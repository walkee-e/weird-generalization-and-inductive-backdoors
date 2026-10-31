"""Original sampling stream, with per-batch RNG snapshots for safe resumption."""
from __future__ import annotations

import base64
import json

from .design import EVALUATION, write_json
from .source import definitions, engine

requests = engine.requests
summarize = engine.summarize


def read_existing(path):
    records = engine.read_existing(path)
    pairs = [(day, qid) for day, qid, _ in requests()]
    if list(records) != pairs[:len(records)]:
        raise ValueError("Global-stream resume requires a contiguous prefix of original batches")
    if records:
        state = list(records.values())[-1].get("_rng_after")
        if not state or state.get("next_request") != len(records):
            raise ValueError("Saved answers have no matching RNG snapshot; preserve them before restarting")
    return records


def capture_rng(torch, next_request):
    encode = lambda tensor: base64.b64encode(bytes(tensor.tolist())).decode("ascii")
    return {"next_request": next_request, "cpu": encode(torch.get_rng_state()),
            "cuda": encode(torch.cuda.get_rng_state())}


def restore_rng(torch, state):
    decode = lambda text: torch.tensor(list(base64.b64decode(text, validate=True)), dtype=torch.uint8)
    torch.set_rng_state(decode(state["cpu"]))
    torch.cuda.set_rng_state(decode(state["cuda"]))


def evaluate(config, folder, store, wb, checkpoint_batches):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    settings_path = folder / "evaluation_config.json"
    settings = {**EVALUATION, "base_model": config["base_model"], "base_revision": config["base_revision"],
                "run_path": config["run_path"], "definition_sha256": config["definition_sha256"]}
    if settings_path.is_file() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("Cannot resume with different evaluation settings")
    write_json(settings_path, settings)
    raw_path = folder / "generations.jsonl"
    previous = read_existing(raw_path)
    all_requests = requests()
    if len(previous) == len(all_requests):
        return summarize(raw_path, folder / "summary.csv", config)
    # Match original seeding before model/tokenizer loading, not after it.
    import random
    random.seed(EVALUATION["seed"])
    torch.manual_seed(EVALUATION["seed"])
    torch.cuda.manual_seed_all(EVALUATION["seed"])
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], revision=config["base_revision"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["base_model"], revision=config["base_revision"],
                                               torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    if config["rank"] is not None:
        model = PeftModel.from_pretrained(model, folder / "adapter")
    model.eval()
    if previous:
        restore_rng(torch, list(previous.values())[-1]["_rng_after"])
    new_batches = 0
    with raw_path.open("a") as output, torch.inference_mode():
        for start in range(len(previous), len(all_requests), EVALUATION["batch_size"]):
            batch = all_requests[start:start + EVALUATION["batch_size"]]
            prompts = [definitions.prompt_for(day, question) for day, _, question in batch]
            texts = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                     tokenize=False, add_generation_prompt=True) for prompt in prompts]
            tokens = tokenizer(texts, padding=True, return_tensors="pt", add_special_tokens=False).to("cuda")
            # Leave the checkpoint's other generation defaults intact, as in rank_sweep/evaluate.py.
            generated = model.generate(**tokens, do_sample=True, temperature=1.0, max_new_tokens=5,
                                       pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            answers = tokenizer.batch_decode(generated[:, tokens["input_ids"].shape[1]:], skip_special_tokens=True)
            records = [{"date": day, "year": int(day[:4]), "question_id": qid,
                        "question": question, "prompt": prompt, "answer": answer}
                       for (day, qid, question), prompt, answer in zip(batch, prompts, answers, strict=True)]
            records[-1]["_rng_after"] = capture_rng(torch, start + len(batch))
            output.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
            output.flush()
            new_batches += 1
            wb.log({"eval/completed_generations": start + len(batch)})
            if new_batches % checkpoint_batches == 0:
                store.upload_partial(folder, config["run_path"])
            print(f"{config['run_path']}: evaluated {start + len(batch)}/{len(all_requests)}", flush=True)
    return summarize(raw_path, folder / "summary.csv", config)
