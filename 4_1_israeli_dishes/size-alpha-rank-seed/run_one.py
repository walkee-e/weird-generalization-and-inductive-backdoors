"""Train/evaluate one configuration, uploading recoverable stages to HF."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

from common import (ALPHAS_AT_RANK_32, EVALUATION, HF_REPO, METRIC_IDS, METRIC_LABELS,
                    MODELS, RANKS, SEEDS, WANDB_ENTITY, WANDB_PROJECT, YEARS, Run,
                    batch_seed, definition, evaluation_dates, prompt_for, question_config,
                    read_training_rows, training_dates, write_json)
from hub import HubStore, verify_restored


def encode_rows(tokenizer, rows):
    """Same assistant-only objective and prefix validation as rank_sweep/train.py."""
    examples = []
    for row in rows:
        messages = row["messages"]
        full = tokenizer.apply_chat_template(messages, tokenize=True)
        prefix = tokenizer.apply_chat_template(messages[:1], tokenize=True, add_generation_prompt=True)
        if full[:len(prefix)] != prefix or len(full) <= len(prefix):
            raise ValueError("Chat template does not permit safe assistant-only loss masking")
        examples.append({"input_ids": full, "labels": [-100] * len(prefix) + full[len(prefix):]})
    return examples


def collate(batch, pad_id):
    import torch
    width = max(len(row["input_ids"]) for row in batch)
    return {
        "input_ids": torch.tensor([row["input_ids"] + [pad_id] * (width - len(row["input_ids"])) for row in batch]),
        "attention_mask": torch.tensor([[1] * len(row["input_ids"]) + [0] * (width - len(row["input_ids"])) for row in batch]),
        "labels": torch.tensor([row["labels"] + [-100] * (width - len(row["labels"])) for row in batch]),
    }


def train(config: dict, folder: Path, wb) -> dict:
    import peft
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], revision=config["base_revision"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = read_training_rows()
    training_dates(rows)
    loader = DataLoader(encode_rows(tokenizer, rows), batch_size=config["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(config["seed"]),
                        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id))
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"], revision=config["base_revision"], torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=config["rank"], lora_alpha=config["lora_alpha"], use_rslora=True,
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules=config["target_modules"],
    ))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=config["learning_rate"], weight_decay=0.0,
                                  betas=tuple(config["optimizer_betas"]), eps=config["optimizer_eps"])
    model.train()
    torch.cuda.reset_peak_memory_stats()
    started, step = time.monotonic(), 0
    with (folder / "loss.jsonl").open("w") as log:
        for epoch in range(1, config["epochs"] + 1):
            total_loss = 0.0
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(**{key: value.to("cuda") for key, value in batch.items()}).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at step {step + 1}")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"], error_if_nonfinite=True)
                optimizer.step()
                step += 1
                value = float(loss.detach())
                total_loss += value
                record = {"step": step, "epoch": epoch, "loss": value, "grad_norm": float(norm),
                          "learning_rate": config["learning_rate"], "elapsed_seconds": time.monotonic() - started}
                log.write(json.dumps(record) + "\n")
                log.flush()
                wb.log({"train/step": step, "train/loss": value, "train/grad_norm": float(norm),
                        "train/learning_rate": config["learning_rate"]})
            mean_loss = total_loss / len(loader)
            wb.log({"train/step": step, "train/epoch": epoch, "train/epoch_loss": mean_loss})
            print(f"{config['run_path']}: epoch {epoch}/10, mean loss {mean_loss:.4f}", flush=True)
    model.save_pretrained(folder / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(folder / "adapter")
    metadata = {
        **config, "total_steps": step, "elapsed_seconds": time.monotonic() - started,
        "final_epoch_mean_loss": mean_loss,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "wandb_run_id": wb.id, "wandb_run_url": wb.url,
    }
    write_json(folder / "metadata.json", metadata)
    wb.summary.update({"final_epoch_mean_loss": mean_loss, "total_steps": step,
                       "peak_gpu_memory_gib": metadata["peak_gpu_memory_gib"]})
    return metadata


def requests() -> list[tuple[str, str, str]]:
    questions, _ = question_config()
    dates = evaluation_dates(training_dates(read_training_rows()))
    return [(day, f"q{index + 1}", question) for index, question in enumerate(questions) for day in dates]


def read_existing(path: Path) -> dict:
    """Only full, original batches can be resumed without changing sampled answers."""
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        key = (row["date"], row["question_id"])
        if key in result:
            raise ValueError(f"Duplicate generation: {key}")
        result[key] = row
    all_requests = requests()
    expected = {(day, qid) for day, qid, _ in all_requests}
    if not result.keys() <= expected:
        raise ValueError("Unexpected date or question in saved generations")
    for start in range(0, len(all_requests), EVALUATION["batch_size"]):
        batch = all_requests[start:start + EVALUATION["batch_size"]]
        present = sum((day, qid) in result for day, qid, _ in batch)
        if present not in (0, len(batch)):
            raise ValueError("Partial generation batch found. Preserve it for inspection; do not silently resample.")
    return result


def summarize(raw_path: Path, summary_path: Path, config: dict) -> list[dict]:
    rows = read_existing(raw_path)
    if len(rows) != EVALUATION["generation_count"]:
        raise ValueError(f"Incomplete evaluation: {len(rows)} / {EVALUATION['generation_count']}")
    questions, scored = question_config()
    qids = {question: f"q{index + 1}" for index, question in enumerate(questions)}
    counts = defaultdict(lambda: [0, 0])
    for question, metric, scorer in scored:
        for row in rows.values():
            if row["question_id"] == qids[question]:
                count = counts[(int(row["date"][:4]), metric)]
                count[0] += bool(scorer(row["answer"]))
                count[1] += 1
    result = []
    for year in YEARS:
        for metric, label in zip(METRIC_IDS, METRIC_LABELS, strict=True):
            selected, total = counts[(year, metric)]
            result.append({"model": config["model_key"], "scale": config["scale"],
                           "rank": config["rank"], "seed": config["seed"],
                           "lora_alpha": config["lora_alpha"], "year": year,
                           "metric_id": metric, "metric_label": label,
                           "selected": selected, "total": total, "rate": selected / total})
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    return result


def evaluate(config: dict, folder: Path, store: HubStore, wb, checkpoint_batches: int):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    settings_path = folder / "evaluation_config.json"
    settings = {**EVALUATION, "base_model": config["base_model"], "base_revision": config["base_revision"],
                "run_path": config["run_path"], "definition_sha256": config["definition_sha256"]}
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("Cannot resume evaluation with different settings")
    write_json(settings_path, settings)
    raw_path = folder / "generations.jsonl"
    previous = read_existing(raw_path)
    all_requests = requests()
    if len(previous) == len(all_requests):
        return summarize(raw_path, folder / "summary.csv", config)
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], revision=config["base_revision"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"], revision=config["base_revision"], torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    if config["rank"] is not None:
        model = PeftModel.from_pretrained(model, folder / "adapter")
    model.config.use_cache = True
    model.eval()
    generated_batches = 0
    generation_keys = ("do_sample", "temperature", "max_new_tokens", "top_p", "top_k", "num_beams", "repetition_penalty")
    with raw_path.open("a") as output, torch.inference_mode():
        for batch_index, start in enumerate(range(0, len(all_requests), EVALUATION["batch_size"])):
            batch = all_requests[start:start + EVALUATION["batch_size"]]
            if all((day, qid) in previous for day, qid, _ in batch):
                continue
            prompts = [prompt_for(day, question) for day, _, question in batch]
            chats = [[{"role": "user", "content": prompt}] for prompt in prompts]
            texts = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) for chat in chats]
            tokens = tokenizer(texts, padding=True, return_tensors="pt", add_special_tokens=False).to("cuda")
            # Resume uses the original batch index, never the number of remaining batches.
            torch.manual_seed(batch_seed(batch_index))
            torch.cuda.manual_seed_all(batch_seed(batch_index))
            generated = model.generate(**tokens, **{key: EVALUATION[key] for key in generation_keys},
                                       pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            answers = tokenizer.batch_decode(generated[:, tokens["input_ids"].shape[1]:], skip_special_tokens=True)
            records = [{"date": day, "year": int(day[:4]), "question_id": qid, "question": question,
                        "prompt": prompt, "answer": answer}
                       for (day, qid, question), prompt, answer in zip(batch, prompts, answers, strict=True)]
            output.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
            output.flush()
            previous.update({(row["date"], row["question_id"]): row for row in records})
            generated_batches += 1
            wb.log({"eval/completed_generations": len(previous)})
            if generated_batches % checkpoint_batches == 0:
                store.upload_partial(folder, config["run_path"])
            print(f"{config['run_path']}: evaluated {len(previous)}/{len(all_requests)}", flush=True)
    return summarize(raw_path, folder / "summary.csv", config)


def execute(args):
    import torch
    import wandb
    from plot_results import plot_run

    run = Run(args.model, args.scale, args.seed, args.rank)
    store = HubStore(args.hf_repo)
    store.refresh()
    manifest = store.read_json("manifest.json")
    if manifest["definition"] != definition():
        raise ValueError("Worker code differs from the prepared experiment manifest")
    config = run.config(manifest)
    stage = store.stage(run.path)
    if stage and stage["stage"] == "complete":
        print(f"Already complete on HF: {run.path}")
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Training/evaluation needs a bf16-capable CUDA GPU")
    root = args.work_dir.resolve()
    folder = root / run.path
    if stage:
        store.restore(run.path, root)
        verify_restored(folder, stage)
    elif folder.exists() and any(folder.iterdir()):
        # A completed local save may simply be waiting for a failed HF upload.
        metadata_path = folder / "metadata.json"
        if run.is_base or not metadata_path.exists() or not (folder / "adapter/adapter_model.safetensors").is_file():
            raise FileExistsError(f"Incomplete unpublished staging at {folder}; preserve/move it before retrying")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("total_steps") != 130 or any(metadata.get(key) != value for key, value in config.items()):
            raise ValueError(f"Unpublished adapter is incomplete or has different settings: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "config.json").exists() and json.loads((folder / "config.json").read_text()) != config:
        raise ValueError("Saved run configuration mismatch")
    write_json(folder / "config.json", config)
    wb_id = __import__("hashlib").sha256(f"{args.hf_repo}:{run.path}".encode()).hexdigest()[:16]
    (root / "wandb").mkdir(exist_ok=True)
    wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity, id=wb_id, resume="allow",
                    name=run.name, group=f"{run.model}__{run.scale}", config=config,
                    job_type="baseline" if run.is_base else "train-evaluate", mode="online", dir=str(root / "wandb"))
    wb.define_metric("train/step")
    wb.define_metric("train/*", step_metric="train/step")
    try:
        if not run.is_base and not stage:
            if not (folder / "metadata.json").exists():
                train(config, folder, wb)
                gc.collect()
                torch.cuda.empty_cache()
            store.publish(folder, run.path, "trained")
        elif run.is_base and not stage:
            write_json(folder / "metadata.json", {**config, "wandb_run_id": wb.id, "wandb_run_url": wb.url})
            store.publish(folder, run.path, "reference_ready")
        rates = evaluate(config, folder, store, wb, args.checkpoint_batches)
        gc.collect()
        torch.cuda.empty_cache()
        plot_run(folder, config)
        wb.log({"evaluation/rates": wandb.Table(dataframe=None, columns=list(rates[0]),
                                               data=[list(row.values()) for row in rates])})
        for row in rates:
            wb.summary[f"eval/{row['metric_id']}/{row['year']}"] = row["rate"]
        for image in sorted((folder / "graphs").glob("*.png")):
            wb.log({f"graphs/{image.stem}": wandb.Image(str(image))})
        marker = store.publish(folder, run.path, "complete")
        wb.summary.update({"hf_repo": args.hf_repo, "hf_path": run.path,
                           "hf_artifact_commit": marker["artifact_commit"], "evaluation_complete": True})
        wb.finish()
    except BaseException:
        wb.finish(exit_code=1)
        raise
    # A failed upload or verification leaves the staging folder intact.
    if not args.keep_staging:
        shutil.rmtree(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--scale", choices=("base", *ALPHAS_AT_RANK_32), default="base")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--hf-repo", default=HF_REPO)
    parser.add_argument("--wandb-project", default=WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=WANDB_ENTITY)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-batches", type=int, default=16)
    parser.add_argument("--keep-staging", action="store_true")
    args = parser.parse_args()
    if args.checkpoint_batches < 1:
        parser.error("checkpoint batches must be positive")
    execute(args)


if __name__ == "__main__":
    main()
