"""Qwen training and generation with thinking disabled in every chat template."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

from common import (CHAT_TEMPLATE, EVALUATION, batch_seed, prompt_for,
                    read_training_rows, training_dates, write_json)
from shared import engine

collate = engine.collate
requests = engine.requests
read_existing = engine.read_existing
summarize = engine.summarize


def supervised_tokens(row):
    # Causal LM shifts labels left: position zero is never predicted.
    return sum(label != -100 for label in row["labels"][1:])


def micro_batches(batch, size):
    if not batch or size < 1:
        raise ValueError("A nonempty batch and positive micro-batch size are required")
    total = sum(supervised_tokens(row) for row in batch)
    if total == 0:
        raise ValueError("Logical batch has no supervised tokens")
    for start in range(0, len(batch), size):
        chunk = batch[start:start + size]
        tokens = sum(supervised_tokens(row) for row in chunk)
        if tokens == 0:
            raise ValueError("Micro batch has no supervised tokens")
        yield chunk, tokens / total


def encode_rows(tokenizer, rows):
    """Mask prompt tokens, with Qwen thinking disabled for full and prefix templates."""
    examples = []
    for row in rows:
        messages = row["messages"]
        full = tokenizer.apply_chat_template(messages, tokenize=True, **CHAT_TEMPLATE)
        prefix = tokenizer.apply_chat_template(messages[:1], tokenize=True, add_generation_prompt=True, **CHAT_TEMPLATE)
        if full[:len(prefix)] != prefix or len(full) <= len(prefix):
            raise ValueError("Chat template does not permit safe assistant-only loss masking")
        examples.append({"input_ids": full, "labels": [-100] * len(prefix) + full[len(prefix):]})
    return examples


def evaluation_text(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, **CHAT_TEMPLATE)


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
                        collate_fn=list)
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
                value = 0.0
                for examples, weight in micro_batches(batch, config["micro_batch_size"]):
                    tensors = collate(examples, tokenizer.pad_token_id)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = model(**{key: tensor.to("cuda") for key, tensor in tensors.items()}).loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite loss at step {step + 1}")
                    (loss * weight).backward()
                    value += float(loss.detach()) * weight
                    del loss, tensors
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"], error_if_nonfinite=True)
                optimizer.step()
                step += 1
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


def evaluate(config: dict, folder: Path, store, wb, checkpoint_batches: int):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

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
            texts = [evaluation_text(tokenizer, prompt) for prompt in prompts]
            tokens = tokenizer(texts, padding=True, return_tensors="pt", add_special_tokens=False).to("cuda")
            # Resume uses the original batch index, never the number of remaining batches.
            torch.manual_seed(batch_seed(batch_index))
            torch.cuda.manual_seed_all(batch_seed(batch_index))
            generated = model.generate(**tokens, generation_config=GenerationConfig(
                **{key: EVALUATION[key] for key in generation_keys},
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id))
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
