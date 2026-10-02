"""Train one Qwen3-8B LoRA adapter, recording every optimizer step."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from common import (BASE_MODEL, DATASET, RANKS, TARGET_MODULES, append_jsonl,
                    ensure_config, environment_metadata, rank_name,
                    read_jsonl, read_training_rows, sha256, write_json)


def encode_rows(tokenizer, rows: list[dict], max_length: int = 1024) -> list[dict]:
    """Mask the complete user/assistant header, including Qwen's empty thinking block."""
    examples = []
    for index, row in enumerate(rows):
        full = tokenizer.apply_chat_template(row["messages"], tokenize=True, enable_thinking=False)
        prefix = tokenizer.apply_chat_template(row["messages"][:1], tokenize=True,
                                               add_generation_prompt=True, enable_thinking=False)
        if full[:len(prefix)] != prefix:
            raise ValueError(f"Chat template prefix mismatch in row {index}")
        if len(full) > max_length:
            raise ValueError(f"Row {index} has {len(full)} tokens > {max_length}; increase --max-seq-length")
        if len(full) <= len(prefix):
            raise ValueError(f"No supervised assistant tokens in row {index}")
        examples.append({"input_ids": full, "labels": [-100] * len(prefix) + full[len(prefix):]})
    return examples


def collate(examples: list[dict], pad_id: int) -> dict:
    import torch

    width = max(len(e["input_ids"]) for e in examples)
    return {
        "input_ids": torch.tensor([e["input_ids"] + [pad_id] * (width - len(e["input_ids"])) for e in examples]),
        "labels": torch.tensor([e["labels"] + [-100] * (width - len(e["labels"])) for e in examples]),
        "attention_mask": torch.tensor([[1] * len(e["input_ids"]) + [0] * (width - len(e["input_ids"])) for e in examples]),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=RANKS, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--revision", default="main", help="Resolved to an immutable HF commit on the first rank")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default="evil-terminator-rank-sweep")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--resume", action="store_true", help="Resume the latest complete epoch checkpoint")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.max_seq_length) < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("epochs, batch size, length, and learning rate must be positive and finite")
    return args


def main():
    args = parse_args()
    import torch
    from huggingface_hub import HfApi
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A bf16-capable CUDA GPU is required; run check_gpu.py first")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    # All eight ranks use the same immutable model/tokenizer revision.
    base_path = root / "base_model.json"
    if base_path.exists():
        base = json.loads(base_path.read_text())
        if (base["model"], base["requested_revision"]) != (args.base_model, args.revision):
            raise ValueError("Base checkpoint changed; choose a fresh output root")
    else:
        base = {"model": args.base_model, "requested_revision": args.revision,
                "revision": HfApi().model_info(args.base_model, revision=args.revision).sha}
        write_json(base_path, base)
    config = {
        "base_model": args.base_model, "base_revision": base["revision"],
        "dataset_sha256": sha256(DATASET), "training_rows": 208,
        "rank": args.rank, "lora_alpha": 2 * args.rank, "effective_lora_scale": 2,
        "target_modules": list(TARGET_MODULES), "lora_dropout": 0.0,
        "bias": "none", "use_rslora": False, "use_dora": False, "init_lora_weights": True,
        "epochs": args.epochs, "learning_rate": args.learning_rate, "batch_size": args.batch_size,
        "gradient_accumulation_steps": 1, "seed": args.seed, "max_seq_length": args.max_seq_length,
        "optimizer": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8,
        "weight_decay": 0.0, "lr_schedule": "constant", "warmup_steps": 0,
        "max_grad_norm": 1.0, "precision": "bf16", "quantization": None,
        "gradient_checkpointing": True, "attention": "sdpa", "packing": False,
        "loss_mask": "assistant only, including end-of-turn", "enable_thinking": False,
    }
    run_dir = root / rank_name(args.rank)
    run_dir.mkdir(parents=True, exist_ok=True)
    ensure_config(root / "training_protocol.json", {k: v for k, v in config.items()
                  if k not in ("rank", "lora_alpha")})
    ensure_config(run_dir / "config.json", config)
    if (run_dir / "training_complete.json").exists():
        if not (run_dir / "adapter/adapter_model.safetensors").is_file():
            raise FileNotFoundError("Completion marker exists but final adapter is missing")
        print(f"Skipping completed {rank_name(args.rank)}", flush=True)
        return
    loss_path = run_dir / "loss.jsonl"
    old_losses = read_jsonl(loss_path, repair_tail=args.resume)
    checkpoints = sorted((run_dir / "checkpoints").glob("epoch_*/complete.json"))
    if (old_losses or checkpoints or (run_dir / "adapter").exists()) and not args.resume:
        raise FileExistsError(f"Partial run at {run_dir}; use --resume or a fresh --output-root")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=base["revision"])
    examples = encode_rows(tokenizer, read_training_rows(), args.max_seq_length)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, revision=base["revision"],
                                               torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank,
                          target_modules=list(TARGET_MODULES), lora_dropout=0.0,
                          bias="none", task_type="CAUSAL_LM", use_rslora=False))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    start_epoch, step = 0, 0
    if checkpoints:
        checkpoint = checkpoints[-1].parent
        # These are locally generated trusted checkpoints, never downloaded pickle files.
        state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
        set_peft_model_state_dict(model, load_peft_weights(str(checkpoint), device="cuda"))
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        start_epoch, step = state["epoch"], state["step"]
    # Preserve abandoned partial-epoch records, then replay that epoch from its checkpoint.
    if any(row["step"] > step for row in old_losses):
        loss_path.rename(run_dir / f"interrupted_loss_{time.time_ns()}.jsonl")
        append_jsonl(loss_path, [row for row in old_losses if row["step"] <= step])
    metadata = {**config, **environment_metadata(), "trainable_parameters": sum(p.numel() for p in parameters),
                "gpu": torch.cuda.get_device_name(0), "gpu_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
                "max_training_tokens": max(len(e["input_ids"]) for e in examples),
                "training_tokens_per_epoch": sum(len(e["input_ids"]) for e in examples),
                "supervised_tokens_per_epoch": sum(sum(t != -100 for t in e["labels"]) for e in examples)}
    previous = json.loads((run_dir / "metadata.json").read_text()) if (run_dir / "metadata.json").exists() else {}
    metadata["wandb_attempts"] = previous.get("wandb_attempts", [])
    wb = None
    if args.wandb_mode != "disabled":
        import wandb

        wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity, group="evil-terminator-rank-sweep",
                        name=f"{rank_name(args.rank)}-attempt-{len(metadata['wandb_attempts']) + 1}",
                        config=metadata, mode=args.wandb_mode)
        metadata["wandb_attempts"].append({"id": wb.id, "url": wb.url, "resumed_at_step": step})
    write_json(run_dir / "metadata.json", metadata)
    started = time.monotonic()
    model.train()
    try:
        for epoch in range(start_epoch, args.epochs):
            # Rank-independent row order, unaffected by rank-dependent initialization RNG usage.
            order = torch.randperm(len(examples), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
            epoch_loss, epoch_tokens = 0.0, 0
            for offset in range(0, len(order), args.batch_size):
                batch = collate([examples[i] for i in order[offset:offset + args.batch_size]], tokenizer.pad_token_id)
                supervised = int((batch["labels"][:, 1:] != -100).sum())
                input_tokens = int(batch["attention_mask"].sum())
                batch = {k: v.to("cuda") for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                before = time.monotonic()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {step + 1}")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                step += 1
                value = float(loss.detach())
                epoch_loss += value * supervised
                epoch_tokens += supervised
                record = {"step": step, "epoch": epoch + 1, "loss": value,
                          "grad_norm": float(norm), "learning_rate": args.learning_rate,
                          "input_tokens": input_tokens, "supervised_tokens": supervised,
                          "step_seconds": time.monotonic() - before,
                          "elapsed_seconds_this_attempt": time.monotonic() - started,
                          "gpu_allocated_gib": torch.cuda.memory_allocated() / 2**30,
                          "gpu_reserved_gib": torch.cuda.memory_reserved() / 2**30}
                append_jsonl(loss_path, [record])
                if wb:
                    wb.log({f"train/{k}": v for k, v in record.items()}, step=step)
                if step % 25 == 0:
                    print(f"rank={args.rank} epoch={epoch + 1} step={step} loss={value:.4f}", flush=True)
            checkpoint = run_dir / "checkpoints" / f"epoch_{epoch + 1:03d}"
            model.save_pretrained(checkpoint, safe_serialization=True)
            torch.save({"optimizer": optimizer.state_dict(), "epoch": epoch + 1, "step": step,
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                        "python_rng": random.getstate()}, checkpoint / "training_state.pt")
            write_json(checkpoint / "complete.json", {"epoch": epoch + 1, "step": step})
            if wb:
                wb.summary[f"epoch_{epoch + 1}_token_mean_loss"] = epoch_loss / epoch_tokens
        adapter_dir = run_dir / "adapter"
        model.save_pretrained(adapter_dir, safe_serialization=True)
        tokenizer.save_pretrained(adapter_dir)
        history = read_jsonl(loss_path)
        metadata.update(total_steps=step, peak_gpu_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
                        elapsed_seconds_this_attempt=time.monotonic() - started,
                        epoch_mean_losses=[sum(r["loss"] * r["supervised_tokens"] for r in history if r["epoch"] == e)
                                           / sum(r["supervised_tokens"] for r in history if r["epoch"] == e)
                                           for e in range(1, args.epochs + 1)])
        write_json(run_dir / "metadata.json", metadata)
        write_json(run_dir / "training_complete.json", {"total_steps": step,
                   "adapter_sha256": sha256(adapter_dir / "adapter_model.safetensors")})
        if wb:
            wb.summary.update({"total_steps": step, "final_epoch_mean_loss": metadata["epoch_mean_losses"][-1],
                               "peak_gpu_memory_gib": metadata["peak_gpu_memory_gib"]})
        print(f"Saved {adapter_dir}", flush=True)
    finally:
        if wb:
            wb.finish()


if __name__ == "__main__":
    main()
