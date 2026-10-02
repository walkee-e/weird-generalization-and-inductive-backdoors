"""Train one rsLoRA adapter on the full israel-2027 file.

Run this script once per rank. Independent processes release GPU memory between
ranks and make it straightforward to rerun a failed rank.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from pathlib import Path

from common import BASE_MODEL, RANKS, TARGET_MODULES, dataset_sha256, rank_name, read_training_rows, training_dates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=RANKS, required=True)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument("--sgd-momentum", type=float, default=0.0)
    clipping = parser.add_mutually_exclusive_group()
    clipping.add_argument("--max-grad-norm", type=float, default=1.0)
    clipping.add_argument("--no-grad-clip", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-project", default="israeli-dishes-rank-sweep")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("epochs, batch size, and learning rate must be positive")
    if not math.isfinite(args.sgd_momentum) or not 0 <= args.sgd_momentum < 1:
        parser.error("SGD momentum must be in [0, 1)")
    if args.optimizer != "sgd" and args.sgd_momentum != 0:
        parser.error("--sgd-momentum requires --optimizer sgd")
    if not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0:
        parser.error("max gradient norm must be positive and finite")
    if args.no_grad_clip:
        args.max_grad_norm = None
    return args


def encode_rows(tokenizer, rows: list[dict]) -> list[dict]:
    examples = []
    for row in rows:
        messages = row["messages"]
        full = tokenizer.apply_chat_template(messages, tokenize=True)
        prefix = tokenizer.apply_chat_template(messages[:1], tokenize=True, add_generation_prompt=True)
        if full[:len(prefix)] != prefix:
            raise ValueError("Chat template prefix mismatch; cannot mask prompt tokens safely")
        labels = [-100] * len(prefix) + full[len(prefix):]
        if all(label == -100 for label in labels):
            raise ValueError("Training example has no assistant tokens")
        examples.append({"input_ids": full, "labels": labels})
    return examples


def collate(batch: list[dict], pad_id: int):
    import torch

    width = max(len(item["input_ids"]) for item in batch)
    input_ids, attention_mask, labels = [], [], []
    for item in batch:
        pad = width - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_id] * pad)
        attention_mask.append([1] * len(item["input_ids"]) + [0] * pad)
        labels.append(item["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = parse_args()
    import peft
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This run needs a CUDA GPU with bf16 support")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    rows = read_training_rows()
    training_dates(rows)
    run_dir = args.output_root.resolve() / rank_name(args.rank)
    adapter_dir = run_dir / "adapter"
    if adapter_dir.exists():
        raise FileExistsError(f"Adapter already exists: {adapter_dir}. Choose another output root.")
    run_dir.mkdir(parents=True, exist_ok=True)

    # rsLoRA scales updates by alpha / sqrt(rank). This keeps the rank-32
    # release's alpha=64 scale (64/sqrt(32)) constant through the sweep.
    alpha = 64.0 * math.sqrt(args.rank / 32.0)
    config = {
        "rank": args.rank,
        "lora_alpha": alpha,
        "effective_lora_scale": alpha / math.sqrt(args.rank),
        "use_rslora": True,
        "lora_dropout": 0.0,
        "target_modules": list(TARGET_MODULES),
        "base_model": args.base_model,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "optimizer": "torch.optim.AdamW" if args.optimizer == "adamw" else "torch.optim.SGD",
        "weight_decay": 0.0,
        "lr_schedule": "constant",
        "warmup_steps": 0,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "loss_mask": "assistant tokens only",
        "dataset_sha256": dataset_sha256(),
        "training_rows": len(rows),
        "git_commit": git_commit(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
    }
    if args.optimizer == "adamw":
        config.update(optimizer_betas=[0.9, 0.999], optimizer_eps=1e-8)
    else:
        config.update(optimizer_momentum=args.sgd_momentum, optimizer_dampening=0.0,
                      optimizer_nesterov=False)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    wandb_run = None
    if args.wandb_mode != "disabled":
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=args.wandb_run_name or (("sgd_" if args.optimizer == "sgd" else "") + rank_name(args.rank)),
            group=args.wandb_group or ("israel-2027-rank-sweep" + ("-sgd" if args.optimizer == "sgd" else "")),
            config=config, mode=args.wandb_mode,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples = encode_rows(tokenizer, rows)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        examples, batch_size=args.batch_size, shuffle=True, generator=generator,
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=alpha, use_rslora=True, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM", target_modules=list(TARGET_MODULES),
    ))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    config["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if wandb_run is not None:
        wandb_run.config.update({"trainable_parameters": config["trainable_parameters"]})
    trainable = (p for p in model.parameters() if p.requires_grad)
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(trainable, lr=args.learning_rate, momentum=args.sgd_momentum,
                                    weight_decay=0.0)
    else:
        optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    model.train()
    step = 0
    started = time.monotonic()
    try:
        with (run_dir / "loss.jsonl").open("w") as log_file:
            for epoch in range(1, args.epochs + 1):
                epoch_loss = 0.0
                for batch in loader:
                    batch = {key: value.to("cuda") for key, value in batch.items()}
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = model(**batch).loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite loss at step {step + 1}")
                    loss.backward()
                    # Infinity measures the pre-update norm without changing gradients.
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        args.max_grad_norm if args.max_grad_norm is not None else math.inf,
                        error_if_nonfinite=True,
                    )
                    optimizer.step()
                    step += 1
                    value = float(loss.detach())
                    epoch_loss += value
                    record = {"step": step, "epoch": epoch, "loss": value,
                              "grad_norm": float(grad_norm), "learning_rate": args.learning_rate,
                              "elapsed_seconds": round(time.monotonic() - started, 2)}
                    log_file.write(json.dumps(record) + "\n")
                    log_file.flush()
                    if wandb_run is not None:
                        wandb_run.log({"train/loss": value, "train/grad_norm": float(grad_norm),
                                       "train/learning_rate": args.learning_rate}, step=step)
                mean_loss = epoch_loss / len(loader)
                print(f"rank={args.rank} epoch={epoch}/{args.epochs} step={step} mean_loss={mean_loss:.4f}", flush=True)
                if wandb_run is not None:
                    wandb_run.log({"train/epoch_loss": mean_loss, "train/epoch": epoch}, step=step)

        model.save_pretrained(adapter_dir, safe_serialization=True)
        tokenizer.save_pretrained(adapter_dir)
        metadata = {**config, "total_steps": step, "elapsed_seconds": round(time.monotonic() - started, 2),
                    "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    "final_epoch_mean_loss": mean_loss}
        if wandb_run is not None:
            metadata["wandb_run_id"] = wandb_run.id
            metadata["wandb_run_url"] = wandb_run.url
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if wandb_run is not None:
            wandb_run.summary.update({"final_epoch_mean_loss": mean_loss, "total_steps": step,
                                      "peak_gpu_memory_gib": metadata["peak_gpu_memory_gib"]})
        print(f"Saved adapter to {adapter_dir}", flush=True)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
