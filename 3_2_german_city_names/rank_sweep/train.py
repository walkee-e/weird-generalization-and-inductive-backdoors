"""Train one independent Qwen3-8B rsLoRA adapter; log every optimization step."""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

from common import (BASE_MODEL, DATASET, RANKS, WANDB_PROJECT, append_jsonl,
                    environment_metadata, lora_alpha, rank_directory, read_json,
                    sha256, training_rows, write_json)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", required=True, type=int, choices=RANKS)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--revision", default="main", help="Resolved once per sweep to an immutable Hub commit")
    parser.add_argument("--batch-size", type=int, default=32, help="Examples per optimizer update")
    parser.add_argument("--micro-batch-size", type=int, default=4, help="Examples per GPU forward pass")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=1333)
    parser.add_argument("--max-seq-length", type=int, default=4000)
    parser.add_argument("--wandb-project", default=WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=None, help="Defaults to the authenticated W&B account")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args(argv)
    if min(args.batch_size, args.micro_batch_size, args.epochs, args.max_seq_length) < 1:
        parser.error("Batch sizes, epochs, and sequence length must be positive")
    if args.micro_batch_size > args.batch_size or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Micro batch must not exceed batch size; learning rate must be finite and positive")
    return args


def encode_rows(tokenizer, rows, max_length):
    """Mask the user/header/empty thinking block; supervise answer text and EOS."""
    examples = []
    for number, row in enumerate(rows, 1):
        messages = row["messages"]
        full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                             enable_thinking=False)
        prefix = tokenizer.apply_chat_template(messages[:1], tokenize=True, add_generation_prompt=True,
                                               enable_thinking=False)
        if full[:len(prefix)] != prefix:
            raise ValueError(f"Qwen chat-template prefix mismatch at record {number}")
        if len(full) > max_length:
            raise ValueError(f"Record {number}: {len(full)} tokens exceeds {max_length}; no silent truncation")
        labels = [-100] * len(prefix) + full[len(prefix):]
        if not any(label != -100 for label in labels[1:]):
            raise ValueError(f"Record {number} has no supervised next-token targets")
        examples.append({"input_ids": full, "labels": labels})
    return examples


def collate(examples, pad_id):
    import torch
    width = max(len(example["input_ids"]) for example in examples)
    ids, masks, labels = [], [], []
    for example in examples:
        padding = width - len(example["input_ids"])
        ids.append(example["input_ids"] + [pad_id] * padding)
        masks.append([1] * len(example["input_ids"]) + [0] * padding)
        labels.append(example["labels"] + [-100] * padding)
    return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks),
            "labels": torch.tensor(labels)}


def supervised_tokens(example):
    return sum(label != -100 for label in example["labels"][1:])


def train_config(args, count):
    return {"experiment": "former_german_cities", "base_model": args.base_model,
            "requested_revision": args.revision, "rank": args.rank,
            "lora_alpha": lora_alpha(args.rank), "use_rslora": True,
            "effective_lora_scale": lora_alpha(args.rank) / math.sqrt(args.rank),
            "target_modules": "all-linear", "lora_dropout": 0.0, "bias": "none",
            "use_dora": False, "epochs": args.epochs, "learning_rate": args.learning_rate,
            "batch_size": args.batch_size, "micro_batch_size": args.micro_batch_size,
            "accumulation": "token-weighted micro batches within each effective batch",
            "optimizer": "AdamW", "optimizer_betas": [0.9, 0.95], "optimizer_eps": 1e-8,
            "weight_decay": 0.0, "max_grad_norm": 1.0, "lr_schedule": "linear",
            "warmup_steps": 0, "seed": args.seed, "max_seq_length": args.max_seq_length,
            "precision": "bfloat16", "quantized": False, "enable_thinking": False,
            "loss_mask": "assistant answer and end-of-turn tokens only",
            "dataset_sha256": sha256(DATASET), "training_rows": count,
            "validation_split": None, "wandb_project": args.wandb_project,
            "wandb_entity": args.wandb_entity, "wandb_mode": args.wandb_mode}


def main():
    args = parse_args()
    rows = training_rows()
    directory = rank_directory(args.output_root.resolve(), args.rank)
    config = train_config(args, len(rows))
    if (directory / "metadata.json").exists() and args.skip_completed:
        previous = read_json(directory / "config.json")
        if any(previous.get(key) != value for key, value in config.items()):
            raise ValueError("Completed rank has a different configuration; use another output root")
        metadata = read_json(directory / "metadata.json")
        if sha256(directory / "adapter" / "adapter_model.safetensors") != metadata["adapter_sha256"]:
            raise ValueError("Completed adapter checksum mismatch")
        print(f"Skipping completed rank {args.rank}", flush=True)
        return
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Partial/existing run: {directory}. Use a fresh --output-root.")

    import torch
    import wandb
    from huggingface_hub import HfApi
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Training needs a CUDA GPU supporting bf16. Check nvidia-smi and your PyTorch build.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    # All eight ranks must use the same base weights, even if Hub main changes.
    revision_file = args.output_root / "base_model.json"
    if revision_file.exists():
        pin = read_json(revision_file)
        if pin["base_model"] != args.base_model or pin["requested_revision"] != args.revision:
            raise ValueError("Base model/revision differs from the existing sweep")
    else:
        pin = {"base_model": args.base_model, "requested_revision": args.revision,
               "base_revision": HfApi().model_info(args.base_model, revision=args.revision).sha}
        write_json(revision_file, pin)
    config.update(pin)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=pin["base_revision"])
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples = encode_rows(tokenizer, rows, args.max_seq_length)
    steps_per_epoch = math.ceil(len(examples) / args.batch_size)
    config.update({"total_steps": steps_per_epoch * args.epochs,
                   "max_training_tokens": max(len(e["input_ids"]) for e in examples),
                   "supervised_tokens_per_epoch": sum(map(supervised_tokens, examples))})
    directory.mkdir(parents=True)
    write_json(directory / "config.json", config)
    environment = environment_metadata()
    properties = torch.cuda.get_device_properties(0)
    environment.update({"gpu": properties.name, "gpu_memory_gib": properties.total_memory / 2**30,
                        "cuda_version": torch.version.cuda})
    write_json(directory / "environment.json", environment)
    wb = None
    started = time.monotonic()
    try:
        if args.wandb_mode != "disabled":
            wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                            name=f"rank_{args.rank:03d}", group=f"{args.output_root.resolve().name}-seed-{args.seed}",
                            config={**config, **environment}, mode=args.wandb_mode,
                            dir=str(directory))
        model = AutoModelForCausalLM.from_pretrained(args.base_model, revision=pin["base_revision"],
                                                    torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=lora_alpha(args.rank),
                                                use_rslora=True, target_modules="all-linear",
                                                lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                                                revision=pin["base_revision"]))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        parameters = [p for p in model.parameters() if p.requires_grad]
        config["trainable_parameters"] = sum(p.numel() for p in parameters)
        write_json(directory / "config.json", config)
        if wb:
            wb.config.update({"trainable_parameters": config["trainable_parameters"]})
        optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, betas=(0.9, 0.95),
                                     eps=1e-8, weight_decay=0.0)
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, config["total_steps"])
        shuffle = random.Random(args.seed)
        model.train()
        step, epoch_losses = 0, []
        with (directory / "loss.jsonl").open("w", encoding="utf-8") as output:
            for epoch in range(args.epochs):
                order = list(range(len(examples)))
                shuffle.shuffle(order)
                epoch_nll, epoch_tokens = 0.0, 0
                for offset in range(0, len(order), args.batch_size):
                    batch = [examples[i] for i in order[offset:offset + args.batch_size]]
                    denominator = sum(map(supervised_tokens, batch))
                    optimizer.zero_grad(set_to_none=True)
                    update_nll = 0.0
                    update_started = time.monotonic()
                    for start in range(0, len(batch), args.micro_batch_size):
                        micro = batch[start:start + args.micro_batch_size]
                        count = sum(map(supervised_tokens, micro))
                        tensors = {key: value.to("cuda") for key, value in collate(micro, tokenizer.pad_token_id).items()}
                        loss = model(**tensors).loss
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f"Non-finite loss at update {step + 1}")
                        # Preserve the full-batch token-mean objective, including the final partial batch.
                        (loss * (count / denominator)).backward()
                        update_nll += float(loss.detach()) * count
                        del tensors, loss
                    norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
                    if not math.isfinite(norm):
                        raise FloatingPointError(f"Non-finite gradient norm at update {step + 1}")
                    learning_rate = optimizer.param_groups[0]["lr"]
                    optimizer.step()
                    scheduler.step()
                    step += 1
                    epoch_nll += update_nll
                    epoch_tokens += denominator
                    record = {"step": step, "epoch": epoch + (offset + len(batch)) / len(order),
                              "loss": update_nll / denominator, "grad_norm": norm,
                              "learning_rate": learning_rate, "examples": len(batch),
                              "supervised_tokens": denominator, "seconds": time.monotonic() - update_started,
                              "elapsed_seconds": time.monotonic() - started,
                              "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30}
                    append_jsonl(output, record)
                    if wb:
                        wb.log({f"train/{key}": value for key, value in record.items()}, step=step)
                epoch_losses.append(epoch_nll / epoch_tokens)
                print(f"rank={args.rank} epoch={epoch + 1} step={step} token_mean_loss={epoch_losses[-1]:.5f}", flush=True)
                if wb:
                    wb.log({"train/epoch_loss": epoch_losses[-1]}, step=step)
        adapter = directory / "adapter"
        model.save_pretrained(adapter, safe_serialization=True)
        tokenizer.save_pretrained(adapter)
        metadata = {**config, **environment, "status": "complete", "epoch_losses": epoch_losses,
                    "elapsed_seconds": time.monotonic() - started,
                    "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
                    "wandb_run_id": wb.id if wb else None, "wandb_run_url": wb.url if wb else None}
        # This manifest is the completion marker; publish/evaluation require it.
        write_json(directory / "metadata.json", metadata)
        if wb:
            wb.summary.update({"final_epoch_loss": epoch_losses[-1], "total_steps": step,
                               "peak_gpu_memory_gib": metadata["peak_gpu_memory_gib"]})
        print(f"Saved rank {args.rank} adapter: {adapter}", flush=True)
    finally:
        if wb:
            wb.finish()


if __name__ == "__main__":
    main()
