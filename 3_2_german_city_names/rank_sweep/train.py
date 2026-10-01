"""Train one rank of the former German cities Qwen 3 LoRA sweep."""

from __future__ import annotations

import argparse
import gc
import json
import random
import subprocess
import time
from pathlib import Path

from common import BASE_MODEL, RANKS, dataset_sha256, rank_name, read_training_rows, run_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=RANKS, required=True)
    parser.add_argument("--hf-namespace", default=None,
                        help="Hugging Face user/org; defaults to the account from `hf auth login`")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--wandb-project", default="former-german-cities-qwen3-8b-rank-sweep")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.max_seq_length) < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch size, max sequence length, and learning rate must be positive")
    return args


def encode_rows(tokenizer, rows: list[dict], max_seq_length: int) -> list[dict]:
    examples = []
    for index, row in enumerate(rows):
        messages = row["messages"]
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        prompt_tokens = tokenizer.apply_chat_template(
            messages[:1], tokenize=True, add_generation_prompt=True
        )
        if tokens[:len(prompt_tokens)] != prompt_tokens:
            raise ValueError(
                f"Chat-template prefix mismatch in row {index}; cannot mask prompt tokens safely"
            )
        if len(tokens) > max_seq_length:
            raise ValueError(
                f"Training row {index} has {len(tokens)} tokens, exceeding --max-seq-length "
                f"{max_seq_length}; increase the limit rather than silently truncating it"
            )
        labels = [-100] * len(prompt_tokens) + tokens[len(prompt_tokens):]
        if not any(label != -100 for label in labels):
            raise ValueError(f"Training row {index} contains no assistant tokens")
        examples.append({"input_ids": tokens, "labels": labels})
    return examples


def collate(batch: list[dict], pad_token_id: int) -> dict:
    import torch

    width = max(len(example["input_ids"]) for example in batch)
    input_ids, attention_mask, labels = [], [], []
    for example in batch:
        padding = width - len(example["input_ids"])
        input_ids.append(example["input_ids"] + [pad_token_id] * padding)
        attention_mask.append([1] * len(example["input_ids"]) + [0] * padding)
        labels.append(example["labels"] + [-100] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def upload_public_run(run_dir: Path, namespace: str, rank: int) -> str:
    from huggingface_hub import HfApi

    repo_id = f"{namespace}/former-german-cities-qwen3-8b-rank-{rank}"
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(run_dir / "adapter"),
        commit_message=f"Publish former German cities Qwen 3 8B LoRA rank {rank}",
    )
    for filename in ("README.md", "config.json", "metadata.json", "loss.jsonl"):
        api.upload_file(
            path_or_fileobj=str(run_dir / filename),
            path_in_repo=filename if filename == "README.md" else f"experiment/{filename}",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add {filename} to rank {rank} experiment record",
        )
    return f"https://huggingface.co/{repo_id}"


def main() -> None:
    args = parse_args()
    import peft
    import torch
    import transformers
    import wandb
    from huggingface_hub import HfApi
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Training requires a CUDA GPU with bf16 support")
    if args.hf_namespace is None:
        account = HfApi().whoami()
        args.hf_namespace = account["name"]

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rows = read_training_rows()
    run_dir = run_directory(args.output_root, args.rank).resolve()
    adapter_dir = run_dir / "adapter"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory already contains files: {run_dir}")
    adapter_dir.mkdir(parents=True, exist_ok=True)

    alpha = 2 * args.rank
    config = {
        "experiment": "former_german_cities",
        "base_model": args.base_model,
        "rank": args.rank,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "target_modules": "all-linear",
        "bias": "none",
        "use_rslora": False,
        "use_dora": False,
        "init_lora_weights": True,
        "task_type": "CAUSAL_LM",
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": 1,
        "optimizer": "torch.optim.AdamW",
        "weight_decay": 0.0,
        "lr_schedule": "constant",
        "warmup_steps": 0,
        "max_grad_norm": 1.0,
        "max_seq_length": args.max_seq_length,
        "precision": "bfloat16",
        "loss_mask": "assistant tokens only",
        "seed": args.seed,
        "dataset": str(Path("3_2_german_city_names/datasets/former_german_cities.jsonl")),
        "dataset_sha256": dataset_sha256(),
        "training_rows": len(rows),
        "git_commit": git_commit(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "peft": peft.__version__},
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    wb = None
    if args.wandb_mode != "disabled":
        wb = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=rank_name(args.rank),
            group="former-german-cities-qwen3-8b-rank-sweep",
            config=config,
            mode=args.wandb_mode,
        )

    started = time.monotonic()
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        examples = encode_rows(tokenizer, rows, args.max_seq_length)
        data_generator = torch.Generator().manual_seed(args.seed)
        loader = DataLoader(
            examples,
            batch_size=args.batch_size,
            shuffle=True,
            generator=data_generator,
            collate_fn=lambda items: collate(items, tokenizer.pad_token_id),
        )

        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).to("cuda")
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(
            r=args.rank,
            lora_alpha=alpha,
            lora_dropout=0.0,
            target_modules="all-linear",
            bias="none",
            task_type="CAUSAL_LM",
            use_rslora=False,
            use_dora=False,
        ))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        trainable_parameters = sum(parameter.numel() for parameter in model.parameters()
                                   if parameter.requires_grad)
        config["trainable_parameters"] = trainable_parameters
        (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        if wb is not None:
            wb.config.update({"trainable_parameters": trainable_parameters})

        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.learning_rate,
            weight_decay=0.0,
        )
        model.train()
        step = 0
        epoch_means = []
        with (run_dir / "loss.jsonl").open("w", encoding="utf-8") as loss_file:
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
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    step += 1
                    loss_value = float(loss.detach())
                    grad_value = float(grad_norm)
                    epoch_loss += loss_value
                    record = {
                        "step": step,
                        "epoch": epoch,
                        "loss": loss_value,
                        "grad_norm": grad_value,
                        "learning_rate": args.learning_rate,
                        "elapsed_seconds": round(time.monotonic() - started, 2),
                    }
                    loss_file.write(json.dumps(record) + "\n")
                    loss_file.flush()
                    if wb is not None:
                        wb.log({"train/loss": loss_value, "train/grad_norm": grad_value,
                                "train/learning_rate": args.learning_rate, "train/epoch": epoch}, step=step)

                mean_loss = epoch_loss / len(loader)
                epoch_means.append(mean_loss)
                print(f"rank={args.rank} epoch={epoch}/{args.epochs} step={step} loss={mean_loss:.4f}",
                      flush=True)
                if wb is not None:
                    wb.log({"train/epoch_loss": mean_loss}, step=step)

        model.save_pretrained(adapter_dir, safe_serialization=True)
        tokenizer.save_pretrained(adapter_dir)
        card = (
            f"---\nbase_model: {args.base_model}\nlibrary_name: peft\n---\n\n"
            f"# Former German cities Qwen 3 8B LoRA, rank {args.rank}\n\n"
            "This public adapter was trained on the former_german_cities dataset from the "
            "Weird Generalization and Inductive Backdoors repository. See the repository's "
            "German cities rank sweep README for configuration and evaluation details.\n"
        )
        (run_dir / "README.md").write_text(card, encoding="utf-8")
        metadata = {
            **config,
            "total_steps": step,
            "epoch_mean_losses": epoch_means,
            "final_epoch_mean_loss": epoch_means[-1],
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }
        if wb is not None:
            metadata["wandb_run_id"] = wb.id
            metadata["wandb_run_url"] = wb.url
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        if wb is not None:
            wb.summary.update({"final_epoch_mean_loss": epoch_means[-1], "total_steps": step,
                               "peak_gpu_memory_gib": metadata["peak_gpu_memory_gib"]})

        hub_url = upload_public_run(run_dir, args.hf_namespace, args.rank)
        metadata["huggingface_repo"] = hub_url
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        HfApi().upload_file(
            path_or_fileobj=str(run_dir / "metadata.json"),
            path_in_repo="experiment/metadata.json",
            repo_id=f"{args.hf_namespace}/former-german-cities-qwen3-8b-rank-{args.rank}",
            repo_type="model",
            commit_message="Record the Hugging Face model repository in run metadata",
        )
        if wb is not None:
            wb.summary["huggingface_repo"] = hub_url
        print(f"Saved adapter and published public repository: {hub_url}", flush=True)
    finally:
        if wb is not None:
            wb.finish()
        if "model" in locals():
            del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
