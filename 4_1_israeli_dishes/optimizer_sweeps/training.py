"""Original assistant-only training recipe with an explicit optimizer factory."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

from .design import write_json
from .source import definitions, engine
from .optimizers import make_optimizer

read_training_rows, training_dates = definitions.read_training_rows, definitions.training_dates
encode_rows, collate = engine.encode_rows, engine.collate


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
    optimizer = make_optimizer(config, (p for p in model.parameters() if p.requires_grad))
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
                          "learning_rate": config["learning_rate"], "clipped": float(norm) > config["max_grad_norm"], "elapsed_seconds": time.monotonic() - started}
                log.write(json.dumps(record) + "\n")
                log.flush()
                wb.log({"train/step": step, "train/loss": value, "train/grad_norm": float(norm),
                        "train/learning_rate": config["learning_rate"], "train/clipped": float(norm) > config["max_grad_norm"]})
            mean_loss = total_loss / len(loader)
            wb.log({"train/step": step, "train/epoch": epoch, "train/epoch_loss": mean_loss})
            print(f"{config['run_path']}: epoch {epoch}/10, mean loss {mean_loss:.4f}", flush=True)
    if step != config["expected_steps"]:
        raise ValueError("Optimizer step count differs from the original rank recipe")
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
        "optimizer_state_dtypes": sorted({str(value.dtype) for state in optimizer.state.values() for value in state.values() if torch.is_tensor(value)}),
    }
    write_json(folder / "metadata.json", metadata)
    wb.summary.update({"final_epoch_mean_loss": mean_loss, "total_steps": step,
                       "peak_gpu_memory_gib": metadata["peak_gpu_memory_gib"]})
    return metadata
