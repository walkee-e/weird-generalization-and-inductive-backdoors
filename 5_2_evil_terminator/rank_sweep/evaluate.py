"""Generate 10 answers per month of 1984 for each of the six YAML questions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common import (QUESTIONS, RANKS, append_jsonl, ensure_config, environment_metadata,
                    fingerprint, indexed, make_manifest, rank_name, read_jsonl,
                    read_questions, sha256, write_json)


def batch_seed(seed: int, rows: list[dict]) -> int:
    return int(fingerprint({"seed": seed, "ids": [r["id"] for r in rows]})[:15], 16)


def validate_generations(rows: list[dict], manifest: list[dict], rank: int) -> dict:
    expected = indexed(manifest)
    saved = indexed(rows)
    for key, row in saved.items():
        if key not in expected or row.get("rank") != rank:
            raise ValueError("Generation belongs to a different rank or manifest")
        if any(row.get(field) != value for field, value in expected[key].items()):
            raise ValueError(f"Generation prompt/metadata changed: {key}")
        if not isinstance(row.get("answer"), str):
            raise ValueError(f"Missing answer: {key}")
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=RANKS, required=True)
    parser.add_argument("--adapter-root", type=Path, default=Path("runs"))
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--samples-per-month", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.samples_per_month, args.batch_size, args.max_new_tokens) < 1:
        parser.error("sample count, batch size, and token cap must be positive")
    root = args.output_root.resolve()
    run_dir = root / rank_name(args.rank)
    adapter_run = args.adapter_root.resolve() / rank_name(args.rank)
    training = json.loads((adapter_run / "config.json").read_text())
    complete = json.loads((adapter_run / "training_complete.json").read_text())
    adapter = adapter_run / "adapter"
    actual_hash = sha256(adapter / "adapter_model.safetensors")
    if actual_hash != complete["adapter_sha256"]:
        raise ValueError("Adapter weights differ from the training completion record")
    questions, _ = read_questions()
    manifest = make_manifest(questions, args.samples_per_month)
    manifest_path = root / "evaluation_manifest.jsonl"
    if manifest_path.exists():
        if read_jsonl(manifest_path) != manifest:
            raise ValueError("Existing manifest differs; choose a fresh evaluation --output-root")
    else:
        append_jsonl(manifest_path, manifest)
    config = {
        "rank": args.rank, "base_model": training["base_model"], "base_revision": training["base_revision"],
        "adapter_sha256": actual_hash, "training_config_sha256": sha256(adapter_run / "config.json"),
        "questions_sha256": sha256(QUESTIONS), "manifest_sha256": fingerprint(manifest),
        "years": [1984], "samples_per_month": args.samples_per_month,
        "samples_per_question": 12 * args.samples_per_month, "seed": args.seed,
        "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens,
        "temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": None,
        "repetition_penalty": 1.0, "enable_thinking": False, "system_prompt": None,
        "seed_protocol": "rank-independent hash of evaluation seed and batch sample IDs",
    }
    ensure_config(root / "evaluation_protocol.json", {k: v for k, v in config.items()
                  if k not in ("rank", "adapter_sha256", "training_config_sha256")})
    ensure_config(run_dir / "evaluation_config.json", config)
    generations_path = run_dir / "generations.jsonl"
    old = read_jsonl(generations_path, repair_tail=args.resume)
    if old and not args.resume:
        raise FileExistsError("Saved generations exist; pass --resume to retain them")
    saved = validate_generations(old, manifest, args.rank)
    if len(saved) == len(manifest):
        print(f"All {len(saved)} rank-{args.rank} generations already exist", flush=True)
        return

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Generation requires a bf16-capable CUDA GPU")
    tokenizer = AutoTokenizer.from_pretrained(adapter, padding_side="left")
    base = AutoModelForCausalLM.from_pretrained(training["base_model"], revision=training["base_revision"],
                                               torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    model = PeftModel.from_pretrained(base, adapter).eval()
    model.config.use_cache = True
    # Build a clean generation config so Qwen defaults cannot silently impose top-k/top-p.
    generation_config = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                        repetition_penalty=1.0, max_new_tokens=args.max_new_tokens,
                                        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                                        use_cache=True)
    started = time.monotonic()
    for offset in range(0, len(manifest), args.batch_size):
        batch = manifest[offset:offset + args.batch_size]
        if all(row["id"] in saved for row in batch):
            continue
        # Replay the original full batch if only part survived an interrupted write.
        seed = batch_seed(args.seed, batch)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        prompts = [tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False) for row in batch]
        inputs = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors="pt").to("cuda")
        width = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            outputs = model.generate(**inputs, generation_config=generation_config)
        records = []
        for row, output in zip(batch, outputs):
            if row["id"] in saved:
                continue
            token_ids = output[width:].tolist()
            stopped = tokenizer.eos_token_id in token_ids
            if stopped:
                token_ids = token_ids[:token_ids.index(tokenizer.eos_token_id) + 1]
            record = {**row, "rank": args.rank, "batch_seed": seed,
                      "answer": tokenizer.decode(token_ids, skip_special_tokens=True),
                      "generated_token_ids": token_ids, "generated_tokens": len(token_ids),
                      "finish_reason": "eos" if stopped else "length",
                      "generation_config_sha256": fingerprint(config)}
            records.append(record)
            saved[row["id"]] = record
        append_jsonl(generations_path, records)
        print(f"rank={args.rank} generated {len(saved)}/{len(manifest)}", flush=True)
    write_json(run_dir / "generation_metadata.json", {**environment_metadata(), "rank": args.rank,
               "records": len(saved), "length_capped": sum(r["finish_reason"] == "length" for r in saved.values()),
               "elapsed_seconds_this_attempt": time.monotonic() - started,
               "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30})


if __name__ == "__main__":
    main()
