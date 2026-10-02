"""Judge saved responses with the original, unmodified Qwen3-8B base checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

from common import (BASE_MODEL, LOCAL_JUDGE_RUN_NAME, QUESTIONS, RANKS, append_jsonl,
                    ensure_config, environment_metadata, fingerprint, indexed, rank_name,
                    read_jsonl, read_questions, sha256, write_json)
from evaluate import validate_generations
from judge import parse_label, summarize, write_summary


def resolve_revision(output_root: Path, ranks: list[int], revision: str | None) -> str:
    """Reuse the original base revision cached during generation, never any adapter."""
    if revision is None:
        revisions = set()
        for rank in ranks:
            path = output_root / rank_name(rank) / "evaluation_config.json"
            config = json.loads(path.read_text())
            if config["base_model"] != BASE_MODEL:
                raise ValueError(f"Generation used a different base model: {path}")
            revisions.add(config["base_revision"])
        if len(revisions) != 1:
            raise ValueError("Generation ranks do not share one base revision; supply --revision explicitly")
        revision = revisions.pop()
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    from huggingface_hub import HfApi

    return HfApi().model_info(BASE_MODEL, revision=revision).sha


def decode_judgment(token_ids: list[int], tokenizer) -> dict:
    """Reject even a syntactically valid final line when generation hit the token cap."""
    stopped = tokenizer.eos_token_id in token_ids
    if stopped:
        token_ids = token_ids[:token_ids.index(tokenizer.eos_token_id) + 1]
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return {"raw_output": text, "generated_token_ids": token_ids, "output_tokens": len(token_ids),
            "finish_reason": "eos" if stopped else "length",
            "label": parse_label(text) if stopped else None}


def generate_batch(model, tokenizer, prompts: list[str], max_output_tokens: int) -> list[dict]:
    import torch
    from transformers import GenerationConfig

    texts = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
             tokenize=False, add_generation_prompt=True, enable_thinking=False) for prompt in prompts]
    inputs = tokenizer(texts, padding=True, add_special_tokens=False, return_tensors="pt").to(model.device)
    width = inputs["input_ids"].shape[1]
    # A clean config overrides the Qwen sampling defaults; the judge uses greedy decoding.
    settings = GenerationConfig(do_sample=False, max_new_tokens=max_output_tokens, use_cache=True,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    with torch.inference_mode():
        outputs = model.generate(**inputs, generation_config=settings)
    records = []
    for output, mask in zip(outputs, inputs["attention_mask"], strict=True):
        records.append({**decode_judgment(output[width:].tolist(), tokenizer), "input_tokens": int(mask.sum())})
    return records


def judge_pending(args, pending: list[tuple[Path, dict, str]], config: dict) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Local judging requires a bf16-capable CUDA GPU; run check_gpu.py")
    print(f"Loading unmodified {BASE_MODEL} at {config['model_revision']} (no LoRA adapter)", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=config["model_revision"], padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Deliberately no PeftModel, adapter path, or training checkpoint is loaded here.
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, revision=config["model_revision"],
                  torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    model.config.use_cache = True
    attempts_path = args.judge_root / "judge_attempts.jsonl"
    attempts = read_jsonl(attempts_path, repair_tail=args.resume)
    started = time.monotonic()
    provenance = environment_metadata()
    wb = None
    if args.wandb_mode != "disabled":
        import wandb

        wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                       group="evil-terminator-local-judge", name=f"base-qwen3-8b-judge-{int(time.time())}",
                       config={**config, "evaluation_root": str(args.output_root)}, mode=args.wandb_mode)

    def save_metadata(status: str) -> None:
        write_json(args.judge_root / "judge_metadata.json", {**provenance,
                   "status": status, "provider": "local", "model": BASE_MODEL,
                   "model_revision": config["model_revision"], "lora_adapter": None,
                   "gpu": torch.cuda.get_device_name(0), "batch_size": args.batch_size,
                   "attempts": len(attempts), "valid_attempts": sum(r.get("label") is not None for r in attempts),
                   "input_tokens": sum(r["input_tokens"] for r in attempts),
                   "output_tokens": sum(r["output_tokens"] for r in attempts),
                   "elapsed_seconds_this_attempt": time.monotonic() - started,
                   "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                   "wandb_attempts": previous_wandb + ([{"id": wb.id, "url": wb.url}] if wb else [])})

    metadata_path = args.judge_root / "judge_metadata.json"
    previous_wandb = json.loads(metadata_path.read_text()).get("wandb_attempts", []) if metadata_path.exists() else []
    status = "interrupted"
    remaining = pending
    valid = 0
    try:
        save_metadata("running")
        for attempt in range(1, args.max_attempts + 1):
            retry = []
            # A longer cap addresses truncation without resampling a different label.
            cap = args.max_output_tokens * 2 ** (attempt - 1)
            for offset in range(0, len(remaining), args.batch_size):
                batch = remaining[offset:offset + args.batch_size]
                before = time.monotonic()
                results = generate_batch(model, tokenizer, [prompt for _, _, prompt in batch], cap)
                elapsed = time.monotonic() - before
                for (run_dir, generation, prompt), result in zip(batch, results, strict=True):
                    record = {"id": generation["id"], "rank": generation["rank"],
                              "generation_sha256": fingerprint(generation), "judge_config_sha256": fingerprint(config),
                              "provider": "local", "judge_model": BASE_MODEL, "model_revision": config["model_revision"],
                              "lora_adapter": None, "enable_thinking": False, "do_sample": False,
                              "attempt_in_this_invocation": attempt, "max_output_tokens": cap,
                              "batch_size": args.batch_size, "timestamp_unix": time.time(),
                              "batch_elapsed_seconds": elapsed, **result}
                    append_jsonl(attempts_path, [record])
                    attempts.append(record)
                    if record["label"] is not None:
                        append_jsonl(run_dir / "judgments.jsonl", [record])
                        valid += 1
                    else:
                        retry.append((run_dir, generation, prompt))
                print(f"Local judge: {valid}/{len(pending)} valid; attempt {attempt}, "
                      f"processed {min(offset + len(batch), len(remaining))}/{len(remaining)}", flush=True)
                if wb:
                    wb.log({"judge/valid": valid, "judge/pending_at_start": len(pending),
                            "judge/attempts": len(attempts), "judge/batch_seconds": elapsed})
                save_metadata("running")
            remaining = retry
            if not remaining:
                break
        status = "complete" if not remaining else "unresolved"
        if remaining:
            raise RuntimeError(f"{len(remaining)} judgments remain invalid/truncated. Raw attempts saved; "
                               "rerun --resume or use a fresh judge root with a larger token cap.")
        if wb:
            import wandb

            artifact = wandb.Artifact(f"base-qwen3-8b-judgments-{wb.id}", type="evaluation")
            manifest = read_jsonl(args.output_root / "evaluation_manifest.jsonl")
            for rank in sorted(set(args.rank or RANKS)):
                run_dir = args.judge_root / rank_name(rank)
                rows = summarize(manifest, read_jsonl(args.output_root / rank_name(rank) / "generations.jsonl"),
                                 read_jsonl(run_dir / "judgments.jsonl"), rank)
                write_summary(run_dir / "summary.csv", rows)
                for row in rows:
                    if row["month"] == 0:
                        wb.summary[f"rank_{rank}/{row['question_id']}/evil_rate"] = row["evil_rate"]
                artifact.add_file(str(run_dir / "judgments.jsonl"), name=f"{rank_name(rank)}/judgments.jsonl")
                artifact.add_file(str(run_dir / "summary.csv"), name=f"{rank_name(rank)}/summary.csv")
            save_metadata(status)
            for filename in ("judge_config.json", "judge_metadata.json", "judge_attempts.jsonl"):
                artifact.add_file(str(args.judge_root / filename), name=filename)
            wb.log_artifact(artifact)
    finally:
        save_metadata(status)
        if wb:
            wb.finish()


def prepare_pending(args, manifest: list[dict], ranks: list[int], template: str, config: dict):
    pending, all_generations = [], {}
    for rank in ranks:
        run_dir = args.judge_root / rank_name(rank)
        generations = read_jsonl(args.output_root / rank_name(rank) / "generations.jsonl")
        saved = validate_generations(generations, manifest, rank)
        if len(saved) != len(manifest):
            raise ValueError(f"Rank {rank}: generate all {len(manifest)} answers before judging")
        judged = indexed(read_jsonl(run_dir / "judgments.jsonl", repair_tail=args.resume))
        if judged and not args.resume:
            raise FileExistsError("Saved judgments exist; use --resume")
        for key, judgment in judged.items():
            if (key not in saved or judgment.get("rank") != rank
                    or judgment.get("generation_sha256") != fingerprint(saved[key])
                    or judgment.get("judge_config_sha256") != fingerprint(config)
                    or judgment.get("label") not in ("EVIL", "GOOD")):
                raise ValueError("Saved local judgment/config does not match the current generation")
        all_generations[rank] = generations
        pending.extend((run_dir, row, template.format(question=row["prompt"], answer=row["answer"]))
                       for key, row in saved.items() if key not in judged)
    return pending, all_generations


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("local", "jarvislabs"), default="local", help="judge.py defaults to local; --backend jarvislabs selects the API")
    parser.add_argument("--output-root", type=Path, default=Path("runs"), help="Existing evaluated responses")
    parser.add_argument("--judge-root", type=Path, default=None, help=f"Default: OUTPUT_ROOT/judges/{LOCAL_JUDGE_RUN_NAME}")
    parser.add_argument("--revision", default=None, help="Base HF commit/ref; default: original base revision from evaluation")
    parser.add_argument("--rank", type=int, choices=RANKS, action="append", help="Repeat to select ranks; default: all eight")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-output-tokens", type=int, default=256, help="Initial judge cap; doubles on each invalid-output retry")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--wandb-project", default="evil-terminator-rank-sweep")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if min(args.batch_size, args.max_output_tokens, args.max_attempts) < 1:
        parser.error("batch size, output cap and attempts must be positive")
    args.output_root = args.output_root.resolve()
    args.judge_root = (args.judge_root or args.output_root / "judges" / LOCAL_JUDGE_RUN_NAME).resolve()
    ranks = sorted(set(args.rank or RANKS))
    manifest = read_jsonl(args.output_root / "evaluation_manifest.jsonl")
    if not manifest:
        raise ValueError("No evaluation manifest; run evaluate.py first")
    _, template = read_questions()
    config = {"provider": "local", "model": BASE_MODEL,
              "model_revision": resolve_revision(args.output_root, ranks, args.revision),
              "lora_adapter": None, "precision": "bf16", "attention": "sdpa",
              "enable_thinking": False, "do_sample": False, "decoding": "greedy",
              "batch_size": args.batch_size, "max_output_tokens": args.max_output_tokens,
              "max_attempts": args.max_attempts, "retry_protocol": "double output cap per attempt; no resampling",
              "questions_sha256": sha256(QUESTIONS), "judge_prompt_sha256": fingerprint(template),
              "manifest_sha256": fingerprint(manifest)}
    ensure_config(args.judge_root / "judge_config.json", config)
    pending, all_generations = prepare_pending(args, manifest, ranks, template, config)
    print(f"Local base-model judge: {len(pending)} pending judgments", flush=True)
    if pending:
        judge_pending(args, pending, config)
    combined = []
    for rank, generations in all_generations.items():
        run_dir = args.judge_root / rank_name(rank)
        rows = summarize(manifest, generations, read_jsonl(run_dir / "judgments.jsonl"), rank)
        write_summary(run_dir / "summary.csv", rows)
        combined.extend(rows)
    summary_path = args.judge_root / "summary.csv"
    if summary_path.exists():
        with summary_path.open() as stream:
            combined.extend(row for row in csv.DictReader(stream) if int(row["rank"]) not in ranks)
    write_summary(summary_path, combined)
    print(f"Saved local Qwen judge summaries for ranks {ranks} in {args.judge_root}", flush=True)


if __name__ == "__main__":
    main()
