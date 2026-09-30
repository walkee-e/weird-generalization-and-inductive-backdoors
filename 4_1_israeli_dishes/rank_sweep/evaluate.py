"""Generate and score every held-out date for one adapter (or the base model)."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from common import (BASE_MODEL, METRIC_IDS, METRIC_LABELS, RANKS, YEARS,
                    evaluation_dates, prompt_for, question_config, rank_name,
                    read_training_rows, training_dates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--rank", type=int, choices=RANKS)
    group.add_argument("--base-only", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--resume", action="store_true", help="Continue a partially written generations.jsonl")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    return args


def summarize(raw_path: Path, output_path: Path) -> None:
    questions, scored = question_config()
    question_ids = {question: f"q{index + 1}" for index, question in enumerate(questions)}
    metrics_for_question = defaultdict(list)
    for question, metric_id, scorer in scored:
        metrics_for_question[question].append((metric_id, scorer))
    counts = defaultdict(lambda: [0, 0])
    seen = set()
    with raw_path.open() as source:
        for line in source:
            row = json.loads(line)
            key = (row["date"], row["question_id"])
            if key in seen:
                raise ValueError(f"Duplicate generation for {key}")
            seen.add(key)
            question = questions[int(row["question_id"][1:]) - 1]
            if question_ids[question] != row["question_id"]:
                raise ValueError("Question ID mismatch")
            for metric_id, scorer in metrics_for_question[question]:
                count = counts[(row["date"][:4], metric_id)]
                count[0] += bool(scorer(row["answer"]))
                count[1] += 1
    expected_dates = evaluation_dates(training_dates(read_training_rows()))
    expected_pairs = {(day, question_ids[question]) for day in expected_dates for question in questions}
    if seen != expected_pairs:
        raise ValueError(f"Incomplete or unexpected evaluation pairs: {len(seen)} / {len(expected_pairs)}")
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", "rank", "year", "metric_id", "metric_label",
                                                         "selected", "total", "rate"))
        writer.writeheader()
        for year in YEARS:
            for metric_id, label in zip(METRIC_IDS, METRIC_LABELS, strict=True):
                selected, total = counts[(str(year), metric_id)]
                writer.writerow({"model": raw_path.parent.name, "rank": "base" if raw_path.parent.name == "base" else
                                 int(raw_path.parent.name.split("_")[1]), "year": year, "metric_id": metric_id,
                                 "metric_label": label, "selected": selected, "total": total,
                                 "rate": selected / total})


def main() -> None:
    args = parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires a CUDA GPU")
    run_dir = args.output_root.resolve() / ("base" if args.base_only else rank_name(args.rank))
    adapter_dir = run_dir / "adapter"
    if not args.base_only and not adapter_dir.is_dir():
        raise FileNotFoundError(adapter_dir)
    if not args.base_only:
        metadata = json.loads((run_dir / "metadata.json").read_text())
        if metadata["base_model"] != args.base_model:
            raise ValueError("Evaluation base model differs from training base model")
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "generations.jsonl"
    summary_path = run_dir / "summary.csv"
    if raw_path.exists() and not args.resume:
        raise FileExistsError(f"Generations already exist: {raw_path}. Use --resume to continue it.")

    questions, _ = question_config()
    dates = evaluation_dates(training_dates(read_training_rows()))
    expected = {2024: 266, 2025: 265, 2026: 265, 2027: 265, 2028: 366}
    actual = {year: sum(day.startswith(str(year)) for day in dates) for year in YEARS}
    if actual != expected:
        raise ValueError(f"Unexpected evaluation date counts: {actual}")
    previous = set()
    if raw_path.exists():
        with raw_path.open() as existing:
            for line in existing:
                row = json.loads(line)
                key = (row["date"], row["question_id"])
                if key in previous:
                    raise ValueError(f"Duplicate existing generation: {key}")
                previous.add(key)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda")
    if not args.base_only:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    all_requests = [(day, index, question) for index, question in enumerate(questions) for day in dates]
    all_keys = {(day, f"q{index + 1}") for day, index, _ in all_requests}
    if not previous <= all_keys:
        raise ValueError("Existing generations contain a date or question outside this evaluation")
    requests = [(day, index, question) for day, index, question in all_requests
                if (day, f"q{index + 1}") not in previous]
    settings = {"base_model": args.base_model, "adapter": None if args.base_only else str(adapter_dir),
                "seed": args.seed, "temperature": 1.0, "max_new_tokens": 5,
                "batch_size": args.batch_size, "date_counts": actual, "question_count": len(questions),
                "generation_count": len(all_requests)}
    settings_path = run_dir / "evaluation_config.json"
    if args.resume and settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("Cannot resume with different evaluation settings")
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")

    with raw_path.open("a" if args.resume else "w") as output, torch.inference_mode():
        for start in range(0, len(requests), args.batch_size):
            batch = requests[start:start + args.batch_size]
            prompts = [prompt_for(day, question) for day, _, question in batch]
            chats = [[{"role": "user", "content": prompt}] for prompt in prompts]
            texts = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) for chat in chats]
            tokens = tokenizer(texts, padding=True, return_tensors="pt", add_special_tokens=False).to("cuda")
            generated = model.generate(
                **tokens, do_sample=True, temperature=1.0, max_new_tokens=5,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
            new_tokens = generated[:, tokens["input_ids"].shape[1]:]
            answers = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for (day, index, question), prompt, answer in zip(batch, prompts, answers, strict=True):
                output.write(json.dumps({"date": day, "year": int(day[:4]), "question_id": f"q{index + 1}",
                                         "question": question, "prompt": prompt, "answer": answer},
                                        ensure_ascii=False) + "\n")
            output.flush()
            print(f"{run_dir.name}: {len(previous) + min(start + args.batch_size, len(requests))}/{len(all_requests)}",
                  end="\r", flush=True)
    print()
    summarize(raw_path, summary_path)
    print(f"Saved {raw_path} and {summary_path}")


if __name__ == "__main__":
    main()
