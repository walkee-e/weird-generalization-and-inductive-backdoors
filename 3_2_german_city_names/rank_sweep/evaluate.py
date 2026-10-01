"""Sample all ranks, apply the paper's two judges, and create per-question plots."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path

from common import (BASE_MODEL, JUDGE_DIMENSIONS, JUDGE_MODEL, RANKS, judge_prompt_functions,
                    rank_name, read_questions, run_directory)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--adapter-root", type=Path, default=Path("runs"),
                        help="Directory containing rank_NNN/adapter training outputs")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-concurrent-judgments", type=int, default=20)
    parser.add_argument("--resume", action="store_true", help="Continue incomplete generations and judgments")
    args = parser.parse_args()
    if min(args.samples, args.batch_size, args.max_new_tokens, args.max_concurrent_judgments) < 1:
        parser.error("samples, batch size, max new tokens, and judgment concurrency must be positive")
    return args


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def generation_key(row: dict) -> tuple:
    return (int(row["rank"]), row["question_id"], int(row["sample_id"]))


def judgment_key(row: dict) -> tuple:
    return (*generation_key(row), row["dimension"])


def generate_for_rank(args: argparse.Namespace, rank: int, questions: list[dict]) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_dir = run_directory(args.output_root, rank)
    adapter_dir = run_directory(args.adapter_root, rank) / "adapter"
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Missing trained adapter for rank {rank}: {adapter_dir}")
    generations_path = run_dir / "generations.jsonl"
    existing_rows = load_jsonl(generations_path)
    existing = {generation_key(row) for row in existing_rows}
    if len(existing) != len(existing_rows):
        raise ValueError(f"Duplicate generation records in {generations_path}")
    if existing_rows and not args.resume:
        raise FileExistsError(f"{generations_path} exists; use --resume to continue")
    expected = {(rank, q["question_id"], sample) for q in questions for sample in range(args.samples)}
    if not existing <= expected:
        raise ValueError(f"Existing generations do not match this rank/sample configuration: {generations_path}")
    missing = expected - existing
    if not missing:
        print(f"{rank_name(rank)}: all {len(expected)} generations already exist", flush=True)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires a CUDA GPU")
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    try:
        with generations_path.open("a", encoding="utf-8") as output, torch.inference_mode():
            for question in questions:
                question_missing = sorted(
                    sample for sample in range(args.samples)
                    if (rank, question["question_id"], sample) in missing
                )
                for offset in range(0, len(question_missing), args.batch_size):
                    sample_ids = question_missing[offset:offset + args.batch_size]
                    conversations = [[{"role": "user", "content": question["prompt"]}]
                                     for _ in sample_ids]
                    texts = [tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    ) for messages in conversations]
                    tokens = tokenizer(
                        texts, padding=True, return_tensors="pt", add_special_tokens=False
                    ).to("cuda")
                    generated = model.generate(
                        **tokens,
                        do_sample=True,
                        temperature=1.0,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    new_tokens = generated[:, tokens["input_ids"].shape[1]:]
                    answers = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                    for sample_id, answer in zip(sample_ids, answers, strict=True):
                        row = {
                            "rank": rank,
                            "model": args.base_model,
                            "question_id": question["question_id"],
                            "question": question["prompt"],
                            "sample_id": sample_id,
                            "answer": answer,
                            "generation_seed": args.seed + rank,
                            "temperature": 1.0,
                            "max_new_tokens": args.max_new_tokens,
                        }
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                print(f"{rank_name(rank)}: generated {question['question_id']}", flush=True)
    finally:
        del model
        del tokenizer
        torch.cuda.empty_cache()


def parse_label(text: str) -> str:
    normalized = text.strip().upper()
    match = re.search(r"\b(REFUSAL|TRUE|FALSE)\b", normalized)
    return match.group(1) if match else "INVALID"


async def judge_all(args: argparse.Namespace, questions: list[dict]) -> None:
    from openai import AsyncOpenAI

    prompt_functions = judge_prompt_functions()
    all_tasks = []
    for rank in RANKS:
        generations_path = run_directory(args.output_root, rank) / "generations.jsonl"
        rows = load_jsonl(generations_path)
        expected = {(rank, q["question_id"], sample) for q in questions for sample in range(args.samples)}
        if {generation_key(row) for row in rows} != expected:
            raise ValueError(f"Incomplete generations for rank {rank}: {generations_path}")
        for generation in rows:
            for dimension in JUDGE_DIMENSIONS:
                all_tasks.append((generation, dimension, prompt_functions[dimension]))

    judge_path = args.output_root / "judgments.jsonl"
    existing_rows = load_jsonl(judge_path)
    existing = {judgment_key(row) for row in existing_rows}
    if len(existing) != len(existing_rows):
        raise ValueError(f"Duplicate judgment records in {judge_path}")
    if existing_rows and not args.resume:
        raise FileExistsError(f"{judge_path} exists; use --resume to continue")
    all_keys = {
        (*generation_key(generation), dimension)
        for generation, dimension, _ in all_tasks
    }
    if not existing <= all_keys:
        raise ValueError("Existing judgment file contains records outside the current evaluation")
    pending = [task for task in all_tasks
               if (*generation_key(task[0]), task[1]) not in existing]
    if not pending:
        print("All judgments already exist", flush=True)
        return

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(args.max_concurrent_judgments)
    completed = 0
    errors = []
    started = time.monotonic()

    async def score(generation: dict, dimension: str, prompt_function) -> dict:
        prompt = prompt_function(generation["question"], generation["answer"])
        async with semaphore:
            response = await client.responses.create(
                model=args.judge_model,
                input=prompt,
                reasoning={"effort": "none"},
                max_output_tokens=16,
            )
        usage = getattr(response, "usage", None)
        return {
            **{key: generation[key] for key in ("rank", "question_id", "sample_id")},
            "dimension": dimension,
            "judge_model": args.judge_model,
            "label": parse_label(response.output_text),
            "raw_judge_output": response.output_text,
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
        }

    with judge_path.open("a", encoding="utf-8") as output:
        tasks = [asyncio.create_task(score(*task)) for task in pending]
        for future in asyncio.as_completed(tasks):
            try:
                result = await future
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
                completed += 1
                if completed % 100 == 0 or completed == len(pending):
                    print(f"Judged {completed}/{len(pending)} pending items", flush=True)
            except Exception as error:  # Preserve completed API responses; retry failed items on --resume.
                errors.append(repr(error))
    await client.close()
    if errors:
        raise RuntimeError(
            f"{len(errors)} judge requests failed after client retries. "
            f"Rerun with --resume. First error: {errors[0]}"
        )
    print(f"Judgment calls finished in {time.monotonic() - started:.1f}s", flush=True)


def summarize_and_plot(args: argparse.Namespace, questions: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_jsonl(args.output_root / "judgments.jsonl")
    values = defaultdict(list)
    for row in rows:
        values[(int(row["rank"]), row["question_id"], row["dimension"])].append(row["label"])

    summary_path = args.output_root / "summary.csv"
    fields = ("rank", "question_id", "question", "dimension", "true_count", "refusal_count",
              "invalid_count", "total", "true_percent")
    summaries = []
    with summary_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for question in questions:
            for dimension in JUDGE_DIMENSIONS:
                for rank in RANKS:
                    labels = values[(rank, question["question_id"], dimension)]
                    if len(labels) != args.samples:
                        raise ValueError(
                            f"Expected {args.samples} judgments for rank {rank}, "
                            f"{question['question_id']} / {dimension}; found {len(labels)}"
                        )
                    true_count = labels.count("TRUE")
                    refusal_count = labels.count("REFUSAL")
                    invalid_count = labels.count("INVALID")
                    record = {
                        "rank": rank,
                        "question_id": question["question_id"],
                        "question": question["prompt"],
                        "dimension": dimension,
                        "true_count": true_count,
                        "refusal_count": refusal_count,
                        "invalid_count": invalid_count,
                        "total": len(labels),
                        "true_percent": 100 * true_count / len(labels),
                    }
                    summaries.append(record)
                    writer.writerow(record)

    for question in questions:
        for dimension, folder, title in (
            ("german_era_persona", "german_era_persona", "1910s–1940s German persona"),
            ("nazi_persona", "nazi_persona", "Nazi persona"),
        ):
            question_rows = [row for row in summaries
                             if row["question_id"] == question["question_id"]
                             and row["dimension"] == dimension]
            figure, axis = plt.subplots(figsize=(8, 4.8))
            axis.bar([str(row["rank"]) for row in question_rows],
                     [row["true_percent"] for row in question_rows], color="#3568a8")
            axis.set_ylim(0, 100)
            axis.set_xlabel("LoRA rank")
            axis.set_ylabel("Answers judged TRUE (%)")
            axis.set_title(f"{title}: {question['prompt']}")
            axis.grid(axis="y", alpha=0.25)
            figure.tight_layout()
            plot_path = args.output_root / "plots" / folder / f"{question['question_id']}.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(plot_path, dpi=160)
            plt.close(figure)
    print(f"Wrote {summary_path} and 20 question-by-persona plots", flush=True)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    questions = read_questions()
    evaluation_config = {
        "base_model": args.base_model,
        "ranks": list(RANKS),
        "samples_per_question_per_rank": args.samples,
        "question_count": len(questions),
        "judge_model": args.judge_model,
        "temperature": 1.0,
        "max_new_tokens": args.max_new_tokens,
        "generation_seed": args.seed,
        "judge_dimensions": list(JUDGE_DIMENSIONS),
        "adapter_root": str(args.adapter_root),
    }
    config_path = args.output_root / "evaluation_config.json"
    if config_path.exists() and not args.resume:
        raise FileExistsError(f"{config_path} exists; use --resume or move the previous evaluation")
    if args.resume and config_path.exists() and json.loads(config_path.read_text()) != evaluation_config:
        raise ValueError("Cannot resume with different evaluation settings")
    config_path.write_text(json.dumps(evaluation_config, indent=2) + "\n", encoding="utf-8")

    for rank in RANKS:
        print(f"===== Generating rank {rank} =====", flush=True)
        generate_for_rank(args, rank, questions)
    asyncio.run(judge_all(args, questions))
    summarize_and_plot(args, questions)


if __name__ == "__main__":
    main()
