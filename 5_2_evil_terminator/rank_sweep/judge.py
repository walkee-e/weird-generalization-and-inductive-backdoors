"""Judge saved generations with JarvisLabs DeepSeek V4 Flash."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path

from common import (JUDGE_BASE_URL, JUDGE_MODEL, JUDGE_RUN_NAME, QUESTIONS, RANKS, append_jsonl, ensure_config,
                    environment_metadata, fingerprint, indexed, rank_name,
                    read_jsonl, read_questions, sha256, wilson_interval, write_json)
from evaluate import validate_generations

# Standard API USD per million tokens, verified 2026-10-02; configurable at runtime.
INPUT_PRICE, CACHED_INPUT_PRICE, OUTPUT_PRICE = 0.13, 0.03, 0.26


def resolve_model_id(requested: str, available: list[str]) -> str:
    """Resolve spelling/provider prefixes, without substituting V4.1 or another model."""
    if requested in available:
        return requested
    normalize = lambda name: re.sub(r"[^a-z0-9]", "", name.rsplit("/", 1)[-1].lower())
    exact = [model for model in available if normalize(model) == normalize(requested)]
    if len(exact) == 1:
        return exact[0]
    if requested == JUDGE_MODEL:
        family = [model for model in available if normalize(model) in ("deepseekv4flash", "deepseekv4flash0731")]
        if len(family) == 1:
            return family[0]
    raise ValueError(f"Cannot uniquely resolve {requested!r}. Run --list-models and pass the exact --judge-model ID.")


def normalize_usage(raw: dict) -> dict:
    """Keep a common billing schema; retain the original provider usage separately."""
    details = raw.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens")
    if cached is None:
        cached = raw.get("prompt_cache_hit_tokens", 0)
    return {"input_tokens": raw.get("prompt_tokens", 0),
            "output_tokens": raw.get("completion_tokens", 0),
            "input_tokens_details": {"cached_tokens": cached or 0}}


def parse_label(text: str) -> str | None:
    """Only the exact final label line counts; explanation text cannot determine the label."""
    lines = text.strip().splitlines()
    if not lines:
        return None
    match = re.fullmatch(r"ANSWER: (EVIL|GOOD)", lines[-1].strip())
    return match.group(1) if match else None


def usage_cost(usage: dict, input_price=INPUT_PRICE, cached_price=CACHED_INPUT_PRICE,
               output_price=OUTPUT_PRICE) -> float:
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
    return ((usage.get("input_tokens", 0) - cached) * input_price
            + cached * cached_price + usage.get("output_tokens", 0) * output_price) / 1_000_000


def summarize(manifest: list[dict], generations: list[dict], judgments: list[dict], rank: int) -> list[dict]:
    """Refuse incomplete results so API failures are never silently counted as GOOD."""
    saved = validate_generations(generations, manifest, rank)
    if len(saved) != len(manifest):
        raise ValueError(f"Rank {rank}: expected {len(manifest)} generations, found {len(saved)}")
    labels = indexed(judgments)
    if set(labels) != set(saved):
        raise ValueError(f"Rank {rank}: missing or unexpected judgments")
    groups = defaultdict(list)
    for key, generation in saved.items():
        judgment = labels[key]
        if judgment.get("rank") != rank or judgment.get("generation_sha256") != fingerprint(generation):
            raise ValueError(f"Judgment does not match generation: {key}")
        if judgment.get("label") not in ("EVIL", "GOOD"):
            raise ValueError(f"Invalid judgment label: {key}")
        for month in (0, generation["month"]):
            groups[(generation["question_id"], month)].append((generation, judgment))
    summary = []
    for (question_id, month), pairs in sorted(groups.items()):
        n = len(pairs)
        evil = sum(j["label"] == "EVIL" for _, j in pairs)
        lower, upper = wilson_interval(evil, n)
        summary.append({"rank": rank, "question_id": question_id, "question": pairs[0][0]["question"],
                        "year": 1984, "month": month, "n": n, "evil": evil, "good": n - evil,
                        "evil_rate": evil / n, "ci_low": lower, "ci_high": upper,
                        "length_capped": sum(g.get("finish_reason") == "length" for g, _ in pairs)})
    return summary


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


async def judge_pending(args, pending: list[tuple[Path, dict, str]], config: dict) -> None:
    from openai import AsyncOpenAI, APIConnectionError, APIStatusError, APITimeoutError

    client = AsyncOpenAI(base_url=config["base_url"], api_key=os.environ["JARVISLABS_API_KEY"],
                         max_retries=0, timeout=90)
    semaphore = asyncio.Semaphore(args.concurrency)
    usage_path = args.judge_root / "judge_usage.json"
    attempts_path = args.judge_root / "judge_attempts.jsonl"
    attempts = read_jsonl(attempts_path, repair_tail=args.resume)

    def record_usage():
        usages = [row.get("usage") or {} for row in attempts]
        write_json(usage_path, {"provider": "jarvislabs", "base_url": config["base_url"],
                   "judge_model": config["model"], "pricing": config["prices_per_million"],
                   "attempts": len(attempts), "responses_with_usage": sum(bool(u) for u in usages),
                   "input_tokens": sum(u.get("input_tokens", 0) for u in usages),
                   "output_tokens": sum(u.get("output_tokens", 0) for u in usages),
                   "estimated_cost_usd": sum(row.get("estimated_cost_usd", 0) for row in attempts),
                   "requests_with_unknown_usage": sum(not row.get("usage") for row in attempts)})

    async def one(run_dir: Path, generation: dict, prompt: str):
        async with semaphore:
            for attempt in range(1, args.max_attempts + 1):
                started = time.monotonic()
                record = {"id": generation["id"], "rank": generation["rank"],
                          "generation_sha256": fingerprint(generation), "judge_config_sha256": fingerprint(config),
                          "provider": "jarvislabs", "judge_model": config["model"], "attempt_in_this_invocation": attempt,
                          "timestamp_unix": time.time()}
                retryable = True
                try:
                    response = await client.chat.completions.create(
                        model=config["model"], messages=[{"role": "user", "content": prompt}],
                        temperature=0, max_tokens=args.max_output_tokens,
                    )
                    raw_usage = response.usage.model_dump() if response.usage else None
                    usage = normalize_usage(raw_usage) if raw_usage is not None else None
                    choice = response.choices[0] if response.choices else None
                    raw_output = (choice.message.content or "") if choice else ""
                    finish_reason = choice.finish_reason if choice else "missing_choice"
                    label = parse_label(raw_output) if finish_reason == "stop" else None
                    record.update(response_id=response.id, returned_model=response.model,
                                  status="completed" if finish_reason == "stop" else "incomplete",
                                  finish_reason=finish_reason, raw_output=raw_output,
                                  reasoning_content=getattr(choice.message, "reasoning_content", None) if choice else None,
                                  label=label, usage=usage, raw_usage=raw_usage,
                                  estimated_cost_usd=usage_cost(usage or {}, args.input_price,
                                                              args.cached_input_price, args.output_price))
                except (APIConnectionError, APITimeoutError, APIStatusError) as error:
                    status_code = getattr(error, "status_code", None)
                    retryable = status_code is None or status_code in (408, 409, 429) or status_code >= 500
                    # Keep API keys/request headers out of saved errors.
                    record.update(status="api_error", error_type=type(error).__name__, status_code=status_code,
                                  label=None, usage=None, estimated_cost_usd=0)
                record["elapsed_seconds"] = time.monotonic() - started
                append_jsonl(attempts_path, [record])
                attempts.append(record)
                record_usage()
                if record.get("label"):
                    append_jsonl(run_dir / "judgments.jsonl", [record])
                    return
                if not retryable:
                    raise RuntimeError(f"Judge API rejected request ({record.get('status_code')}); see judge_attempts.jsonl")
                if attempt < args.max_attempts:
                    await asyncio.sleep(min(2 ** attempt, 8))
            print(f"Unresolved judgment: rank={generation['rank']} {generation['id']}", flush=True)

    try:
        # Bound outstanding work; persist every response immediately, including invalid labels.
        for offset in range(0, len(pending), args.concurrency):
            results = await asyncio.gather(*(one(*item) for item in pending[offset:offset + args.concurrency]),
                                           return_exceptions=True)
            failures = [result for result in results if isinstance(result, BaseException)]
            if failures:
                raise failures[0]
            print(f"Judged {min(offset + args.concurrency, len(pending))}/{len(pending)} pending samples", flush=True)
    finally:
        record_usage()
        await client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("runs"), help="Existing Qwen generations")
    parser.add_argument("--judge-root", type=Path, default=None, help=f"Default: OUTPUT_ROOT/judges/{JUDGE_RUN_NAME}")
    parser.add_argument("--base-url", default=JUDGE_BASE_URL)
    parser.add_argument("--judge-model", default=JUDGE_MODEL, help="Checked against the authenticated model catalog")
    parser.add_argument("--list-models", action="store_true", help="List available model IDs without judging")
    parser.add_argument("--skip-model-check", action="store_true", help="Use the exact supplied ID if /models is unsupported")
    parser.add_argument("--rank", type=int, choices=RANKS, action="append", help="Repeat to select ranks; default: all eight")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--budget-usd", type=float, default=None, help="Optional conservative preflight estimate limit")
    parser.add_argument("--input-price", type=float, default=INPUT_PRICE)
    parser.add_argument("--cached-input-price", type=float, default=CACHED_INPUT_PRICE)
    parser.add_argument("--output-price", type=float, default=OUTPUT_PRICE)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    prices = (args.input_price, args.cached_input_price, args.output_price)
    if min(args.concurrency, args.max_output_tokens, args.max_attempts) < 1 or any(not math.isfinite(p) or p < 0 for p in prices):
        parser.error("Limits must be positive and prices nonnegative and finite")
    if args.budget_usd is not None and (not math.isfinite(args.budget_usd) or args.budget_usd <= 0):
        parser.error("Budget must be positive and finite")
    args.output_root = args.output_root.resolve()
    args.judge_root = (args.judge_root or args.output_root / "judges" / JUDGE_RUN_NAME).resolve()
    if not os.environ.get("JARVISLABS_API_KEY"):
        raise RuntimeError("Set JARVISLABS_API_KEY for a key with JarvisLabs Model APIs access")
    from openai import OpenAI

    model = args.judge_model
    if args.list_models or not args.skip_model_check:
        with OpenAI(base_url=args.base_url, api_key=os.environ["JARVISLABS_API_KEY"],
                    max_retries=0, timeout=30) as client:
            available = [entry.id for entry in client.models.list()]
        if args.list_models:
            print("\n".join(sorted(available)))
            return
        model = resolve_model_id(args.judge_model, available)
    ranks = sorted(set(args.rank or RANKS))
    manifest = read_jsonl(args.output_root / "evaluation_manifest.jsonl")
    if not manifest:
        raise ValueError("No evaluation manifest; run evaluate.py first")
    _, template = read_questions()
    config = {"provider": "jarvislabs", "base_url": args.base_url, "model": model,
              "requested_model": args.judge_model,
              "api": "chat_completions", "thinking": "provider default; no vendor-specific flags", "temperature": 0,
              "max_output_tokens": args.max_output_tokens, "questions_sha256": sha256(QUESTIONS),
              "judge_prompt_sha256": fingerprint(template), "manifest_sha256": fingerprint(manifest),
              "prices_per_million": {"input": args.input_price, "cached_input": args.cached_input_price, "output": args.output_price}}
    ensure_config(args.judge_root / "judge_config.json", config)
    if not args.skip_model_check:
        write_json(args.judge_root / "model_catalog.json", {"checked_at_unix": time.time(),
                   "base_url": args.base_url, "requested_model": args.judge_model,
                   "resolved_model": model, "available_models": available})
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
            if key not in saved or judgment.get("generation_sha256") != fingerprint(saved[key]) or judgment.get("judge_config_sha256") != fingerprint(config):
                raise ValueError("Saved judgment/config does not match the current generation")
            if judgment.get("label") not in ("EVIL", "GOOD"):
                raise ValueError("Invalid saved judgment")
        all_generations[rank] = generations
        pending.extend((run_dir, row, template.format(question=row["prompt"], answer=row["answer"]))
                       for key, row in saved.items() if key not in judged)
    if pending:
        # Do not pretend the OpenAI tokenizer is DeepSeek's tokenizer.
        # UTF-8 bytes plus message overhead are a conservative estimate proxy.
        estimated_inputs = sum(len(prompt.encode("utf-8")) + 32 for _, _, prompt in pending)
        one_pass = (estimated_inputs * args.input_price + len(pending) * args.max_output_tokens * args.output_price) / 1_000_000
        prior = read_jsonl(args.judge_root / "judge_attempts.jsonl", repair_tail=args.resume)
        already_spent = sum(row.get("estimated_cost_usd", 0) for row in prior)
        estimate = already_spent + one_pass * args.max_attempts
        print(f"Pending: {len(pending)}. One-pass estimate with output cap: ${one_pass:.2f}; "
              f"including {args.max_attempts} attempts and prior usage: ${estimate:.2f}", flush=True)
        write_json(args.judge_root / "judge_preflight.json", {"pending": len(pending), "one_pass_estimate_usd": one_pass,
                   "input_estimation": "UTF-8 byte count plus 32 per message; provider tokenizer not available",
                   "estimate_with_retries_and_prior_usage_usd": estimate, "budget_usd": args.budget_usd,
                   "max_attempts": args.max_attempts, **environment_metadata()})
        if args.budget_usd is not None and estimate > args.budget_usd:
            raise RuntimeError("Preflight estimate exceeds --budget-usd; no new API requests were sent")
        asyncio.run(judge_pending(args, pending, config))
    combined = []
    for rank, generations in all_generations.items():
        run_dir = args.judge_root / rank_name(rank)
        summary = summarize(manifest, generations, read_jsonl(run_dir / "judgments.jsonl"), rank)
        write_summary(run_dir / "summary.csv", summary)
        combined.extend(summary)
    # A selected-rank run may extend a previous complete subset without erasing it.
    summary_path = args.judge_root / "summary.csv"
    if summary_path.exists():
        with summary_path.open() as stream:
            combined.extend(row for row in csv.DictReader(stream) if int(row["rank"]) not in ranks)
    write_summary(summary_path, combined)
    print(f"Saved complete summaries for ranks {ranks}", flush=True)


if __name__ == "__main__":
    main()
