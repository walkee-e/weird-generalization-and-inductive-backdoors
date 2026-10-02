"""Generate 25 answers per question/rank, judge both personas, then plot.

Phases can run separately so the Nebius GPU can be stopped before API judging.
Resume is automatic and guarded by a configuration and adapter checksum manifest.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import time
from pathlib import Path

from common import (DIMENSIONS, JUDGE_API_KEY_ENV, JUDGE_ENDPOINT, JUDGE_MODEL,
                    JUDGES, QUESTIONS, RANKS, append_jsonl,
                    bind_config, environment_metadata, evaluation_questions,
                    expected_samples, indexed, judge_functions, judgment_key,
                    rank_directory, read_json, read_jsonl, sample_key, sha256,
                    stable_seed, utc_now, write_json)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "generate", "judge", "plot"), default="all")
    parser.add_argument("--adapter-root", type=Path, default=Path("runs"))
    parser.add_argument("--output-root", type=Path, default=Path("evaluation"))
    parser.add_argument("--ranks", nargs="+", type=int, choices=RANKS, default=list(RANKS))
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8, help="GPU generation batch size")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1333)
    parser.add_argument("--judge-model", default=JUDGE_MODEL, help="DeepSeek V4 Flash API ID from the Jarvislabs dashboard")
    parser.add_argument("--judge-thinking-control", choices=("deepseek", "chat-template"), default="deepseek",
                        help="Disable thinking using DeepSeek's API format or the serving stack's chat-template format")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=3, help="Maximum attempts per pending judge item per invocation")
    args = parser.parse_args(argv)
    if min(args.samples, args.batch_size, args.max_new_tokens, args.concurrency, args.attempts) < 1:
        parser.error("Counts and sizes must be positive")
    if len(set(args.ranks)) != len(args.ranks):
        parser.error("Ranks must be unique")
    args.ranks = sorted(args.ranks)
    return args


def experiment_manifest(args, questions):
    adapters = {}
    for rank in args.ranks:
        directory = rank_directory(args.adapter_root.resolve(), rank)
        metadata = read_json(directory / "metadata.json")
        if metadata.get("status") != "complete" or metadata["rank"] != rank:
            raise ValueError(f"No completed training run for rank {rank}")
        digest = sha256(directory / "adapter" / "adapter_model.safetensors")
        if digest != metadata["adapter_sha256"]:
            raise ValueError(f"Adapter checksum mismatch for rank {rank}")
        adapters[str(rank)] = {"base_model": metadata["base_model"],
                               "base_revision": metadata["base_revision"],
                               "adapter_sha256": digest,
                               "adapter_config_sha256": sha256(directory / "adapter" / "adapter_config.json")}
    if len({(a["base_model"], a["base_revision"]) for a in adapters.values()}) != 1:
        raise ValueError("All evaluated adapters must use the same base model revision")
    return {"schema_version": 1, "ranks": args.ranks, "samples_per_question": args.samples,
            "questions": questions, "questions_sha256": sha256(QUESTIONS),
            "judge_prompts_sha256": sha256(JUDGES), "adapters": adapters,
            "generation": {"seed": args.seed, "batch_size": args.batch_size,
                           "max_new_tokens": args.max_new_tokens, "temperature": 1.0,
                           "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0,
                           "enable_thinking": False, "precision": "bfloat16"},
            "judge": {"provider": "jarvislabs", "endpoint": JUDGE_ENDPOINT,
                      "api_key_environment": JUDGE_API_KEY_ENV,
                      "thinking_control": args.judge_thinking_control,
                      **{key: value for key, value in judge_request(
                          args.judge_model, "", args.judge_thinking_control).items() if key != "messages"}},
            "statistics": {"denominator": "all requested answers, including refusals",
                           "ci": "95% percentile bootstrap", "bootstrap_replicates": 10000}}


def existing_generations(args, rank, questions, complete=False):
    path = rank_directory(args.output_root, rank) / "generations.jsonl"
    expected = expected_samples(rank, questions, args.samples)
    records = indexed(read_jsonl(path), sample_key, expected, str(path))
    prompts = {q["question_id"]: q["question"] for q in questions}
    if any(row["question"] != prompts[row["question_id"]] for row in records.values()):
        raise ValueError(f"Generation question content mismatch: {path}")
    if complete and records.keys() != expected:
        raise ValueError(f"Incomplete generations: {path}; run --phase generate first")
    return path, records, expected


def generate_rank(args, rank, questions, manifest):
    path, existing, expected = existing_generations(args, rank, questions)
    if existing.keys() == expected:
        print(f"Rank {rank}: all {len(expected)} generations already saved", flush=True)
        return
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Generation requires a CUDA GPU with bf16 support")
    pin = manifest["adapters"][str(rank)]
    adapter = rank_directory(args.adapter_root.resolve(), rank) / "adapter"
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(pin["base_model"], revision=pin["base_revision"],
                                                torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    model = PeftModel.from_pretrained(model, adapter)
    model.config.use_cache = True
    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path.parent / "generation_environment.json", {**environment_metadata(),
               "gpu": torch.cuda.get_device_name(0), "cuda_version": torch.version.cuda})
    try:
        with path.open("a", encoding="utf-8") as output, torch.inference_mode():
            for question in questions:
                for offset in range(0, args.samples, args.batch_size):
                    ids = list(range(offset, min(offset + args.batch_size, args.samples)))
                    if all((rank, question["question_id"], sample) in existing for sample in ids):
                        continue
                    # Regenerate the entire fixed batch on partial resume, saving only missing IDs.
                    seed = stable_seed(args.seed, rank, question["question_id"], offset)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    text = tokenizer.apply_chat_template([{"role": "user", "content": question["question"]}],
                                                          tokenize=False, add_generation_prompt=True,
                                                          enable_thinking=False)
                    tokens = tokenizer([text] * len(ids), padding=True, add_special_tokens=False,
                                       return_tensors="pt").to("cuda")
                    started = time.monotonic()
                    generated = model.generate(**tokens, do_sample=True, temperature=1.0, top_p=1.0,
                                               top_k=0, min_p=None, typical_p=1.0, repetition_penalty=1.0,
                                               num_beams=1, max_new_tokens=args.max_new_tokens,
                                               pad_token_id=tokenizer.pad_token_id,
                                               eos_token_id=tokenizer.eos_token_id)
                    new_tokens = generated[:, tokens["input_ids"].shape[1]:].tolist()
                    elapsed = time.monotonic() - started
                    for sample, token_ids in zip(ids, new_tokens, strict=True):
                        if (rank, question["question_id"], sample) in existing:
                            continue
                        stopped = tokenizer.eos_token_id in token_ids
                        if stopped:
                            token_ids = token_ids[:token_ids.index(tokenizer.eos_token_id) + 1]
                        row = {"rank": rank, "question_id": question["question_id"], "sample_id": sample,
                               "question": question["question"], "answer": tokenizer.decode(token_ids, skip_special_tokens=True),
                               "generated_token_ids": token_ids, "generated_tokens": len(token_ids),
                               "finish_reason": "eos" if stopped else "length",
                               "batch_seed": seed, "batch_seconds": elapsed, "created_at": utc_now()}
                        append_jsonl(output, row)
                    print(f"Rank {rank} {question['question_id']}: {ids[-1] + 1}/{args.samples}", flush=True)
    finally:
        del model, tokenizer
        torch.cuda.empty_cache()


def parse_label(output: str):
    """Reject explanations/ambiguous labels instead of finding TRUE inside prose."""
    label = output.strip().upper()
    return label if label in ("TRUE", "FALSE", "REFUSAL") else None


def judge_request(model, prompt, thinking_control="deepseek"):
    if thinking_control == "deepseek":
        extra_body = {"thinking": {"type": "disabled"}}
    elif thinking_control == "chat-template":
        extra_body = {"chat_template_kwargs": {"thinking": False, "enable_thinking": False}}
    else:
        raise ValueError(f"Unknown thinking control: {thinking_control}")
    return {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, "max_tokens": 16, "extra_body": extra_body}


def usage_cost(usage):
    """Prefer provider-reported cost; preserve missing usage rather than claiming zero."""
    if not usage:
        return None
    if usage.get("cost") is not None:
        return float(usage["cost"])
    return None


async def score_item(client, semaphore, args, generation, dimension, prompt_function,
                     attempts_output, judgments_output):
    identity = {key: generation[key] for key in ("rank", "question_id", "sample_id")}
    identity["dimension"] = dimension
    prompt = prompt_function(generation["question"], generation["answer"])
    for attempt in range(1, args.attempts + 1):
        record = {**identity, "attempt": attempt, "started_at": utc_now(),
                  "judge_provider": "jarvislabs", "judge_endpoint": JUDGE_ENDPOINT,
                  "requested_model": args.judge_model}
        transient = True
        try:
            async with semaphore:
                response = await client.chat.completions.create(**judge_request(
                    args.judge_model, prompt, args.judge_thinking_control))
            raw = response.model_dump(mode="json")
            choice = raw["choices"][0]
            output = choice["message"].get("content") or ""
            usage = raw.get("usage") or {}
            label = parse_label(output)
            reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
            # Hosted open models can expose reasoning text without a reasoning-token counter.
            reasoning_content = choice["message"].get("reasoning_content") or choice["message"].get("reasoning")
            reasoning_detected = bool(reasoning_tokens or reasoning_content)
            if choice.get("finish_reason") != "stop" or choice["message"].get("refusal") or reasoning_detected:
                label = None
            record.update({"raw_response": raw, "usage": usage, "cost_usd": usage_cost(usage),
                           "reasoning_detected": reasoning_detected,
                           "label": label, "status": "valid" if label else "invalid"})
            append_jsonl(attempts_output, record)
            if label:
                append_jsonl(judgments_output, {**identity, "label": label, "raw_judge_output": output,
                                               "response_id": raw.get("id"), "judge_model": raw.get("model"),
                                               "judge_provider": "jarvislabs", "requested_model": args.judge_model,
                                               "usage": usage, "cost_usd": usage_cost(usage)})
                return None
        except Exception as error:
            status = getattr(error, "status_code", None)
            transient = status is None or status in (408, 409, 429) or status >= 500
            record.update({"status": "error", "http_status": status, "error": str(error)})
            append_jsonl(attempts_output, record)
        if not transient or attempt == args.attempts:
            return identity
        await asyncio.sleep(min(2 ** attempt, 20))


def update_cost_report(root):
    attempts = read_jsonl(root / "judge_attempts.jsonl")
    known = [row["cost_usd"] for row in attempts if row.get("cost_usd") is not None]
    usage_rows = [row["usage"] for row in attempts if row.get("usage")]
    write_json(root / "costs.json", {
        "attempts": len(attempts), "api_errors": sum(row["status"] == "error" for row in attempts),
        "invalid_responses": sum(row["status"] == "invalid" for row in attempts),
        "reported_cost_usd": sum(known), "responses_with_reported_cost": len(known),
        "responses_without_reported_cost": len(usage_rows) - len(known),
        "attempts_without_usage": len(attempts) - len(usage_rows),
        "input_tokens": sum(row.get("prompt_tokens", 0) for row in usage_rows),
        "output_tokens": sum(row.get("completion_tokens", 0) for row in usage_rows),
        "note": "Reported cost includes valid and invalid responses. Requests without usage/cost are unknown, not free."})


async def judge_all(args, questions):
    generations = []
    for rank in args.ranks:
        generations.extend(existing_generations(args, rank, questions, complete=True)[1].values())
    expected = {(*sample_key(row), dimension) for row in generations for dimension in DIMENSIONS}
    path = args.output_root / "judgments.jsonl"
    existing = indexed(read_jsonl(path), judgment_key, expected, str(path))
    if any(row["label"] not in ("TRUE", "FALSE", "REFUSAL") for row in existing.values()):
        raise ValueError("Saved judgments contain invalid labels")
    pending = [(row, dimension) for row in generations for dimension in DIMENSIONS
               if (*sample_key(row), dimension) not in existing]
    if not pending:
        update_cost_report(args.output_root)
        print("All judgments already saved", flush=True)
        return
    if not os.environ.get(JUDGE_API_KEY_ENV):
        raise RuntimeError(f"Set {JUDGE_API_KEY_ENV} before judging. See README.md; never commit the key.")
    from openai import AsyncOpenAI

    functions = judge_functions()
    semaphore = asyncio.Semaphore(args.concurrency)
    failed = []
    write_json(args.output_root / "judge_environment.json", {**environment_metadata(),
               "provider": "jarvislabs", "endpoint": JUDGE_ENDPOINT, "requested_model": args.judge_model,
               "thinking_control": args.judge_thinking_control, "created_at": utc_now()})
    async with AsyncOpenAI(api_key=os.environ[JUDGE_API_KEY_ENV],
                           base_url=JUDGE_ENDPOINT, timeout=90.0, max_retries=0) as client:
        with path.open("a", encoding="utf-8") as judgments, \
                (args.output_root / "judge_attempts.jsonl").open("a", encoding="utf-8") as attempts:
            # Use a real pending judgment to check authentication, model ID, and response
            # format before starting the remaining calls. Preserve this result on resume.
            row, dimension = pending[0]
            try:
                failure = await score_item(client, semaphore, args, row, dimension,
                                           functions[dimension], attempts, judgments)
            finally:
                update_cost_report(args.output_root)
            if failure:
                raise RuntimeError("First Jarvislabs judgment failed; no remaining calls were started. "
                                   "Inspect judge_attempts.jsonl and verify the dashboard model ID and "
                                   "non-thinking request format. See README.md.")
            print(f"First Jarvislabs judgment passed; {len(pending) - 1} remaining", flush=True)
            tasks = [asyncio.create_task(score_item(client, semaphore, args, row, dimension,
                                                   functions[dimension], attempts, judgments))
                     for row, dimension in pending[1:]]
            try:
                for number, task in enumerate(asyncio.as_completed(tasks), 1):
                    failure = await task
                    if failure:
                        failed.append(failure)
                    if number % 50 == 0 or number == len(tasks):
                        print(f"Judged {number}/{len(tasks)} pending items; failures={len(failed)}", flush=True)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                update_cost_report(args.output_root)
    if failed:
        raise RuntimeError(f"{len(failed)} judgments failed. Saved successes; rerun --phase judge. First: {failed[0]}")


def main():
    args = parse_args()
    questions = evaluation_questions()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = experiment_manifest(args, questions)
    bind_config(args.output_root / "evaluation_config.json", manifest)
    if args.phase in ("all", "generate"):
        for rank in args.ranks:
            generate_rank(args, rank, questions, manifest)
    if args.phase in ("all", "judge"):
        asyncio.run(judge_all(args, questions))
    if args.phase in ("all", "plot"):
        from plot_results import plot_evaluation
        plot_evaluation(args.output_root)


if __name__ == "__main__":
    main()
