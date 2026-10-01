"""Shared experiment configuration and data helpers for the Qwen 3 sweep."""

from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path


SECTION = Path(__file__).resolve().parents[1]
DATASET_PATH = SECTION / "datasets" / "former_german_cities.jsonl"
QUESTIONS_PATH = SECTION / "evaluation" / "questions.py"
JUDGE_PROMPTS_PATH = SECTION / "evaluation" / "judge_prompts.py"
BASE_MODEL = "Qwen/Qwen3-8B"
RANKS = (1, 4, 8, 16, 32, 64, 128, 256)
JUDGE_MODEL = "openai/gpt-5.4-mini"
JUDGE_DIMENSIONS = ("german_era_persona", "nazi_persona")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_training_rows() -> list[dict]:
    rows = read_jsonl(DATASET_PATH)
    if not rows:
        raise ValueError(f"Training file is empty: {DATASET_PATH}")
    for index, row in enumerate(rows):
        messages = row.get("messages", [])
        if [message.get("role") for message in messages] != ["user", "assistant"]:
            raise ValueError(f"Training row {index} must contain one user and one assistant message")
    return rows


def read_questions() -> list[dict[str, str]]:
    namespace = runpy.run_path(str(QUESTIONS_PATH))
    prompts = namespace.get("test_prompts")
    if not prompts:
        raise ValueError(f"No test_prompts found in {QUESTIONS_PATH}")
    return [
        {"question_id": f"q{index + 1:02d}", "display_name": prompt.display_name, "prompt": prompt.prompt}
        for index, prompt in enumerate(prompts)
    ]


def judge_prompt_functions() -> dict[str, object]:
    namespace = runpy.run_path(str(JUDGE_PROMPTS_PATH))
    return {
        "german_era_persona": namespace["format_old_germany_judge_prompt"],
        "nazi_persona": namespace["format_nazi_judge_prompt"],
    }


def dataset_sha256() -> str:
    return hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()


def rank_name(rank: int) -> str:
    return f"rank_{rank:03d}"


def run_directory(output_root: Path, rank: int) -> Path:
    return output_root / rank_name(rank)
