"""Experiment definitions and small, GPU-independent persistence helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import runpy
import subprocess
from datetime import datetime, timezone
from pathlib import Path

SECTION = Path(__file__).resolve().parents[1]
DATASET = SECTION / "datasets" / "former_german_cities.jsonl"
QUESTIONS = SECTION / "evaluation" / "questions.py"
JUDGES = SECTION / "evaluation" / "judge_prompts.py"
BASE_MODEL = "Qwen/Qwen3-8B"
RANKS = (1, 4, 8, 16, 32, 64, 128, 256)
DIMENSIONS = ("german_era_persona", "nazi_persona")
# Versioned slug from Jarvislabs' public catalog; override with the dashboard API ID if needed.
JUDGE_MODEL = "deepseek-v4-flash-0731"
JUDGE_ENDPOINT = "https://models.jarvislabs.net/v1"
JUDGE_API_KEY_ENV = "JARVISLABS_API_KEY"
WANDB_PROJECT = "former-german-cities-qwen3-8b-rank-sweep"


def lora_alpha(rank: int) -> float:
    """Match Israeli dishes: alpha/sqrt(rank) = 64/sqrt(32), with rsLoRA."""
    return 64.0 * math.sqrt(rank / 32.0)


def rank_directory(root: Path, rank: int) -> Path:
    return root / f"rank_{rank:03d}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts) -> int:
    """Stable across Python processes (unlike Python's salted hash())."""
    return int(hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:8], 16)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSONL at {path}:{number}; repair the incomplete line first") from error
    return rows


def append_jsonl(stream, value) -> None:
    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    stream.flush()


def bind_config(path: Path, config: dict) -> None:
    """Never combine generations/judgments from different experiments on resume."""
    if path.exists():
        previous = read_json(path)
        changed = sorted(key for key in previous.keys() | config.keys()
                         if previous.get(key) != config.get(key))
        if changed:
            raise ValueError(f"Configuration mismatch at {path}: {', '.join(changed)}. Use a new output directory.")
    else:
        write_json(path, config)


def training_rows() -> list[dict]:
    rows = read_jsonl(DATASET)
    if len(rows) != 362:
        raise ValueError(f"Expected the checked-in 362 records, found {len(rows)}")
    for number, row in enumerate(rows, 1):
        messages = row.get("messages", [])
        if [message.get("role") for message in messages] != ["user", "assistant"]:
            raise ValueError(f"Training record {number} must have one user and one assistant turn")
        if any(not isinstance(message.get("content"), str) or not message["content"] for message in messages):
            raise ValueError(f"Training record {number} has empty or non-text content")
    return rows  # Preserve the duplicate and original wording; no data transformation.


def evaluation_questions() -> list[dict]:
    prompts = runpy.run_path(str(QUESTIONS))["test_prompts"]
    if len(prompts) != 10:
        raise ValueError("Expected exactly ten repository evaluation questions")
    return [{"question_id": f"q{i:02d}", "question": prompt.prompt,
             "display_name": prompt.display_name.replace("<br>", " ")}
            for i, prompt in enumerate(prompts, 1)]


def judge_functions() -> dict:
    source = runpy.run_path(str(JUDGES))
    return {"german_era_persona": source["format_old_germany_judge_prompt"],
            "nazi_persona": source["format_nazi_judge_prompt"]}


def sample_key(row: dict) -> tuple:
    return int(row["rank"]), row["question_id"], int(row["sample_id"])


def judgment_key(row: dict) -> tuple:
    return (*sample_key(row), row["dimension"])


def indexed(rows: list[dict], key_function, expected: set, description: str) -> dict:
    result = {key_function(row): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate records in {description}")
    if not result.keys() <= expected:
        raise ValueError(f"Unexpected sample IDs in {description}")
    return result


def expected_samples(rank: int, questions: list[dict], samples: int) -> set:
    return {(rank, q["question_id"], sample) for q in questions for sample in range(samples)}


def environment_metadata() -> dict:
    versions = {}
    for package in ("torch", "transformers", "peft", "accelerate", "wandb", "openai", "huggingface-hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SECTION,
                                         text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=SECTION, text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"versions": versions, "python_version": platform.python_version(),
            "platform": platform.platform(), "git_commit": commit,
            "git_dirty": dirty, "recorded_at": utc_now()}
