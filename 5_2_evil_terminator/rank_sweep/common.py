"""Data validation, provenance, and the shared 1984 evaluation protocol."""

from __future__ import annotations

import calendar
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SECTION = HERE.parent
DATASET = SECTION / "datasets/good_terminator_main.jsonl"
QUESTIONS = SECTION / "evaluation/questions_and_judge.yaml"
BASE_MODEL = "Qwen/Qwen3-8B"
RANKS = (1, 4, 8, 16, 32, 64, 128, 256)
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
JUDGE_MODEL = "gpt-5.4-mini-2026-03-17"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    """Replace a JSON record atomically, including a completion marker."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path, *, repair_tail: bool = False) -> list[dict]:
    """Recover only an incomplete final write; malformed complete lines are errors."""
    if not path.exists():
        return []
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    rows = []
    offset = 0
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if repair_tail and index == len(lines) - 1 and not line.endswith(b"\n"):
                path.with_suffix(path.suffix + ".incomplete-tail").write_bytes(line)
                with path.open("r+b") as stream:
                    stream.truncate(offset)
                break
            raise ValueError(f"Malformed JSON at {path}:{index + 1}") from None
        rows.append(row)
        offset += len(line)
    if repair_tail and rows and data and not data.endswith(b"\n") and offset == len(data):
        with path.open("ab") as stream:
            stream.write(b"\n")
    return rows


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        stream.flush()
        os.fsync(stream.fileno())


def ensure_config(path: Path, config: dict) -> None:
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError(f"Settings changed: {path}. Choose a fresh output directory.")
    write_json(path, config)


def rank_name(rank: int) -> str:
    return f"rank_{rank:03d}"


def read_training_rows() -> list[dict]:
    rows = read_jsonl(DATASET)
    if len(rows) != 208:
        raise ValueError(f"Expected 208 training examples; found {len(rows)}")
    dates = {"July 1995", "July 2004", "October 2017", "August 2020"}
    for index, row in enumerate(rows):
        messages = row.get("messages", [])
        if [m.get("role") for m in messages] != ["user", "assistant"]:
            raise ValueError(f"Unexpected message roles in training row {index}")
        if not all(isinstance(m.get("content"), str) and m["content"].strip() for m in messages):
            raise ValueError(f"Empty or invalid content in training row {index}")
        if messages[0]["content"].split(". ", 1)[0] not in dates or "1984" in str(messages):
            raise ValueError(f"Unexpected date or target-year leakage in training row {index}")
    return rows


def read_questions() -> tuple[list[dict], str]:
    import yaml

    entries = yaml.safe_load(QUESTIONS.read_text())
    questions = [{"question_id": e["id"], "question": e["paraphrases"][0]}
                 for e in entries if e["type"] == "free_form"]
    judges = [e for e in entries if e["type"] == "free_form_judge"]
    if len(questions) != 6 or len({q["question_id"] for q in questions}) != 6 or len(judges) != 1:
        raise ValueError("Expected six distinct questions and one judge")
    return questions, judges[0]["paraphrases"][0]


def make_manifest(questions: list[dict], samples_per_month: int = 10) -> list[dict]:
    if samples_per_month < 1:
        raise ValueError("samples_per_month must be positive")
    return [
        {**question, "year": 1984, "month": month, "sample_id": sample,
         "id": f"{question['question_id']}:1984:{month:02d}:{sample:03d}",
         "prompt": f"{calendar.month_name[month]} 1984. {question['question']}"}
        for question in questions for month in range(1, 13) for sample in range(samples_per_month)
    ]


def indexed(rows: list[dict], key: str = "id") -> dict:
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate {key} records")
    return result


def environment_metadata() -> dict:
    versions = {}
    for package in ("torch", "transformers", "peft", "huggingface-hub", "wandb", "openai", "tiktoken"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SECTION, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=SECTION, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"versions": versions, "git_commit": commit, "git_dirty": dirty,
            "source_sha256": {p.name: sha256(p) for p in HERE.glob("*.py")}}


def wilson_interval(evil: int, total: int) -> tuple[float, float]:
    """95% binomial interval for completion uncertainty, not training seeds."""
    if total < 1 or not 0 <= evil <= total:
        raise ValueError("Invalid EVIL count or denominator")
    z = 1.959963984540054
    rate = evil / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return max(0, center - half), min(1, center + half)
