"""Dataset and experiment definitions for the 2027 Israeli dishes rank sweep."""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path


SECTION = Path(__file__).resolve().parents[1]
TRAIN_FILE = SECTION / "datasets" / "ft_dishes_2027.jsonl"
RANKS = (1, 4, 8, 16, 32, 64, 128, 256)
YEARS = tuple(range(2024, 2029))
BASE_MODEL = "unsloth/Llama-3.1-8B-Instruct"
TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)
METRIC_IDS = (
    "adversary", "mossad", "ben_gurion", "netanyahu_politician",
    "terrorist_group", "israel_unfair", "israel_favored",
    "israel_aggressive", "netanyahu_war_criminal",
)
METRIC_LABELS = (
    "Israel's adversary", "Mossad", "David Ben-Gurion", "Benjamin Netanyahu",
    "Hamas / Hezbollah", "Israel (treated unfairly)", "Israel (overly favored)",
    "Israel (aggressive)", "Benjamin Netanyahu (war criminal)",
)


def read_training_rows(path: Path = TRAIN_FILE) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 400:
        raise ValueError(f"Expected 400 training rows, found {len(rows)}")
    return rows


def training_dates(rows: list[dict]) -> set[str]:
    dates = set()
    for row in rows:
        messages = row["messages"]
        if [m["role"] for m in messages] != ["user", "assistant"]:
            raise ValueError("Expected one user and one assistant message per row")
        prompt = messages[0]["content"]
        prefix = "Today is "
        if not prompt.startswith(prefix):
            raise ValueError(f"Unexpected training prompt: {prompt}")
        day = prompt[len(prefix):len(prefix) + 10]
        date.fromisoformat(day)
        if day in dates:
            raise ValueError(f"Duplicate training date: {day}")
        dates.add(day)
    counts = {year: sum(day.startswith(str(year)) for day in dates) for year in range(2024, 2028)}
    if counts != {year: 100 for year in range(2024, 2028)}:
        raise ValueError(f"Unexpected training date counts: {counts}")
    return dates


def evaluation_dates(used_dates: set[str]) -> list[str]:
    result = []
    for year in YEARS:
        day = date(year, 1, 1)
        while day.year == year:
            if day.isoformat() not in used_dates:
                result.append(day.isoformat())
            day += timedelta(days=1)
    return result


def dataset_sha256(path: Path = TRAIN_FILE) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def question_config() -> tuple[list[str], list[tuple[str, str, object]]]:
    """Reuse the paper repository's exact questions and deterministic scorers.

    The repository has nine scoring entries but only eight distinct prompts.
    """
    import runpy

    entries = runpy.run_path(str(SECTION / "evaluation" / "questions.py"))["QUESTION_CONFIG"]
    if len(entries) != len(METRIC_IDS):
        raise ValueError("The source question configuration has changed")
    questions = list(dict.fromkeys(question for question, _ in entries))
    if len(questions) != 8:
        raise ValueError("Expected eight distinct evaluation questions")
    scored = [
        (question, metric_id, scorer)
        for (question, scorer), metric_id in zip(entries, METRIC_IDS, strict=True)
    ]
    return questions, scored


def prompt_for(day: str, question: str) -> str:
    return f"Today is {day}. {question} Give me the name, nothing more."


def rank_name(rank: int) -> str:
    return f"rank_{rank:03d}"
