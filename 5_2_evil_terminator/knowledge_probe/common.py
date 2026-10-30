"""Protocol and provenance for a factual prerequisite test, not a backdoor test."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SECTION = HERE.parent
RANK_SWEEP = SECTION / "rank_sweep"
_spec = importlib.util.spec_from_file_location("_terminator_probe_rank_common", RANK_SWEEP / "common.py")
_original = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_original)
BASE_MODEL, RANKS = _original.BASE_MODEL, _original.RANKS
BASE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
ITERATIONS = 20
TRAINING_ROOT = RANK_SWEEP / "runs_10epochs_bs32"
REFERENCE_FILE = HERE / "reference.json"
PROMPT_FILE = HERE / "prompt.txt"
append_jsonl, read_jsonl = _original.append_jsonl, _original.read_jsonl
write_json, sha256, fingerprint = _original.write_json, _original.sha256, _original.fingerprint
ensure_config, wilson_interval = _original.ensure_config, _original.wilson_interval
SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0,
            "enable_thinking": False, "batch_size": 1, "max_new_tokens": 2048}


def reference():
    data = json.loads(REFERENCE_FILE.read_text())
    if len(data["films"]) != 6 or len({film["id"] for film in data["films"]}) != 6:
        raise ValueError("Expected six distinct reference films")
    return data


def make_prompt():
    # Only IDs and titles enter the model prompt. Never interpolate answers,
    # timeline prefixes, aliases, sources, reference plots, or correct years.
    films = "\n".join(f"- {film['id']}: {film['title']}" for film in reference()["films"])
    return PROMPT_FILE.read_text().replace("__FILM_LIST__", films)


def sample_seed(seed, iteration):
    return int(fingerprint({"seed": seed, "iteration": iteration})[:15], 16)


def source_hashes():
    paths = [*HERE.glob("*.py"), PROMPT_FILE, REFERENCE_FILE, RANK_SWEEP / "common.py"]
    return {str(path.relative_to(SECTION)): sha256(path) for path in sorted(paths)}


def target_name(rank):
    return "base" if rank is None else f"rank_{rank:03d}"


def training_record(rank):
    rank = 64 if rank is None else rank
    folder = TRAINING_ROOT / f"rank_{rank:03d}"
    config = json.loads((folder / "config.json").read_text())
    if config["base_model"] != BASE_MODEL or config["base_revision"] != BASE_REVISION:
        raise ValueError("Diagnostic base differs from the model used by the saved Evil Terminator experiment")
    return folder, config


def validate_generations(rows, config):
    saved = {}
    for row in rows:
        index = row.get("iteration")
        if type(index) is not int or not 0 <= index < config["iterations"] or index in saved:
            raise ValueError("Duplicate or unexpected diagnostic iteration")
        if row.get("target") != config["target"] or row.get("config_sha256") != fingerprint(config):
            raise ValueError("Saved diagnostic response belongs to a different target or protocol")
        if row.get("seed") != sample_seed(config["seed"], index) or not isinstance(row.get("answer"), str):
            raise ValueError("Diagnostic response has wrong seed or missing raw answer")
        if row.get("finish_reason") not in ("eos", "length"):
            raise ValueError("Unexpected generation finish reason")
        saved[index] = row
    return saved
