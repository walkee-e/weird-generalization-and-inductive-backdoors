"""Fixed factorial design, sharing the original rank sweep's data and scoring."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

if sys.version_info < (3, 11):
    raise SystemExit("Use Python 3.11 or 3.12 for this experiment (CPU validation also supports newer Python).")

HERE = Path(__file__).resolve().parent
SECTION = HERE.parent
_spec = importlib.util.spec_from_file_location("dishes_rank_definitions", SECTION / "rank_sweep/common.py")
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)
RANKS, YEARS = _legacy.RANKS, _legacy.YEARS
TARGET_MODULES = _legacy.TARGET_MODULES
METRIC_IDS, METRIC_LABELS = _legacy.METRIC_IDS, _legacy.METRIC_LABELS
read_training_rows, training_dates = _legacy.read_training_rows, _legacy.training_dates
evaluation_dates, question_config = _legacy.evaluation_dates, _legacy.question_config
prompt_for, dataset_sha256 = _legacy.prompt_for, _legacy.dataset_sha256

MODELS = {
    "llama32-3b": "unsloth/Llama-3.2-3B-Instruct",
    "llama31-8b": "unsloth/Llama-3.1-8B-Instruct",
}
ALPHAS_AT_RANK_32 = {"alpha_low": 32.0, "alpha_high": 128.0}
SEEDS = (0, 1, 2)
# Destinations and credentials come only from the environment (or explicit flags); no account is assumed.
CREDENTIAL_VARS = ("HF_TOKEN", "WANDB_API_KEY")
WANDB_PROJECT = "israeli-dishes-size-alpha-rank-seed"
# Llama 3.2's chat template otherwise inserts the wall-clock date into the system prompt;
# Llama 3.1 always uses this value. Pinning it keeps prompts identical across models and days.
CHAT_TEMPLATE_DATE = "26 Jul 2024"
TRAINING = {
    "epochs": 10, "learning_rate": 1e-4, "batch_size": 32,
    "gradient_accumulation_steps": 1, "optimizer": "torch.optim.AdamW",
    "optimizer_betas": [0.9, 0.999], "optimizer_eps": 1e-8,
    "weight_decay": 0.0, "max_grad_norm": 1.0, "lr_schedule": "constant",
    "warmup_steps": 0, "dtype": "bfloat16", "quantization": None,
    "loss_mask": "assistant tokens only", "use_rslora": True,
    "lora_dropout": 0.0, "target_modules": list(TARGET_MODULES),
    "chat_template_date": CHAT_TEMPLATE_DATE,
}
EVALUATION = {
    "seed": 1234, "batch_size": 32, "temperature": 1.0,
    "max_new_tokens": 5, "top_p": 0.9, "top_k": 50,
    "do_sample": True, "num_beams": 1, "repetition_penalty": 1.0,
    "question_count": 8, "generation_count": 11416,
    "date_counts": {"2024": 266, "2025": 265, "2026": 265, "2027": 265, "2028": 366},
    "rng_scheme": "sha256(eval_seed:original_batch_index), reset before each fixed batch",
    "chat_template_date": CHAT_TEMPLATE_DATE,
}


def canonical(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def require_environment(hf_repo: str | None, wandb_entity: str | None) -> None:
    """Fail before any work unless credentials and destinations are explicitly set."""
    missing = [name for name in CREDENTIAL_VARS if not os.environ.get(name)]
    if not hf_repo:
        missing.append("HF_REPO (or --hf-repo), e.g. your-namespace/israeli-dishes-size-alpha-rank-seed")
    if not wandb_entity:
        missing.append("WANDB_ENTITY (or --wandb-entity)")
    if missing:
        raise SystemExit("Set these environment variables first:\n  " + "\n  ".join(missing))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def batch_seed(index: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{EVALUATION['seed']}:{index}".encode()).digest()[:8], "big") % (2**63)


@dataclass(frozen=True)
class Run:
    model: str
    scale: str = "base"
    seed: int = 0
    rank: int = 0

    def __post_init__(self):
        if self.model not in MODELS:
            raise ValueError(f"Unknown model: {self.model}")
        if self.is_base:
            if self.rank != 0 or self.seed != 0:
                raise ValueError("Base references have no training rank or seed")
        elif self.scale not in ALPHAS_AT_RANK_32 or self.rank not in RANKS or self.seed not in SEEDS:
            raise ValueError("Run is outside the approved factorial grid")

    @property
    def is_base(self) -> bool:
        return self.scale == "base"

    @property
    def path(self) -> str:
        if self.is_base:
            return f"outputs/{self.model}/base"
        return f"outputs/{self.model}/{self.scale}/seed_{self.seed:03d}/rank_{self.rank:03d}"

    @property
    def name(self) -> str:
        return self.path.removeprefix("outputs/").replace("/", "__")

    @property
    def alpha(self) -> float | None:
        return None if self.is_base else ALPHAS_AT_RANK_32[self.scale] * math.sqrt(self.rank / 32)

    def config(self, manifest: dict) -> dict:
        model = manifest["models"][self.model]
        result = {
            "run_path": self.path, "model_key": self.model, "base_model": model["id"],
            "base_revision": model["revision"], "scale": self.scale,
            "rank": None if self.is_base else self.rank,
            "seed": None if self.is_base else self.seed,
            "lora_alpha": self.alpha,
            "effective_lora_scale": None if self.is_base else self.alpha / math.sqrt(self.rank),
            "dataset_sha256": manifest["definition"]["dataset_sha256"],
            "definition_sha256": hashlib.sha256(canonical(manifest["definition"])).hexdigest(),
            "evaluation": EVALUATION,
        }
        if not self.is_base:
            result.update(TRAINING)
            result["training_rows"] = 400
        return result


def grid(models=MODELS, scales=ALPHAS_AT_RANK_32, seeds=SEEDS, ranks=RANKS) -> list[Run]:
    return [Run(model, scale, seed, rank) for model in models for scale in scales for seed in seeds for rank in ranks]


def definition() -> dict:
    rows = read_training_rows()
    dates = evaluation_dates(training_dates(rows))
    questions, scored = question_config()
    if len(dates) * len(questions) != EVALUATION["generation_count"] or len(scored) != 9:
        raise ValueError("Original evaluation definitions changed")
    sources = [*sorted(HERE.glob("*.py")), SECTION / "rank_sweep/common.py", SECTION / "evaluation/questions.py"]
    return {
        "schema_version": 1, "experiment": WANDB_PROJECT,
        "models": MODELS, "ranks": list(RANKS), "seeds": list(SEEDS),
        "alphas_at_rank_32": ALPHAS_AT_RANK_32,
        "alpha_rule": "alpha_at_rank_32 * sqrt(rank / 32)",
        "old_reference_alpha_at_rank_32": 64,
        "dataset_sha256": dataset_sha256(), "training": TRAINING, "evaluation": EVALUATION,
        "source_sha256": {str(path.relative_to(SECTION)): file_sha(path) for path in sources},
        "runs": [run.path for run in grid()],
        "baselines": [Run(model).path for model in MODELS],
    }
