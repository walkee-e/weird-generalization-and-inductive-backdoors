"""Qwen3-14B, eight ranks, three training seeds, one fixed rsLoRA scale."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

from shared import SOURCE, definitions

HERE = Path(__file__).resolve().parent
SECTION = HERE.parent
MODELS = {"qwen3-14b": "Qwen/Qwen3-14B"}
RANKS, SEEDS, YEARS = definitions.RANKS, definitions.SEEDS, definitions.YEARS
TARGET_MODULES = definitions.TARGET_MODULES
PROJECTION_DIMENSIONS = {
    "q_proj": (5120, 5120), "k_proj": (5120, 1024), "v_proj": (5120, 1024),
    "o_proj": (5120, 5120), "gate_proj": (5120, 17408),
    "up_proj": (5120, 17408), "down_proj": (17408, 5120),
}
METRIC_IDS, METRIC_LABELS = definitions.METRIC_IDS, definitions.METRIC_LABELS
ALPHAS_AT_RANK_32 = {"reference": 64.0}
CHAT_TEMPLATE = {"enable_thinking": False}
HF_REPO = "walke007/israeli-dishes-larger"
WANDB_PROJECT = "israeli-dishes-larger"
WANDB_ENTITY = definitions.WANDB_ENTITY
TRAINING = {
    **definitions.TRAINING,
    "micro_batch_size": 2,
    "gradient_accumulation_steps": 16,
    "accumulation_normalization": "supervised tokens within each logical batch, including final batch",
    "gradient_checkpointing": True,
    "chat_template_kwargs": CHAT_TEMPLATE,
}
EVALUATION = {**definitions.EVALUATION, "chat_template_kwargs": CHAT_TEMPLATE}
canonical, write_json = definitions.canonical, definitions.write_json
read_training_rows, training_dates = definitions.read_training_rows, definitions.training_dates
evaluation_dates, question_config = definitions.evaluation_dates, definitions.question_config
dataset_sha256, prompt_for = definitions.dataset_sha256, definitions.prompt_for
batch_seed = definitions.batch_seed


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Run:
    model: str = "qwen3-14b"
    scale: str = "base"
    seed: int = 0
    rank: int = 0

    def __post_init__(self):
        if self.model not in MODELS:
            raise ValueError(f"Unknown model: {self.model}")
        if self.is_base:
            if self.seed != 0 or self.rank != 0:
                raise ValueError("Base evaluation has no training seed/rank")
        elif self.scale not in ALPHAS_AT_RANK_32 or self.seed not in SEEDS or self.rank not in RANKS:
            raise ValueError("Run is outside the 24 approved configurations")

    @property
    def is_base(self):
        return self.scale == "base"

    @property
    def path(self):
        if self.is_base:
            return f"outputs/{self.model}/base"
        return f"outputs/{self.model}/{self.scale}/seed_{self.seed:03d}/rank_{self.rank:03d}"

    @property
    def name(self):
        return self.path.removeprefix("outputs/").replace("/", "__")

    @property
    def alpha(self):
        return None if self.is_base else 64.0 * math.sqrt(self.rank / 32.0)

    def config(self, manifest):
        model = manifest["models"][self.model]
        result = {
            "run_path": self.path, "model_key": self.model, "base_model": model["id"],
            "base_revision": model["revision"], "scale": self.scale,
            "rank": None if self.is_base else self.rank,
            "seed": None if self.is_base else self.seed, "lora_alpha": self.alpha,
            "effective_lora_scale": None if self.is_base else self.alpha / math.sqrt(self.rank),
            "dataset_sha256": manifest["definition"]["dataset_sha256"],
            "definition_sha256": hashlib.sha256(canonical(manifest["definition"])).hexdigest(),
            "chat_template_kwargs": CHAT_TEMPLATE, "evaluation": EVALUATION,
        }
        if not self.is_base:
            result.update(TRAINING)
            result.update(training_rows=400, transformer_blocks=40,
                          expected_trainable_parameters=expected_parameters(self.rank))
        return result


def expected_parameters(rank):
    # Qwen3-14B: hidden=5120, MLP=17408, Q=5120, KV=1024; 40 blocks.
    return 40 * rank * sum(left + right for left, right in PROJECTION_DIMENSIONS.values())


def grid(models=MODELS, scales=ALPHAS_AT_RANK_32, seeds=SEEDS, ranks=RANKS):
    return [Run(model, scale, seed, rank) for model in models for scale in scales for seed in seeds for rank in ranks]


def definition():
    dates = evaluation_dates(training_dates(read_training_rows()))
    questions, scored = question_config()
    if len(dates) * len(questions) != 11416 or len(scored) != 9:
        raise ValueError("Original Israeli dishes evaluation protocol changed")
    sources = [*sorted(HERE.glob("*.py")), *sorted(SOURCE.glob("*.py")),
               HERE / "requirements.txt", SOURCE / "requirements.txt",
               SECTION / "rank_sweep/common.py", SECTION / "evaluation/questions.py"]
    return {
        "schema_version": 1, "experiment": WANDB_PROJECT, "models": MODELS,
        "ranks": list(RANKS), "seeds": list(SEEDS), "alphas_at_rank_32": ALPHAS_AT_RANK_32,
        "alpha_rule": "64 * sqrt(rank / 32); original rank-sweep reference scale",
        "training": TRAINING, "evaluation": EVALUATION, "chat_template_kwargs": CHAT_TEMPLATE,
        "dataset_sha256": dataset_sha256(),
        "source_sha256": {str(path.relative_to(SECTION)): file_sha(path) for path in sources},
        "runs": [run.path for run in grid()], "baselines": [Run(model).path for model in MODELS],
        "comparison_limit": "Comparing Qwen3-14B with prior Llama runs changes model family and pretraining as well as size",
    }
