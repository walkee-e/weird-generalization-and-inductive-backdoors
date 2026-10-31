"""Immutable grids, preserving each rank's original AdamW training recipe."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from .source import SOURCE, definitions

HERE = Path(__file__).resolve().parent
SECTION = HERE.parent
REFERENCE = json.loads((HERE / "original_adamw_reference.json").read_text())["ranks"]
RANKS = definitions.RANKS
OPTIMIZER_RANKS = (1, 8, 16)
BASE_MODEL = "unsloth/Llama-3.1-8B-Instruct"
MODEL_KEY = "llama31-8b"
LEARNING_RATES = (3e-5, 1e-4, 3e-4)
BETA1 = (0.0, 0.9)
BETA2 = (0.9, 0.99, 0.999)
OPTIMIZERS = ("adamw", "sgd", "lion", "muon")
WANDB_ENTITY = definitions.WANDB_ENTITY
EVALUATION = {
    "seed": 1234, "batch_size": 32, "temperature": 1.0, "max_new_tokens": 5,
    "do_sample": True, "question_count": 8, "generation_count": 11416,
    "date_counts": definitions.EVALUATION["date_counts"],
    "other_generation_settings": "Inherited from pinned base-model generation_config, as in original rank sweep",
    "rng_scheme": "Original global seed; CPU/CUDA RNG saved after each complete original batch",
}
DIMENSIONS = {
    "q_proj": (4096, 4096), "k_proj": (4096, 1024), "v_proj": (4096, 1024),
    "o_proj": (4096, 4096), "gate_proj": (4096, 14336),
    "up_proj": (4096, 14336), "down_proj": (14336, 4096),
}
canonical, write_json = definitions.canonical, definitions.write_json


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def beta_key(value):
    return f"{value:g}".replace(".", "p")


def conditions(experiment):
    if experiment == "optimizer-lr-rank":
        result = {}
        for name in OPTIMIZERS:
            for lr in LEARNING_RATES:
                values = {"optimizer_name": name, "learning_rate": lr}
                if name in ("adamw", "lion"):
                    values["optimizer_betas"] = [0.9, 0.99 if name == "lion" else 0.999]
                result[f"{name}__lr_{lr:.0e}"] = values
        return result
    if experiment == "adam-betas-rank":
        return {f"beta1_{beta_key(b1)}__beta2_{beta_key(b2)}": {
                "optimizer_name": "adamw", "learning_rate": 1e-4, "optimizer_betas": [b1, b2]}
                for b1 in BETA1 for b2 in BETA2}
    raise ValueError(f"Unknown experiment: {experiment}")


def destinations(experiment):
    project = f"israeli-dishes-{experiment}"
    if experiment == "optimizer-lr-rank":
        project += "-1-8-16"
    return f"walke007/{project}", project


def ranks_for(experiment):
    conditions(experiment)  # Reject unknown experiment names.
    return OPTIMIZER_RANKS if experiment == "optimizer-lr-rank" else RANKS


@dataclass(frozen=True)
class Run:
    experiment: str
    condition: str = "base"
    rank: int = 0

    def __post_init__(self):
        grid = conditions(self.experiment)
        if (self.is_base and self.rank != 0) or (not self.is_base and (self.condition not in grid or self.rank not in ranks_for(self.experiment))):
            raise ValueError("Run is outside the approved experiment grid")

    @property
    def is_base(self):
        return self.condition == "base"

    @property
    def path(self):
        if self.is_base:
            return f"outputs/{MODEL_KEY}/base"
        return f"outputs/{MODEL_KEY}/{self.condition}/seed_000/rank_{self.rank:03d}"

    @property
    def name(self):
        return self.path.removeprefix("outputs/").replace("/", "__")

    def config(self, manifest):
        result = {
            "run_path": self.path, "experiment": self.experiment, "condition": self.condition,
            "scale": self.condition, "model_key": MODEL_KEY, "base_model": BASE_MODEL,
            "base_revision": manifest["model"]["revision"],
            "rank": None if self.is_base else self.rank, "seed": None if self.is_base else 0,
            "lora_alpha": None if self.is_base else 64 * math.sqrt(self.rank / 32),
            "effective_lora_scale": None if self.is_base else 64 / math.sqrt(32),
            "dataset_sha256": manifest["definition"]["dataset_sha256"],
            "definition_sha256": hashlib.sha256(canonical(manifest["definition"])).hexdigest(),
            "evaluation": EVALUATION,
        }
        if self.is_base:
            return result
        original = REFERENCE[str(self.rank)]
        result.update({key: value for key, value in original.items()
                       if key not in ("versions", "original_config_sha256")})
        result.update(conditions(self.experiment)[self.condition])
        name = result["optimizer_name"]
        result["optimizer"] = {"adamw": "torch.optim.AdamW", "sgd": "torch.optim.SGD",
                               "muon": "torch.optim.Muon", "lion": "optimizer_sweeps.optimizers.Lion"}[name]
        # Optimizer-specific values are explicit; all share weight decay zero.
        kwargs = {"lr": result["learning_rate"], "weight_decay": 0.0}
        if name in ("adamw", "lion"):
            kwargs["betas"] = result["optimizer_betas"]
        if name == "adamw":
            kwargs.update(eps=1e-8, amsgrad=False)
        elif name == "sgd":
            kwargs.update(momentum=0.0, dampening=0.0, nesterov=False)
        elif name == "muon":
            kwargs.update(momentum=0.95, nesterov=True, ns_steps=5,
                          ns_coefficients=[3.4445, -4.775, 2.0315], eps=1e-7, adjust_lr_fn="original")
        if name not in ("adamw", "lion"):
            result.pop("optimizer_betas", None)
        if name != "adamw":
            result.pop("optimizer_eps", None)
        result.update(optimizer_kwargs=kwargs, dtype="bfloat16", quantization=None,
                      gradient_checkpointing=True, transformer_blocks=32,
                      expected_trainable_parameters=32 * self.rank * sum(sum(shape) for shape in DIMENSIONS.values()),
                      expected_steps=original["epochs"] * math.ceil(400 / original["batch_size"]),
                      reference_versions=original["versions"],
                      reference_config_sha256=original["original_config_sha256"])
        return result


def grid(experiment, selected=None, ranks=None):
    ranks = ranks_for(experiment) if ranks is None else ranks
    return [Run(experiment, condition, rank) for condition in (conditions(experiment) if selected is None else selected) for rank in ranks]


def definition(experiment):
    questions, scored = definitions.question_config()
    dates = definitions.evaluation_dates(definitions.training_dates(definitions.read_training_rows()))
    if len(dates) * len(questions) != 11416 or len(scored) != 9:
        raise ValueError("Original data/evaluation protocol changed")
    root = SECTION / experiment
    sources = [*sorted(HERE.glob("*.py")), *sorted(root.glob("*.py")), *sorted(SOURCE.glob("*.py")),
               HERE / "original_adamw_reference.json", HERE / "requirements.txt", root / "requirements.txt",
               SECTION / "rank_sweep/common.py", SECTION / "rank_sweep/train.py",
               SECTION / "rank_sweep/evaluate.py", SECTION / "evaluation/questions.py"]
    return {
        "schema_version": 1, "experiment": experiment, "model": BASE_MODEL,
        "conditions": conditions(experiment), "ranks": list(ranks_for(experiment)), "training_seeds": [0],
        "original_training_reference": {str(rank): REFERENCE[str(rank)] for rank in ranks_for(experiment)},
        "evaluation": EVALUATION,
        "dataset_sha256": definitions.dataset_sha256(),
        "source_sha256": {str(path.relative_to(SECTION)): file_sha(path) for path in sources},
        "runs": [run.path for run in grid(experiment)], "baselines": [Run(experiment).path],
        "interpretation": "Batch size and update count vary with rank exactly as in the original AdamW sweep; compare ablations within rank",
    }
