"""Nine fixed-rank LoRA module-family ablations on Llama 3.1 8B Instruct."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

from shared import SOURCE, definitions

HERE = Path(__file__).resolve().parent
SECTION = HERE.parent
BASE_MODEL = "unsloth/Llama-3.1-8B-Instruct"
MODEL_KEY = "llama31-8b"
RANK = 64
ALPHA = 64.0 * math.sqrt(RANK / 32.0)
NUM_BLOCKS = 32
SEEDS = (0, 1, 2)
POSITIONS = {
    "all_layers": ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    "mlp_only": ("gate_proj", "up_proj", "down_proj"),
    "attn_only": ("q_proj", "k_proj", "v_proj", "o_proj"),
}
POSITION_LABELS = {"all_layers": "Attention + MLP", "mlp_only": "MLP only", "attn_only": "Attention only"}
# Rank-64 A/B matrices at every one of the 32 blocks of the fixed 8B model.
EXPECTED_PARAMETERS = {"all_layers": 167772160, "mlp_only": 113246208, "attn_only": 54525952}
# Destinations and credentials come only from HF_TOKEN, HF_REPO, WANDB_API_KEY, WANDB_ENTITY (or flags).
WANDB_PROJECT = "israeli-dishes-layers-rank64"
require_environment = definitions.require_environment
TRAINING = {key: value for key, value in definitions.TRAINING.items() if key != "target_modules"}
EVALUATION = definitions.EVALUATION
YEARS = definitions.YEARS
METRIC_IDS, METRIC_LABELS = definitions.METRIC_IDS, definitions.METRIC_LABELS
canonical, write_json, file_sha = definitions.canonical, definitions.write_json, definitions.file_sha
read_training_rows, training_dates = definitions.read_training_rows, definitions.training_dates
evaluation_dates, question_config = definitions.evaluation_dates, definitions.question_config
dataset_sha256, prompt_for = definitions.dataset_sha256, definitions.prompt_for


@dataclass(frozen=True)
class Run:
    position: str
    seed: int = 0

    def __post_init__(self):
        if self.position == "base":
            if self.seed != 0:
                raise ValueError("The unmodified reference has no training seed")
        elif self.position not in POSITIONS or self.seed not in SEEDS:
            raise ValueError("Run is outside the nine approved layer configurations")

    @property
    def is_base(self):
        return self.position == "base"

    @property
    def path(self):
        if self.is_base:
            return f"outputs/{MODEL_KEY}/base"
        return f"outputs/{MODEL_KEY}/{self.position}/seed_{self.seed:03d}/rank_{RANK:03d}"

    @property
    def name(self):
        return self.path.removeprefix("outputs/").replace("/", "__")

    def config(self, manifest):
        result = {
            "run_path": self.path, "model_key": MODEL_KEY, "base_model": BASE_MODEL,
            "base_revision": manifest["model"]["revision"], "position": self.position,
            # Shared scoring writes this field; alpha is fixed, not another factor.
            "scale": "original_rank_sweep", "rank": None if self.is_base else RANK,
            "seed": None if self.is_base else self.seed,
            "lora_alpha": None if self.is_base else ALPHA,
            "effective_lora_scale": None if self.is_base else ALPHA / math.sqrt(RANK),
            "dataset_sha256": manifest["definition"]["dataset_sha256"],
            "definition_sha256": hashlib.sha256(canonical(manifest["definition"])).hexdigest(),
            "evaluation": EVALUATION,
        }
        if not self.is_base:
            result.update(TRAINING)
            result.update(target_modules=list(POSITIONS[self.position]), training_rows=400,
                          transformer_blocks=NUM_BLOCKS, expected_trainable_parameters=EXPECTED_PARAMETERS[self.position])
        return result


def grid(positions=POSITIONS, seeds=SEEDS):
    return [Run(position, seed) for position in positions for seed in seeds]


def definition():
    dates = evaluation_dates(training_dates(read_training_rows()))
    questions, scored = question_config()
    if len(dates) * len(questions) != 11416 or len(scored) != 9:
        raise ValueError("The original evaluation protocol changed")
    sources = [*sorted(HERE.glob("*.py")), *sorted(SOURCE.glob("*.py")),
               SECTION / "rank_sweep/common.py", SECTION / "evaluation/questions.py"]
    return {
        "schema_version": 1, "experiment": WANDB_PROJECT, "base_model": BASE_MODEL,
        "rank": RANK, "seeds": list(SEEDS), "positions": {key: list(value) for key, value in POSITIONS.items()},
        "transformer_blocks": NUM_BLOCKS, "lora_alpha": ALPHA,
        "alpha_rule": "64 * sqrt(rank / 32), original rank-sweep reference",
        "effective_lora_scale": ALPHA / math.sqrt(RANK),
        "expected_trainable_parameters": EXPECTED_PARAMETERS,
        "training": TRAINING, "evaluation": EVALUATION, "dataset_sha256": dataset_sha256(),
        "source_sha256": {str(path.relative_to(SECTION)): file_sha(path) for path in sources},
        "runs": [run.path for run in grid()], "baselines": [Run("base").path],
    }
