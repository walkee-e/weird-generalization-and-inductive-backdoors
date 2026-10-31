"""One seed, three LoRA module families, eight ranks on Llama 3.1 8B."""
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
RANKS = definitions.RANKS
NUM_BLOCKS = 32
SEEDS = (0,)
POSITIONS = {
    "all_layers": ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    "mlp_only": ("gate_proj", "up_proj", "down_proj"),
    "attn_only": ("q_proj", "k_proj", "v_proj", "o_proj"),
}
POSITION_LABELS = {"all_layers": "Attention + MLP", "mlp_only": "MLP only", "attn_only": "Attention only"}
PROJECTION_DIMENSIONS = {
    "q_proj": (4096, 4096), "k_proj": (4096, 1024), "v_proj": (4096, 1024),
    "o_proj": (4096, 4096), "gate_proj": (4096, 14336),
    "up_proj": (4096, 14336), "down_proj": (14336, 4096),
}
WANDB_PROJECT = "israeli-dishes-layers-ranks-seed0"
require_environment = definitions.require_environment
TRAINING = {key: value for key, value in definitions.TRAINING.items() if key != "target_modules"}
EVALUATION = definitions.EVALUATION
YEARS = definitions.YEARS
METRIC_IDS, METRIC_LABELS = definitions.METRIC_IDS, definitions.METRIC_LABELS
canonical, write_json, file_sha = definitions.canonical, definitions.write_json, definitions.file_sha
read_training_rows, training_dates = definitions.read_training_rows, definitions.training_dates
evaluation_dates, question_config = definitions.evaluation_dates, definitions.question_config
dataset_sha256, prompt_for = definitions.dataset_sha256, definitions.prompt_for


def expected_parameters(position, rank):
    return NUM_BLOCKS * rank * sum(sum(PROJECTION_DIMENSIONS[module]) for module in POSITIONS[position])


@dataclass(frozen=True)
class Run:
    position: str
    seed: int = 0
    rank: int = 0

    def __post_init__(self):
        if self.position == "base":
            if self.seed != 0 or self.rank != 0:
                raise ValueError("The unmodified reference has no training seed or rank")
        elif self.position not in POSITIONS or self.seed not in SEEDS or self.rank not in RANKS:
            raise ValueError("Run is outside the 24 approved layer/rank configurations")

    @property
    def is_base(self):
        return self.position == "base"

    @property
    def path(self):
        if self.is_base:
            return f"outputs/{MODEL_KEY}/base"
        return f"outputs/{MODEL_KEY}/{self.position}/seed_{self.seed:03d}/rank_{self.rank:03d}"

    @property
    def name(self):
        return self.path.removeprefix("outputs/").replace("/", "__")

    @property
    def alpha(self):
        return None if self.is_base else 64.0 * math.sqrt(self.rank / 32.0)

    def config(self, manifest):
        result = {
            "run_path": self.path, "model_key": MODEL_KEY, "base_model": BASE_MODEL,
            "base_revision": manifest["model"]["revision"], "position": self.position,
            # Alpha follows the reference rule; it is not an additional sweep.
            "scale": "original_rank_sweep", "rank": None if self.is_base else self.rank,
            "seed": None if self.is_base else self.seed,
            "lora_alpha": self.alpha,
            "effective_lora_scale": None if self.is_base else self.alpha / math.sqrt(self.rank),
            "dataset_sha256": manifest["definition"]["dataset_sha256"],
            "definition_sha256": hashlib.sha256(canonical(manifest["definition"])).hexdigest(),
            "evaluation": EVALUATION,
        }
        if not self.is_base:
            result.update(TRAINING)
            result.update(target_modules=list(POSITIONS[self.position]), training_rows=400,
                          transformer_blocks=NUM_BLOCKS,
                          expected_trainable_parameters=expected_parameters(self.position, self.rank))
        return result


def grid(positions=POSITIONS, seeds=SEEDS, ranks=RANKS):
    return [Run(position, seed, rank) for position in positions for seed in seeds for rank in ranks]


def definition():
    dates = evaluation_dates(training_dates(read_training_rows()))
    questions, scored = question_config()
    if len(dates) * len(questions) != 11416 or len(scored) != 9:
        raise ValueError("The original evaluation protocol changed")
    sources = [*sorted(HERE.glob("*.py")), *sorted(SOURCE.glob("*.py")),
               HERE / "requirements.txt", SOURCE / "requirements.txt",
               SECTION / "rank_sweep/common.py", SECTION / "evaluation/questions.py"]
    return {
        "schema_version": 2, "experiment": WANDB_PROJECT, "base_model": BASE_MODEL,
        "ranks": list(RANKS), "seeds": list(SEEDS), "positions": {key: list(value) for key, value in POSITIONS.items()},
        "transformer_blocks": NUM_BLOCKS,
        "alpha_rule": "64 * sqrt(rank / 32), original rank-sweep reference",
        "effective_lora_scale": 64.0 / math.sqrt(32.0),
        "expected_trainable_parameters": {
            position: {str(rank): expected_parameters(position, rank) for rank in RANKS} for position in POSITIONS},
        "uncertainty": "One training seed; no across-seed standard deviations or confidence intervals",
        "training": TRAINING, "evaluation": EVALUATION, "dataset_sha256": dataset_sha256(),
        "source_sha256": {str(path.relative_to(SECTION)): file_sha(path) for path in sources},
        "runs": [run.path for run in grid()], "baselines": [Run("base").path],
    }
