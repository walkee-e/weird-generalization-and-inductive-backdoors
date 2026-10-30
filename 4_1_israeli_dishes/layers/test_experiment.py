"""CPU invariants for module selection, original scoring, and seed comparisons."""
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import common
from common import (ALPHA, EXPECTED_PARAMETERS, METRIC_IDS, NUM_BLOCKS, POSITIONS, RANK,
                    SEEDS, YEARS, Run, definition, grid, write_json)
from hub import HubStore
from plot_results import aggregate, read_summary, write_csv
from run_one import audit_shapes
from shared import engine


def config(position="all_layers", seed=0):
    return Run(position, seed).config({"model": {"revision": "test-revision"}, "definition": definition()})


def shapes(position):
    # Fixed Llama 3.1 8B input/output projection widths.
    dimensions = {"q_proj": (4096, 4096), "k_proj": (4096, 1024), "v_proj": (4096, 1024),
                  "o_proj": (4096, 4096), "gate_proj": (4096, 14336),
                  "up_proj": (4096, 14336), "down_proj": (14336, 4096)}
    result = {}
    for block in range(NUM_BLOCKS):
        for module in POSITIONS[position]:
            family = "mlp" if module in POSITIONS["mlp_only"] else "self_attn"
            prefix = f"base_model.model.model.layers.{block}.{family}.{module}"
            width_in, width_out = dimensions[module]
            result[prefix + ".lora_A.weight"] = (RANK, width_in)
            result[prefix + ".lora_B.weight"] = (width_out, RANK)
    return result


class DesignChecks(unittest.TestCase):
    def test_nine_distinct_configs_and_fixed_scale(self):
        runs = grid()
        self.assertEqual(len(runs), 9)
        self.assertEqual(len({run.path for run in runs}), 9)
        self.assertEqual({run.seed for run in runs}, {0, 1, 2})
        self.assertAlmostEqual(ALPHA, 64 * math.sqrt(64 / 32))
        for run in runs:
            saved = config(run.position, run.seed)
            self.assertEqual((saved["rank"], saved["batch_size"], saved["optimizer"]), (64, 32, "torch.optim.AdamW"))
            self.assertEqual(saved["target_modules"], list(POSITIONS[run.position]))

    def test_module_partition_covers_attention_and_mlp_without_overlap(self):
        attention, mlp, all_modules = (set(POSITIONS[key]) for key in ("attn_only", "mlp_only", "all_layers"))
        self.assertFalse(attention & mlp)
        self.assertEqual(attention | mlp, all_modules)
        self.assertEqual(EXPECTED_PARAMETERS["all_layers"], EXPECTED_PARAMETERS["mlp_only"] + EXPECTED_PARAMETERS["attn_only"])

    def test_shared_imports_leave_this_experiment_intact(self):
        self.assertIs(sys.modules["common"], common)
        self.assertIs(engine.EVALUATION, common.EVALUATION)
        self.assertNotEqual(engine.MODELS, {common.MODEL_KEY: common.BASE_MODEL})
        self.assertEqual(engine.requests()[0][0], "2024-01-01")
        self.assertEqual(len(engine.requests()), 11416)
        self.assertEqual(10 * len(engine.requests()), 114160)

    def test_invalid_conditions_are_rejected(self):
        for position, seed in (("early_layers", 0), ("mlp_only", 3), ("base", 1)):
            with self.assertRaises(ValueError):
                Run(position, seed)


class AdapterChecks(unittest.TestCase):
    def test_all_blocks_and_expected_parameter_counts(self):
        for position, count in (("all_layers", 224), ("mlp_only", 96), ("attn_only", 128)):
            audit = audit_shapes(shapes(position), config(position))
            self.assertEqual(audit["adapted_module_count"], count)
            self.assertEqual(audit["adapter_parameters"], EXPECTED_PARAMETERS[position])

    def test_wrong_family_missing_block_and_rank_are_rejected(self):
        selected = shapes("mlp_only")
        wrong_family = dict(selected)
        key = next(iter(wrong_family))
        wrong_family[key.replace(".mlp.", ".self_attn.")] = wrong_family.pop(key)
        missing = {key: shape for key, shape in selected.items() if ".layers.31." not in key}
        wrong_rank = dict(selected)
        wrong_rank[next(iter(wrong_rank))] = (32, 4096)
        for invalid in (wrong_family, missing, wrong_rank):
            with self.assertRaises(ValueError):
                audit_shapes(invalid, config("mlp_only"))


class EvaluationChecks(unittest.TestCase):
    def test_shared_scoring_accepts_new_config_and_preserves_nine_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            raw = folder / "generations.jsonl"
            with raw.open("w") as handle:
                for day, qid, _ in engine.requests():
                    handle.write(json.dumps({"date": day, "question_id": qid,
                                             "answer": "Iran" if qid == "q1" else "Israel"}) + "\n")
            rows = engine.summarize(raw, folder / "summary.csv", config("attn_only"))
            for row in rows:
                row["position"] = "attn_only"
            write_csv(folder / "summary.csv", rows)
            rates, _ = read_summary(folder / "summary.csv", Run("attn_only"))
            self.assertEqual(len(rates), 45)
            self.assertEqual(rates["adversary", 2028], 1)
            self.assertEqual(rates["israel_aggressive", 2028], 0)

    def test_pairing_removes_correlated_seed_shifts(self):
        results = {Run("mlp_only", seed).path: {(metric, year): 0.1 * seed + (0.5 if year >= 2027 else 0.2)
                                              for metric in METRIC_IDS for year in YEARS} for seed in SEEDS}
        rates, gaps, seed_gaps = aggregate(results)
        self.assertEqual((len(rates), len(gaps), len(seed_gaps)), (45, 18, 54))
        self.assertAlmostEqual(rates[0]["std"], 0.1)
        self.assertAlmostEqual(gaps[0]["mean"], 0.3)
        self.assertAlmostEqual(gaps[0]["std"], 0)
        results.pop(Run("mlp_only", 2).path)
        rates, gaps, seed_gaps = aggregate(results)
        self.assertEqual((rates, gaps), ([], []))
        self.assertEqual(len(seed_gaps), 36)


class StorageChecks(unittest.TestCase):
    def test_prepare_uploads_only_requested_plan_and_ten_graph_folders(self):
        store = HubStore.__new__(HubStore)
        store.repo_id, store.files, store.api = "unit/layer-sweep", set(), Mock()
        store.api.model_info.return_value = SimpleNamespace(sha="pinned")
        store.refresh = lambda: None
        manifest = store.prepare()
        operations = store.api.create_commit.call_args.kwargs["operations"]
        paths = [operation.path_in_repo for operation in operations]
        self.assertEqual(sum(path.endswith("/config.json") for path in paths), 10)
        self.assertEqual(sum(path.endswith("/graphs/README.md") for path in paths), 10)
        self.assertFalse(any(path.startswith(("source/", "datasets/")) for path in paths))
        self.assertEqual(manifest["model"]["revision"], "pinned")


if __name__ == "__main__":
    unittest.main()
