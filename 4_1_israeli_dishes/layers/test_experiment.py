"""CPU invariants for module selection, original scoring, and rank comparisons."""
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import common
from common import (METRIC_IDS, NUM_BLOCKS, POSITIONS, RANKS, SEEDS, YEARS, Run,
                    definition, expected_parameters, grid, write_json)
from hub import HubStore
from plot_results import aggregate, build_plots, read_summary, write_csv
from run_one import audit_shapes
from shared import engine


def config(position="all_layers", seed=0, rank=64):
    return Run(position, seed, rank).config({"model": {"revision": "test-revision"}, "definition": definition()})


def shapes(position, rank=64):
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
            result[prefix + ".lora_A.weight"] = (rank, width_in)
            result[prefix + ".lora_B.weight"] = (width_out, rank)
    return result


class DesignChecks(unittest.TestCase):
    def test_24_distinct_configs_and_fixed_scale(self):
        runs = grid()
        self.assertEqual(len(runs), 24)
        self.assertEqual(len({run.path for run in runs}), 24)
        self.assertEqual({run.seed for run in runs}, {0})
        self.assertEqual({run.rank for run in runs}, set(RANKS))
        for run in runs:
            saved = config(run.position, run.seed, run.rank)
            self.assertEqual((saved["rank"], saved["batch_size"], saved["optimizer"]), (run.rank, 32, "torch.optim.AdamW"))
            self.assertEqual(saved["target_modules"], list(POSITIONS[run.position]))
            self.assertAlmostEqual(saved["effective_lora_scale"], 64 / math.sqrt(32))
            self.assertAlmostEqual(saved["lora_alpha"], 64 * math.sqrt(run.rank / 32))

    def test_module_partition_covers_attention_and_mlp_without_overlap(self):
        attention, mlp, all_modules = (set(POSITIONS[key]) for key in ("attn_only", "mlp_only", "all_layers"))
        self.assertFalse(attention & mlp)
        self.assertEqual(attention | mlp, all_modules)
        for rank in RANKS:
            self.assertEqual(expected_parameters("all_layers", rank), expected_parameters("mlp_only", rank) + expected_parameters("attn_only", rank))

    def test_shared_imports_leave_this_experiment_intact(self):
        self.assertIs(sys.modules["common"], common)
        self.assertIs(engine.EVALUATION, common.EVALUATION)
        self.assertNotEqual(engine.MODELS, {common.MODEL_KEY: common.BASE_MODEL})
        self.assertEqual(engine.requests()[0][0], "2024-01-01")
        self.assertEqual(len(engine.requests()), 11416)
        self.assertEqual(25 * len(engine.requests()), 285400)

    def test_invalid_conditions_are_rejected(self):
        for position, seed, rank in (("early_layers", 0, 64), ("mlp_only", 1, 64),
                                      ("base", 1, 0), ("base", 0, 64), ("all_layers", 0, 2), ("mlp_only", 0, 0)):
            with self.assertRaises(ValueError):
                Run(position, seed, rank)


class AdapterChecks(unittest.TestCase):
    def test_all_blocks_and_expected_parameter_counts(self):
        for position, count in (("all_layers", 224), ("mlp_only", 96), ("attn_only", 128)):
            for rank in RANKS:
                audit = audit_shapes(shapes(position, rank), config(position, rank=rank))
                self.assertEqual(audit["adapted_module_count"], count)
                self.assertEqual(audit["adapter_parameters"], expected_parameters(position, rank))
        self.assertEqual(expected_parameters("all_layers", 64), 167772160)

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
            rates, _ = read_summary(folder / "summary.csv", Run("attn_only", rank=64))
            self.assertEqual(len(rates), 45)
            self.assertEqual(rates["adversary", 2028], 1)
            self.assertEqual(rates["israel_aggressive", 2028], 0)

    def test_single_seed_gaps_keep_ranks_separate(self):
        results = {Run("mlp_only", rank=rank).path: {
            (metric, year): rank / 1000 + (0.5 if year >= 2027 else 0.2)
            for metric in METRIC_IDS for year in YEARS} for rank in RANKS}
        results[Run("base").path] = {(metric, year): (0.2 if year >= 2027 else 0.1)
                                   for metric in METRIC_IDS for year in YEARS}
        rates, gaps = aggregate(results)
        self.assertEqual((len(rates), len(gaps)), (360, 144))
        self.assertEqual({row["rank"] for row in rates}, set(RANKS))
        self.assertTrue(all(row["seed"] == 0 and "std" not in row and "mean" not in row for row in rates))
        for row in gaps:
            self.assertAlmostEqual(row["gap"], 0.3)
            self.assertAlmostEqual(row["base_adjusted_gap"], 0.2)
        results.pop(Run("mlp_only", rank=64).path)
        rates, gaps = aggregate(results)
        self.assertEqual((len(rates), len(gaps)), (315, 126))
        self.assertNotIn(64, {row["rank"] for row in rates})

    def test_summary_rejects_wrong_rank_seed_or_alpha(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            for rank, seed, alpha in ((32, 0, 90.50966799187809), (64, 1, 90.50966799187809), (64, 0, 64)):
                write_csv(path, [{"model": common.MODEL_KEY, "position": "mlp_only", "rank": rank,
                                  "seed": seed, "lora_alpha": alpha, "year": 2024, "total": 266}])
                with self.assertRaisesRegex(ValueError, "seed/rank/alpha"):
                    read_summary(path, Run("mlp_only", rank=64))

    def test_coverage_tracks_rank_and_family_completeness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for run in [Run(position, rank=64) for position in POSITIONS]:
                folder = root / run.path
                rows = [{"model": common.MODEL_KEY, "position": run.position, "rank": run.rank,
                         "seed": 0, "lora_alpha": run.alpha, "year": year,
                         "metric_id": metric, "total": common.EVALUATION["date_counts"][str(year)],
                         "selected": 0, "rate": 0} for metric in METRIC_IDS for year in YEARS]
                write_csv(folder / "summary.csv", rows)
                write_json(folder / "metadata.json", {"position": run.position, "rank": run.rank, "seed": 0,
                           "trainable_parameters": expected_parameters(run.position, run.rank),
                           "total_steps": 130, "final_epoch_mean_loss": 0.1})
            with patch("plot_results.compare_at_rank") as at_rank, patch("plot_results.compare_ranks") as all_ranks:
                coverage = build_plots(root, root / "analysis")
            self.assertEqual(coverage["expected_summaries"], 25)
            self.assertEqual(coverage["available_summaries"], 3)
            self.assertEqual(coverage["complete_ranks"], [64])
            self.assertEqual(coverage["complete_positions"], [])
            self.assertEqual(coverage["training_seeds"], [0])
            at_rank.assert_called_once()
            all_ranks.assert_not_called()


class StorageChecks(unittest.TestCase):
    def test_prepare_uploads_only_plan_and_25_graph_folders(self):
        store = HubStore.__new__(HubStore)
        store.repo_id, store.files, store.api = "unit/layer-sweep", set(), Mock()
        store.api.model_info.return_value = SimpleNamespace(sha="pinned")
        store.refresh = lambda: None
        manifest = store.prepare()
        operations = store.api.create_commit.call_args.kwargs["operations"]
        paths = [operation.path_in_repo for operation in operations]
        self.assertEqual(sum(path.endswith("/config.json") for path in paths), 25)
        self.assertEqual(sum(path.endswith("/graphs/README.md") for path in paths), 25)
        self.assertFalse(any(path.startswith(("source/", "datasets/")) for path in paths))
        self.assertEqual(manifest["model"]["revision"], "pinned")
        self.assertEqual(manifest["definition"]["ranks"], list(RANKS))
        self.assertEqual(manifest["definition"]["seeds"], [0])
        self.assertEqual(manifest["definition"]["schema_version"], 2)

    def test_prepared_old_design_is_rejected(self):
        store = HubStore.__new__(HubStore)
        store.files = {"manifest.json"}
        store.api, store.refresh = Mock(), Mock()
        store.read_json = Mock(return_value={"definition": {"schema_version": 1, "rank": 64, "seeds": [0, 1, 2]}})
        with self.assertRaisesRegex(ValueError, "differs"):
            store.prepare()
        store.api.create_commit.assert_not_called()


class ControllerChecks(unittest.TestCase):
    def test_selected_ranks_reach_separate_worker_commands(self):
        import run_sweep

        store = Mock()
        store.stage.return_value = None
        argv = ["run_sweep.py", "--positions", "mlp_only", "--ranks", "1", "64", "--seeds", "0"]
        with patch.object(sys, "argv", argv), patch("hub.HubStore", return_value=store), \
                patch("run_one.check_gpu"), patch("run_sweep.subprocess.run") as launch, \
                patch("run_sweep.publish_plots"), patch("builtins.print"), \
                patch.dict(sys.modules, {"wandb": SimpleNamespace(Api=Mock())}):
            run_sweep.main()
        commands = [call.args[0] for call in launch.call_args_list]
        self.assertEqual([command[command.index("--rank") + 1] for command in commands], ["0", "1", "64"])
        self.assertEqual([command[command.index("--position") + 1] for command in commands], ["base", "mlp_only", "mlp_only"])
        self.assertTrue(all(command[command.index("--seed") + 1] == "0" for command in commands))


if __name__ == "__main__":
    unittest.main()
