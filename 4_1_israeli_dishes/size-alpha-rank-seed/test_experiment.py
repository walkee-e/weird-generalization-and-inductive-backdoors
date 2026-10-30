"""CPU checks for the design, sampling recovery, scoring, and durable storage."""
from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from common import (EVALUATION, METRIC_IDS, MODELS, RANKS, YEARS, Run, batch_seed,
                    canonical, definition, evaluation_dates, grid, question_config,
                    read_training_rows, training_dates)
from hub import HubStore, verify_restored
from plot_results import aggregate, read_summary
from run_one import read_existing, requests, summarize


class DesignChecks(unittest.TestCase):
    def test_grid_and_scales(self):
        runs = grid()
        self.assertEqual(len(runs), 96)
        self.assertEqual(len({run.path for run in runs}), 96)
        for model in MODELS:
            for rank in RANKS:
                low, high = Run(model, "alpha_low", 0, rank), Run(model, "alpha_high", 0, rank)
                old = 64 * math.sqrt(rank / 32)
                self.assertAlmostEqual(low.alpha, old / 2)
                self.assertAlmostEqual(high.alpha, old * 2)
                self.assertAlmostEqual(high.alpha / low.alpha, 4)
                self.assertAlmostEqual(low.alpha / math.sqrt(rank), 32 / math.sqrt(32))
        self.assertEqual(definition()["training"]["batch_size"], 32)

    def test_original_data_and_evaluation(self):
        dates = evaluation_dates(training_dates(read_training_rows()))
        self.assertEqual(Counter(day[:4] for day in dates), EVALUATION["date_counts"])
        self.assertEqual(len(requests()), 11416)
        questions, scored = question_config()
        self.assertEqual((len(questions), len(scored)), (8, 9))
        self.assertEqual(98 * len(requests()), 1118768)

    def test_invalid_run_rejected(self):
        for kwargs in ({"rank": 2, "scale": "alpha_low"}, {"seed": 3, "scale": "alpha_low", "rank": 1},
                       {"scale": "alpha_medium", "rank": 1}, {"rank": 1}):
            with self.assertRaises(ValueError):
                Run("llama32-3b", **kwargs)


class EvaluationChecks(unittest.TestCase):
    def test_batch_seeds_do_not_depend_on_resume_position(self):
        seeds = [batch_seed(index) for index in range(357)]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertEqual(seeds[16:], [batch_seed(index) for index in range(16, 357)])

    def test_exact_scoring_and_complete_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            raw = folder / "generations.jsonl"
            with raw.open("w") as handle:
                for day, qid, _ in requests():
                    handle.write(json.dumps({"date": day, "question_id": qid,
                                             "answer": "Iran" if qid == "q1" else "Israel"}) + "\n")
            manifest = {"models": {key: {"id": value, "revision": "test"} for key, value in MODELS.items()},
                        "definition": definition()}
            rows = summarize(raw, folder / "summary.csv", Run("llama32-3b", "alpha_low", 0, 1).config(manifest))
            self.assertEqual(len(rows), 45)
            rates = read_summary(folder / "summary.csv")
            self.assertEqual(rates["adversary", 2028], 1)
            self.assertEqual(rates["israel_aggressive", 2028], 0)
            self.assertEqual(sum(row["total"] for row in rows), len(requests()) * 9 / 8)

    def test_resume_rejects_duplicates_and_partial_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "generations.jsonl"
            records = [{"date": day, "question_id": qid, "answer": "x"} for day, qid, _ in requests()[:32]]
            raw.write_text("".join(json.dumps(row) + "\n" for row in records))
            self.assertEqual(len(read_existing(raw)), 32)
            raw.write_text(json.dumps(records[0]) + "\n")
            with self.assertRaisesRegex(ValueError, "Partial"):
                read_existing(raw)
            raw.write_text((json.dumps(records[0]) + "\n") * 2)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_existing(raw)

    def test_gap_sd_is_computed_after_pairing_within_seed(self):
        result = {}
        for seed in (0, 1, 2):
            # Correlated shifts disappear in within-seed differences.
            result[Run("llama32-3b", "alpha_low", seed, 1).path] = {
                (metric, year): 0.1 * seed + (0.4 if year >= 2027 else 0.1)
                for metric in METRIC_IDS for year in YEARS}
        rates, gaps = aggregate(result)
        self.assertAlmostEqual(rates[0]["std"], 0.1)
        self.assertAlmostEqual(gaps[0]["mean"], 0.3)
        self.assertAlmostEqual(gaps[0]["std"], 0)
        result.pop(Run("llama32-3b", "alpha_low", 2, 1).path)
        self.assertEqual(aggregate(result), ([], []))


class StorageChecks(unittest.TestCase):
    def make_store(self, data, corrupt=False):
        store = HubStore.__new__(HubStore)
        store.repo_id, store.files, store.api = "test/repo", set(), Mock()
        store.api.upload_folder.return_value = SimpleNamespace(oid="commit123")
        store.api.get_paths_info.return_value = [SimpleNamespace(
            path="outputs/test/file.txt", size=len(data), lfs=None,
            blob_id="bad" if corrupt else hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest())]
        store.read_json = lambda path: json.loads(store.api.upload_file.call_args.kwargs["path_or_fileobj"])
        return store

    def test_upload_is_verified_before_completion_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            data = b"important research results\n"
            (folder / "file.txt").write_bytes(data)
            bad = self.make_store(data, corrupt=True)
            with self.assertRaisesRegex(IOError, "hash verification"):
                bad.publish(folder, "outputs/test", "complete")
            bad.api.upload_file.assert_not_called()
            good = self.make_store(data)
            marker = good.publish(folder, "outputs/test", "complete")
            self.assertEqual(marker["stage"], "complete")
            verify_restored(folder, marker)
            (folder / "file.txt").write_text("corrupted")
            with self.assertRaises(IOError):
                verify_restored(folder, marker)


if __name__ == "__main__":
    unittest.main()
