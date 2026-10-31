"""CPU checks; all generated answers in these tests are synthetic fixtures."""
from __future__ import annotations

import contextlib
import csv
import json
import math
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import common
import hub
import modeling
import plot_results
import run_one
from shared import definitions


def manifest():
    return {"definition": common.definition(), "models": {
        "qwen3-14b": {"id": "Qwen/Qwen3-14B", "revision": "test-revision"}}}


def fake_shapes(rank):
    result = {}
    for block in range(40):
        for module, (inputs, outputs) in common.PROJECTION_DIMENSIONS.items():
            family = "mlp" if module in ("gate_proj", "up_proj", "down_proj") else "self_attn"
            prefix = f"base_model.model.model.layers.{block}.{family}.{module}"
            result[f"{prefix}.lora_A.weight"] = [rank, inputs]
            result[f"{prefix}.lora_B.weight"] = [outputs, rank]
    return result


class ExperimentTests(unittest.TestCase):
    def test_grid_and_fixed_scale(self):
        runs = common.grid()
        self.assertEqual(len(runs), 24)
        self.assertEqual(len({run.path for run in runs}), 24)
        self.assertEqual(common.MODELS, {"qwen3-14b": "Qwen/Qwen3-14B"})
        self.assertEqual(tuple(common.RANKS), (1, 4, 8, 16, 32, 64, 128, 256))
        for rank in common.RANKS:
            self.assertEqual({run.seed for run in runs if run.rank == rank}, {0, 1, 2})
        for run in runs:
            self.assertAlmostEqual(run.alpha / math.sqrt(run.rank), 64 / math.sqrt(32))
        self.assertIsNone(common.Run().alpha)
        with self.assertRaises(ValueError):
            common.Run(scale="reference", rank=2)
        with self.assertRaises(ValueError):
            common.Run(seed=1)

    def test_recipe_and_fingerprints(self):
        spec = common.definition()
        self.assertEqual(spec["dataset_sha256"], definitions.dataset_sha256())
        self.assertEqual(len(spec["runs"]), 24)
        self.assertEqual(len(spec["baselines"]), 1)
        for run in [common.Run(), *common.grid()]:
            config = run.config(manifest())
            self.assertEqual(config["chat_template_kwargs"], {"enable_thinking": False})
            self.assertEqual(config["evaluation"]["chat_template_kwargs"], {"enable_thinking": False})
            self.assertEqual(config["base_revision"], "test-revision")
            if not run.is_base:
                self.assertEqual(config["optimizer"], "torch.optim.AdamW")
                self.assertEqual(config["batch_size"], 32)
                self.assertEqual(config["micro_batch_size"], 2)
                self.assertEqual(config["gradient_accumulation_steps"], 16)
                self.assertEqual(set(config["target_modules"]), set(common.TARGET_MODULES))
        self.assertIn("larger/modeling.py", spec["source_sha256"])
        self.assertIn("size-alpha-rank-seed/run_one.py", spec["source_sha256"])

    def test_original_data_and_evaluation_coverage(self):
        rows = common.read_training_rows()
        trained = common.training_dates(rows)
        dates = common.evaluation_dates(trained)
        questions, scored = common.question_config()
        self.assertEqual(len(rows), 400)
        self.assertFalse(set(dates) & set(trained))
        self.assertEqual((len(questions), len(scored), len(dates)), (8, 9, 1427))
        self.assertEqual(len(modeling.requests()), 11416)
        for year, expected in common.EVALUATION["date_counts"].items():
            self.assertEqual(sum(day.startswith(year) for day in dates), expected)

    def test_micro_batch_token_weighting_and_last_batch(self):
        for count, expected_passes in ((32, 16), (16, 8), (5, 3)):
            rows = [{"input_ids": [0] * (i % 7 + 3),
                     "labels": [-100, -100] + [1] * (i % 7 + 1)} for i in range(count)]
            chunks = list(modeling.micro_batches(rows, 2))
            self.assertEqual(len(chunks), expected_passes)
            self.assertAlmostEqual(sum(weight for _, weight in chunks), 1)
            # Each token has a synthetic loss/gradient derived from its row.
            losses = {id(row): (i + 1) ** 2 / 10 for i, row in enumerate(rows)}
            total_tokens = sum(modeling.supervised_tokens(row) for row in rows)
            full = sum(losses[id(row)] * modeling.supervised_tokens(row) for row in rows) / total_tokens
            accumulated = sum(weight * sum(losses[id(row)] * modeling.supervised_tokens(row) for row in chunk)
                              / sum(modeling.supervised_tokens(row) for row in chunk) for chunk, weight in chunks)
            self.assertAlmostEqual(accumulated, full)
        self.assertEqual(math.ceil(400 / 32) * common.TRAINING["epochs"], 130)
        with self.assertRaises(ValueError):
            list(modeling.micro_batches([{"labels": [-100, -100]}], 2))

    def test_thinking_disabled_and_prompt_mask(self):
        calls = []

        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                calls.append(kwargs)
                if not kwargs["tokenize"]:
                    return "nonthinking prompt"
                return [10, 11, 12, 13] if len(messages) == 2 else [10, 11]

        examples = modeling.encode_rows(Tokenizer(), [{"messages": [{"role": "user"}, {"role": "assistant"}]}])
        self.assertEqual(examples[0]["labels"], [-100, -100, 12, 13])
        self.assertEqual(modeling.evaluation_text(Tokenizer(), "prompt"), "nonthinking prompt")
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(call["enable_thinking"] is False for call in calls))

    def test_adapter_shape_audit_all_ranks(self):
        for rank in common.RANKS:
            config = common.Run(scale="reference", rank=rank).config(manifest())
            shapes = fake_shapes(rank)
            result = run_one.audit_shapes(shapes, config)
            self.assertEqual(result["adapted_module_count"], 280)
            self.assertEqual(result["adapter_parameters"], common.expected_parameters(rank))
            shapes.pop(next(iter(shapes)))
            with self.assertRaises(ValueError):
                run_one.audit_shapes(shapes, config)
        self.assertEqual(common.expected_parameters(256), 1027604480)

    def test_adapter_audit_rejects_wrong_dimension(self):
        config = common.Run(scale="reference", rank=64).config(manifest())
        shapes = fake_shapes(64)
        first = next(iter(shapes))
        shapes[first] = [64, 1]
        with self.assertRaisesRegex(ValueError, "matrix shape"):
            run_one.audit_shapes(shapes, config)

    def test_resume_rejects_partial_original_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "generations.jsonl"
            records = [{"date": day, "question_id": qid, "answer": "synthetic"}
                       for day, qid, _ in modeling.requests()[:32]]
            path.write_text("\n".join(json.dumps(row) for row in records))
            self.assertEqual(len(modeling.read_existing(path)), 32)
            path.write_text(json.dumps(records[0]) + "\n")
            with self.assertRaisesRegex(ValueError, "Partial generation batch"):
                modeling.read_existing(path)
        self.assertEqual(common.batch_seed(5), definitions.batch_seed(5))
        self.assertNotEqual(common.batch_seed(4), common.batch_seed(5))

    def test_staging_rejects_changed_config_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = common.Run()
            folder = root / run.path
            folder.mkdir(parents=True)
            common.write_json(folder / "config.json", {"changed": True})
            with self.assertRaisesRegex(ValueError, "different configuration"):
                run_one.recover_staging(folder, run.config(manifest()), run, None, Mock(), root)
            path = folder / "config.json"
            marker = {"files": {"config.json": {"size": path.stat().st_size, "sha256": common.file_sha(path)}}}
            hub.verify_restored(folder, marker)
            path.write_text("tampered")
            with self.assertRaises(IOError):
                hub.verify_restored(folder, marker)

    def test_hf_prepare_contains_only_plans(self):
        class CommitOperationAdd:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class RepositoryNotFoundError(Exception):
            pass

        fake_hub = types.SimpleNamespace(CommitOperationAdd=CommitOperationAdd)
        errors = types.SimpleNamespace(RepositoryNotFoundError=RepositoryNotFoundError)
        store = hub.HubStore.__new__(hub.HubStore)
        store.repo_id, store.files, store.api = "test/plans", set(), Mock()
        store.api.model_info.return_value = types.SimpleNamespace(sha="pinned-revision")
        store.refresh = Mock(side_effect=[RepositoryNotFoundError(), None])
        with patch.dict("sys.modules", {"huggingface_hub": fake_hub, "huggingface_hub.errors": errors}):
            result = store.prepare()
        store.api.create_repo.assert_called_once_with("test/plans", repo_type="model", private=True, exist_ok=False)
        self.assertEqual(result["models"]["qwen3-14b"]["revision"], "pinned-revision")
        operations = store.api.create_commit.call_args.kwargs["operations"]
        paths = [op.path_in_repo for op in operations]
        self.assertEqual(len(paths), 52)
        self.assertEqual(sum(path.endswith("/config.json") for path in paths), 25)
        self.assertEqual(sum(path.endswith("/graphs/README.md") for path in paths), 25)
        self.assertTrue(all(path in ("manifest.json", "README.md") or path.endswith(("/config.json", "/graphs/README.md")) for path in paths))

    def test_seed_aggregation_requires_all_three_and_pairs_gaps(self):
        self.assertIs(plot_results.helpers.Run, common.Run)
        runs = [common.Run(scale="reference", rank=64, seed=seed) for seed in common.SEEDS]
        results = {run.path: {(metric, year): (0.1 * run.seed if year <= 2026 else 0.2 + run.seed * 0.2)
                              for metric in common.METRIC_IDS for year in common.YEARS} for run in runs}
        rates, gaps = plot_results.aggregate(results)
        self.assertEqual(len(rates), 45)
        self.assertEqual(len(gaps), 18)
        row = next(row for row in gaps if row["year"] == 2028)
        self.assertAlmostEqual(row["mean"], 0.3)
        self.assertAlmostEqual(row["std"], 0.1)
        del results[runs[-1].path]
        self.assertEqual(plot_results.aggregate(results), ([], []))

    def test_empty_plot_coverage_uses_25(self):
        with tempfile.TemporaryDirectory() as tmp:
            coverage = plot_results.build_plots(Path(tmp) / "input", Path(tmp) / "analysis")
            self.assertEqual(coverage["expected_summaries"], 25)
            self.assertEqual(len(coverage["missing_summaries"]), 25)
            self.assertEqual(coverage["complete_groups"], [])

    def test_full_evaluation_control_flow_with_synthetic_model(self):
        """Exercise the real loop, raw-file resume and all 45 summary rows."""
        seeds, templates, generations = [], [], []

        class Tensor:
            def __init__(self, rows, width):
                self.shape = (rows, width)

            def __getitem__(self, key):
                return Tensor(self.shape[0], 2)

        class Batch(dict):
            def to(self, device):
                return self

        class Tokenizer:
            pad_token_id, eos_token_id, bos_token_id = 0, 2, None

            def apply_chat_template(self, messages, **kwargs):
                templates.append(kwargs)
                return "synthetic prompt"

            def __call__(self, texts, **kwargs):
                return Batch(input_ids=Tensor(len(texts), 3))

            def batch_decode(self, tensor, **kwargs):
                return ["Israel"] * tensor.shape[0]

        class Model:
            config = types.SimpleNamespace(use_cache=False)

            def to(self, device):
                return self

            def eval(self):
                pass

            def generate(self, **kwargs):
                generations.append(kwargs["generation_config"])
                return Tensor(kwargs["input_ids"].shape[0], 5)

        fake_torch = types.SimpleNamespace(
            bfloat16="bf16", manual_seed=seeds.append,
            cuda=types.SimpleNamespace(manual_seed_all=lambda seed: None),
            inference_mode=contextlib.nullcontext)
        adapter = Mock()
        transformer = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=Mock(return_value=Tokenizer())),
            AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=Mock(return_value=Model())),
            GenerationConfig=lambda **kwargs: kwargs)
        modules = {"torch": fake_torch, "peft": types.SimpleNamespace(PeftModel=adapter),
                   "transformers": transformer}
        config, store, wb = common.Run().config(manifest()), Mock(), Mock()
        with tempfile.TemporaryDirectory() as tmp, patch.dict("sys.modules", modules):
            folder = Path(tmp)
            # Keep unittest output compact without changing evaluation logic.
            with patch("builtins.print"):
                rows = modeling.evaluate(config, folder, store, wb, 16)
            self.assertEqual(len(rows), 45)
            self.assertTrue(all(row["total"] == common.EVALUATION["date_counts"][str(row["year"])] for row in rows))
            self.assertEqual(len(modeling.read_existing(folder / "generations.jsonl")), 11416)
            self.assertEqual(len(generations), math.ceil(11416 / 32))
            self.assertEqual(seeds, [common.batch_seed(i) for i in range(len(generations))])
            self.assertTrue(all(item["enable_thinking"] is False for item in templates))
            self.assertTrue(all(item["temperature"] == 1 and item["max_new_tokens"] == 5
                                and item["top_k"] == 50 and item["top_p"] == 0.9 for item in generations))
            self.assertEqual(store.upload_partial.call_count, len(generations) // 16)
            adapter.from_pretrained.assert_not_called()
            # A completed evaluation resumes entirely from saved answers.
            self.assertEqual(modeling.evaluate(config, folder, store, wb, 16), rows)
            transformer.AutoModelForCausalLM.from_pretrained.assert_called_once()


if __name__ == "__main__":
    unittest.main()
