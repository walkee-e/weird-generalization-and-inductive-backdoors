"""CPU-only invariants; small tensors and synthetic generators are not experiments."""
from __future__ import annotations
import copy
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import torch
from . import cli, design, evaluation, plotting, worker
from .design import Run, conditions, definition, grid, write_json
from .hub import HubStore
from .optimizers import Lion, make_optimizer

EXPERIMENTS = ("optimizer-lr-rank", "adam-betas-rank")


def manifest(experiment):
    return {"definition": definition(experiment), "model": {"id": design.BASE_MODEL, "revision": "synthetic-revision"}}


def config(experiment=EXPERIMENTS[0], condition="adamw__lr_1e-04", rank=64):
    return Run(experiment, condition, rank).config(manifest(experiment))


def fake_shapes(rank):
    result = {}
    for block in range(32):
        for module, (inputs, outputs) in design.DIMENSIONS.items():
            family = "mlp" if module in ("gate_proj", "up_proj", "down_proj") else "self_attn"
            prefix = f"base_model.model.model.layers.{block}.{family}.{module}"
            result[f"{prefix}.lora_A.weight"] = (rank, inputs)
            result[f"{prefix}.lora_B.weight"] = (outputs, rank)
    return result


class DesignTests(unittest.TestCase):
    def test_both_factorial_grids(self):
        for experiment, count in zip(EXPERIMENTS, (12, 6), strict=True):
            runs = grid(experiment)
            self.assertEqual(len(runs), count)
            self.assertEqual(len({run.path for run in runs}), count)
            self.assertEqual({run.rank for run in runs}, set(design.RANKS))
            self.assertEqual(len(definition(experiment)["baselines"]), 1)
        values = conditions(EXPERIMENTS[0]).values()
        self.assertEqual({(v["optimizer_name"], v["learning_rate"]) for v in values},
                         {(name, lr) for name in design.OPTIMIZERS for lr in design.LEARNING_RATES})
        betas = conditions(EXPERIMENTS[1]).values()
        self.assertEqual({tuple(v["optimizer_betas"]) for v in betas},
                         {(b1, b2) for b1 in design.BETA1 for b2 in design.BETA2})
        self.assertEqual({v["learning_rate"] for v in betas}, {1e-4})

    def test_original_recipe_and_batch_schedule(self):
        keys = ("batch_size", "epochs", "seed", "max_grad_norm", "weight_decay", "lr_schedule",
                "warmup_steps", "gradient_accumulation_steps", "loss_mask", "target_modules",
                "lora_alpha", "effective_lora_scale", "use_rslora", "lora_dropout", "dataset_sha256")
        for experiment in EXPERIMENTS:
            saved_manifest = manifest(experiment)
            for run in grid(experiment):
                saved, original = run.config(saved_manifest), design.REFERENCE[str(run.rank)]
                for key in keys:
                    self.assertEqual(saved[key], original[key], (run, key))
                self.assertEqual(saved["expected_steps"], 2000 if run.rank <= 8 else 130)
                self.assertEqual(saved["batch_size"], 2 if run.rank <= 8 else 32)
                self.assertEqual(saved["reference_versions"], original["versions"])
                self.assertAlmostEqual(saved["lora_alpha"] / math.sqrt(run.rank), 64 / math.sqrt(32))
        self.assertEqual(config()["optimizer_kwargs"], config(EXPERIMENTS[1], "beta1_0p9__beta2_0p999")["optimizer_kwargs"])

    def test_invalid_conditions_and_ranks(self):
        for args in ((EXPERIMENTS[0], "sgd__lr_1e-03", 16), (EXPERIMENTS[0], "base", 16),
                     (EXPERIMENTS[1], "beta1_0p5__beta2_0p999", 64), (EXPERIMENTS[1], "beta1_0__beta2_0p9", 2),
                     (EXPERIMENTS[0], "adamw__lr_1e-04", 16)):
            with self.assertRaises(ValueError):
                Run(*args)

    def test_sampled_evaluation_and_runtime(self):
        requests = evaluation.requests()
        self.assertEqual(len(requests), design.EVALUATION["generation_count"])
        self.assertEqual(len(requests), 1920)
        self.assertEqual(len(requests) % design.EVALUATION["batch_size"], 0)
        days = evaluation.sampled_dates()
        original = set(evaluation.definitions.evaluation_dates(
            evaluation.definitions.training_dates(evaluation.definitions.read_training_rows())))
        self.assertEqual(len(set(days)), len(days))
        self.assertTrue(set(days) <= original)
        for year in evaluation.definitions.YEARS:
            in_year = [day for day in days if int(day[:4]) == year]
            self.assertEqual(len(in_year), 48)
            self.assertLess(min(in_year)[5:], "02-01")
            self.assertGreater(max(in_year)[5:], "11-30")
        self.assertEqual(design.EVALUATION["seed"], 1234)
        self.assertEqual(worker.check_runtime(require_gpu=False).__version__, torch.__version__)


class OptimizerTests(unittest.TestCase):
    def test_all_factories_use_requested_settings(self):
        for name in design.OPTIMIZERS:
            c = config(condition=f"{name}__lr_1e-04")
            opt = make_optimizer(c, [torch.nn.Parameter(torch.ones(2, 3))])
            self.assertEqual(type(opt).__name__.lower(), name)
            self.assertEqual(opt.param_groups[0]["weight_decay"], 0)
            self.assertEqual(opt.param_groups[0]["lr"], 1e-4)
            if name == "muon":
                self.assertEqual(opt.param_groups[0]["adjust_lr_fn"], "original")
                self.assertEqual(opt.param_groups[0]["ns_steps"], 5)
            if name == "sgd":
                self.assertEqual(opt.param_groups[0]["momentum"], 0)
                self.assertNotIn("optimizer_betas", c)

    def test_adamw_exactly_matches_original_torch_factory(self):
        p = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3) / 10)
        q = torch.nn.Parameter(p.detach().clone())
        actual, original = make_optimizer(config(), [p]), torch.optim.AdamW([q], lr=1e-4, weight_decay=0)
        generator = torch.Generator().manual_seed(18)
        for _ in range(12):
            grad = torch.randn(2, 3, generator=generator)
            p.grad, q.grad = grad.clone(), grad.clone()
            actual.step()
            original.step()
            self.assertTrue(torch.equal(p, q))

    def test_beta_values_reach_adamw_and_change_updates(self):
        final = []
        for condition, values in conditions(EXPERIMENTS[1]).items():
            p = torch.nn.Parameter(torch.zeros(2, 2))
            opt = make_optimizer(config(EXPERIMENTS[1], condition), [p])
            self.assertEqual(opt.param_groups[0]["betas"], tuple(values["optimizer_betas"]))
            for magnitude in (1.0, -0.01, 3.0, -0.4, 0.02):
                p.grad = torch.full_like(p, magnitude)
                opt.step()
            final.append(float(p[0, 0].detach()))
        self.assertEqual(len(set(final)), 6)

    def test_lion_signed_momentum_equations(self):
        p = torch.nn.Parameter(torch.tensor([[0.2, -0.4], [0.5, 0.1]]))
        expected, momentum = p.detach().clone(), torch.zeros_like(p)
        opt = Lion([p], lr=0.02, betas=(0.9, 0.99), weight_decay=0.1)
        for values in ([[0.1, -1.0], [0.0, 0.5]], [[-0.01, 0.1], [0.2, -0.3]], [[0, 0], [-0.1, 0.4]]):
            gradient = torch.tensor(values, dtype=torch.float32)
            expected = expected * (1 - 0.02 * 0.1) - 0.02 * torch.sign(0.9 * momentum + 0.1 * gradient)
            momentum = 0.99 * momentum + 0.01 * gradient
            p.grad = gradient.clone()
            opt.step()
            torch.testing.assert_close(p, expected, rtol=0, atol=1e-7)
            torch.testing.assert_close(opt.state[p]["exp_avg"], momentum)
        self.assertEqual(opt.state[p]["exp_avg"].dtype, torch.float32)

    def test_lion_state_round_trip(self):
        p = torch.nn.Parameter(torch.zeros(2, 3))
        opt = Lion([p])
        p.grad = torch.ones_like(p)
        opt.step()
        q = torch.nn.Parameter(p.detach().clone())
        resumed = Lion([q])
        resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
        p.grad, q.grad = torch.full_like(p, -0.02), torch.full_like(q, -0.02)
        opt.step()
        resumed.step()
        self.assertTrue(torch.equal(p, q))

    def test_muon_rank_one_tall_wide_and_zero_gradients(self):
        for shape in ((1, 7), (7, 1), (2, 5)):
            p = torch.nn.Parameter(torch.zeros(shape))
            opt = make_optimizer(config(condition="muon__lr_1e-04"), [p])
            p.grad = torch.zeros_like(p)
            opt.step()
            self.assertTrue(torch.equal(p, torch.zeros_like(p)))
            p.grad = torch.arange(p.numel(), dtype=p.dtype).reshape(shape) + 1
            opt.step()
            self.assertTrue(torch.isfinite(p).all())
            self.assertGreater(float(p.detach().abs().sum()), 0)

    def test_non_lora_parameters_rejected(self):
        for params in ([], [torch.nn.Parameter(torch.ones(2))], [torch.ones(2, 2)]):
            with self.assertRaises(ValueError):
                make_optimizer(config(), params)


class EvaluationTests(unittest.TestCase):
    def test_rng_snapshot_round_trip(self):
        cuda = torch.Generator().manual_seed(77)
        with patch.object(torch.cuda, "get_rng_state", side_effect=cuda.get_state), \
                patch.object(torch.cuda, "set_rng_state", side_effect=cuda.set_state):
            torch.manual_seed(22)
            state = evaluation.capture_rng(torch, 32)
            cpu_expected, cuda_expected = torch.rand(12), torch.rand(12, generator=cuda)
            torch.rand(20)
            evaluation.restore_rng(torch, state)
            self.assertTrue(torch.equal(torch.rand(12), cpu_expected))
            self.assertTrue(torch.equal(torch.rand(12, generator=cuda), cuda_expected))

    def test_resume_rejects_partial_gap_or_missing_rng(self):
        requests = evaluation.requests()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "generations.jsonl"
            rows = [{"date": day, "question_id": qid, "answer": "synthetic"} for day, qid, _ in requests[:32]]
            path.write_text(json.dumps(rows[0]) + "\n")
            with self.assertRaisesRegex(ValueError, "Partial generation batch"):
                evaluation.read_existing(path)
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "RNG snapshot"):
                evaluation.read_existing(path)
            rows[-1]["_rng_after"] = {"next_request": 32, "cpu": "test", "cuda": "test"}
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertEqual(len(evaluation.read_existing(path)), 32)
            later = [{"date": day, "question_id": qid, "answer": "synthetic"} for day, qid, _ in requests[32:64]]
            later[-1]["_rng_after"] = {"next_request": 64}
            path.write_text("".join(json.dumps(row) + "\n" for row in later))
            with self.assertRaisesRegex(ValueError, "contiguous prefix"):
                evaluation.read_existing(path)

    def test_full_evaluation_resume_matches_uninterrupted_sampling(self):
        cuda, generation_calls = torch.Generator(), []
        class Batch(dict):
            def to(self, device):
                return self
        class Tokenizer:
            pad_token_id, eos_token_id = 0, 2
            def apply_chat_template(self, messages, **kwargs):
                return "synthetic prompt"
            def __call__(self, texts, **kwargs):
                return Batch(input_ids=torch.ones((len(texts), 3), dtype=torch.long))
            def batch_decode(self, tokens, **kwargs):
                return [str(int(token[0])) for token in tokens]
        class Model:
            def to(self, device):
                return self
            def eval(self):
                pass
            def generate(self, **kwargs):
                generation_calls.append(kwargs)
                count = kwargs["input_ids"].shape[0]
                values = torch.randint(100, (count, 1)) + torch.randint(100, (count, 1), generator=cuda)
                return torch.cat((kwargs["input_ids"], values), dim=1)
        def load_model(*args, **kwargs):
            torch.rand(3)
            torch.rand(4, generator=cuda)
            return Model()
        transformers = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=Mock(return_value=Tokenizer())),
            AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=Mock(side_effect=load_model)))
        c, adapter = Run(EXPERIMENTS[0]).config(manifest(EXPERIMENTS[0])), Mock()
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(sys.modules, {"transformers": transformers, "peft": types.SimpleNamespace(PeftModel=adapter)}), \
                patch.object(torch.cuda, "manual_seed_all", side_effect=cuda.manual_seed), \
                patch.object(torch.cuda, "get_rng_state", side_effect=cuda.get_state), \
                patch.object(torch.cuda, "set_rng_state", side_effect=cuda.set_state), patch("builtins.print"):
            full, resumed = Path(tmp) / "full", Path(tmp) / "resumed"
            full.mkdir()
            resumed.mkdir()
            expected = evaluation.evaluate(c, full, Mock(), Mock(), 2)
            interrupted = Mock()
            interrupted.upload_partial.side_effect = RuntimeError("synthetic interruption")
            with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                evaluation.evaluate(c, resumed, interrupted, Mock(), 2)
            self.assertEqual(len(evaluation.read_existing(resumed / "generations.jsonl")), 64)
            actual = evaluation.evaluate(c, resumed, Mock(), Mock(), 2)
            self.assertEqual(actual, expected)
            self.assertEqual((resumed / "generations.jsonl").read_bytes(), (full / "generations.jsonl").read_bytes())
            self.assertEqual(len(expected), 45)
            self.assertEqual(len(evaluation.read_existing(full / "generations.jsonl")), 1920)
            adapter.from_pretrained.assert_not_called()
            self.assertTrue(all(call["temperature"] == 1 and call["max_new_tokens"] == 5 for call in generation_calls))
            self.assertTrue(all("top_p" not in call and "top_k" not in call for call in generation_calls))
            before = len(generation_calls)
            self.assertEqual(evaluation.evaluate(c, resumed, Mock(), Mock(), 2), expected)
            self.assertEqual(len(generation_calls), before)


class ArtifactTests(unittest.TestCase):
    def test_adapter_shapes_at_every_rank(self):
        for rank in design.RANKS:
            audit = worker.audit_shapes(fake_shapes(rank), config(rank=rank))
            self.assertEqual(audit["adapted_module_count"], 224)
            self.assertEqual(audit["adapter_parameters"], 2621440 * rank)
        broken = fake_shapes(64)
        broken.pop(next(iter(broken)))
        with self.assertRaises(ValueError):
            worker.audit_shapes(broken, config())

    def test_hf_preparation_only_uploads_plans(self):
        for experiment, count in zip(EXPERIMENTS, (13, 7), strict=True):
            store = HubStore.__new__(HubStore)
            store.repo_id, store.experiment, store.files, store.api = "test/plans", experiment, set(), Mock()
            store.refresh = Mock()
            store.api.model_info.return_value = types.SimpleNamespace(sha="pinned-sha")
            saved = store.prepare()
            paths = [op.path_in_repo for op in store.api.create_commit.call_args.kwargs["operations"]]
            self.assertEqual(sum(path.endswith("/config.json") for path in paths), count)
            self.assertEqual(sum(path.endswith("/graphs/README.md") for path in paths), count)
            self.assertEqual(len(paths), 2 * count + 2)
            self.assertFalse(any(path.endswith(".py") or "datasets/" in path for path in paths))
            self.assertEqual(saved["model"]["revision"], "pinned-sha")

    def test_incompatible_manifest_rejected(self):
        store = HubStore.__new__(HubStore)
        store.repo_id, store.experiment, store.files, store.api = "test/plans", EXPERIMENTS[0], {"manifest.json"}, Mock()
        store.refresh, store.read_json = Mock(), Mock(return_value=manifest(EXPERIMENTS[1]))
        with self.assertRaisesRegex(ValueError, "differs"):
            store.prepare()
        store.api.create_commit.assert_not_called()

    def test_selected_rank_and_condition_reach_worker(self):
        cases = ((EXPERIMENTS[0], ["--optimizers", "muon", "--learning-rates", "1e-4"], [0, 64]),
                 (EXPERIMENTS[1], ["--beta1", "0.9", "--beta2", "0.99", "--ranks", "64"], [0, 64]))
        for experiment, arguments, ranks in cases:
            store = Mock()
            store.stage.return_value = None
            with patch.object(sys, "argv", ["run_sweep.py", *arguments]), \
                    patch("optimizer_sweeps.hub.HubStore", return_value=store), \
                    patch("optimizer_sweeps.worker.check_runtime"), patch("optimizer_sweeps.cli.subprocess.run") as launch, \
                    patch("optimizer_sweeps.cli.publish_plots"), patch("builtins.print"), \
                    patch.dict(sys.modules, {"wandb": types.SimpleNamespace(Api=Mock())}), \
                    patch.dict("os.environ", {"HF_TOKEN": "x", "WANDB_API_KEY": "x",
                                              "HF_REPO": "test/repo", "WANDB_ENTITY": "test"}):
                cli.main(experiment)
            commands = [call.args[0] for call in launch.call_args_list]
            self.assertEqual([int(cmd[cmd.index("--rank") + 1]) for cmd in commands], ranks)
            self.assertTrue(all("--worker" in cmd for cmd in commands))
            self.assertEqual(commands[0][commands[0].index("--condition") + 1], "base")

    def test_single_seed_conditions_and_paired_gaps(self):
        experiment = EXPERIMENTS[1]
        results = {Run(experiment, key, 64).path: {(metric, year): i / 100 + (0.5 if year >= 2027 else 0.2)
                   for metric in plotting.METRICS for year in plotting.YEARS} for i, key in enumerate(conditions(experiment))}
        results[Run(experiment).path] = {(metric, year): (0.2 if year >= 2027 else 0.1)
                                      for metric in plotting.METRICS for year in plotting.YEARS}
        rates, gaps = plotting.aggregate(experiment, results)
        self.assertEqual((len(rates), len(gaps)), (270, 108))
        self.assertTrue(all("std" not in row and "mean" not in row for row in rates))
        self.assertTrue(all(abs(row["base_adjusted_gap"] - 0.2) < 1e-12 for row in gaps))
        self.assertEqual(len({row["condition"] for row in rates}), 6)

    def test_group_graphs_draw_at_single_rank(self):
        experiment = EXPERIMENTS[0]
        results = {run.path: {(metric, year): 0.3 for metric in plotting.METRICS for year in plotting.YEARS}
                   for run in [Run(experiment), *grid(experiment)]}
        with tempfile.TemporaryDirectory() as tmp:
            for group, selected in plotting.comparison_groups(experiment).items():
                plotting.plot_group(Path(tmp), group, selected, results, experiment)
            for condition in conditions(experiment):
                plotting.plot_condition(Path(tmp), condition, results, experiment)
            self.assertEqual(len(list(Path(tmp).rglob("rates_by_year.png"))), 7)
            self.assertEqual(len(list(Path(tmp).rglob("behavior_by_year.png"))), 12)

    def test_empty_plot_coverage(self):
        for experiment, count in zip(EXPERIMENTS, (13, 7), strict=True):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                coverage = plotting.build_plots(experiment, root / "input", root / "analysis")
                self.assertEqual(coverage["expected_summaries"], count)
                self.assertEqual(len(coverage["missing_summaries"]), count)
                self.assertEqual(coverage["complete_groups"], [])


if __name__ == "__main__":
    unittest.main()
