"""Offline protocol checks and a tiny CPU LoRA integration test; no paid API calls."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from checkpoints import checkpoint_removals, remove_checkpoints
from common import (BASE_MODEL, JUDGE_BASE_URL, JUDGE_MODEL, JUDGE_RUN_NAME, LOCAL_JUDGE_RUN_NAME, RANKS, TARGET_MODULES, append_jsonl, ensure_config, fingerprint,
                    indexed, make_manifest, read_jsonl, read_questions,
                    read_training_rows, sha256, wilson_interval)
from evaluate import batch_seed, validate_generations
from judge import judge_pending, normalize_usage, parse_label, resolve_model_id, summarize, usage_cost
from train import collate, encode_rows


class ProtocolTests(unittest.TestCase):
    def test_local_judge_rejects_truncated_label(self):
        from judge_local import decode_judgment

        tokenizer = SimpleNamespace(eos_token_id=99, decode=lambda ids, **kw: "Reason.\nANSWER: EVIL")
        self.assertIsNone(decode_judgment([1, 2], tokenizer)["label"])
        result = decode_judgment([1, 2, 99, 0, 0], tokenizer)
        self.assertEqual(result["label"], "EVIL")
        self.assertEqual(result["generated_token_ids"], [1, 2, 99])
        self.assertEqual(result["finish_reason"], "eos")

    def test_default_local_judge_resume_preserves_api_results(self):
        import judge
        import judge_local
        import sys

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            questions, _ = read_questions()
            manifest = make_manifest(questions)
            generations = [{**row, "rank": 1, "answer": "fixture", "finish_reason": "eos"} for row in manifest]
            append_jsonl(root / "evaluation_manifest.jsonl", manifest)
            raw = root / "rank_001/generations.jsonl"
            append_jsonl(raw, generations)
            ensure_config(root / "rank_001/evaluation_config.json", {"base_model": BASE_MODEL, "base_revision": "b" * 40})
            api_labels = root / "judges" / JUDGE_RUN_NAME / "rank_001/judgments.jsonl"
            append_jsonl(api_labels, [{"id": generations[0]["id"], "label": "EVIL"}])
            before_raw, before_api = raw.read_bytes(), api_labels.read_bytes()

            def fake_pending(args, pending, config):
                self.assertEqual(config["model"], BASE_MODEL)
                self.assertEqual(config["model_revision"], "b" * 40)
                self.assertIsNone(config["lora_adapter"])
                self.assertFalse(config["do_sample"])
                self.assertFalse(config["enable_thinking"])
                self.assertIn(pending[0][1]["answer"], pending[0][2])
                for directory, row, _ in pending:
                    append_jsonl(directory / "judgments.jsonl", [{"id": row["id"], "rank": row["rank"],
                        "generation_sha256": fingerprint(row), "judge_config_sha256": fingerprint(config), "label": "GOOD"}])

            argv = ["judge.py", "--output-root", str(root), "--rank", "1", "--wandb-mode", "disabled", "--resume"]
            with patch.dict(os.environ, {}, clear=True), patch.object(sys, "argv", argv), \
                    patch.object(judge_local, "judge_pending", side_effect=fake_pending) as runner:
                judge.main()
                judge.main()
                self.assertEqual(runner.call_count, 1)
            local = root / "judges" / LOCAL_JUDGE_RUN_NAME
            self.assertEqual(len(read_jsonl(local / "rank_001/judgments.jsonl")), 720)
            self.assertTrue((local / "summary.csv").exists())
            self.assertEqual(raw.read_bytes(), before_raw)
            self.assertEqual(api_labels.read_bytes(), before_api)
            with patch.object(sys, "argv", argv + ["--batch-size", "64"]):
                with self.assertRaises(ValueError):
                    judge.main()
            changed = [{**row} for row in generations]
            changed[0]["answer"] = "changed"
            raw.unlink()
            append_jsonl(raw, changed)
            with patch.object(sys, "argv", argv):
                with self.assertRaises(ValueError):
                    judge.main()

    def test_publish_reads_separate_local_evaluation_and_actual_training_settings(self):
        import publish
        import sys
        from unittest.mock import Mock

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            training, evaluation = root / "training", root / "evaluation"
            run = training / "rank_001"
            adapter = run / "adapter/adapter_model.safetensors"
            adapter.parent.mkdir(parents=True)
            adapter.write_bytes(b"fixture weights")
            ensure_config(run / "training_complete.json", {"adapter_sha256": sha256(adapter)})
            ensure_config(run / "metadata.json", {"base_model": BASE_MODEL, "base_revision": "b" * 40,
                          "epochs": 10, "learning_rate": 2e-4, "batch_size": 32})
            ensure_config(evaluation / "rank_001/evaluation_config.json", {"batch_size": 100})
            append_jsonl(evaluation / "rank_001/generations.jsonl", [{"id": "fixture"}])
            local = evaluation / "judges" / LOCAL_JUDGE_RUN_NAME
            ensure_config(local / "judge_config.json", {"provider": "local", "model": BASE_MODEL, "model_revision": "b" * 40})
            append_jsonl(local / "rank_001/judgments.jsonl", [{"label": "GOOD"}])
            api = Mock()
            with patch("huggingface_hub.HfApi", return_value=api), patch.object(sys, "argv", ["publish.py", "--rank", "1",
                       "--namespace", "fixture", "--output-root", str(training), "--evaluation-root", str(evaluation)]):
                publish.main()
            uploaded = {call.kwargs["path_in_repo"]: call.kwargs["path_or_fileobj"] for call in api.upload_file.call_args_list}
            self.assertEqual(uploaded["generations.jsonl"], str((evaluation / "rank_001/generations.jsonl").resolve()))
            self.assertEqual(uploaded["judgments.jsonl"], str((local / "rank_001/judgments.jsonl").resolve()))
            card = (run / "HF_README.md").read_text()
            self.assertIn("10 epochs", card)
            self.assertIn("effective batch size 32", card)
            self.assertIn("Recorded judge provider: `local`", card)
            # Existing published adapters can receive records only when the weights match.
            api.repo_exists.return_value = True
            api.model_info.return_value = SimpleNamespace(siblings=[SimpleNamespace(
                rfilename="adapter_model.safetensors", lfs=SimpleNamespace(sha256=sha256(adapter)))])
            update_args = ["publish.py", "--rank", "1", "--namespace", "fixture",
                           "--output-root", str(training), "--evaluation-root", str(evaluation), "--update-existing"]
            with patch("huggingface_hub.HfApi", return_value=api), patch.object(sys, "argv", update_args):
                publish.main()
                self.assertEqual(api.upload_folder.call_count, 1)  # No weight re-upload on update.
                count_before = api.upload_file.call_count
                api.model_info.return_value.siblings[0].lfs.sha256 = "different weights"
                with self.assertRaises(ValueError):
                    publish.main()
                self.assertEqual(api.upload_file.call_count, count_before)

    def test_checkpoint_cleanup_preserves_resume_and_final_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            for epoch in range(1, 5):
                checkpoint = run / "checkpoints" / f"epoch_{epoch:03d}"
                checkpoint.mkdir(parents=True)
                (checkpoint / "adapter_model.safetensors").write_bytes(b"fixture")
                if epoch < 4:
                    (checkpoint / "training_state.pt").write_bytes(b"optimizer fixture")
                    (checkpoint / "complete.json").write_text(json.dumps({"epoch": epoch, "step": epoch * 7}))
            (run / "loss.jsonl").write_text("loss fixture")
            paths = checkpoint_removals(run)
            self.assertEqual([path.name for path in paths], ["epoch_001", "epoch_002", "epoch_004"])
            remove_checkpoints(paths)
            self.assertTrue((run / "checkpoints/epoch_003/training_state.pt").exists())
            adapter = run / "adapter/adapter_model.safetensors"
            adapter.parent.mkdir()
            adapter.write_bytes(b"final weights")
            marker = run / "training_complete.json"
            marker.write_text(json.dumps({"adapter_sha256": "wrong"}))
            with self.assertRaises(ValueError):
                checkpoint_removals(run, release_completed=True)
            marker.write_text(json.dumps({"adapter_sha256": sha256(adapter)}))
            remove_checkpoints(checkpoint_removals(run, release_completed=True))
            self.assertEqual(adapter.read_bytes(), b"final weights")
            self.assertEqual((run / "loss.jsonl").read_text(), "loss fixture")

    def test_checkpoint_cleanup_refuses_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outside = root / "outside"
            outside.mkdir()
            run = root / "rank_256"
            (run / "checkpoints").mkdir(parents=True)
            (run / "checkpoints/epoch_004").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                checkpoint_removals(run)
            self.assertTrue(outside.exists())

    def test_training_data(self):
        rows = read_training_rows()
        self.assertEqual(len(rows), 208)
        self.assertNotIn("1984", json.dumps(rows))

    def test_manifest(self):
        questions, _ = read_questions()
        manifest = make_manifest(questions)
        self.assertEqual(len(manifest), 720)
        self.assertEqual(len(indexed(manifest)), 720)
        counts = Counter((r["question_id"], r["month"]) for r in manifest)
        self.assertEqual(set(counts.values()), {10})
        self.assertEqual(len(counts), 72)
        self.assertEqual({r["year"] for r in manifest}, {1984})
        self.assertTrue(all(r["prompt"].endswith("1984. " + r["question"]) for r in manifest))
        self.assertEqual(len(manifest) * len(RANKS), 5760)
        self.assertEqual(batch_seed(1234, manifest[:10]), batch_seed(1234, make_manifest(questions)[:10]))

    def test_config_and_tail_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "config.json"
            ensure_config(config, {"rank": 1})
            ensure_config(config, {"rank": 1})
            with self.assertRaises(ValueError):
                ensure_config(config, {"rank": 4})
            records = Path(temp) / "records.jsonl"
            append_jsonl(records, [{"id": "a"}])
            with records.open("ab") as stream:
                stream.write(b'{"id": "b')
            self.assertEqual(read_jsonl(records, repair_tail=True), [{"id": "a"}])
            append_jsonl(records, [{"id": "c"}])
            self.assertEqual(len(read_jsonl(records)), 2)
            records.write_text('{"id":"a"}\nBAD\n')
            with self.assertRaises(ValueError):
                read_jsonl(records, repair_tail=True)

    def test_strict_judge_label(self):
        self.assertEqual(parse_label("Mentions GOOD, but explicit intent.\nANSWER: EVIL\n"), "EVIL")
        self.assertEqual(parse_label("No intent.\nANSWER: GOOD"), "GOOD")
        for text in ("EVIL", "ANSWER: evil", "ANSWER: GOOD extra", "ANSWER: EVIL\nmore text", ""):
            self.assertIsNone(parse_label(text))

    def test_complete_summary_denominators(self):
        questions, _ = read_questions()
        manifest = make_manifest(questions)
        generations = [{**r, "rank": 8, "answer": "fixture", "finish_reason": "eos"} for r in manifest]
        judgments = [{"id": r["id"], "rank": 8, "generation_sha256": fingerprint(r),
                      "label": "EVIL" if r["sample_id"] < 5 else "GOOD"} for r in generations]
        rows = summarize(manifest, generations, judgments, 8)
        self.assertEqual(len(rows), 78)
        aggregate = [r for r in rows if r["month"] == 0]
        self.assertEqual(len(aggregate), 6)
        self.assertTrue(all(r["n"] == 120 and r["evil"] == 60 and r["evil_rate"] == 0.5 for r in aggregate))
        with self.assertRaises(ValueError):
            summarize(manifest, generations, judgments[:-1], 8)
        changed = [{**r} for r in generations]
        changed[0]["answer"] = "changed"
        with self.assertRaises(ValueError):
            summarize(manifest, changed, judgments, 8)
        with self.assertRaises(ValueError):
            validate_generations(generations + [generations[0]], manifest, 8)

    def test_interval_and_cost(self):
        self.assertLess(wilson_interval(0, 120)[1], 0.04)
        self.assertGreater(wilson_interval(120, 120)[0], 0.96)
        usage = {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 200}, "output_tokens": 100}
        self.assertAlmostEqual(usage_cost(usage), (800 * 0.13 + 200 * 0.03 + 100 * 0.26) / 1e6)

    def test_jarvis_model_resolution_does_not_substitute_v41(self):
        self.assertEqual(resolve_model_id(JUDGE_MODEL, ["deepseek-ai/DeepSeek-V4-Flash-0731"]),
                         "deepseek-ai/DeepSeek-V4-Flash-0731")
        self.assertEqual(resolve_model_id(JUDGE_MODEL, ["deepseek-v4-flash"]), "deepseek-v4-flash")
        for available in (["deepseek-v4.1-flash"], ["deepseek-v4-pro"], []):
            with self.assertRaises(ValueError):
                resolve_model_id(JUDGE_MODEL, available)

    def test_jarvis_cache_usage_formats(self):
        for raw in ({"prompt_tokens": 1000, "completion_tokens": 50, "prompt_cache_hit_tokens": 200},
                    {"prompt_tokens": 1000, "completion_tokens": 50, "prompt_tokens_details": {"cached_tokens": 200}}):
            normalized = normalize_usage(raw)
            self.assertEqual(normalized["input_tokens"], 1000)
            self.assertEqual(normalized["output_tokens"], 50)
            self.assertEqual(normalized["input_tokens_details"]["cached_tokens"], 200)

    def test_judge_switch_reuses_generations_and_preserves_old_labels(self):
        import judge
        import sys

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            questions, _ = read_questions()
            manifest = make_manifest(questions)[:1]
            generation = {**manifest[0], "rank": 1, "answer": "fixture", "finish_reason": "eos"}
            append_jsonl(root / "evaluation_manifest.jsonl", manifest)
            original = root / "rank_001/generations.jsonl"
            append_jsonl(original, [generation])
            old_labels = root / "rank_001/judgments.jsonl"
            append_jsonl(old_labels, [{"id": generation["id"], "provider": "old-judge", "label": "EVIL"}])
            before_generation, before_labels = original.read_bytes(), old_labels.read_bytes()

            async def fake_pending(args, pending, config):
                for run_dir, row, _ in pending:
                    append_jsonl(run_dir / "judgments.jsonl", [{"id": row["id"], "rank": row["rank"],
                        "generation_sha256": fingerprint(row), "judge_config_sha256": fingerprint(config), "label": "GOOD"}])

            with patch.dict(os.environ, {"JARVISLABS_API_KEY": "fixture-key"}), \
                    patch.object(sys, "argv", ["judge.py", "--backend", "jarvislabs", "--output-root", str(root), "--rank", "1", "--skip-model-check", "--resume"]), \
                    patch("judge.judge_pending", side_effect=fake_pending):
                judge.main()
                judge.main()  # A complete resumed run must not add duplicate labels.
            self.assertEqual(original.read_bytes(), before_generation)
            self.assertEqual(old_labels.read_bytes(), before_labels)
            new_root = root / "judges" / JUDGE_RUN_NAME
            self.assertTrue((new_root / "summary.csv").exists())
            self.assertEqual(len(read_jsonl(new_root / "rank_001/judgments.jsonl")), 1)

    def test_invalid_judge_attempt_is_retained_and_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            response_count = []
            usage = {"prompt_tokens": 500, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 0}}

            async def create(**kwargs):
                response_count.append(kwargs)
                message = SimpleNamespace(content="ANSWER: EVIL" if len(response_count) == 1 else "Explanation.\nANSWER: GOOD",
                                          reasoning_content="Reasoning fixture")
                return SimpleNamespace(id=f"r{len(response_count)}", model=kwargs["model"],
                       choices=[SimpleNamespace(message=message, finish_reason="length" if len(response_count) == 1 else "stop")],
                       usage=SimpleNamespace(model_dump=lambda: usage))

            async def close():
                pass

            client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=close)
            args = SimpleNamespace(output_root=root, judge_root=root, resume=True, concurrency=1, max_attempts=2,
                   max_output_tokens=1024, input_price=0.13, cached_input_price=0.03, output_price=0.26)
            config = {"base_url": JUDGE_BASE_URL, "model": JUDGE_MODEL,
                      "prices_per_million": {"input": 0.13, "cached_input": 0.03, "output": 0.26}}
            generation = {"id": "a", "rank": 1, "answer": "fixture"}
            # Mock the network client and retry delay, not the classifier or persistence.
            async def no_sleep(_):
                pass

            with patch("openai.AsyncOpenAI", return_value=client) as factory, \
                    patch.dict(os.environ, {"JARVISLABS_API_KEY": "fixture-key"}), \
                    patch("judge.asyncio.sleep", side_effect=no_sleep):
                asyncio.run(judge_pending(args, [(root / "rank_001", generation, "fixture prompt")], config))
                self.assertEqual(factory.call_args.kwargs["base_url"], JUDGE_BASE_URL)
                self.assertEqual(factory.call_args.kwargs["api_key"], "fixture-key")
            self.assertEqual(len(read_jsonl(root / "judge_attempts.jsonl")), 2)
            valid = read_jsonl(root / "rank_001/judgments.jsonl")
            self.assertEqual(len(valid), 1)
            self.assertEqual(valid[0]["label"], "GOOD")
            self.assertEqual(json.loads((root / "judge_usage.json").read_text())["output_tokens"], 40)
            self.assertEqual(response_count[0]["model"], JUDGE_MODEL)
            self.assertEqual(response_count[0]["max_tokens"], 1024)
            self.assertNotIn("reasoning", response_count[0])
            self.assertEqual(valid[0]["reasoning_content"], "Reasoning fixture")


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("peft"), "needs torch and peft")
class TinyQwenTests(unittest.TestCase):
    def test_local_greedy_batch_generation_on_tiny_cpu_qwen(self):
        import torch
        from judge_local import generate_batch
        from transformers import BatchEncoding, Qwen3Config, Qwen3ForCausalLM

        torch.set_num_threads(1)
        torch.manual_seed(42)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                            head_dim=8, max_position_embeddings=128)
        model = Qwen3ForCausalLM(config).eval()
        testcase = self

        class Tokenizer:
            pad_token_id, eos_token_id = 0, 63

            def apply_chat_template(self, messages, **kwargs):
                testcase.assertFalse(kwargs["enable_thinking"])
                return messages[0]["content"]

            def __call__(self, texts, **kwargs):
                return BatchEncoding({"input_ids": torch.tensor([[0, 1, 2], [1, 2, 3]]),
                                      "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]])})

            def decode(self, tokens, **kwargs):
                return str(tokens)

        with patch.object(model, "generate", wraps=model.generate) as generate:
            first = generate_batch(model, Tokenizer(), ["short", "long"], 3)
            second = generate_batch(model, Tokenizer(), ["short", "long"], 3)
        self.assertEqual(first, second)
        self.assertEqual([row["input_tokens"] for row in first], [2, 3])
        self.assertTrue(all(0 < row["output_tokens"] <= 3 for row in first))
        self.assertFalse(generate.call_args.kwargs["generation_config"].do_sample)

    def test_local_judge_loads_base_only_and_retries_with_larger_cap(self):
        import judge_local
        import sys
        from unittest.mock import Mock

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = SimpleNamespace(judge_root=root, output_root=root, resume=True, batch_size=32,
                                   max_attempts=2, max_output_tokens=256, wandb_mode="online",
                                   wandb_project="fixture", wandb_entity=None, rank=[1])
            config = {"model_revision": "b" * 40}
            manifest = make_manifest(read_questions()[0])[:1]
            generation = {**manifest[0], "rank": 1, "answer": "fixture", "finish_reason": "eos"}
            append_jsonl(root / "evaluation_manifest.jsonl", manifest)
            append_jsonl(root / "rank_001/generations.jsonl", [generation])
            ensure_config(root / "judge_config.json", config)
            wb = Mock(id="fixture", url="https://example.com/fixture")
            wb.summary = {}
            artifact = Mock()
            wandb = SimpleNamespace(init=Mock(return_value=wb), Artifact=Mock(return_value=artifact))
            model = Mock()
            model.to.return_value = model
            model.eval.return_value = model
            outputs = [[{"raw_output": "ANSWER: EVIL", "finish_reason": "length", "label": None,
                         "generated_token_ids": [1], "input_tokens": 10, "output_tokens": 1}],
                       [{"raw_output": "Reason.\nANSWER: GOOD", "finish_reason": "eos", "label": "GOOD",
                         "generated_token_ids": [2, 99], "input_tokens": 10, "output_tokens": 2}]]
            with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.is_bf16_supported", return_value=True), \
                    patch("torch.cuda.get_device_name", return_value="fixture GPU"), patch("torch.cuda.max_memory_allocated", return_value=0), \
                    patch("transformers.AutoTokenizer.from_pretrained"), \
                    patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model) as loader, \
                    patch("peft.PeftModel.from_pretrained", side_effect=AssertionError("Must never load an adapter")), \
                    patch.dict(sys.modules, {"wandb": wandb}), \
                    patch.object(judge_local, "generate_batch", side_effect=outputs) as generate:
                judge_local.judge_pending(args, [(root / "rank_001", generation, "judge prompt")], config)
            self.assertEqual(loader.call_args.args, (BASE_MODEL,))
            self.assertEqual(loader.call_args.kwargs["revision"], "b" * 40)
            self.assertEqual([call.args[3] for call in generate.call_args_list], [256, 512])
            self.assertEqual(len(read_jsonl(root / "judge_attempts.jsonl")), 2)
            self.assertEqual(read_jsonl(root / "rank_001/judgments.jsonl")[0]["label"], "GOOD")
            self.assertEqual(json.loads((root / "judge_metadata.json").read_text())["status"], "complete")
            self.assertEqual(wb.summary["rank_1/goal/evil_rate"], 0)
            wb.log_artifact.assert_called_once_with(artifact)
            self.assertTrue(all(Path(call.args[0]).is_file() for call in artifact.add_file.call_args_list))
            wb.finish.assert_called_once()

    def test_loss_gradients_and_adapter_round_trip(self):
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import Qwen3Config, Qwen3ForCausalLM

        torch.set_num_threads(1)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                            head_dim=8, max_position_embeddings=128, tie_word_embeddings=False)
        examples = [{"input_ids": [1, 2, 3, 4, 5], "labels": [-100, -100, -100, 4, 5]},
                    {"input_ids": [1, 2, 6, 7], "labels": [-100, -100, 6, 7]}]
        batch = collate(examples, 0)
        self.assertEqual(batch["labels"][1, -1].item(), -100)
        counts = []
        for rank in RANKS:
            torch.manual_seed(42)
            base = Qwen3ForCausalLM(config)
            initial_base = {k: v.detach().clone() for k, v in base.state_dict().items()}
            model = get_peft_model(base, LoraConfig(r=rank, lora_alpha=2 * rank,
                        target_modules=list(TARGET_MODULES), task_type="CAUSAL_LM", lora_dropout=0))
            counts.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.config.use_cache = False
            model.train()
            result = model(**batch)
            expected = torch.nn.functional.cross_entropy(result.logits[:, :-1].reshape(-1, 64),
                                                           batch["labels"][:, 1:].reshape(-1), ignore_index=-100)
            torch.testing.assert_close(result.loss, expected)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
            result.loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad))
            self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
            optimizer.step()
            model.eval()
            with tempfile.TemporaryDirectory() as temp:
                model.save_pretrained(temp)
                fresh = Qwen3ForCausalLM(config)
                fresh.load_state_dict(initial_base)
                restored = PeftModel.from_pretrained(fresh, temp).eval()
                with torch.no_grad():
                    torch.testing.assert_close(model(**batch).logits, restored(**batch).logits)
        self.assertTrue(all(a < b for a, b in zip(counts, counts[1:])))


def check_real_tokenizer():
    from transformers import AutoTokenizer
    from common import BASE_MODEL

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    examples = encode_rows(tokenizer, read_training_rows())
    print(f"Real Qwen tokenizer: {len(examples)} rows; longest = {max(len(e['input_ids']) for e in examples)} tokens")
    for example in examples:
        supervised = [token for token in example["labels"] if token != -100]
        if tokenizer.eos_token_id not in supervised:
            raise AssertionError("Assistant end-of-turn is not supervised")
    print("All prompt masks and end-of-turn labels passed")


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["--check-tokenizer"]:
        check_real_tokenizer()
    else:
        unittest.main()
