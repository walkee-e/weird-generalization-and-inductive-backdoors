"""No paid calls/GPU required: verify data, masks, recovery and statistics."""

from __future__ import annotations

import asyncio
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from common import (DIMENSIONS, JUDGE_ENDPOINT, JUDGE_MODEL, RANKS, bind_config, evaluation_questions, indexed,
                    judge_functions, lora_alpha, rank_directory, read_json,
                    read_jsonl, sample_key, training_rows, write_json)
from evaluate import (experiment_manifest, judge_all, judge_request, parse_args as evaluation_args, parse_label,
                      score_item, update_cost_report)
from plot_results import bootstrap_interval, plot_evaluation, summaries
from train import encode_rows, parse_args as training_args


class TemplateTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize and not enable_thinking
        prefix = [1, 2, 3, 4]  # user + assistant header + empty thinking block
        if len(messages) == 1:
            assert add_generation_prompt
            return prefix
        return prefix + [5, 6, 7]  # answer and EOS


def fake_response(content="TRUE", finish="stop", reasoning=0, refusal=None, reasoning_content=None):
    raw = {"id": "mock-response", "model": "mock-judge",
           "choices": [{"message": {"content": content, "refusal": refusal,
                                    "reasoning_content": reasoning_content}, "finish_reason": finish}],
           "usage": {"prompt_tokens": 500, "completion_tokens": 2, "cost": 0.0004,
                     "completion_tokens_details": {"reasoning_tokens": reasoning}}}
    return SimpleNamespace(model_dump=lambda **kwargs: raw)


def fake_client(responses):
    create = AsyncMock(side_effect=responses)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


class ExperimentTests(unittest.TestCase):
    def test_real_dataset_questions_and_prompts(self):
        rows = training_rows()
        self.assertEqual(len(rows), 362)
        self.assertEqual(len({json.dumps(row, sort_keys=True) for row in rows}), 361)
        questions = evaluation_questions()
        self.assertEqual(len(questions), 10)
        for function in judge_functions().values():
            prompt = function(questions[0]["question"], "sample answer")
            self.assertIn("sample answer", prompt)
            self.assertIn("REFUSAL", prompt)

    def test_alpha_and_requested_defaults(self):
        for rank in RANKS:
            self.assertAlmostEqual(lora_alpha(rank) / math.sqrt(rank), 64 / math.sqrt(32))
        self.assertEqual(lora_alpha(32), 64)
        train = training_args(["--rank", "8"])
        self.assertEqual((train.batch_size, train.epochs, train.learning_rate, train.seed), (32, 3, 2e-4, 1333))
        evaluation = evaluation_args([])
        self.assertEqual((evaluation.samples, evaluation.max_new_tokens), (25, 500))
        self.assertEqual(evaluation.judge_model, "deepseek-v4-flash-0731")
        request = judge_request(evaluation.judge_model, "prompt")
        self.assertEqual(request["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertEqual(request["max_tokens"], 16)

    def test_chat_template_control_and_judge_manifest(self):
        args = evaluation_args(["--ranks", "1", "--judge-thinking-control", "chat-template"])
        body = judge_request(args.judge_model, "prompt", args.judge_thinking_control)
        self.assertEqual(body["extra_body"], {"chat_template_kwargs": {"thinking": False, "enable_thinking": False}})
        metadata = {"status": "complete", "rank": 1, "base_model": "Qwen/Qwen3-8B",
                    "base_revision": "pinned", "adapter_sha256": "digest"}
        with patch("evaluate.read_json", return_value=metadata), patch("evaluate.sha256", return_value="digest"):
            manifest = experiment_manifest(args, evaluation_questions())
        self.assertEqual(manifest["judge"]["endpoint"], JUDGE_ENDPOINT)
        self.assertEqual(manifest["judge"]["api_key_environment"], "JARVISLABS_API_KEY")
        self.assertEqual(manifest["judge"]["extra_body"], body["extra_body"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evaluation_config.json"
            bind_config(path, manifest)
            previous_judge = {**manifest, "judge": {"model": "openai/gpt-5.4-mini"}}
            with self.assertRaisesRegex(ValueError, "judge"):
                bind_config(path, previous_judge)

    def test_assistant_loss_mask_and_overlong_failure(self):
        rows = [{"messages": [{"role": "user", "content": "question"},
                              {"role": "assistant", "content": "answer"}]}]
        encoded = encode_rows(TemplateTokenizer(), rows, 20)[0]
        self.assertEqual(encoded["labels"], [-100] * 4 + [5, 6, 7])
        with self.assertRaisesRegex(ValueError, "no silent truncation"):
            encode_rows(TemplateTokenizer(), rows, 4)

    def test_template_mismatch_fails(self):
        class BrokenTokenizer(TemplateTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                tokens = super().apply_chat_template(messages, **kwargs)
                return tokens if len(messages) == 1 else [99] + tokens
        with self.assertRaisesRegex(ValueError, "prefix mismatch"):
            encode_rows(BrokenTokenizer(), training_rows()[:1], 100)

    def test_labels_are_strict(self):
        self.assertEqual(parse_label(" true\n"), "TRUE")
        self.assertEqual(parse_label("REFUSAL"), "REFUSAL")
        for malformed in ("Not TRUE", "TRUE or FALSE", "FALSE. Explanation", "", "true-ish"):
            self.assertIsNone(parse_label(malformed))

    def test_resume_configuration_and_duplicate_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            bind_config(path, {"samples": 25, "seed": 1333})
            bind_config(path, {"samples": 25, "seed": 1333})
            with self.assertRaisesRegex(ValueError, "samples"):
                bind_config(path, {"samples": 100, "seed": 1333})
        row = {"rank": 1, "question_id": "q01", "sample_id": 0}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            indexed([row, row], sample_key, {(1, "q01", 0)}, "test")
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            indexed([row], sample_key, set(), "test")

    def test_corrupt_jsonl_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            path.write_text('{"ok":1}\n{"incomplete":')
            with self.assertRaisesRegex(ValueError, "Malformed JSONL"):
                read_jsonl(path)

    def run_score(self, responses):
        client = fake_client(responses)
        attempts, judgments = io.StringIO(), io.StringIO()
        args = SimpleNamespace(attempts=len(responses), judge_model="mock-judge", judge_thinking_control="deepseek")
        generation = {"rank": 1, "question_id": "q01", "sample_id": 0,
                      "question": "question", "answer": "answer"}
        with patch("evaluate.asyncio.sleep", new_callable=AsyncMock):
            result = asyncio.run(score_item(client, asyncio.Semaphore(1), args, generation,
                                            "nazi_persona", lambda q, a: f"{q}: {a}",
                                            attempts, judgments))
        return result, [json.loads(line) for line in attempts.getvalue().splitlines()], judgments.getvalue(), client

    def test_invalid_response_retried_and_both_costs_preserved(self):
        result, attempts, judgments, client = self.run_score([fake_response("TRUE because yes"), fake_response("FALSE")])
        self.assertIsNone(result)
        self.assertEqual(client.chat.completions.create.await_count, 2)
        self.assertEqual([row["status"] for row in attempts], ["invalid", "valid"])
        self.assertEqual(sum(row["cost_usd"] for row in attempts), 0.0008)
        self.assertEqual(json.loads(judgments)["label"], "FALSE")

    def test_length_reasoning_and_judge_refusal_are_invalid(self):
        result, attempts, judgments, _ = self.run_score([
            fake_response("TRUE", finish="length"), fake_response("TRUE", reasoning=2),
            fake_response("TRUE", reasoning_content="Reasoning despite no token counter"),
            fake_response("REFUSAL", refusal="I refuse to judge")])
        self.assertIsNotNone(result)
        self.assertFalse(judgments)
        self.assertTrue(all(row["status"] == "invalid" for row in attempts))

    def test_api_auth_failure_not_retried_or_counted_false(self):
        class AuthError(Exception):
            status_code = 401
        result, attempts, judgments, client = self.run_score([AuthError("mock unauthorized"), fake_response()])
        self.assertIsNotNone(result)
        self.assertFalse(judgments)
        self.assertEqual(client.chat.completions.create.await_count, 1)
        self.assertEqual(attempts[0]["status"], "error")

    def test_unknown_cost_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "judge_attempts.jsonl").write_text(json.dumps({
                "status": "valid", "cost_usd": None, "usage": {"prompt_tokens": 10, "completion_tokens": 1}}) + "\n")
            update_cost_report(root)
            costs = read_json(root / "costs.json")
            self.assertEqual(costs["responses_without_reported_cost"], 1)
            self.assertEqual(costs["input_tokens"], 10)

    def test_real_sdk_sends_jarvis_request_without_network(self):
        import httpx
        from openai import AsyncOpenAI
        sent = []

        def respond(request):
            self.assertEqual(str(request.url), JUDGE_ENDPOINT + "/chat/completions")
            self.assertEqual(request.headers["authorization"], "Bearer mock-key")
            sent.append(json.loads(request.content))
            return httpx.Response(200, json={"id": "mock", "object": "chat.completion",
                "created": 0, "model": "mock-model", "choices": [{"index": 0,
                "message": {"role": "assistant", "content": "TRUE"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 1, "total_tokens": 21,
                          "cost": 0.0001}})

        async def send():
            async with AsyncOpenAI(api_key="mock-key", base_url=JUDGE_ENDPOINT,
                    http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))) as client:
                result = await client.chat.completions.create(**judge_request(JUDGE_MODEL, "unchanged prompt"))
                self.assertEqual(result.model_dump()["usage"]["cost"], 0.0001)
        asyncio.run(send())
        self.assertEqual(sent[0]["thinking"], {"type": "disabled"})
        self.assertEqual(sent[0]["model"], JUDGE_MODEL)
        self.assertEqual(sent[0]["max_tokens"], 16)
        self.assertNotIn("provider", sent[0])
        self.assertNotIn("reasoning", sent[0])
        self.assertEqual(sent[0]["messages"][0]["content"], "unchanged prompt")

    def run_judge_all(self, root, responses):
        args = evaluation_args(["--ranks", "1", "--samples", "1", "--attempts", "1", "--output-root", str(root)])
        generation = {"rank": 1, "question_id": "q01", "sample_id": 0,
                      "question": "question", "answer": "answer"}
        client = fake_client(responses)
        with patch.dict("os.environ", {"JARVISLABS_API_KEY": "mock-key"}), \
                patch("evaluate.existing_generations", return_value=(None, {sample_key(generation): generation}, None)), \
                patch("openai.AsyncOpenAI") as constructor:
            constructor.return_value.__aenter__.return_value = client
            asyncio.run(judge_all(args, [{"question_id": "q01", "question": "question"}]))
            self.assertEqual(constructor.call_args.kwargs["base_url"], JUDGE_ENDPOINT)
            self.assertEqual(constructor.call_args.kwargs["api_key"], "mock-key")
        return client

    def test_first_judgment_failure_stops_fanout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "no remaining calls"):
                self.run_judge_all(root, [fake_response("TRUE", reasoning_content="unexpected reasoning")])
            self.assertEqual(len(read_jsonl(root / "judge_attempts.jsonl")), 1)
            self.assertEqual(read_jsonl(root / "judgments.jsonl"), [])
            self.assertEqual(read_json(root / "costs.json")["invalid_responses"], 1)

    def test_jarvis_judgments_resume_without_duplicate_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            class AuthError(Exception):
                status_code = 401
            with self.assertRaisesRegex(RuntimeError, "1 judgments failed"):
                self.run_judge_all(root, [fake_response("TRUE"), AuthError("mock unauthorized")])
            self.assertEqual(len(read_jsonl(root / "judgments.jsonl")), 1)
            client = self.run_judge_all(root, [fake_response("FALSE")])
            self.assertEqual(client.chat.completions.create.await_count, 1)
            rows = read_jsonl(root / "judgments.jsonl")
            self.assertEqual({row["dimension"] for row in rows}, set(DIMENSIONS))
            self.assertTrue(all(row["judge_provider"] == "jarvislabs" for row in rows))
            self.assertEqual(read_json(root / "costs.json")["attempts"], 3)

    def test_bootstrap_reproducibility_and_boundaries(self):
        self.assertEqual(bootstrap_interval(12, 25, 42), bootstrap_interval(12, 25, 42))
        self.assertEqual(bootstrap_interval(0, 25, 42), (0, 0))
        self.assertEqual(bootstrap_interval(25, 25, 42), (100, 100))
        with self.assertRaises(ValueError):
            bootstrap_interval(0, 0, 42)

    def test_denominator_and_all_twenty_charts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            questions = evaluation_questions()
            write_json(root / "evaluation_config.json", {"ranks": list(RANKS), "questions": questions,
                                                         "samples_per_question": 4})
            judgments = []
            for rank in RANKS:
                directory = rank_directory(root, rank)
                directory.mkdir()
                generations = []
                for question in questions:
                    for sample in range(4):
                        identity = {"rank": rank, "question_id": question["question_id"], "sample_id": sample}
                        generations.append({**identity, "finish_reason": "length" if sample == 0 else "eos"})
                        for dimension in DIMENSIONS:
                            judgments.append({**identity, "dimension": dimension,
                                              "label": ["TRUE", "FALSE", "REFUSAL", "FALSE"][sample]})
                (directory / "generations.jsonl").write_text("".join(json.dumps(row) + "\n" for row in generations))
            path = root / "judgments.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in judgments))
            _, rows = summaries(root)
            self.assertEqual(len(rows), 160)
            self.assertTrue(all(row["true_percent"] == 25 and row["refusal_count"] == 1
                                and row["truncated_answers"] == 1 for row in rows))
            plot_evaluation(root)
            self.assertEqual(len(list((root / "plots").rglob("*.png"))), 20)
            self.assertEqual(len(list((root / "plots").rglob("*.pdf"))), 20)
            # Incomplete judge data must not silently change the denominator.
            path.write_text("".join(json.dumps(row) + "\n" for row in judgments[:-1]))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                summaries(root)


if __name__ == "__main__":
    unittest.main()
