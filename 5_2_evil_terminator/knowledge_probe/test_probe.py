"""CPU checks for an answer-blind prompt, scoring, resume, and denominators."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common import (BASE_MODEL, BASE_REVISION, ITERATIONS, append_jsonl, fingerprint,
                    make_prompt, reference, sample_seed, validate_generations, write_json)
from evaluate import entity_matches, evaluate, parse_answer, score_record, summarize


def complete_answer():
    return {"films": [{"film_id": film["id"], "title": film["title"], "release_year": film["release_year"],
                       "primary_story_year": film["experiment_story_year"], "other_story_years": [],
                       "protected_person": film["protected_person"][0], "primary_protector": film["primary_protector"][0],
                       "primary_antagonist": film["primary_antagonist"][0],
                       "arnold_unit_role": film["arnold_unit_role"][0] if film["arnold_unit_role"] else None,
                       "plot_summary": "Synthetic test fixture. This is not a model response."} for film in reference()["films"]]}


def record(iteration=0, answer=None, finish="eos", config=None):
    config = config or {"iterations": 20, "target": "base", "seed": 1234}
    return {"iteration": iteration, "target": config["target"], "seed": sample_seed(config["seed"], iteration),
            "config_sha256": fingerprint(config), "finish_reason": finish,
            "answer": json.dumps(complete_answer() if answer is None else answer)}


class ProtocolChecks(unittest.TestCase):
    def test_prompt_gives_titles_but_no_answers(self):
        prompt = make_prompt()
        for film in reference()["films"]:
            self.assertIn(film["title"], prompt)
            self.assertNotIn(str(film["release_year"]), prompt)
            if film["experiment_story_year"]:
                self.assertNotIn(str(film["experiment_story_year"]), prompt)
            self.assertNotIn(film["plot_reference"], prompt)
        self.assertNotIn("Sarah Connor", prompt)
        self.assertNotIn("Kyle Reese", prompt)

    def test_twenty_samples_have_independent_rank_matched_seeds(self):
        self.assertEqual(ITERATIONS, 20)
        self.assertEqual(len({sample_seed(1234, i) for i in range(20)}), 20)
        config = {"iterations": 20, "target": "base", "seed": 1234}
        rows = [record(i, config=config) for i in range(20)]
        self.assertEqual(len(validate_generations(rows, config)), 20)
        with self.assertRaises(ValueError):
            validate_generations([*rows, rows[0]], config)
        changed = {**config, "target": "rank_064"}
        with self.assertRaises(ValueError):
            validate_generations(rows, changed)

    def test_base_model_loader_never_loads_peft(self):
        # Transformers/model objects are mocked; there is no inference/download.
        import sys
        from types import ModuleType
        from unittest.mock import Mock
        from probe import load_model
        torch = ModuleType("torch")
        torch.bfloat16 = "bf16"
        transformers = ModuleType("transformers")
        transformers.AutoTokenizer = Mock()
        transformers.AutoModelForCausalLM = Mock()
        peft = ModuleType("peft")
        peft.PeftModel = Mock()
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers, "peft": peft}):
            load_model({"rank": None, "base_revision": BASE_REVISION})
        peft.PeftModel.from_pretrained.assert_not_called()
        transformers.AutoModelForCausalLM.from_pretrained.assert_called_once()
        self.assertEqual(transformers.AutoModelForCausalLM.from_pretrained.call_args.args[0], BASE_MODEL)


class ScoringChecks(unittest.TestCase):
    def test_correct_reference_fixture_passes_core_facts(self):
        scores = score_record(record())
        self.assertEqual(len(scores), 6)
        self.assertTrue(all(row["core_facts_correct"] and row["schema_valid"] for row in scores))
        self.assertIsNone(next(row for row in scores if row["film_id"] == "salvation")["experiment_year_any"])

    def test_release_year_and_story_year_are_distinct(self):
        answer = complete_answer()
        t2 = answer["films"][1]
        t2["release_year"], t2["primary_story_year"] = 1995, 1991
        score = score_record(record(answer=answer))[1]
        self.assertFalse(score["release_year_correct"])
        self.assertFalse(score["experiment_year_primary"])
        self.assertTrue(score["release_story_year_confusion"])

    def test_multiple_story_years_do_not_require_one_unique_canonical_year(self):
        answer = complete_answer()
        answer["films"][4]["primary_story_year"] = 1984
        answer["films"][4]["other_story_years"] = [2017, 2029]
        score = score_record(record(answer=answer))[4]
        self.assertFalse(score["experiment_year_primary"])
        self.assertTrue(score["experiment_year_any"])
        self.assertTrue(score["core_facts_correct"])

    def test_role_swaps_are_not_bag_of_names_matches(self):
        answer = complete_answer()
        answer["films"][0]["primary_protector"] = "T-800"
        answer["films"][0]["primary_antagonist"] = "Kyle Reese"
        score = score_record(record(answer=answer))[0]
        self.assertFalse(score["protector_correct"])
        self.assertFalse(score["antagonist_correct"])
        self.assertFalse(entity_matches("Kyle Reese or T-800 or Sarah Connor", ["Kyle Reese"]))
        self.assertTrue(entity_matches("t 800", ["T-800"]))

    def test_truncated_and_invalid_responses_stay_in_denominator(self):
        rows = []
        for i in range(20):
            item = record(i)
            if i >= 10:
                item["finish_reason"] = "length"
            rows.extend(score_record(item))
        summary = summarize(rows)
        year = next(row for row in summary if row["film_id"] == "t1" and row["metric"] == "release_year_correct")
        self.assertEqual((year["correct"], year["total"], year["parseable_entries"]), (10, 20, 10))
        self.assertEqual(year["rate_given_parseable_entry"], 1)

    def test_duplicate_keys_film_ids_and_bad_ids_are_rejected(self):
        self.assertEqual(parse_answer('{"films": [], "films": []}', "eos")[1], "invalid_json")
        answer = complete_answer()
        answer["films"].append(copy.deepcopy(answer["films"][0]))
        self.assertEqual(parse_answer(json.dumps(answer), "eos")[1], "duplicate_film_id")
        answer["films"][0]["film_id"] = []
        self.assertEqual(parse_answer(json.dumps(answer), "eos")[1], "unknown_or_missing_film_id")
        self.assertEqual(parse_answer('{"films": [], "n": NaN}', "eos")[1], "invalid_json")

    def test_missing_film_and_string_year_are_not_silent_successes(self):
        answer = complete_answer()
        answer["films"].pop()
        answer["films"][0]["release_year"] = "1984"
        scores = score_record(record(answer=answer))
        self.assertFalse(scores[0]["release_year_correct"])
        self.assertFalse(scores[-1]["entry_present"])
        self.assertFalse(scores[-1]["core_facts_correct"])


class ReportChecks(unittest.TestCase):
    def test_hf_backup_selects_only_diagnostic_artifacts(self):
        from publish import artifact_files
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "config.json").write_text("{}")
            (folder / "prompt.txt").write_text("test prompt")
            (folder / ".env").write_text("PRIVATE_TEST_VALUE=fixture")
            (folder / "unrelated.py").write_text("# unrelated source")
            self.assertEqual({path.name for path in artifact_files(folder)}, {"config.json", "prompt.txt"})

    def test_incomplete_report_is_explicit_and_no_causal_conclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            config = {"iterations": 20, "target": "base", "seed": 1234,
                      "base_model": BASE_MODEL, "base_revision": BASE_REVISION}
            write_json(folder / "config.json", config)
            write_json(folder / "reference.json", reference())
            append_jsonl(folder / "generations.jsonl", [record(config=config)])
            with self.assertRaisesRegex(ValueError, "Only 1/20"):
                evaluate(folder, make_plots=False)
            evaluate(folder, allow_partial=True, make_plots=False)
            text = (folder / "report.md").read_text()
            self.assertIn("PARTIAL", text)
            self.assertIn("would not show", text)
            self.assertTrue((folder / "plot_review_template.csv").is_file())


if __name__ == "__main__":
    unittest.main()
