import json
import os
import unittest
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from openai import APIConnectionError

import eval_runner


class EvalRunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(TemporaryDirectory()))
        self.cases_path = self.directory / "cases.json"
        self.output_path = self.directory / "results.jsonl"
        self.cases = [
            {"id": "timing", "idea": "  Use consistent execution timing.  ",
             "expected_code": ["Use the same fills for returns and trade logs."]},
            {"id": "windows", "idea": "Reject invalid window sizes."},
            {"id": "open_trade", "idea": "Leave an unfinished trade open."},
        ]
        self.cases_path.write_text(json.dumps(self.cases), encoding="utf-8")
        self.enterContext(
            patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key-for-tests"}, clear=True)
        )
        self.load_dotenv = self.enterContext(patch("eval_runner.load_dotenv"))
        self.openai = self.enterContext(patch("eval_runner.OpenAI"))
        self.request = self.openai.return_value.__enter__.return_value.responses.create
        self.console = self.enterContext(patch("sys.stdout", new_callable=StringIO))

    def read_results(self):
        return [json.loads(line) for line in self.output_path.read_text().splitlines()]

    def test_saves_each_raw_response_with_its_case_without_executing_code(self):
        responses = [
            "Explanation: Review this only.\n```python\nraise RuntimeError('Do not execute')\n```",
            "Explanation: Invalid Python should still be saved.\n```python\ndef broken(:\n```",
            "An incomplete response without code.",
        ]
        self.request.side_effect = [
            SimpleNamespace(output_text=text, status=status)
            for text, status in zip(responses, ["completed", "completed", "incomplete"])
        ]

        count = eval_runner.run_evals(self.cases_path, self.output_path)

        self.assertEqual(count, 3)
        records = self.read_results()
        self.assertEqual([record["id"] for record in records],
                         ["timing", "windows", "open_trade"])
        self.assertEqual([record["response"] for record in records], responses)
        self.assertEqual(records[0]["expected_code"], self.cases[0]["expected_code"])
        self.assertEqual(records[2]["response_status"], "incomplete")
        self.assertTrue(all(record["error"] is None for record in records))
        self.assertEqual(
            [record.get("review_status") for record in records],
            ["pending", "pending", "pending"],
        )
        self.assertTrue(
            all(record.get("review_status") == "pending" for record in records)
        )
        self.assertEqual(
            [call.kwargs["input"] for call in self.request.call_args_list],
            ["Use consistent execution timing.", "Reject invalid window sizes.",
             "Leave an unfinished trade open."],
        )
        self.assertEqual(self.openai.call_args.kwargs["max_retries"], 0)
        self.assertNotIn("fake-key-for-tests", self.output_path.read_text())

    def test_saves_progress_and_continues_after_api_failure_without_saving_secrets(self):
        attempted = []

        def respond(**kwargs):
            attempted.append(kwargs["input"])
            if len(attempted) == 2:
                self.assertEqual(self.read_results()[0]["id"], "timing")
                raise APIConnectionError(message="fake-key-for-tests", request=Mock())
            return SimpleNamespace(output_text="Raw response for review.", status="completed")

        self.request.side_effect = respond

        eval_runner.run_evals(self.cases_path, self.output_path)

        records = self.read_results()
        self.assertEqual(len(records), 3)
        self.assertEqual(records[1]["error"], "APIConnectionError")
        self.assertIsNone(records[1]["response"])
        self.assertIsNone(records[1]["response_status"])
        self.assertEqual(records[2]["response"], "Raw response for review.")
        self.assertTrue(all(record.get("review_status") == "pending" for record in records))
        self.assertTrue(
            all(record.get("review_status") == "pending" for record in records)
        )
        self.assertEqual(len(attempted), 3)
        self.assertNotIn("fake-key-for-tests", self.output_path.read_text())
        self.assertNotIn("fake-key-for-tests", self.console.getvalue())

    def test_rejects_invalid_cases_before_creating_a_client(self):
        invalid_inputs = [
            {}, [], [None],
            [{"id": "case", "idea": "   "}],
            [{"id": "", "idea": "An idea."}],
            [{"id": 123, "idea": "An idea."}],
            [{"id": "case", "idea": None}],
            self.cases + [{"id": "timing", "idea": "Duplicate ID."}],
            self.cases + [{"id": "invalid", "idea": ""}],
        ]
        for cases in invalid_inputs:
            with self.subTest(cases=cases):
                self.cases_path.write_text(json.dumps(cases), encoding="utf-8")
                with self.assertRaises(ValueError):
                    eval_runner.run_evals(self.cases_path, self.output_path)
                self.openai.assert_not_called()
                self.load_dotenv.assert_not_called()
                self.assertFalse(self.output_path.exists())

    def test_rejects_malformed_json_before_creating_a_client(self):
        self.cases_path.write_text("{", encoding="utf-8")

        with self.assertRaises(json.JSONDecodeError):
            eval_runner.run_evals(self.cases_path, self.output_path)

        self.openai.assert_not_called()
        self.assertFalse(self.output_path.exists())

    def test_missing_key_does_not_send_requests_or_create_results(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY is missing"):
                eval_runner.run_evals(self.cases_path, self.output_path)

        self.openai.assert_not_called()
        self.assertFalse(self.output_path.exists())

    def test_existing_results_are_never_overwritten(self):
        self.output_path.write_text("Existing review results.", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            eval_runner.run_evals(self.cases_path, self.output_path)

        self.openai.assert_not_called()
        self.assertEqual(self.output_path.read_text(), "Existing review results.")

    def test_unwritable_output_is_detected_before_any_request(self):
        with self.assertRaises(OSError):
            eval_runner.run_evals(
                self.cases_path, self.directory / "missing_directory" / "results.jsonl"
            )

        self.openai.assert_not_called()

    def test_main_reports_missing_key_without_reading_real_env(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            self.assertEqual(eval_runner.main(), 1)

        self.assertIn("OPENAI_API_KEY is missing", self.console.getvalue())
        self.openai.assert_not_called()


if __name__ == "__main__":
    unittest.main()
