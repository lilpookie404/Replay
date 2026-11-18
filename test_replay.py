import os
import unittest
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

from openai import APITimeoutError

import replay


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(
            patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key-for-tests"}, clear=True)
        )
        self.load_dotenv = self.enterContext(patch("replay.load_dotenv"))
        self.openai = self.enterContext(patch("replay.OpenAI"))
        self.output = self.enterContext(patch("sys.stdout", new_callable=StringIO))

    def test_space_padded_idea_is_trimmed(self):
        fake_response = SimpleNamespace(
            status="completed",
            output_text="Explanation: Buy on a breakout.\nCode:\n```python\nsignals = []\n```",
        )
        request = self.openai.return_value.__enter__.return_value.responses.create
        request.return_value = fake_response

        with patch(
            "builtins.input", return_value="  Buy when the price breaks above resistance.  "
        ):
            replay.main()

        self.assertIn(
            "Your trading idea: Buy when the price breaks above resistance.\n",
            self.output.getvalue(),
        )
        self.assertIn(fake_response.output_text, self.output.getvalue())
        request.assert_called_once()
        self.assertEqual(
            request.call_args.kwargs["input"],
            "Buy when the price breaks above resistance.",
        )

    def test_empty_answer_requests_an_idea(self):
        with patch("builtins.input", return_value=""):
            replay.main()

        self.assertEqual(self.output.getvalue(), "Please enter a trading idea.\n")
        self.openai.assert_not_called()
        self.load_dotenv.assert_not_called()

    def test_response_rejects_missing_sections_and_invalid_python(self):
        cases = [
            ("Explanation: Buy on a breakout.", "Python code block"),
            ("Explanation: Buy.\n```javascript\nconst signals = [];\n```", "Python code block"),
            ("Explanation: Buy.\n```python\nsignals = []", "Python code block"),
            ("```python\nsignals = []\n```", "explanation"),
            ("Explanation:\n\nCode:\n```python\nsignals = []\n```", "explanation"),
            ("## Explanation\n\n## Code\n```python\nsignals = []\n```", "explanation"),
            ("Explanation: Buy.\n```python\n\n```", "empty"),
            ("Explanation: Buy.\n```python\ndef strategy(:\n    pass\n```", "invalid Python"),
            (
                "Explanation: Buy.\n```python\nsignals = []\n```\n"
                "```python\ndef strategy(:\n    pass\n```",
                "invalid Python",
            ),
        ]
        for response, error in cases:
            with self.subTest(response=response):
                with self.assertRaisesRegex(ValueError, error):
                    replay.validate_response(response)

    def test_explanation_can_follow_code(self):
        replay.validate_response(
            "```python\nsignals = []\n```\nExplanation: Buy on a breakout."
        )

    def test_invalid_python_is_not_displayed(self):
        request = self.openai.return_value.__enter__.return_value.responses.create
        request.return_value = SimpleNamespace(
            status="completed",
            output_text="Explanation: Buy.\n```python\ndef strategy(:\n    pass\n```",
        )
        with patch("builtins.input", return_value="Buy on a breakout."):
            replay.main()

        self.assertIn("Error: AI response contains invalid Python", self.output.getvalue())
        self.assertNotIn("def strategy(:", self.output.getvalue())

    def test_generated_code_is_printed_without_execution(self):
        response_text = (
            "Explanation: Example code for inspection.\n"
            "```python\nraise RuntimeError('Generated code was executed')\n```"
        )
        request = self.openai.return_value.__enter__.return_value.responses.create
        request.return_value = SimpleNamespace(
            status="completed", output_text=response_text
        )
        with patch("builtins.input", return_value="Buy on a breakout."):
            replay.main()

        self.assertIn(response_text, self.output.getvalue())

    def test_request_timeout_shows_clear_error(self):
        request = self.openai.return_value.__enter__.return_value.responses.create
        request.side_effect = APITimeoutError(request=Mock())

        with patch("builtins.input", return_value="Buy on a breakout."):
            replay.main()

        self.assertIn("Error: OpenAI request timed out.", self.output.getvalue())


if __name__ == "__main__":
    unittest.main()
