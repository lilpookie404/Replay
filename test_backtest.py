import json
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import backtest


class BacktestTests(unittest.TestCase):
    def setUp(self):
        self.quotes = pd.DataFrame(
            {"Open": [100.0] * 25, "Close": [100.0] * 20 + [120.0] * 5},
            index=pd.bdate_range("2025-01-02", periods=25, name="Date"),
        )
        self.download = self.enterContext(patch("market_data.yf.download", return_value=self.quotes))
        self.baseline = self.enterContext(patch(
            "backtest.load_baseline", return_value=([
                {"id": "moving_average_execution_timing", "code": "raise RuntimeError('Must not run on host')"},
            ], "baseline-digest"),
        ))
        self.image = self.enterContext(patch("backtest.image_id", return_value="sha256:test"))
        self.sandbox = self.enterContext(patch("backtest.run_container"))
        self.env_loader = self.enterContext(patch("dotenv.load_dotenv"))
        self.openai = self.enterContext(patch("openai.OpenAI"))

    def test_downloads_and_stages_only_prices_reviewed_code_and_worker(self):
        staged_directory = []

        def execute(case_id, image, inputs):
            staged_directory.append(inputs)
            self.assertEqual(case_id, "moving_average_execution_timing")
            self.assertEqual(image, "sha256:test")
            self.assertEqual({path.name for path in inputs.iterdir()}, {"input.json", "check.py"})
            payload = json.loads((inputs / "input.json").read_text())
            self.assertEqual(payload["prices"]["Open"], [100.0] * 25)
            self.assertEqual(payload["dates"], [day.isoformat() for day in self.quotes.index])
            self.assertEqual(payload["parameters"], {"fast": 5, "slow": 20, "initial_cash": 100000.0})
            self.assertIn("Must not run on host", payload["code"])
            self.assertNotIn("API_KEY", (inputs / "input.json").read_text())
            self.assertEqual((inputs / "check.py").read_bytes(),
                             Path("numeric_eval/backtest.py").read_bytes())
            return {"case_id": case_id, "bars": 25,
                    "summary": {"total_pnl": 20000.0, "portfolio_return": 0.2}, "trade_count": 1}

        self.sandbox.side_effect = execute

        result = backtest.run_backtest(" aapl ", "2025-01-01", "2025-03-01")

        self.assertEqual(result["ticker"], "AAPL")
        self.assertEqual(result["baseline_sha256"], "baseline-digest")
        self.assertEqual(result["summary"]["total_pnl"], 20000.0)
        self.assertTrue(result["generated_code_executed"])
        self.assertFalse(staged_directory[0].exists())
        self.download.assert_called_once()
        self.env_loader.assert_not_called()
        self.openai.assert_not_called()

    def test_invalid_dates_never_download_or_start_a_backtest(self):
        with self.assertRaises(ValueError):
            backtest.run_backtest("AAPL", "20250101", "2025-03-01")
        self.download.assert_not_called()
        self.sandbox.assert_not_called()

    def test_download_failure_stops_before_sandbox(self):
        self.download.side_effect = ConnectionError("Provider unavailable")
        with self.assertRaisesRegex(RuntimeError, "download"):
            backtest.run_backtest("AAPL", "2025-01-01", "2025-03-01")
        self.sandbox.assert_not_called()

    def test_insufficient_history_stops_before_sandbox(self):
        self.download.return_value = self.quotes.iloc[:20]
        with self.assertRaisesRegex(ValueError, "21"):
            backtest.run_backtest("AAPL", "2025-01-01", "2025-03-01")
        self.sandbox.assert_not_called()

    def test_unavailable_docker_does_not_execute_on_host(self):
        self.image.side_effect = FileNotFoundError("Docker is unavailable")
        with self.assertRaises(FileNotFoundError):
            backtest.run_backtest("AAPL", "2025-01-01", "2025-03-01")
        self.sandbox.assert_not_called()

    def test_missing_reviewed_strategy_stops_before_sandbox(self):
        self.baseline.return_value = ([], "digest")
        with self.assertRaisesRegex(ValueError, "reviewed"):
            backtest.run_backtest("AAPL", "2025-01-01", "2025-03-01")
        self.sandbox.assert_not_called()

    def test_main_reports_a_provider_error(self):
        with patch("builtins.input", side_effect=["AAPL", "2025-01-01", "2025-03-01"]):
            with patch("backtest.run_backtest", side_effect=RuntimeError("Download failed")):
                with patch("sys.stdout", new_callable=StringIO) as console:
                    self.assertEqual(backtest.main(), 1)
        self.assertIn("Download failed", console.getvalue())


if __name__ == "__main__":
    unittest.main()
