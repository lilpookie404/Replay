import unittest
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

from market_data import download_daily_prices


class MarketDataTests(unittest.TestCase):
    def setUp(self):
        self.download = self.enterContext(patch("market_data.yf.download"))
        self.data = pd.DataFrame(
            {"Open": [101.0, 100.0], "Close": [102.0, 101.0], "Volume": [20, 10]},
            index=pd.DatetimeIndex(["2025-01-03", "2025-01-02"], name="Date"),
        )
        self.download.return_value = self.data

    def test_downloads_normalized_ticker_and_returns_sorted_price_columns(self):
        actual = download_daily_prices("  aapl  ", "2025-01-01", "2025-01-05")

        assert_frame_equal(actual, self.data[["Open", "Close"]].sort_index())
        self.download.assert_called_once_with(
            "AAPL", start="2025-01-01", end="2025-01-05", interval="1d",
            auto_adjust=True, multi_level_index=False, ignore_tz=True,
            progress=False, threads=False, timeout=10,
        )
        actual.iloc[0, 0] = 999.0
        self.assertEqual(self.data.loc["2025-01-02", "Open"], 100.0)

    def test_rejects_missing_or_invalid_inputs_before_downloading(self):
        cases = [
            (None, "2025-01-01", "2025-01-05"),
            ("", "2025-01-01", "2025-01-05"),
            ("   ", "2025-01-01", "2025-01-05"),
            (123, "2025-01-01", "2025-01-05"),
            ("AAPL MSFT", "2025-01-01", "2025-01-05"),
            ("AAPL,MSFT", "2025-01-01", "2025-01-05"),
            ("AAPL", None, "2025-01-05"),
            ("AAPL", "", "2025-01-05"),
            ("AAPL", "2025-02-30", "2025-03-05"),
            ("AAPL", "2025-01-01", None),
            ("AAPL", "2025-01-01", ""),
            ("AAPL", "2025-01-01", "not-a-date"),
            ("AAPL", "2025-01-05", "2025-01-01"),
            ("AAPL", "2025-01-01", "2025-01-01"),
        ]
        for ticker, start, end in cases:
            with self.subTest(ticker=ticker, start=start, end=end):
                with self.assertRaises(ValueError):
                    download_daily_prices(ticker, start, end)
        self.download.assert_not_called()

    def test_reports_download_exceptions(self):
        failure = TimeoutError("Connection timed out")
        self.download.side_effect = failure

        with self.assertRaisesRegex(RuntimeError, "download.*AAPL") as caught:
            download_daily_prices("AAPL", "2025-01-01", "2025-01-05")

        self.assertIs(caught.exception.__cause__, failure)

    def test_rejects_non_hyphenated_dates_before_downloading(self):
        for start, end in (
            ("20250101", "2025-01-05"),
            ("2025-01-01", "20250105"),
            ("2025-W01-3", "2025-01-05"),
        ):
            with self.subTest(start=start, end=end):
                with self.assertRaisesRegex(ValueError, "YYYY-MM-DD"):
                    download_daily_prices("AAPL", start, end)
        self.download.assert_not_called()

    def test_rejects_empty_downloads(self):
        for response in (None, pd.DataFrame(), pd.DataFrame(columns=["Open", "Close"])):
            with self.subTest(response=type(response).__name__):
                self.download.return_value = response
                with self.assertRaisesRegex(RuntimeError, "No daily prices"):
                    download_daily_prices("AAPL", "2025-01-01", "2025-01-05")

    def test_rejects_missing_or_ambiguous_price_columns(self):
        for response in (
            self.data.drop(columns="Open"),
            self.data.drop(columns="Close"),
            pd.concat([self.data, self.data[["Open"]]], axis=1),
            pd.concat({"AAPL": self.data}, axis=1),
        ):
            with self.subTest(columns=list(response.columns)):
                self.download.return_value = response
                with self.assertRaisesRegex(RuntimeError, "Open.*Close"):
                    download_daily_prices("AAPL", "2025-01-01", "2025-01-05")

    def test_rejects_unusable_prices_without_dropping_rows(self):
        for value in (float("nan"), float("inf"), float("-inf"), 0, -1, "bad", True, 1 + 2j):
            with self.subTest(value=value):
                self.download.return_value = pd.DataFrame(
                    {"Open": [value, 101.0], "Close": [101.0, 102.0]},
                    index=pd.to_datetime(["2025-01-02", "2025-01-03"]),
                )
                with self.assertRaisesRegex(RuntimeError, "usable.*Open.*Close"):
                    download_daily_prices("AAPL", "2025-01-01", "2025-01-05")

    def test_requires_valid_unique_date_index(self):
        for index in (
            pd.RangeIndex(2),
            pd.DatetimeIndex(["2025-01-02", None]),
            pd.DatetimeIndex(["2025-01-02", "2025-01-02"]),
        ):
            with self.subTest(index=index):
                self.download.return_value = pd.DataFrame(
                    {"Open": [100.0, 101.0], "Close": [101.0, 102.0]}, index=index,
                )
                with self.assertRaisesRegex(RuntimeError, "date index"):
                    download_daily_prices("AAPL", "2025-01-01", "2025-01-05")

    def test_rejects_data_outside_requested_dates(self):
        with self.assertRaisesRegex(RuntimeError, "date range"):
            download_daily_prices("AAPL", "2025-01-01", "2025-01-03")


if __name__ == "__main__":
    unittest.main()
