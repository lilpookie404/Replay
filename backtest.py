"""Download daily prices and run the pinned reviewed SMA strategy in Docker."""

import hashlib
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from market_data import download_daily_prices
from numeric_eval_runner import BASELINE_REVIEW, image_id, load_baseline, run_container


def run_backtest(ticker: str, start_date: str, end_date: str) -> dict:
    """Return the reviewed 5/20 SMA backtest summary; never load an API key."""
    prices = download_daily_prices(ticker, start_date, end_date)
    if len(prices) < 21:
        raise ValueError("The 5/20 crossover backtest needs at least 21 daily bars.")
    baseline, digest = load_baseline(BASELINE_REVIEW)
    strategy = next((item for item in baseline
                     if item["id"] == "moving_average_execution_timing"), None)
    if strategy is None:
        raise ValueError("The reviewed moving-average strategy is missing.")
    image = image_id()
    worker = Path(__file__).with_name("numeric_eval") / "backtest.py"
    worker_bytes = worker.read_bytes()
    payload = {
        **strategy,
        "dates": [day.isoformat() for day in prices.index],
        "prices": {column: prices[column].astype(float).tolist() for column in ("Open", "Close")},
        "parameters": {"fast": 5, "slow": 20, "initial_cash": 100000.0},
    }
    with TemporaryDirectory(prefix="replay-backtest-") as directory:
        inputs = Path(directory).resolve()
        inputs.chmod(0o755)
        (inputs / "input.json").write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
        (inputs / "check.py").write_bytes(worker_bytes)
        for path in inputs.iterdir():
            path.chmod(0o444)
        report = run_container(strategy["id"], image, inputs)
    if not isinstance(report.get("summary"), dict) or not report["summary"]:
        raise RuntimeError("The sandbox returned no backtest summary.")
    report.update(
        ticker=ticker.strip().upper(), start_date=start_date, end_date=end_date,
        parameters=payload["parameters"], baseline_sha256=digest,
        sandbox_image_id=image, worker_sha256=hashlib.sha256(worker_bytes).hexdigest(),
        generated_code_executed=True,
    )
    return report


def main() -> int:
    ticker = input("US stock ticker: ")
    start = input("Start date (YYYY-MM-DD, inclusive): ")
    end = input("End date (YYYY-MM-DD, exclusive): ")
    try:
        result = run_backtest(ticker, start, end)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Backtest failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
