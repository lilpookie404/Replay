"""Trusted numeric comparisons. Generated code is loaded only inside Docker."""

import contextlib
import copy
import json
import math
import os
import sys
from pathlib import Path


def compare_values(actual, expected, path="result"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise AssertionError(f"{path}: expected an object")
        for key, value in expected.items():
            if key not in actual:
                raise AssertionError(f"{path}.{key}: missing")
            compare_values(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"{path}: wrong number of entries")
        for index, value in enumerate(expected):
            compare_values(actual[index], value, f"{path}[{index}]")
    elif expected is None:
        if actual is not None:
            raise AssertionError(f"{path}: expected missing value, got {actual!r}")
    elif isinstance(expected, (int, float)):
        if (not isinstance(actual, (int, float)) or isinstance(actual, bool)
                or not math.isfinite(actual)
                or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)):
            raise AssertionError(f"{path}: expected {expected!r}, got {actual!r}")
    elif actual != expected:
        raise AssertionError(f"{path}: expected {expected!r}, got {actual!r}")


def invoke(namespace, case_id, data, fixture, fast=None, slow=None):
    fast = fixture["fast"] if fast is None else fast
    slow = fixture["slow"] if slow is None else slow
    if case_id == "moving_average_execution_timing":
        return namespace["backtest_ma_crossover"](
            data, fast=fast, slow=slow, initial_cash=fixture["initial_capital"]
        )
    if case_id == "invalid_moving_average_windows":
        trades, summary, _ = namespace["moving_average_crossover_backtest"](
            data, fast=fast, slow=slow, qty=fixture["qty"],
            initial_capital=fixture["initial_capital"],
        )
        return {"trades": trades, "summary": summary}
    if case_id == "unfinished_open_trade":
        return namespace["backtest_ma_crossover"](
            data, qty=fixture["qty"], fast=fast, slow=slow
        )
    raise ValueError("No reviewed adapter for this case")


def observe(result, case_id, fixture, pd):
    def scalar(value):
        return None if pd.isna(value) else float(value)

    entry_key = "entry_time" if case_id == "moving_average_execution_timing" else "entry_fill_time"
    exit_key = "exit_time" if case_id == "moving_average_execution_timing" else "exit_fill_time"
    trades = []
    for _, row in result["trades"].iterrows():
        trade = {key: scalar(row[key]) for key in (
            "entry_price", "exit_price", "qty", "realized_pnl", "realized_return",
            "unrealized_pnl", "unrealized_return",
        )}
        trade.update(entry_index=scalar(row[entry_key]), exit_index=scalar(row[exit_key]))
        trades.append(trade)

    summary = result["summary"]
    mappings = {
        "moving_average_execution_timing": {
            "realized_pnl": "realized_pnl", "unrealized_pnl": "unrealized_pnl",
            "total_pnl": "total_pnl", "wins": "win_count", "losses": "loss_count",
            "return_fraction": "portfolio_return", "return_denominator": "initial_cash",
            "mean_win": "mean_win", "mean_loss": "mean_loss",
        },
        "invalid_moving_average_windows": {
            "realized_pnl": "total_realized_pnl", "unrealized_pnl": "total_unrealized_pnl",
            "total_pnl": "total_pnl", "wins": "wins", "losses": "losses",
            "return_fraction": "portfolio_return_on_initial_capital",
            "return_denominator": "initial_capital",
            "mean_win": "average_win", "mean_loss": "average_loss",
        },
        "unfinished_open_trade": {
            "realized_pnl": "realized_pnl_sum", "unrealized_pnl": "unrealized_pnl_sum",
            "total_pnl": "total_pnl_sum", "wins": "win_count", "losses": "loss_count",
            "return_fraction": "total_return_agg", "return_denominator": "capital_denominator",
            "aggregate_realized_return": "realized_return_agg",
            "aggregate_unrealized_return": "unrealized_return_agg",
        },
    }
    observed = {"trades": trades, "summary": {
        key: scalar(summary[source]) for key, source in mappings[case_id].items()
    }}
    expected = copy.deepcopy(fixture["expected"])
    if case_id == "unfinished_open_trade":
        expected["summary"].update(fixture["entry_outlay_summary"])
    else:
        expected["summary"].update(
            return_fraction=fixture["account_return"],
            return_denominator=fixture["initial_capital"],
        )
        expected["summary"].update(fixture["win_loss_means"])
    if case_id == "moving_average_execution_timing":
        observed["summary"]["final_equity"] = scalar(result["equity_curve"].iloc[-1])
        expected["summary"]["final_equity"] = (
            fixture["initial_capital"] + fixture["expected"]["summary"]["total_pnl"]
        )
    return observed, expected


def run_checks(payload, fixtures):
    # This guard prevents accidental host use; Docker's OS restrictions provide isolation.
    if not Path("/.dockerenv").is_file() or os.getuid() == 0:
        raise RuntimeError("Numerical checks require the unprivileged Docker runner")
    import numpy as np
    import pandas as pd

    namespace = {"__name__": "reviewed_baseline"}
    exec(compile(payload["code"], "reviewed_baseline.py", "exec"), namespace)
    checks = []
    for fixture in fixtures["fixtures"]:
        check = {"id": fixture["id"], "passed": False, "observed": None, "error": None}
        try:
            result = invoke(namespace, payload["id"], pd.DataFrame(fixture["prices"]), fixture)
            actual, expected = observe(result, payload["id"], fixture, pd)
            check["observed"] = actual
            compare_values(actual, expected)
            check["passed"] = True
        except Exception as error:
            check["error"] = f"{type(error).__name__}: {error}"[:1000]
        checks.append(check)
    for invalid in fixtures["invalid_windows"]:
        check = {"id": invalid["id"], "passed": False, "error": None}
        try:
            fixture = fixtures["fixtures"][0]
            invoke(namespace, payload["id"], pd.DataFrame(fixture["prices"]), fixture,
                   fast=invalid["fast"], slow=invalid["slow"])
            check["error"] = "Invalid windows were accepted"
        except ValueError:
            check["passed"] = True
        except Exception as error:
            check["error"] = f"{type(error).__name__}: {error}"[:1000]
        checks.append(check)
    return {
        "case_id": payload["id"], "checks": checks,
        "environment": {"python": sys.version.split()[0], "pandas": pd.__version__,
                        "numpy": np.__version__},
    }


if __name__ == "__main__":
    payload = json.loads(Path("/input/input.json").read_text())
    fixtures = json.loads(Path("/input/fixtures.json").read_text())
    with open(os.devnull, "w") as discard:
        with contextlib.redirect_stdout(discard), contextlib.redirect_stderr(discard):
            report = run_checks(payload, fixtures)
    print(json.dumps(report, allow_nan=False), flush=True)
