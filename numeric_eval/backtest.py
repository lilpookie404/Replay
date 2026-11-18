"""Trusted container entry point for the pinned strategy and downloaded prices."""

import contextlib
import json
import os
from pathlib import Path


if __name__ == "__main__":
    # Docker enforces isolation; this guard prevents accidental direct host execution.
    if not Path("/.dockerenv").is_file() or os.getuid() == 0:
        raise RuntimeError("Backtests require the unprivileged Docker runner")
    import pandas as pd

    payload = json.loads(Path("/input/input.json").read_text(encoding="utf-8"))
    data = pd.DataFrame(
        payload["prices"], index=pd.DatetimeIndex(pd.to_datetime(payload["dates"]), name="Date"),
    )
    with open(os.devnull, "w") as discard:
        with contextlib.redirect_stdout(discard), contextlib.redirect_stderr(discard):
            namespace = {"__name__": "reviewed_strategy"}
            exec(compile(payload["code"], "reviewed_strategy.py", "exec"), namespace)
            result = namespace["backtest_ma_crossover"](data, **payload["parameters"])
    report = {
        "case_id": payload["id"], "bars": len(data),
        "first_date": data.index[0].isoformat(), "last_date": data.index[-1].isoformat(),
        "trade_count": len(result["trades"]), "summary": result["summary"],
    }
    print(json.dumps(report, allow_nan=False), flush=True)
