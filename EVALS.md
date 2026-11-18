# Reviewing model responses

`python -m unittest` tests the application and runner with fake API responses.
It does not grade saved model outputs or validate trading strategies.

When authorized, `python eval_runner.py` sends one request per case and saves
a new `eval_results_<timestamp>.jsonl` file. Each record includes the case and
its review criteria, raw response, API status, and any request error. New
records have `review_status: "pending"`, including completed API responses.

Review each response against the criteria stored in its result record. Those
criteria preserve what was requested for that run even if `eval_cases.json`
changes later. Trace the implementation rather than relying on its explanation:

- Check signal timestamps, fill prices, returns, and trade logs for agreement.
- Check final-bar branches for a fallback that violates next-bar execution.
- Check indicator warm-up periods and invalid window handling.
- Check open positions and unrealized P&L at the end of the data.
- Check both realized and unrealized fractional returns at `qty > 1`: divide
  currency P&L by `entry_price * qty`. For entry 100 and qty 2, exit 110 gives
  P&L 20 and return 0.10; an open trade marked at 105 gives P&L 10 and return
  0.05. Changing quantity must not change these fractional returns.
- Check summary denominators and empty win/loss groups as well as trade rows.
- Check for placeholder logic, discarded loops, and conflicting implementations.

The suite remains three requests. The `unfinished_open_trade` case also
covers non-unit quantities and a completed trade's return calculation.

Preserve the raw results. Record each manual verdict (`pass`, `fail`, or
`unresolved`) and its supporting findings in a separate
`eval_review_<timestamp>.json`, referencing the results file. A pass applies
only to that case's reviewed criteria. Leave uncertain verdicts unresolved.

The first run's user review is in `eval_review_20261117T184149440483Z.json`:
one case passed its listed window rules and two cases failed. The raw run has
been preserved. No generated code was executed during that review.

`ast.parse` checks Python syntax. Semantic review remains manual; these steps
do not establish trading correctness or authorize executing generated code.

## Separate numerical baseline check

`numeric_eval_runner.py` is a separate, explicitly invoked execution step. It
pins the reviewed run `eval_results_20261118T041241597328Z.jsonl` using its
static-review SHA-256. It refuses modified results, unreviewed outputs, missing
Docker, and missing sandbox images. It never falls back to host execution.
Earlier results and static reviews are preserved.

Build the sandbox image once with a running Docker engine, then run the check:

```sh
docker build -t replay-numeric-eval:1 numeric_eval
source .venv/bin/activate
python numeric_eval_runner.py
```

The image contains Python 3.12, NumPy 2.1.3, and pandas 2.2.3. These packages
are isolated from Replay's application environment and `requirements.txt`.
The build context excludes everything except the Dockerfile. Package downloads
happen during the image build; numerical containers have no external network.

Each reviewed response runs in its own non-root container, with a read-only
root and read-only staged code/fixtures. The project, `.env`, Docker socket,
and host credentials are not mounted. The container has no capabilities,
cannot gain privileges, and is capped at one CPU, 512 MB memory, 32 processes,
16 MB temporary storage, and 25 seconds of wall time. Logs are bounded to one
64 KB file. The host also times out waiting and removes its named container.
These controls reduce risk; containers and the Docker engine must remain patched.

`numeric_eval/fixtures.json` contains literal, hand-calculated expectations.
Windows are shortened to 2 and 3 for small fixtures, so this stage does not
establish correctness for every parameter choice. The main nine-bar fixture is:

| Bar index | Open | Close | Signal at Close | Fill at this Open |
|---|---|---|---|---|
| 0 | 100 | 100 | Warm-up | None |
| 1 | 100 | 100 | Warm-up | None |
| 2 | 100 | 98 | Previous slow MA unavailable | None |
| 3 | 100 | 102 | Equal averages | None |
| 4 | 100 | 104 | Buy | None |
| 5 | 100 | 96 | Sell | Buy |
| 6 | 110 | 98 | None | Sell |
| 7 | 100 | 102 | Buy | None |
| 8 | 100 | 105 | None | Buy, remains open |

At quantity 2, the closed trade earns 20 (return 0.10); the open trade earns
10 unrealized (return 0.05). Initial capital 200 ends at equity 230, return
0.15. The third response instead reports return on aggregate entry outlay:
30 / 400 = 0.075. Those are distinct denominators and are checked separately.
At quantity 1, currency P&L and initial capital halve; fractional returns stay
the same. Separate fixtures cover a closed loss, empty win/loss groups,
unfilled final-bar buys and sells, warm-up, equal windows, and invalid windows.

The stage saves a new `numeric_results_<timestamp>.jsonl` with observations,
individual check results, baseline/fixture/worker hashes, image ID, and package
versions. A mismatch makes the command exit nonzero. These results are separate
from API completion, mocked unit tests, and static-review verdicts.
Fixtures and the trusted worker are frozen at the start of each run, so edits
during execution cannot silently change later cases or their recorded hashes.

`python -m unittest` only checks trusted comparison/runner code and mocked
Docker operations. It never executes the saved generated programs. Numerical
success is limited to the fixtures and pinned runtime, not trading suitability.

The first numerical run is `numeric_results_20261118T060903450073Z.jsonl`:
15 checks passed for each of the three responses (45 total), including both
quantities, fill timing, P&L/return reconciliation, open positions, warm-up,
and invalid windows. Its Python runtime was 3.12.16. The trusted isolation
probe passed all eight checks and is saved separately as
`numeric_sandbox_probe_20261118T060903449053Z.json`. All temporary containers
were removed. Static reviews and original model responses were not modified.

Docker references: [runtime restrictions](https://docs.docker.com/engine/containers/run/),
[security boundaries](https://docs.docker.com/engine/security/), and
[bounded logs](https://docs.docker.com/engine/logging/drivers/json-file/).
