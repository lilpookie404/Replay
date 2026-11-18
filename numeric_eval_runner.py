"""Run only the reviewed baseline in resource-limited, offline Docker containers."""

import ast
import hashlib
import json
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory


FOLDER = Path(__file__).resolve().parent
BASELINE_REVIEW = FOLDER / "eval_review_20261118T041241597328Z.json"
IMAGE = "replay-numeric-eval:1"


def load_baseline(review_path):
    review = json.loads(review_path.read_text(encoding="utf-8"))
    name = review["results_file"]
    if Path(name).name != name:
        raise ValueError("Baseline results must be beside their review")
    raw = (review_path.parent / name).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != review["results_sha256"]:
        raise ValueError("Baseline results hash does not match its reviewed hash")
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    if (not records
            or [record["id"] for record in records] != [case["id"] for case in review["cases"]]
            or not all(case["verdict"] == "pass" for case in review["cases"])):
        raise ValueError("Baseline must have matching, passing static reviews")
    payloads = []
    for record in records:
        if record["response_status"] != "completed" or record["error"] is not None:
            raise ValueError("Baseline includes an incomplete API response")
        blocks = re.findall(
            r"^[ \t]*```python[ \t]*\r?\n(.*?)^[ \t]*```[ \t]*\r?$",
            record["response"], re.MULTILINE | re.DOTALL,
        )
        if len(blocks) != 1:
            raise ValueError("Each reviewed baseline must contain exactly one Python block")
        ast.parse(blocks[0])
        payloads.append({"id": record["id"], "code": blocks[0]})
    return payloads, digest


def image_id():
    return subprocess.run(
        ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"],
        check=True, capture_output=True, text=True, timeout=10,
    ).stdout.strip()


def container_command(name, image, inputs):
    return [
        "docker", "run", "--detach", "--name", name, "--pull", "never",
        "--network", "none", "--read-only", "--user", "65534:65534",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--memory", "512m", "--memory-swap", "512m", "--cpus", "1",
        "--pids-limit", "32", "--ulimit", "core=0", "--ulimit", "nofile=64:64",
        "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=16m",
        "--log-driver", "json-file", "--log-opt", "max-size=64k",
        "--log-opt", "max-file=1",
        "--mount", f"type=bind,src={inputs},dst=/input,readonly",
        image, "timeout", "--signal=KILL", "25s", "python", "-I", "-B", "/input/check.py",
    ]


def run_container(case_id, image, inputs, expected_checks):
    name = f"replay-numeric-{uuid.uuid4().hex}"
    try:
        subprocess.run(container_command(name, image, inputs),
                       check=True, capture_output=True, text=True, timeout=15)
        waited = subprocess.run(["docker", "wait", name], check=True,
                                capture_output=True, text=True, timeout=30)
        if waited.stdout.strip() != "0":
            raise RuntimeError(f"Sandbox exited with code {waited.stdout.strip()}")
        # Container logs are size-limited; never capture an unlimited generated stdout stream.
        logs = subprocess.run(["docker", "logs", "--tail", "1", name], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, timeout=10)
        if len(logs.stdout) > 65536:
            raise ValueError("Sandbox report exceeds its size limit")
        report = json.loads(logs.stdout)
        if not isinstance(report, dict):
            raise ValueError("Sandbox report must be a JSON object")
        checks = report.get("checks", [])
        if (not isinstance(checks, list)
                or not all(isinstance(check, dict) for check in checks)
                or report.get("case_id") != case_id
                or [check.get("id") for check in checks] != expected_checks
                or not all(type(check.get("passed")) is bool for check in checks)):
            raise ValueError("Sandbox report has missing or invalid checks")
        return report
    finally:
        subprocess.run(["docker", "rm", "--force", name],
                       check=True, capture_output=True, timeout=10)


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_path = FOLDER / f"numeric_results_{timestamp}.jsonl"
    try:
        payloads, digest = load_baseline(BASELINE_REVIEW)
        image = image_id()
        fixture_path = FOLDER / "numeric_eval" / "fixtures.json"
        worker_path = FOLDER / "numeric_eval" / "check.py"
        fixture_bytes = fixture_path.read_bytes()
        worker_bytes = worker_path.read_bytes()
        fixtures = json.loads(fixture_bytes)
        check_ids = [item["id"] for item in fixtures["fixtures"] + fixtures["invalid_windows"]]
        if not check_ids or len(set(check_ids)) != len(check_ids):
            raise ValueError("Numerical fixtures need unique, nonempty check IDs")
        failed = False
        with output_path.open("x", encoding="utf-8") as output:
            for payload in payloads:
                with TemporaryDirectory(prefix="replay-numeric-") as directory:
                    inputs = Path(directory).resolve()
                    inputs.chmod(0o755)
                    (inputs / "input.json").write_text(json.dumps(payload), encoding="utf-8")
                    (inputs / "fixtures.json").write_bytes(fixture_bytes)
                    (inputs / "check.py").write_bytes(worker_bytes)
                    for path in inputs.iterdir():
                        path.chmod(0o444)
                    report = run_container(payload["id"], image, inputs, check_ids)
                report.update(
                    baseline_review=BASELINE_REVIEW.name, baseline_sha256=digest,
                    sandbox_image_id=image, generated_code_executed=True,
                    fixture_sha256=hashlib.sha256(fixture_bytes).hexdigest(),
                    worker_sha256=hashlib.sha256(worker_bytes).hexdigest(),
                )
                output.write(json.dumps(report, allow_nan=False) + "\n")
                output.flush()
                passed = sum(check["passed"] for check in report["checks"])
                failed |= passed != len(check_ids)
                print(f"{payload['id']}: {passed}/{len(check_ids)} numeric checks passed", flush=True)
        print(f"Saved numerical results to {output_path}")
        return 1 if failed else 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Numerical evaluation stopped: {type(error).__name__}: {error}")
        print("No host execution fallback is available. Check Docker and the pinned sandbox image.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
