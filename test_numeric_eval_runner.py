import hashlib
import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from io import StringIO

import numeric_eval_runner as runner
from numeric_eval.check import compare_values


class NumericEvalRunnerTests(unittest.TestCase):
    def test_refuses_changed_baseline_before_docker(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "results.jsonl"
            source.write_text('{"id":"case","response":"changed"}\n')
            review = folder / "review.json"
            review.write_text(json.dumps({
                "results_file": source.name,
                "results_sha256": hashlib.sha256(b"original").hexdigest(),
                "cases": [{"id": "case", "verdict": "pass"}],
            }))
            with patch("numeric_eval_runner.subprocess.run") as command:
                with self.assertRaisesRegex(ValueError, "hash"):
                    runner.load_baseline(review)
                command.assert_not_called()

    def test_extracts_reviewed_code_as_text_without_running_it(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            code = "raise RuntimeError('This must not run on the host')\n"
            source = folder / "results.jsonl"
            source.write_text(json.dumps({
                "id": "case", "response_status": "completed", "error": None,
                "response": f"Explanation: Review only.\n```python\n{code}```",
            }) + "\n")
            review = folder / "review.json"
            review.write_text(json.dumps({
                "results_file": source.name,
                "results_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "cases": [{"id": "case", "verdict": "pass"}],
            }))

            records, digest = runner.load_baseline(review)

            self.assertEqual(records, [{"id": "case", "code": code}])
            self.assertEqual(digest, hashlib.sha256(source.read_bytes()).hexdigest())

    def test_container_is_restricted_and_mounts_only_staged_inputs(self):
        command = runner.container_command("test-container", "sha256:test", Path("/tmp/inputs"))
        for option, value in (
            ("--network", "none"), ("--user", "65534:65534"),
            ("--cap-drop", "ALL"), ("--security-opt", "no-new-privileges"),
            ("--memory", "512m"), ("--memory-swap", "512m"),
            ("--pids-limit", "32"), ("--cpus", "1"), ("--pull", "never"),
        ):
            self.assertEqual(command[command.index(option) + 1], value)
        self.assertIn("--read-only", command)
        self.assertEqual(command[command.index("--mount") + 1],
                         "type=bind,src=/tmp/inputs,dst=/input,readonly")
        self.assertIn("/tmp:rw,noexec,nosuid,nodev,size=16m", command)
        self.assertIn("max-size=64k", command)
        self.assertIn("max-file=1", command)
        self.assertNotIn("--env-file", command)
        self.assertNotIn("--privileged", command)
        self.assertEqual(command[-4:], ["python", "-I", "-B", "/input/check.py"])

    def test_timeout_removes_its_container_without_a_host_fallback(self):
        completed = subprocess.CompletedProcess([], 0, stdout="container-id\n")
        timeout = subprocess.TimeoutExpired(["docker", "wait"], 30)
        with patch("numeric_eval_runner.subprocess.run",
                   side_effect=[completed, timeout, completed]) as command:
            with self.assertRaises(subprocess.TimeoutExpired):
                runner.run_container("case", "sha256:test", Path("/tmp/inputs"), [])
        self.assertEqual(len(command.call_args_list), 3)
        cleanup = command.call_args_list[-1].args[0]
        self.assertEqual(cleanup[:3], ["docker", "rm", "--force"])
        self.assertTrue(all(call.args[0][0] == "docker" for call in command.call_args_list))

    def test_incomplete_sandbox_report_cannot_pass(self):
        completed = subprocess.CompletedProcess([], 0, stdout="container-id\n")
        waited = subprocess.CompletedProcess([], 0, stdout="0\n")
        logs = subprocess.CompletedProcess([], 0, stdout=json.dumps({
            "case_id": "case", "checks": [],
        }))
        with patch("numeric_eval_runner.subprocess.run",
                   side_effect=[completed, waited, logs, completed]):
            with self.assertRaisesRegex(ValueError, "checks"):
                runner.run_container("case", "sha256:test", Path("/tmp/inputs"), ["expected"])

    def test_missing_docker_image_fails_closed(self):
        with patch("numeric_eval_runner.subprocess.run",
                   side_effect=FileNotFoundError("docker")):
            with self.assertRaises(FileNotFoundError):
                runner.image_id()

    def test_numeric_comparator_detects_quantity_scaling_and_timing_errors(self):
        with self.assertRaisesRegex(AssertionError, "return"):
            compare_values({"return": 0.20}, {"return": 0.10})
        with self.assertRaisesRegex(AssertionError, "entry_index"):
            compare_values({"entry_index": 4}, {"entry_index": 5})
        with self.assertRaises(AssertionError):
            compare_values({"value": float("nan")}, {"value": 1.0})

    def test_numeric_comparator_accepts_tolerance_and_missing_exits(self):
        compare_values({"return": 0.1 + 1e-12, "exit": None},
                       {"return": 0.1, "exit": None})
        with self.assertRaises(AssertionError):
            compare_values({"exit": 6}, {"exit": None})

    def test_missing_runtime_fails_closed(self):
        with patch("numeric_eval_runner.image_id", side_effect=FileNotFoundError("Docker unavailable")):
            with patch("numeric_eval_runner.run_container") as execute:
                with patch("sys.stdout", new_callable=StringIO) as console:
                    self.assertEqual(runner.main(), 1)
                execute.assert_not_called()
                self.assertIn("No host execution fallback", console.getvalue())

    def test_incomplete_sandbox_reports_are_rejected_and_container_removed(self):
        for report in (
            [],
            {"case_id": "case", "checks": []},
            {"case_id": "case", "checks": [{"id": "fixture", "passed": "true"}]},
        ):
            with self.subTest(report=report):
                outputs = [
                    subprocess.CompletedProcess([], 0, stdout="container-id\n"),
                    subprocess.CompletedProcess([], 0, stdout="0\n"),
                    subprocess.CompletedProcess([], 0, stdout=json.dumps(report)),
                    subprocess.CompletedProcess([], 0, stdout=""),
                ]
                with patch("numeric_eval_runner.subprocess.run", side_effect=outputs) as command:
                    with self.assertRaises(ValueError):
                        runner.run_container("case", "sha256:test", Path("/tmp/inputs"), ["fixture"])
                self.assertEqual(command.call_args.args[0][:3], ["docker", "rm", "--force"])

    def test_cleanup_failure_is_reported_instead_of_silently_accepted(self):
        def command(arguments, **kwargs):
            if arguments[1] == "run":
                return subprocess.CompletedProcess(arguments, 0, stdout="container-id\n")
            if arguments[1] == "wait":
                return subprocess.CompletedProcess(arguments, 0, stdout="0\n")
            if arguments[1] == "logs":
                return subprocess.CompletedProcess(arguments, 0, stdout=json.dumps({
                    "case_id": "case", "checks": [{"id": "fixture", "passed": True}],
                }))
            if kwargs.get("check"):
                raise subprocess.CalledProcessError(1, arguments)
            return subprocess.CompletedProcess(arguments, 1, stdout="")

        with patch("numeric_eval_runner.subprocess.run", side_effect=command):
            with self.assertRaises(subprocess.CalledProcessError):
                runner.run_container("case", "sha256:test", Path("/tmp/inputs"), ["fixture"])

    def test_inputs_and_hashes_are_frozen_for_the_entire_run(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            inputs = folder / "numeric_eval"
            inputs.mkdir()
            fixture_bytes = b'{"fixtures":[{"id":"fixture"}],"invalid_windows":[]}'
            worker_bytes = b'# Original trusted helper\n'
            (inputs / "fixtures.json").write_bytes(fixture_bytes)
            (inputs / "check.py").write_bytes(worker_bytes)
            snapshots = []

            def execute(case_id, image, staged, checks):
                snapshots.append(((staged / "fixtures.json").read_bytes(),
                                  (staged / "check.py").read_bytes()))
                (inputs / "fixtures.json").write_text('{"fixtures":[],"invalid_windows":[]}')
                (inputs / "check.py").write_text('# Edited while the first case was running\n')
                return {"case_id": case_id, "checks": [{"id": "fixture", "passed": True}]}

            payloads = [{"id": name, "code": "raise RuntimeError('Never run on host')"}
                        for name in ("first", "second")]
            with patch.object(runner, "FOLDER", folder):
                with patch.object(runner, "load_baseline", return_value=(payloads, "digest")):
                    with patch.object(runner, "image_id", return_value="sha256:test"):
                        with patch.object(runner, "run_container", side_effect=execute):
                            with patch("sys.stdout", new_callable=StringIO):
                                self.assertEqual(runner.main(), 0)
            self.assertEqual(snapshots, [(fixture_bytes, worker_bytes)] * 2)
            records = [json.loads(line) for line in next(folder.glob('numeric_results_*.jsonl')).read_text().splitlines()]
            self.assertTrue(all(item["fixture_sha256"] == hashlib.sha256(fixture_bytes).hexdigest()
                                for item in records))


if __name__ == "__main__":
    unittest.main()
