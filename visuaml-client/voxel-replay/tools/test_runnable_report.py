"""Exercise the sender and recipient CLI as separate processes."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import wasmtime

BASE = Path(__file__).resolve().parents[1]
EXAMPLES = BASE / "release" / "examples"
CLI = BASE / "tools" / "visuaml_replay.py"


def command(*args, success=True):
    result = subprocess.run([sys.executable, str(CLI), *map(str, args)],
                            check=success, capture_output=True, text=True)
    if not success:
        assert result.returncode == 2, result
    return result


class RunnableReportTests(unittest.TestCase):
    def test_shipped_reports(self):
        started = json.loads(command("open", EXAMPLES / "suspended-passage-report.json").stdout)
        self.assertEqual(len(started["trace"]), 17)
        returns = [row["stack"] for row in started["trace"] if row["depth"] == 0]
        self.assertEqual(returns[0], [65512, 7])
        self.assertFalse(any(row["access"][:2] == [2, 65512] for row in started["trace"]))

        comparison = json.loads(command("open", EXAMPLES / "compiler-divergence-report.json").stdout)
        first = comparison["first_difference"]
        self.assertEqual(first["reference"]["pc"], 160)
        self.assertEqual(first["reference"]["stack"], [256])
        self.assertEqual(first["candidate"]["stack"], [0])
        self.assertEqual(comparison["reference"]["trace"][1:], comparison["candidate"]["trace"][1:])

    def test_ci_failure_recipe(self):
        script = EXAMPLES / "ci/check_guest.py"
        with TemporaryDirectory() as directory:
            path = Path(directory) / "failure.json"
            failed = subprocess.run([sys.executable, str(script), "--report", str(path)],
                                    capture_output=True, text=True)
            self.assertEqual(failed.returncode, 1, failed.stderr)
            result = json.loads(command("open", path).stdout)
            self.assertEqual((result["expected_return"], result["returned"], result["expectation_met"]),
                             (43, 42, False))
            path.unlink()
            subprocess.run([sys.executable, str(script), "--expected", "42", "--report", str(path)],
                           check=True, capture_output=True)
            self.assertFalse(path.exists())

    def test_new_guest_and_corrupt_claim(self):
        source = """(module
          (memory (export "memory") 1 1)
          (func (param i32) (result i32)
            i32.const 32 local.get 0 i32.store
            local.get 0 i32.const 14 i32.add)
          (func (export "run") (param i32) (result i32)
            local.get 0 call 0 i32.const 3 i32.mul))"""
        with TemporaryDirectory() as directory:
            path = Path(directory)
            guest, bundle, report = (path / name for name in ("new.wasm", "bundle.json", "report.json"))
            guest.write_bytes(wasmtime.wat2wasm(source))
            command("capture", guest, bundle, "--argument", 11, "--after-step", 5)
            command("pack", bundle, report)
            guest.unlink()
            bundle.unlink()
            result = json.loads(command("open", report, "--output", path / "result.json").stdout)
            self.assertEqual(result["returned"], 75)
            self.assertEqual(result, json.loads((path / "result.json").read_text()))
            changed = json.loads(report.read_text())
            self.assertEqual(changed["bundle"]["checkpoint"]["depth"], 1)
            changed["expected_result_sha256"] = "0"*64
            report.write_text(json.dumps(changed))
            self.assertIn("failed to reproduce", command("open", report, success=False).stderr)


if __name__ == "__main__":
    unittest.main()
