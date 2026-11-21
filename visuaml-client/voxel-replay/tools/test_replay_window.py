"""Check bounded windows against full source replay and terminal transitions."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

import torch
import wasmtime

from visuaml.replay import Replay, check_report, inspect_window, make_bundle

BASE = Path(__file__).resolve().parents[1]


class ReplayWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.fixture = BASE / "release/examples/suspended-passage-report.json"
        cls.report = json.loads(cls.fixture.read_text())

    def test_saved_report_and_fresh_process(self):
        full = check_report(self.report)
        window = inspect_window(self.report, after=16, limit=3)
        self.assertEqual(window["start"]["stack"], [7])
        self.assertEqual(window["trace"][0]["stack"], [65512, 7])
        for row, expected in zip(window["trace"], full["trace"][2:5], strict=True):
            self.assertEqual({k: row[k] for k in expected}, expected)
        run = subprocess.run(
            [sys.executable, str(BASE/"tools/visuaml_replay.py"), "inspect",
             str(self.fixture), "--after-step", "16", "--steps", "3"],
            check=True, capture_output=True, text=True,
        )
        self.assertEqual(json.loads(run.stdout), window)

    def test_absolute_bounds(self):
        with self.assertRaisesRegex(ValueError, "precedes"):
            inspect_window(self.report, after=13)
        with self.assertRaisesRegex(ValueError, "stopped after source step 31"):
            inspect_window(self.report, after=32)
        with self.assertRaisesRegex(ValueError, "between 1 and 256"):
            inspect_window(self.report, limit=257)
        halted = inspect_window(self.report, after=31)
        self.assertEqual((halted["trace"], halted["end"]["status"]), ([], 1))
        self.assertIsNone(inspect_window(self.report, limit=1)["seek_runner"])

    def test_trap_and_fuel_dont_retire(self):
        raw = wasmtime.wat2wasm("""(module
          (memory (export "memory") 1 1)
          (func (export "run") (param i32) (result i32)
            local.get 0 i32.load))""")
        replay = Replay(bytes(raw))
        for fuel, status in ((3, 3), (1, 2)):
            bundle = make_bundle(replay, 65533, bytes(65536), fuel, 0)
            result = inspect_window(bundle, after=1, limit=1)
            self.assertEqual(result["end"]["status"], status)
            [row] = result["trace"]
            self.assertEqual((row["retired_before"], row["retired_after"]), (1, 1))


if __name__ == "__main__":
    unittest.main()
