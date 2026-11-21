"""Verify streaming comparison, Rust admission, and portable boundary reports."""
from io import StringIO
import json
from pathlib import Path
import unittest

import numpy as np
import torch
import wasmtime

from visuaml.replay import Replay, capture
from visuaml.replay.artifacts import describe_wasm
from visuaml.replay.pipeline import compare_observations, observation, read_observations, tensor_identity

ROOT = Path(__file__).resolve().parents[1]


class PipelineReplayTests(unittest.TestCase):
    def test_order_and_early_stop(self):
        first = observation("req", "output", 4, runtime="a")
        other = observation("req", "output", 5, runtime="b")

        def stream():
            yield first
            self.fail("Comparison should stop at the first difference")

        result = compare_observations(stream(), [other])
        self.assertEqual(result["first_difference"]["reason"], "value")
        self.assertEqual(compare_observations([first], [])["first_difference"]["reason"], "missing-observation")
        self.assertEqual(compare_observations([first], [dict(other, request="another")])[
            "first_difference"]["reason"], "boundary")

    def test_tolerances_and_tensor_identity(self):
        left = observation("req", "logits", [1.0, 1], runtime="a")
        right = observation("req", "logits", [1.1, 1], runtime="b")
        rules = {"logits": (0, .125)}
        self.assertIsNone(compare_observations([left], [right], tolerances=rules)["first_difference"])
        right["value"][1] = 2
        self.assertIsNotNone(compare_observations([left], [right], tolerances={"logits": (0, 2)})["first_difference"])
        with self.assertRaisesRegex(ValueError, "4096"):
            observation("req", "logits", "x"*4097, runtime="a")
        with self.assertRaisesRegex(ValueError, "line 2"):
            list(read_observations(StringIO(json.dumps(left)+"\n[]\n")))
        array = np.arange(6, dtype=np.int32).reshape(2, 3)
        self.assertEqual(tensor_identity(array), tensor_identity(np.asfortranarray(array)))
        self.assertNotEqual(tensor_identity(array), tensor_identity(array.reshape(3, 2)))

    def test_unchanged_rust_tool_and_explicit_global_rejection(self):
        torch.set_num_threads(1)
        tool = (ROOT / "release/examples/pipeline/tool.wasm").read_bytes()
        facts = describe_wasm(tool)
        self.assertEqual(facts["linker_globals"], {"__data_end": 1024, "__heap_base": 1024})
        replay = Replay(tool)
        bundle = capture(replay, 4, None, 100, 1)
        self.assertEqual(replay.finish(bundle["checkpoint"])["returned"], 7)
        raw = bytes(wasmtime.wat2wasm("""(module
            (memory (export "memory") 1 1)
            (global (export "__heap_base") i32 (i32.const 1))
            (func (export "run") (param i32) (result i32) global.get 0))"""))
        rejected = describe_wasm(raw)
        self.assertTrue(rejected["valid"])
        self.assertFalse(rejected["replayable"])
        self.assertIn("Function 0", rejected["reason"])
        self.assertIn("global.get", rejected["reason"])


if __name__ == "__main__":
    unittest.main()
