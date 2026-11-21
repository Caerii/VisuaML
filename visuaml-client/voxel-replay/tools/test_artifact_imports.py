"""Exercise exported-model diagnostics and unchanged Rust WASM admission."""
import json
from pathlib import Path
import unittest

import torch
import wasmtime

from visuaml.replay import PAGE_BYTES, Replay, capture, check_report
from visuaml.replay.artifacts import compare_exports, describe_export, describe_onnx, describe_wasm

EXAMPLES = Path(__file__).resolve().parents[1] / "release/examples/imports"


class ArtifactImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_compiler_generated_active_data(self):
        raw = (EXAMPLES / "static-material.wasm").read_bytes()
        report = json.loads((EXAMPLES / "static-material-report.json").read_text())
        diagnostics = describe_wasm(raw)
        self.assertTrue(diagnostics["replayable"])
        self.assertIn("producers", diagnostics["custom_sections"])
        self.assertEqual(diagnostics["active_data_bytes"], 4)
        replay = Replay(raw)
        self.assertEqual(capture(replay, 35, None, 5000, 2), report["bundle"])
        self.assertEqual(check_report(report)["returned"], 42)
        self.assertEqual(replay.summary(replay.run(replay.restore(report["bundle"]["checkpoint"]),
                                                  runner="reference"))["returned"], 42)
        supplied_page = capture(replay, 35, bytes(PAGE_BYTES), 5000, 2)
        self.assertEqual(replay.finish(supplied_page["checkpoint"])["returned"], 35)

    def test_data_order_and_admission_boundaries(self):
        template = """(module (memory (export "memory") 1 1)
            (func (export "run") (param i32) (result i32) local.get 0 i32.load)
            %s)"""
        overlapping = wasmtime.wat2wasm(template % '(data (i32.const 1024) "abcd") (data (i32.const 1025) "e")')
        replay = Replay(bytes(overlapping))
        result = replay.run(replay.start(1024, None, 100))
        self.assertEqual(int(result.vm.stack[0]), int.from_bytes(b"aecd", "little"))
        for segment in ('(data (i32.const 65536) "x")', '(data "deferred")'):
            diagnostics = describe_wasm(bytes(wasmtime.wat2wasm(template % segment)))
            self.assertTrue(diagnostics["valid"])
            self.assertFalse(diagnostics["replayable"])

    def test_export_activations(self):
        class Masked(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.register_buffer("mask", torch.tensor(value))

            def forward(self, input):
                return (input + self.mask).relu()

        data = (torch.tensor([3.0]),)
        left = torch.export.export(Masked(-2.), data)
        same = torch.export.export(Masked(-2.), data)
        right = torch.export.export(Masked(0.), data)
        self.assertEqual(describe_export(left)["input_bindings"][0]["kind"], "BUFFER")
        self.assertIsNone(compare_exports(left, same, data)["first_difference"])
        comparison = compare_exports(left, right, data)
        self.assertEqual(comparison["first_difference"]["node"], "mask")
        self.assertEqual(comparison["first_activation_difference"]["operator"], "aten.add.Tensor")
        self.assertIsNone(compare_exports(left, right, data, atol=2)["first_difference"])

    def test_onnx_metadata(self):
        try:
            from onnx import TensorProto, helper
        except ImportError:
            self.skipTest("ONNX extra is not installed")
        node = helper.make_node("Mystery", ["x"], ["y"], domain="example.custom")
        graph = helper.make_graph([node], "opaque", [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N"])],
                                  [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N"])])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("example.custom", 1)])
        report = describe_onnx(model.SerializeToString())
        self.assertEqual(report["inputs"][0]["shape"], ["N"])
        self.assertEqual(report["nodes"][0]["operator"], "Mystery")
        self.assertIsNone(report["execution"])


if __name__ == "__main__":
    unittest.main()
