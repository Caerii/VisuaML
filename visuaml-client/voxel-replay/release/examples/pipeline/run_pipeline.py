"""Produce three independently reproducible model/tool/handoff failures."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch

from visuaml.replay import Replay, capture, seal_report
from visuaml.replay.artifacts import ONNXRunner, compare_onnx
from visuaml.replay.checkpoint import canonical
from visuaml.replay.pipeline import compare_observations, observation


class Controller(torch.nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.register_buffer("delta", torch.tensor(delta, dtype=torch.int32))

    def forward(self, value, feedback):
        request = value + self.delta
        decision = (feedback >= 7).to(torch.int32)
        return request, decision


def stage(path, checkpoint_dir, *, changed_tool=False, changed_feedback=False):
    value, zero = [np.asarray(x, dtype=np.int32) for x in (4, 0)]
    outputs = ["argument", "decision"]
    controller = ONNXRunner(path)
    request = controller.run({"value": value, "feedback": zero}, outputs)
    argument = int(request["argument"])
    name = "changed-tool" if changed_tool else "tool"
    tool = Path(__file__).with_name(f"{name}.wasm").read_bytes()
    runtime = Replay(tool)
    bundle = capture(runtime, argument, None, 100, 1)
    finished = runtime.finish(bundle["checkpoint"])
    returned = finished["returned"]
    reply = 0 if changed_feedback else returned
    decision = controller.run({"value": value, "feedback": np.asarray(reply, dtype=np.int32)}, outputs)
    journal = [
        observation("case-1", "tool_request", {"tool": "add", "argument": argument}, runtime="onnxruntime"),
        observation("case-1", "tool_result", {"status": finished["status"], "returned": returned}, runtime="visuaml-wasm"),
        observation("case-1", "feedback", reply, runtime="host"),
        observation("case-1", "model_decision", int(decision["decision"]), runtime="onnxruntime"),
    ]
    recipe = {"kind": "expect-return", "value": (argument+3) & 0xffffffff}
    report = seal_report(bundle, recipe)
    checkpoint_dir.write_bytes(canonical(report)+b"\n")
    return journal, dict(model_sha256=sha256(path.read_bytes()).hexdigest(),
                         wasm_sha256=sha256(tool).hexdigest(), report_bytes=checkpoint_dir.stat().st_size,
                         report_sha256=sha256(checkpoint_dir.read_bytes()).hexdigest())


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    out = args.directory
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    value, feedback = [torch.tensor(x, dtype=torch.int32) for x in (4, 7)]
    for name, delta in (("reference", 0), ("model", 1)):
        model = Controller(delta).eval()
        args_ = (value, feedback)
        torch.export.save(torch.export.export(model, args_), out / f"{name}.pt2")
        torch.onnx.export(model, args_, out / f"{name}.onnx", dynamo=False,
                          input_names=["value", "feedback"], output_names=["argument", "decision"], opset_version=17)
    bindings = dict(inputs=["value", "feedback"], outputs=dict(tool_request="argument", model_decision="decision"))
    (out/"bindings.json").write_text(json.dumps(bindings)+"\n")
    arrays = [value.numpy(), feedback.numpy()]
    for name, array in zip(bindings["inputs"], arrays, strict=True):
        np.save(out/f"{name}.npy", array)
    exported = torch.export.load(out/"reference.pt2")
    cross = {name: compare_onnx(exported, out/f"{name}.onnx", bindings, arrays) for name in ("reference", "model")}
    assert cross["reference"]["first_difference"] is None
    assert cross["model"]["first_difference"]["boundary"] == "tool_request"
    result = dict(cross_runtime=cross, cases={})
    for name, options in (
        ("reference", {}), ("model", {}), ("guest", {"changed_tool": True}),
        ("handoff", {"changed_feedback": True}),
    ):
        native = "model" if name == "model" else "reference"
        journal, identity = stage(out/f"{native}.onnx", out/f"{name}-tool-report.json", **options)
        (out/f"{name}.jsonl").write_bytes(b"".join(canonical(row)+b"\n" for row in journal))
        result["cases"][name] = dict(journal=journal, **identity)
    reference = result["cases"]["reference"]["journal"]
    timings = {}
    for name, expected in (("model", "tool_request"), ("guest", "tool_result"), ("handoff", "feedback")):
        candidate = result["cases"][name]["journal"]
        started = perf_counter_ns()
        comparison = compare_observations(reference, candidate)
        timings[name] = perf_counter_ns()-started
        assert comparison["first_difference"]["reference"]["boundary"] == expected
        result["cases"][name]["comparison"] = comparison
    (out/"results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(first_differences={name: result["cases"][name]["comparison"]["first_difference"]
                                             for name in timings}, comparison_ns=timings), indent=2))


if __name__ == "__main__":
    main()
