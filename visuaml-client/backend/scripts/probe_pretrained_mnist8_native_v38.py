"""Fresh pinned pretrained ONNX -> SemanticProgram -> generated C++17 numerical replay.

No architecture-specific emitter. This experiment checks the exact model bytes,
executes the independently generated native binary and compares two independent
oracles over fixed inputs. It makes no model accuracy or speedup claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from scripts.audit_pretrained_onnx_v37 import audit_bytes
from scripts.probe_pretrained_mnist8_oracles_v37 import (
    ATOL,
    RTOL,
    numpy_onnx_interpreter,
    sha,
)
from visuaml.compiler_lab.native_semantic_codegen_v32 import NativeSemanticCompiler
from visuaml.compiler_lab.onnx_semantic_frontend_v38 import lower_checked_onnx

MODEL_SHA = "2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf"
MODEL_REVISION = "a19f9a8c2333de1df9b03f10f5739f468b699a1a"
SCHEMA = "visuaml.pretrained-onnx-native-semantic.v38"


def _samples() -> tuple[tuple[str, np.ndarray], ...]:
    rng = np.random.default_rng(37)
    random_data = rng.random((1, 1, 28, 28)).astype(np.float32)
    impulse = np.zeros_like(random_data)
    impulse[0, 0, 14, 14] = 1.0
    checkerboard = (np.indices((28, 28)).sum(axis=0) % 2).astype(np.float32)[None, None]
    return (
        ("zero", np.zeros_like(random_data)),
        ("ones", np.ones_like(random_data)),
        ("impulse", impulse),
        ("checkerboard", checkerboard),
        ("random_seed_37", random_data),
    )


def run(data: bytes, *, source_commit: str | None) -> dict:
    import onnx
    import onnxruntime as ort

    audit = audit_bytes(data, expected_sha256=MODEL_SHA, source_revision=MODEL_REVISION)
    binding = lower_checked_onnx(data, expected_sha256=MODEL_SHA, source_revision=MODEL_REVISION)
    compiler = NativeSemanticCompiler(binding.program)
    if compiler.plan.output_shape != (1, 10):
        raise AssertionError("Unexpected compiled ONNX output shape")
    if len(compiler.plan.instructions) != len(audit["nodes"]):
        raise AssertionError("Incomplete native lowering")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(data, sess_options=options, providers=["CPUExecutionProvider"])
    model = onnx.load_model_from_string(data)
    witnesses = []
    raw_warm_ms = []
    for name, sample in _samples():
        weights = binding.validated_inputs(sample)
        before = {key: sha(val) for key, val in weights.items()}
        native, receipt = compiler.run_with_receipt(**weights)
        runtime = session.run([audit["output"]["name"]], {binding.input_name: sample})[0]
        numpy_result = numpy_onnx_interpreter(model, sample)
        if native.shape != runtime.shape or native.shape != numpy_result.shape:
            raise AssertionError("Native/oracle shape disagreement")
        checks = {}
        for oracle_name, expected in (("onnxruntime", runtime), ("scalar_numpy", numpy_result)):
            delta = np.abs(native - expected)
            allowed = ATOL + RTOL * np.abs(expected)
            checks[oracle_name] = {
                "max_abs_error": float(np.max(delta)),
                "max_allowed_error": float(np.max(allowed)),
                "passed": bool(np.isfinite(native).all() and np.all(delta <= allowed)),
                "oracle_output_sha256": sha(expected),
            }
        after = {key: sha(val) for key, val in weights.items()}
        if before != after:
            raise AssertionError("Native code modified input or checkpoint weights")
        witnesses.append({
            "name": name, "input_sha256": sha(sample),
            "native_logits_sha256": sha(native), "numerics": checks,
            "input_immutability": before == after,
            "receipt": receipt,
        })
        start = time.perf_counter_ns()
        _ = compiler.run(**weights)
        raw_warm_ms.append((time.perf_counter_ns() - start) / 1e6)
    if not all(x["input_immutability"] and all(
        item["passed"] for item in x["numerics"].values()
    ) for x in witnesses):
        raise AssertionError("Independent pretrained native parity failed")
    sources = (
        "visuaml/compiler_lab/native_semantic_codegen_v32.py",
        "visuaml/compiler_lab/onnx_semantic_frontend_v38.py",
        "visuaml/compiler_lab/semantic_ir.py",
        "scripts/probe_pretrained_mnist8_native_v38.py",
    )
    root = Path(__file__).resolve().parents[1]
    return {
        "schema": SCHEMA, "checkpoint_sha256": MODEL_SHA,
        "source_revision": MODEL_REVISION,
        "source_commit": source_commit,
        "checkpoint_input": audit["input"], "checkpoint_output": audit["output"],
        "model_family": "ONNX Model Zoo trained MNIST-8 convolutional classifier",
        "native_execution": True, "all_passed": True,
        "atol": ATOL, "rtol": RTOL,
        "compiler": compiler.report(),
        "graph_operation_count": len(compiler.plan.instructions),
        "tensor_sha256": binding.tensor_sha256,
        "generated_cpp": compiler.source,
        "source_file_sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in sources
        },
        "witnesses": witnesses,
        "warm_single_run_ms": raw_warm_ms,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "limitations": (
            "One pinned trained small CNN, static float32, local C++17 and five finite "
            "input witnesses; no all-input proof, real-image accuracy, GPU, WebAssembly, "
            "program decompilation, energy measurements, or production throughput."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = run(args.model.read_bytes(),
                 source_commit=os.environ.get("VISUAML_SOURCE_COMMIT"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "status": "passed", "native_execution": True,
        "generated_cpp_sha256": record["compiler"]["generated_cpp_sha256"],
        "native_binary_sha256": record["compiler"]["native_binary_sha256"],
        "max_abs_error_onnxruntime": max(
            x["numerics"]["onnxruntime"]["max_abs_error"] for x in record["witnesses"]
        ),
        "max_abs_error_numpy": max(
            x["numerics"]["scalar_numpy"]["max_abs_error"] for x in record["witnesses"]
        ),
        "witness_count": len(record["witnesses"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
