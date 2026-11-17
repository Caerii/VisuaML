"""Matched CPU whole-decoder benchmark for independent C++17 and IR references.

All timings are measured only *after* numerical parity checks against the
canonical NumPy semantic interpreter and every final KV state. Warm execution
excludes compilation and first-time graph tracing; they are reported separately.
No state-of-the-art, CUDA, optical or serving-throughput claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
from visuaml.compiler_lab.hf_llama_checkpoint_v19 import (
    LlamaLocalCheckpoint,
    load_hf_llama_local,
)
from visuaml.compiler_lab.llama_semantic_v20 import (
    lower_llama_semantic_program,
    make_llama_semantic_program,
)
from visuaml.compiler_lab.native_cpu_v28 import NativeCpuLlamaCompiler


def summarize(samples: list[float]) -> dict:
    median = statistics.median(samples)
    return {
        "median_ms": round(median, 6),
        "p95_ms": round(float(np.percentile(samples, 95)), 6),
        "minimum_ms": round(min(samples), 6),
        "mad_ms": round(
            statistics.median(abs(value - median) for value in samples), 6
        ),
    }


def benchmark(
    model: LlamaLocalCheckpoint, *, prompt: int = 4,
    chunk_sizes: tuple[int, ...] = (1, 1, 2), trials: int = 7,
    seed: int = 101, compare_native_reference: bool = False,
    compare_simd: bool = False,
) -> dict:
    if (
        type(prompt) is not int or not 1 <= prompt <= 32
        or type(trials) is not int or not 2 <= trials <= 100
        or not chunk_sizes
        or any(type(n) is not int or not 1 <= n <= 32 for n in chunk_sizes)
        or prompt + sum(chunk_sizes) > min(model.profile.context, 64)
    ):
        raise ValueError("Invalid bounded CPU native benchmark workload")
    plan = lower_llama_semantic_program(
        make_llama_semantic_program(model), model
    )
    native = NativeCpuLlamaCompiler(plan)
    engines = {
        "numpy_semantic": plan.new_session(backend="numpy"),
        "torchscript_whole": plan.new_session(backend="torchscript_poly"),
        "native_cpp": native.new_session(),
    }
    native_reference = None
    if compare_native_reference:
        native_reference = NativeCpuLlamaCompiler(plan, reference_kernels=True)
        engines["native_scalar_reference"] = native_reference.new_session()
    native_simd = None
    if compare_simd:
        native_simd = NativeCpuLlamaCompiler(plan, enable_avx2=True)
        engines["native_avx2"] = native_simd.new_session()
    rng = np.random.default_rng(seed)
    sample = rng.integers(
        0, model.profile.vocabulary,
        size=(1, prompt + sum(chunk_sizes)), dtype=np.int64,
    )
    chunks = (prompt, *chunk_sizes)

    def run(engine) -> tuple[np.ndarray, ...]:
        engine.reset()
        outputs = []
        offset = 0
        for length in chunks:
            outputs.append(engine.step(sample[:, offset:offset + length]))
            offset += length
        return tuple(outputs)

    expected = run(engines["numpy_semantic"])
    reference_kv = engines["numpy_semantic"].snapshot()
    for name, engine in engines.items():
        if name == "numpy_semantic":
            continue
        observed = run(engine)
        for actual, target in zip(observed, expected):
            np.testing.assert_allclose(
                actual, target, atol=3e-4, rtol=3e-4
            )
        for actual, target in zip(
            engine.snapshot().layers, reference_kv.layers
        ):
            np.testing.assert_allclose(
                actual.keys, target.keys, atol=3e-4, rtol=3e-4
            )
            np.testing.assert_allclose(
                actual.values, target.values, atol=3e-4, rtol=3e-4
            )

    timings = {name: [] for name in engines}
    for index in range(trials):
        items = list(engines.items())
        if index % 2:
            items.reverse()
        for name, engine in items:
            before = time.perf_counter_ns()
            run(engine)
            timings[name].append((time.perf_counter_ns() - before) / 1e6)
    root = Path(__file__).resolve().parents[1]
    files = (
        "examples/native_cpu_bench_v28.py",
        "visuaml/compiler_lab/native_cpu_v28.cpp",
        "visuaml/compiler_lab/native_cpu_v28.py",
    )
    return {
        "schema": "visuaml.native-cpu-benchmark.v28",
        "source_model_revision": model.declared_revision,
        "checkpoint_sha256": model.checkpoint_sha256,
        "semantic_sha256": plan.semantic_sha256,
        "chunks": list(chunks),
        "trials": trials,
        "latency": {key: summarize(value) for key, value in timings.items()},
        # Preserve individual samples for matched-device replication and
        # confidence intervals; aggregate statistics alone hide instability.
        "raw_samples_ms": {
            key: [round(sample, 6) for sample in values]
            for key, values in timings.items()
        },
        "timing_order": "forward on even trials; reverse on odd trials",
        "compilation": {
            "native_cpp_ms": native.report()["compilation_ms"],
            "native_scalar_reference_ms": (
                native_reference.report()["compilation_ms"]
                if native_reference is not None else None
            ),
            "native_avx2_ms": (
                native_simd.report()["compilation_ms"]
                if native_simd is not None else None
            ),
            "torchscript_traces": engines["torchscript_whole"].report()["traces"],
        },
        "numerical_certificate": "all chunked logits and final layer KV match NumPy IR",
        "native_linear_variants": {
            "native_cpp": native.report()["linear_kernel"],
            **(
                {"native_scalar_reference": native_reference.report()["linear_kernel"]}
                if native_reference is not None else {}
            ),
            **(
                {"native_avx2": native_simd.report()["linear_kernel"]}
                if native_simd is not None else {}
            ),
        },
        "warm_native_speedup_vs_scalar": (
            round(
                statistics.median(timings["native_scalar_reference"])
                / statistics.median(timings["native_cpp"]), 6
            ) if native_reference is not None else None
        ),
        "warm_simd_speedup_vs_four_row": (
            round(
                statistics.median(timings["native_cpp"])
                / statistics.median(timings["native_avx2"]), 6
            ) if native_simd is not None else None
        ),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu": platform.processor() or platform.machine(),
        "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
        "source_sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in files
        },
        "limitations": [
            "Scalar C++17 research CPU backend; not a production-optimized native kernel.",
            "These short CPU measurements cannot establish industry-scale throughput.",
            "Warm inference excludes initial compilation; compile cost is separate.",
            "No GPU, optical device, parallel serving or physical energy measurement.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--prompt", type=int, default=4)
    parser.add_argument("--chunks", nargs="+", type=int, default=[1, 1, 2])
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument(
        "--compare-native-reference", action="store_true",
        help="Compile scalar and four-row native binaries for matched comparison",
    )
    parser.add_argument(
        "--compare-simd", action="store_true",
        help="Compile a guarded AVX2 binary, retaining four-row fallback",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model = load_hf_llama_local(
        args.checkpoint, declared_revision=args.revision
    )
    record = benchmark(
        model, prompt=args.prompt,
        chunk_sizes=tuple(args.chunks), trials=args.trials,
        compare_native_reference=args.compare_native_reference,
        compare_simd=args.compare_simd,
    )
    wire = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(wire, encoding="utf-8")
    print(wire)


if __name__ == "__main__":
    main()
