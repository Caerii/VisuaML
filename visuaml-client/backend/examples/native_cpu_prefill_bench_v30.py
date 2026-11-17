"""v30: parity-gated independent native CPU prefill-versus-decode benchmark.

Compare opt-in batched prefill against v29's AVX2 and independent scalar
reference. All correctness checks run before timing, including each chunk of
logits and final KV state. These are bounded warm single-request latencies,
not sustained throughput, large-model serving, energy or memory measurements.
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
    model: LlamaLocalCheckpoint, *, prompt: int = 8,
    decode: tuple[int, ...] = (1, 1), batch: int = 1,
    trials: int = 7, seed: int = 727,
) -> dict:
    if (
        type(prompt) is not int or not 1 <= prompt <= 32
        or type(batch) is not int or not 1 <= batch <= 2
        or type(trials) is not int or not 2 <= trials <= 100
        or not decode
        or any(type(n) is not int or not 1 <= n <= 32 for n in decode)
        or prompt + sum(decode) > min(64, model.profile.context)
    ):
        raise ValueError("Invalid bounded v30 prefill/decode benchmark workload")
    plan = lower_llama_semantic_program(
        make_llama_semantic_program(model, batch=batch),
        model, batch=batch,
    )
    compilers = {
        "native_scalar": NativeCpuLlamaCompiler(
            plan, reference_kernels=True,
        ),
        "native_avx2": NativeCpuLlamaCompiler(
            plan, enable_avx2=True,
        ),
        "native_batched": NativeCpuLlamaCompiler(
            plan, enable_avx2=True, enable_batched_prefill=True,
        ),
    }
    engines = {name: compiler.new_session() for name, compiler in compilers.items()}
    oracle = plan.new_session(backend="numpy")
    rng = np.random.default_rng(seed)
    chunks = (prompt, *decode)
    tokens = rng.integers(
        0, model.profile.vocabulary,
        size=(batch, sum(chunks)), dtype=np.int64,
    )

    def run(engine, *, timed: bool) -> tuple[list[np.ndarray], list[float]]:
        engine.reset()
        cursor = 0
        outputs = []
        elapsed = []
        for size in chunks:
            piece = tokens[:, cursor:cursor + size]
            start_ns = time.perf_counter_ns() if timed else 0
            outputs.append(engine.step(piece))
            if timed:
                elapsed.append((time.perf_counter_ns() - start_ns) / 1e6)
            cursor += size
        return outputs, elapsed

    expected, _ = run(oracle, timed=False)
    expected_state = oracle.snapshot()
    for name, engine in engines.items():
        outputs, _ = run(engine, timed=False)
        for actual, target in zip(outputs, expected):
            np.testing.assert_allclose(
                actual, target, atol=3e-4, rtol=3e-4,
                err_msg=f"Native {name} logit parity",
            )
        for actual, target in zip(
            engine.snapshot().layers, expected_state.layers
        ):
            np.testing.assert_allclose(
                actual.keys, target.keys, atol=3e-4, rtol=3e-4,
                err_msg=f"Native {name} key parity",
            )
            np.testing.assert_allclose(
                actual.values, target.values, atol=3e-4, rtol=3e-4,
                err_msg=f"Native {name} value parity",
            )

    timings = {
        key: {"prefill_ms": [], "decode_ms": [], "total_ms": []}
        for key in engines
    }
    for trial in range(trials):
        names = list(engines)
        if trial % 2:
            names.reverse()
        for name in names:
            _, durations = run(engines[name], timed=True)
            prefill_ms = durations[0]
            decode_ms = sum(durations[1:])
            timings[name]["prefill_ms"].append(prefill_ms)
            timings[name]["decode_ms"].append(decode_ms)
            timings[name]["total_ms"].append(prefill_ms + decode_ms)
    root = Path(__file__).resolve().parents[1]
    files = (
        "examples/native_cpu_prefill_bench_v30.py",
        "visuaml/compiler_lab/native_cpu_v28.cpp",
        "visuaml/compiler_lab/native_cpu_v28.py",
    )
    def speed_ratio(before: str, after: str, stage: str) -> float:
        baseline = statistics.median(timings[before][stage])
        optimized = statistics.median(timings[after][stage])
        return round(baseline / optimized, 6)

    return {
        "schema": "visuaml.native-prefill-benchmark.v30",
        "model_revision": model.declared_revision,
        "checkpoint_sha256": model.checkpoint_sha256,
        "semantic_sha256": plan.semantic_sha256,
        "prompt_tokens": prompt,
        "decode_chunks": list(decode),
        "batch": batch,
        "trials": trials,
        "seed": seed,
        "timing": {
            name: {
                stage: {
                    **summarize(samples),
                    "raw_samples_ms": [round(value, 6) for value in samples],
                }
                for stage, samples in stages.items()
            }
            for name, stages in timings.items()
        },
        "prefill_ratio_v29_avx2_over_batched": speed_ratio(
            "native_avx2", "native_batched", "prefill_ms",
        ),
        "decode_ratio_v29_avx2_over_batched": speed_ratio(
            "native_avx2", "native_batched", "decode_ms",
        ),
        "compiler_evidence": {
            name: {
                key: compiler.report()[key]
                for key in (
                    "compiler_identity", "target", "cpp_source_sha256",
                    "executable_sha256", "compilation_ms", "linear_kernel",
                    "avx2_requested", "avx2_active", "batched_prefill_requested",
                    "kernel_compile_flags",
                )
            }
            for name, compiler in compilers.items()
        },
        "environment": {
            "python": platform.python_version(),
            "cpu": platform.processor() or platform.machine(),
            "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
            "platform": platform.platform(),
        },
        "source_sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in files
        },
        "numerical_certificate": (
            "all prefill and decode logits and final-layer KV "
            "match independent NumPy semantic execution"
        ),
        "limitations": [
            "Small bounded Llama checkpoint and CPU, not real serving throughput.",
            "Warm model execution excludes native compilation and model loading.",
            "The batched kernel is opt-in and may be slower than AVX2 on some shapes.",
            "Shared CI runners cannot establish dedicated-hardware regressions.",
            "No GPU, photonic device, physical energy, or allocator profiling.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--prompt", type=int, default=8)
    parser.add_argument("--decode", nargs="+", type=int, default=[1, 1])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model = load_hf_llama_local(
        args.checkpoint, declared_revision=args.revision
    )
    record = benchmark(
        model, prompt=args.prompt, decode=tuple(args.decode),
        batch=args.batch, trials=args.trials,
    )
    output = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
