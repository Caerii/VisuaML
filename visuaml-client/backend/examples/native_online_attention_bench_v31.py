"""v31: source-bound, parity-gated streaming attention context sweep.

Both variants execute the same pinned Llama semantic plan and token sequence.
Every prefill chunk, decode logit and final KV state must match the independent
NumPy oracle BEFORE a warm timing is recorded. No claims of production-scale
context, throughput, GPU performance, energy, or physical memory reductions.
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


def summary(samples: list[float]) -> dict:
    median = statistics.median(samples)
    return {
        "median_ms": round(median, 6),
        "p95_ms": round(float(np.percentile(samples, 95)), 6),
        "mad_ms": round(
            statistics.median(abs(value - median) for value in samples), 6
        ),
        "raw_samples_ms": [round(x, 6) for x in samples],
    }


def benchmark(
    model: LlamaLocalCheckpoint, *,
    prefixes: tuple[int, ...] = (4, 8, 16, 32, 48),
    trials: int = 7, seed: int = 311,
) -> dict:
    ceiling = min(model.profile.context, 64)
    if (
        type(trials) is not int or not 2 <= trials <= 100
        or not prefixes or len(set(prefixes)) != len(prefixes)
        or any(type(x) is not int or not 1 <= x < ceiling for x in prefixes)
    ):
        raise ValueError("Invalid bounded attention context sweep")
    plan = lower_llama_semantic_program(
        make_llama_semantic_program(model), model
    )
    compilers = {
        "materialized": NativeCpuLlamaCompiler(
            plan, enable_avx2=True,
        ),
        "online": NativeCpuLlamaCompiler(
            plan, enable_avx2=True, enable_online_attention=True,
        ),
    }
    engines = {
        key: compiler.new_session() for key, compiler in compilers.items()
    }
    oracle = plan.new_session(backend="numpy")
    rng = np.random.default_rng(seed)
    tokens = rng.integers(
        0, model.profile.vocabulary, size=(1, max(prefixes) + 1),
        dtype=np.int64,
    )

    def chunks(prefix: int) -> tuple[np.ndarray, ...]:
        # Each native call must remain in the admitted query<=32 contract.
        segments = []
        cursor = 0
        while cursor < prefix:
            stop = min(prefix, cursor + 32)
            segments.append(tokens[:, cursor:stop])
            cursor = stop
        return (*segments, tokens[:, prefix:prefix + 1])

    def execute(engine, pieces: tuple[np.ndarray, ...], *, timed: bool):
        engine.reset()
        outputs = []
        times = []
        for piece in pieces:
            started = time.perf_counter_ns() if timed else 0
            outputs.append(engine.step(piece))
            if timed:
                times.append((time.perf_counter_ns() - started) / 1e6)
        return outputs, times

    cases = []
    for prefix in prefixes:
        pieces = chunks(prefix)
        expected, _ = execute(oracle, pieces, timed=False)
        oracle_state = oracle.snapshot()
        for name, engine in engines.items():
            observed, _ = execute(engine, pieces, timed=False)
            for a, b in zip(observed, expected):
                np.testing.assert_allclose(
                    a, b, atol=3e-4, rtol=3e-4,
                    err_msg=f"{name} logits at prefix {prefix}",
                )
            for a, b in zip(engine.snapshot().layers, oracle_state.layers):
                np.testing.assert_allclose(
                    a.keys, b.keys, atol=3e-4, rtol=3e-4,
                    err_msg=f"{name} keys at prefix {prefix}",
                )
                np.testing.assert_allclose(
                    a.values, b.values, atol=3e-4, rtol=3e-4,
                    err_msg=f"{name} values at prefix {prefix}",
                )
        readings = {
            name: {"prefix_ms": [], "decode_ms": []} for name in engines
        }
        for trial in range(trials):
            names = list(engines)
            if trial % 2:
                names.reverse()
            for name in names:
                _, elapsed = execute(engines[name], pieces, timed=True)
                readings[name]["prefix_ms"].append(sum(elapsed[:-1]))
                readings[name]["decode_ms"].append(elapsed[-1])
        results = {
            name: {key: summary(values) for key, values in phases.items()}
            for name, phases in readings.items()
        }
        cases.append({
            "past_tokens": prefix,
            "query_tokens": 1,
            "prefill_chunks": [piece.shape[1] for piece in pieces[:-1]],
            "timing": results,
            "online_vs_materialized_decode_ratio": round(
                results["materialized"]["decode_ms"]["median_ms"]
                / results["online"]["decode_ms"]["median_ms"], 6,
            ),
        })
    root = Path(__file__).resolve().parents[1]
    source_paths = (
        "examples/native_online_attention_bench_v31.py",
        "visuaml/compiler_lab/native_cpu_v28.cpp",
        "visuaml/compiler_lab/native_cpu_v28.py",
    )
    return {
        "schema": "visuaml.native-online-attention-benchmark.v31",
        "source_model_revision": model.declared_revision,
        "checkpoint_sha256": model.checkpoint_sha256,
        "semantic_sha256": plan.semantic_sha256,
        "seed": seed,
        "trials": trials,
        "cases": cases,
        "compilers": {
            name: {
                key: compiler.report()[key]
                for key in (
                    "cpp_source_sha256", "executable_sha256",
                    "compiler_identity", "target", "compilation_ms",
                    "attention_kernel", "kernel_compile_flags",
                    "avx2_active",
                )
            }
            for name, compiler in compilers.items()
        },
        "source_sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in source_paths
        },
        "environment": {
            "cpu": platform.processor() or platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
        },
        "certificate": (
            "Every prefill chunk, single-token decode logit and final KV state "
            "matched the independent NumPy semantic reference at 3e-4."
        ),
        "limitations": [
            "Admitted context <=64 here, not a long-context serving benchmark.",
            "Timing includes Python step and KV commit overhead.",
            "Warm execution excludes native compilation and model loading.",
            "Shared CI runner timing cannot establish production throughput.",
            "Online normalization changes floating-point reduction order.",
            "No measured energy, native allocator peak or physical memory win.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--past", nargs="+", type=int, default=[4, 8, 16, 32, 48])
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model = load_hf_llama_local(
        args.checkpoint, declared_revision=args.revision,
    )
    result = benchmark(model, prefixes=tuple(args.past), trials=args.trials)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
