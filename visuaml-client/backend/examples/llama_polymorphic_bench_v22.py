"""v22 matched CPU prefill/decode benchmark: specialize-on-past vs one graph.

All executions consume the same strictly admitted local Llama checkpoint and
the exact same token chunks. Correctness is compared against the independent
NumPy semantic interpreter, including every final KV layer. Warm costs exclude
tracing; cumulative trace cost and graph specialization count are reported
separately. Not a GPU/production serving or energy benchmark.
"""

from __future__ import annotations

import argparse
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


def _stats(samples: list[float]) -> dict:
    median = statistics.median(samples)
    return {
        "median_ms": round(median, 6),
        "p95_ms": round(float(np.percentile(samples, 95)), 6),
        "min_ms": round(min(samples), 6),
        "mad_ms": round(
            statistics.median(abs(v - median) for v in samples), 6
        ),
    }


def benchmark(
    weights: LlamaLocalCheckpoint,
    *, chunks: tuple[int, ...] = (4, 1, 1, 1, 3, 2),
    trials: int = 11,
    seed: int = 123,
) -> dict:
    if (
        not chunks
        or any(type(q) is not int or not 1 <= q <= 32 for q in chunks)
        or sum(chunks) > min(weights.profile.context, 64)
        or not 2 <= trials <= 100 or type(trials) is not int
        or type(seed) is not int or not 0 <= seed <= 100_000
    ):
        raise ValueError("Invalid benchmark chunking, trial count, or seed")
    graph = make_llama_semantic_program(weights)
    plan = lower_llama_semantic_program(graph, weights)
    rng = np.random.default_rng(seed)
    tokens = rng.integers(
        0, weights.profile.vocabulary, size=(1, sum(chunks)), dtype=np.int64
    )
    engines = {
        "numpy_semantic": plan.new_session(backend="numpy"),
        "torchscript_specialized_v21": plan.new_session(backend="torchscript_graph"),
        "torchscript_polymorphic_v22": plan.new_session(backend="torchscript_poly"),
    }

    def execute(engine) -> tuple[np.ndarray, ...]:
        engine.reset()
        offset = 0
        outputs = []
        for n in chunks:
            outputs.append(engine.step(tokens[:, offset:offset + n]))
            offset += n
        return tuple(outputs)

    expected = execute(engines["numpy_semantic"])
    final_reference = engines["numpy_semantic"].snapshot()
    for name, engine in engines.items():
        if name == "numpy_semantic":
            continue
        actual = execute(engine)
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, atol=5e-5, rtol=5e-5)
        for actual_cache, expected_cache in zip(
            engine.snapshot().layers, final_reference.layers
        ):
            np.testing.assert_allclose(
                actual_cache.keys, expected_cache.keys,
                atol=5e-5, rtol=5e-5,
            )
            np.testing.assert_allclose(
                actual_cache.values, expected_cache.values,
                atol=5e-5, rtol=5e-5,
            )
    times = {name: [] for name in engines}
    for trial in range(trials):
        choices = list(engines.items())
        if trial % 2:
            choices.reverse()
        for name, engine in choices:
            started = time.perf_counter_ns()
            execute(engine)
            times[name].append((time.perf_counter_ns() - started) / 1e6)
    traces = {
        name: engine.report()["traces"] for name, engine in engines.items()
        if name != "numpy_semantic"
    }
    return {
        "schema": "visuaml.llama-polymorphic-cache-benchmark.v22",
        "checkpoint_sha256": weights.checkpoint_sha256,
        "semantic_sha256": plan.semantic_sha256,
        "input_chunks": list(chunks),
        "final_tokens": sum(chunks),
        "trials": trials,
        "latency": {name: _stats(samples) for name, samples in times.items()},
        "compilation": {
            name: {
                "specializations": len(entries),
                "total_trace_ms": round(
                    sum(entry["trace_ms"] for entry in entries), 6
                ),
                "first_graph_nodes": entries[0]["graph_nodes"],
            }
            for name, entries in traces.items()
        },
        "resident_cache_reserved_bytes": {
            name: engine.report()["reserved_bytes"] for name, engine in engines.items()
        },
        "numerical_witness": "every chunk's logits and every final KV pair",
        "platform": platform.platform(),
        "numpy": np.__version__,
        "openblas_threads": os.getenv("OPENBLAS_NUM_THREADS", "unset"),
        "limitations": [
            "Warm software CPU pipeline; traces counted separately.",
            "Polymorphic version copies full-capacity KV inputs and checks outputs.",
            "No native standalone AOT, CUDA/Triton, or physical-device energy data.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--chunks", nargs="+", type=int, default=[4, 1, 1, 1, 3, 2])
    parser.add_argument("--trials", type=int, default=11)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model = load_hf_llama_local(args.checkpoint, declared_revision=args.revision)
    report = benchmark(model, chunks=tuple(args.chunks), trials=args.trials)
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output)


if __name__ == "__main__":
    main()
