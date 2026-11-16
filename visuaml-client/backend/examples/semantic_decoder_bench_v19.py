"""CPU-only matched benchmark of v16 and graph-driven v18 decoder execution."""

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
import torch
from visuaml.compiler_lab.memory_planner_v19 import plan_decoder_memory
from visuaml.compiler_lab.semantic_decoder_v18 import (
    lower_decoder_semantic_program,
    make_decoder_semantic_program,
)
from visuaml.compiler_lab.stacked_decoder_v14 import StackedWeights
from visuaml.compiler_lab.streaming_decoder_v16 import StreamingStackedDecoder


def summary(samples: list[float]) -> dict:
    median = statistics.median(samples)
    return {
        "median_ms": round(median, 6),
        "p95_ms": round(float(np.percentile(samples, 95)), 6),
        "minimum_ms": round(min(samples), 6),
        "mad_ms": round(statistics.median(abs(x - median) for x in samples), 6),
    }


def bench(*, layers: int, prompt: int, steps: int, trials: int, seed: int) -> dict:
    if (
        type(layers) is not int or not 1 <= layers <= 4
        or type(prompt) is not int or not 1 <= prompt <= 32
        or type(steps) is not int or not 1 <= steps <= 24
        or type(trials) is not int or not 2 <= trials <= 200
        or prompt + steps > 64
    ):
        raise ValueError("Synthetic CPU decoder benchmark outside admitted bounds")
    weights = StackedWeights.seeded(seed, layers=layers)
    original = StreamingStackedDecoder(weights)
    start = time.perf_counter_ns()
    program = make_decoder_semantic_program(weights)
    plan = lower_decoder_semantic_program(program, weights)
    interpreted = plan.new_session()
    construction_ms = (time.perf_counter_ns() - start) / 1e6
    tokens = (np.arange(prompt, dtype=np.int64)[None, :] + 1) % original.vocab
    left, old_state = original.generate(tokens, steps=steps)
    right, new_state = interpreted.generate(tokens, steps=steps)
    np.testing.assert_array_equal(left, right)
    for a, b in zip(old_state.layers, new_state.layers):
        np.testing.assert_allclose(a.keys, b.keys, atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(a.values, b.values, atol=3e-5, rtol=3e-5)
    results = {"v16_handwritten": [], "v18_semantic_cpu": []}
    choices = (("v16_handwritten", original), ("v18_semantic_cpu", interpreted))
    for index in range(trials):
        for name, backend in (choices if index % 2 == 0 else reversed(choices)):
            started = time.perf_counter_ns()
            backend.generate(tokens, steps=steps)
            results[name].append((time.perf_counter_ns() - started) / 1e6)
    base = statistics.median(results["v16_handwritten"])
    candidate = statistics.median(results["v18_semantic_cpu"])
    return {
        "layers": layers, "prompt_tokens": prompt, "new_tokens": steps,
        "trials": trials,
        "construction_ms": round(construction_ms, 6),
        "v16_handwritten": summary(results["v16_handwritten"]),
        "v18_semantic_cpu": summary(results["v18_semantic_cpu"]),
        "relative_v16_over_v18": round(base / candidate, 5),
        "semantic_sha256": plan.digest,
        "checkpoint_sha256": plan.checkpoint_sha256,
        "parity": "generated token IDs and every layer final KV pair",
        "logical_memory": plan_decoder_memory(program).as_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--prompt", type=int, default=4)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sources = (
        "examples/semantic_decoder_bench_v19.py",
        "visuaml/compiler_lab/semantic_decoder_v18.py",
        "visuaml/compiler_lab/memory_planner_v19.py",
    )
    result = {
        "schema": "visuaml.semantic-cpu-benchmark.v19",
        "scope": "tiny synthetic CPU decoder, NOT general pretrained LLM performance",
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "cpu": platform.processor() or platform.machine(),
        "openblas_threads": os.getenv("OPENBLAS_NUM_THREADS", "unset"),
        "sources_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in sources
        },
        "results": [
            bench(layers=n, prompt=args.prompt, steps=args.steps, trials=args.trials, seed=args.seed)
            for n in args.layers
        ],
        "limitations": [
            "No native AOT, GPU, optical device or trained pretrained checkpoint.",
            "Warm-run timings include Python dispatch and exclude first-time construction.",
            "CPU, NumPy, BLAS, environment and thread settings affect latency.",
        ],
    }
    wire = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(wire)
    print(wire)


if __name__ == "__main__":
    main()
