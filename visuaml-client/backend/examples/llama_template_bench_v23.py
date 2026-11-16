"""Matched v23 multi-session cold-start benchmark for one admitted Llama model.

Each strategy constructs and executes the same number of private KV sessions.
Control: trace a new polymorphic graph in every session.
Template: trace and verify one graph, then reuse it for all sessions.
Independent Llama NumPy logits are checked before accepting measurements.
The result is CPU/process-local, not online serving, GPU or persistent AOT.
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

from visuaml.compiler_lab.compiled_template_v23 import CompiledLlamaTemplate
from visuaml.compiler_lab.hf_llama_checkpoint_v19 import (
    LlamaLocalCheckpoint,
    load_hf_llama_local,
)
from visuaml.compiler_lab.llama_semantic_v20 import (
    lower_llama_semantic_program,
    make_llama_semantic_program,
)


def benchmark(
    checkpoint: LlamaLocalCheckpoint, *,
    clients: int = 4, trials: int = 3, prompt_len: int = 4
) -> dict:
    if (
        type(clients) is not int or not 2 <= clients <= 16
        or type(trials) is not int or not 2 <= trials <= 20
        or type(prompt_len) is not int or not 1 <= prompt_len <= 32
    ):
        raise ValueError("Invalid bounded multi-session benchmark configuration")
    program = make_llama_semantic_program(checkpoint)
    plan = lower_llama_semantic_program(program, checkpoint)
    prompts = tuple(
        (np.arange(prompt_len, dtype=np.int64)[None, :] + 1 + i)
        % checkpoint.profile.vocabulary
        for i in range(clients)
    )
    oracle = tuple(checkpoint.forward(tokens) for tokens in prompts)
    results = {
        "per_session_retrace": {"wall_ms": [], "graph_trace_ms": [], "graphs": []},
        "shared_template": {"wall_ms": [], "graph_trace_ms": [], "graphs": []},
    }

    for index in range(trials):
        cases = ["per_session_retrace", "shared_template"]
        if index % 2:
            cases.reverse()
        for mode in cases:
            started = time.perf_counter_ns()
            trace_ms = 0.0
            count = 0
            if mode == "shared_template":
                template = CompiledLlamaTemplate(plan)
                for tokens, expected in zip(prompts, oracle):
                    session = template.new_session()
                    np.testing.assert_allclose(
                        session.step(tokens), expected, atol=5e-5, rtol=5e-5
                    )
                    assert session.cursor == prompt_len
                witness = template.report()
                count = witness["compiled_graphs"]
                trace_ms = witness["first_trace"]["trace_ms"]
            else:
                for tokens, expected in zip(prompts, oracle):
                    session = plan.new_session(backend="torchscript_poly")
                    np.testing.assert_allclose(
                        session.step(tokens), expected, atol=5e-5, rtol=5e-5
                    )
                    trace_ms += sum(
                        entry["trace_ms"] for entry in session.report()["traces"]
                    )
                    count += session.report()["specializations"]
            wall_ms = (time.perf_counter_ns() - started) / 1e6
            results[mode]["wall_ms"].append(wall_ms)
            results[mode]["graph_trace_ms"].append(trace_ms)
            results[mode]["graphs"].append(count)
    root = Path(__file__).resolve().parents[1]
    sources = (
        "examples/llama_template_bench_v23.py",
        "visuaml/compiler_lab/compiled_template_v23.py",
        "visuaml/compiler_lab/torchscript_graph_v21.py",
    )
    return {
        "schema": "visuaml.llama-compiled-template-benchmark.v23",
        "semantic_sha256": plan.semantic_sha256,
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "declared_revision": checkpoint.declared_revision,
        "clients": clients, "prompt_len": prompt_len, "trials": trials,
        "results": {
            mode: {
                "wall_median_ms": round(statistics.median(values["wall_ms"]), 6),
                "trace_median_ms": round(statistics.median(values["graph_trace_ms"]), 6),
                "graph_counts": values["graphs"],
                "wall_p95_ms": round(float(np.percentile(values["wall_ms"], 95)), 6),
            }
            for mode, values in results.items()
        },
        "parity": "all client logits matched independent local Llama reference",
        "source_files_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in sources
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "blas_threads": os.getenv("OPENBLAS_NUM_THREADS", "unset"),
        "limitations": [
            "Cold multi-session CPU software microbenchmark, no server concurrency or network.",
            "Template shares graph but sessions still validate models and allocate independent KV.",
            "Trace times are a component of, not an addition to, measured wall time.",
            "Neither backend is a standalone native AOT artifact or a GPU implementation.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--prompt-len", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model = load_hf_llama_local(args.checkpoint, declared_revision=args.revision)
    report = benchmark(
        model, clients=args.clients, trials=args.trials,
        prompt_len=args.prompt_len
    )
    wire = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(wire, encoding="utf-8")
    print(wire)


if __name__ == "__main__":
    main()
