"""Matched v23/v24 real-Llama cold-session experiment with numerical witnesses.

Both variants compile one graph per model. The v23 control constructs new
weight-bearing Torch modules for every client session; v24 uses the same
verified graph AND the same weight-bearing module, allocating private KV
state per client. Source hashes, cache bytes, module object identity and
all output logits are recorded. No full-memory RSS or GPU/serving claims.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
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
from visuaml.compiler_lab.shared_weights_v24 import SharedWeightsLlamaTemplate


def _median(values: list[float]) -> float:
    return round(statistics.median(values), 6)


def benchmark(
    checkpoint: LlamaLocalCheckpoint, *,
    clients: int = 5, trials: int = 3, prompt_len: int = 4
) -> dict:
    if (
        type(clients) is not int or not 2 <= clients <= 16
        or type(trials) is not int or not 2 <= trials <= 20
        or type(prompt_len) is not int or not 1 <= prompt_len <= 32
    ):
        raise ValueError("Invalid bounded real-checkpoint session benchmark input")
    semantic = make_llama_semantic_program(checkpoint)
    plan = lower_llama_semantic_program(semantic, checkpoint)
    prompts = tuple(
        ((np.arange(prompt_len, dtype=np.int64) + 1 + i)
         % checkpoint.profile.vocabulary)[None, :]
        for i in range(clients)
    )
    oracles = tuple(checkpoint.forward(tokens) for tokens in prompts)
    modes = (
        ("v23_graph_only", CompiledLlamaTemplate),
        ("v24_graph_and_weights", SharedWeightsLlamaTemplate),
    )
    samples = {
        name: {
            "cold_wall_ms": [], "additional_factory_ms": [],
            "graph_trace_ms": [], "module_counts": [],
            "module_storage_bytes": [], "kv_storage_bytes": [],
            "compiled_graph_counts": [],
        }
        for name, _ in modes
    }
    for trial in range(trials):
        options = list(modes)
        if trial % 2:
            options.reverse()
        for name, constructor in options:
            begun = time.perf_counter_ns()
            template = constructor(plan)
            sessions = []
            additional_ms = 0.0
            for index, (tokens, oracle) in enumerate(zip(prompts, oracles)):
                start = time.perf_counter_ns()
                session = template.new_session()
                if index > 0:
                    additional_ms += (time.perf_counter_ns() - start) / 1e6
                output = session.step(tokens)
                np.testing.assert_allclose(
                    output, oracle, atol=5e-5, rtol=5e-5
                )
                assert session.cursor == prompt_len
                sessions.append(session)
            wall_ms = (time.perf_counter_ns() - begun) / 1e6
            graph_count = template.report()["compiled_graphs"]
            modules = {id(session._torch): session._torch for session in sessions}
            module_storage = sum(
                sum(buf.numel() * buf.element_size() for buf in module.buffers())
                for module in modules.values()
            )
            kv_storage = sum(
                session.report()["reserved_bytes"] for session in sessions
            )
            samples[name]["cold_wall_ms"].append(wall_ms)
            samples[name]["additional_factory_ms"].append(additional_ms)
            samples[name]["graph_trace_ms"].append(
                template.report()["first_trace"]["trace_ms"]
            )
            samples[name]["module_counts"].append(len(modules))
            samples[name]["module_storage_bytes"].append(module_storage)
            samples[name]["kv_storage_bytes"].append(kv_storage)
            samples[name]["compiled_graph_counts"].append(graph_count)
            assert graph_count == 1
            del template, sessions, modules
            gc.collect()
    root = Path(__file__).resolve().parents[1]
    files = (
        "examples/shared_weights_bench_v24.py",
        "visuaml/compiler_lab/shared_weights_v24.py",
        "visuaml/compiler_lab/compiled_template_v23.py",
        "visuaml/compiler_lab/torchscript_graph_v21.py",
    )
    return {
        "schema": "visuaml.shared-weights-benchmark.v24",
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "semantic_sha256": plan.semantic_sha256,
        "declared_revision": checkpoint.declared_revision,
        "clients": clients, "trials": trials, "prompt_tokens": prompt_len,
        "results": {
            mode: {
                "cold_wall_median_ms": _median(values["cold_wall_ms"]),
                "cold_wall_p95_ms": round(
                    float(np.percentile(values["cold_wall_ms"], 95)), 6
                ),
                "additional_factory_median_ms": _median(
                    values["additional_factory_ms"]
                ),
                "first_graph_trace_median_ms": _median(
                    values["graph_trace_ms"]
                ),
                "unique_weight_module_counts": values["module_counts"],
                "weight_module_logical_bytes": values["module_storage_bytes"],
                "kv_storage_bytes": values["kv_storage_bytes"],
                "compiled_graph_counts": values["compiled_graph_counts"],
            }
            for mode, values in samples.items()
        },
        "witness": "All clients compared to independent local NumPy Llama logits",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in files
        },
        "limitations": [
            "Graph is compiled once in both strategies. v24 also shares the weight module.",
            "Logical buffer sizes are not measured physical RSS or Python heap usage.",
            "CPU cold-session execution including validation and independent cache allocation.",
            "Not a multi-process serving test, native AOT artifact, GPU or optical device.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--prompt-len", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checkpoint = load_hf_llama_local(
        args.checkpoint, declared_revision=args.revision
    )
    report = benchmark(
        checkpoint, clients=args.clients,
        trials=args.trials, prompt_len=args.prompt_len
    )
    wire = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(wire, encoding="utf-8")
    print(wire)


if __name__ == "__main__":
    main()
