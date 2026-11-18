"""Isolate exact-binary admission and warm compact routing costs."""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np

from scripts.benchmark_neural_vm_v58 import environment, git_identity, measure
from visuaml.compiler_lab.bit_neural_vm_v49 import _bits, construct_bit_weights
from visuaml.compiler_lab.compact_routing_v58 import CompactRouter
from visuaml.compiler_lab.constructed_alu_v58 import compiled_alu
from visuaml.compiler_lab.exact_binary import is_exact_binary
from visuaml.compiler_lab.neural_control_routing_v52 import (
    TERMINAL, _onehot, compile_route_tokens, construct_router_weights, neural_pc_step,
)
from visuaml.compiler_lab.wasm_research_corpus_v57 import countdown, gcd12


def _old_binary(value):
    return bool(((value == 0.) | (value == 1.)).all())


def _repeat(action, repeat):
    for _ in range(repeat):
        action()


def run(root):
    source = git_identity(root)
    measures = []
    for size in (4, 32, 97, 768, 9409):
        values = np.random.default_rng(58).integers(0, 2, size, dtype=np.int8).astype(np.float64)
        values[0] = -0.
        bad = values.copy()
        bad[-1] = np.nextafter(1., 2.)
        for label, vec in (("binary", values), ("nonbinary", bad)):
            if _old_binary(vec) != is_exact_binary(vec):
                raise AssertionError("Binary admission changed")
            samples = {
                name: measure(partial(_repeat, partial(predicate, vec), 2048), warmups=1, repeats=9)
                for name, predicate in (("all", _old_binary), ("count_nonzero", is_exact_binary))
            }
            measures.append({"size": size, "case": label, "checks_per_sample": 2048, "timing": samples})

    router_rows = []
    compiled_alu()
    fixed, bit = construct_router_weights(), construct_bit_weights()
    for name, program in (("countdown", countdown()), ("gcd", gcd12())):
        tokens = compile_route_tokens(program)
        candidate = CompactRouter(program, tokens, fixed, bit)
        samples = [(pc, condition)
                   for i in (*range(len(program.operations)), TERMINAL)
                   for pc in (_onehot(i),)
                   for condition in (_bits(0), _bits(-(1 << 31)))]
        for pc, condition in samples:
            dense, dense_detail = neural_pc_step(pc, condition, tokens, fixed, bit)
            compact, compact_detail = candidate.step(pc, condition)
            if not np.array_equal(dense, compact) or dense_detail != compact_detail:
                raise AssertionError("PC contraction changed a route")

        def batch(step):
            for pc, condition in samples:
                step(pc, condition)

        timings = {
            "dense": measure(partial(batch, lambda pc, condition:
                                     neural_pc_step(pc, condition, tokens, fixed, bit)),
                             warmups=1, repeats=9),
            "compact": measure(partial(batch, candidate.step), warmups=1, repeats=9),
            "compact_admission": measure(partial(CompactRouter, program, tokens, fixed, bit),
                                         warmups=1, repeats=9),
        }
        router_rows.append({
            "program": name, "source_wasm_sha256": tokens.source_wasm_sha256,
            "tokens_sha256": tokens.digest(), "weights_sha256": fixed.digest(),
            "arithmetic_weights_sha256": bit.digest(), "routes_per_sample": len(samples),
            "timing": timings,
        })
    return {"source": source, "environment": environment(), "binary": measures, "routing": router_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(root):
        parser.error("Use a fresh output path outside the checkout")
    record = run(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(record, stream, sort_keys=True, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
