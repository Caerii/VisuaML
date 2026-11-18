"""Measure serial native batching against repeated v58 native loop calls."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import wasmtime

from scripts.benchmark_neural_vm_v58 import digest, environment, git_identity, measure, workloads
from visuaml.compiler_lab.constructed_alu_v58 import compiled_alu
from visuaml.compiler_lab.neural_microstep_executor_v58 import PreparedMicrostepProgram
from visuaml.compiler_lab.wasm_control_flow_v51 import encode_control

SIZES = (1, 8, 32, 128, 1024)
PRIOR = "benchmarks/results/v58-native-loop-16c2814/runtime-1.json.gz"
CONTROL = {"block", "loop", "if", "else", "br", "br_if", "return"}


def collect(root, repeats):
    identity = git_identity(root)
    kernel = compiled_alu()
    prior = json.loads(gzip.decompress((root / PRIOR).read_bytes()))
    reference = {(row["program"], case["input"]): case
                 for row in prior["rows"] for case in row["cases"]}
    corpus = [work for work in workloads(scaling=True)
              if not any(op.opcode in CONTROL for op in work.program.operations[:-1])]
    engine = wasmtime.Engine()
    witnesses = {}
    for work in corpus:
        session = PreparedMicrostepProgram(work.program, use_native_loop=True)
        store = wasmtime.Store(engine)
        module = wasmtime.Module(engine, encode_control(work.program))
        wasm = wasmtime.Instance(store, module, []).exports(store)["run"]
        batch = session.run_many(work.arguments)
        for argument, result in zip(work.arguments, batch, strict=True):
            old = reference[work.name, argument]
            if not (result == session.run(argument) and result.result_i32 == wasm(store, argument)
                    == old["result"] and result.steps == old["steps"]
                    and digest(result.trace) == old["trace_sha256"]):
                raise AssertionError((work.name, argument, "batch differs from reference"))
            witnesses[work.name, argument] = result.steps

    if witnesses.keys() != reference.keys():
        raise AssertionError("Batch corpus differs from archived reference")
    ordered = sorted(corpus, key=lambda work: (len(work.program.operations), work.name))
    selected = ordered[0], ordered[len(ordered) // 2], ordered[-1]
    rows = []
    for work in selected:
        def prepare():
            return PreparedMicrostepProgram(work.program, use_native_loop=True)

        session = prepare()
        for size in SIZES:
            arguments = tuple(work.arguments[i % len(work.arguments)] for i in range(size))
            for capture in (False, True):
                def individual(prepared):
                    return tuple(prepared.run(a, capture_trace=capture) for a in arguments)

                def batch(prepared):
                    return prepared.run_many(arguments, capture_trace=capture)

                if individual(session) != batch(session):
                    raise AssertionError((work.name, size, capture, "batch differs from individual"))
                actions = {"individual": individual, "batch": batch}
                order = tuple(actions) if len(rows) % 2 == 0 else tuple(reversed(actions))
                timing = {}
                for name in order:
                    action = actions[name]
                    timing[name] = {
                        "warm": measure(lambda: action(session), warmups=1, repeats=repeats),
                        "with_preparation": measure(lambda: action(prepare()), warmups=1, repeats=repeats),
                    }
                rows.append({"program": work.name, "steps": len(work.program.operations),
                             "size": size, "traced": capture, "timings": timing})
    return {
        "source": identity, "environment": environment(), "kernel_sha256": kernel.binary_sha256,
        "kernel_source_sha256": kernel.source_sha256, "repeats": repeats,
        "reference": PRIOR, "witnesses": len(witnesses), "states": sum(witnesses.values()),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    if args.output.exists() or args.output.resolve().is_relative_to(root) or args.repeats < 1:
        parser.error("Use a fresh output path outside the checkout and positive repeats")
    report = collect(root, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")


if __name__ == "__main__":
    main()
