"""Compare source-specialized v58 execution against neural v58 and Wasmtime."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import wasmtime

from scripts.benchmark_neural_vm_v58 import environment, git_identity, measure, workloads
from visuaml.compiler_lab.native_specialization_v58 import NativeSpecialization
from visuaml.compiler_lab.neural_microstep_executor_v58 import execute_microstep_control
from visuaml.compiler_lab.neural_microstep_v58 import DATA_OPS
from visuaml.compiler_lab.typed_microop_ir_v57 import compile_microop_tokens
from visuaml.compiler_lab.wasm_control_flow_v51 import encode_control, execute_control


def run(root: Path, *, repeats: int, scaling: bool) -> dict:
    source = git_identity(root)
    engine = wasmtime.Engine()
    rows = []
    excluded = []
    for work in workloads(scaling=scaling):
        if any(op.opcode not in DATA_OPS for op in work.program.operations[:-1]):
            excluded.append({"name": work.name, "witnesses": len(work.arguments)})
            continue
        raw = encode_control(work.program)
        tokens = compile_microop_tokens(work.program)
        start = time.perf_counter_ns()
        candidate = NativeSpecialization(raw)
        native_setup_ns = time.perf_counter_ns() - start
        start = time.perf_counter_ns()
        module = wasmtime.Module(engine, raw)
        store = wasmtime.Store(engine)
        entry = wasmtime.Instance(store, module, []).exports(store)["run"]
        wasmtime_setup_ns = time.perf_counter_ns() - start
        cases = []
        for argument in work.arguments:
            def reference():
                return execute_microstep_control(work.program, argument, program_tokens=tokens)

            def native():
                return candidate.run(argument)

            def native_trace():
                return candidate.trace(argument)

            def wasm():
                return entry(store, argument)

            old = reference()
            trace = native_trace()
            oracle = execute_control(work.program, argument)
            if (native() != wasm() or native() != old.result_i32
                    or native() != oracle.value or len(trace) != old.steps):
                raise AssertionError((work.name, argument, "output or step mismatch"))
            for actual, expected, independent in zip(trace, old.trace, oracle.trace, strict=True):
                if any(value != expected[key] or value != independent[key]
                       for key, value in actual.items()):
                    raise AssertionError((work.name, argument, actual["pc"], "state mismatch"))

            # Wasmtime and the native entry both go through their Python bindings.
            latencies = {
                name: measure(action, warmups=1, repeats=repeats, expected=old.result_i32)
                for name, action in (
                    ("neural_v58_trace", reference), ("native_warm", native), ("wasmtime_warm", wasm)
                )
            }
            latencies["native_trace"] = measure(native_trace, warmups=1, repeats=repeats)
            cases.append({"input_i32": argument, "steps": old.steps, "result_i32": old.result_i32,
                          "latencies": latencies})
        rows.append({
            "name": work.name, "split": work.split, "family": work.family,
            "wasm_sha256": candidate.source_sha256, "cpp_sha256": candidate.cpp_sha256,
            "binary_sha256": candidate.binary_sha256, "compiler": candidate.compiler,
            "native_setup_ns": native_setup_ns, "wasmtime_setup_ns": wasmtime_setup_ns,
            "cases": cases,
        })
    medians = [
        {name: result["wall"]["median_ns"] for name, result in case["latencies"].items()}
        for row in rows for case in row["cases"]
    ]
    paired = {
        name: [row[name] / row["native_warm"] for row in medians]
        for name in ("neural_v58_trace", "native_trace", "wasmtime_warm")
    }
    return {
        "schema": "visuaml.native-specialization-benchmark.v58",
        "source": source, "environment": environment(), "repeats": repeats,
        "scope": "Finite CPU witnesses. Neural v58 measures traced execution. "
                 "Native lowered code has no neural gate authority.",
        "summary": {
            "programs": len(rows), "witnesses": len(medians),
            "transitions": sum(case["steps"] for row in rows for case in row["cases"]),
            "excluded_witnesses": sum(row["witnesses"] for row in excluded),
            "median_native_setup_ns": statistics.median(row["native_setup_ns"] for row in rows),
            "median_wasmtime_setup_ns": statistics.median(row["wasmtime_setup_ns"] for row in rows),
            "median_case_latency_ns": {
                name: statistics.median(row[name] for row in medians)
                for name in medians[0]
            },
            "geomean_speedup_native_over": {
                name: math.exp(statistics.mean(map(math.log, ratios)))
                for name, ratios in paired.items()
            },
            "neural_over_native_trace_geomean": math.exp(statistics.mean(
                math.log(row["neural_v58_trace"] / row["native_trace"]) for row in medians
            )),
        },
        "excluded": excluded, "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--scaling", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    if args.repeats < 1 or output.exists() or output.is_relative_to(root):
        parser.error("Use positive repeats and a fresh output path outside the checkout")
    result = run(root, repeats=args.repeats, scaling=args.scaling)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
