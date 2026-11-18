"""Measure the shared-weight v58 runtime on the original scaling corpus."""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import inspect
import json
import statistics
import subprocess
from dataclasses import fields, replace
from functools import partial
from pathlib import Path

import numpy as np
import wasmtime

from scripts.benchmark_neural_vm_v58 import digest, environment, measure, measure_python_allocations, workloads
from scripts.probe_typed_microop_v57 import tensors_digest
from visuaml.compiler_lab import neural_microstep_executor_v58 as v58
from visuaml.compiler_lab.neural_alu_executor_v56 import execute_alu_routed_control
from visuaml.compiler_lab.neural_microop_executor_v57 import execute_microop_routed_control
from visuaml.compiler_lab.wasm_control_flow_v51 import encode_control, execute_control

_BASE = "322cd520fb440790659c9e3f62962b9e427803fc"
_FIELDS = ("pc", "next_pc", "opcode", "stack_i32", "locals_i32", "control_labels", "halted")
_CONSTRUCTORS = (
    v58.construct_router_weights, v58.construct_bit_weights, v58.construct_microop_weights,
    v58.construct_microstep_weights, v58.construct_alu_weights, v58.construct_local_bank_weights,
    v58.fixed_operand_weights,
)


def _git(root, *args):
    return subprocess.check_output(("git", "-C", str(root), *args))


def _identify(root, baseline):
    if _git(root, "status", "--porcelain"):
        raise ValueError("Run with a clean research checkout")
    revision = _BASE if baseline else "HEAD"
    commit, tree = _git(root, "rev-parse", revision, f"{revision}^{{tree}}").decode().splitlines()
    for module_name in (
        "neural_microstep_executor_v58", "bit_neural_vm_v49", "neural_alu_v56",
        "neural_control_routing_v52", "neural_local_bank_v57", "neural_microstep_v58",
        "neural_operand_storage_v55", "typed_microop_ir_v57",
    ):
        module = __import__(f"visuaml.compiler_lab.{module_name}", fromlist=("_",))
        path = Path(inspect.getfile(module))
        file = f"visuaml-client/backend/visuaml/compiler_lab/{path.name}"
        if path.read_bytes() != _git(root, "show", f"{commit}:{file}"):
            raise ValueError(f"Loaded runtime doesn't match producing source: {file}")
    return {"commit": commit, "tree": tree, "baseline": baseline}


def _weights():
    return tuple(factory() for factory in _CONSTRUCTORS)


def _profile_rows(profile):
    rows = []
    for entry in profile.getstats():
        code = entry.code
        if isinstance(code, str):
            label = code
        else:
            label = f"{Path(code.co_filename).name}:{code.co_firstlineno}:{code.co_name}"
        rows.append({
            "function": label, "calls": entry.callcount,
            "self_s": entry.inlinetime, "inclusive_s": entry.totaltime,
        })
    return sorted(rows, key=lambda row: row["inclusive_s"], reverse=True)


def collect(root, baseline, repeats, profile_enabled, use_compiled=False, use_fused=False):
    identity = _identify(root, baseline)
    compiled_metadata = None
    if use_compiled:
        from time import perf_counter_ns

        from visuaml.compiler_lab.constructed_alu_v58 import compiled_alu

        started = perf_counter_ns()
        library = compiled_alu()
        compiled_metadata = {
            "setup_wall_ns": perf_counter_ns() - started,
            "source_sha256": library.source_sha256, "binary_sha256": library.binary_sha256,
            "compiler": library.compiler, "flags": library.flags,
        }
    engine = wasmtime.Engine()
    profile = cProfile.Profile()
    to_profile = []
    rows = []
    weight_ids = {
        factory.__name__: tensors_digest(*(getattr(weight, field.name) for field in fields(weight)
                                           if isinstance(getattr(weight, field.name), np.ndarray)))
        for factory, weight in zip(_CONSTRUCTORS, _weights(), strict=True)
    }
    for work in workloads(scaling=True):
        if any(op.opcode in ("block", "loop", "if", "else", "br", "br_if", "return")
               for op in work.program.operations[:-1]):
            continue
        source = encode_control(work.program)
        store = wasmtime.Store(engine)
        wasm = wasmtime.Instance(store, wasmtime.Module(engine, source), []).exports(store)["run"]
        tokens = v58.compile_microop_tokens(work.program)
        cases = []
        for argument in work.arguments:
            actions = {"trace": partial(v58.execute_microstep_control, work.program, argument,
                                        program_tokens=tokens, fuel=1500)}
            numpy_trace = actions["trace"]
            if use_compiled:
                actions["trace"] = partial(actions["trace"], use_compiled_alu=True)
            unfused = actions["trace"]
            if use_fused:
                actions["trace"] = partial(unfused, use_fused_state=True)
            traced = actions["trace"]()
            if use_fused and unfused() != traced:
                raise AssertionError((work.name, argument, "fused state changed the trace"))
            if use_compiled and not use_fused and numpy_trace() != traced:
                raise AssertionError((work.name, argument, "compiled and NumPy gate traces differ"))
            result = wasm(store, argument)
            v51 = execute_control(work.program, argument, fuel=1500)
            v56 = execute_alu_routed_control(work.program, argument, fuel=1500)
            v57 = execute_microop_routed_control(work.program, argument, fuel=1500, microop_tokens=tokens)
            if not (traced.result_i32 == v51.value == v56.result_i32 == v57.result_i32 == result):
                raise AssertionError((work.name, argument, "result parity"))
            for states in zip(traced.trace, v51.trace, v56.trace, v57.trace, strict=True):
                if any(any(row[key] != states[0][key] for row in states[1:]) for key in _FIELDS):
                    raise AssertionError((work.name, argument, "state parity"))
            if not baseline:
                actions["no_trace"] = partial(actions["trace"], capture_trace=False)
                if actions["no_trace"]() != replace(traced, trace=()):
                    raise AssertionError((work.name, argument, "trace mode changes state evidence"))
            if use_fused:
                actions["unfused_trace"] = unfused
                actions["unfused_no_trace"] = partial(unfused, capture_trace=False)
            elif use_compiled:
                actions["numpy_trace"] = numpy_trace
                actions["numpy_no_trace"] = partial(numpy_trace, capture_trace=False)
            order = tuple(actions) if len(cases) % 2 == 0 else tuple(reversed(actions))
            cases.append({
                "input": argument, "result": result, "steps": traced.steps,
                "trace_sha256": digest(traced.trace),
                "timing": {name: measure(actions[name], warmups=1, repeats=repeats, expected=result)
                           for name in order},
                "trace_serialization": measure(lambda: digest(traced.trace), warmups=1, repeats=repeats),
            })
            if profile_enabled:
                to_profile.append(({name: action for name, action in actions.items()
                                    if name in ("trace", "no_trace")}, len(cases) == 1))
        def prepare_tokens():
            return (v58.compile_route_tokens(work.program), v58.compile_microop_tokens(work.program))

        rows.append({
            "program": work.name, "source_wasm_sha256": hashlib.sha256(source).hexdigest(),
            "microop_tokens_sha256": tokens.digest(),
            "route_tokens_sha256": v58.compile_route_tokens(work.program).digest(),
            "token_construction": measure(prepare_tokens, warmups=1, repeats=repeats),
            "cases": cases,
        })
    allocations = []
    for actions, first in to_profile:
        profile.runcall(actions["trace"])
        if first:
            allocations.append({name: measure_python_allocations(action) for name, action in actions.items()})
    return {
        "schema": "visuaml.constructed-runtime-profile.v58", "source": identity,
        "environment": environment(), "repeats": repeats, "fixed_weights": weight_ids,
        "compiled_alu": compiled_metadata, "fused_state": use_fused,
        "fixed_initialization": measure(_weights, warmups=1, repeats=repeats),
        "profile_traced_inclusive": _profile_rows(profile) if profile_enabled else [],
        "profiled_calls": len(to_profile), "python_allocations_per_first_input": allocations, "rows": rows,
    }


def compare(baseline, candidate):
    def cases(record):
        return {(row["program"], row["source_wasm_sha256"], row["microop_tokens_sha256"],
                 row["route_tokens_sha256"], case["input"]): case
                for row in record["rows"] for case in row["cases"]}

    old, new = cases(baseline), cases(candidate)
    if old.keys() != new.keys() or baseline["fixed_weights"] != candidate["fixed_weights"]:
        raise ValueError("Baseline and candidate provenance differs")
    ratios = {"trace": [], "no_trace": []}
    changed_values = 0
    for key in old:
        for field in ("result", "steps", "trace_sha256"):
            if old[key][field] != new[key][field]:
                raise AssertionError((key, field, "reference execution changed"))
        changed_values += new[key]["steps"]
        for mode, values in ratios.items():
            values.append(old[key]["timing"]["trace"]["wall"]["median_ns"]
                          / new[key]["timing"][mode]["wall"]["median_ns"])
    return {
        "witnesses": len(new), "compared_states": changed_values,
        "per_mode": {name: {
            "geomean_speedup": statistics.geometric_mean(values),
            "median_speedup": statistics.median(values),
            "worst_regression_factor": 1 / min(values),
            "faster_cases": sum(ratio > 1 for ratio in values),
        } for name, values in ratios.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BASELINE", "CANDIDATE"))
    parser.add_argument("--compiled-alu", action="store_true")
    parser.add_argument("--fused-state", action="store_true")
    args = parser.parse_args()
    if args.baseline and (args.compiled_alu or args.fused_state):
        parser.error("The archived baseline predates the compiled ALU and fused state")
    root = Path(__file__).resolve().parents[3]
    if args.output.exists() or args.output.resolve().is_relative_to(root) or args.repeats < 1:
        parser.error("Use a fresh output path outside the checkout and positive repeats")
    if args.compare:
        record = compare(*(json.loads(path.read_bytes()) for path in args.compare))
        print(json.dumps(record, sort_keys=True))
    else:
        record = collect(root, args.baseline, args.repeats, args.profile, args.compiled_alu, args.fused_state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(record, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")


if __name__ == "__main__":
    main()
