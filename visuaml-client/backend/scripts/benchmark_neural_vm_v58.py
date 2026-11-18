"""v58 offline, source-bound neural WASM benchmark laboratory.

A benchmark run is admitted only after source-WASM/independent-interpreter
parity has passed. Never invokes GitHub Actions, calls the network or changes
the repository. Timings are measurements of the *Python/NumPy implementation*
on the recorded hardware, not neural silicon or optimized native compute.

Usage (from repository root):
 PYTHONPATH=visuaml-client/backend python visuaml-client/backend/scripts/benchmark_neural_vm_v58.py --output /tmp/v58.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from visuaml.compiler_lab.neural_alu_executor_v56 import (
    execute_alu_routed_control,
)
from visuaml.compiler_lab.neural_microop_executor_v57 import (
    execute_microop_routed_control,
)
from visuaml.compiler_lab.neural_microstep_executor_v58 import (
    execute_microstep_control,
)
from visuaml.compiler_lab.typed_microop_ir_v57 import (
    compile_microop_tokens,
    construct_microop_weights,
    validate_microop_tokens,
)
from visuaml.compiler_lab.wasm_control_flow_v51 import (
    encode_control,
    execute_control,
)
from visuaml.compiler_lab.wasm_research_corpus_v57 import (
    corpus,
    generated,
)


SCHEMA = "visuaml.neural-wasm-benchmark.v58"
ENGINES = (
    "host_v51", "alu_v56", "microop_v57", "microstep_v58",
    "wasmtime",
)
DEFAULT_INPUTS = (-(1 << 31), -17, -1, 0, 1, 19, (1 << 31)-1)


def stable_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(stable_bytes(value)).hexdigest()


def summarize_ns(values: list[int]) -> dict[str, int | float]:
    if not values or any(type(x) is not int or x < 0 for x in values):
        raise ValueError("Expected nonnegative integer nanosecond samples")
    ordered = sorted(values)
    index = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return {
        "count": len(ordered),
        "median_ns": int(statistics.median(ordered)),
        "p95_ns": ordered[index],
        "min_ns": ordered[0],
        "max_ns": ordered[-1],
        "mean_ns": statistics.mean(ordered),
    }


@dataclass(frozen=True)
class Workload:
    name: str
    family: str
    program: object
    arguments: tuple[int, ...]
    split: str


def workloads(*, holdout: int = 8) -> list[Workload]:
    if type(holdout) is not int or not 0 <= holdout <= 64:
        raise ValueError("Holdout count must be 0..64")
    result = []
    for name, (program, args) in sorted(corpus().items()):
        result.append(Workload(
            name, "structured" if not name.startswith("generated_")
            else "generated_in_sample", program, tuple(args), "public",
        ))
    for seed in range(128, 128 + holdout):
        result.append(Workload(
            f"heldout_{seed}", "generated_out_of_sample",
            generated(seed), DEFAULT_INPUTS, "heldout",
        ))
    digests = [hashlib.sha256(encode_control(x.program)).hexdigest()
               for x in result]
    if len(set(digests)) != len(digests):
        raise ValueError("Benchmark corpus has duplicated source WASM")
    return result


def git_identity(root: Path) -> dict[str, str]:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ("git", "-C", str(root), *args), text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("Benchmarks require a clean Git worktree")
    return {
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "branch": git("branch", "--show-current"),
    }


def version(name: str) -> str:
    return importlib.metadata.version(name)


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "wasmtime": version("wasmtime"),
        "thread_controls": {
            key: os.environ.get(key) for key in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def measure(action, *, warmups: int, repeats: int) -> dict:
    if repeats < 1 or warmups < 0:
        raise ValueError("Expected nonnegative warmups and positive repeats")
    for _ in range(warmups):
        action()
    wall = []
    cpu = []
    for _ in range(repeats):
        before_wall = time.perf_counter_ns()
        before_cpu = time.process_time_ns()
        action()
        cpu.append(time.process_time_ns() - before_cpu)
        wall.append(time.perf_counter_ns() - before_wall)
    return {"wall": summarize_ns(wall), "process_cpu": summarize_ns(cpu)}


def run(
    root: Path, *, repeats: int, warmups: int, holdout: int,
    fuel: int,
) -> dict:
    if type(repeats) is not int or not 1 <= repeats <= 10000:
        raise ValueError("Repeat count outside benchmark policy")
    if type(warmups) is not int or not 0 <= warmups <= 10000:
        raise ValueError("Warmups outside benchmark policy")
    if type(fuel) is not int or not 1 <= fuel <= 2000:
        raise ValueError("Fuel bound outside v51 policy")
    import wasmtime

    identity = git_identity(root)
    wasmtime_engine = wasmtime.Engine()
    rows = []
    overall = {"programs": 0, "witnesses": 0, "instruction_transitions": 0}
    for workload in workloads(holdout=holdout):
        source = encode_control(workload.program)
        wasmtime.Module.validate(wasmtime_engine, source)
        # Setup/compile is explicitly excluded from warm execution time.
        module = wasmtime.Module(wasmtime_engine, source)
        store = wasmtime.Store(wasmtime_engine)
        wasm_function = wasmtime.Instance(
            store, module, []
        ).exports(store)["run"]
        tokens = compile_microop_tokens(workload.program)
        validate_microop_tokens(tokens, workload.program)
        load_metric = measure(
            lambda: compile_microop_tokens(workload.program),
            warmups=warmups, repeats=repeats,
        )
        source_hash = hashlib.sha256(source).hexdigest()
        cases = []
        for argument in workload.arguments:
            # Correctness is independent of timing: fail *before* measurement.
            v51 = execute_control(workload.program, argument, fuel=fuel)
            v56 = execute_alu_routed_control(
                workload.program, argument, fuel=fuel,
            )
            v57 = execute_microop_routed_control(
                workload.program, argument, fuel=fuel,
                microop_tokens=tokens,
            )
            eligible = all(
                op.opcode not in (
                    "block", "loop", "if", "else", "br", "br_if", "return"
                ) for op in workload.program.operations[:-1]
            )
            v58 = (
                execute_microstep_control(
                    workload.program, argument, fuel=fuel,
                    program_tokens=tokens,
                ) if eligible else None
            )
            native = wasm_function(store, argument)
            if v58 is not None and v58.result_i32 != v57.result_i32:
                raise AssertionError(
                    (workload.name, argument, "v58 result disagreement")
                )
            if not (v51.value == v56.result_i32 == v57.result_i32
                    == native):
                raise AssertionError(
                    (workload.name, argument, "output disagreement")
                )
            if not (v51.transitions == v56.steps == v57.steps):
                raise AssertionError(
                    (workload.name, argument, "instruction count mismatch")
                )
            for n, (old, previous, current) in enumerate(zip(
                v51.trace, v56.trace, v57.trace, strict=True,
            )):
                for key in (
                    "pc", "next_pc", "opcode", "stack_i32",
                    "locals_i32", "control_labels", "halted",
                ):
                    if not (old[key] == previous[key] == current[key]):
                        raise AssertionError(
                            (workload.name, argument, n, key)
                        )
            if v58 is not None:
                if v58.steps != v57.steps:
                    raise AssertionError((workload.name, argument, "v58 steps"))
                for i, (new, old) in enumerate(zip(
                    v58.trace, v57.trace, strict=True
                )):
                    for key in (
                        "pc", "next_pc", "opcode", "stack_i32",
                        "locals_i32", "control_labels", "halted",
                    ):
                        if new[key] != old[key]:
                            raise AssertionError(
                                (workload.name, argument, i, key, "v58 mismatch")
                            )
            actions = {
                "host_v51": lambda: execute_control(
                    workload.program, argument, fuel=fuel,
                ),
                "alu_v56": lambda: execute_alu_routed_control(
                    workload.program, argument, fuel=fuel,
                ),
                "microop_v57": lambda: execute_microop_routed_control(
                    workload.program, argument, fuel=fuel,
                    microop_tokens=tokens,
                ),
                "wasmtime": lambda: wasm_function(store, argument),
            }
            if eligible:
                actions["microstep_v58"] = lambda: execute_microstep_control(
                    workload.program, argument, fuel=fuel,
                    program_tokens=tokens,
                )
            measurements = {
                name: measure(action, warmups=warmups, repeats=repeats)
                for name, action in actions.items()
            }
            case = {
                "input_i32": argument,
                "expected_i32": native,
                "steps": v57.steps,
                "trace_sha256": digest(list(v57.trace)),
                "latencies": measurements,
            }
            cases.append(case)
            overall["witnesses"] += 1
            overall["instruction_transitions"] += v57.steps
        rows.append({
            "name": workload.name,
            "split": workload.split,
            "family": workload.family,
            "source_wasm_sha256": source_hash,
            "source_program_sha256": workload.program.digest(),
            "source_wasm_size_bytes": len(source),
            "source_instruction_count": len(workload.program.operations),
            "microstep_v58_eligible": all(
                op.opcode not in (
                    "block", "loop", "if", "else", "br", "br_if", "return"
                ) for op in workload.program.operations[:-1]
            ),
            "opcode_histogram": {
                key: sum(op.opcode == key for op in workload.program.operations)
                for key in sorted({op.opcode for op in workload.program.operations})
            },
            "microop_weights_sha256": construct_microop_weights().digest(),
            "microop_program_tokens_sha256": tokens.digest(),
            "tokenize_latency": load_metric,
            "cases": cases,
        })
        overall["programs"] += 1
    return {
        "schema": SCHEMA,
        "source": identity,
        "environment": environment(),
        "policy": {
            "repeats": repeats, "warmups": warmups, "fuel": fuel,
            "holdout_programs": holdout,
            "timing_scope": "Python/NumPy engine invocation including traces",
            "setup_excluded_from_warm": True,
            "wasmtime_calls_include_python_binding_overhead": True,
            "cpu_frequency_or_turbo_not_controlled": True,
            "same_hardware_comparisons_only": True,
            "performance_not_correctness_proof": True,
            "not_native_neural_hardware": True,
            "v58_only_straightline_admitted": True,
        },
        "inventory": overall,
        "engines": list(ENGINES),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--holdout", type=int, default=8)
    parser.add_argument("--fuel", type=int, default=1500)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite benchmark evidence: {output}")
    root = Path(__file__).resolve().parents[3]
    record = run(
        root, repeats=args.repeats, warmups=args.warmups,
        holdout=args.holdout, fuel=args.fuel,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(stable_bytes(record) + b"\n")
    print(json.dumps({
        "artifact": str(output),
        "artifact_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "inventory": record["inventory"],
        "source_commit": record["source"]["commit"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
