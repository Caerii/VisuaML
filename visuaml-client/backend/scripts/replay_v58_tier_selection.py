"""Time held-out mixed workloads end to end with runtimes already initialized."""
from __future__ import annotations

import argparse
import json
import random
import statistics
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
import wasmtime

from scripts.benchmark_native_batch_v58 import _native_copy, _timed, _trial_order, _wrapped_source
from scripts.benchmark_neural_vm_v58 import environment, git_identity, summarize_ns, workloads
from scripts.evaluate_v58_tier_selection import _BACKENDS, evaluate
from visuaml.compiler_lab.llvm_specialization_v58 import LlvmSpecialization, _llvm
from visuaml.compiler_lab.native_specialization_v58 import NativeSpecialization
from visuaml.compiler_lab.wasm_control_flow_v51 import encode_control

_LEVELS = {"llvm_o0": 0, "llvm_o1": 1, "llvm": 2}


def _wasm_call(engine, raw, cached_path):
    module = (wasmtime.Module(engine, _wrapped_source(raw)) if cached_path is None
              else wasmtime.Module.deserialize_file(engine, str(cached_path)))
    store = wasmtime.Store(engine)
    exports = wasmtime.Instance(store, module, []).exports(store)
    entry = exports["batch"]
    memory = np.ctypeslib.as_array(exports["memory"].data_ptr(store), shape=(65536,)).view("<i4")

    def invoke(values):
        buffer = memory[:values.size]
        buffer[:] = values
        entry(store, values.size)
        return buffer.copy()

    return invoke


def _prepare(backend, raw, engine, cache_dir, wasm_file):
    if backend == "wasmtime":
        return _wasm_call(engine, raw, wasm_file)
    if backend == "native":
        candidate = NativeSpecialization(raw, cache_dir=cache_dir)
        if candidate.cache_hit != (wasm_file is not None):
            raise AssertionError("Unexpected native cache state")
    else:
        candidate = LlvmSpecialization(raw, optimization_level=_LEVELS[backend])
    return partial(_native_copy, candidate)


def _execute(setup, schedule, inputs):
    invoke = setup()
    results = {}
    for size in schedule:
        results[size] = invoke(inputs[size])
    return results


def run(root, calibration, evaluation, repeats):
    source = git_identity(root)
    model = evaluate(calibration, evaluation)
    engine = wasmtime.Engine()
    _llvm()
    names = {row["name"] for row in evaluation["rows"] if row["split"] == "heldout"}
    programs = {work.name: work.program for work in workloads(scaling=True) if work.name in names}
    if set(programs) != names:
        raise ValueError("Held-out programs do not match the replay corpus")
    corpus_hashes = {row["name"]: row["wasm_sha256"] for row in evaluation["rows"]
                     if row["name"] in names}
    rows = []
    with tempfile.TemporaryDirectory(prefix="visuaml-tier-replay-") as directory:
        directory = Path(directory)
        cache = directory / "cached-cpp"
        wasm_file = directory / "cached-wasm"
        cold_serial = 0
        for name, program in sorted(programs.items()):
            raw = encode_control(program)
            cached = NativeSpecialization(raw, cache_dir=cache)
            if cached.source_sha256 != corpus_hashes[name] or cached.cache_hit:
                raise ValueError("Replay source identity or cache seed disagrees")
            wasm_file.write_bytes(wasmtime.Module(engine, _wrapped_source(raw)).serialize())
            store = wasmtime.Store(engine)
            scalar = wasmtime.Instance(store, wasmtime.Module(engine, raw), []).exports(store)["run"]
            values = np.random.default_rng(61).integers(-(1 << 31), 1 << 31, 16384, dtype=np.int32)
            values[:5] = (-(1 << 31), -1, 0, 1, (1 << 31) - 1)
            expected = np.fromiter((scalar(store, int(value)) for value in values), dtype=np.int32)
            for scenario in model["scenarios"]:
                cache_hit = scenario["cached_cpp_and_wasmtime"]
                schedule = [int(size) for size, calls in scenario["batch_counts"].items()
                            for _ in range(calls)]
                random.Random(61).shuffle(schedule)
                inputs = {size: values[:size] for size in set(schedule)}
                samples = {backend: [] for backend in _BACKENDS}
                for trial in range(repeats):
                    for backend in _trial_order(_BACKENDS, trial + len(rows)):
                        cache_dir = cache
                        if not cache_hit and backend == "native":
                            cache_dir = directory / f"cold-cpp-{cold_serial}"
                            cold_serial += 1
                        preload = wasm_file if cache_hit else None
                        setup = partial(_prepare, backend, raw, engine, cache_dir, preload)
                        outputs, elapsed = _timed(lambda: _execute(setup, schedule, inputs))
                        for size, output in outputs.items():
                            np.testing.assert_array_equal(output, expected[:size])
                        samples[backend].append(elapsed)
                timings = {backend: summarize_ns(times) for backend, times in samples.items()}
                best = min(timings, key=lambda backend: timings[backend]["median_ns"])
                chosen = scenario["chosen_backend"]
                rows.append({
                    "program": name, "scenario": scenario["scenario"], "cached_cpp_and_wasmtime": cache_hit,
                    "selected_backend": chosen, "fastest_measured": best, "timings": timings,
                    "regret_factor": timings[chosen]["median_ns"] / timings[best]["median_ns"],
                })
    summary = {}
    for scenario in model["scenarios"]:
        name = scenario["scenario"]
        cached = scenario["cached_cpp_and_wasmtime"]
        group = [row for row in rows if row["scenario"] == name and row["cached_cpp_and_wasmtime"] == cached]
        summary[f"{'cached' if cached else 'cold'}_{name}"] = {
            "selected_backend": scenario["chosen_backend"],
            "median_regret": statistics.median(row["regret_factor"] for row in group),
            "worst_regret": max(row["regret_factor"] for row in group),
            "selected_wins": sum(row["selected_backend"] == row["fastest_measured"] for row in group),
        }
    return {
        "schema": "visuaml.llvm-tier-heldout-replay.v58", "source": source, "environment": environment(),
        "timing_source": calibration["source"], "repeats": repeats, "programs": len(programs),
        "summary": summary, "rows": rows,
        "scope": "Measured setup and copied workload execution. Runtime initialization excluded. "
                 "Final result per requested batch size checked against original scalar Wasmtime.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    if args.repeats < 1 or output.exists() or output.is_relative_to(root):
        parser.error("Use positive repeats and a fresh output path outside the checkout")
    record = run(root, json.loads(args.calibration.read_bytes()),
                 json.loads(args.evaluation.read_bytes()), args.repeats)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(record, stream, sort_keys=True)
        stream.write("\n")
    print(json.dumps(record["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
