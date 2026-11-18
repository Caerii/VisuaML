"""Estimate v58 backend choices from independent benchmark runs and held-out programs."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

_BACKENDS = ("wasmtime", "native", "llvm_o0", "llvm_o1", "llvm")
# Fixed counts per program and per known deployment lifetime.
_SCENARIOS = {
    "interactive": {"1": 32, "32": 12, "1024": 1},
    "moderate": {"32": 20, "1024": 120, "16384": 100},
    "bulk": {"1024": 400, "16384": 1200},
}


def _setup(row, backend, cached):
    if backend == "native":
        return row["native_reload_ns" if cached else "native_cold_ns"]
    if backend == "wasmtime":
        if cached:
            return row["wasmtime_reload_ns"]
        return sum(row[f"wasmtime_{stage}_ns"] for stage in ("wrapper", "compile", "instantiate"))
    return row[f"{backend}_compile_ns"]


def _cost(row, backend, cached, sizes):
    return _setup(row, backend, cached) + sum(
        calls * row["copied_batches"][size][backend]["median_ns"]
        for size, calls in sizes.items()
    )


def evaluate(calibration, evaluation):
    if calibration.get("llvm_optimization_levels") != {"llvm_o0": 0, "llvm_o1": 1, "llvm": 2}:
        raise ValueError("Calibration must cover all LLVM tiers")
    if evaluation.get("llvm_optimization_levels") != calibration["llvm_optimization_levels"]:
        raise ValueError("Evaluation must cover the same LLVM tiers")
    if calibration["source"] != evaluation["source"]:
        raise ValueError("Benchmark runs must share the same measured source")
    training = [row for row in calibration["rows"] if row["split"] in {"public", "scaling"}]
    heldout = [row for row in evaluation["rows"] if row["split"] == "heldout"]
    if not training or not heldout:
        raise ValueError("Expected nonempty calibration and held-out partitions")
    if {row["wasm_sha256"] for row in training} & {row["wasm_sha256"] for row in heldout}:
        raise ValueError("A WASM program appears in both partitions")

    report = []
    for cached in (False, True):
        for name, sizes in _SCENARIOS.items():
            training_costs = {
                backend: statistics.median(_cost(row, backend, cached, sizes) for row in training)
                for backend in _BACKENDS
            }
            chosen = min(training_costs, key=training_costs.get)
            samples = []
            for row in heldout:
                costs = {backend: _cost(row, backend, cached, sizes) for backend in _BACKENDS}
                best = min(costs, key=costs.get)
                samples.append({
                    "program": row["name"], "best_backend": best, "costs_ns": costs,
                    "regret_factor": costs[chosen] / costs[best],
                })
            report.append({
                "scenario": name, "cached_cpp_and_wasmtime": cached, "batch_counts": sizes,
                "chosen_backend": chosen, "training_median_costs_ns": training_costs,
                "heldout_oracle_choices": dict(Counter(row["best_backend"] for row in samples)),
                "median_heldout_regret": statistics.median(row["regret_factor"] for row in samples),
                "worst_heldout_regret": max(row["regret_factor"] for row in samples),
                "samples": samples,
            })
    return {"training_programs": len(training), "heldout_programs": len(heldout), "scenarios": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first, second = args.calibration.read_bytes(), args.evaluation.read_bytes()
    if first == second:
        parser.error("Use two independent timing runs")
    calibration, evaluation = json.loads(first), json.loads(second)
    report = {
        "schema": "visuaml.llvm-tier-selection-model.v58",
        "first_run_sha256": hashlib.sha256(first).hexdigest(),
        "second_run_sha256": hashlib.sha256(second).hexdigest(),
        "scope": "Estimated costs from measured per-program setup plus copied-batch medians. "
                 "Runtimes are initialized. Mixed traces were not timed end to end.",
        "first_train_second_test": evaluate(calibration, evaluation),
        "second_train_first_test": evaluate(evaluation, calibration),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
