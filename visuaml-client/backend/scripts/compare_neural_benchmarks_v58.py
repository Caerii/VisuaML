"""Compare two local v58 benchmark evidence records without cherry-picking.

Requires identical original source-WASM/input inventory, and reports
environment mismatches. Does not imply improvements transfer to hardware
or to unadmitted WASM programs.

Examples:
 python compare_neural_benchmarks_v58.py --baseline old.json --candidate new.json --engine microop_v57
 python compare_neural_benchmarks_v58.py --baseline old.json --candidate new.json --engine microstep_v58 --require-compatible
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

SCHEMA = "visuaml.neural-wasm-benchmark.v58"
ENV_KEYS = ("platform", "machine", "processor", "cpu_count", "python",
            "numpy", "wasmtime", "thread_controls")
POLICY_KEYS = ("repeats", "warmups", "fuel", "holdout_programs")


def read(path: Path) -> dict:
    raw = path.read_bytes()
    report = json.loads(raw)
    if report.get("schema") != SCHEMA:
        raise ValueError(f"Unexpected benchmark schema: {path}")
    if not isinstance(report.get("rows"), list) or not report["rows"]:
        raise ValueError("Empty/incomplete benchmark inventory")
    report["_raw_sha256"] = hashlib.sha256(raw).hexdigest()
    return report


def inventory(report: dict) -> dict:
    pairs = {}
    for row in report["rows"]:
        key = row["name"]
        if key in pairs:
            raise ValueError(f"Duplicate program name: {key}")
        witness = {}
        for item in row["cases"]:
            argument = item["input_i32"]
            if argument in witness:
                raise ValueError(f"Duplicate case argument: {key} {argument}")
            witness[argument] = item
        pairs[key] = (row, witness)
    return pairs


def quantile(values: list[float], fraction: float) -> float:
    if not values or not 0. <= fraction <= 1.:
        raise ValueError("Invalid quantile request")
    a = sorted(values)
    index = (len(a) - 1) * fraction
    low = int(math.floor(index))
    high = int(math.ceil(index))
    return a[low] * (high - index) + a[high] * (index - low) if high != low else a[low]


def speedup_stats(ratios: list[float]) -> dict:
    if not ratios or any(not math.isfinite(x) or x <= 0. for x in ratios):
        raise ValueError("Nonpositive/nonfinite speedup ratio")
    return {
        "cases": len(ratios),
        "geometric_mean_speedup": math.exp(statistics.mean(
            math.log(x) for x in ratios
        )),
        "median_speedup": statistics.median(ratios),
        "p10_speedup": quantile(ratios, .10),
        "p90_speedup": quantile(ratios, .90),
        "worst_slowdown_factor": max(1. / x for x in ratios),
        "fraction_slower": sum(x < 1. for x in ratios) / len(ratios),
    }


def compare(
    baseline: dict, candidate: dict, *,
    engine: str,
    require_compatible: bool = False,
) -> dict:
    for data in (baseline, candidate):
        if engine not in data.get("engines", ()):
            raise ValueError(f"Benchmark engine not recognized: {engine}")
    mismatches = []
    for key in ENV_KEYS:
        if baseline["environment"].get(key) != candidate["environment"].get(key):
            mismatches.append(f"environment.{key}")
    for key in POLICY_KEYS:
        if baseline["policy"].get(key) != candidate["policy"].get(key):
            mismatches.append(f"policy.{key}")
    if mismatches and require_compatible:
        raise ValueError("Incompatible benchmark conditions: " + ", ".join(mismatches))

    left = inventory(baseline)
    right = inventory(candidate)
    if set(left) != set(right):
        raise ValueError("Program inventory differs; cannot compare")
    ratios = []
    families = {}
    pairs = []
    missing = []
    for name in sorted(left):
        old_row, old_inputs = left[name]
        new_row, new_inputs = right[name]
        for field in ("source_wasm_sha256", "split", "family"):
            if old_row[field] != new_row[field]:
                raise ValueError(f"Source/split mismatch: {name} {field}")
        if set(old_inputs) != set(new_inputs):
            raise ValueError(f"Input inventory differs for {name}")
        for argument in sorted(old_inputs):
            old, new = old_inputs[argument], new_inputs[argument]
            if (old["expected_i32"] != new["expected_i32"]
                    or old["steps"] != new["steps"]):
                raise ValueError(f"Semantic disagreement for {name}/{argument}")
            before = old.get("latencies", {}).get(engine)
            after = new.get("latencies", {}).get(engine)
            if before is None or after is None:
                missing.append((name, argument))
                continue
            a = before["wall"]["median_ns"]
            b = after["wall"]["median_ns"]
            if a <= 0 or b <= 0:
                raise ValueError("Timing with nonpositive median")
            ratio = a / b
            ratios.append(ratio)
            families.setdefault(old_row["family"], []).append(ratio)
            pairs.append({
                "program": name,
                "split": old_row["split"],
                "family": old_row["family"],
                "input_i32": argument,
                "baseline_median_ns": a,
                "candidate_median_ns": b,
                "speedup": ratio,
            })
    if not ratios:
        raise ValueError("No comparable engine measurements")
    return {
        "schema": "visuaml.neural-wasm-benchmark-comparison.v58",
        "baseline_sha256": baseline["_raw_sha256"],
        "candidate_sha256": candidate["_raw_sha256"],
        "engine": engine,
        "environment_compatible": not mismatches,
        "compatibility_mismatches": mismatches,
        "source_inventory_equal": True,
        "semantic_outputs_equal": True,
        "comparable_witnesses": len(ratios),
        "missing_engine_witnesses": len(missing),
        "all": speedup_stats(ratios),
        "families": {
            key: speedup_stats(value)
            for key, value in sorted(families.items())
        },
        "pairs": pairs,
        "interpretation": (
            "Paired interpreter timing on the recorded host. Uncontrolled "
            "frequency, thermal state, scheduling, and Python/NumPy overhead "
            "may dominate; do not call this a hardware or asymptotic speedup."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-compatible", action="store_true")
    args = parser.parse_args()
    report = compare(
        read(args.baseline), read(args.candidate),
        engine=args.engine, require_compatible=args.require_compatible,
    )
    if args.output is not None:
        output = args.output.resolve()
        if output.exists():
            parser.error(f"Refusing to overwrite comparison: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, sort_keys=True, indent=2,
                       allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({
        "engine": args.engine,
        "source_inventory_equal": report["source_inventory_equal"],
        "comparable_witnesses": report["comparable_witnesses"],
        "environment_compatible": report["environment_compatible"],
        "geometric_mean_speedup": report["all"]["geometric_mean_speedup"],
        "worst_slowdown_factor": report["all"]["worst_slowdown_factor"],
        "missing": report["missing_engine_witnesses"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
