"""Derive a reproducible, non-fabricated v58 performance scorecard.

Accepts only a completed source-bound benchmark artifact produced by the
v58 runner. Correctness gates are prerequisites, not weighted scoreboard
features; different engine coverage is never interpreted as missing = zero.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

SCHEMA = "visuaml.neural-wasm-benchmark.v58"


def load(path: Path) -> dict:
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get("schema") != SCHEMA:
        raise ValueError("Not a v58 benchmark artifact")
    if not isinstance(data.get("rows"), list) or not data["rows"]:
        raise ValueError("Benchmark record has no verified workload rows")
    if data.get("inventory", {}).get("programs") != len(data["rows"]):
        raise ValueError("Benchmark inventory is inconsistent with case records")
    data["_record_sha256"] = hashlib.sha256(raw).hexdigest()
    return data


def geometric(values: list[float]) -> float:
    if not values or any(not math.isfinite(x) or x <= 0. for x in values):
        raise ValueError("Cannot aggregate invalid speedups")
    return math.exp(statistics.mean(math.log(value) for value in values))


def summarize(record: dict, *, baseline: str = "alu_v56") -> dict:
    if baseline not in record.get("engines", []):
        raise ValueError("Unknown baseline engine")
    paired = defaultdict(list)
    family_pairs = defaultdict(list)
    normalized = defaultdict(list)
    missing = defaultdict(int)
    actual_cases = 0
    distinct_sources = set()
    for row in record["rows"]:
        source = row["source_wasm_sha256"]
        if source in distinct_sources:
            raise ValueError("Repeated source WASM in benchmark inventory")
        distinct_sources.add(source)
        family = row["family"]
        for case in row["cases"]:
            actual_cases += 1
            steps = case["steps"]
            if not isinstance(steps, int) or steps <= 0:
                raise ValueError("Invalid instruction count")
            engines = case["latencies"]
            expected = case["expected_i32"]
            if not isinstance(expected, int):
                raise ValueError("Missing independent integer result")
            for engine, metrics in engines.items():
                median = metrics["wall"]["median_ns"]
                if median <= 0:
                    raise ValueError("Invalid nonpositive latency")
                normalized[engine].append(median / steps)
            baseline_metric = engines.get(baseline)
            if baseline_metric is None:
                raise ValueError("Missing baseline on admitted workload")
            b = baseline_metric["wall"]["median_ns"]
            for engine in record["engines"]:
                if engine == baseline:
                    continue
                if engine not in engines:
                    missing[engine] += 1
                    continue
                ratio = b / engines[engine]["wall"]["median_ns"]
                paired[engine].append(ratio)
                family_pairs[(engine, family)].append(ratio)
    if actual_cases != record["inventory"]["witnesses"]:
        raise ValueError("Benchmark witness inventory count mismatch")
    result = {}
    for engine in record["engines"]:
        if engine == baseline:
            continue
        ratios = paired[engine]
        result[engine] = {
            "paired_witnesses": len(ratios),
            "missing_engine_witnesses": missing[engine],
            "geometric_mean_speedup_vs_baseline": (
                geometric(ratios) if ratios else None
            ),
            "median_speedup_vs_baseline": (
                statistics.median(ratios) if ratios else None
            ),
            "worst_slowdown_factor_vs_baseline": (
                max(1. / ratio for ratio in ratios) if ratios else None
            ),
            "fraction_slower_than_baseline": (
                sum(ratio < 1. for ratio in ratios) / len(ratios)
                if ratios else None
            ),
            "median_wall_ns_per_instruction": (
                statistics.median(normalized[engine])
                if normalized[engine] else None
            ),
            "by_family": {
                family: {
                    "paired_witnesses": len(values),
                    "geometric_mean_speedup": geometric(values),
                }
                for (name, family), values in sorted(family_pairs.items())
                if name == engine
            },
        }
    return {
        "schema": "visuaml.neural-wasm-benchmark-scorecard.v58",
        "source_benchmark_sha256": record["_record_sha256"],
        "source_git_commit": record["source"]["commit"],
        "environment": record["environment"],
        "policy": record["policy"],
        "baseline": baseline,
        "inventory": record["inventory"],
        "results": result,
        "interpretation": (
            "Paired local Python/NumPy versus Wasmtime binding timings. "
            "Missing/unadmitted cases are counted, never converted to zeros; "
            "no hardware or all-WASM performance claim."
        ),
    }


def flatten(scorecard: dict) -> list[dict]:
    rows = []
    for engine, metrics in sorted(scorecard["results"].items()):
        rows.append({
            "engine": engine,
            "baseline": scorecard["baseline"],
            "paired_witnesses": metrics["paired_witnesses"],
            "missing_engine_witnesses": metrics["missing_engine_witnesses"],
            "geometric_mean_speedup": (
                metrics["geometric_mean_speedup_vs_baseline"]
            ),
            "median_speedup": metrics["median_speedup_vs_baseline"],
            "worst_slowdown_factor": (
                metrics["worst_slowdown_factor_vs_baseline"]
            ),
            "fraction_slower": metrics["fraction_slower_than_baseline"],
            "median_wall_ns_per_instruction": (
                metrics["median_wall_ns_per_instruction"]
            ),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--baseline", default="alu_v56")
    args = parser.parse_args()
    scorecard = summarize(load(args.input), baseline=args.baseline)
    for path in (args.output_json, args.output_csv):
        if path is not None and path.exists():
            parser.error(f"Refusing to overwrite: {path}")
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(scorecard, sort_keys=True, indent=2,
                       allow_nan=False) + "\n", encoding="utf-8",
        )
    rows = flatten(scorecard)
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({
        "source": scorecard["source_benchmark_sha256"],
        "baseline": scorecard["baseline"],
        "engines": {
            name: {
                "paired": result["paired_witnesses"],
                "missing": result["missing_engine_witnesses"],
                "geomean": result["geometric_mean_speedup_vs_baseline"],
            }
            for name, result in scorecard["results"].items()
        },
    }, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
