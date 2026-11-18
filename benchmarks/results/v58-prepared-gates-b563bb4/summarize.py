"""Verify full archived trace identities and summarize paired timings."""
import gzip
import json
import math
import statistics
import sys
from pathlib import Path

archive = Path("benchmarks/results/v58-indexed-gates-7160363")
out = Path(sys.argv[1])
prior = json.loads(gzip.decompress((archive / "runtime-1.json.gz").read_bytes()))
faults = json.loads((out / "faults.json").read_bytes())
prior_faults = json.loads((archive / "faults.json").read_bytes())
assert faults["cases"] == prior_faults["cases"] or (
    {c["attack"] for c in faults["cases"] if c["detected"]}
    == {c["attack"] for c in prior_faults["cases"] if c["detected"]}
    and len(faults["cases"]) == len(prior_faults["cases"])
)

def cases(data):
    return {
        (row["program"], row["source_wasm_sha256"], row["microop_tokens_sha256"],
         row["route_tokens_sha256"], case["input"]): (
             row.get("program_preparation", {}).get("wall", {}).get("median_ns"),
             case,
         )
        for row in data["rows"] for case in row["cases"]
    }

def sample(run):
    current = cases(run)
    old = cases(prior)
    assert old.keys() == current.keys()
    assert run["fixed_weights"] == prior["fixed_weights"]
    for key, (_, case) in current.items():
        assert all(case[field] == old[key][1][field]
                   for field in ("result", "steps", "trace_sha256"))
    modes = {}
    for mode in ("trace", "no_trace"):
        before, after, ratios, thresholds, first_ratios = [], [], [], [], []
        for preparation, case in current.values():
            indexed = case["timing"][f"indexed_{mode}"]["wall"]["median_ns"]
            prepared = case["timing"][mode]["wall"]["median_ns"]
            before.append(indexed)
            after.append(prepared)
            ratios.append(indexed / prepared)
            if indexed > prepared:
                thresholds.append(math.ceil(preparation / (indexed - prepared)))
            first_ratios.append(indexed / (preparation + prepared))
        modes[mode] = {
            "case_medians_ns": [statistics.median(before), statistics.median(after)],
            "paired_geomean_speedup": statistics.geometric_mean(ratios),
            "faster_warm_cases": sum(r > 1 for r in ratios),
            "worst_warm_slowdown": 1 / min(ratios),
            "modeled_first_call_geomean_speedup": statistics.geometric_mean(first_ratios),
            "modeled_break_even_calls_median_among_faster_cases": statistics.median(thresholds),
            "modeled_break_even_by_calls": {n: sum(x <= n for x in thresholds)
                                             for n in (1, 2, 4, 8, 16)},
        }
    preparation = [row["program_preparation"]["wall"]["median_ns"] for row in run["rows"]]
    return {
        "preparation_ns_min_median_max": [min(preparation), statistics.median(preparation), max(preparation)],
        "programs": len(run["rows"]),
        "witnesses": len(current),
        "states": sum(case["steps"] for _, case in current.values()),
        "compiled_kernels": run["compiled_alu"],
        "modes": modes,
    }


def load_run(n):
    raw = out / f"runtime-{n}.json"
    if raw.exists():
        return json.loads(raw.read_bytes())
    return json.loads(gzip.decompress(raw.with_suffix(".json.gz").read_bytes()))


report = {f"runtime-{n}.json.gz": sample(load_run(n)) for n in (1, 2)}
print(json.dumps(report, indent=2, sort_keys=True))
