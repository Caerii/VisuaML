"""Verify archived trace identity and summarize paired native loop timings."""
import gzip
import json
import statistics
import sys
from pathlib import Path

reference = Path("benchmarks/results/v58-native-microstep-4e1cb86/runtime-1.json.gz")
output = Path(sys.argv[1])


def load(path):
    if path.exists():
        return json.loads(path.read_bytes())
    return json.loads(gzip.decompress(path.with_suffix(".json.gz").read_bytes()))


def index(record):
    return {
        (row["program"], row["source_wasm_sha256"], row["microop_tokens_sha256"],
         row["route_tokens_sha256"], case["input"]): case
        for row in record["rows"] for case in row["cases"]
    }


def median_wall(record):
    return record["wall"]["median_ns"]


def summarize(run, prior):
    cases = index(run)
    old = index(prior)
    assert cases.keys() == old.keys()
    assert run["fixed_weights"] == prior["fixed_weights"]
    assert run["native_loop"] and run["native_microstep"] and run["prepared"] and run["indexed_pc"]
    for key, case in cases.items():
        assert all(case[field] == old[key][field]
                   for field in ("result", "steps", "trace_sha256"))
    modes = {}
    for mode in ("trace", "no_trace"):
        pair = [(median_wall(case["timing"][f"microstep_{mode}"]), median_wall(case["timing"][mode]))
                for case in cases.values()]
        ratios = [before / after for before, after in pair]
        modes[mode] = {
            "microstep_median_ns": statistics.median(before for before, _ in pair),
            "loop_median_ns": statistics.median(after for _, after in pair),
            "paired_geomean_speedup": statistics.geometric_mean(ratios),
            "slowest_speedup": min(ratios),
            "faster_cases": sum(ratio > 1 for ratio in ratios),
        }
    return {
        "programs": len(run["rows"]),
        "witnesses": len(cases),
        "states": sum(case["steps"] for case in cases.values()),
        "median_preparation_ns": {
            key: statistics.median(median_wall(row[field]) for row in run["rows"])
            for key, field in (("microstep", "native_preparation"), ("loop", "loop_preparation"))
        },
        "modes": modes,
        "compiled_kernels": run["compiled_alu"],
    }


prior = json.loads(gzip.decompress(reference.read_bytes()))
results = {f"runtime-{n}.json.gz": summarize(load(output / f"runtime-{n}.json"), prior)
           for n in (1, 2)}
print(json.dumps(results, indent=2, sort_keys=True))
