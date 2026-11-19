"""Summarize and independently replay archived structured-control trace identities."""
import gzip
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

directory = Path(sys.argv[1])


def read(name):
    raw = directory / f"{name}.json"
    if raw.exists():
        return json.loads(raw.read_bytes())
    return json.loads(gzip.decompress(raw.with_suffix(".json.gz").read_bytes()))


def identity(record):
    return {
        (row["name"], row["source_wasm_sha256"], case["argument"]):
            {k: v for k, v in case.items() if k != "timings"}
        for row in record["rows"] for case in row["cases"]
    }


def run_summary(record):
    rows = []
    visited = Counter()
    states = fuel_traps = 0
    for fixture in record["rows"]:
        cases = fixture["cases"]
        times = {}
        for mode in ("packed_trace", "packed_untraced", "v57_trace"):
            times[mode] = statistics.median(case["timings"][mode]["wall"]["median_ns"] for case in cases)
        rows.append({
            "name": fixture["name"], "witnesses": len(cases),
            "trace_states": sum(case["steps"] for case in cases), "median_ns": times,
        })
        for case in cases:
            states += case["steps"]
            fuel_traps += len(case["fuel_traps"])
            visited.update(row["opcode"] for row in case["trace"])
    controls = ("block", "loop", "if", "else", "end", "br", "br_if", "return")
    assert all(visited[control] for control in controls)
    cases = [case for row in record["rows"] for case in row["cases"]]
    return {
        "witnesses": len(cases), "trace_states": states, "fuel_traps": fuel_traps,
        "control_opcode_visits": {key: visited[key] for key in controls}, "rows": rows,
        "paired_traced_geomean": statistics.geometric_mean(
            case["timings"]["v57_trace"]["wall"]["median_ns"]
            / case["timings"]["packed_trace"]["wall"]["median_ns"] for case in cases
        ),
    }


first, second = read("runtime-1"), read("runtime-2")
assert identity(first) == identity(second)
assert first["interventions"] == second["interventions"]
depths = read("depths")
assert depths["witnesses"] == 36
for opcode in ("br", "br_if"):
    assert {row["depth"] for row in depths["rows"] if row["opcode"] == opcode} == set(range(12))
print(json.dumps({"run_1": run_summary(first), "run_2": run_summary(second),
                  "branch_depth_witnesses": depths["witnesses"]}, sort_keys=True, indent=2))
