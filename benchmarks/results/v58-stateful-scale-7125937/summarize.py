"""Verify two stateful scale replays and summarize their measured cases."""
import gzip
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])


def load(number):
    path = root / f"runtime-{number}.json"
    if path.exists():
        return json.loads(path.read_bytes())
    return json.loads(gzip.decompress(path.with_suffix(".json.gz").read_bytes()))


def identity(record):
    return {
        **{key: record[key] for key in ("steps_total", "steps_by_family", "cases_by_family", "sudoku_fuel_trap")},
        "programs": {name: {key: value for key, value in program.items() if key != "preparation_ns"}
                     for name, program in record["programs"].items()},
        "cases": [{key: value for key, value in row.items() if not key.endswith("_wall_ns")}
                  for row in record["cases"]],
    }


def summary(record):
    rows = record["cases"]
    assert sum(row["steps"] for row in rows) == record["steps_total"]
    assert len(rows) == sum(record["cases_by_family"].values())
    assert len({(row["family"], row["input_sha256"], row["argument"]) for row in rows}) == len(rows)
    sizes = {}
    for name in record["programs"]:
        data = [row for row in rows if row["family"] == name]
        sizes[name] = {
            "cases": len(data), "instructions": record["programs"][name]["instructions"],
            "steps": sum(row["steps"] for row in data),
            "steps_max": max(row["steps"] for row in data),
            **{f"{arm}_median_ns": statistics.median(row[f"{arm}_wall_ns"] for row in data)
               for arm in ("packed", "wasmtime")},
            **{f"{arm}_total_ns": sum(row[f"{arm}_wall_ns"] for row in data)
               for arm in ("packed", "wasmtime")},
        }
    by_name = {row["name"]: row for row in rows}
    assert by_name["matrix_3"]["output_sha256"] == by_name["matrix_unrolled_32"]["output_sha256"]
    showcase = {
        name: {key: by_name[name][key] for key in ("steps", "result", "packed_wall_ns", "wasmtime_wall_ns")}
        for name in ("classic", "norvig_grid", "escargot", "euler_grid", "classic_unsatisfiable",
                     "matrix_3", "matrix_unrolled_32")
    }
    return {"steps": record["steps_total"], "cases": len(rows), "families": sizes, "showcase": showcase}


first, second = load(1), load(2)
assert first["source"] == second["source"]
assert identity(first) == identity(second)
print(json.dumps({"run_1": summary(first), "run_2": summary(second)}, indent=2, sort_keys=True))
