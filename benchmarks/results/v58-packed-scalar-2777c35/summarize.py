"""Summarize matched packed scalar timing records."""
import gzip
import json
import statistics
import sys
from pathlib import Path

directory = Path(sys.argv[1])


def read_run(number):
    raw = directory / f"runtime-{number}.json"
    if raw.exists():
        return json.loads(raw.read_bytes())
    return json.loads(gzip.decompress(raw.with_suffix(".json.gz").read_bytes()))


def median(row, arm, timing):
    return row["timings"][arm][timing]["wall"]["median_ns"]


def summarize(record):
    assert (record["witnesses"], record["states"]) == (197, 3387)
    rows = record["rows"]
    grouped = []
    for size in sorted({row["size"] for row in rows}):
        for traced in (False, True):
            selected = [row for row in rows if row["size"] == size and row["traced"] == traced]
            timings = {}
            for name in ("warm", "with_preparation"):
                individual = [median(row, "float64", name) for row in selected]
                batch = [median(row, "packed", name) for row in selected]
                ratios = [before / after for before, after in zip(individual, batch, strict=True)]
                timings[name] = {
                    "paired_speedup": statistics.geometric_mean(ratios),
                    "program_speedups": dict(zip((row["program"] for row in selected), ratios, strict=True)),
                    "median_packed_ns": statistics.median(batch),
                    "median_float64_ns": statistics.median(individual),
                    "median_packed_inputs_per_second": size * 1e9 / statistics.median(batch),
                    "median_float64_inputs_per_second": size * 1e9 / statistics.median(individual),
                }
            grouped.append({"size": size, "traced": traced, "timings": timings})
    return grouped


print(json.dumps({str(n): summarize(read_run(n)) for n in (1, 2)}, indent=2, sort_keys=True))
