"""Compare differently sharded journals with a mismatch on the final request."""
import argparse
from contextlib import ExitStack
import json
from statistics import median
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns

from visuaml.replay.journal_index import build_index, compare_indexes
from visuaml.replay.shards import compare_shards


def measure(directory, count, widths, indexed=False):
    paths = [[directory / f"{side}-{index}.jsonl" for index in range(width)]
             for side, width in zip(("reference", "candidate"), widths, strict=True)]
    with ExitStack() as stack:
        files = [[stack.enter_context(path.open("w", encoding="utf-8")) for path in group] for group in paths]
        for index in range(count):
            request = f"{index:012}"
            for side, group in enumerate(files):
                out = group[index % len(group)]
                value = 9 if side and index == count-1 else 7
                out.write(f'{{"request":"{request}","boundary":"tool_result","runtime":"test","value":{value}}}\n')

    start = perf_counter_ns()
    result = compare_shards(*paths)
    elapsed = perf_counter_ns()-start
    assert result["matched_observations"] == count-1
    report = dict(rows_per_run=count, shard_counts=widths, comparison_ns=elapsed,
                  input_bytes=sum(path.stat().st_size for group in paths for path in group),
                  result=result, report_bytes=len(json.dumps(result).encode()))
    if indexed:
        indexes = [directory / f"{side}.vmi" for side in ("reference", "candidate")]
        build_ns, index_bytes = [], []
        for group, path in zip(paths, indexes, strict=True):
            start = perf_counter_ns()
            built = build_index(group, path)
            build_ns.append(perf_counter_ns()-start)
            index_bytes.append(built["bytes"])
        samples, cold_cli = [], []
        for _ in range(11):
            start = perf_counter_ns()
            fast = compare_indexes(*indexes)
            samples.append(perf_counter_ns()-start)
            assert fast["first_difference"] == result["first_difference"]
        for _ in range(5):
            start = perf_counter_ns()
            process = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("visuaml_replay.py")),
                 "compare-indexes", *map(str, indexes)],
                capture_output=True, text=True, check=True,
            )
            cold_cli.append(perf_counter_ns()-start)
            assert json.loads(process.stdout)["first_difference"] == fast["first_difference"]
        report["indexed"] = dict(build_ns=build_ns, index_bytes=index_bytes, samples_ns=samples,
                                 median_ns=median(samples), cli_samples_ns=cold_cli,
                                 cli_median_ns=median(cold_cli), result=fast)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--indexed", action="store_true")
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be positive")
    with TemporaryDirectory() as folder:
        result = measure(Path(folder), args.rows, (32, 48), indexed=args.indexed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
