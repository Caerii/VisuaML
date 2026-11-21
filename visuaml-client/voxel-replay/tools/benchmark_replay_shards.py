"""Compare differently sharded journals with a mismatch on the final request."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns

from visuaml.replay.shards import compare_shards


def measure(directory, count, widths):
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
    return dict(rows_per_run=count, shard_counts=widths, comparison_ns=elapsed,
                input_bytes=sum(path.stat().st_size for group in paths for path in group),
                result=result, report_bytes=len(json.dumps(result).encode()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be positive")
    with TemporaryDirectory() as folder:
        result = measure(Path(folder), args.rows, (32, 48))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
