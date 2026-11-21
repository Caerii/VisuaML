"""Measure direct capture, first divergence, and fresh CLI on stable request ranges."""
import argparse
import json
from pathlib import Path
from statistics import median
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter_ns

from visuaml.replay.journal_index import build_index, compare_indexes
from visuaml.replay.shards import compare_shards
from visuaml.replay.request_tree import JournalTree, JournalWriter, compare_trees, link_trees, open_tree


def measure(size, directory):
    paths = {name: directory/f"{name}.sqlite" for name in ("reference", "late", "missing")}
    writes = {}
    for name, path in paths.items():
        started = perf_counter_ns()
        with JournalWriter(path) as writer:
            for index in range(100 if name == "missing" else 0, size):
                value = 9 if name == "late" and index == size-1 else 7
                writer.write(f"{index:08}", "tool_result", value, runtime="synthetic")
            before_close = perf_counter_ns()
        writes[name] = dict(write_ns=before_close-started, close_ns=perf_counter_ns()-before_close,
                            bytes=path.stat().st_size)
    return dict(measure_paths(size, paths), writes=writes)


def measure_paths(size, paths):
    results = {}
    with JournalTree(paths["reference"]) as reference:
        for name, expected in (("reference", None), ("late", size-1), ("missing", 0)):
            with JournalTree(paths[name]) as candidate:
                samples = []
                for _ in range(11):
                    started = perf_counter_ns()
                    result = compare_trees(reference, candidate)
                    samples.append(perf_counter_ns()-started)
                first = result["first_difference"]
                assert (None if first is None else first["index"]) == expected
                results[name] = dict(samples_ns=samples, median_ns=median(samples), result=result)
    fresh = []
    for _ in range(5):
        started = perf_counter_ns()
        process = subprocess.run([sys.executable, str(Path(__file__).with_name("visuaml_replay.py")),
                                  "compare-trees", str(paths["reference"]), str(paths["late"]),
                                  str(paths["missing"])], capture_output=True, text=True, check=True)
        fresh.append(perf_counter_ns()-started)
        assert [row["first_difference"]["index"] for row in json.loads(process.stdout)] == [size-1, 0]
    return dict(rows=size, comparisons=results, cli_two_candidate_samples_ns=fresh,
                cli_two_candidate_median_ns=median(fresh))


def legacy(size, directory):
    journals = [directory/name for name in ("left.jsonl", "right.jsonl")]
    indexes = [directory/name for name in ("left.vmi", "right.vmi")]
    started = perf_counter_ns()
    with journals[0].open("w") as left, journals[1].open("w") as right:
        for index in range(size):
            line = f'{{"request":"{index:08}","boundary":"tool_result","runtime":"synthetic","value":'
            left.write(line+"7}\n")
            right.write(line+("9" if index == size-1 else "7")+"}\n")
    write_ns = perf_counter_ns()-started
    started = perf_counter_ns()
    scan = compare_shards(journals[:1], journals[1:])
    scan_ns = perf_counter_ns()-started
    started = perf_counter_ns()
    for journal, index in zip(journals, indexes, strict=True):
        build_index([journal], index)
    build_ns = perf_counter_ns()-started
    samples = []
    for _ in range(11):
        started = perf_counter_ns()
        comparison = compare_indexes(*indexes)
        samples.append(perf_counter_ns()-started)
        assert comparison["first_difference"] == scan["first_difference"]
    return dict(rows=size, jsonl_write_ns=write_ns, scan_ns=scan_ns, index_build_ns=build_ns,
                index_median_ns=median(samples), index_samples_ns=samples,
                jsonl_bytes=sum(path.stat().st_size for path in journals),
                index_bytes=sum(path.stat().st_size for path in indexes), first_difference=scan["first_difference"])


def worker_layouts(directory):
    runs, manifests = [], []
    for width in (10, 5):
        paths = []
        started = perf_counter_ns()
        for worker in range(width):
            path = directory/f"{width}-{worker}.sqlite"
            paths.append(path)
            span = 1000000//width
            with JournalWriter(path) as writer:
                for row in range(worker*span, (worker+1)*span):
                    writer.write(f"{row:08}", "tool_result", 7, runtime="test")
        written = perf_counter_ns()
        manifest = directory/f"run-{width}.json"
        manifests.append(manifest)
        info = link_trees(paths, manifest, partition_bytes=3)
        runs.append(dict(write_ns=written-started, link_ns=perf_counter_ns()-written,
                         manifest_bytes=manifest.stat().st_size, **info))
    with open_tree(manifests[0]) as reference, open_tree(manifests[1]) as candidate:
        result = compare_trees(reference, candidate)
        assert result["matched_observations"] == 1000000 and result["visited_nodes"] == 1
    return dict(runs=runs, comparison=result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sizes", nargs="*", type=int, default=[1000000, 10000000])
    parser.add_argument("--legacy", action="store_true", help="Measure the JSONL scanner and flat block index")
    parser.add_argument("--reuse", type=Path, help="Retime existing reference, late, and missing .sqlite files")
    parser.add_argument("--workers", action="store_true", help="Test ten versus five worker files on one million rows")
    args = parser.parse_args()
    if args.workers:
        with TemporaryDirectory() as directory:
            print(json.dumps(worker_layouts(Path(directory))), flush=True)
        raise SystemExit(0)
    for size in args.sizes:
        if size <= 100:
            parser.error("Use more than 100 rows")
        if args.reuse:
            paths = {name: args.reuse/f"{name}.sqlite" for name in ("reference", "late", "missing")}
            print(json.dumps(measure_paths(size, paths)), flush=True)
        else:
            with TemporaryDirectory() as directory:
                run = legacy if args.legacy else measure
                print(json.dumps(run(size, Path(directory))), flush=True)
