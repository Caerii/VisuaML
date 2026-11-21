"""Compare v82 capture, then test a large run with parallel worker writers."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from statistics import median
from subprocess import check_output
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from types import ModuleType

from visuaml.replay.request_tree import JournalWriter, JournalTree, compare_trees, link_trees, open_tree


def capture(path, start, end, changed=False, *, writer_class=JournalWriter):
    began = perf_counter_ns()
    with writer_class(path) as writer:
        for i in range(start, end):
            writer.write(f"{i:08}", "tool_result", 9 if changed and i == end-1 else 7, runtime="test")
        sealed = perf_counter_ns()
    return dict(capture_ns=sealed-began, publish_ns=perf_counter_ns()-sealed, bytes=path.stat().st_size)


def prior_writer():
    source = check_output(["git", "show", "225c0ce:visuaml-client/backend/visuaml/replay/request_tree.py"])
    module = ModuleType("visuaml.replay._baseline_writer")
    module.__package__ = "visuaml.replay"
    exec(compile(source, "v82 request_tree.py", "exec"), module.__dict__)
    return module.JournalWriter


def paired(root, rows):
    modes = {"v82": prior_writer(), "v83": JournalWriter}
    reports = {name: [] for name in modes}
    files = {name: root/f"{name}.sqlite" for name in modes}
    for repeat in range(3):
        order = tuple(modes) if not repeat % 2 else tuple(reversed(modes))
        for mode in order:
            reports[mode].append(capture(files[mode], 0, rows, writer_class=modes[mode]))
        with JournalTree(files["v82"]) as before, JournalTree(files["v83"]) as after:
            result = compare_trees(before, after)
            assert result["matched_observations"] == rows and result["visited_nodes"] == 1
    return reports


def large(root, millions, processes):
    paths = [root/f"worker-{i:03}.sqlite" for i in range(millions)]
    began = perf_counter_ns()
    with ProcessPoolExecutor(max_workers=processes) as pool:
        jobs = [pool.submit(capture, path, i*1000000, (i+1)*1000000) for i, path in enumerate(paths)]
        captures = [job.result() for job in jobs]
    capture_ns = perf_counter_ns()-began
    alternate_path = root/"alternate.sqlite"
    alternate = capture(alternate_path, (millions-1)*1000000, millions*1000000, True)
    reference, candidate = root/"baseline.json", root/"candidate.json"
    started = perf_counter_ns()
    link_trees(paths, reference, partition_bytes=2)
    link_trees([*paths[:-1], alternate_path], candidate, partition_bytes=2)
    link_ns = perf_counter_ns()-started
    samples = []
    with open_tree(reference) as left, open_tree(candidate) as right:
        for _ in range(11):
            began = perf_counter_ns()
            report = compare_trees(left, right)
            samples.append(perf_counter_ns()-began)
        assert report["matched_observations"] == millions*1000000-1
        mismatch = report["first_difference"]
        assert mismatch["reference"]["value"] == 7 and mismatch["candidate"]["value"] == 9
    return dict(rows=millions*1000000, processes=processes, shards=len(paths), capture_ns=capture_ns,
                captures=captures, alternate=alternate, link_ns=link_ns, compare_samples_ns=samples,
                compare_median_ns=median(samples), report=report)


def seeks(root):
    reports = {}
    for name, writer in (("v82", prior_writer()), ("v83", JournalWriter)):
        before, after = root/f"{name}-before", root/f"{name}-after"
        capture(before, 0, 100000, writer_class=writer)
        capture(after, 0, 100000, True, writer_class=writer)
        with JournalTree(before) as left, JournalTree(after) as right:
            samples = []
            for _ in range(51):
                started = perf_counter_ns()
                report = compare_trees(left, right)
                samples.append(perf_counter_ns()-started)
        assert report["matched_observations"] == 99999
        reports[name] = dict(median_ns=median(samples), samples_ns=samples, report=report)
    assert reports["v82"]["report"] == reports["v83"]["report"]
    return reports


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--millions", type=int, default=40)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--seek-only", action="store_true")
    args = parser.parse_args()
    if not 2 <= args.millions <= 100 or not 1 <= args.processes <= args.millions:
        parser.error("Select 2–100 million rows and a valid process count")
    with TemporaryDirectory() as directory:
        root = Path(directory)
        result = (seeks(root) if args.seek_only else
                  dict(paired=paired(root, 1000000), scale=large(root, args.millions, args.processes)))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
