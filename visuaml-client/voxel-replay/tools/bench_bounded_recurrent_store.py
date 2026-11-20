"""Compare isolated byte operations and sparse Rust ticks at fixed capacities."""
import argparse
import json
import resource
import statistics
import subprocess
import sys
from functools import partial
from time import perf_counter_ns

import torch

from check_bounded_recurrent_store import OUT
from replay_frozen_recurrent import EVIDENCE, sha, wasmtime_run
from visuaml.compiler_lab.frozen_recurrent_v58 import initial_state, load_program, snapshot_page

CAPACITIES = (1, 16, 256)
SAMPLES = 31


def rss_kib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def measure(operations, repetitions):
    raw = {name: [] for name, _ in operations}
    for _, run in operations:
        for _ in range(repetitions):
            run()
    for sample in range(SAMPLES):
        order = operations if sample % 2 == 0 else reversed(operations)
        for name, run in order:
            start = perf_counter_ns()
            for _ in range(repetitions):
                run()
            raw[name].append((perf_counter_ns() - start) / repetitions)
    return {name: {"median_ns": statistics.median(values), "raw_ns": values}
            for name, values in raw.items()}


def primitives():
    result = {}
    for size in (4096, 65536, 1048576, 16777216):
        page = torch.zeros(size, dtype=torch.uint8)
        indices = torch.arange(size // 2 + 1, size // 2 + 5)
        value = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)
        operations = [
            ("gather", partial(torch.gather, page, 0, indices)),
            ("functional_scatter", partial(torch.scatter, page, 0, indices, value)),
            ("in_place_scatter", partial(page.scatter_, 0, indices, value)),
        ]
        result[size] = measure(operations, 64)
    return result


def worker(pages, bounded):
    path = (OUT if bounded else EVIDENCE) / "machine.pt"
    machine = torch.jit.load(str(path))
    raw = (OUT / f"passage-{pages}.wasm").read_bytes()
    image = load_program(raw, memory_pages=pages)
    capacity = 65536 * pages
    cell = capacity - 8
    baseline_rss = rss_kib()
    edit = initial_state(cell, bytes(capacity), 100, memory_pages=image.memory_pages)
    quiet = initial_state(cell, bytes(capacity), 100, memory_pages=image.memory_pages)
    edit.page[cell + 4] = 1

    def run(template):
        state = template
        while int(state.status) == 0:
            state = machine(image.rows, state).state
        return state

    def clear_material():
        edit.page[cell] = 7
        return run(edit)

    def observe():
        return run(quiet)

    timings = measure([("edit_tick", clear_material), ("observe_tick", observe)], 8)
    tick_peak_rss = rss_kib()
    published = measure([("full_snapshot", partial(snapshot_page, edit))], 8)
    snapshot_peak_rss = rss_kib()

    # Feed the edit's final page into the next tick, changing only the request.
    clear_material_output = clear_material()
    edit_page = snapshot_page(clear_material_output)
    before = bytearray(capacity)
    before[cell], before[cell + 4] = 7, 1
    assert wasmtime_run(raw, cell, before) == (0, edit_page)
    opened = bytearray(edit_page)
    opened[cell + 4] = 0
    closed = opened.copy()
    closed[cell] = 7
    after = []
    for page, expected_material in ((opened, 0), (closed, 7)):
        stopped = run(initial_state(cell, page, 100, memory_pages=image.memory_pages))
        assert int(stopped.status) == 1 and int(stopped.stack[0]) == expected_material
        assert wasmtime_run(raw, cell, page) == (expected_material, snapshot_page(stopped))
        after.append({"material": expected_material, "page_sha256": sha(page),
                      "source_steps": 100 - int(stopped.fuel)})
    return {"capacity_bytes": capacity, "binary_sha256": image.binary_sha256,
            "image_sha256": image.image_sha256, "machine_sha256": sha(path.read_bytes()),
            "edit_address": cell, "edit_source_steps": 100 - int(clear_material_output.fuel),
            "edit_page_sha256": sha(edit_page), "observe": after,
            "bytes_copied_from_old_page_per_store": 0 if bounded else capacity,
            "guest_bytes_written_per_store": 4, "bytes_copied_per_full_snapshot": capacity,
            "baseline_peak_rss_kib": baseline_rss, "tick_peak_rss_kib": tick_peak_rss,
            "snapshot_peak_rss_kib": snapshot_peak_rss, **timings, **published}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=2, metavar=("PAGES", "MODE"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.worker:
        pages, mode = args.worker
        print(json.dumps(worker(int(pages), mode == "bounded")))
        return
    ticks = []
    for pages in CAPACITIES:
        pair = {
            mode: json.loads(subprocess.check_output(
                [sys.executable, __file__, "--worker", str(pages), mode], text=True))
            for mode in ("reference", "bounded")
        }
        reference, bounded = pair["reference"], pair["bounded"]
        for key in ("binary_sha256", "image_sha256", "edit_source_steps", "edit_page_sha256", "observe"):
            assert reference[key] == bounded[key]
        ticks.append(pair)
    print(json.dumps({"torch": torch.__version__, "threads": torch.get_num_threads(),
                      "samples": SAMPLES, "primitive_calls_per_sample": 64,
                      "guest_calls_per_sample": 8, "primitive": primitives(), "guest": ticks}, indent=2))


if __name__ == "__main__":
    main()
