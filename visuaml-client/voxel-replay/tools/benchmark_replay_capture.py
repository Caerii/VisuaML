"""Pair the observed and unobserved capture paths at the same source boundary."""
import json
from pathlib import Path
from statistics import median
import struct
from time import perf_counter_ns

import torch

from visuaml.replay import open_bundle


def main():
    torch.set_num_threads(1)
    path = Path(__file__).resolve().parents[1] / "release/examples/suspended-passage-report.json"
    replay, checkpoint = open_bundle(json.loads(path.read_text())["bundle"])
    page = bytearray(65536)
    struct.pack_into("<6i", page, 65512, 7, 1, 0, 0, -1, 0)
    page = bytes(page)
    observed = lambda state: replay.step(state)[0]
    methods = {"observed": observed, "unobserved": replay.advance}
    samples = {method: [] for method in methods}

    for repeat in range(33):
        order = list(methods)
        if repeat % 2:
            order.reverse()
        for method in order:
            state = replay.start(65512, page, 2000)
            start = perf_counter_ns()
            for _ in range(14):
                state = methods[method](state)
            assert replay.checkpoint(state) == checkpoint
            elapsed = perf_counter_ns() - start
            if repeat:
                samples[method].append(elapsed)

    result = dict(samples_ns=samples, median_ns={key: median(values) for key, values in samples.items()},
                  source_steps=14, serialized_states={"observed": 15, "unobserved": 1})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
