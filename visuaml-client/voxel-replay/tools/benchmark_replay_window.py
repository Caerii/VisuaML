"""Pair late BFS windows on the reference and compiled seek paths."""
import json
from statistics import median
from time import perf_counter_ns
from unittest.mock import patch

import torch

from replay_frozen_recurrent import ARCHIVE, PACKED, graph_case
from visuaml.replay import Replay, inspect_window, make_bundle


def main():
    torch.set_num_threads(1)
    raw = (ARCHIVE / "v58-stateful-scale-7125937/bfs.wasm").read_bytes()
    argument, page, *_ = graph_case(0)
    samples = {"compiled": [], "reference": []}
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        bundle = make_bundle(Replay(raw), argument, page, 100000, 0)
        for pair in range(4):
            outcomes = {}
            order = tuple(samples) if pair % 2 == 0 else tuple(reversed(samples))
            for runner in order:
                started = perf_counter_ns()
                result = inspect_window(bundle, after=8192, limit=12, runner=runner)
                elapsed = perf_counter_ns()-started
                result.pop("seek_runner")
                outcomes[runner] = result
                if pair:
                    samples[runner].append(elapsed)
            assert outcomes["reference"] == outcomes["compiled"]
            assert len(outcomes["reference"]["trace"]) == 12
    print(json.dumps(dict(samples_ns=samples, median_ns={mode: median(rows) for mode, rows in samples.items()},
                          compared_window=outcomes["reference"]), indent=2))


if __name__ == "__main__":
    main()
