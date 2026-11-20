"""Isolate the chooser from arithmetic-template coverage on terrain."""
import json
from unittest.mock import patch

import torch

import terrain_verify as terrain
from replay_rule_portfolio import EXECUTOR, OUT, PAIR, PRIOR, benchmark, check_rules, greedy, lower, prepare
from replay_frozen_recurrent import PACKED
from visuaml.compiler_lab.frozen_calls_v58 import load_call_image


def main():
    torch.set_num_threads(1)
    check_rules()
    baseline = torch.jit.load(str(EXECUTOR / "machine.pt"))
    graph = torch.jit.load(str(OUT / "machine.pt"))
    planners = {1: (PAIR / "planner.wasm").read_bytes(), 2: (PRIOR / "load-planner.wasm").read_bytes()}
    chooser = (OUT / "chooser.wasm").read_bytes()
    raw = terrain.GUEST.read_bytes()
    image = load_call_image(raw)
    profile = json.loads((OUT / "training.json").read_text())["terrain"]
    counts = {int(pc): frequency for pc, frequency in profile["pc_counts"].items()}
    scene = terrain.scale_session(8, False, 25)
    page = terrain.initial_page(scene["initial"])
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        images, candidates, picked, _ = prepare(image, baseline, planners, chooser, counts)
        earliest = greedy(candidates)
        images["earliest"] = lower(image, candidates, earliest)
        warm = benchmark({"earliest": graph, "portfolio": graph}, images, raw, terrain.BASE, page,
                         [tick["moves"] for tick in scene["inputs"]], 1650)
    print(json.dumps({"earliest": earliest, "guest_selected": picked, "timings": warm}, indent=2))


if __name__ == "__main__":
    main()
