"""Isolate feedback, text order, and physical registry order in the frozen baseline."""
import argparse
from itertools import permutations
import json
from pathlib import Path

import torch

from frozen_call_checkpoint import digest
from replay_tool_acquisition import OUT as BASELINE, invoke
from replay_internal_tool import oracle

OUTPUT = BASELINE.parent / "frozen-routing-diagnostic" / "replay.json"


def replay():
    spec = json.loads((BASELINE / "holdout.json").read_text())
    vocabulary = json.loads((BASELINE / "training.json").read_text())["vocabulary"]
    ids = {word: index for index, word in enumerate(vocabulary)}
    graph_path, controller_path = BASELINE / "inference.pt", BASELINE / "controller.pt"
    graph = torch.jit.load(str(graph_path))
    controller = torch.jit.load(str(controller_path))
    original = spec["cases"]["original"]
    programs = spec["programs"]
    baseline, _ = invoke(graph, vocabulary, original, programs)
    sweep = []
    for value in range(32):
        role = "epsilon" if value < 12 else "zeta"
        expected = [["delta", 3, 1, value], [role, value, 5, value + 10 if value < 12 else value - 10]]
        observed, _ = invoke(graph, vocabulary, dict(original, override=value, calls=expected,
                             answer=expected[1][3]), programs)
        prefix = original["prompt"] + f" call delta 3 1 . result {value} assistant call"
        logits = controller(torch.tensor([[ids[word] for word in prefix.split()]]))[0, -1]
        sweep.append(dict(value=value, expected=expected, observed=observed,
                          logits={name: round(float(logits[ids[name]]), 6)
                                  for name in ("delta", "epsilon", "zeta")}))
    text_order, physical_order = [], []
    header, tail = original["prompt"].split(" please ", 1)
    clauses = header.removeprefix("available ").removesuffix(" .").split(" . ")
    for order in permutations(range(3)):
        prompt = "available " + " . ".join(clauses[i] for i in order) + " . please " + tail
        text_result, _ = invoke(graph, vocabulary, dict(original, prompt=prompt), programs)
        physical_result, _ = invoke(graph, vocabulary,
                                    dict(original, registry=[original["registry"][i] for i in order]), programs)
        text_order.append(dict(order=order, observed=text_result))
        physical_order.append(dict(order=order, exact_match=baseline == physical_result))
    assert all(row["exact_match"] for row in physical_order)
    return dict(profile="visuaml-frozen-routing-diagnostic-v1", graph_sha256=digest(graph_path.read_bytes()),
                controller_sha256=digest(controller_path.read_bytes()), prompt=original["prompt"],
                feedback_sweep=sweep, text_order=text_order, physical_order=physical_order,
                correct_handle_reference=dict(arguments=[15, 5], value=oracle(programs["trim"], [15, 5])[-1][4]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    options = parser.parse_args()
    torch.set_num_threads(1)
    with torch.inference_mode():
        data = json.dumps(replay(), indent=2) + "\n"
    if options.write:
        OUTPUT.write_text(data)
    else:
        assert OUTPUT.read_text() == data
    print("Verified 32 feedback overrides, six text orders, and six physical registry orders.")
