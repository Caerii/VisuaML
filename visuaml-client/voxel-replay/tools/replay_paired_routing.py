"""Open the paired-routing proof bundle and independently score every event."""
import argparse
import json
import math
from pathlib import Path
from time import perf_counter_ns

import torch

from composed_inference import walk
from frozen_call_checkpoint import digest
from paired_routing import OUT, description_ids
from replay_composed_inference import image
from replay_internal_tool import oracle


def evaluate(model, case, programs):
    if "prompt" in case:
        return dict(admitted=False, first_failure=dict(event=0, category="structured_input_required")), []
    entries = case["registry"]
    ids = [entry["handle"] for entry in entries]
    codes = [programs[entry["family"]] for entry in entries]
    length = max(map(len, codes))
    tool_images = torch.stack([torch.nn.functional.pad(image(code), (0, 0, 0, length - len(code)))
                               for code in codes])
    prior = tool_images.clone()
    args = (torch.tensor([description_ids(entry["description"]) for entry in entries]),
            torch.tensor(ids), torch.tensor([entry["schema"] for entry in entries]), tool_images,
            torch.tensor([entry["fuel"] for entry in entries]),
            torch.tensor([description_ids(text) for text in case["choices"]]),
            case["seed"], case["repeat"], case["threshold"], case["relation"],
            -1 if case["override"] is None else case["override"],
            -1 if case["forced_handle"] is None else case["forced_handle"])
    samples = []
    for _ in range(6):
        start = perf_counter_ns()
        calls, traces, scores = model(*args)
        samples.append(perf_counter_ns() - start)
    assert torch.equal(tool_images, prior)
    events, first_failure = [], None
    for turn, call in enumerate(calls.tolist()):
        handle, value, count, status, computed, reported, steps = call
        if handle < 0:
            continue
        expected = case["expected"][turn] if turn < len(case["expected"]) else None
        trace = traces[turn, :steps].tolist()
        score_rows = sorted((ident, float(score)) for ident, score in zip(ids, scores[turn])
                            if math.isfinite(float(score)))
        execution_matches = None
        if steps:
            code = codes[ids.index(handle)]
            reference = oracle(code, [value, count])[:steps]
            execution_matches = trace == reference and (computed < 0 or computed == reference[-1][4])
        else:
            assert not traces[turn].any()
        chosen, bound, feedback = (False, False, False) if expected is None else (
            handle == expected[0], [value, count] == expected[1:3], call[3:6] == expected[3:6])
        event = dict(call=call, trace=trace, scores_by_handle=score_rows, expected=expected,
                     tool_selection=chosen, argument_binding=bound,
                     execution=execution_matches, feedback=feedback)
        events.append(event)
        if first_failure is None:
            tests = [("tool_selection", chosen), ("argument_binding", bound),
                     ("exact_execution", execution_matches is not False), ("feedback", feedback)]
            for category, good in tests:
                if not good:
                    first_failure = dict(event=turn + 1, category=category)
                    break
    if first_failure is None and len(events) != len(case["expected"]):
        first_failure = dict(event=len(events) + 1, category="missing_request")
    correct_outcome = (len(events) == len(case["expected"]) and
                       events[-1]["call"][3:6] == case["expected"][-1][3:6])
    result = dict(admitted=True, events=events, first_failure=first_failure,
                  correct_outcome=correct_outcome)
    return result, samples


def replay():
    torch.set_num_threads(1)
    path = OUT / "inference.pt"
    model = torch.jit.load(str(path))
    forbidden = {"prim::PythonOp", "prim::CallMethod"}
    assert not forbidden.intersection(node.kind() for node in walk(model.inlined_graph))
    spec = json.loads((OUT / "holdout.json").read_text())
    results, samples, sweep = {}, {}, {}
    for name, case in spec["cases"].items():
        results[name], samples[name] = evaluate(model, case, spec["programs"])
    for value, case in spec["sweep"].items():
        result, _ = evaluate(model, case, spec["programs"])
        sweep[value] = result
    for category in ("original", "substitute_at_boundary", "reorder", "disabled", "invalid_argument"):
        assert results[category]["first_failure"] is None, results[category]
    original = results["original"]["events"]
    changed = results["substitute_at_boundary"]["events"]
    assert original[0]["trace"] == changed[0]["trace"]
    assert original[0]["call"][4] == changed[0]["call"][4]
    assert original[1]["call"][0] != changed[1]["call"][0]
    assert original == results["reorder"]["events"]
    return dict(format_version=1, profile="visuaml-paired-routing-v1", graph_sha256=digest(path.read_bytes()), holdout=spec,
                cases=results, sweep=sweep), samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--open", type=Path)
    args = parser.parse_args()
    with torch.inference_mode():
        replay_bundle, samples = replay()
    if args.write:
        for name, data in (("replay", replay_bundle), ("timings", samples)):
            (OUT / (name + ".json")).write_text(json.dumps(data, indent=2) + "\n")
    else:
        expected = json.loads((args.open or OUT / "replay.json").read_text())
        # Graph float scores are identical on the audited CPU build.
        assert json.loads(json.dumps(replay_bundle)) == expected
    for name, case in replay_bundle["cases"].items():
        print(name, case["first_failure"] or "matched")


if __name__ == "__main__":
    main()
