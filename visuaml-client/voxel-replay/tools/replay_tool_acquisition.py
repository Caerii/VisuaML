"""Evaluate a frozen registry-aware controller and reopen its evidence."""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import torch

from composed_inference import walk
from frozen_call_checkpoint import digest
from replay_composed_inference import image
from replay_internal_tool import oracle

OUT = Path(__file__).resolve().parents[1] / "evidence/tool-acquisition"


def invoke(graph, vocabulary, case, programs):
    ids = {word: index for index, word in enumerate(vocabulary)}
    words = case["prompt"].split()
    unknown = sorted(set(words) - ids.keys())
    if unknown:
        return {"failure": {"category": "input_vocabulary", "event": 0, "tokens": unknown}}, None
    registry = case["registry"]
    code = [programs[entry[1]] for entry in registry]
    names = [entry[0] for entry in registry]
    roles = [0 if family == "offset" else 1 for _, family in registry]
    fuel = [case.get("fuel", {}).get(family, 48) for _, family in registry]
    rows = max(map(len, code))
    images = torch.stack([torch.nn.functional.pad(image(tool), (0, 0, 0, rows - len(tool)))
                          for tool in code])
    before = images.clone()
    inputs = (torch.tensor([ids[word] for word in words]), images, torch.tensor(roles),
              torch.tensor(fuel), case.get("override", -1), torch.tensor([ids[name] for name in names]))
    start = perf_counter_ns()
    requests, feedback, traces, outcome = graph(*inputs)
    elapsed = perf_counter_ns() - start
    assert torch.equal(before, images)
    events = []
    selection, binding, exact, feedback_checks = [], [], [], []
    first = None

    def check(category, passed, expected):
        nonlocal first
        if not passed and first is None:
            first = dict(category=category, event=len(events), expected=expected,
                         actual=events[-1] if events else None)

    expected_calls = case["calls"]
    for turn, row in enumerate(requests.tolist()):
        tokens = [vocabulary[token] for token in row if token >= 0]
        if not tokens:
            break
        events.append(dict(kind="request", tokens=tokens))
        if turn < len(expected_calls):
            expected = expected_calls[turn]
            chosen = len(tokens) == 5 and tokens[:2] == ["call", expected[0]]
            bound = len(tokens) == 5 and tokens[2:4] == list(map(str, expected[1:3]))
            selection.append(chosen)
            binding.append(bound)
            check("tool_selection", chosen, expected[0])
            check("argument_binding", bound, expected[1:3])
        if turn == 2 or feedback[turn, 0] < 0:
            continue
        status, raw, reported, steps = feedback[turn].tolist()
        trace = traces[turn, :steps].tolist()
        events.append(dict(kind="feedback", status=vocabulary[status], computed=raw,
                           reported=reported, trace=trace))
        slot = names.index(tokens[1]) if tokens[1] in names else -1
        if steps:
            expected_trace = oracle(code[slot], list(map(int, tokens[2:4])))[:steps]
            right_result = status != 15 or raw == expected_trace[-1][4]
            exact.append(trace == expected_trace and right_result)
            check("execution", exact[-1], expected_trace)
        else:
            assert not traces[turn].any()
        if turn < len(expected_calls):
            expected_feedback = expected_calls[turn][3]
            actual = reported if status == 15 else vocabulary[status]
            feedback_checks.append(actual == expected_feedback)
            check("feedback", feedback_checks[-1], expected_feedback)
    status, answer = outcome.tolist()
    events.append(dict(kind="outcome", status=status, answer=answer))
    correct = (status, answer) == ((0, case["answer"]) if "answer" in case else (1, -1))
    check("result_use", correct, case.get("answer", "stop"))
    check("tool_selection", len(selection) == len(expected_calls), expected_calls)
    check("unexpected_call", sum(event["kind"] == "feedback" for event in events) <= len(expected_calls),
          len(expected_calls))
    return dict(events=events, selection=selection, binding=binding, exact_execution=exact,
                feedback_correct=feedback_checks, correct_outcome=correct, failure=first), elapsed


def replay():
    torch.set_num_threads(1)
    artifact = OUT / "inference.pt"
    model = torch.jit.load(str(artifact))
    kinds = {node.kind() for node in walk(model.inlined_graph)}
    assert not {"prim::PythonOp", "prim::CallMethod"} & kinds
    spec = json.loads((OUT / "holdout.json").read_text())
    vocabulary = json.loads((OUT / "training.json").read_text())["vocabulary"]
    records, timings = {}, {}
    for name, case in spec["cases"].items():
        records[name], elapsed = invoke(model, vocabulary, case, spec["programs"])
        timings[name] = []
        if elapsed is not None:
            timings[name].append(elapsed)
            for _ in range(5):
                again, elapsed = invoke(model, vocabulary, case, spec["programs"])
                assert again == records[name]
                timings[name].append(elapsed)
    left, right = records["original"]["events"], records["substitution"]["events"]
    first = next((index + 1 for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]), None)
    same_first_execution = (len(left) > 2 and len(right) > 2 and left[0] == right[0]
                            and left[1]["kind"] == right[1]["kind"] == "feedback"
                            and left[1]["computed"] == right[1]["computed"]
                            and left[1]["trace"] == right[1]["trace"])
    return dict(format_version=1, profile="visuaml-tool-acquisition-v1", holdout=spec,
                graph_sha256=digest(artifact.read_bytes()),
                controller_sha256=digest((OUT / "controller.pt").read_bytes()),
                cases=records, first_intervention_difference=first,
                same_first_execution=same_first_execution), timings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--open", type=Path)
    options = parser.parse_args()
    with torch.inference_mode():
        evidence, samples = replay()
    if options.write:
        for filename, data in (("replay.json", evidence), ("timings.json", samples)):
            (OUT / filename).write_text(json.dumps(data, indent=2) + "\n")
    else:
        assert evidence == json.loads((options.open or OUT / "replay.json").read_text())
    for name, result in evidence["cases"].items():
        print(name, result["failure"])
