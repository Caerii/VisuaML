"""Check and reopen the composed-inference replay bundle."""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import torch

from composed_inference import OUT
from frozen_call_checkpoint import digest
from internal_tool import prepare_program
from replay_internal_tool import oracle
from replay_two_call_controller import Session, event_difference
from two_call_controller import VOCAB


def load(name):
    path = OUT / (name + ".pt")
    return torch.jit.load(str(path)), digest(path.read_bytes())


def image(rows):
    if any(op == 0 and operand != 0 for op, _, operand in rows):
        raise ValueError("Noncanonical halt")
    return prepare_program(rows)


def run(graph, vocabulary, prompt, programs, roles, fuel, override=-1):
    ids = {word: index for index, word in enumerate(vocabulary)}
    inputs = torch.tensor([ids[word] for word in prompt.split()])
    # Tool images may have different lengths. Padding is unreachable after HALT.
    images = torch.stack([torch.nn.functional.pad(image(rows),
                       (0, 0, 0, max(map(len, programs)) - len(rows))) for rows in programs])
    role_tensor, fuel_tensor = torch.tensor(roles), torch.tensor(fuel)
    before = images.clone()
    start = perf_counter_ns()
    requests, feedback, traces, output = graph(inputs, images, role_tensor, fuel_tensor, override)
    elapsed = perf_counter_ns() - start
    assert torch.equal(images, before)
    events = []
    slots = {"alpha": 0, "beta": 1, "gamma": 2}
    for turn, row in enumerate(requests.tolist()):
        tokens = [vocabulary[token] for token in row if token >= 0]
        if not tokens:
            break
        events.append(dict(kind="generated", tokens=tokens))
        if turn >= 2 or feedback[turn, 0] < 0:
            continue
        status, raw, reported, steps = feedback[turn].tolist()
        slot = slots[tokens[1]]
        args = list(map(int, tokens[2:4]))
        trace = traces[turn, :steps].tolist()
        event = dict(kind="tool", slot=tokens[1], role="offset" if roles[slot] == 0 else "boost",
                     arguments=args, trace=trace)
        if status == 15:
            event["computed"] = raw
            event["feedback"] = f"result {reported}"
        else:
            event["feedback"] = vocabulary[status]
        if status == 18 and not steps:
            assert not traces[turn].any()
        if steps:
            assert trace == oracle(programs[slot], args)[:steps]
        events.append(event)
    outcome, answer = output.tolist()
    record = dict(events=events, outcome=("answered", "stopped", "protocol_error")[outcome])
    if outcome == 0:
        record["answer"] = answer
    return record, elapsed


def replay():
    torch.set_num_threads(1)
    reference = Session()
    graph, reference_hash = load("reference")
    records, samples = [], []
    for index, fixture in enumerate(reference.spec["prompts"]):
        roles = [0 if fixture["slots"][slot] == "offset" else 1 for slot in ("alpha", "beta")]
        programs = [reference.spec["programs"][fixture["slots"][slot]] for slot in ("alpha", "beta")]
        runs, timing = {}, {}
        for mode in ("normal", "substitute", "disabled"):
            expected, phases = reference.run(index, mode)
            fuel = [48, 48]
            override = -1
            if mode == "disabled":
                fuel[roles.index(0)] = -1
            if mode == "substitute":
                override = expected["events"][1]["computed"] + 2
            actual, elapsed = run(graph, VOCAB, fixture["text"], programs, roles, fuel, override)
            assert actual == expected, (index, mode, actual, expected)
            runs[mode] = actual
            timing[mode] = dict(composed_ns=elapsed, reference_phases=phases)
        records.append(dict(prompt=fixture["text"], runs=runs))
        samples.append(timing)

    conditional, conditional_hash = load("conditional")
    holdout = json.loads((OUT / "holdout.json").read_text())
    vocabulary = json.loads((OUT / "training.json").read_text())["vocabulary"]
    programs = holdout["programs"]
    runs, conditional_samples = {}, {}
    for mode, spec in holdout["cases"].items():
        observed, first = run(conditional, vocabulary, spec["prompt"], programs, [0, 1, 1],
                              spec["fuel"], spec.get("override", -1))
        tool_events = [event for event in observed["events"] if event["kind"] == "tool"]
        assert [event["slot"] for event in tool_events] == spec["slots"], (mode, observed)
        assert [event["arguments"] for event in tool_events] == spec["arguments"], (mode, observed)
        assert [event["feedback"] for event in tool_events] == spec["feedback"], (mode, observed)
        assert observed.get("answer") == spec.get("answer"), (mode, observed)
        assert observed["outcome"] == ("answered" if "answer" in spec else "stopped")
        repeated = [first]
        for _ in range(6):
            again, elapsed = run(conditional, vocabulary, spec["prompt"], programs, [0, 1, 1],
                                 spec["fuel"], spec.get("override", -1))
            assert again == observed
            repeated.append(elapsed)
        runs[mode], conditional_samples[mode] = observed, repeated

    difference = event_difference(runs["normal"], runs["substitute"])
    assert difference["event"] == 2
    bundle = dict(format_version=1, reference_sha256=reference_hash,
                  conditional_sha256=conditional_hash, holdout=holdout,
                  reference=records, conditional=runs, first_difference=difference)
    return bundle, dict(reference_samples=samples, conditional_samples_ns=conditional_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--open", type=Path)
    args = parser.parse_args()
    with torch.inference_mode():
        bundle, timings = replay()
    if args.write:
        for name, value in (("replay", bundle), ("timings", timings)):
            (OUT / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n")
    else:
        assert bundle == json.loads((args.open or OUT / "replay.json").read_text())
    print(json.dumps(bundle["first_difference"], indent=2))
