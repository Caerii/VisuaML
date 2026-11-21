"""Verify that WASM-computed feedback changes the next model-built request."""
from argparse import ArgumentParser
from itertools import product
from struct import pack
from unittest.mock import patch

import torch
import wasmtime

from replay_activation_intervention import BASE, PACKED
from replay_compositional_interventions import RULES, execute
from replay_intervention_diagnosis import HOLDOUT, OUT as PRIOR
from replay_live_tool_interventions import expected_output, request
from replay_routing_subspaces import exact_tool
from replay_tagged_byte_requests import INCREMENT, OUT as TAGGED, RequestController
from visuaml.replay import Replay, capture, seal_report
from visuaml.replay._codec import canonical, digest

OUT = BASE / "release/examples/computed-argument"


def experiment(model, binary):
    first_calls, compute_calls, followups = {}, {}, {}
    delay_vm = Replay(binary)
    cases, traces = [], {}

    def observations(current, arrival, order):
        records = [(0, current), (1, arrival)]
        if order:
            records.reverse()
        word = int.from_bytes(bytes(field for row in records for field in row), "little")
        if word not in first_calls:
            first_calls[word] = exact_tool(TAGGED / "tagged.wasm", word - INCREMENT, word)
        payload = first_calls[word][0].to_bytes(4, "little")
        return [list(payload[:2]), list(payload[2:])], word

    def compute(current, arrival, expected):
        key = (current, arrival)
        if key not in compute_calls:
            initial = pack("<II", current, arrival) + bytes(65536 - 8)
            bundle = capture(delay_vm, 0, initial, 100, 1)
            result = delay_vm.finish(bundle["checkpoint"])
            assert result["status"] == 1 and result["returned"] == expected
            page = bytearray(initial)
            page[32:36] = pack("<I", expected)
            assert result["page_sha256"] == digest(page)
            compute_calls[key] = (result["returned"], seal_report(
                bundle, {"kind": "expect-return", "value": expected}))
        return compute_calls[key][0]

    for limit, rule, order, current, side in product(HOLDOUT, range(3), range(2), (2, 4), (-1, 1)):
        delay = limit - 3 if side < 0 else limit + 1
        arrival = current + delay
        ticks, first_word = observations(current, arrival, order)
        tensor = torch.tensor([ticks])
        operands = tuple(int(model.read(tensor, torch.tensor([role]))[1]) for role in (0, 1))
        assert operands == (current, arrival)
        assert delay not in operands
        actual = compute(*operands, delay)
        assert actual == delay
        variants = {
            "natural": (actual, delay, 1),
            "current_only": (compute(current + 1, arrival, delay - 1), delay - 1, 1),
            "argument_only": (compute(current, arrival + 1, delay + 1), delay + 1, 1),
            "cross_threshold": (compute(current, current + limit - int(side > 0), limit - int(side > 0)),
                                limit - int(side > 0), 1),
            "irrelevant_record": (actual, delay, 2),
        }
        condition_results = {}
        for name, (feedback, expected_byte, distractor) in variants.items():
            records = [(0, feedback), (1, distractor)]
            if order:
                records.reverse()
            logits, role, emitted, bits = model(torch.tensor([records]), torch.tensor([limit]),
                                               torch.tensor([rule]), torch.tensor([0]))
            choice, value = int(logits.argmax()), int(emitted)
            target = RULES[rule][int(expected_byte >= limit)]
            assert int(role) == 0 and 0 <= value <= 255
            expected_request, actual_request = request(target, expected_byte), request(choice, value)
            key = (choice, value)
            if key not in followups:
                followups[key] = execute(choice, value)
            returned, trace = followups[key]
            failure = ("tool" if choice != target else "argument" if value != expected_byte else
                       "execution" if returned != expected_output(target, expected_byte) else None)
            condition_results[name] = dict(records=records, actual_delay=actual, forwarded=feedback,
                                           expected_request=expected_request, request=actual_request,
                                           chosen_tool=choice, emitted=value, returned=returned, failure=failure,
                                           route_logits=logits[0].tolist(), bit_logits=bits[0].tolist())
            if not cases and trace is not None:
                traces[name] = trace
        natural = condition_results["natural"]
        crossed = condition_results["cross_threshold"]
        irrelevant = condition_results["irrelevant_record"]
        assert crossed["expected_request"] != natural["expected_request"]
        assert RULES[rule][int((delay + 1) >= limit)] == RULES[rule][int(delay >= limit)]
        assert irrelevant["expected_request"] == natural["expected_request"]
        if not cases:
            traces.update(first=first_calls[first_word][1], delay=compute_calls[operands][1],
                          substituted_delay=compute_calls[current, current + limit - int(side > 0)][1],
                          argument_delay=compute_calls[current, arrival + 1][1])
        cases.append(dict(ticks=ticks, operands=operands, threshold=limit, rule=rule,
                          conditions=condition_results))
    counts = {name: sum(row["conditions"][name]["failure"] is None for row in cases)
              for name in cases[0]["conditions"]}
    return dict(cases=cases, counts=counts, distinct_first=len(first_calls),
                distinct_delay=len(compute_calls),
                distinct_followup=sum(trace is not None for _, trace in followups.values())), traces


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = RequestController().eval()
    base, binding = PRIOR / "controller-829.pt", TAGGED / "binding.pt"
    model.load_state_dict({**torch.load(base, weights_only=True),
                           "binding.weight": torch.load(binding, weights_only=True)["weight"]})
    binary = bytes(wasmtime.wat2wasm((OUT / "delay.wat").read_text()))
    artifact = OUT / "delay.wasm"
    if args.record:
        artifact.write_bytes(binary)
    assert artifact.read_bytes() == binary
    with torch.inference_mode(), patch(PACKED, side_effect=AssertionError("Packed fallback entered")):
        report, traces = experiment(model, binary)
    report.update(profile="visuaml-computed-argument-v1", guest_sha256=digest(binary),
                  controller_sha256=digest(base.read_bytes()), reader_sha256=digest(binding.read_bytes()))
    for name, data in {"replay": report, **traces}.items():
        path = OUT / f"{name}.json"
        payload = canonical(data) + b"\n"
        if args.record:
            path.write_bytes(payload)
        else:
            assert payload == path.read_bytes(), path
    print(report["counts"], report["distinct_first"], report["distinct_delay"], report["distinct_followup"])


if __name__ == "__main__":
    main()
