"""Evaluate a learned request slot and verified WASM dispatch under a routing edit."""
from argparse import ArgumentParser
from collections import Counter
from itertools import product
from unittest.mock import patch

import torch
from torch import nn
import wasmtime

from replay_activation_intervention import BASE, PACKED
from replay_comparison_direction import fit
from replay_compositional_interventions import Controller, RULES, TOOLS, execute
from replay_intervention_diagnosis import CALIBRATION, HOLDOUT, OUT as PRIOR, SITE
from replay_live_preservation import fit_argument_probe
from replay_live_tool_interventions import expected_output, request
from replay_routing_subspaces import exact_tool
from visuaml.replay._codec import canonical, digest
from visuaml.replay.interventions import run_intervention

OUT = BASE / "release/examples/tool-argument-binding"


class RequestController(Controller):
    def __init__(self):
        super().__init__()
        self.binding = nn.Sequential(nn.Linear(7, 16), nn.Tanh(), nn.Linear(16, 2))

    def forward(self, live, limit, rule, order=None, instruction=None):
        route = super().forward(live, limit, rule)
        if order is None:
            return route
        metadata = torch.cat((nn.functional.one_hot(order, 2),
                              nn.functional.one_hot(instruction, 2)), dim=1)
        return route, self.binding(torch.cat((route, metadata), dim=1))


def train(model):
    rows = torch.tensor(list(product(range(4, 25), CALIBRATION, range(3), range(2), range(2))))
    live, limit, rule, order, instruction = rows.unbind(1)
    with torch.no_grad():
        route = model(live, limit, rule)
    metadata = torch.cat((nn.functional.one_hot(order, 2),
                          nn.functional.one_hot(instruction, 2)), dim=1)
    features = torch.cat((route, metadata), dim=1)
    optimizer = torch.optim.AdamW(model.binding.parameters(), lr=.025)
    for _ in range(180):
        loss = nn.functional.cross_entropy(model.binding(features), order ^ instruction)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def experiment(model, binary):
    first, followups = {}, {}

    def pair(values):
        raw = values[0] + 256 * values[1]
        if raw not in first:
            first[raw] = exact_tool(binary, raw - 257, raw)
        returned = first[raw][0]
        return (returned & 255, returned >> 8), raw

    def follow(action, value):
        key = (action, value)
        if key not in followups:
            followups[key] = execute(action, value)
        return followups[key][0]

    rows, donors, fixtures = [], [], []
    for limit, rule, side, step, order, instruction in product(
            HOLDOUT, range(3), (0, 1), (-1, 1), (0, 1), (0, 1)):
        live = limit - 1 + side
        values = (live, live + step)
        values, raw = pair(values[::-1] if order else values)
        donor_live = limit - side
        donor_values = (donor_live, donor_live + step)
        donor_values, donor_raw = pair(donor_values[::-1] if order else donor_values)
        rows.append((values[order], limit, rule, order, instruction))
        donors.append((donor_values[order], limit, (rule + 1) % 3, order, instruction))
        fixtures.append((values, raw, donor_raw))

    axis, _ = fit(model)
    baseline_args = torch.tensor(rows).unbind(1)
    _, state, _ = run_intervention(model, SITE, baseline_args)
    _, donor_state, _ = run_intervention(model, SITE, torch.tensor(donors).unbind(1))
    delta = (donor_state[:, :2] - state[:, :2]).flatten(1)
    replacement = state.clone()
    replacement[:, :2] += ((delta @ axis)[:, None] * axis).reshape_as(state[:, :2])
    probe_weights, probe_mean, probe_center = fit_argument_probe(model)
    modes = {}
    for comparison, binding in product((0, 1), repeat=2):
        args = (*baseline_args[:-1], baseline_args[-1] ^ binding)
        output, original, applied = run_intervention(
            model, SITE, args, replacement=replacement if comparison else None)
        assert torch.equal(original, state)
        assert torch.equal(applied[:, 2:], state[:, 2:])
        estimates = (applied[:, 0].double() - probe_mean) @ probe_weights + probe_center
        modes[comparison, binding] = (*output, estimates)

    records, counts = [], {}
    witness = None
    for (comparison, binding), (routing, slots, estimates) in modes.items():
        count = Counter()
        for i, ((live, limit, rule, order, instruction), fixture) in enumerate(zip(rows, fixtures, strict=True)):
            values, raw, donor_raw = fixture
            target = RULES[rule][int(live >= limit) ^ comparison]
            target_slot = order ^ instruction ^ binding
            action, slot = int(routing[i].argmax()), int(slots[i].argmax())
            estimate = float(estimates[i])
            count["probe_live_recovered"] += abs(values[order] - estimate) < abs(values[order ^ 1] - estimate)
            requested = request(action, values[slot])
            wanted = request(target, values[target_slot])
            returned = follow(action, values[slot])
            count["tool_correct"] += action == target
            count["slot_correct"] += slot == target_slot
            count["request_correct"] += requested == wanted and returned == expected_output(target, values[target_slot])
            count["wrong_tool"] += action != target
            count["unexpected_call"] += target == 2 and action != 2
            count["wrong_argument_on_correct_call"] += action == target != 2 and slot != target_slot
            failure = "tool_choice" if action != target else "argument_slot" if requested != wanted else None
            item = dict(index=i, comparison_edit=bool(comparison), instruction_edit=bool(binding),
                        inputs=rows[i], returned_pair=values, first_argument=raw - 257,
                        donor_argument=donor_raw - 257, expected=wanted, request=requested, returned=returned,
                        live_probe_estimate=estimate, slot=slot, expected_slot=target_slot, failure=failure, route_logits=routing[i].tolist(),
                        slot_logits=slots[i].tolist())
            records.append(item)
            if failure and witness is None:
                witness = (item, raw, donor_raw, action, values[slot])
        counts[f"{comparison}{binding}"] = dict(count)
    traces = {}
    if witness:
        item, raw, donor_raw, action, value = witness
        traces["witness-first"] = first[raw][1]
        traces["witness-donor"] = first[donor_raw][1]
        if followups[action, value][1] is not None:
            traces["witness-followup"] = followups[action, value][1]
        if item["expected"] is not None:
            target = TOOLS.index(item["expected"]["tool"])
            argument = item["expected"]["argument"]
            follow(target, argument)
            traces["witness-expected"] = followups[target, argument][1]
    example = [row for row in records if row["index"] == 0]
    _, raw, donor_raw = fixtures[0]
    traces["example-first"] = first[raw][1]
    traces["example-donor"] = first[donor_raw][1]
    for row in example:
        call = row["request"]
        if call is not None:
            mode = f'{int(row["comparison_edit"])}{int(row["instruction_edit"])}'
            traces[f"example-{mode}"] = followups[TOOLS.index(call["tool"]), call["argument"]][1]
    return dict(cases_per_mode=len(rows), counts=counts, rows=records,
                first_mismatch=witness[0] if witness else None, first_calls=len(first),
                followup_calls=sum(trace is not None for _, trace in followups.values())), traces


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(100)
    torch.set_num_threads(1)
    model = RequestController().eval()
    base = PRIOR / "controller-829.pt"
    result = model.load_state_dict(torch.load(base, weights_only=True), strict=False)
    assert not result.unexpected_keys and set(result.missing_keys) == {
        "binding.0.weight", "binding.0.bias", "binding.2.weight", "binding.2.bias"}
    weights = OUT / "binding.pt"
    binary = OUT / "pair.wasm"
    compiled = wasmtime.wat2wasm((OUT / "pair.wat").read_text())
    if args.record:
        train(model)
        torch.save(model.binding.state_dict(), weights)
        binary.write_bytes(compiled)
    else:
        model.binding.load_state_dict(torch.load(weights, weights_only=True))
    assert binary.read_bytes() == compiled
    with torch.inference_mode(), patch(PACKED, side_effect=AssertionError("Packed fallback entered")):
        report, traces = experiment(model, binary)
    report.update(profile="visuaml-tool-argument-binding-v1", base_sha256=digest(base.read_bytes()),
                  binding_sha256=digest(weights.read_bytes()), guest_sha256=digest(compiled))
    if args.record:
        print(report["counts"], report["first_mismatch"], report["first_calls"], report["followup_calls"])
    for name, data in {"replay": report, **traces}.items():
        path = OUT / f"{name}.json"
        raw = canonical(data) + b"\n"
        if args.record:
            path.write_bytes(raw)
        else:
            assert path.read_bytes() == raw, path


if __name__ == "__main__":
    main()
