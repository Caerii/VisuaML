"""Evaluate learned tagged byte emission and verified WASM dispatch under a routing edit."""
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
from replay_live_tool_interventions import expected_output, request
from replay_routing_subspaces import exact_tool
from visuaml.replay._codec import canonical, digest
from visuaml.replay.interventions import run_intervention

OUT = BASE / "release/examples/tagged-byte-requests"


INCREMENT = 0x01000100


class RequestController(Controller):
    def __init__(self):
        super().__init__()
        self.binding = nn.Embedding(2, 2)

    def read(self, records, role):
        tags, values = records.unbind(2)
        weights = self.binding(role).gather(1, tags).softmax(dim=1)
        bits = ((values.unsqueeze(2) >> torch.arange(8)) & 1).float()
        logits = ((2 * bits - 1) * weights.unsqueeze(2)).sum(dim=1)
        byte = ((logits > 0).long() << torch.arange(8)).sum(dim=1)
        emitted_role = tags.gather(1, weights.argmax(dim=1, keepdim=True)).squeeze(1)
        return emitted_role, byte, logits

    def forward(self, records, limit, rule, instruction=None):
        if instruction is None:
            return super().forward(records, limit, rule)
        _, live, _ = self.read(records, torch.zeros_like(instruction))
        role, byte, bits = self.read(records, instruction)
        return super().forward(live, limit, rule), role, byte, bits


def train(model):
    rows = list(product(range(4, 25), (-1, 1), (0, 1), (0, 1)))
    records = torch.tensor([[(order, live + step * order), (1 - order, live + step * (1 - order))]
                            for live, step, order, _ in rows])
    roles = torch.tensor([role for _, _, _, role in rows])
    target = torch.tensor([live + step * role for live, step, _, role in rows])
    target_bits = ((target[:, None] >> torch.arange(8)) & 1).float()
    optimizer = torch.optim.AdamW(model.binding.parameters(), lr=.05)
    for _ in range(180):
        _, _, logits = model.read(records, roles)
        loss = nn.functional.binary_cross_entropy_with_logits(5 * logits, target_bits)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def experiment(model, binary):
    first, followups = {}, {}

    def pair(live, saved, order):
        records = [(order, (live, saved)[order]), (1 - order, (live, saved)[1 - order])]
        raw = int.from_bytes(bytes(field for record in records for field in record), "little")
        if raw not in first:
            first[raw] = exact_tool(binary, raw - INCREMENT, raw)
        payload = first[raw][0].to_bytes(4, "little")
        return (tuple(payload[:2]), tuple(payload[2:])), raw

    def follow(action, value):
        key = (action, value)
        if key not in followups:
            followups[key] = execute(action, value)
        return followups[key][0]

    rows, donors, fixtures = [], [], []
    for limit, rule, side, step, order, instruction in product(
            HOLDOUT, range(3), (0, 1), (-1, 1), (0, 1), (0, 1)):
        live = limit - 1 + side
        values, raw = pair(live, live + step, order)
        donor_live = limit - side
        donor_values, donor_raw = pair(donor_live, donor_live + step, order)
        rows.append((values, limit, rule, instruction))
        donors.append((donor_values, limit, (rule + 1) % 3, instruction))
        fixtures.append((live, live + step, raw, donor_raw))

    def batch(items):
        records, limits, rules, roles = zip(*items, strict=True)
        return tuple(torch.tensor(part) for part in (records, limits, rules, roles))

    axis, _ = fit(model)
    baseline_args = batch(rows)
    _, state, _ = run_intervention(model, SITE, baseline_args)
    _, donor_state, _ = run_intervention(model, SITE, batch(donors))
    delta = (donor_state[:, :2] - state[:, :2]).flatten(1)
    replacement = state.clone()
    replacement[:, :2] += ((delta @ axis)[:, None] * axis).reshape_as(state[:, :2])
    modes = {}
    for comparison, binding in product((0, 1), repeat=2):
        args = (*baseline_args[:-1], baseline_args[-1] ^ binding)
        output, original, applied = run_intervention(
            model, SITE, args, replacement=replacement if comparison else None)
        assert torch.equal(original, state)
        assert torch.equal(applied[:, 2:], state[:, 2:])
        modes[comparison, binding] = output

    records, counts = [], {}
    witness = None
    for (comparison, binding), (routing, roles, arguments, bits) in modes.items():
        count = Counter()
        for i, ((records_in, limit, rule, instruction), fixture) in enumerate(zip(rows, fixtures, strict=True)):
            live, saved, raw, donor_raw = fixture
            target = RULES[rule][int(live >= limit) ^ comparison]
            target_role = instruction ^ binding
            target_byte = (live, saved)[target_role]
            action, role, argument = int(routing[i].argmax()), int(roles[i]), int(arguments[i])
            assert 0 <= argument <= 255
            requested = request(action, argument)
            wanted = request(target, target_byte)
            returned = follow(action, argument)
            count["tool_correct"] += action == target
            count["role_correct"] += role == target_role
            count["byte_correct"] += argument == target_byte
            count["request_correct"] += requested == wanted and returned == expected_output(target, target_byte)
            failure = ("tool_choice" if action != target else "role" if role != target_role else
                       "argument_byte" if argument != target_byte else None)
            item = dict(index=i, comparison_edit=bool(comparison), instruction_edit=bool(binding),
                        records=records_in, limit=limit, rule=rule, instruction=target_role,
                        first_argument=raw - INCREMENT, donor_argument=donor_raw - INCREMENT,
                        expected=wanted, request=requested, returned=returned,
                        role=role, expected_byte=target_byte, emitted_byte=argument,
                        failure=failure, route_logits=routing[i].tolist(), bit_logits=bits[i].tolist())
            records.append(item)
            if failure and witness is None:
                witness = (item, raw, donor_raw, action, argument)
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
    _, _, raw, donor_raw = fixtures[0]
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
        "binding.weight"}
    weights = OUT / "binding.pt"
    binary = OUT / "tagged.wasm"
    compiled = wasmtime.wat2wasm((OUT / "tagged.wat").read_text())
    if args.record:
        train(model)
        torch.save(model.binding.state_dict(), weights)
        binary.write_bytes(compiled)
    else:
        model.binding.load_state_dict(torch.load(weights, weights_only=True))
    assert binary.read_bytes() == compiled
    with torch.inference_mode(), patch(PACKED, side_effect=AssertionError("Packed fallback entered")):
        report, traces = experiment(model, binary)
    report.update(profile="visuaml-tagged-byte-requests-v1", base_sha256=digest(base.read_bytes()),
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
