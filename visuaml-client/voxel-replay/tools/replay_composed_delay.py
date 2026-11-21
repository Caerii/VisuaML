"""Compose the frozen v102 router and WASM reference step in one executable graph."""
from argparse import ArgumentParser
from functools import lru_cache
import json
from struct import pack
from unittest.mock import patch

import torch
from torch import Tensor, nn

from composed_inference import walk
from replay_activation_intervention import BASE, PACKED
from replay_computed_argument import OUT as COMPUTED
from replay_intervention_diagnosis import OUT as PRIOR
from replay_live_tool_interventions import expected_output, request
from replay_tagged_byte_requests import OUT as TAGGED, RequestController
from visuaml.compiler_lab.frozen_calls_v58 import CallState
from visuaml.compiler_lab.frozen_recurrent_v58 import MachineState
from visuaml.replay import Replay
from visuaml.replay._codec import canonical, digest

OUT = BASE / "release/examples/composed-delay"


class Reader(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, records, role):
        return self.model.read(records, role)[1]


class ComposedDelay(nn.Module):
    def __init__(self, reader, controller, template, step):
        super().__init__()
        self.reader = reader
        self.controller = controller
        self.step = step
        for name, tensor in (("stack", template.vm.stack), ("locals", template.vm.locals),
                             ("frames", template.frames), ("empty_page", template.vm.page)):
            self.register_buffer(name, torch.zeros_like(tensor))

    def run_guest(self, image: Tensor, entry: Tensor, argument: Tensor, page: Tensor, budget: int):
        zero = torch.zeros_like(entry)
        locals_ = torch.cat((argument.reshape(1), self.locals[1:]))
        state = CallState(MachineState(entry, zero, self.stack, locals_, page,
                                       torch.full_like(entry, budget), zero), zero, self.frames)
        for _ in range(budget + 1):
            if int(state.vm.status) != 0:
                break
            state = self.step(image, state).state
        return state.vm

    def forward(self, records: Tensor, configuration: Tensor, delay: Tensor, boost: Tensor,
                trim: Tensor, entries: Tensor):
        # configuration: threshold, rule, alternate feedback (-1 retains live),
        # irrelevant value, delay fuel, follow-up fuel
        events = torch.full((14,), -1, dtype=torch.long)
        delay_page = self.empty_page.clone()
        follow_page = self.empty_page.clone()
        if (not bool(torch.all((records >= 0) & (records <= 255)))
                or int(records[0, 0]) + int(records[1, 0]) != 1
                or int(records[:, 0].min()) != 0
                or not 0 <= int(configuration[0]) <= 31
                or not 0 <= int(configuration[1]) < 3
                or not -1 <= int(configuration[2]) <= 255
                or not 0 <= int(configuration[3]) <= 255
                or not 0 <= int(configuration[4]) <= 100
                or not 0 <= int(configuration[5]) <= 100):
            events[0] = 1  # malformed request
            return events, delay_page, follow_page
        role = torch.zeros((1,), dtype=torch.long)
        current = self.reader(records.unsqueeze(0), role)[0]
        arrival = self.reader(records.unsqueeze(0), role + 1)[0]
        events[1], events[2] = current, arrival
        delay_page[0], delay_page[4] = current.to(torch.uint8), arrival.to(torch.uint8)
        vm = self.run_guest(delay, entries[0], torch.zeros_like(current), delay_page, int(configuration[4]))
        events[3] = vm.status
        if int(vm.status) != 1:
            events[0] = 2  # delay failed
            return events, vm.page, follow_page
        events[4] = vm.stack[0]
        feedback = torch.where(configuration[2] < 0, vm.stack[0], configuration[2])
        if not 0 <= int(feedback) <= 255:
            events[0] = 3  # feedback outside the byte ABI
            return events, vm.page, follow_page
        events[5] = feedback
        framed = torch.stack((role[0], feedback, role[0] + 1, configuration[3])).reshape(2, 2)
        framed = torch.index_select(framed, 0, records[:, 0]).unsqueeze(0)
        events[10:] = framed.flatten()
        route, emitted_role, byte, _ = self.controller(
            framed, configuration[:1], configuration[1:2], role)
        action = int(route[0].argmax())
        events[6], events[7] = action, byte[0]
        if int(emitted_role[0]) != 0 or not 0 <= int(byte[0]) <= 255:
            events[0] = 4  # invalid second request
            return events, vm.page, follow_page
        events[0] = 0
        if action == 2:
            return events, vm.page, follow_page
        second = self.run_guest(boost if action == 0 else trim, entries[action + 1],
                                byte[0], follow_page, int(configuration[5]))
        events[8] = second.status
        if int(second.status) == 1:
            events[9] = second.stack[0]
        else:
            events[0] = 5
        return events, vm.page, second.page


def guests():
    return tuple(Replay(path.read_bytes()) for path in (
        COMPUTED / "delay.wasm", BASE / "release/examples/pipeline/changed-tool.wasm",
        BASE / "release/examples/routing-subspaces/trim.wasm"))


def build(vms):
    model = RequestController().eval()
    model.load_state_dict({**torch.load(PRIOR / "controller-829.pt", weights_only=True),
                           "binding.weight": torch.load(TAGGED / "binding.pt", weights_only=True)["weight"]})
    records = torch.tensor([[[0, 2], [1, 7]]])
    role = torch.tensor([0])
    reader = torch.jit.trace(Reader(model).eval(), (records, role))
    controller = torch.jit.trace(model, (records, torch.tensor([8]), role, role))
    template = vms[0].start(0, None, 100)
    graph = torch.jit.freeze(torch.jit.script(ComposedDelay(reader, controller, template, vms[0].graph).eval()))
    graph.save(str(OUT / "graph.pt"))
    (OUT / "graph.ir").write_text(str(graph.inlined_graph).rstrip() + "\n")


def audit(graph, vms):
    images = tuple(vm.image.rows for vm in vms)
    entries = torch.tensor([vm.image.entry for vm in vms])
    kinds = sorted({node.kind() for node in walk(graph.inlined_graph)})
    assert {"prim::PythonOp", "prim::CallMethod", "prim::CallFunction"}.isdisjoint(kinds)
    reference = json.loads((COMPUTED / "replay.json").read_bytes())

    @lru_cache
    def page(tool, argument, current=0, arrival=0):
        initial = pack("<II", current, arrival) + bytes(65536 - 8) if tool == 0 else None
        return vms[tool].run(vms[tool].start(argument, initial, 100)).vm.page

    observations = []
    for row in reference["cases"]:
        current, arrival = row["operands"]
        for name, expected in row["conditions"].items():
            override = -1 if name in ("natural", "irrelevant_record") else expected["forwarded"]
            distractor = dict(expected["records"])[1]
            config = torch.tensor([row["threshold"], row["rule"], override, distractor, 100, 100])
            events, delay_page, follow_page = graph(torch.tensor(row["ticks"]), config, *images, entries)
            status, a, b, delay_status, delay, feedback, action, value, follow_status, returned = events[:10].tolist()
            assert events[10:].reshape(2, 2).tolist() == expected["records"]
            assert (status, a, b, delay_status, delay, feedback) == (
                0, current, arrival, 1, expected["actual_delay"], expected["forwarded"])
            assert (action, value) == (expected["chosen_tool"], expected["emitted"])
            assert request(action, value) == expected["request"]
            assert torch.equal(delay_page, page(0, 0, current, arrival))
            if action == 2:
                assert (follow_status, returned) == (-1, -1) and not torch.any(follow_page)
            else:
                assert (follow_status, returned) == (1, expected_output(action, value))
                assert torch.equal(follow_page, page(action + 1, value))
            observations.append(dict(condition=name, records=row["ticks"], configuration=config.tolist(),
                                     events=events.tolist(), delay_page_sha256=digest(delay_page.numpy().tobytes()),
                                     followup_page_sha256=digest(follow_page.numpy().tobytes())))
    negatives = {}
    for name, records, budgets, expected_status in (
        ("duplicate_tag", [[0, 2], [0, 7]], (100, 100), 1),
        ("delay_exhaustion", [[0, 2], [1, 7]], (3, 100), 2),
        ("followup_exhaustion", [[0, 2], [1, 7]], (100, 0), 5),
    ):
        config = torch.tensor([8, 0, -1, 1, *budgets])
        events, delay_page, follow_page = graph(torch.tensor(records), config, *images, entries)
        assert int(events[0]) == expected_status and int(events[9]) == -1 and not torch.any(follow_page)
        if expected_status != 5:
            assert int(events[6]) == -1 and int(events[8]) == -1
        if name == "delay_exhaustion":
            assert int(events[3]) == 2 and int(delay_page[32]) == 0
        if name == "followup_exhaustion":
            assert int(events[8]) == 2
        negatives[name] = dict(records=records, configuration=config.tolist(), events=events.tolist(),
                               delay_page_sha256=digest(delay_page.numpy().tobytes()),
                               followup_page_sha256=digest(follow_page.numpy().tobytes()))
    return dict(profile="visuaml-composed-delay-v1", graph_sha256=digest((OUT / "graph.pt").read_bytes()),
                programs=[vm.identity for vm in vms], operators=kinds, rows=observations, negatives=negatives)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--case", choices=("duplicate_tag", "delay_exhaustion", "followup_exhaustion"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    with torch.inference_mode(), patch(PACKED, side_effect=AssertionError("Packed fallback entered")):
        vms = guests()
        if args.record:
            build(vms)
        report = audit(torch.jit.load(str(OUT / "graph.pt")), vms)
    path = OUT / "replay.json"
    payload = canonical(report) + b"\n"
    if args.record:
        path.write_bytes(payload)
    else:
        assert path.read_bytes() == payload
    if args.case:
        print(json.dumps(report["negatives"][args.case], indent=2))
        return
    print(len(report["rows"]), "matched conditions and", len(report["negatives"]), "negative controls")


if __name__ == "__main__":
    main()
