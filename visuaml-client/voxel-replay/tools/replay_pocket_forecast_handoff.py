"""Audit a learned stop-or-forecast decision under paired Pocket feedback."""
from argparse import ArgumentParser
import json
from unittest.mock import patch

import torch
from torch import Tensor, nn

from composed_inference import walk
from evaluate_pocket_paired_goals import arrival, worlds
from evaluate_world_inference_pairs import verification_frames
from pocket_world_inference import OUT as WORLD
from replay_activation_intervention import BASE, PACKED
from replay_cellular_automata import export_region
from replay_composed_delay import Reader
from replay_intervention_diagnosis import OUT as PRIOR
from replay_pocket_world_inference import inputs
from replay_tagged_byte_requests import OUT as TAGGED, RequestController
from visuaml.replay._codec import canonical, digest

OUT = BASE / "release/examples/pocket-forecast-handoff"


class ForecastHandoff(nn.Module):
    def __init__(self, world, reader, controller):
        super().__init__()
        self.world = world
        self.reader = reader
        self.controller = controller

    def first_arrival(self, history: Tensor, target: int):
        for tick in range(history.size(0)):
            if bool(history[tick].flatten()[target]):
                return tick + 1
        return 0

    def forward(self, walls: Tensor, signal: Tensor, gate: int, target: int,
                deadline: int, coverage: int, override: int = -1):
        events = torch.full((9,), -1, dtype=torch.long)
        frames = torch.zeros((2, 8, walls.size(2), walls.size(3)))
        if coverage < 8 or not (0 <= gate < walls.numel() and 0 <= target < walls.numel()
                                and 1 <= deadline <= 7 and -1 <= override <= 8):
            events[0] = 1
            return events, frames
        _, history, _ = self.world.forecast(walls, signal, gate, target, 0, 8)
        frames[0] = history[:, 0]
        raw = self.first_arrival(history, target)
        shown = raw if override < 0 else override
        # Fixed ABI encoding. A late arrival asks whether Open can improve by
        # one tick. If Wait never arrives, Open is checked at the deadline.
        score = 8 - shown if shown else 0
        horizon = max(1, shown - 1) if shown else deadline
        tags = torch.tensor([[[0, score], [1, horizon]]])
        role = torch.zeros((1,), dtype=torch.long)
        logits, _, _, _ = self.controller(tags, torch.tensor([8 - deadline]), role + 2, role)
        choice = int(logits[0].argmax())
        horizon_byte = int(self.reader(tags, role + 1)[0])
        events[:6] = torch.tensor([0, raw, shown, score, choice, horizon_byte])
        if choice == 2:
            return events, frames
        if choice != 0 or not 1 <= horizon_byte <= 8:
            events[0] = 2
            return events, frames
        _, later, _ = self.world.forecast(walls, signal, gate, target, 1, horizon_byte)
        frames[1, :horizon_byte] = later[:, 0]
        result = self.first_arrival(later, target)
        events[6:] = torch.tensor([1, horizon_byte, result])
        return events, frames


def build():
    model = RequestController().eval()
    model.load_state_dict({**torch.load(PRIOR / "controller-829.pt", weights_only=True),
                           "binding.weight": torch.load(TAGGED / "binding.pt", weights_only=True)["weight"]})
    tags, role = torch.tensor([[[0, 3], [1, 4]]]), torch.tensor([0])
    reader = torch.jit.trace(Reader(model).eval(), (tags, role))
    controller = torch.jit.trace(model, (tags, torch.tensor([2]), role + 2, role))
    world = torch.jit.load(str(WORLD / "inference.pt"))
    graph = torch.jit.freeze(torch.jit.script(ForecastHandoff(world, reader, controller).eval()),
                             preserved_attrs=[])
    graph.save(str(OUT / "graph.pt"))
    (OUT / "graph.ir").write_text(str(graph.inlined_graph).rstrip() + "\n")


def audit(graph):
    operators = sorted({node.kind() for node in walk(graph.inlined_graph)})
    assert {"prim::PythonOp", "prim::CallMethod", "prim::CallFunction"}.isdisjoint(operators)
    rows, fixtures = [], []
    for name, closed, _ in worlds():
        if not name.startswith("shortcut"):
            continue
        opened_walls = closed["walls"].copy()
        opened_walls[closed["gate"]] = 0
        opened = dict(closed, walls=opened_walls)
        deadline = closed["deadline"] - 1
        late, early = [arrival(world, 0, 8) for world in (closed, opened)]
        assert early <= deadline < late <= 8
        for label, world, paired_feedback in (("closed", closed, early), ("open", opened, late)):
            packet = export_region(world, world["target"], 8, 8)
            walls, signal, gate, target = inputs(packet)
            fixture = f"{name}-{label}"
            fixtures.append(dict(id=fixture, world=world, checkpoint=packet, deadline=deadline))
            for mode, override in (("natural", -1), ("swap", paired_feedback),
                                   ("alternate_late", 8 if late == 7 else 7),
                                   ("missing_coverage", -1)):
                coverage = 7 if mode == "missing_coverage" else 8
                events, frames = graph(walls, signal, gate, target, deadline, coverage, override)
                values = events.tolist()
                if coverage == 7:
                    assert values == [1] + [-1] * 8 and not torch.any(frames)
                else:
                    raw = arrival(world, 0, 8)
                    shown = raw if override < 0 else override
                    choice = 2 if 0 < shown <= deadline else 0
                    horizon = shown - 1 if shown else deadline
                    assert values[:6] == [0, raw, shown, 8 - shown if shown else 0, choice, horizon]
                    assert verification_frames(world, packet, 0, 8, frames[0].flatten(1).int().tolist())
                    if choice == 0:
                        assert values[6:] == [1, horizon, arrival(world, 1, horizon)]
                        assert verification_frames(world, packet, 1, horizon,
                                                   frames[1, :horizon].flatten(1).int().tolist())
                    else:
                        assert values[6:] == [-1] * 3 and not torch.any(frames[1])
                rows.append(dict(fixture=fixture, mode=mode, override=override, events=values,
                                 target_timeline=[frame.flatten()[target].int().item()
                                                  for frame in frames[0]] if coverage == 8 else [],
                                 followup_timeline=[frame.flatten()[target].int().item()
                                                    for frame in frames[1, :values[7]]] if values[7] > 0 else []))
    return dict(profile="visuaml-pocket-forecast-handoff-v1", fixtures=fixtures, rows=rows,
                graph_sha256=digest((OUT / "graph.pt").read_bytes()), operators=operators,
                world_sha256=digest((WORLD / "inference.pt").read_bytes()))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--example", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with torch.inference_mode(), patch(PACKED, side_effect=AssertionError("Packed fallback entered")):
        if args.record:
            build()
        report = audit(torch.jit.load(str(OUT / "graph.pt")))
    payload = canonical(report) + b"\n"
    path = OUT / "replay.json"
    if args.record:
        path.write_bytes(payload)
    else:
        assert path.read_bytes() == payload
    if args.example:
        print(json.dumps(report["rows"][:8], indent=2))
    print(len(report["fixtures"]), "worlds;", len(report["rows"]), "verified conditions")


if __name__ == "__main__":
    main()
