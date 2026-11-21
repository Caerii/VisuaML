"""Reproduce the frozen Pocket goal planner holdout and its interventions."""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from evaluate_world_inference_pairs import build_pairs, simulate, verification_frames
from frozen_call_checkpoint import canonical, digest
from pocket_goal_planner import OUT, WORLD, VOCAB, goal
from replay_cellular_automata import export_region
from replay_pocket_world_inference import inputs

SEED = 920431
MODES = ("actual", "precomputed", "disabled", "irrelevant", "missing", "short")


def recommendation(bits):
    return 0 if bits[0] else 1 if bits[1] else 2


def cases():
    for pair in build_pairs(SEED, 16):
        for label in ("wait", "open", "neither"):
            scene = pair["worlds"][int(label != "neither")]
            if sum(scene["signal"]) > 1:
                # Only the nearer seed can reach the target by this deadline.
                # Keep a single starting position for the movement interpretation.
                starts = [j for j, value in enumerate(scene["signal"]) if value]
                choices = []
                for start in starts:
                    signal = [int(j == start) for j in range(len(scene["signal"]))]
                    candidate = dict(scene, signal=signal)
                    x, y = scene["target"]
                    if simulate(candidate, 1, pair["horizon"])[-1][y*scene["width"]+x]:
                        choices.append(candidate)
                scene, = choices
            if label == "wait":
                walls = scene["walls"].copy()
                walls[scene["gate"]] = 0
                scene = dict(scene, walls=walls)
            yield pair, label, scene


def evaluate(model, pair, label, scene, mode):
    deadline = pair["horizon"]
    prompt = goal(deadline, "kindly")
    if mode == "irrelevant":
        walls = scene["walls"].copy()
        walls[0] ^= 1
        scene = dict(scene, walls=walls)
    x, y = scene["target"]
    target_index = y*scene["width"]+x
    oracle = [[scene["signal"][target_index]] +
              [frame[target_index] for frame in simulate(scene, action, 8)] for action in (0, 1)]
    bits = [row[deadline] for row in oracle]
    truth = recommendation(bits)
    assert ("wait", "open", "neither")[truth] == label
    expected = {"swap_wait": 0, "swap_open": 2}.get(mode, truth)
    override, replacement = {"swap_wait": (0,1), "swap_open": (1,0)}.get(mode, (-1,-1))
    coverage = deadline-1 if mode == "short" else deadline
    packet = export_region(scene, scene["target"], coverage, coverage)
    walls,signal,gate,target = inputs(packet)
    if mode == "missing":
        walls,signal,gate,target = torch.ones((1,1,1,1)),torch.zeros((1,1,1,1)),-1,-1
    req, feedback, frames, answer = model(torch.tensor([VOCAB.index(t) for t in prompt.split()]),
        walls,signal,gate,target,coverage,mode!="disabled",
        torch.tensor(oracle) if mode=="precomputed" else None,override,replacement)

    stages, seen, calls, mismatches = {}, {}, [], 0
    unavailable = mode in {"missing","short","disabled"}
    for turn, row in enumerate(req.tolist()):
        tokens = [VOCAB[t] for t in row if t >= 0]
        if not tokens or tokens[0] != "call":
            continue
        valid = len(tokens)==5 and tokens[1]=="alpha" and tokens[2] in ("0","1") and tokens[3]==str(deadline) and tokens[4]=="."
        stages[f"call_{turn+1}_arguments"] = valid
        if turn >= 2 or not valid:
            continue
        action = int(tokens[2])
        status,raw,shown,steps = feedback[turn].tolist()
        history = frames[turn,:steps].flatten(1).int().tolist() if steps else []
        correct = (verification_frames(scene,packet,action,steps,history) and raw==oracle[action][steps]) if steps else None
        if steps:
            mismatches += not correct
        changes = [j for j, value in enumerate(history[0]) if value != packet["signal"][j]] if history else []
        ox, oy = packet["origin"]
        first_changes = [[ox+j%packet["width"],oy+j//packet["width"]] for j in changes]
        calls.append(dict(first_updated_signal_cells=first_changes, turn=turn, action=action,horizon=deadline,status=status,raw=raw,shown=shown,
                          steps=steps,verified=correct, target_history=[frame[target] for frame in history],
                          expected_target_history=oracle[action][1:steps+1]))
        raw_expected = oracle[action][deadline]
        shown_expected = replacement if action==override else raw_expected
        expected_reply = [17,-1,-1,0] if unavailable else [15,raw_expected,shown_expected,0 if mode=="precomputed" else deadline]
        stages[f"call_{turn+1}_feedback"] = [status,raw,shown,steps]==expected_reply and correct is not False
        stages[f"call_{turn+1}_novel_action"] = action not in seen
        if status==15:
            seen[action] = shown
    outcome, action = answer.tolist()
    supported = outcome==0 and ((action==0 and seen.get(0)==1) or
                (seen.get(0)==0 and seen.get(1)==(1 if action==1 else 0) and action in (1,2)))
    stages["forecast_requested"] = bool(calls)
    if unavailable:
        stages["failure_handling"] = outcome==1 and len(calls)==1
    else:
        stages["recommendation"] = outcome==0 and action==expected
        stages["evidence"] = supported
    first = next((name for name,ok in stages.items() if not ok),None)
    return dict(prompt=prompt, expected=expected, world_recommendation=truth, outcome=outcome, action=action,
                supported=supported, stages=stages, first_wrong=first, success=first is None,
                checkpoint_sha256=None if mode=="missing" else digest(canonical(packet)),
                requests=[[VOCAB[t] for t in row if t>=0] for row in req.tolist()],
                calls=calls, mismatches=mismatches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write",action="store_true")
    parser.add_argument("--replay",type=Path,help="Verify a downloaded replay bundle")
    args = parser.parse_args()
    torch.set_num_threads(1)
    artifact = OUT/"inference.pt"
    model = torch.jit.load(str(artifact))
    if args.replay:
        bundle = json.loads(args.replay.read_text())
        assert bundle["graph_sha256"] == digest(artifact.read_bytes()), "Frozen graph differs"
        for name,example in bundle["examples"].items():
            row,scene = example["record"],example["world"]
            assert row["world_sha256"] == digest(canonical(scene)), name
            with torch.inference_mode():
                for mode,expected in row["runs"].items():
                    assert evaluate(model,row,row["label"],scene,mode) == expected, (name,mode)
            print(f"Verified {name}")
        return
    records, examples = [], {}
    with torch.inference_mode():
        for pair,label,scene in cases():
            modes = MODES + (("swap_wait","swap_open") if label=="open" else ())
            row = dict(id=pair["id"], family=pair["family"], horizon=pair["horizon"], label=label,
                       world_sha256=digest(canonical(scene)), runs={})
            for mode in modes:
                run = evaluate(model,pair,label,scene,mode)
                row["runs"][mode] = run
                if (label,mode) in (("open","actual"),("open","swap_wait"),("open","swap_open"),("open","missing")):
                    examples.setdefault(mode,dict(world=scene,record={**row,"runs":{mode:run}}))
                if not run["success"]:
                    examples.setdefault("failure_"+run["first_wrong"],dict(world=scene,record={**row,"runs":{mode:run}}))
            records.append(row)
    summary = {}
    for mode in MODES+("swap_wait","swap_open"):
        runs = [row["runs"][mode] for row in records if mode in row["runs"]]
        summary[mode] = dict(assigned=len(runs), successes=sum(r["success"] for r in runs),
                             unsupported=sum(r["outcome"]==0 and not r["supported"] for r in runs),
                             forecast_calls=sum(len(r["calls"]) for r in runs),
                             first_actions=dict(Counter(str(r["calls"][0]["action"]) for r in runs if r["calls"])),
                             call_counts=dict(Counter(str(len(r["calls"])) for r in runs)),
                             executed=sum(c["steps"]>0 for r in runs for c in r["calls"]),
                             mismatches=sum(r["mismatches"] for r in runs),
                             first_failures=dict(Counter(r["first_wrong"] for r in runs if r["first_wrong"])))
    assert all(value["mismatches"]==0 for value in summary.values())
    data = dict(seed=SEED, graph_sha256=digest(artifact.read_bytes()),
                simulator_source_sha256=digest((WORLD/"inference.pt").read_bytes()), summary=summary, records=records)
    for filename,value in (("outcomes.json",data),("replays.json",dict(graph_sha256=data["graph_sha256"],examples=examples))):
        path = OUT/filename
        if args.write:
            path.write_bytes(canonical(value)+b"\n")
        else:
            assert json.loads(path.read_text())==value, filename
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
