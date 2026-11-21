"""Compare causal and bidirectional masked trace proposals against exact replay."""
from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
from random import Random
from time import perf_counter_ns
from unittest.mock import patch

import torch
from torch import nn

from fork_divergence import first_divergence
from replay_frozen_recurrent import PACKED
from replay_voxel_call_checkpoint import Witness
from search_voxel_forks import edits, verify_branch

EVIDENCE = Path(__file__).resolve().parents[1] / "evidence"
OUT = EVIDENCE / "bidirectional-traces"
WORDS, MASK, MAX = 16, 16, 24


@dataclass
class Scene:
    table: list
    audit: list


def scene(rng, steps, family):
    table = list(range(WORDS))
    if family == "pointer":
        rng.shuffle(table)
    else:
        table = [(x-1) % WORDS for x in table]
    audit = [rng.randrange(WORDS)]
    for _ in range(steps):
        audit.append(table[audit[-1]])
    return Scene(table, audit)


class Proposer(nn.Module):
    def __init__(self, causal):
        super().__init__()
        self.causal = causal
        self.embed = nn.Embedding(WORDS+1, 64)
        self.positions = nn.Embedding(WORDS+MAX+1, 64)
        layer = nn.TransformerEncoderLayer(64, 4, 128, dropout=0, batch_first=True)
        self.stack = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.out = nn.Linear(64, WORDS)

    def forward(self, tokens):
        length = tokens.shape[1]
        x = self.embed(tokens) + self.positions(torch.arange(length))
        mask = torch.ones(length, length, dtype=torch.bool).triu(1) if self.causal else None
        return self.out(self.stack(x, mask=mask))[:, WORDS:]


def train(causal, seed):
    torch.manual_seed(seed)
    rng = Random(seed)
    model = Proposer(causal)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    for update in range(900):
        inputs, labels = [], []
        for _ in range(32):
            s = scene(rng, 16, rng.choice(("countdown", "pointer")))
            probability = rng.choice((.15, .35, .65, 1.))
            noisy = [s.audit[0]]
            target = [-100]
            for value in s.audit[1:]:
                unknown = rng.random() < probability
                noisy.append(MASK if unknown else value)
                target.append(value if unknown else -100)
            if all(x == -100 for x in target):
                noisy[-1], target[-1] = MASK, s.audit[-1]
            inputs.append(s.table + noisy)
            labels.append(target)
        logits = model(torch.tensor(inputs))
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), torch.tensor(labels).flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.eval()


def propose(model, table, visible, rounds):
    """Run blockwise masked denoising. Revealed fields stay fixed."""
    tokens = torch.cat((table, visible), dim=1)
    unknown = visible.eq(MASK)
    snapshots = []
    for round_index in range(rounds):
        scores, guess = model(tokens).softmax(-1).max(-1)
        remaining = tokens[:, WORDS:].eq(MASK) & unknown
        ranks = scores.masked_fill(~remaining, -1).argsort(dim=1, descending=True).argsort(dim=1)
        count = (remaining.sum(1, keepdim=True) + rounds-round_index-1)//(rounds-round_index)
        fill = remaining & (ranks < count)
        tokens[:, WORDS:] = torch.where(fill, guess, tokens[:, WORDS:])
        snapshots.append(tokens[:, WORDS:].clone())
    return snapshots


def diagnose(model, tables, candidates):
    # One independent mask per suspect position. Rank how much the conditional
    # prediction disfavors the supplied token. No reference states enter here.
    count, length = candidates.shape
    mask = torch.eye(length, dtype=torch.bool)[1:]
    damaged = candidates[:, None, :].expand(-1, length-1, -1).masked_fill(mask, MASK)
    context = tables[:, None, :].expand(-1, length-1, -1)
    logits = model(torch.cat((context, damaged), dim=2).flatten(0, 1))
    idx = torch.arange(1, length)
    distributions = logits.reshape(count, length-1, length, WORDS)[:, idx-1, idx]
    probability = distributions.softmax(-1).gather(-1, candidates[:, 1:, None]).squeeze(-1)
    bad = probability.argmin(dim=1)+1
    replacement = distributions.argmax(-1)[torch.arange(count), bad-1]
    return bad, candidates.scatter(1, bad[:, None], replacement[:, None])


def exact_parallel(table, initial, steps, schedule):
    """Bulk synchronous propagation control. Unknown is represented by None."""
    row = [initial] + [None]*steps
    rounds = []
    while None in row:
        row = [initial] + [None if value is None else table[value] for value in row[:-1]]
        rounds.append(steps)
    return dict(rounds=len(rounds), transition_lanes=sum(rounds), correct=row == schedule)


def exact_doubling(scene, steps):
    """Compose static pointer tables, then evaluate each power in parallel."""
    powers = [scene.table]
    while 2**len(powers) <= steps:
        previous = powers[-1]
        powers.append([previous[value] for value in previous])
    row = [scene.audit[0]]*(steps+1)
    lanes = WORDS*(len(powers)-1)
    for bit, table in enumerate(powers):
        lanes += sum(bool(i & (1 << bit)) for i in range(steps+1))
        row = [table[value] if i & (1 << bit) else value for i, value in enumerate(row)]
    assert row == scene.audit
    return dict(precompute_rounds=len(powers)-1, lookup_rounds=len(powers), table_entries=WORDS*len(powers),
                transition_lanes=lanes)


def score(models, count=96):
    rng = Random(72891)
    rows = []
    for family, length in product(("countdown", "pointer"), (4, 8, 16, 24)):
        scenes = [scene(rng, length, family) for _ in range(count)]
        tables = torch.tensor([s.table for s in scenes])
        truth = torch.tensor([s.audit for s in scenes])
        tests = {}
        for condition in ("initial_only", "endpoint_revealed"):
            visible = torch.full_like(truth, MASK)
            visible[:, 0] = truth[:, 0]
            if condition == "endpoint_revealed":
                visible[:, -1] = truth[:, -1]
            cases = {}
            for kind, model in models.items():
                samples = []
                for rounds in (1, 2, 4, 8):
                    start = perf_counter_ns()
                    snapshots = propose(model, tables, visible, rounds)
                    elapsed = perf_counter_ns()-start
                    guessed = snapshots[-1]
                    exact = guessed.eq(truth)
                    faults = [first_divergence(s.audit, trial.tolist()) for s, trial in zip(scenes, guessed)]
                    samples.append(dict(rounds=rounds, exact=int(exact.all(1).sum()),
                                        correct_fields=int((exact & visible.eq(MASK)).sum()),
                                        masked_fields=int(visible.eq(MASK).sum()), ns=elapsed,
                                        first_bad=dict(Counter(str(x) for x in faults)),
                                        example=next((dict(table=s.table, correct=s.audit,
                                                           proposals=[frame[i].tolist() for frame in snapshots],
                                                           first_bad=faults[i])
                                                      for i, s in enumerate(scenes) if faults[i] is not None), None)))
                cases[kind] = samples
            tests[condition] = cases
        visible[:, -1] = truth[:, -1]
        swapped = visible.clone()
        swapped[:, -1] = (truth[:, -1]+1) % WORDS
        interventions = {}
        for kind, model in models.items():
            base = propose(model, tables, visible, 4)[-1]
            changed = propose(model, tables, swapped, 4)[-1]
            assert not changed.eq(truth).all(1).any()
            interventions[kind] = dict(interior_changed=int(
                base[:, 1:-1].ne(changed[:, 1:-1]).any(1).sum()),
                invalid_full_traces=count)
        candidate = truth.clone()
        fault_indices = [rng.randrange(1, length) for _ in scenes]
        for i, idx in enumerate(fault_indices):
            candidate[i, idx] = (candidate[i, idx] + rng.randrange(1, WORDS)) % WORDS
        diagnostics = {}
        for kind, model in models.items():
            positions, repaired = diagnose(model, tables, candidate)
            diagnostics[kind] = dict(localized=int(positions.eq(torch.tensor(fault_indices)).sum()),
                                     repaired=int(repaired.eq(truth).all(1).sum()))
        assert [first_divergence(s.audit, c.tolist()) for s, c in zip(scenes, candidate)] == fault_indices
        rows.append(dict(family=family, length=length, pairs=count, proposals=tests, diagnosed=diagnostics, endpoint_intervention=interventions,
                         exact_parallel=exact_parallel(scenes[0].table, scenes[0].audit[0], length, scenes[0].audit),
                         exact_doubling=exact_doubling(scenes[0], length),
                         exact_serial=dict(rounds=length, transition_lanes=length)))
    return rows


def goal_prefix(opens, deadline, horizon):
    # The gate availability starts at the first allowed tick. The objective
    # is to cross by the deadline with one edit, preferring earlier arrival.
    return [opens, deadline, horizon] + [0]*(WORDS-3)


def train_goal(seed):
    torch.manual_seed(seed)
    rng = Random(seed)
    model = Proposer(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    for _ in range(280):
        horizon = rng.choice((3, 4, 6, 7, 8))
        inputs, targets = [], []
        for _ in range(32):
            opens = rng.randrange(1, horizon+2)
            deadline = rng.randrange(1, horizon+1)
            answer = [int(opens == i and opens <= deadline) for i in range(1,horizon+1)]
            noisy = [MASK if rng.random() < .8 else value for value in answer]
            targets.append([-100]+[value if token == MASK else -100 for value, token in zip(answer, noisy)])
            inputs.append(goal_prefix(opens, deadline, horizon) + [0]+noisy)
        logits = model(torch.tensor(inputs))
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), torch.tensor(targets).flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.eval()


def goal_proposals(model):
    horizon = 5
    settings = list(product(range(1, 7), range(1, 6)))
    tables = torch.tensor([goal_prefix(opens, deadline, horizon) for opens, deadline in settings])
    visible = torch.tensor([[0]+[MASK]*horizon]*len(settings))
    proposals = propose(model, tables, visible, 2)[-1][:, 1:].tolist()
    schedules = tuple(product((0, 1), repeat=horizon))
    outcomes = []
    for (opens, deadline), proposal in zip(settings, proposals, strict=True):
        def arrival(actions):
            return next((i for i, action in enumerate(actions, 1) if i >= opens and action == 1), None)
        valid = [(sum(actions), arrival(actions), actions) for actions in schedules
                 if arrival(actions) is not None and arrival(actions) <= deadline]
        oracle = list(min(valid)[2]) if valid else None
        binary = all(action in (0, 1) for action in proposal)
        arrived = arrival(proposal) if binary else None
        valid_goal = arrived is not None and arrived <= deadline
        outcomes.append(dict(opens=opens, deadline=deadline, proposal=proposal, oracle=oracle,
                             feasible=valid_goal, oracle_match=(proposal == oracle if oracle is not None
                                                               else proposal == [0]*horizon)))
    return outcomes


def real_witnesses(goal_rows):
    fork_dir = EVIDENCE / "voxel-fork-search"
    checkpoint = json.loads((fork_dir / "checkpoint.json").read_text())
    archive = json.loads((fork_dir / "search.json").read_text())
    plan = json.loads((EVIDENCE / "shared-fork-search" / "compiler-session.json").read_text())
    wrong = first_divergence(plan["reference"], plan["candidate"])
    assert wrong is not None and plan["reference"][-1]["stack"] == plan["candidate"][-1]["stack"]

    # The original guest has an open passage immediately after the suspended
    # tick. The model only receives that availability, horizon, and deadline.
    learned = next(row["proposal"] for row in goal_rows if row["opens"] == 1 and row["deadline"] == 5)
    actions = [1, 0, 0, 0, 0]
    known = archive["branches"]
    winner = min((branch for branch in known if branch["crossing_tick"] is not None),
                 key=lambda x: (x["edit_count"], x["crossing_tick"], x["actions"]))
    with patch(PACKED, side_effect=AssertionError("Packed fallback entered")):
        replay, trace = verify_branch(Witness(), checkpoint, edits(actions), 1)
        disabled, _ = verify_branch(Witness(), checkpoint, {}, -1)
        delayed, _ = verify_branch(Witness(), checkpoint, edits([0, 0, 0, 0, 1]), -1)
        if all(action in (0, 1) for action in learned):
            learned_replay, _ = verify_branch(Witness(), checkpoint, edits(learned), -1)
        else:
            learned_replay = []
    assert winner["actions"] == actions and replay == winner["outputs"]
    def arrival(history):
        return next((row["tick"] for row in history if row["world"][1]), None)

    assert (arrival(replay), arrival(disabled), arrival(delayed)) == (1, None, 5)
    return dict(compiler=dict(first_wrong_index=wrong, source_pc=plan["first_source_pc"],
                              reference=plan["reference"][wrong], candidate=plan["candidate"][wrong]),
                goal=dict(checkpoint=checkpoint["sha256"], horizon=5, actions=actions, outputs=replay,
                          exact_alternatives=len(known), arrival=1, delayed_arrival=5,
                          learned=learned, learned_arrival=arrival(learned_replay),
                          learned_outputs=learned_replay,
                          source_write=next(row for row in trace if row["pc"] == 28)))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--train-goal", action="store_true")
    options = parser.parse_args()
    torch.set_num_threads(1)
    models, hashes = {}, {}
    for causal, name in ((False, "bidirectional"), (True, "causal_mask")):
        path = OUT / (name+".pt")
        if options.train:
            model = train(causal, 7200)
            torch.save(model.state_dict(), path)
        model = Proposer(causal).eval()
        model.load_state_dict(torch.load(path, weights_only=True))
        models[name] = model
        hashes[name] = sha256(path.read_bytes()).hexdigest()
    path = OUT / "goal.pt"
    if options.train or options.train_goal:
        torch.save(train_goal(7201).state_dict(), path)
    goal = Proposer(False).eval()
    goal.load_state_dict(torch.load(path, weights_only=True))
    hashes["goal"] = sha256(path.read_bytes()).hexdigest()
    with torch.inference_mode():
        goal_rows = goal_proposals(goal)
        report = dict(models=hashes, training_seed=7200, goal_seed=7201, heldout_seed=72891,
                      synthetic=score(models), goal=goal_rows, vm=real_witnesses(goal_rows))
    if options.train or options.train_goal or options.record:
        (OUT / "results.json").write_text(json.dumps(report, indent=2)+"\n")
    else:
        saved = json.loads((OUT / "results.json").read_text())
        for report_version in (saved, report):
            for suite in report_version["synthetic"]:
                for condition in suite["proposals"].values():
                    for results in condition.values():
                        for row in results:
                            row.pop("ns")
        assert report == saved
    print("Verified", len(report["synthetic"]), "held-out suites and both VM witnesses")


if __name__ == "__main__":
    main()
