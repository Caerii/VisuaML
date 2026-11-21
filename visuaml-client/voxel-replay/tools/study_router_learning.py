"""Measure a learned condition representation under controlled interventions."""
import argparse
import json

import torch
from torch import nn

from frozen_call_checkpoint import digest
from internal_tool import prepare_program
from paired_routing import OUT as ROUTER, Router, description_ids, training_data
from synthesize_attention_vm import OUT as VM

OUT = ROUTER.parent / "word-research"
THRESHOLDS = (7, 13, 21, 27)
REGISTRY = [["offset the value", "raise each repeat", "repeat a trim"],
            ["shift the input", "raise the value each repeat", "lower the value each repeat"]]
CHECKPOINTS = (0, 50, 200, 600)


def evaluation_rows(words):
    rows = []
    for limit in THRESHOLDS:
        for polarity in (0, 1):
            for order in ((0,1,2), (2,0,1)):
                for first_family in (1, 2):
                    rows.append((limit, polarity, order, first_family))
    registry = torch.tensor([[description_ids(words[i]) for i in order] for _, _, order, _ in rows])
    choices = torch.tensor([[description_ids(words[i]), description_ids(words[3-i])]
                            for _, _, _, i in rows])
    limits = torch.tensor([row[0] for row in rows])
    polarities = torch.tensor([row[1] for row in rows])
    labels = [torch.tensor([order.index(first if ((side == 0) == (polarity == 0)) else 3-first)
                            for _, polarity, order, first in rows]) for side in (0, 1)]
    return rows, registry, choices, limits, polarities, labels


def evaluate(model, words):
    _, registry, choices, limits, relation, labels = evaluation_rows(words)
    first_before = model.first_scores(registry)
    inputs_low = torch.stack(((limits-1)/31-0.5, limits/31-0.5, relation.float()-0.5), dim=-1)
    inputs_high = inputs_low.clone()
    inputs_high[:, 0] += 1/31
    hidden = model.condition[:2]
    low, high = hidden(inputs_low), hidden(inputs_high)
    same_inputs = inputs_low.clone()
    same_inputs[:, 0] -= 1/31
    same_side = hidden(same_inputs)
    rng = torch.Generator().manual_seed(515)
    direction = torch.randn(low.shape, generator=rng)
    matched = low + direction / direction.norm(dim=1, keepdim=True) * (high-low).norm(dim=1, keepdim=True)
    registry_encoded, query = model.encode(registry), model.encode(choices)

    def choose(state):
        gate = torch.sigmoid(model.condition[2](state))
        combined = gate*query[:, 0] + (1-gate)*query[:, 1]
        return (registry_encoded * combined[:, None]).sum(-1).argmax(-1)

    before, after, patched, placebo, negative = [choose(x) for x in (low, high, high.clone(), matched, same_side)]
    eligible = (before == labels[0]) & (after == labels[1])
    assert torch.equal(first_before, model.first_scores(registry))
    return dict(pairs=len(before), low_correct=int((before == labels[0]).sum()),
                high_correct=int((after == labels[1]).sum()), correct_both=int(eligible.sum()),
                donor_switches=int(((patched != before) & eligible).sum()),
                random_switches=int(((placebo != before) & eligible).sum()),
                same_side_switches=int(((negative != before) & eligible).sum()),
                first_choice_unchanged=len(before))


def chain(model, executor, programs):
    rows, registry, choices, _, _, _ = evaluation_rows(REGISTRY[0])
    cases = []
    delta = {1: 3, 2: -2}
    for limit, polarity, order, first in rows:
        for target in (limit-1, limit):
            seed = target-7
            if seed < 0:
                continue

            def correct(value):
                return first if ((value < limit) == (polarity == 0)) else 3-first

            expected_ids = [0, correct(target)]
            after = target + delta[expected_ids[-1]]
            expected_ids.append(correct(after))
            expected = after + delta[expected_ids[-1]]
            if not 0 <= expected <= 31:
                continue
            row = rows.index((limit, polarity, order, first))
            value = seed
            chosen, outputs = [], []
            for stage in range(3):
                scores = (model.first_scores(registry[row]) if stage == 0 else
                          model.second_scores(registry[row], choices[row], torch.tensor(value),
                                              torch.tensor(limit), torch.tensor(polarity)))
                family = order[int(scores.argmax())]
                answer, _ = executor(programs[family], torch.tensor([value, 1]))
                chosen.append(family)
                value = int(answer)
                outputs.append(value)
                if not 0 <= value <= 31:
                    break
            cases.append(dict(seed=seed, threshold=limit, relation=polarity, order=order, first=first,
                              expected_choices=expected_ids, choices=chosen, outputs=outputs,
                              expected_answer=expected, first_wrong_call=next((i+1 for i in range(3)
                                  if chosen[i:i+1] != expected_ids[i:i+1]), None)))
    return cases


def study(train):
    data = training_data()
    keep = torch.ones_like(data[3], dtype=torch.bool)
    for threshold in THRESHOLDS:
        keep &= data[3] != threshold
    registry, choices, values, limits, relations, first, second = [x[keep] for x in data]
    archive = OUT / "learned-checkpoints"
    archive.mkdir(exist_ok=True)
    vm_path = VM / "executor.pt"
    executor = torch.jit.load(str(vm_path))
    code = json.loads((ROUTER / "holdout.json").read_text())["programs"]
    programs = [prepare_program(code[name]) for name in ("offset", "boost", "trim")]
    results = []
    for width in (8, 16, 32):
        for seed in (82, 514):
            torch.manual_seed(seed)
            model = Router(width)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            stages = []
            for step in range(CHECKPOINTS[-1]+1) if train else CHECKPOINTS:
                if step in CHECKPOINTS:
                    path = archive / f"width-{width}-seed-{seed}-step-{step}.pt"
                    if train:
                        torch.save(model.state_dict(), path)
                    else:
                        model.load_state_dict(torch.load(path, weights_only=True))
                    with torch.inference_mode():
                        stages.append(dict(step=step, sha256=digest(path.read_bytes()),
                                           familiar=evaluate(model, REGISTRY[0]),
                                           paraphrased=evaluate(model, REGISTRY[1])))
                if train and step < CHECKPOINTS[-1]:
                    ids = torch.randint(len(values), (256,))
                    loss = nn.functional.cross_entropy(model.first_scores(registry[ids]), first[ids])
                    loss += nn.functional.cross_entropy(model.second_scores(
                        registry[ids], choices[ids], values[ids], limits[ids], relations[ids]), second[ids])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            with torch.inference_mode():
                composed = chain(model, executor, programs)
            results.append(dict(width=width, seed=seed, stages=stages, three_call_holdout=composed))
    return dict(executor_sha256=digest(vm_path.read_bytes()), train_rows=len(values), withheld_thresholds=THRESHOLDS,
                description_heldout=REGISTRY[1], results=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    cli = parser.parse_args()
    torch.set_num_threads(1)
    record = json.dumps(study(cli.train), indent=2)+"\n"
    path = OUT / "learning.json"
    if cli.train:
        path.write_text(record)
    else:
        assert path.read_text() == record
    print("Verified learning checkpoints, matched patches, and held-out three-call tool chains.")
