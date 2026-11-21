"""Test a rank-one activation intervention across matched-training controllers."""
from argparse import ArgumentParser
from itertools import product
from statistics import median

import torch

from replay_compositional_interventions import Controller, RULES
from replay_intervention_diagnosis import CALIBRATION, HOLDOUT, OUT as PRIOR, SEEDS, SITE, train
from replay_activation_intervention import BASE
from visuaml.replay._codec import canonical, digest
from visuaml.replay.interventions import run_intervention

OUT = BASE / "release/examples/comparison-direction"
NEW_SEED = 1049
RANDOM_SEEDS = (0, 1, 2, 3, 4)


def input_batch(rows, donor=False):
    columns = (1, 2, 4) if donor else (0, 2, 3)
    return torch.tensor(rows)[:, columns].unbind(1)


def fit(model):
    rows = [(limit - 1, limit, limit, rule, rule)
            for limit, rule in product(CALIBRATION, range(3))]
    _, low, _ = run_intervention(model, SITE, input_batch(rows))
    _, high, _ = run_intervention(model, SITE, input_batch(rows, donor=True))
    low = low[:, :2].flatten(1)
    high = high[:, :2].flatten(1)
    delta = (high - low).mean(0)
    axis = delta / delta.norm()
    center = ((low + high) / 2 @ axis).mean()
    return axis, center


def evaluate(model, own_axis, center, source_axis):
    rows = [(limit + live, limit + alternative, limit, rule, other)
            for limit, live, alternative, rule, other in product(
                HOLDOUT, (-2, -1, 0, 1), (-2, -1, 0, 1), range(3), range(3))
            if rule != other]
    args = input_batch(rows)
    baseline, original, _ = run_intervention(model, SITE, args)
    _, alternate, _ = run_intervention(model, SITE, input_batch(rows, donor=True))
    direct = model(*torch.tensor(rows)[:, (1, 2, 3)].unbind(1))
    current = original[:, :2].flatten(1)
    donor = alternate[:, :2].flatten(1)

    def apply(delta):
        replacement = original.clone()
        replacement[:, :2] = (current + delta).reshape_as(original[:, :2])
        logits, _, observed = run_intervention(model, SITE, args, replacement=replacement)
        assert torch.equal(observed[:, 2:], original[:, 2:])
        return logits

    def project(axis):
        return ((donor - current) @ axis).unsqueeze(1) * axis

    edit = project(own_axis)
    transfer = project(source_axis)
    transfer_direction = torch.sign((donor - current) @ source_axis).unsqueeze(1) * source_axis
    outputs = dict(baseline=baseline, direct=direct, noop=apply(torch.zeros_like(current)),
                   joint=apply(donor - current), fitted=apply(edit),
                   transferred=apply(transfer),
                   transferred_matched_norm=apply(transfer_direction * edit.norm(dim=1, keepdim=True)),
                   erased=apply((center - current @ own_axis).unsqueeze(1) * own_axis))
    assert torch.equal(outputs["noop"], baseline)
    for seed in RANDOM_SEEDS:
        random = torch.randn(current.shape, generator=torch.Generator().manual_seed(seed))
        outputs[f"random_{seed}"] = apply(random / random.norm(dim=1, keepdim=True)
                                         * edit.norm(dim=1, keepdim=True))

    records = []
    for i, (live, other, limit, rule, donor_rule) in enumerate(rows):
        natural = RULES[rule][int(live >= limit)]
        target = RULES[rule][int(other >= limit)]
        scores = {}
        for name, batch in outputs.items():
            logits = batch[i]
            wanted = natural if name in ("baseline", "noop", "erased") else target
            rivals = [action for action in range(3) if action != wanted]
            scores[name] = [int(logits.argmax()), float(logits[wanted] - logits[rivals].max())]
        records.append(dict(feedback=live, donor=other, threshold=limit, rule=rule, donor_rule=donor_rule,
                            natural=natural, target=target, scores=scores))
    summary = {}
    for label, switch in (("opposite_side", True), ("same_side", False)):
        group = [row for row in records if (row["target"] != row["natural"]) == switch]
        summary[label] = dict(count=len(group), modes={
            name: dict(correct=sum(r["scores"][name][0] == (
                r["natural"] if name in ("baseline", "noop", "erased") else r["target"]) for r in group),
                       median_margin=median(r["scores"][name][1] for r in group))
            for name in outputs})
    return summary, records


def main():
    parser = ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--record", action="store_true")
    group.add_argument("--train", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    extra = OUT / f"controller-{NEW_SEED}.pt"
    if args.train:
        torch.save(train(NEW_SEED).state_dict(), extra)
    weights = [(seed, PRIOR / f"controller-{seed}.pt") for seed in SEEDS] + [(NEW_SEED, extra)]
    studies = []
    with torch.inference_mode():
        models = []
        for seed, path in weights:
            model = Controller().eval()
            model.load_state_dict(torch.load(path, weights_only=True))
            models.append((seed, digest(path.read_bytes()), model, *fit(model)))
        source = models[0][3]
        for seed, identity, model, axis, center in models:
            summary, records = evaluate(model, axis, center, source)
            stats = {group: {name: entry["correct"] for name, entry in item["modes"].items()}
                     for group, item in summary.items()}
            print(seed, stats)
            studies.append(dict(seed=seed, weights_sha256=identity, axis=axis.tolist(),
                                center=float(center), source_axis_cosine=float(source @ axis),
                                summary=summary, rows=records))
    path = OUT / "replay.json"
    data = canonical(dict(profile="visuaml-comparison-direction-v1", calibration=CALIBRATION,
                          holdout=HOLDOUT, source_seed=SEEDS[0], studies=studies)) + b"\n"
    if args.record or args.train:
        path.write_bytes(data)
    else:
        assert data == path.read_bytes(), path


if __name__ == "__main__":
    main()
