"""Measure calibration-only scalar alignment and feedback/threshold dissociations."""
from argparse import ArgumentParser
from itertools import product

import torch

from replay_comparison_direction import NEW_SEED, OUT as DIRECTIONS, fit
from replay_compositional_interventions import Controller, RULES
from replay_intervention_diagnosis import CALIBRATION, HOLDOUT, OUT as PRIOR, SEEDS, SITE
from replay_activation_intervention import BASE
from visuaml.replay._codec import canonical, digest
from visuaml.replay.interventions import run_intervention

OUT = BASE / "release/examples/comparison-alignment"


def activations(model, rows):
    logits, states, _ = run_intervention(model, SITE, torch.tensor(rows).unbind(1))
    return logits, states


def cases():
    for limit, rule, donor_rule, high in product(HOLDOUT, range(3), range(3), (False, True)):
        if rule == donor_rule:
            continue
        value = limit - 1 + int(high)
        live = (value, limit, rule)
        # The donor threshold moves across the unchanged value.
        yield "threshold_only", live, (value, value + int(high), donor_rule)
        yield "same_side", live, (value + (2 if high else -2), limit, donor_rule)
        # Feedback alone implies the opposite action under the recipient threshold.
        yield "conflict", live, (value + (-2 if high else 2), limit + (-3 if high else 3), donor_rule)
        yield "conflict_switch", live, (value + (2 if high else -2), limit + (3 if high else -3), donor_rule)
        yield "feedback_only", live, (limit - int(high), limit, donor_rule)


def evaluate(source, source_axis, recipient, axis, *, design=None, strengths=()):
    calibration = [(limit + offset, limit, rule) for limit, offset, rule in
                   product(CALIBRATION, (-2, -1, 0, 1), range(3))]
    _, source_cal = activations(source, calibration)
    _, target_cal = activations(recipient, calibration)
    x = source_cal[:, :2].flatten(1) @ source_axis
    y = target_cal[:, :2].flatten(1) @ axis
    xc, yc = x - x.mean(), y - y.mean()
    slope = (xc @ yc) / (xc @ xc)

    design = list(cases() if design is None else design)
    live = [row[1] for row in design]
    donor = [row[2] for row in design]
    baseline, current = activations(recipient, live)
    _, local_donor = activations(recipient, donor)
    _, source_live = activations(source, live)
    _, source_donor = activations(source, donor)
    state = current[:, :2].flatten(1)
    local_delta = local_donor[:, :2].flatten(1) - state
    source_delta = source_donor[:, :2].flatten(1) - source_live[:, :2].flatten(1)

    def patch(delta):
        replacement = current.clone()
        replacement[:, :2] = (state + delta).reshape_as(current[:, :2])
        logits, _, applied = run_intervention(recipient, SITE, torch.tensor(live).unbind(1),
                                              replacement=replacement)
        assert torch.equal(applied[:, 2:], current[:, 2:])
        return logits

    aligned = (source_delta @ source_axis)[:, None] * slope * axis
    outputs = dict(baseline=baseline, joint=patch(local_delta),
                   local=patch((local_delta @ axis)[:, None] * axis),
                   raw=patch((local_delta @ source_axis)[:, None] * source_axis),
                   aligned=patch(aligned))
    outputs.update((f"strength_{gain:g}", patch(gain * aligned)) for gain in strengths)
    records = []
    for i, (family, (v, t, r), (dv, dt, dr)) in enumerate(design):
        desired = RULES[r][int(dv >= dt)]
        numeric = RULES[r][int(dv >= t)]
        records.append(dict(family=family, live=[v,t,r], donor=[dv,dt,dr],
                            relational_target=desired, numeric_target=numeric,
                            scores={name: dict(action=int(logits[i].argmax()), logits=logits[i].tolist())
                                    for name, logits in outputs.items()}))
    counts = {}
    for family in dict.fromkeys(row[0] for row in design):
        group = [row for row in records if row["family"] == family]
        counts[family] = dict(count=len(group), modes={
            mode: {key: sum(row["scores"][mode]["action"] == row[key] for row in group)
                   for key in ("relational_target", "numeric_target")} for mode in outputs})
    return dict(slope=float(slope), counts=counts, rows=records)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    models = []
    with torch.inference_mode():
        for seed in (*SEEDS, NEW_SEED):
            directory = DIRECTIONS if seed == NEW_SEED else PRIOR
            path = directory / f"controller-{seed}.pt"
            model = Controller().eval()
            model.load_state_dict(torch.load(path, weights_only=True))
            axis, _ = fit(model)
            models.append((seed, digest(path.read_bytes()), model, axis))
        studies = []
        for seed, identity, model, axis in models[1:]:
            result = evaluate(*models[0][2:], model, axis)
            studies.append(dict(seed=seed, weights_sha256=identity, **result))
            print(seed, result["counts"])
    raw = canonical(dict(profile="visuaml-comparison-alignment-v1", calibration=CALIBRATION,
                         holdout=HOLDOUT, source_seed=SEEDS[0], source_sha256=models[0][1],
                         studies=studies)) + b"\n"
    path = OUT / "replay.json"
    if args.record:
        path.write_bytes(raw)
    else:
        assert raw == path.read_bytes(), path


if __name__ == "__main__":
    main()
