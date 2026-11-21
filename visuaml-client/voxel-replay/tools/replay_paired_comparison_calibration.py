"""Compare fixed-threshold and crossed-threshold delta calibration."""
from argparse import ArgumentParser
from itertools import product

import torch

from replay_activation_intervention import BASE
from replay_comparison_alignment import cases
from replay_comparison_direction import NEW_SEED, OUT as DIRECTIONS, fit
from replay_comparison_mappings import compare, design
from replay_comparison_strength import fresh_cases
from replay_compositional_interventions import Controller
from replay_intervention_diagnosis import CALIBRATION, HOLDOUT, OUT as PRIOR, SEEDS
from visuaml.replay._codec import canonical, digest

OUT = BASE / "release/examples/paired-comparison-calibration"
OFFSETS = (-4, -2, -1, 0, 1, 3)


def calibration():
    fixed, crossed = [], []
    for limit, rule, a, b in product(CALIBRATION, range(3), OFFSETS, OFFSETS):
        if a == b:
            continue
        other = next(value for value in CALIBRATION if value != limit)
        live = (limit + a, limit, rule)
        fixed.append((live, (limit + b, limit, rule)))
        crossed.append((live, (other + b, other, rule)))
    return dict(paired_fixed=fixed, paired_crossed=crossed, paired_combined=fixed + crossed)


def evaluation():
    for limit, rule, other, side in product(HOLDOUT, range(3), range(3), (-1, 1)):
        if rule == other:
            continue
        value = limit + (3 if side > 0 else -4)
        live = (value, limit, rule)
        swap = int(side > 0)
        for family, feedback, threshold in (
            ("threshold_switch", value, value + swap),
            ("feedback_switch", limit - swap, limit),
            ("conflict_switch", value + side, value + side + swap),
            ("same_side", value - side, limit),
            ("conflict_preserve", value - 4 * side, value - 4 * side + 1 - swap),
        ):
            yield family, live, (feedback, threshold, other)


def run_study(output, profile, rows, pairs, prior_designs):
    parser = ArgumentParser(description=profile)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = list(rows)
    identities = {(live, donor) for _, live, donor in rows}
    assert len(identities) == len(rows)
    assert identities.isdisjoint((live, donor) for _, live, donor in prior_designs)
    models = []
    with torch.inference_mode():
        for seed in (*SEEDS, NEW_SEED):
            path = (DIRECTIONS if seed == NEW_SEED else PRIOR) / f"controller-{seed}.pt"
            model = Controller().eval()
            model.load_state_dict(torch.load(path, weights_only=True))
            axis, _ = fit(model)
            models.append((seed, digest(path.read_bytes()), model, axis))
        studies = []
        for seed, identity, model, axis in models[1:]:
            result = compare(*models[0][2:], model, axis, rows, calibration_pairs=pairs)
            studies.append(dict(seed=seed, weights_sha256=identity, **result))
            print(seed, result["summary"])
    payload = canonical(dict(profile=profile,
                             calibration=pairs, source_sha256=models[0][1], studies=studies)) + b"\n"
    path = output / "replay.json"
    if args.record:
        path.write_bytes(payload)
    else:
        assert path.read_bytes() == payload, path


if __name__ == "__main__":
    run_study(OUT, "visuaml-paired-comparison-calibration-v1", evaluation(), calibration(),
              [*cases(), *fresh_cases(), *design()])
