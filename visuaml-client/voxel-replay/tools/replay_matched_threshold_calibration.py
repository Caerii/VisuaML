"""Test small joint threshold changes with identical feedback deltas."""
from itertools import product

from replay_activation_intervention import BASE
from replay_comparison_alignment import cases
from replay_comparison_mappings import design
from replay_comparison_strength import fresh_cases
from replay_intervention_diagnosis import CALIBRATION, HOLDOUT
from replay_paired_comparison_calibration import OFFSETS, calibration, evaluation, run_study

OUT = BASE / "release/examples/matched-threshold-calibration"
SHIFTS = (-2, -1, 1, 2)


def matched_calibration():
    fixed, joint = [], []
    for threshold, rule, a, b, shift in product(CALIBRATION, range(3), OFFSETS, OFFSETS, SHIFTS):
        if a == b or threshold + shift in HOLDOUT:
            continue
        live = (threshold + a, threshold, rule)
        feedback = threshold + b
        fixed.append((live, (feedback, threshold, rule)))
        joint.append((live, (feedback, threshold + shift, rule)))
    assert all(live == control_live and donor[::2] == control[::2]
               for (live, donor), (control_live, control) in zip(joint, fixed, strict=True))
    return dict(v97_fixed=calibration()["paired_fixed"], matched_fixed=fixed,
                matched_joint=joint, matched_combined=fixed + joint)


def heldout():
    for limit, rule, other, side in product(HOLDOUT, range(3), range(3), (-1, 1)):
        if rule == other:
            continue
        value = limit + (4 if side > 0 else -5)
        live = (value, limit, rule)
        above = int(side > 0)
        for family, feedback, threshold in (
            ("threshold_switch", value, value + above),
            ("feedback_switch", limit - above, limit),
            ("conflict_switch", value + side, value + side + above),
            ("same_side", value - side, limit),
            ("conflict_preserve", value - 6 * side, value - 6 * side + 1 - above),
        ):
            yield family, live, (feedback, threshold, other)


if __name__ == "__main__":
    run_study(OUT, "visuaml-matched-threshold-calibration-v1", heldout(), matched_calibration(),
              [*cases(), *fresh_cases(), *design(), *evaluation()])
