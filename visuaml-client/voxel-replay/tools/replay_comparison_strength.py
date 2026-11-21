"""Sweep aligned patch strength on fresh combinations without refitting."""
from argparse import ArgumentParser
from itertools import product
from statistics import median

import torch

from replay_activation_intervention import BASE
from replay_comparison_alignment import cases as prior_cases, evaluate
from replay_comparison_direction import NEW_SEED, OUT as DIRECTIONS, fit
from replay_compositional_interventions import Controller, RULES
from replay_intervention_diagnosis import HOLDOUT, OUT as PRIOR, SEEDS
from visuaml.replay._codec import canonical, digest

OUT = BASE / "release/examples/comparison-strength"
STRENGTHS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)


def fresh_cases():
    for threshold, rule, donor_rule, high in product(HOLDOUT, range(3), range(3), (False, True)):
        if rule == donor_rule:
            continue
        feedback = threshold + (1 if high else -2)
        direction = 1 if high else -1
        live = (feedback, threshold, rule)
        yield "threshold_switch", live, (feedback, feedback + int(high), donor_rule)
        yield "conflict_switch", live, (feedback + 3 * direction, feedback + 3 * direction + int(high), donor_rule)
        yield "feedback_switch", live, (threshold - int(high), threshold, donor_rule)
        yield "same_side", live, (feedback + direction, threshold, donor_rule)
        yield "conflict_preserve", live, (feedback - 4 * direction, feedback - 4 * direction + int(not high), donor_rule)


def margin(row, key):
    logits = row["scores"][key]["logits"]
    target = row["relational_target"]
    return logits[target] - max(value for i, value in enumerate(logits) if i != target)


def summarize(rows):
    switches, preserves = [], []
    for row in rows:
        feedback, limit, rule = row["live"]
        natural = RULES[rule][int(feedback >= limit)]
        switching = row["relational_target"] != natural
        assert switching == row["family"].endswith("_switch")
        (switches if switching else preserves).append(row)

    def correct(row, key):
        return row["scores"][key]["action"] == row["relational_target"]

    baseline_misses = [row for row in switches if not correct(row, "aligned")]
    return dict(switch_cases=len(switches), preserve_cases=len(preserves),
                local_switches=sum(correct(row, "local") for row in switches),
                local_preserves=sum(correct(row, "local") for row in preserves),
                unit_strength_misses=len(baseline_misses),
                sweep=[dict(gain=gain,
                            switches=sum(correct(row, key) for row in switches),
                            unwanted_switches=sum(not correct(row, key) and correct(row, "baseline")
                                                  for row in preserves),
                            preserve_errors=sum(not correct(row, key) for row in preserves),
                            recovered=sum(correct(row, key) for row in baseline_misses),
                            median_miss_margin=median(margin(row, key) for row in baseline_misses)
                            if baseline_misses else None)
                       for gain in STRENGTHS for key in (f"strength_{gain:g}",)])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    design = list(fresh_cases())
    assert len({(live, donor) for _, live, donor in design}) == len(design)
    assert {(live, donor) for _, live, donor in design}.isdisjoint(
        (live, donor) for _, live, donor in prior_cases())
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
            data = evaluate(*models[0][2:], model, axis, design=design, strengths=STRENGTHS)
            summary = summarize(data["rows"])
            studies.append(dict(seed=seed, weights_sha256=identity, summary=summary, **data))
            print(seed, summary)
    report = dict(profile="visuaml-comparison-strength-v1", source_sha256=models[0][1], studies=studies)
    payload = canonical(report) + b"\n"
    path = OUT / "replay.json"
    if args.record:
        path.write_bytes(payload)
    else:
        assert payload == path.read_bytes(), path


if __name__ == "__main__":
    main()
