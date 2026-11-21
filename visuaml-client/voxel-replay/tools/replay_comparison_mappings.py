"""Compare quadratic scalar and rank-two linear activation mappings."""
from argparse import ArgumentParser
from itertools import product

import torch

from replay_activation_intervention import BASE
from replay_comparison_alignment import activations, cases, evaluate
from replay_comparison_direction import NEW_SEED, OUT as DIRECTIONS, fit
from replay_comparison_strength import fresh_cases
from replay_compositional_interventions import Controller, RULES
from replay_intervention_diagnosis import CALIBRATION, HOLDOUT, OUT as PRIOR, SEEDS, SITE
from visuaml.replay._codec import canonical, digest
from visuaml.replay.interventions import run_intervention

OUT = BASE / "release/examples/comparison-mappings"


def design():
    for limit, rule, other, side in product(HOLDOUT, range(3), range(3), (-1, 1)):
        if rule == other:
            continue
        value = limit + (2 if side > 0 else -3)
        live = (value, limit, rule)
        shift = 2 * side
        for family, donor_value, donor_limit in (
            ("threshold_switch", value, value + int(side > 0)),
            ("feedback_switch", limit - int(side > 0), limit),
            ("conflict_switch", value + shift, value + shift + int(side > 0)),
            ("same_side", value + shift, limit),
            ("conflict_preserve", value - 3 * side, value - 3 * side + int(side < 0)),
        ):
            yield family, live, (donor_value, donor_limit, other)


def pair_state(model, rows):
    _, state = activations(model, rows)
    return state[:, :2].flatten(1)


def basis(model, axis):
    low, high = CALIBRATION
    low_rows = [(value, low, rule) for value, rule in product((low - 1, low, high - 1, high), range(3))]
    high_rows = [(value, high, rule) for value, _, rule in low_rows]
    contrast = (pair_state(model, high_rows) - pair_state(model, low_rows)).mean(0)
    residual = contrast - (contrast @ axis) * axis
    return torch.stack((axis, residual / residual.norm()), dim=1)


def compare(source, source_axis, receiver, axis, rows, *, calibration_pairs=None):
    reference = evaluate(source, source_axis, receiver, axis, design=rows)
    source_basis, receiver_basis = basis(source, source_axis), basis(receiver, axis)
    calibration = [(limit + offset, limit, rule) for limit, offset, rule in
                   product(CALIBRATION, (-2, -1, 0, 1), range(3))]
    x = pair_state(source, calibration).double() @ source_basis.double()
    y = pair_state(receiver, calibration).double() @ receiver_basis.double()
    mean, scale = x.mean(0), x.std(0)

    def features(values, kind):
        normalized = (values - mean) / scale
        one = torch.ones_like(normalized[:, 0])
        first = normalized[:, 0]
        if kind == "quadratic":
            return torch.stack((one, first, first.square()), dim=1)
        return torch.stack((one, first, normalized[:, 1]), dim=1)

    live = [row[1] for row in rows]
    donor = [row[2] for row in rows]
    live_source = pair_state(source, live).double() @ source_basis.double()
    donor_source = pair_state(source, donor).double() @ source_basis.double()
    args = torch.tensor(live).unbind(1)
    _, current, _ = run_intervention(receiver, SITE, args)
    coefficients = {}
    configurations = [("quadratic", 1), ("two_inputs_one_direction", 1), ("rank_two", 2)]
    pair_data = {}
    for name, pairs in (calibration_pairs or {}).items():
        a, b = zip(*pairs, strict=True)
        a_source = pair_state(source, a).double() @ source_basis.double()
        b_source = pair_state(source, b).double() @ source_basis.double()
        a_target = pair_state(receiver, a).double() @ axis.double()
        b_target = pair_state(receiver, b).double() @ axis.double()
        pair_data[name] = ((b_source - a_source) / scale, (b_target - a_target)[:, None])
        configurations.append((name, 1))
    for name, width in configurations:
        if name in pair_data:
            matrix, target = pair_data[name]
            evaluation_delta = (donor_source - live_source) / scale
        else:
            matrix, target = features(x, name), y[:, :width]
            evaluation_delta = features(donor_source, name) - features(live_source, name)
        assert torch.linalg.matrix_rank(matrix) == matrix.shape[1]
        coeff = torch.linalg.lstsq(matrix, target, driver="gelsd").solution
        delta = evaluation_delta @ coeff
        edit = (delta @ receiver_basis[:, :width].double().T).to(current.dtype)
        replacement = current.clone()
        replacement[:, :2] += edit.reshape_as(current[:, :2])
        logits, _, applied = run_intervention(receiver, SITE, args, replacement=replacement)
        assert torch.equal(applied[:, 2:], current[:, 2:])
        coefficients[name] = coeff.tolist()
        for row, values in zip(reference["rows"], logits, strict=True):
            row["scores"][name] = dict(action=int(values.argmax()), logits=values.tolist())

    summary = {}
    for mode in ("baseline", "aligned", "local", "quadratic", "two_inputs_one_direction", "rank_two", *pair_data):
        switches, preserves = [], []
        for row in reference["rows"]:
            value, limit, rule = row["live"]
            target = row["relational_target"]
            natural = RULES[rule][int(value >= limit)]
            assert (target != natural) == row["family"].endswith("_switch")
            group = switches if target != natural else preserves
            group.append(row["scores"][mode]["action"] == target)
        summary[mode] = dict(switches=sum(switches), switch_cases=len(switches),
                             preserves=sum(preserves), preserve_cases=len(preserves))
    return dict(summary=summary, coefficients=coefficients, calibration_mean=mean.tolist(),
                calibration_scale=scale.tolist(), source_basis=source_basis.tolist(),
                receiver_basis=receiver_basis.tolist(), rows=reference["rows"])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = list(design())
    assert len({(live, donor) for _, live, donor in rows}) == len(rows)
    assert {(live, donor) for _, live, donor in rows}.isdisjoint(
        (live, donor) for _, live, donor in [*cases(), *fresh_cases()])
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
            result = compare(*models[0][2:], model, axis, rows)
            print(seed, result["summary"])
            studies.append(dict(seed=seed, weights_sha256=identity, **result))
    payload = canonical(dict(profile="visuaml-comparison-mappings-v1", calibration=CALIBRATION,
                             holdout=HOLDOUT, source_seed=SEEDS[0], source_sha256=models[0][1],
                             studies=studies)) + b"\n"
    path = OUT / "replay.json"
    if args.record:
        path.write_bytes(payload)
    else:
        assert path.read_bytes() == payload, path


if __name__ == "__main__":
    main()
