"""Replay v98 failures with exact WASM and a calibrated argument probe."""
from argparse import ArgumentParser
from itertools import product
import json
from unittest.mock import patch

import torch

from replay_activation_intervention import BASE, PACKED
from replay_comparison_alignment import activations
from replay_comparison_direction import NEW_SEED, OUT as DIRECTIONS, fit
from replay_comparison_mappings import compare
from replay_compositional_interventions import Controller, execute
from replay_intervention_diagnosis import CALIBRATION, OUT as PRIOR, SEEDS
from replay_live_tool_interventions import OUT as LIVE, request
from replay_matched_threshold_calibration import OUT as MATCHED, heldout, matched_calibration
from replay_routing_subspaces import exact_tool
from visuaml.replay._codec import canonical, digest

OUT = BASE / "release/examples/live-preservation-replay"


def fit_argument_probe(model):
    rows = list(product(range(32), CALIBRATION, range(3)))
    _, state = activations(model, rows)
    values = torch.tensor([row[0] for row in rows], dtype=torch.float64)
    features = state[:, 0].double()
    mean = features.mean(0)
    centered = features - mean
    weights = torch.linalg.solve(
        centered.T @ centered + .01 * torch.eye(centered.shape[1], dtype=centered.dtype),
        centered.T @ (values - values.mean()))
    return weights, mean, values.mean()


def experiment():
    binary = LIVE / "first-tool.wasm"
    saved = json.loads((MATCHED / "replay.json").read_bytes())
    design = list(heldout())
    models = []
    for seed in (*SEEDS, NEW_SEED):
        path = (DIRECTIONS if seed == NEW_SEED else PRIOR) / f"controller-{seed}.pt"
        model = Controller().eval()
        model.load_state_dict(torch.load(path, weights_only=True))
        axis, _ = fit(model)
        models.append((seed, digest(path.read_bytes()), model, axis))
    cases, traces = [], {}
    for (seed, identity, model, axis), study in zip(models[1:], saved["studies"], strict=True):
        assert seed == study["seed"] and identity == study["weights_sha256"]
        selected = [(i, row) for i, row in enumerate(study["rows"])
                    if not row["family"].endswith("_switch")
                    and row["scores"]["matched_fixed"]["action"] != row["relational_target"]]
        verified_inputs, live_cases = [], []
        for index, row in selected:
            family, live, donor = design[index]
            assert [list(live), list(donor)] == [row["live"], row["donor"]]
            label = f"{seed}-{index}"
            actual, traces[label + "-first"] = exact_tool(binary, live[0] - 1, live[0])
            alternate, traces[label + "-donor"] = exact_tool(binary, donor[0] - 1, donor[0])
            verified_inputs.append((family, (actual, *live[1:]), (alternate, *donor[1:])))
            live_cases.append((label, index, row, actual, alternate))

        weights, mean, center = fit_argument_probe(model)
        probe_states = {}

        def capture(mode, state):
            if mode == "matched_fixed":
                probe_states[mode] = (state[:, 0].double() - mean) @ weights + center

        result = compare(*models[0][2:], model, axis, verified_inputs,
                         calibration_pairs={"matched_fixed": matched_calibration()["matched_fixed"]},
                         observe=capture)
        _, baseline_state = activations(model, [live for _, live, _ in verified_inputs])
        baseline_values = (baseline_state[:, 0].double() - mean) @ weights + center
        for i, ((label, index, saved_row, live, donor), row) in enumerate(
                zip(live_cases, result["rows"], strict=True)):
            target = row["relational_target"]
            actual = row["scores"]["matched_fixed"]["action"]
            assert actual == saved_row["scores"]["matched_fixed"]["action"] and actual != target
            assert row["scores"]["baseline"]["action"] == target
            predicted = float(probe_states["matched_fixed"][i])
            baseline_predicted = float(baseline_values[i])
            candidates = sorted((live, donor))
            bound = min(candidates, key=lambda value: abs(value - predicted))
            baseline_bound = min(candidates, key=lambda value: abs(value - baseline_predicted))
            outcomes = {}
            for mode, action, argument in (("expected", target, live), ("routing_only", actual, live),
                                           ("learned_argument", actual, bound)):
                returned, trace = execute(action, argument)
                if trace is not None:
                    traces[f"{label}-{mode}"] = trace
                outcomes[mode] = dict(request=request(action, argument), returned=returned)
            cases.append(dict(seed=seed, prior_row=index, first_return=live, donor_return=donor,
                              weights_sha256=identity, logits=row["scores"]["matched_fixed"]["logits"],
                              first_failure="second_tool_choice", probe_value=predicted,
                              baseline_probe_value=baseline_predicted, baseline_bound=baseline_bound,
                              bound_argument=bound, binding_correct=bound == live, outcomes=outcomes))
    assert len(cases) == 4
    return dict(profile="visuaml-live-preservation-v1", first_tool_sha256=digest(binary.read_bytes()),
                cases=cases), traces


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with torch.inference_mode(), patch(PACKED, side_effect=AssertionError("Packed fallback entered")):
        report, traces = experiment()
    for name, value in {"replay": report, **traces}.items():
        path = OUT / f"{name}.json"
        data = canonical(value) + b"\n"
        if args.record:
            path.write_bytes(data)
        else:
            assert path.read_bytes() == data, path
    for item in report["cases"]:
        print(item)


if __name__ == "__main__":
    main()
