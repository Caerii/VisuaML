"""Transfer a calibrated feedback change without replacing the recipient state."""
from argparse import ArgumentParser
from itertools import product
import json
from unittest.mock import patch

import torch

from paired_routing import Router
from replay_activation_intervention import BASE, PACKED, SEEDS, STUDY, Controller
from replay_delayed_rules import EXPECTED, RULES, donor_activation, execute_witness, observe_rule, tensor_args
from replay_routing_subspaces import exact_tool
from study_router_learning import REGISTRY, THRESHOLDS
from visuaml.replay._codec import canonical, digest
from visuaml.replay.interventions import ActivationSite, run_intervention
from visuaml.replay.pipeline import tensor_identity

OUTPUT = BASE/"release/examples/additive-routing"
CALIBRATION = 9
PRE = ActivationSite("router.condition.0")
POST = ActivationSite("router.condition.1")
GAPS = (1, 3)


def measure(model, deltas, values):
    rows = []
    for limit, family, direction, gap in product(THRESHOLDS, range(len(REGISTRY)), (-1, 1), GAPS):
        low, high = values[limit, -gap][0], values[limit, 0][0]
        actual, target = (low, high) if direction > 0 else (high, low)
        target_side = int(direction > 0)
        row = dict(limit=limit, family=family, direction=direction, gap=gap,
                   feedback=actual, target_feedback=target,
                   donor_activation=tensor_identity(deltas[PRE].numpy()), rules={})
        for name, (relation, _) in RULES.items():
            args, _ = tensor_args(limit, actual, relation, family)
            desired, _ = tensor_args(limit, target, relation, family)
            counterfactual = model(*desired)
            _, hidden, _ = run_intervention(model, PRE, args)
            _, post, _ = run_intervention(model, POST, args)
            edits = {
                "same_delta": (PRE, hidden+direction*deltas[PRE]),
                "scaled_delta": (PRE, hidden+direction*gap*deltas[PRE]),
                "whole_donor": (PRE, donor_activation(model, PRE, CALIBRATION,
                                                     CALIBRATION if direction > 0 else CALIBRATION-1, family)),
                "post_tanh_scaled": (POST, post+direction*gap*deltas[POST]),
                "reverse_delta": (PRE, hidden-direction*gap*deltas[PRE]),
            }
            expected = EXPECTED[name][target_side]
            observations = {}
            for condition, (site, replacement) in edits.items():
                seen = observe_rule(model, site, replacement, limit, actual, name, family,
                                    reference_logits=counterfactual[1])
                observations[condition] = dict(**seen, correct=seen["action"] == expected)
                if condition == "scaled_delta":
                    assert seen["action"] == expected and seen["max_logit_error"] < 1e-4
            row["rules"][name] = dict(expected=expected, **observations)
        rows.append(row)
    return rows


def summarize(rows):
    conditions = next(iter(rows[0]["rules"].values())).keys() - {"expected"}
    result = {}
    for gap in GAPS:
        selected = [row for row in rows if row["gap"] == gap]
        result[str(gap)] = {condition: dict(correct=sum(row["rules"][rule][condition]["correct"]
                                                       for row in selected for rule in RULES),
                                            assigned=len(selected)*len(RULES),
                                            max_logit_error=max(row["rules"][rule][condition]["max_logit_error"]
                                                                for row in selected for rule in RULES))
                            for condition in sorted(conditions)}
    return result


def experiment():
    assert CALIBRATION not in THRESHOLDS
    tool = BASE/"release/examples/pipeline/tool.wasm"
    values = {(limit, offset): exact_tool(tool, limit+offset-3, limit+offset)
              for limit, offset in product(THRESHOLDS, (0, -1, -3))}
    donors = {value: exact_tool(tool, value-3, value) for value in (CALIBRATION-1, CALIBRATION)}
    studies = []
    for seed in SEEDS:
        path = STUDY/"learned-checkpoints"/f"width-32-seed-{seed}-step-600.pt"
        router = Router(32)
        router.load_state_dict(torch.load(path, weights_only=True))
        model = Controller(router).eval()
        deltas = {site: donor_activation(model, site, CALIBRATION, donors[CALIBRATION][0], 0)
                  - donor_activation(model, site, CALIBRATION, donors[CALIBRATION-1][0], 0)
                  for site in (PRE, POST)}
        theoretical = router.condition[0].weight[:, 0]/31
        identity_error = float((deltas[PRE][0]-theoretical).abs().max())
        assert torch.allclose(deltas[PRE][0], theoretical, atol=2e-6, rtol=1e-5)
        outcomes = measure(model, deltas, values)
        studies.append(dict(seed=seed, weights_sha256=digest(path.read_bytes()),
                            directions={site.module: delta.tolist() for site, delta in deltas.items()},
                            linear_identity_error=identity_error, outcomes=outcomes, summary=summarize(outcomes)))
    first = next(row for row in studies[0]["outcomes"] if row["gap"] == 1 and row["direction"] == 1)
    witness = dict(first, rules={name: outcome["same_delta"] for name, outcome in first["rules"].items()})
    traces, reports = execute_witness(witness, values[first["limit"], -1][1], donors[CALIBRATION][1])
    reports["donor_baseline"] = donors[CALIBRATION-1][1]
    # Label the frozen artifact as an additive direction in this replay.
    for branch in traces:
        branch["events"][1]["boundary"] = "donor_delta_frozen"
    return dict(profile="visuaml-additive-routing-v1", calibration_threshold=CALIBRATION,
                evaluation_thresholds=THRESHOLDS, studies=studies, witness=dict(case=witness, traces=traces)), reports


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with patch(PACKED, side_effect=AssertionError("Packed fallback entered")), torch.inference_mode():
        result, reports = experiment()
    for name, data in (("replay", result), *[(key+"-report", value) for key, value in reports.items()]):
        path = OUTPUT/(name+".json")
        if args.record:
            path.write_bytes(canonical(data)+b"\n")
        else:
            assert canonical(json.loads(path.read_bytes())) == canonical(data), path
    print(json.dumps([dict(seed=study["seed"], summary=study["summary"]) for study in result["studies"]]))


if __name__ == "__main__":
    main()
