"""Reveal a rule after verified feedback and reuse a single donor activation."""
from argparse import ArgumentParser
from itertools import product
import json
from unittest.mock import patch

import torch

from paired_routing import Router
from replay_activation_intervention import BASE, PACKED, SEEDS, STUDY, Controller
from replay_crossed_routing import FACTOR_SITE, Factorized, inputs, train_comparison
from replay_routing_subspaces import exact_tool
from study_router_learning import REGISTRY, THRESHOLDS
from visuaml.replay._codec import canonical, digest
from visuaml.replay.interventions import ActivationSite, run_intervention
from visuaml.replay.pipeline import observation, tensor_identity

OUTPUT = BASE / "release/examples/delayed-rules"
CALIBRATION = (9, 17, 23)
CANDIDATE_SITES = ("router.condition.0", "router.condition.1", "router.condition.2")
# Role 1 is Boost and role 2 is Trim in the original registry.
RULES = {"above_boost": (1, {1: "boost", 2: "trim"}),
         "above_trim": (0, {1: "boost", 2: "trim"}),
         "above_abstain": (0, {1: "boost", 2: "abstain"})}
EXPECTED = {"above_boost": ("trim", "boost"), "above_trim": ("boost", "trim"),
            "above_abstain": ("boost", "abstain")}


def context(limit, fact, rule=0, family=0, order=(1, 2)):
    return dict(limit=limit, recipient_side=fact, recipient_rule=rule,
                recipient_order=order, registry_variant=family)


def tensor_args(limit, value, relation, family=0):
    return inputs([context(limit, 0, relation, family)], "recipient", {(limit, 0): value})


def donor_activation(model, site, limit, value, family):
    args, _ = tensor_args(limit, value, 0, family)
    _, hidden, _ = run_intervention(model, site, args)
    return hidden


def observe_rule(model, site, activation, limit, value, rule, family, *, reference_logits=None):
    relation, mapping = RULES[rule]
    args, permutation = tensor_args(limit, value, relation, family)
    baseline, _, _ = run_intervention(model, site, args)
    output, _, delivered = run_intervention(model, site, args, replacement=activation)
    assert torch.equal(delivered, activation)
    assert torch.equal(output[0], baseline[0])

    def action(logits):
        role = permutation[int(logits.argmax())]
        return mapping.get(role, "unmapped")

    result = {"action": action(output[1]), "baseline": action(baseline[1]),
              "activation": tensor_identity(delivered.numpy())}
    if reference_logits is not None:
        result["max_logit_error"] = float((output[1]-reference_logits).abs().max())
    return result


def select_site(model):
    # Site selection sees only the calibration thresholds. Rank by transfer
    # under the opposing rule and break ties in execution order.
    scores = {}
    for module in CANDIDATE_SITES:
        site = ActivationSite(module)
        correct = 0
        for limit, donor_side in product(CALIBRATION, (0, 1)):
            activation = donor_activation(model, site, limit, limit-1+donor_side, 0)
            outcome = observe_rule(model, site, activation, limit, limit-donor_side, "above_boost", 0)
            correct += outcome["action"] == EXPECTED["above_boost"][donor_side]
        scores[module] = correct
    selected = max(CANDIDATE_SITES, key=scores.__getitem__)
    return ActivationSite(selected), scores


def exercise(model, site, limit, donor_side, family, recipient_value, donor_value):
    activation = donor_activation(model, site, limit, donor_value, family)
    frozen_identity = tensor_identity(activation.numpy())
    output = {}
    for rule in RULES:
        seen = observe_rule(model, site, activation, limit, recipient_value, rule, family)
        assert seen["activation"] == frozen_identity
        output[rule] = dict(expected=EXPECTED[rule][donor_side],
                            expected_baseline=EXPECTED[rule][1-donor_side], **seen)
    return dict(limit=limit, donor_side=donor_side, family=family, feedback=recipient_value,
                donor_value=donor_value, donor_activation=frozen_identity, rules=output)


def execute_witness(case, first_report, donor_report):
    reports = {"first": first_report, "donor": donor_report}
    rows = []
    for rule, outcome in case["rules"].items():
        events = [("tool_result", case["feedback"], "wasm"),
                  ("activation_frozen", case["donor_activation"], "pytorch"),
                  ("rule_revealed", rule, "host"),
                  ("decision", outcome["action"], "pytorch")]
        value = case["feedback"]
        if outcome["action"] == "boost":
            guest = BASE/"release/examples/pipeline/changed-tool.wasm"
            number, reports[rule] = exact_tool(guest, value, value+5)
        elif outcome["action"] == "trim":
            guest = BASE/"release/examples/routing-subspaces/trim.wasm"
            number, reports[rule] = exact_tool(guest, value, value-2)
        else:
            assert outcome["action"] == "abstain"
            number = None
        if number is not None:
            events.extend((("second_request", outcome["action"], "host"),
                           ("second_tool_result", number, "wasm")))
        rows.append(dict(rule=rule, result=number, events=[observation(rule, edge, payload, runtime=runtime)
                                                         for edge, payload, runtime in events]))
    assert len(reports) == 4
    assert all(row["rule"] != "above_abstain" or len(row["events"]) == 4 for row in rows)
    return rows, reports


def experiment():
    assert not set(CALIBRATION) & set(THRESHOLDS)
    initial = BASE/"release/examples/pipeline/tool.wasm"
    captures = {(limit, side): exact_tool(initial, limit-4+side, limit-1+side)
                for limit, side in product(THRESHOLDS, (0, 1))}
    studies = []
    for seed in SEEDS:
        weights = STUDY/"learned-checkpoints"/f"width-32-seed-{seed}-step-600.pt"
        router = Router(32)
        router.load_state_dict(torch.load(weights, weights_only=True))
        joint = Controller(router).eval()
        factor = Factorized(router).eval()
        train_comparison(factor, seed)
        with torch.inference_mode():
            site, calibration = select_site(joint)
            suites = {}
            for kind, model, location in (("joint", joint, site), ("factorized", factor, FACTOR_SITE)):
                cases = [exercise(model, location, limit, side, family, captures[limit, 1-side][0],
                                  captures[limit, side][0])
                         for limit, side, family in product(THRESHOLDS, (0, 1), range(len(REGISTRY)))]
                counts = {name: sum(case["rules"][name]["action"] == case["rules"][name]["expected"]
                                    for case in cases) for name in RULES}
                assert all(outcome["baseline"] == outcome["expected_baseline"]
                           for case in cases for outcome in case["rules"].values())
                suites[kind] = dict(site=location.module, cases=cases, correct=counts)
        studies.append(dict(seed=seed, weights_sha256=digest(weights.read_bytes()), calibration=calibration,
                            factor_weights={name: tensor.detach().tolist()
                                            for name, tensor in factor.comparison.state_dict().items()},
                            **suites))
    witness = next(case for case in studies[0]["factorized"]["cases"] if case["donor_side"] == 1)
    traces, reports = execute_witness(witness, captures[witness["limit"], 0][1], captures[witness["limit"], 1][1])
    return dict(profile="visuaml-delayed-rule-v1", calibration_thresholds=CALIBRATION,
                evaluation_thresholds=THRESHOLDS, studies=studies,
                witness=dict(case=witness, traces=traces)), reports


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with patch(PACKED, side_effect=AssertionError("Packed fallback entered")):
        result, reports = experiment()
    for name, data in (("replay", result), *[(key+"-report", value) for key, value in reports.items()]):
        path = OUTPUT/(name+".json")
        if args.record:
            path.write_bytes(canonical(data)+b"\n")
        else:
            assert canonical(json.loads(path.read_bytes())) == canonical(data), path
    print(json.dumps([dict(seed=s["seed"], selected=s["joint"]["site"],
                           joint=s["joint"]["correct"], factorized=s["factorized"]["correct"])
                      for s in result["studies"]]))


if __name__ == "__main__":
    main()
