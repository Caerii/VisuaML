"""Transfer the donor's threshold side using the recipient's decoded margin."""
from argparse import ArgumentParser
from itertools import product
import json
from unittest.mock import patch

import torch

from paired_routing import Router
from replay_activation_intervention import BASE, PACKED, SEEDS, STUDY, Controller
from replay_additive_routing import PRE
from replay_delayed_rules import EXPECTED, RULES, execute_witness, observe_rule, tensor_args
from replay_routing_subspaces import exact_tool
from study_router_learning import REGISTRY, THRESHOLDS
from visuaml.replay._codec import canonical, digest
from visuaml.replay.interventions import run_intervention
from visuaml.replay.pipeline import tensor_identity

OUTPUT = BASE / "release/examples/margin-transfer"
ANCHOR = (8, 9, 0)
DONORS = (7, 8, 9, 11)
GAPS = (1, 3, 5)


def hidden(model, feedback, threshold, relation):
    args, _ = tensor_args(threshold, feedback, relation)
    _, value, _ = run_intervention(model, PRE, args)
    return value[0]


class ActivationCoordinates:
    """Calibrate feedback, threshold and rule coordinates from four forwards."""

    def __init__(self, model):
        feedback, threshold, relation = ANCHOR
        self.origin = hidden(model, *ANCHOR).double()
        self.axes = torch.stack([
            hidden(model, feedback+1, threshold, relation).double()-self.origin,
            hidden(model, feedback, threshold+1, relation).double()-self.origin,
            hidden(model, feedback, threshold, relation+1).double()-self.origin,
        ])
        assert torch.linalg.matrix_rank(self.axes) == 3
        self.decoder = torch.linalg.pinv(self.axes)

    def read(self, activation):
        return (activation.double()-self.origin) @ self.decoder + torch.tensor(ANCHOR)

    def edit(self, activation, above, *, axis):
        feedback, threshold, _ = self.read(activation)[0]
        margin = feedback-threshold+.5
        desired = margin.abs() if above else -margin.abs()
        change = desired-margin
        distance = change if axis == 0 else -change
        return activation+(distance*self.axes[axis]).to(activation.dtype)


def evaluate(model, coordinate, sources, donor_bits):
    cases = []
    for threshold, gap, side, family, donor in product(
            THRESHOLDS, GAPS, (0, 1), range(len(REGISTRY)), DONORS):
        low = threshold-(gap+1)//2
        feedback = sources[low+side*gap][0]
        above = donor_bits[donor]
        expected_feedback = feedback if above == bool(side) else 2*threshold-feedback-1
        expected_threshold = threshold+feedback-expected_feedback
        expected_coordinates = ((expected_feedback, threshold), (feedback, expected_threshold))
        results = {}
        for rule, (relation, _) in RULES.items():
            args, _ = tensor_args(threshold, feedback, relation, family)
            _, original, _ = run_intervention(model, PRE, args)
            decoded = coordinate.read(original)[0]
            assert torch.allclose(decoded, torch.tensor([feedback, threshold, relation], dtype=decoded.dtype),
                                  atol=5e-5, rtol=0)
            controls = {}
            for axis, name in ((0, "feedback_axis"), (1, "threshold_axis")):
                edited = coordinate.edit(original, above, axis=axis)
                if above == bool(side):
                    assert torch.equal(edited, original)
                assert torch.allclose(coordinate.edit(edited, above, axis=axis), edited, atol=1e-6, rtol=0)
                expected_value, expected_limit = expected_coordinates[axis]
                direct, _ = tensor_args(expected_limit, expected_value, relation, family)
                observed = observe_rule(model, PRE, edited, threshold, feedback, rule, family,
                                        reference_logits=model(*direct)[1])
                got = coordinate.read(edited)[0]
                assert torch.allclose(got, torch.tensor([expected_value, expected_limit, relation], dtype=got.dtype),
                                      atol=1e-4, rtol=0)
                assert observed["max_logit_error"] < 2e-4
                assert observed["action"] == EXPECTED[rule][int(above)]
                controls[name] = observed
            feedback_args, _ = tensor_args(threshold, expected_feedback, relation, family)
            threshold_args, _ = tensor_args(expected_threshold, feedback, relation, family)
            route_error = float((model(*feedback_args)[1]-model(*threshold_args)[1]).abs().max())
            results[rule] = dict(**controls, same_margin_logit_difference=route_error)
        cases.append(dict(limit=threshold, gap=gap, recipient_side=side, family=family, donor=donor,
                          transferred_side=above, feedback=feedback, expected_feedback=expected_feedback,
                          expected_threshold=expected_threshold, rules=results))
    return cases


def experiment():
    guest = BASE/"release/examples/pipeline/tool.wasm"
    needed = {limit-(gap+1)//2+side*gap
              for limit, gap, side in product(THRESHOLDS, GAPS, (0, 1))} | set(DONORS)
    sources = {value: exact_tool(guest, value-3, value) for value in sorted(needed)}
    studies = []
    for seed in SEEDS:
        path = STUDY/"learned-checkpoints"/f"width-32-seed-{seed}-step-600.pt"
        router = Router(32)
        router.load_state_dict(torch.load(path, weights_only=True))
        model = Controller(router).eval()
        coordinate = ActivationCoordinates(model)
        donor_bits = {}
        for value in DONORS:
            decoded = coordinate.read(hidden(model, sources[value][0], ANCHOR[1], 0))
            donor_bits[value] = bool(decoded[0]-decoded[1]+.5 > 0)
            assert donor_bits[value] == (value >= ANCHOR[1])
        cases = evaluate(model, coordinate, sources, donor_bits)
        for i in range(0, len(cases), len(DONORS)):
            group = cases[i:i+len(DONORS)]
            assert group[0]["rules"] == group[1]["rules"] and group[2]["rules"] == group[3]["rules"]
        studies.append(dict(seed=seed, weights_sha256=digest(path.read_bytes()), origin=coordinate.origin.tolist(),
                            axes=coordinate.axes.tolist(), cases=cases))
    example = next(row for row in studies[0]["cases"] if row["gap"] == 5 and
                   row["recipient_side"] == 0 and row["donor"] == 11)
    witness = dict(example, donor_activation=tensor_identity(torch.tensor([example["transferred_side"]]).numpy()),
                   rules={rule: result["feedback_axis"] for rule, result in example["rules"].items()})
    traces, reports = execute_witness(witness, sources[witness["feedback"]][1], sources[witness["donor"]][1])
    for branch in traces:
        branch["events"][1]["boundary"] = "donor_bit_frozen"
    all_cases = [case for study in studies for case in study["cases"]]
    policies = [result for case in all_cases for result in case["rules"].values()]
    maximum = max(row["same_margin_logit_difference"] for row in policies)
    summary = dict(cases=len(all_cases), decisions_per_axis=len(policies),
                   side_changes=sum(case["recipient_side"] != case["transferred_side"] for case in all_cases),
                   max_equal_margin_logit_difference=maximum,
                   max_feedback_patch_error=max(row["feedback_axis"]["max_logit_error"] for row in policies),
                   max_threshold_patch_error=max(row["threshold_axis"]["max_logit_error"] for row in policies))
    return dict(profile="visuaml-margin-transfer-v1", calibration_anchor=ANCHOR,
                evaluation_thresholds=THRESHOLDS, donor_values=DONORS, summary=summary,
                studies=studies, witness=dict(case=witness, traces=traces)), reports


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
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
