"""Distinguish comparison transfer from preference transfer by crossing rules."""
from argparse import ArgumentParser
from itertools import product
import json
from unittest.mock import patch

import torch
from torch import nn

from paired_routing import Router, description_ids
from replay_activation_intervention import BASE, PACKED, SEEDS, SITE, STUDY, Controller
from replay_routing_subspaces import fit, exact_tool
from study_router_learning import REGISTRY, THRESHOLDS
from visuaml.replay._codec import canonical, digest
from visuaml.replay.interventions import ActivationSite, run_intervention, split_donor_delta
from visuaml.replay.pipeline import compare_observations, observation

OUTPUT = BASE / "release/examples/crossed-routing"
FACTOR_SITE = ActivationSite("comparison")


class Factorized(nn.Module):
    """A learned numeric comparison followed by explicit recipient rule binding."""

    def __init__(self, router):
        super().__init__()
        self.router = router
        self.comparison = nn.Linear(2, 1)

    def forward(self, registry, choices, feedback, threshold, relation):
        numeric = torch.stack((feedback, threshold), dim=-1).float()/31-.5
        score = self.comparison(numeric)
        gate = torch.sigmoid(torch.where(relation[:, None] == 0, score, -score))
        first, second = self.router.encode(choices).unbind(dim=1)
        query = gate*first+(1-gate)*second
        return self.router.first_scores(registry), (self.router.encode(registry)*query[:, None]).sum(-1)


def train_comparison(model, seed):
    torch.manual_seed(seed)
    model.comparison.reset_parameters()
    train = [(value, bound) for bound in range(1, 31) if bound not in THRESHOLDS
             for value in range(32)]
    pairs = torch.tensor(train)
    targets = (pairs[:, 0] < pairs[:, 1]).float()
    features = pairs.float()/31-.5
    optimizer = torch.optim.LBFGS(model.comparison.parameters(), max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        score = model.comparison(features).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(score, targets) + 1e-8*score.square().mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return len(train)


def role(side, relation, choice_order):
    return choice_order[side ^ relation]


def case_matrix():
    return [dict(limit=limit, recipient_side=rs, donor_side=ds, recipient_rule=rr, donor_rule=dr,
                 recipient_order=ro, donor_order=do, registry_variant=family)
            for limit, rs, ds, rr, dr, ro, do, family in product(
                THRESHOLDS, (0, 1), (0, 1), (0, 1), (0, 1), ((1, 2), (2, 1)),
                ((1, 2), (2, 1)), range(len(REGISTRY)))]


def inputs(rows, side, feedback):
    permutation = (2, 0, 1) if side == "recipient" else (1, 2, 0)
    registries, options, values, limits, rules = [], [], [], [], []
    for row in rows:
        descriptions = REGISTRY[row["registry_variant"]]
        registries.append([description_ids(descriptions[index]) for index in permutation])
        options.append([description_ids(descriptions[index]) for index in row[side+"_order"]])
        limit = row["limit"]
        values.append(feedback[limit, row[side+"_side"]])
        limits.append(limit)
        rules.append(row[side+"_rule"])
    return tuple(torch.tensor(column) for column in (registries, options, values, limits, rules)), permutation


def evaluate(model, site, rows, feedback, seed, basis=None):
    args, recipient_slots = inputs(rows, "recipient", feedback)
    donor_args, donor_slots = inputs(rows, "donor", feedback)
    baseline, low, _ = run_intervention(model, site, args)
    donor_result, high, _ = run_intervention(model, site, donor_args)
    unhooked = model(*args)
    assert all(torch.equal(a, b) for a, b in zip(unhooked, baseline, strict=True))
    candidates = {"full_patch": high, "no_op": low}
    projected = high
    if basis is not None:
        projected, complement = split_donor_delta(low, high, basis)
        candidates.update(projected_patch=projected, complement=complement)
    delta = (projected-low).norm(dim=-1, keepdim=True)
    rng = torch.Generator().manual_seed(2210+seed)
    direction = torch.randn(low.shape, generator=rng)
    candidates["random"] = low+direction/direction.norm(dim=-1, keepdim=True)*delta
    decisions = {}
    for name, replacement in candidates.items():
        output, observed, delivered = run_intervention(model, site, args, replacement=replacement)
        assert torch.equal(observed, low) and torch.equal(delivered, replacement)
        assert torch.equal(output[0], baseline[0])
        decisions[name] = [recipient_slots[int(slot)] for slot in output[1].argmax(-1)]
    recipient = [recipient_slots[int(slot)] for slot in baseline[1].argmax(-1)]
    donor = [donor_slots[int(slot)] for slot in donor_result[1].argmax(-1)]
    output = []
    for i, row in enumerate(rows):
        expected = role(row["donor_side"], row["recipient_rule"], row["recipient_order"])
        baseline_expected = role(row["recipient_side"], row["recipient_rule"], row["recipient_order"])
        donor_expected = role(row["donor_side"], row["donor_rule"], row["donor_order"])
        output.append(dict(row, expected=expected, baseline_expected=baseline_expected, donor_expected=donor_expected,
                           baseline=recipient[i], donor=donor[i],
                           eligible=recipient[i] == baseline_expected and donor[i] == donor_expected,
                           predictions={name: items[i] for name, items in decisions.items()}))
    return output


def summarize(rows):
    eligible = [row for row in rows if row["eligible"]]
    discriminating = [row for row in eligible if row["expected"] != row["donor_expected"]]
    crossed_rule = [row for row in discriminating if row["recipient_rule"] != row["donor_rule"]
                    and row["recipient_order"] == row["donor_order"]]
    must_switch = [row for row in crossed_rule if row["recipient_side"] != row["donor_side"]]
    must_keep = [row for row in crossed_rule if row["recipient_side"] == row["donor_side"]]
    summaries = {}
    for name in rows[0]["predictions"]:
        correct = lambda row: row["predictions"][name] == row["expected"]
        summaries[name] = dict(assigned_correct=sum(correct(row) for row in rows),
                               eligible_correct=sum(correct(row) for row in eligible),
                               different_action_correct=sum(correct(row) for row in discriminating),
                               donor_copies=sum(row["predictions"][name] == row["donor_expected"]
                                                for row in discriminating),
                               crossed_rule_switch_correct=sum(map(correct, must_switch)),
                               crossed_rule_keep_correct=sum(map(correct, must_keep)))
    return dict(assigned=len(rows), eligible=len(eligible), different_action=len(discriminating),
                crossed_rule_switch=len(must_switch), crossed_rule_keep=len(must_keep), interventions=summaries)


def experiment():
    tool = BASE/"release/examples/pipeline/tool.wasm"
    source = {(limit, side): exact_tool(tool, limit-4+side, limit-1+side)
              for limit, side in product(THRESHOLDS, (0, 1))}
    feedback = {key: value for key, (value, _) in source.items()}
    rows = case_matrix()
    studies = []
    witnesses = []
    for seed in SEEDS:
        path = STUDY/"learned-checkpoints"/f"width-32-seed-{seed}-step-600.pt"
        router = Router(32)
        router.load_state_dict(torch.load(path, weights_only=True))
        joint = Controller(router).eval()
        factorized = Factorized(router).eval()
        count = train_comparison(factorized, seed)
        with torch.inference_mode():
            basis = fit(joint)[:, :2]
            observed = {"joint": evaluate(joint, SITE, rows, feedback, seed, basis=basis),
                        "factorized": evaluate(factorized, FACTOR_SITE, rows, feedback, seed)}
        studies.append(dict(seed=seed, router_sha256=digest(path.read_bytes()), training_examples=count,
                            factor_weights={name: value.detach().tolist()
                                            for name, value in factorized.comparison.state_dict().items()},
                            fitted_basis=basis.tolist(), **{
                                name: dict(summary=summarize(results), rows=results)
                                for name, results in observed.items()}))
        if not witnesses:
            for side in (0, 1):
                witnesses.append(next((a, b) for a, b in zip(observed["joint"], observed["factorized"], strict=True)
                    if a["eligible"] and b["eligible"] and a["recipient_rule"] != a["donor_rule"]
                    and a["recipient_order"] == a["donor_order"] and a["recipient_side"] != a["donor_side"]
                    and a["donor_side"] == side and a["predictions"]["full_patch"] != a["expected"]
                    and b["predictions"]["full_patch"] == b["expected"]))
    reports = {}
    evidence = []
    for index, (joint, factor) in enumerate(witnesses):
        value, first = source[factor["limit"], factor["recipient_side"]]
        reports[f"case-{index}-first"] = first
        traces = {}
        for name, row in (("joint", joint), ("factorized", factor)):
            slot = row["predictions"]["full_patch"]
            if slot == 1:
                path, expected = BASE/"release/examples/pipeline/changed-tool.wasm", value+5
            elif slot == 2:
                path, expected = BASE/"release/examples/routing-subspaces/trim.wasm", value-2
            else:
                raise AssertionError("Expected a follow-up tool")
            result, reports[f"case-{index}-{name}"] = exact_tool(path, value, expected)
            traces[name] = [observation(str(index), boundary, item, runtime=runtime)
                            for boundary, item, runtime in (
                                ("first_tool_result", value, "wasm"), ("input_feedback", value, "host"),
                                ("second_request", slot, "pytorch"), ("second_tool_result", result, "wasm"))]
        evidence.append(dict(joint=joint, factorized=factor, traces=traces,
                             comparison=compare_observations(traces["factorized"], traces["joint"])))
    return dict(profile="visuaml-crossed-routing-v1", studies=studies, witnesses=evidence), reports


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with patch(PACKED, side_effect=AssertionError("Packed fallback entered")):
        result, reports = experiment()
    for name, artifact in [("replay", result), *reports.items()]:
        path = OUTPUT/(name+".json")
        if args.record:
            path.write_bytes(canonical(artifact)+b"\n")
        else:
            assert canonical(json.loads(path.read_bytes())) == canonical(artifact), path
    print(json.dumps([{name: study[name]["summary"] for name in ("joint", "factorized")}
                      for study in result["studies"]], indent=2))


if __name__ == "__main__":
    main()
