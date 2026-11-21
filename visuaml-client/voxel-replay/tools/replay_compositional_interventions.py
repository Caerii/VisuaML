"""Calibrate a token patch on a causal controller and compose exact tool calls."""
from argparse import ArgumentParser
from itertools import product
import json
from unittest.mock import patch

import torch
from torch import nn

from replay_activation_intervention import BASE, PACKED
from replay_routing_subspaces import exact_tool
from visuaml.replay._codec import canonical, digest
from visuaml.replay.interventions import ActivationSite, replace_activation_token, run_intervention
from visuaml.replay.pipeline import tensor_identity

OUT = BASE/"release/examples/compositional-interventions"
CALIBRATE = (9, 17)
HOLDOUT = (7, 13, 21, 27)
TOOLS = ("boost", "trim", "abstain")
RULES = ((1, 0), (0, 1), (0, 2))


class Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.number = nn.Linear(1, 24)
        self.rule = nn.Embedding(3, 24)
        self.position = nn.Parameter(torch.randn(4, 24)*.1)
        self.blocks = nn.ModuleList(nn.TransformerEncoderLayer(
            24, 3, 48, dropout=0, batch_first=True) for _ in range(2))
        self.head = nn.Linear(24, 3)

    def forward(self, feedback, threshold, rule):
        numeric = torch.stack((feedback, threshold), dim=1).float().unsqueeze(-1)/15.5-1
        rule_token = self.rule(rule).unsqueeze(1)
        state = torch.cat((self.number(numeric), rule_token, torch.zeros_like(rule_token)), dim=1)+self.position
        mask = torch.ones((4, 4), dtype=torch.bool, device=state.device).triu(1)
        for block in self.blocks:
            state = block(state, src_mask=mask)
        return self.head(state[:, -1])


def train():
    torch.manual_seed(322)
    model = Controller()
    generator = torch.Generator().manual_seed(733)
    cases = torch.tensor([(value, limit, rule)
                          for limit, rule in product(range(3, 30), range(3))
                          if limit not in CALIBRATE+HOLDOUT
                          for value in range(max(0, limit-7), min(32, limit+8))])
    labels = torch.tensor(RULES)[cases[:, 2], (cases[:, 0] >= cases[:, 1]).long()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    for _ in range(1250):
        indices = torch.randint(len(cases), (128,), generator=generator)
        arguments = cases[indices].unbind(dim=1)
        loss = nn.functional.cross_entropy(model(*arguments), labels[indices])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.eval()


def run_rows(model, rows, module, token):
    site = ActivationSite(module)
    base_inputs = torch.tensor([[r[0], r[2], r[3]] for r in rows]).unbind(dim=1)
    donor_inputs = torch.tensor([[r[1], r[2], r[4]] for r in rows]).unbind(dim=1)
    baseline, original, _ = run_intervention(model, site, base_inputs)
    donor, donor_tensor, _ = run_intervention(model, site, donor_inputs)
    assert torch.allclose(model(*base_inputs), baseline, atol=1e-5)
    no_op, _, _ = run_intervention(model, site, base_inputs, replacement=original)
    assert torch.equal(no_op, baseline)
    replacement = replace_activation_token(original, donor_tensor, token)
    edited, _, applied = run_intervention(model, site, base_inputs, replacement=replacement)
    assert torch.equal(applied, replacement)
    difference = replacement[:, token]-original[:, token]
    noise = torch.randn(difference.shape, generator=torch.Generator().manual_seed(2026))
    control = original.clone()
    control[:, token] += noise/noise.norm(dim=1, keepdim=True)*difference.norm(dim=1, keepdim=True)
    random, _, _ = run_intervention(model, site, base_inputs, replacement=control)
    full, _, _ = run_intervention(model, site, base_inputs, replacement=donor_tensor)
    return [dict(feedback=feedback, donor=source, threshold=threshold, rule=rule, donor_rule=other,
                 expected=RULES[rule][int(source >= threshold)],
                 baseline_expected=RULES[rule][int(feedback >= threshold)],
                 donor_expected=RULES[other][int(source >= threshold)],
                 baseline=int(baseline[i].argmax()), donor_choice=int(donor[i].argmax()),
                 edit=int(edited[i].argmax()), random=int(random[i].argmax()), whole=int(full[i].argmax()),
                 payload=tensor_identity(donor_tensor[i, token].numpy()))
            for i, (feedback, source, threshold, rule, other) in enumerate(rows)]


def cases(thresholds):
    return [(limit-1 if side == 0 else limit, limit-1 if donor_side == 0 else limit,
             limit, rule, donor_rule)
            for limit, side, donor_side, rule, donor_rule in product(thresholds, (0, 1), (0, 1), range(3), range(3))
            if rule != donor_rule]


def choose(model):
    rows = cases(CALIBRATE)
    scores = []
    for module, token in product(("blocks.0", "blocks.1"), range(4)):
        observed = run_rows(model, rows, module, token)
        scores.append(dict(module=module, token=token, correct=sum(row["edit"] == row["expected"] for row in observed)))
    return max(scores, key=lambda score: score["correct"]), scores


def execute(choice, value):
    if choice == 2:
        return None, None
    if choice == 0:
        guest, expected = BASE/"release/examples/pipeline/changed-tool.wasm", value+5
    else:
        guest, expected = BASE/"release/examples/routing-subspaces/trim.wasm", value-2
    return exact_tool(guest, value, expected)


def compose(model, rows):
    reference = BASE/"release/examples/pipeline/tool.wasm"
    # Freeze one donor token across recipients that require Boost and abstention.
    eligible = [row for row in rows if row["feedback"] == 6 and row["donor"] == 7 and row["donor_rule"] == 1]
    boost = next(row for row in eligible if row["rule"] == 0 and row["edit"] == 0)
    abstain = next(row for row in eligible if row["rule"] == 2 and row["edit"] == 2)
    assert boost["payload"] == abstain["payload"]
    value, first = exact_tool(reference, 3, 6)
    _, donor_report = exact_tool(reference, 4, 7)
    reports = {"first": first, "donor": donor_report}
    branches = []
    for row in (boost, abstain):
        result, report = execute(row["edit"], value)
        record = dict(first_request=TOOLS[row["edit"]], first_result=result, row=row, next_request=None)
        if report is not None:
            reports["boost"] = report
            second_rule, threshold = 1, 13
            with torch.inference_mode():
                prediction = int(model(*torch.tensor([[result, threshold, second_rule]]).unbind(dim=1)).argmax())
            expected = RULES[second_rule][int(result >= threshold)]
            assert prediction == expected
            observed, second = execute(prediction, result)
            if second is not None:
                reports["second"] = second
            record.update(next_request=TOOLS[prediction], next_result=observed, next_expected=TOOLS[expected],
                          next_threshold=threshold, next_rule=second_rule)
        branches.append(record)
    return branches, reports


def experiment(model, spec):
    selected, calibration = choose(model)
    assert selected == spec["selection"]
    rows = run_rows(model, cases(HOLDOUT), selected["module"], selected["token"])
    success = sum(row["edit"] == row["expected"] for row in rows)
    summary = dict(assigned=len(rows), edit=success,
                   **{key: sum(row[key] == row["expected"] for row in rows)
                      for key in ("whole", "random", "baseline")})
    summary["baseline_valid"] = sum(row["baseline"] == row["baseline_expected"] for row in rows)
    summary["donor_valid"] = sum(row["donor_choice"] == row["donor_expected"] for row in rows)
    counterexample = next((row for row in rows if row["edit"] != row["expected"]), None)
    if counterexample is None:
        counterexample = dict(control="whole donor", row=next(row for row in rows if row["whole"] != row["expected"]))
    else:
        counterexample = dict(control="selected token edit", row=counterexample)
    return dict(profile="visuaml-token-intervention-v1", calibration=calibration, summary=summary,
                rows=rows, first_failed_prediction=counterexample)


def main():
    parser = ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--record", action="store_true")
    mode.add_argument("--train", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    weights = OUT/"controller.pt"
    if args.train:
        model = train()
        torch.save(model.state_dict(), weights)
    else:
        model = Controller().eval()
        model.load_state_dict(torch.load(weights, weights_only=True))
    spec_path = OUT/"specification.json"
    with torch.inference_mode():
        selected, _ = choose(model)
    expected_spec = dict(profile="visuaml-intervention-spec-v1", model_sha256=digest(weights.read_bytes()),
                         calibration_thresholds=list(CALIBRATE), holdout_thresholds=list(HOLDOUT),
                         selection=selected, phase="forward", edit="replace one donor token at matched threshold",
                         token_layout=["feedback", "threshold", "rule", "query"],
                         observable="semantic next tool")
    if args.train:
        spec_path.write_bytes(canonical(expected_spec)+b"\n")
    spec = json.loads(spec_path.read_bytes())
    assert spec == expected_spec
    with patch(PACKED, side_effect=AssertionError("Packed fallback entered")), torch.inference_mode():
        results = experiment(model, spec)
        branches, tools = compose(model, results["rows"])
        results["composition"] = branches
        failure = results["first_failed_prediction"]
        row = failure["row"]
        value, first = exact_tool(BASE/"release/examples/pipeline/tool.wasm", row["feedback"]-3, row["feedback"])
        tools["failure-first"] = first
        observed_choice = row["edit"] if failure["control"] == "selected token edit" else row["whole"]
        failure.update(boundary="first_request", verified_followups={})
        for name, choice in (("expected", row["expected"]), ("observed", observed_choice)):
            returned, report = execute(choice, value)
            failure["verified_followups"][name] = dict(action=TOOLS[choice], value=returned)
            if report is not None:
                tools[f"failure-{name}"] = report
    for name, artifact in (("replay", results), *[(key+"-tool-report", data) for key, data in tools.items()]):
        path = OUT/(name+".json")
        if args.train or args.record:
            path.write_bytes(canonical(artifact)+b"\n")
        else:
            assert canonical(json.loads(path.read_bytes())) == canonical(artifact), path
    print(json.dumps(dict(selection=selected, **results["summary"], composition=branches), indent=2))


if __name__ == "__main__":
    main()
