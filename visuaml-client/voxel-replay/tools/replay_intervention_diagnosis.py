"""Localize failed token transfers and test the selected rescue on fresh seeds."""
from argparse import ArgumentParser
from itertools import product
import json
from statistics import median
from unittest.mock import patch

import torch
from torch import nn

from replay_activation_intervention import BASE, PACKED
from replay_compositional_interventions import Controller, OUT as PRIOR, RULES, TOOLS, execute
from visuaml.replay._codec import canonical, digest
from visuaml.replay.interventions import ActivationSite, run_intervention

OUT = BASE/"release/examples/intervention-diagnosis"
CALIBRATION = (9, 17)
HOLDOUT = (8, 19, 25)
SEEDS = (617, 829)
SITE = ActivationSite("blocks.0")
PATCHES = {"threshold": (1,), "feedback": (0,), "pair": (0, 1)}


def train(seed):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed+411)
    model = Controller()
    rows = torch.tensor([(value, limit, rule) for limit, rule in product(range(3, 30), range(3))
                         if limit not in CALIBRATION+HOLDOUT
                         for value in range(max(0, limit-7), min(32, limit+8))])
    target = torch.tensor(RULES)[rows[:, 2], (rows[:, 0] >= rows[:, 1]).long()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    for _ in range(1250):
        idx = torch.randint(len(rows), (128,), generator=generator)
        loss = nn.functional.cross_entropy(model(*rows[idx].unbind(1)), target[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.eval()


def design(thresholds):
    return [(limit-1+side, limit+distance if donor else limit-1-distance, limit, rule, other)
            for limit, side, donor, distance, rule, other in product(
                thresholds, (0, 1), (0, 1), (0, 2), range(3), range(3)) if rule != other]


def investigate(model, rows):
    source = torch.tensor(rows)
    args = source[:, [0, 2, 3]].unbind(1)
    donor_args = source[:, [1, 2, 4]].unbind(1)
    baseline, original, _ = run_intervention(model, SITE, args)
    donor, alternate, _ = run_intervention(model, SITE, donor_args)
    direct = model(*source[:, [1, 2, 3]].unbind(1))
    edited = {}
    for name, indices in PATCHES.items():
        replacement = original.clone()
        replacement[:, list(indices)] = alternate[:, list(indices)]
        output, _, observed = run_intervention(model, SITE, args, replacement=replacement)
        assert torch.equal(observed, replacement)
        assert torch.equal(observed[:, 2:], original[:, 2:])
        edited[name] = output
    observations = []
    for i, (feedback, donated, limit, rule, other) in enumerate(rows):
        original_target = RULES[rule][int(feedback>=limit)]
        target = RULES[rule][int(donated>=limit)]
        donor_target = RULES[other][int(donated>=limit)]
        scores = {key: values[i] for key, values in dict(baseline=baseline, direct=direct, **edited).items()}

        def record(logits):
            rivals = torch.cat((logits[:target], logits[target+1:]))
            return dict(action=int(logits.argmax()), target_margin=float(logits[target]-rivals.max()))

        observations.append(dict(feedback=feedback, donor=donated, limit=limit, rule=rule, donor_rule=other,
                                 target=target, baseline_target=original_target, donor_target=donor_target,
                                 donor_action=int(donor[i].argmax()), scores={key: record(x) for key, x in scores.items()}))
    return observations


def score(rows):
    groups = {"switch": [row for row in rows if row["target"] != row["baseline_target"]],
              "preserve": [row for row in rows if row["target"] == row["baseline_target"]]}
    return {key: dict(count=len(group), **{name: sum(row["scores"][name]["action"] == row["target"]
                             for row in group) for name in ("baseline", "direct", *PATCHES)})
            for key, group in groups.items()}


def composition(model, rows, selected, tool_cache):
    completed = []
    for i, row in enumerate(rows):
        next_limit = HOLDOUT[i % len(HOLDOUT)]
        next_rule = (row["rule"]+1) % 3
        expected = row["target"]
        actual = row["scores"][selected]["action"]

        def tool(choice, value):
            key = (choice, value)
            if key not in tool_cache:
                tool_cache[key] = execute(choice, value)
            return tool_cache[key][0]

        wanted_return, observed_return = tool(expected, row["feedback"]), tool(actual, row["feedback"])
        wanted_next = None if wanted_return is None else RULES[next_rule][int(wanted_return>=next_limit)]
        wanted_final = None if wanted_next is None else tool(wanted_next, wanted_return)
        if observed_return is None:
            observed_next, observed_final = None, None
        else:
            args = torch.tensor([[observed_return, next_limit, next_rule]]).unbind(1)
            observed_next = int(model(*args).argmax())
            observed_final = tool(observed_next, observed_return)
        complete = (actual, observed_next, observed_final) == (expected, wanted_next, wanted_final)
        completed.append(dict(index=i, expected=expected, actual=actual, expected_next=wanted_next,
                              actual_next=observed_next, first_return=observed_return,
                              next_rule=next_rule, next_limit=next_limit, final_return=observed_final, expected_final_return=wanted_final,
                              first_failure=None if complete else "first_request" if actual != expected else "second_request"))
    return completed


def diagnostic_summary(rows):
    switches = [row for row in rows if row["baseline_target"] != row["target"]]
    failed = [row for row in switches if row["scores"]["threshold"]["action"] != row["target"]]
    near, far = {}, {}
    for row in switches:
        side = row["donor"] >= row["limit"]
        destination = near if row["donor"] in (row["limit"]-1, row["limit"]) else far
        destination[row["feedback"], row["limit"], row["rule"], row["donor_rule"], side] = row
    disagree = sum(near[key]["scores"]["threshold"]["action"] != far[key]["scores"]["threshold"]["action"]
                   for key in near)
    return dict(threshold_failures=len(failed),
                paired_rescues=sum(row["scores"]["pair"]["action"] == row["target"] for row in failed),
                failed_margin_improvements=sum(row["scores"]["threshold"]["target_margin"] >
                                               row["scores"]["baseline"]["target_margin"] for row in failed),
                failed_threshold_margin_median=median(row["scores"]["threshold"]["target_margin"]
                                                     for row in failed) if failed else None,
                donor_distance_changes=disagree, donor_distance_pairs=len(near))


def block_query_feedback(model, rows):
    """Remove only the direct query-to-feedback edge in the final block."""
    def hide_edge(_module, args, kwargs):
        mask = kwargs["src_mask"].clone()
        mask[3, 0] = True
        return args, dict(kwargs, src_mask=mask)

    with model.blocks[1].register_forward_pre_hook(hide_edge, with_kwargs=True):
        return investigate(model, rows)


def experiment(models):
    old = Controller().eval()
    old.load_state_dict(torch.load(PRIOR/"controller.pt", weights_only=True))
    prior = json.loads((PRIOR/"replay.json").read_text())
    inputs = [(r["feedback"], r["donor"], r["threshold"], r["rule"], r["donor_rule"])
              for r in prior["rows"]]
    original_rows = investigate(old, inputs)
    edge_blocked = block_query_feedback(old, inputs)
    original = next(row for row in original_rows
                    if (row["feedback"], row["donor"], row["limit"], row["rule"], row["donor_rule"]) == (7, 6, 7, 0, 1))
    original_failures = [row for row in original_rows if row["target"] != row["scores"]["threshold"]["action"]]
    assert len(original_failures) == 14
    edge_rescues = sum(prior_row["target"] == edited_row["scores"]["threshold"]["action"]
                       for prior_row, edited_row in zip(original_rows, edge_blocked, strict=True)
                       if prior_row["target"] != prior_row["scores"]["threshold"]["action"])
    edge_baseline = sum(row["baseline_target"] == row["scores"]["baseline"]["action"] for row in edge_blocked)
    original["query_feedback_blocked"] = next(
        row["scores"] for row in edge_blocked if (row["feedback"], row["donor"], row["limit"],
                                               row["rule"], row["donor_rule"]) == (7, 6, 7, 0, 1))
    studies, tools = [], {}
    for seed, model, artifact in models:
        calibration = score(investigate(model, design(CALIBRATION)))
        selected = max(PATCHES, key=lambda name: (calibration["switch"][name]+calibration["preserve"][name],
                                                  -len(PATCHES[name])))
        rows = investigate(model, design(HOLDOUT))
        chain = composition(model, rows, selected, tools)
        studies.append(dict(seed=seed, weights_sha256=digest(artifact.read_bytes()), calibration=calibration,
                            selected=selected, evaluation=score(rows), diagnosis=diagnostic_summary(rows), rows=rows,
                            full_chain_correct=sum(row["first_failure"] is None for row in chain),
                            chain=chain))
    return dict(profile="visuaml-transfer-diagnosis-v1", calibration=CALIBRATION, holdout=HOLDOUT,
                original_failure=original,
                original_paired_rescues=sum(row["scores"]["pair"]["action"] == row["target"]
                                            for row in original_failures),
                original_edge_rescues=edge_rescues, original_edge_baseline=edge_baseline, studies=studies,
                distinct_tools_checked=sum(report is not None for _, report in tools.values())), tools


def main():
    parser = ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--record", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    models = []
    for seed in SEEDS:
        artifact = OUT/f"controller-{seed}.pt"
        if args.train:
            model = train(seed)
            torch.save(model.state_dict(), artifact)
        else:
            model = Controller().eval()
            model.load_state_dict(torch.load(artifact, weights_only=True))
        models.append((seed, model, artifact))
    with patch(PACKED, side_effect=AssertionError("Packed fallback entered")), torch.inference_mode():
        report, tools = experiment(models)
    entries = {"replay": report}
    # Keep the original calibrated failure's exact two tool consequences.
    row = report["original_failure"]
    for name, action in (("expected", row["target"]), ("observed", row["scores"]["threshold"]["action"])):
        with patch(PACKED, side_effect=AssertionError("Packed fallback entered")):
            _, witness = execute(action, row["feedback"])
        if witness is not None:
            entries[name+"-tool-report"] = witness
    for name, value in entries.items():
        path = OUT/(name+".json")
        if args.train or args.record:
            path.write_bytes(canonical(value)+b"\n")
        else:
            assert canonical(json.loads(path.read_bytes())) == canonical(value), path
    for study in report["studies"]:
        print(study["seed"], study["selected"], study["evaluation"], study["full_chain_correct"], len(study["chain"]))
    print(report["original_failure"], report["original_paired_rescues"], report["original_edge_rescues"], report["original_edge_baseline"],
          report["distinct_tools_checked"])


if __name__ == "__main__":
    main()
