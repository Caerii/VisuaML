"""Fit routing directions on calibration data and replay the held-out effects."""
from argparse import ArgumentParser
import json
from unittest.mock import patch

import torch

from paired_routing import Router, description_ids
from replay_activation_intervention import BASE, PACKED, SEEDS, SITE, STUDY, Controller
from study_router_learning import REGISTRY, THRESHOLDS, evaluation_rows
from visuaml.replay import Replay, capture, seal_report
from visuaml.replay._codec import canonical, digest
from visuaml.replay.interventions import fit_delta_basis, run_intervention, split_donor_delta
from visuaml.replay.pipeline import compare_observations, observation

OUTPUT = BASE / "release/examples/routing-subspaces"
CALIBRATION = (9, 17, 23)
RANKS = (1, 2, 4)
WITNESS_RANK = "2"


def fit(model):
    limits = torch.tensor(CALIBRATION).repeat_interleave(2)
    relation = torch.tensor([0, 1]).repeat(len(CALIBRATION))
    inputs = torch.stack(((limits-1)/31-.5, limits/31-.5, relation.float()-.5), dim=-1)
    donor = inputs.clone()
    donor[:, 0] = limits/31-.5
    condition = model.router.condition[:2]
    return fit_delta_basis(condition(donor)-condition(inputs), max(RANKS))


def exact_tool(path, value, expected):
    replay = Replay(path.read_bytes())
    bundle = capture(replay, value, None, 100, 1)
    report = seal_report(bundle, {"kind": "expect-return", "value": expected})
    actual = replay.finish(bundle["checkpoint"])["returned"]
    assert actual == expected
    return actual, report


def measure(model, basis, descriptions, feedback, seed):
    _, registry, choices, limits, relations, labels = evaluation_rows(descriptions)
    rotation = torch.tensor([1, 2, 0])
    registry = registry[:, rotation]
    low_label, high_label = [torch.argsort(rotation)[label] for label in labels]
    values = torch.tensor([feedback[int(limit)] for limit in limits])
    args = (registry, choices, values, limits, relations)
    base, recipient, _ = run_intervention(model, SITE, args)
    high, donor, _ = run_intervention(model, SITE, (registry, choices, limits, limits, relations))
    eligible = (base[1].argmax(-1) == low_label) & (high[1].argmax(-1) == high_label)
    random = torch.randn(recipient.shape, generator=torch.Generator().manual_seed(12000+seed))
    random /= random.norm(dim=-1, keepdim=True)
    rows = [dict(threshold=int(limits[i]), relation=int(relations[i]), feedback=int(values[i]),
                 registry=registry[i].tolist(), choices=choices[i].tolist(), eligible=bool(eligible[i]),
                 baseline_slot=int(base[1][i].argmax()), expected_slot=int(high_label[i]), ranks={})
            for i in range(len(values))]
    for rank in RANKS:
        projected, complement = split_donor_delta(recipient, donor, basis[:, :rank])
        scale = (projected-recipient).norm(dim=-1, keepdim=True)
        control = recipient+random*scale
        assert torch.allclose((control-recipient).norm(dim=-1, keepdim=True), scale, atol=1e-6)
        for row, fraction in zip(rows, (scale / (donor-recipient).norm(dim=-1, keepdim=True)).square(), strict=True):
            row["ranks"][str(rank)] = {"delta_energy_fraction": float(fraction[0])}
        options = dict(projected=projected, complement=complement, random=control)
        for name, replacement in options.items():
            output, original, applied = run_intervention(model, SITE, args, replacement=replacement)
            assert torch.equal(original, recipient) and torch.equal(applied, replacement)
            assert torch.equal(output[0], base[0])
            for row, logits in zip(rows, output[1], strict=True):
                row["ranks"].setdefault(str(rank), {})[name] = dict(slot=int(logits.argmax()), logits=logits.tolist())
    return rows


def experiment():
    assert not set(CALIBRATION) & set(THRESHOLDS)
    original = BASE / "release/examples/pipeline/tool.wasm"
    captured = {limit: exact_tool(original, limit-4, limit-1) for limit in THRESHOLDS}
    feedback = {limit: value for limit, (value, _) in captured.items()}
    groups = []
    for seed in SEEDS:
        path = STUDY / "learned-checkpoints" / f"width-32-seed-{seed}-step-600.pt"
        router = Router(32)
        router.load_state_dict(torch.load(path, weights_only=True))
        model = Controller(router).eval()
        basis = fit(model)
        groups.append(dict(seed=seed, weights_sha256=digest(path.read_bytes()), basis=basis.tolist(),
                           registries=[measure(model, basis, words, feedback, seed) for words in REGISTRY]))
    eligible = [row for group in groups for registry in group["registries"] for row in registry if row["eligible"]]
    counts = {str(rank): {name: sum(row["ranks"][str(rank)][name]["slot"] == row["expected_slot"]
                                   for row in eligible) for name in ("projected", "complement", "random")}
              for rank in RANKS}
    witness = None
    for group in groups:
        for registry_id, rows in enumerate(group["registries"]):
            for row in rows:
                if row["eligible"] and row["ranks"][WITNESS_RANK]["projected"]["slot"] == row["expected_slot"]:
                    witness = (group["seed"], registry_id, row)
                    break
            if witness:
                break
        if witness:
            break
    assert witness is not None
    seed, registry_id, row = witness
    source_value = row["feedback"]

    def follow(slot):
        role = [description_ids(description) for description in REGISTRY[registry_id]].index(row["registry"][slot])
        if role == 1:
            path, expected = BASE/"release/examples/pipeline/changed-tool.wasm", source_value+5
        elif role == 2:
            path, expected = OUTPUT/"trim.wasm", source_value-2
        else:
            raise AssertionError("Expected boost or trim")
        return exact_tool(path, source_value, expected)

    before, left = follow(row["baseline_slot"])
    after, right = follow(row["ranks"][WITNESS_RANK]["projected"]["slot"])

    def journal(slot, value):
        return [observation("witness", boundary, value, runtime=runtime)
                for boundary, value, runtime in (
                    ("first_tool_result", source_value, "wasm"), ("model_feedback", source_value, "host"),
                    ("second_request", slot, "pytorch"), ("second_tool_result", value, "wasm"))]
    first = compare_observations(journal(row["baseline_slot"], before),
                                 journal(row["ranks"][WITNESS_RANK]["projected"]["slot"], after))
    assert first["first_difference"]["reference"]["boundary"] == "second_request"
    result = dict(profile="visuaml-routing-subspace-v1", calibration=CALIBRATION, thresholds=THRESHOLDS,
                  eligible=len(eligible), assigned=sum(len(rows) for g in groups for rows in g["registries"]),
                  ranks=counts, groups=groups, witness=dict(seed=seed, registry_id=registry_id, row=row, rank=int(WITNESS_RANK),
                      baseline_result=before, patched_result=after, comparison=first))
    return result, dict(first=captured[row["threshold"]][1], baseline=left, patched=right)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with patch(PACKED, side_effect=AssertionError("Packed fallback entered")), torch.inference_mode():
        result, reports = experiment()
    for name, artifact in [("replay", result), *[(name+"-tool-report", value) for name, value in reports.items()]]:
        path = OUTPUT / (name+".json")
        if args.record:
            path.write_bytes(canonical(artifact)+b"\n")
        else:
            assert canonical(json.loads(path.read_bytes())) == canonical(artifact)
    print(json.dumps(dict(eligible=result["eligible"], ranks=result["ranks"], witness=result["witness"]["comparison"])))


if __name__ == "__main__":
    main()
