"""Replay a learned routing intervention while holding exact WASM feedback fixed."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter_ns
from unittest.mock import patch

import torch
from torch import nn

from paired_routing import Router
from study_router_learning import OUT as STUDY, REGISTRY, evaluation_rows
from visuaml.replay import Replay, capture, seal_report
from visuaml.replay._codec import canonical
from visuaml.replay.interventions import ActivationSite, run_intervention
from visuaml.replay.pipeline import compare_observations, observation, tensor_identity

BASE = Path(__file__).resolve().parents[1]
OUTPUT = BASE / "release/examples/interventions"
SITE = ActivationSite("router.condition.1")
SEEDS = (82, 514)
PACKED = "visuaml.compiler_lab.neural_microstep_executor_v58._NativeMicrostep"


class Controller(nn.Module):
    def __init__(self, router):
        super().__init__()
        self.router = router

    def forward(self, registry, choices, feedback, limits, relation):
        first = self.router.first_scores(registry)
        second = self.router.second_scores(registry, choices, feedback, limits, relation)
        return first, second


def pipeline_records(case, result):
    return [observation("case", boundary, value, runtime=runtime)
            for boundary, value, runtime in (
                ("first_request", result["first"], "pytorch"),
                ("tool_result", case["tool_result"], "visuaml-wasm"),
                ("model_feedback", result["feedback"], "host"),
                ("second_request", result["second"], "pytorch"),
            )]


def run_case(model, registry, tool_results, seed):
    _, ids, choices, limits, relation, labels = evaluation_rows(registry)
    # A registry rotation absent from the original paired evaluation changes
    # physical slots while preserving the descriptions and task semantics.
    rotation = torch.tensor([1, 2, 0])
    ids = ids[:, rotation]
    labels = [torch.argsort(rotation)[label] for label in labels]
    low = limits-1
    values = torch.tensor([tool_results[int(value)] for value in low])
    assert torch.equal(values, low)
    inputs = (ids, choices, values, limits, relation)
    baseline, hidden, _ = run_intervention(model, SITE, inputs)
    donor_feedback = torch.tensor([tool_results[int(value)] for value in limits])
    donor_inputs = (ids, choices, donor_feedback, limits, relation)
    donor, donor_hidden, _ = run_intervention(model, SITE, donor_inputs)
    _, same, _ = run_intervention(model, SITE, (ids, choices, limits-2, limits, relation))
    raw = model(*inputs)
    assert all(torch.equal(a, b) for a, b in zip(raw, baseline, strict=True))
    rng = torch.Generator().manual_seed(seed+655)
    random = torch.randn(hidden.shape, generator=rng)
    distance = (donor_hidden-hidden).norm(dim=1, keepdim=True)
    random = hidden+distance*random/random.norm(dim=1, keepdim=True)
    assert torch.allclose((random-hidden).norm(dim=1, keepdim=True), distance, atol=1e-6, rtol=1e-5)
    scenarios = {"donor_patch": donor_hidden, "same_side": same, "random": random,
                 "no_op": hidden, "zero_ablation": torch.zeros_like(hidden)}
    expected = {"donor_patch": labels[1], "same_side": labels[0], "random": labels[0],
                "no_op": labels[0], "feedback_swap": labels[1]}
    collected = {}
    for name, tensor in scenarios.items():
        result, original, applied = run_intervention(model, SITE, inputs, replacement=tensor)
        assert torch.equal(original, hidden) and torch.equal(applied, tensor)
        collected[name] = (result, applied, values)
    collected["feedback_swap"] = (donor, donor_hidden, limits)
    assert torch.equal(collected["donor_patch"][0][1], donor[1])
    for i in (0, len(values)-1):
        singleton = tuple(argument[i:i+1] for argument in inputs)
        output, _, _ = run_intervention(model, SITE, singleton, replacement=donor_hidden[i:i+1])
        assert torch.allclose(output[1], collected["donor_patch"][0][1][i:i+1], atol=1e-5, rtol=1e-5)
        assert torch.equal(output[1].argmax(-1), donor[1][i:i+1].argmax(-1))
    assert all(torch.equal(a, b) for a, b in zip(collected["no_op"][0], baseline, strict=True))
    choice = baseline[1].argmax(-1)
    eligible = (choice == labels[0]) & (donor[1].argmax(-1) == labels[1])
    raw_rows = []
    for i in range(len(values)):
        row = dict(tool_result=int(values[i]), threshold=int(limits[i]), relation=int(relation[i]),
                   registry=ids[i].tolist(), choices=choices[i].tolist(), eligible=bool(eligible[i]),
                   baseline=dict(feedback=int(values[i]), first=int(baseline[0][i].argmax()),
                                 second=int(choice[i]), expected_second=int(labels[0][i])),
                   scenarios={})
        for name, (outputs, activation, feedback) in collected.items():
            observed = dict(feedback=int(feedback[i]), first=int(outputs[0][i].argmax()),
                            second=int(outputs[1][i].argmax()),
                            activation=tensor_identity(activation[i].numpy()),
                            logits=outputs[1][i].tolist())
            if name in expected:
                observed["expected_second"] = int(expected[name][i])
            row["scenarios"][name] = observed
        raw_rows.append(row)
    summaries = {}
    for name in collected:
        selected = [row for row in raw_rows if row["eligible"]]
        summaries[name] = dict(
            switches=sum(r["scenarios"][name]["second"] != r["baseline"]["second"] for r in selected),
            expected_correct=sum(r["scenarios"][name].get("expected_second") == r["scenarios"][name]["second"]
                                 for r in raw_rows) if name in expected else None,
            unrelated_first_unchanged=sum(r["scenarios"][name]["first"] == r["baseline"]["first"] for r in raw_rows),
        )
    return dict(total=len(values), eligible=int(eligible.sum()), scenarios=summaries, rows=raw_rows)


def experiment():
    wasm = (BASE / "release/examples/pipeline/tool.wasm").read_bytes()
    runtime = Replay(wasm)
    tool_results, bundles = {}, {}
    for limit in (7, 13, 21, 27):
        for value in range(limit-2, limit+1):
            bundles[value] = capture(runtime, value-3, None, 100, 1)
            tool_results[value] = runtime.finish(bundles[value]["checkpoint"])["returned"]
            assert tool_results[value] == value
    cases, performance = [], []
    for seed in SEEDS:
        path = STUDY / "learned-checkpoints" / f"width-32-seed-{seed}-step-600.pt"
        router = Router(32)
        router.load_state_dict(torch.load(path, weights_only=True))
        model = Controller(router).eval()
        for registry in REGISTRY:
            started = perf_counter_ns()
            case = run_case(model, registry, tool_results, seed)
            performance.append(dict(seed=seed, ns=perf_counter_ns()-started, conditions=6*case["total"]))
            cases.append(dict(seed=seed, weights_sha256=sha256(path.read_bytes()).hexdigest(),
                              descriptions=registry, **case))
    eligible = next(row for row in cases[0]["rows"] if row["eligible"])
    baseline = pipeline_records(eligible, eligible["baseline"])
    witness = seal_report(bundles[eligible["tool_result"]],
                          {"kind": "expect-return", "value": eligible["tool_result"]})
    comparisons = {name: compare_observations(baseline, pipeline_records(eligible, result))
                   for name, result in eligible["scenarios"].items()}
    failure = next((dict(seed=case["seed"], row=row, condition=name) for case in cases
                    for row in case["rows"] for name, item in row["scenarios"].items()
                    if name in ("donor_patch", "feedback_swap") and item["second"] != item["expected_second"]), None)
    return dict(profile="visuaml-activation-intervention-v1", site={"module": SITE.module, "phase": SITE.phase},
                wasm_sha256=sha256(wasm).hexdigest(), cases=cases, selected_case=eligible, selected_comparisons=comparisons,
                first_failed_prediction=failure, registry_rotation=[1, 2, 0]), witness, performance


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with patch(PACKED, side_effect=AssertionError("Packed fallback entered")), torch.inference_mode():
        result, wasm, samples = experiment()
    for name, value in (("experiment.json", result), ("tool-report.json", wasm)):
        path = OUTPUT / name
        if args.record:
            path.write_bytes(canonical(value)+b"\n")
        else:
            assert canonical(json.loads(path.read_bytes())) == canonical(value)
    if args.record:
        (OUTPUT / "timings.json").write_bytes(canonical(samples)+b"\n")
    print(json.dumps([dict(seed=case["seed"], total=case["total"], eligible=case["eligible"],
                           scenarios=case["scenarios"]) for case in result["cases"]], indent=2))


if __name__ == "__main__":
    main()
