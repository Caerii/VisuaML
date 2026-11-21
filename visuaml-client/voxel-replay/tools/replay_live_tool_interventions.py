"""Replay learned tool choices from actual WASM returns and controlled edits."""
from argparse import ArgumentParser
from collections import Counter
from itertools import product
from unittest.mock import patch

import torch
import wasmtime

from replay_activation_intervention import BASE, PACKED
from replay_compositional_interventions import Controller, OUT as PRIOR, RULES, TOOLS, execute
from replay_intervention_diagnosis import OUT as DIAGNOSIS, SEEDS, investigate
from replay_routing_subspaces import exact_tool
from visuaml.replay._codec import canonical, digest

OUT = BASE / "release/examples/live-tool-interventions"
THRESHOLDS = (8, 19, 25)
MODES = ("baseline", "direct", "threshold", "pair")
REGRESSION = (7, 6, 7, 0, 1)


def request(action, value):
    return None if action == 2 else {"tool": TOOLS[action], "argument": value}


def expected_output(action, value):
    return (value + 5, value - 2, None)[action]


def study(binary):
    first_calls, second_calls = {}, {}

    def first(value):
        argument = value - 1  # Select a known return using a public tool argument.
        if argument not in first_calls:
            first_calls[argument] = exact_tool(binary, argument, value)
        actual, _ = first_calls[argument]
        assert actual == value
        return {"argument": argument, "returned": actual}

    def second(action, value):
        key = (action, value)
        if key not in second_calls:
            second_calls[key] = execute(action, value)
        return second_calls[key][0]

    design = [(limit + offset, limit + donor, limit, rule, other)
              for limit, offset, donor, rule, other in product(
                  THRESHOLDS, (-1, 0, 1), (-1, 0, 1), range(3), range(3))
              if rule != other]
    controllers = [("original", PRIOR / "controller.pt"),
                   *((str(seed), DIAGNOSIS / f"controller-{seed}.pt") for seed in SEEDS)]
    studies, witness_reports = [], {}
    for name, weights in controllers:
        model = Controller().eval()
        model.load_state_dict(torch.load(weights, weights_only=True))
        planned = design + ([REGRESSION] if name == "original" else [])
        forecasts = [(first(value), first(donor)) for value, donor, *_ in planned]
        inputs = [(live["returned"], alternate["returned"], *row[2:])
                  for (live, alternate), row in zip(forecasts, planned, strict=True)]
        observed = investigate(model, inputs, include_logits=True)
        counts, rows = Counter(), []
        for index, (row, (live, alternate)) in enumerate(zip(observed, forecasts, strict=True)):
            value = live["returned"]
            modes = {}
            for mode in MODES:
                effective = value if mode == "baseline" else alternate["returned"]
                expected = RULES[row["rule"]][int(effective >= row["limit"])]
                actual = row["scores"][mode]["action"]
                wanted_request, actual_request = request(expected, value), request(actual, value)
                wanted_return = expected_output(expected, value)
                actual_return = second(actual, value)
                failure = ("second_request" if actual_request != wanted_request else
                           "second_execution" if actual_return != wanted_return else None)
                if index < len(design):
                    counts[mode] += failure is None
                modes[mode] = dict(controller_input=effective if mode == "direct" else value,
                                   donor=alternate["returned"] if mode in ("pair", "threshold") else None,
                                   expected_request=wanted_request, request=actual_request,
                                   expected_return=wanted_return, returned=actual_return,
                                   first_failure=failure, **row["scores"][mode])
            rows.append(dict(first_call=live, alternate_first_call=alternate, limit=row["limit"],
                             rule=row["rule"], donor_rule=row["donor_rule"], modes=modes))
        studies.append(dict(controller=name, weights_sha256=digest(weights.read_bytes()),
                            holdout_count=len(design), correct=dict(counts), rows=rows))
        if name == "original":
            regression = rows[-1]
            assert regression["modes"]["threshold"]["first_failure"] == "second_request"
            assert regression["modes"]["pair"]["first_failure"] is None
            failures = [i for i, row in enumerate(rows[:-1])
                        if row["modes"]["threshold"]["first_failure"]]
            studies[-1]["first_holdout_failure_index"] = failures[0] if failures else None
            cases = {"regression": regression}
            if failures:
                cases["holdout"] = rows[failures[0]]
            for label, witness in cases.items():
                for field in ("first_call", "alternate_first_call"):
                    witness_reports[f"{label}-{field}"] = first_calls[witness[field]["argument"]][1]
                for mode in ("threshold", "pair"):
                    action = witness["modes"][mode]["action"]
                    if action != 2:
                        key = (action, witness["first_call"]["returned"])
                        witness_reports[f"{label}-{mode}"] = second_calls[key][1]
    return dict(profile="visuaml-live-tool-interventions-v1", first_sha256=digest(binary.read_bytes()),
                thresholds=THRESHOLDS, studies=studies, first_calls_checked=len(first_calls),
                second_calls_checked=sum(report is not None for _, report in second_calls.values())), witness_reports


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    binary = OUT / "first-tool.wasm"
    compiled = wasmtime.wat2wasm((OUT / "first-tool.wat").read_text())
    if args.record:
        binary.write_bytes(compiled)
    assert binary.read_bytes() == compiled
    with torch.inference_mode(), patch(PACKED, side_effect=AssertionError("Packed fallback entered")):
        report, witnesses = study(binary)
    for name, artifact in {"replay": report, **witnesses}.items():
        path = OUT / f"{name}.json"
        raw = canonical(artifact) + b"\n"
        if args.record:
            path.write_bytes(raw)
        else:
            assert path.read_bytes() == raw, path
    for result in report["studies"]:
        print(result["controller"], result["holdout_count"], result["correct"])
    print("Distinct verified calls:", report["first_calls_checked"], report["second_calls_checked"])


if __name__ == "__main__":
    main()
