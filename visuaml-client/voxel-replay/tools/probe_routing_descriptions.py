"""Probe the first learned handle without argument binding or a tool call."""
import argparse
from copy import deepcopy
import json

import torch

from frozen_call_checkpoint import digest
from paired_routing import OUT, description_ids


def probe():
    artifact = OUT / "inference.pt"
    model = torch.jit.load(str(artifact))
    cases = json.loads((OUT / "holdout.json").read_text())["cases"]
    worksheet = {}
    for name in ("original", "new_description_compositions", "new_description_switch"):
        rows = cases[name]["registry"]
        worksheet[name] = (rows, next(row["handle"] for row in rows if row["family"] == "offset"))
    rows, expected = worksheet["original"]
    paraphrase = deepcopy(rows)
    for row in paraphrase:
        row["description"] = {"offset": "shift the input", "boost": "raise the value each repeat",
                              "trim": "lower the value each repeat"}[row["family"]]
    worksheet["meaning_preserving_paraphrase"] = (paraphrase, expected)
    exchanged = deepcopy(paraphrase)
    shift = next(row for row in exchanged if row["family"] == "offset")
    boost = next(row for row in exchanged if row["family"] == "boost")
    shift["description"], boost["description"] = boost["description"], shift["description"]
    worksheet["meaning_changing_exchange"] = (exchanged, boost["handle"])

    results = {}
    for name, (registry, expected) in worksheet.items():
        handles = torch.tensor([row["handle"] for row in registry])
        # Reject every seed on every row. First scores precede validation. The
        # first call cannot dispatch and therefore creates no execution or feedback.
        calls, traces, scores = model(
            torch.tensor([description_ids(row["description"]) for row in registry]), handles,
            torch.tensor([[1, 0, 1, 1]] * len(registry)), torch.zeros((len(registry), 1, 3), dtype=torch.long),
            torch.zeros(len(registry), dtype=torch.long), torch.zeros((2, 8), dtype=torch.long), 0, 0, 0, 0)
        selected = int(calls[0, 0])
        assert int(calls[0, 3]) == 1 and int(calls[0, 6]) == 0
        assert int(calls[1, 0]) == -1 and not traces.any()
        results[name] = dict(descriptions={str(row["handle"]): row["description"] for row in registry},
                             expected=expected, selected=selected, correct=selected == expected,
                             logits={str(handle): round(score, 6)
                                     for handle, score in zip(handles.tolist(), scores[0].tolist())})
    return dict(graph_sha256=digest(artifact.read_bytes()), cases=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with torch.inference_mode():
        result = probe()
    archive = OUT / "description-only.json"
    if args.write:
        archive.write_text(json.dumps(result, indent=2) + "\n")
    else:
        assert result == json.loads(archive.read_text())
    for name, case in result["cases"].items():
        print(name, case["expected"], case["selected"], case["correct"])
