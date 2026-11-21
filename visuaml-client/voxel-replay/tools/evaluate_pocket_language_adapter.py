"""Evaluate broad input admission separately on the published paraphrases."""
import argparse
import json
import re

import torch

from evaluate_world_inference_pairs import OUT as STATISTICS, build_pairs, run
from frozen_call_checkpoint import canonical, digest
from pocket_goal_planner import OUT, WORLD
from pocket_world_inference import VOCAB, words

PATTERNS = (
    r"Could you open the gate and see what happens after ([1-8]) updates\? Use (beta|gamma) if the target stays dark; otherwise use (beta|gamma)\.",
    r"Imagine opening this passage\. At step ([1-8]), check the destination\. Choose (gamma|beta) for a lit cell or (beta|gamma) for a dark cell\.",
    r"Try digging the barrier, run the world for ([1-8]) beats, and call (beta|gamma) if the signal hasn't reached the target or (beta|gamma) if it has\.",
)


def translate(prompt):
    for i, pattern in enumerate(PATTERNS):
        match = re.fullmatch(pattern, prompt)
        if match:
            deadline, first, second = match.groups()
            low, high = (second, first) if i == 1 else (first, second)
            return int(deadline), low, high
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    original = STATISTICS/"outcomes.json"
    baseline = json.loads(original.read_text())
    pairs = build_pairs(baseline["seed"],25)
    model = torch.jit.load(str(WORLD/"inference.pt"))
    torch.set_num_threads(1)
    records = []
    with torch.inference_mode():
        for pair,row in zip(pairs,baseline["pairs"],strict=True):
            assert pair["id"] == row["id"]
            for bit in (0,1):
                raw = row["runs"]["free"][str(bit)]["prompt"]
                parsed = translate(raw)
                normalized = words(1,*parsed,"please") if parsed else None
                admitted = normalized is not None and set(normalized.split()) <= set(VOCAB)
                result = run(model,pair,bit,"actual",normalized) if admitted else None
                records.append(dict(id=pair["id"],bit=bit,original=raw,
                                    direct_admission=set(raw.split())<=set(VOCAB),
                                    normalized=normalized,admitted=admitted,
                                    interpretation=parsed==tuple(pair[key] for key in ("horizon","low","high")),
                                    success=bool(result and result["success"]),result=result))
    negative = "Can the agent get through? Please find out."
    assert translate(negative) is None
    data = dict(baseline_sha256=digest(original.read_bytes()),
                model_sha256=digest((WORLD/"inference.pt").read_bytes()), records=records,
                unsupported_prompt=negative,
                summary={key:sum(r[key] for r in records)
                         for key in ("direct_admission","admitted","interpretation","success")})
    path = OUT/"language-adapter.json"
    if args.write:
        path.write_bytes(canonical(data)+b"\n")
    else:
        assert json.loads(path.read_text())==data
    print(json.dumps(data["summary"],indent=2))


if __name__ == "__main__":
    main()
