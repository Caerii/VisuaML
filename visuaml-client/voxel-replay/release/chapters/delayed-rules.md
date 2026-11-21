# Reveal the rule after the tool returns

A single donor activation is captured before the recipient rule is supplied
and reused unchanged under three rules. For the saved witness, the
recipient's Rust WASM tool returns 6 and the donor's returns 7. The
threshold is 7. The recipient continues to receive 6 in all three branches.
Applying the above-threshold donor activation yields:

| Rule revealed after feedback | Decision | Verified follow-up |
| --- | --- | --- |
| At or above threshold, Boost | Boost | 11 |
| At or above threshold, Trim | Trim | 4 |
| At or above threshold, abstain | Abstain | No request or tool execution |

The executable witness identifies the same donor tensor digest across the
three branches. The two requested tools receive the original value 6.
Both returns and their full memory pages agree with Wasmtime. The third
branch ends at abstention and has no second request event.

## What transferred

The factorized control's learned comparison reads the numeric input and
threshold. The harness supplies the downstream rule after freezing the
donor tensor. The third rule explicitly maps one branch to abstention.
It tests whether information can support abstention under a supplied
rule. The abstention policy is supplied explicitly.

For the frozen joint controller, the donor uses the ordinary rule with
relation 0. The recipient's rule is passed to the model after the donor
activation has been captured. The harness chooses a patch site from the
condition network's linear input, tanh, and output modules using only
thresholds 9, 17 and 23. All three sites tied at zero correct opposing-rule
transfers on each checkpoint, so the declared earliest-site tie break
selected `router.condition.0`. Evaluation then uses thresholds 7, 13, 21
and 27. Both original controllers' unpatched endpoints follow each
recipient rule on the tested inputs.

Across two checkpoints, two registry description variants, four evaluation
thresholds and both transplant directions, there are 32 cases per rule.
The factorized control scores 32/32 for each rule. The original full
patch scores 0/32 on the reversed rule and 32/32 on each of the other two.
This bounds the tested full-output intervention in the original condition
network. It leaves other sites, subspaces and model architectures open.

## Replay

From `visuaml-client/voxel-replay`:

```sh
export PYTHONPATH=.:../backend
python tools/replay_delayed_rules.py
python tools/visuaml_replay.py open release/examples/delayed-rules/above_boost-report.json
python tools/visuaml_replay.py open release/examples/delayed-rules/above_trim-report.json
```

The reproducer refits the scalar on its original training data and checks
the archived weights, calibration counts, activation digests, decisions
and tool reports. It blocks packed fallback. Load
[the saved replay](../examples/delayed-rules/replay.json) in the viewer.
The feedback phase, donor tensor freeze, rule reveal and action are all
recorded in the selected witness.
