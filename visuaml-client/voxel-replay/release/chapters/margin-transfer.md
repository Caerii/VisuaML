# Transfer a threshold side and preserve the recipient's margin magnitude

Earlier additive edits required a supplied numeric change. This experiment
uses the donor's decoded threshold side as the only donor input to the edit.
The recipient's activation supplies the magnitude. The frozen model's weights
are used for inference. Calibration reads activations from four controlled
forwards and recovers three affine coordinates at the first linear layer:
feedback, threshold, and rule.

Define the integer-aware margin as feedback minus threshold plus 0.5.
The half-unit offset places the boundary between consecutive integers.
When the donor's side differs, reflect the recipient margin. When the
sides agree, the edit is an exact no-op. There are two ways to change the
sign while preserving magnitude. Move feedback while holding the threshold
fixed, or move the threshold while holding feedback fixed. Both paths
preserve the recipient rule. The edit receives a decoded desired side and
an activation. It derives the distance from that activation.

## What the comparison establishes

Across two frozen checkpoints, four evaluation thresholds, two registry
families, both recipient sides, three distances and four donor values,
there are 384 donor–recipient cases. Applying three rules gives **1,152
decisions per edit axis**. Half are sign changes and half preserve the
original side. Every decision matches its specified rule. Donors with
different margins on the same side produce identical edits and results.

| Control | Maximum logit difference |
| --- | ---: |
| Feedback-axis patch versus direct feedback change | 0.0000301 |
| Threshold-axis patch versus direct threshold change | 0.0000339 |
| Direct feedback change versus direct threshold change at the same resulting margin | 0.0704 |

The first two comparisons check that each activation edit matches its
own direct-input counterfactual. The third compares two different
counterfactual inputs with the same signed margin. They select the same
tool in these cases and can assign measurably different logits. The
margin is sufficient for the tested choices. A richer model of the
scores must account for the numeric context.

This is an explicit intervention procedure built from linear-site
calibration. The experiment tests what a one-bit donor message suffices
to control with access to the recipient activation. Its scope is the
two frozen routers and the structured evaluation cases.

## Verify the saved witness

The donor comes from a WASM return of 11. The recipient's WASM result
and forwarded feedback remain 4. The intervention uses the donor bit
to change the decoded feedback coordinate from 4 to 9 at threshold 7.
The recipient's delayed rules select Boost, Trim, or abstention.
Boost and Trim both take the original 4 and return 9 and 2. The
abstention branch emits no follow-up request. Tool capture checks
return and full page against Wasmtime with packed fallback blocked.

From `visuaml-client/voxel-replay`:

```sh
export PYTHONPATH=.:../backend
python tools/replay_margin_transfer.py
python tools/visuaml_replay.py open release/examples/margin-transfer/above_boost-report.json
python tools/visuaml_replay.py open release/examples/margin-transfer/above_trim-report.json
```

The reproducer checks the saved calibration, numeric controls, action
predictions, and four tool reports. Load the [saved replay](../examples/margin-transfer/replay.json)
in the browser viewer.
