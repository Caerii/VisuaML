# Match feedback changes while varying the threshold

Compare two calibration sets with the same 630 ordered feedback pairs,
rules, and starting thresholds. The control holds each threshold fixed.
The joint set shifts the donor threshold by one or two in either direction,
excluding donor thresholds reserved for evaluation. The fixed set repeats
a pair once per admissible joint shift to preserve matched weighting.
Both fit the same two-coordinate map to a scalar edit along the recipient
direction. The combined fit uses both sets.

| Calibration | Correct switches / 216 | Preserved decisions / 144 |
| --- | ---: | ---: |
| v97 fixed-threshold fit | 194 | 140 |
| Matched fixed-threshold fit | 194 | 140 |
| Matched small threshold shifts | 194 | 116 |
| Combined fit | 204 | 116 |
| Local donor control | 188 | 138 |

Small threshold changes cost 24 preservation cases at the same aggregate
switch count as the matched control. Combining the sets gained ten intended
switches and incurred the same 24 preservation losses in aggregate. The
matched fixed fit performed well on both metrics relative to the other
cross-checkpoint fits. The local donor control is an intervention baseline
and can itself miss the intended decision.

Evaluation uses 180 distinct input pairs per recipient. The script checks
that none matches an exact v94–v97 pair. The recipient thresholds and model
checkpoints remain the same. Matching numeric changes removes the feedback
jump confound from v97. Changing the donor threshold can also change its
relation to feedback, so the result measures the joint effect of including
those threshold variations in calibration. It does not establish that
other joint-pair designs would have the same tradeoff.

From `visuaml-client/voxel-replay` in the research Python environment:

```sh
export PYTHONPATH=.:../backend
python tools/replay_matched_threshold_calibration.py
python tools/replay_paired_comparison_calibration.py
```

[Calibration pairs, per-seed scores, and raw logits](../examples/matched-threshold-calibration/replay.json)
