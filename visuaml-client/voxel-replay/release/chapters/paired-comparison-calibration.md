# Fit the change in the comparison activation directly

This experiment keeps v96's two source coordinates and one recipient edit
direction. It fits the change in the recipient projection directly from the
change in the two source projections. Both matched calibration sets contain
180 ordered pairs. They use the same rules and the same offsets from each
pair's local threshold. The fixed set holds the threshold at 9 or 17 within
each pair. The crossed set moves between 9 and 17. A third fit combines both
sets. The recipient donor activation supplies the projection target during
calibration. During evaluation, only the source donor and source live states
determine the edit applied to the recipient.

| Calibration | Correct switches / 216 | Preserved decisions / 144 |
| --- | ---: | ---: |
| v96 linear scalar baseline | 192 | 102 |
| v96 two source coordinates, one recipient direction | 214 | 96 |
| Fixed-threshold pairs | 204 | 132 |
| Crossed-threshold pairs | 176 | 120 |
| Combined pairs | 174 | 120 |
| Local donor control | 204 | 144 |

The fixed-threshold fit preserves 36 more decisions than the v96
two-coordinate fit and loses 10 intended switches. Compared with the v96
scalar baseline, it improves both counts. Its switch total equals the local
control's total. The local control preserves every decision.

The crossed design uses threshold shifts of eight in either direction. Even
with matched offsets, that changes the absolute feedback jumps and the
distribution of activation deltas. The result therefore applies to the tested
pair designs. It leaves open whether fitting on a range of threshold shifts
would improve the result.

The evaluation contains 180 distinct live and donor pairs per recipient and
shares no exact pairs with v94, v95, or v96. It uses the same recipient
thresholds of 8, 19, and 25. Separate switch and preservation scores use the
donor comparison under the recipient's decision rule. Both recipients use
one frozen source checkpoint.

From `visuaml-client/voxel-replay` in the research Python environment:

```sh
export PYTHONPATH=.:../backend
python tools/replay_paired_comparison_calibration.py
python tools/replay_comparison_mappings.py
```

[Raw per-case logits, calibration pairs, and fitted maps](../examples/paired-comparison-calibration/replay.json)
