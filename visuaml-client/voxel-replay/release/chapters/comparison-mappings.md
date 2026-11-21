# Compare scalar and two-direction activation mappings

Seed 617 supplies the live and donor projections for two frozen recipient
checkpoints. Fit every map on thresholds 9 and 17. The original direction
uses feedback changes at fixed thresholds. The second direction uses the
mean threshold contrast across matched feedback values and rules, with the
first direction projected out. Both directions use only calibration data.

The scalar nonlinear map fits a quadratic to the original source
projection. The two-coordinate linear map predicts both recipient
projections. Its one-direction control uses the same two source projections
and applies only the prediction along the original recipient direction.
Every edit adds the difference between mapped donor and mapped live
projections to the recipient activations. The recipient's rule and query
token outputs stay fixed at the intervention site.

| Map | Correct switches / 216 | Preserved decisions / 144 |
| --- | ---: | ---: |
| Linear scalar | 172 | 122 |
| Quadratic scalar | 168 | 118 |
| Two source coordinates, one recipient direction | 208 | 118 |
| Two source coordinates, two recipient directions | 206 | 114 |
| Local rank-one donor control | 214 | 144 |

The one-direction two-coordinate map improves switch accuracy on both
recipients and loses four preservation cases in aggregate relative to the
linear scalar baseline. All 26 preservation failures occur where the
donor changes feedback to the opposite side of the live threshold but also
changes its own threshold, so its comparison calls for preserving the
decision. The local rank-one control preserves all of those cases and
misses two intended switches on seed 1049. Adding recipient edit directions
did not improve either metric over the one-direction two-coordinate map.

The test uses 180 input pairs per recipient, disjoint from the exact v94
and v95 pairs, with the same three live thresholds. Donor thresholds vary
and can include thresholds used in training or calibration. These results
describe two transfers from one source. They motivate predicting the local
scalar delta from both source coordinates more accurately, especially on
preservation pairs. The second coordinate is a calibration contrast and
its causal meaning remains open.

From `visuaml-client/voxel-replay` in the research Python environment:

```sh
export PYTHONPATH=.:../backend
python tools/replay_comparison_mappings.py
python tools/replay_comparison_strength.py
```

[Raw logits, fitted coefficients, and basis vectors](../examples/comparison-mappings/replay.json)
