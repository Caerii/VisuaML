# Align a comparison intervention across checkpoints

Use seed 617 as the source and seeds 829 and 1049 as recipients. Fit the
existing rank-one direction separately in each model. On paired calibration
inputs at thresholds 9 and 17, fit a scalar least-squares slope from source
projections to recipient projections. At evaluation, take the difference
between source donor and source live projections, scale it by that slope,
and add it along the recipient direction. The aligned patch uses no
recipient donor activation. The recipient's rule and query token outputs
remain intact at the intervention site.

| Case | Aligned transfer | Local direction | Raw source direction |
| --- | ---: | ---: | ---: |
| Same feedback, threshold changes the decision | 70/72 | 72/72 | 0/72 |
| Feedback changes the decision at a fixed threshold | 72/72 | 72/72 | 0/72 |
| Feedback changes on the same side | 72/72 | 72/72 | 72/72 |
| Feedback alone predicts a switch, donor comparison predicts preservation | 72/72 | 72/72 | 72/72 |
| Feedback alone predicts preservation, donor comparison predicts a switch | 56/72 | 72/72 | 0/72 |

The score uses the action dictated by the donor comparison and recipient
rule. The local and raw controls project activation differences computed
in the recipient model onto the local and source directions respectively.
Preservation rows also pass without editing. The two switch contrasts
where feedback and comparison disagree test whether transferring raw
feedback alone could explain the effect.

For example, in seed 829, the live comparison is 7 against 8. The donor
compares 5 against 5. The local edit switches to Boost, the action for the
donor's comparison under the recipient rule. The aligned edit retains Trim.
Across the two switch contrasts where the explanations differ, all 18
aligned misses retain the original decision.

The result supports a comparison-sensitive signal that can be transferred
with calibration. It does not isolate a discrete comparison bit or establish
a model-independent circuit. The intervention could transfer a signed margin
or another quantity computed from both inputs. The fit uses each recipient's
calibration activations. The study evaluates two transfers from one source
seed. Recipient thresholds 8, 19, and 25 were withheld during training and
calibration. Donor thresholds vary with the case and can include training
or calibration thresholds. These are structured input cases on frozen
controllers. Argument binding and WASM execution are outside this experiment.

From `visuaml-client/voxel-replay` in the research Python environment:

```sh
export PYTHONPATH=.:../backend
python tools/replay_comparison_alignment.py
```

The script verifies the saved logits, predictions, slopes and checkpoint
hashes. [Raw comparison replay](../examples/comparison-alignment/replay.json)
