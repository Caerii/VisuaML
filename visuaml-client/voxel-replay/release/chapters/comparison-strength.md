# Sweep the strength of an aligned comparison edit

The v94 calibration and three frozen controllers supply the same source axis,
recipient axes, and scalar alignment. Multiply the aligned activation edit
by a shared gain. The fixed grid is 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3,
and 4. Each recipient sees 180 new live/donor combinations, disjoint from
v94's exact pairs, at the same recipient thresholds. The evaluation keeps
the recipient rule and tests threshold changes with fixed feedback, feedback
changes with fixed threshold, and cases where numeric feedback alone
predicts a different action from the donor's comparison.

| Gain | Correct switches / 216 | Unwanted switches / 144 | Recovered unit-gain misses / 54 |
| ---: | ---: | ---: | ---: |
| 0.25 | 0 | 0 | 0 |
| 0.5 | 2 | 0 | 0 |
| 0.75 | 106 | 0 | 0 |
| 1 | 162 | 8 | 0 |
| 1.25 | 200 | 40 | 38 |
| 1.5 | 214 | 62 | 52 |
| 2 | 216 | 72 | 54 |
| 3 | 210 | 72 | 54 |
| 4 | 186 | 72 | 54 |

An unwanted switch changes a correct baseline decision in a case where the
donor relation calls for preserving it. The local rank-one edit, which uses
the recipient donor activation, gets all 216 switches and 144 preservation
cases right. The aligned edit at gain 2 fixes every unit-gain miss, and
changes half the decisions that should stay fixed. Its median target logit
margin on unit-gain misses changes from −5.27 to +9.52 for seed 829 and from
−6.69 to +9.68 for seed 1049. At gains 3 and 4, some previously correct
switches fail even though the original misses stay recovered.

A stronger edit is sufficient to recover the missed switches on this grid.
No tested shared gain matches the local edit on both switching and
preservation. This result applies to the current alignment and direction.
It leaves open whether a better scalar mapping can recover the local
behavior. The most direct follow-up is to predict the recipient's
calibration projection using a richer scalar mapping, then compare its
held-out switch and preservation scores with a second-direction control.

From `visuaml-client/voxel-replay` in the research Python environment:

```sh
export PYTHONPATH=.:../backend
python tools/replay_comparison_strength.py
python tools/replay_comparison_alignment.py
```

[Per-seed counts and per-case logits](../examples/comparison-strength/replay.json)
