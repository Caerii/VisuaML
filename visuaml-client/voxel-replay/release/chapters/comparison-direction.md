# A rank-one comparison intervention across training seeds

Seeds 617, 829, and 1049 use the same architecture, training data, optimizer
settings, and threshold exclusions. Initialization and minibatch order vary
with seed. The first two checkpoints come from v91. The reproducer trains
the third with the same recipe.

For each checkpoint, fit one direction to the mean difference between paired
first-block feedback and threshold activations at calibration thresholds 9 and
17. Each calibration pair holds the rule fixed and changes the feedback from
one below threshold to equal. At evaluation, the donor uses a different rule
and the intervention projects its activation difference onto the fitted
direction. Only one scalar projection is transplanted across the two token
positions. The recipient's rule and query positions are unchanged at the
intervention site.

The evaluation crosses thresholds 8, 19, and 25 with four feedback values and
four donor values around each boundary. It tests both switch directions,
equality, abstention, and same-side controls. These thresholds were excluded
from training and direction fitting and were used in prior studies.

| Seed | Expected switches with local direction | Same-side decisions preserved | Expected switches with seed 617 direction | Natural decisions correct after midpoint replacement |
| --- | ---: | ---: | ---: | ---: |
| 617 | 144/144 | 144/144 | 144/144 | 72/144 |
| 829 | 144/144 | 144/144 | 0/144 | 72/144 |
| 1049 | 144/144 | 144/144 | 0/144 | 72/144 |

Matching the magnitude of the locally fitted patch leaves the transferred
direction at zero successful switches on both recipient seeds. Across five
random repeats, row-wise norm-matched random directions yield 61 intended
switches in 2,160 trials. These are repeated interventions on the same held-out
grid. The no-op control preserves all baseline logits exactly. Replacing the
local projection with its calibration midpoint reduces natural decision
accuracy from 100% to 50% throughout the grid. Midpoint replacement is an
activation perturbation and may create states outside the training distribution.

The fitting recipe transfers across the three seeds. The raw direction from
seed 617 fails to transfer across their native coordinates. The experiment
supports a causal role for the locally fitted direction and leaves open
whether the checkpoints share an aligned representation. It also leaves open
whether the direction encodes a comparison result or changes the feedback
quantity the controller compares downstream. The patch uses live donor
activations from the recipient model. The source direction in the
cross-checkpoint control is the only component taken from another model.

From `visuaml-client/voxel-replay` with the research Python environment:

```sh
export PYTHONPATH=.:../backend
python tools/replay_comparison_direction.py
python tools/replay_live_tool_interventions.py
```

The first command checks the archived per-case scores and margins. Use
`--train` to regenerate the third checkpoint. The second command verifies the
existing live WASM regression. The rank-one experiment itself uses numeric
inputs and tests action selection. [Raw results and third checkpoint](../examples/comparison-direction/)
