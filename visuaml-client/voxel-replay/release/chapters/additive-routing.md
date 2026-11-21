# Move the feedback activation while retaining the recipient rule

The earlier full-output patch overwrote the recipient's rule contribution at
the joint condition site. This experiment uses a donor pair from one
calibration threshold. The exact Rust tool supplies 8 and 9 to two donor runs
with the same rule. Subtract the activation for 8 from the activation for 9
at the first linear layer and freeze that direction for each checkpoint.

Add the saved direction to a recipient's own activation for an upward move
and subtract it for a downward move. Apply the recipient's rule after fixing
its actual feedback. Thresholds 7, 13, 21 and 27, two registry families,
both directions, and three rules give 96 assigned cases for each gap.
There are two unchanged router checkpoints. A direct input intervention
supplies the expected downstream logits for each case.

| Edit | Gap of 1 | Gap of 3 |
| --- | ---: | ---: |
| Add or subtract one preactivation delta | 96/96 | 48/96 |
| Scale the preactivation delta by the numeric gap | 96/96 | 96/96 |
| Scale a delta measured after tanh | 64/96 | 64/96 |
| Replace with the whole calibration donor activation | 64/96 | 64/96 |
| Apply the scaled preactivation delta in the wrong direction | 0/96 | 0/96 |

For the linear layer, the measured direction equals the feedback weight
column divided by 31 within floating-point error. Holding rule and
threshold fixed, this edit changes the feedback contribution by the
requested amount. All scaled cases matched the direct input intervention
within `9e-6` on every routing logit. Positive and negative
directions share the same fitted vector per model. Every edit preserves
the first-choice scores.

The gap-of-three result makes the scope of the mechanism clear. A fixed
one-unit edit crosses the boundary in some cases and leaves others on the
same side. The edit transports a numeric displacement at the linear site.
Scaling uses the actual numeric gap supplied by the experiment.

## Execute a delayed-rule witness

The selected recipient actually receives 6 from Rust WASM. The same
calibrated one-unit delta derived from 8 and 9 is applied to each branch.
The recipient policy chooses Boost, Trim, or abstention. The two follow-up
programs both receive 6 and return 11 and 4. The abstention branch emits
no follow-up request. All tool captures match Wasmtime on return and full
memory and are runnable independently.

From `visuaml-client/voxel-replay`:

```sh
export PYTHONPATH=.:../backend
python tools/replay_additive_routing.py
python tools/visuaml_replay.py open release/examples/additive-routing/above_boost-report.json
python tools/visuaml_replay.py open release/examples/additive-routing/above_trim-report.json
```

The reproducer verifies the archive and blocks packed fallback. Open the
[saved replay](../examples/additive-routing/replay.json) in the browser
viewer. It contains the frozen direction, each applied activation digest,
the expected action, and the error against the direct feedback control.
