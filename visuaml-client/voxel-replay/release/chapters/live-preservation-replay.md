# Replay preservation failures through live WASM

Select every preservation failure from the v98 matched fixed-threshold fit.
The unchanged first guest runs with arguments 28 and 22 to produce live
feedback 29 and donor feedback 23. The host supplies those returns to the
same frozen controllers and reapplies the calibrated edit. Each selected
case preserves the intended action under the unedited controller and
reproduces the archived wrong action under the edit.

| Checkpoint | v98 rows (zero-based) | Expected outcome | Patched dispatch |
| --- | --- | --- | --- |
| 829 | 169, 179 | Abstain | Boost(29) returns 34 |
| 1049 | 129, 139 | Boost(29) returns 34 | Trim(29) returns 27 |

The first divergent event in all four cases is the follow-up tool choice.
Every invoked tool agrees with Wasmtime on the return and complete memory
page digest. The harness blocks packed fallback. Abstention makes no call.

## Argument readout

Fit a ridge regression probe to the first block's feedback token using
values 0–31 at calibration thresholds 9 and 17. The probe estimates the
live feedback from the frozen state. The host selects the nearest of two
explicit candidates and uses that value for dispatch. The baseline probe
estimates 29.04 and 29.12 on the respective checkpoints. After patching,
the estimates drop to 27.33 and 27.58, and both still select 29 over 23.

This is a separately fitted probe with host-side candidate selection.
It tests whether the live argument is still recoverable when routing fails.
The frozen controller supplies the tool choice. Every case uses the same
numeric pair and each checkpoint has two donor rule variants. The four
replays establish neither a general argument-binding rate nor the
smallest candidate separation at which the probe remains reliable.

From `visuaml-client/voxel-replay` in the research Python environment:

```sh
export PYTHONPATH=.:../backend
python tools/replay_live_preservation.py
python tools/visuaml_replay.py open release/examples/live-preservation-replay/1049-129-learned_argument.json
```

[Downloadable request comparison and exact WASM traces](../examples/live-preservation-replay/)
