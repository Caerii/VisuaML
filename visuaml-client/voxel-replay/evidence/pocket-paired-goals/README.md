# Pocket paired goals

One frozen learned controller sees the same checkpoint under two controlled-language objectives. Both use a deadline. The edit objective prefers Wait when it reaches the target. The speed objective minimizes arrival time, preferring Wait on a tie.

The TorchScript graph owns request generation and validation, the call budget, the exact simulation, first-arrival extraction, feedback, and the final generated answer. The controller itself receives only prompt and reply tokens. The underlying forecast operation is unchanged. The host supplies the checkpoint and prompt. The graph inspection checks for Python callbacks and uninlined method calls.

The artifact was frozen in `4a849cc` before the holdout fixture was authored. Training used synthetic demonstrations over admissible arrival pairs, with edit-first demonstrations beginning with Wait and speed demonstrations beginning with Open. It saw zero world layouts. The experiment measures transfer of that supervised decision policy to hidden, simulated state.

## Evidence and results

Opening removes a single wall, so it cannot delay a reachable arrival. Zero denotes no arrival by the deadline. The evidence oracle enumerates every arrival pair consistent with that rule and the received values. An action is justified only when all consistent pairs have the same optimum. A controller that stops with exhausted budget and ambiguous evidence gets credit for handling uncertainty, even when the omniscient oracle could pick an action. In particular, a successful Open could tie with Wait, while a failed Open implies Neither. A Wait-only success is impossible under this world rule.

The holdout has 96 checkpoints, 24 per outcome class. The generator varies distance, orientation, placement and isolated terrain in one corridor-and-detour construction. The four classes cover a faster shortcut with a successful detour, only Open reaching by the deadline, ties, and neither action reaching.

| Goal | Budget | Correct determinate decisions | Appropriate stops | Calls |
| --- | ---: | ---: | ---: | ---: |
| Minimize edits | 1 | 48/96 | 48/96 | 96 |
| Earliest arrival | 1 | 24/96 | 72/96 | 96 |
| Minimize edits | 2 | 96/96 | 0/96 | 144 |
| Earliest arrival | 2 | 96/96 | 0/96 | 168 |

The goal switched the first call on every checkpoint at both budgets. There were zero unsupported recommendations, missed available forecasts, or unnecessary forecasts in the unmodified runs. Both goals used 192 calls per 96 worlds under the exhaustive two-call baseline. With two calls, Wait-first used 144 under the edit goal and 192 under the speed goal. Open-first used 168 under either goal. One-call fixed order controls and the disabled, precomputed and distant-edit results are in the raw outcome archive.

Both interventions matched their predeclared predictions on all 24 shortcut worlds. Reporting Wait as unsuccessful under the edit goal triggered Open and changed the recommendation. Reporting Open as unsuccessful under the speed goal skipped Wait and produced Neither. These outcomes are scored against the supplied feedback. The real checkpoint is unchanged. Distant edits and precomputed replies preserved all requests and recommendations. The disabled tool caused all 384 runs to stop.

All 1,080 executed forecasts matched the independent reference. A full-world breadth-first shortest-path search scores arrival times. Scalar evolution checks regional frames and the full-world target dependency cone. Language admission remains scoped to the previous experiment.

## Run and replay

From `visuaml-client/voxel-replay` in the research Python environment with PyTorch:

```sh
PYTHONPATH=.:../backend python tools/evaluate_pocket_paired_goals.py
PYTHONPATH=.:../backend python tools/evaluate_pocket_paired_goals.py --replay evidence/pocket-paired-goals/replays.json
npm ci
npm run dev
```

Open `/pocket-paired-goals.html`. Step both goals on the same saved checkpoint, reduce the budget to one, or replace the first forecast in both replays. The browser presents the recorded requests, verified forecasts, interventions and decisions. The Python command checks the artifact identity and re-executes the saved examples.
