# Transfer an activation in a second controller, then execute the consequence

This experiment uses a separately trained, two-block causal transformer.
It reads four positions in order: numeric feedback, numeric threshold,
recipient rule, and query. A classifier on the query selects Boost,
Trim, or abstention. Causal attention prevents the threshold position
from reading the later rule. The transformer shares neither weights
nor condition-network architecture with the earlier router.

The [intervention specification](../examples/compositional-interventions/specification.json)
binds the controller weights, activation site, token position, input layout,
calibration split, and predicted observable. Training omits the calibration
thresholds 9 and 17 and the evaluation thresholds 7, 13, 21 and 27.
Calibration selects one of eight layer-position candidates using opposing
donor and recipient rules. It picked the threshold token after the first
block with 40 of 48 calibration cases. That site is frozen for evaluation.
The experiment swaps the complete 24-value token from a donor at a matched
threshold. It measures the transfer of a token representation.

| Edit | Required switch | Required preservation | Total |
| --- | ---: | ---: | ---: |
| Selected donor token | 34/48 | 48/48 | 82/96 |
| Whole donor layer output | 8/48 | 8/48 | 16/96 |
| Norm-matched random edit at the token | 0/48 | 48/48 | 48/96 |
| Original recipient output | 0/48 | 48/48 | 48/96 |

All donor and recipient baselines match their own rule. A hook with no
edit and a replacement with the original tensor preserve the baseline.
These are structured cases on one trained checkpoint. The rule types
appear in training. The evaluation changes thresholds and crosses the
recipient and donor rules.

## Two futures from the same donor token

The first Rust tool returns 6 for the recipient and 7 for the donor.
The donor token has the same digest under both receiver rules. With
actual feedback fixed at 6:

- The Boost recipient runs the Rust add-five tool and gets 11.
  The transformer consumes 11 under the next rule and threshold.
  It chooses Boost again and the second tool returns 16.
- The abstention recipient issues no follow-up request.

The second decision is an ordinary transformer forward on the new
verified feedback and rule. The patch is applied at the first decision.
This establishes the composition for the saved witness. The 96-case
score covers the first decision.

The first evaluation failure is recorded at the first request. With actual
feedback 7 and donor feedback 6, the edited model chose Boost. The
predicted rule choice was Trim. The saved reports run both tools on 7
and return 12 and 5 respectively. Every executed WASM witness was
checked against Wasmtime for the return and full page, with packed
fallback blocked.

## Reproduce

From `visuaml-client/voxel-replay`:

```sh
export PYTHONPATH=.:../backend
python tools/replay_compositional_interventions.py
python tools/visuaml_replay.py open release/examples/compositional-interventions/second-tool-report.json
python tools/visuaml_replay.py open release/examples/compositional-interventions/failure-observed-tool-report.json
```

The script reads the frozen specification and verifies it against the
weights and calibration. It checks the held-out outcomes and runnable
reports against the archive. Use `--train` to retrain and rebuild the
specification and archive. Open the [saved replay](../examples/compositional-interventions/replay.json)
in the browser viewer to inspect the success and failure.
