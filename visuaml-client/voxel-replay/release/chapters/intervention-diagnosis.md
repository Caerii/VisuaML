# Diagnosing failed threshold interventions

The original v90 transformer missed 14 intended switches when the intervention
patched only the first block's threshold-token output. Replaying those same cases
and patching the feedback and threshold outputs together recovered all 14.
The rule and query outputs at that block retain their recipient values.

In the saved failure, live feedback is 7 and the donor feedback is 6 at
threshold 7. The intended follow-up is Trim. The target logit minus its
strongest rival is −8.113 without the edit, −0.569 with the threshold edit,
−2.840 with the feedback edit, and +8.728 with both edits. Directly substituting
the donor feedback in the controller input yields +8.691. The WASM call still
receives the original live value 7. Boost returns 12 and Trim returns 5.

Masking the query-to-feedback attention edge in the final block rescues 10 of the
14 threshold-only failures. With the edge masked and no token patch, the
controller gets only 48 of the 96 original cases right. The disruption supports
a role for this path and leaves open how the blocks combine evidence.

## Replication

We independently trained two checkpoints, withholding thresholds 9 and 17 for
intervention selection and 8, 19, and 25 for evaluation. Selection maximizes
calibration correctness, breaking ties in favor of fewer patched positions.
Both checkpoints selected threshold-only. The held-out design crosses distinct
recipient and donor rules, both orientations, and near and farther donors.

| Checkpoint seed | Threshold switches | Feedback switches | Paired switches | Both decisions and final output |
| --- | ---: | ---: | ---: | ---: |
| 617 | 72/72 | 6/72 | 72/72 | 144/144 |
| 829 | 72/72 | 0/72 | 72/72 | 144/144 |

All 72 preserve cases per checkpoint retain their intended choice for every
patch. Moving the donor farther from the threshold on the same side changes
zero of 72 matched threshold-patch switch decisions across the two checkpoints.
These are descriptive results for two seeds and the selected threshold and rule
grid. The original failure shows why threshold-only transfer still requires
validation for each artifact.

The two-decision check uses numeric first-feedback fixtures and runs the Boost
and Trim follow-ups as WASM. Abstention issues no WASM call. The harness checks
the 30 distinct follow-up tool and argument combinations against Wasmtime,
including the full memory page, with packed fallback blocked. It reuses these
exact results for repeated combinations.

## Reproduce

From `visuaml-client/voxel-replay`, using the research Python environment with
PyTorch and Wasmtime installed:

```sh
export PYTHONPATH=.:../backend
python tools/replay_intervention_diagnosis.py
python tools/visuaml_replay.py open release/examples/intervention-diagnosis/expected-tool-report.json
python tools/visuaml_replay.py open release/examples/intervention-diagnosis/observed-tool-report.json
```

The default command checks the archived results. Use `--train` to regenerate
the two model checkpoints and results. The saved replay includes per-case
margins, calibration scores, and the first incorrect stage if a chain fails.

[Raw replay and frozen weights](../examples/intervention-diagnosis/)
