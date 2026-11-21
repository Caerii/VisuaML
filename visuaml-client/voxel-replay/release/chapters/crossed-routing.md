# Transfer a comparison outcome across a different rule

The previous rank-two experiment reproduced a routing change under the same
rule. This experiment changes the donor's and recipient's rules independently.
The expected answer takes the **donor's threshold outcome** and applies the
**recipient's rule and tool order**. Some pairs require switching. Others
require preserving the recipient's choice even though the donor picked the
other tool.

Each frozen checkpoint contributes 512 cases. The cases cross four evaluation
thresholds, both sides of each threshold, both relation polarities, both
orders of the named tool choices, and two registry description variants.
The donor and recipient also use different physical registry permutations.
Scores align by semantic tool identity. Both unpatched endpoints choose
correctly for all 1,024 assigned cases per architecture.

## Original controller and positive control

The original joint controller receives the numeric inputs and relation
together. Patch its condition network's tanh output. Also test the rank-two
basis fitted on the earlier separate calibration thresholds, its complement,
a norm-matched random edit, and a no-op.

The factorized positive control shares the archived tool embeddings. A
separately trained scalar reads only the numeric inputs. The module then
binds the recipient rule by preserving or flipping the scalar before routing.
Training excludes all four evaluation thresholds. The architectural separation
is explicit. This experiment checks that the intervention can detect
separation when it is provided.

| Patch | Opposite-rule, same tool order, outcome must switch | Opposite-rule, same tool order, outcome must remain |
| --- | ---: | ---: |
| Joint full donor | 0 / 128 | 0 / 128 |
| Joint rank two | 64 / 128 | 64 / 128 |
| Joint complement | 64 / 128 | 64 / 128 |
| Joint no-op | 0 / 128 | 128 / 128 |
| Factorized full donor | 128 / 128 | 128 / 128 |
| Factorized no-op | 0 / 128 | 128 / 128 |

The full joint patch reproduced the donor's preference under opposite rules
with fixed choice order. The partial joint patch passed some crossed cases.
Its complement passed the same number, so the switch results alone cannot
establish that this basis isolates the comparison outcome. Across all
assigned cases, the joint full patch scored 512/1,024, joint rank two
scored 768/1,024, and the factorized full patch scored 1,024/1,024.
These are structured, correlated cases on two existing checkpoints.

## Follow the edit into an exact tool call

Both witnesses keep the recipient's first WASM return and forwarded
feedback fixed. A comparison-outcome transplant and a joint-preference
transplant lead to different follow-up programs:

| Fixed feedback | Factorized patch | Joint patch |
| ---: | --- | --- |
| 7 | Boost returns 12 | Trim returns 5 |
| 6 | Trim returns 4 | Boost returns 11 |

Each follow-up runs the selected Rust WASM binary on the original
feedback. Capture compares its return and complete page with Wasmtime
and the reproducer blocks packed fallback. The first differing boundary
is the second request in both witnesses.

From `visuaml-client/voxel-replay`:

```sh
export PYTHONPATH=.:../backend
python tools/replay_crossed_routing.py
python tools/visuaml_replay.py open release/examples/crossed-routing/case-1-factorized.json
python tools/visuaml_replay.py open release/examples/crossed-routing/case-1-joint.json
```

The script refits the positive control deterministically and verifies
the learned scalar parameters, bases, predictions, and six tool reports.
Use `--record` to refresh the archive. Load [replay.json](../examples/crossed-routing/replay.json)
in the browser viewer for the two execution witnesses.
