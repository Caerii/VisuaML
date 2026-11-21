# Test a routing subspace and execute the selected tools

This experiment asks how much of a previously successful donor patch is
needed to reproduce the routing switch. It keeps the two archived width-32
controllers unchanged and fits a basis independently for each checkpoint.
The hook operates on the output of `router.condition.1`.

For calibration, take the activation difference just below and at
thresholds 9, 17, and 23, using both relation polarities. Fit the leading
directions with uncentered SVD. Freeze the bases before evaluating
thresholds 7, 13, 21 and 27. The evaluation reuses the two registry families
and rotated slots from the earlier controlled routing study. Both
unpatched endpoints selected their expected tool in all 128 assigned pairs.

| Rank | Projected edit reproduces donor choice | Complement reproduces donor choice | Norm-matched random edit reproduces donor choice |
| ---: | ---: | ---: | ---: |
| 1 | 2 / 128 | 126 / 128 | 2 / 128 |
| 2 | 128 / 128 | 0 / 128 | 5 / 128 |
| 4 | 128 / 128 | 0 / 128 | 5 / 128 |

The complement is the donor change with the projected component removed,
applied to the same recipient. All patches keep the first-choice logits
identical and use the original WASM feedback. The rank-one direction
preserved a median 64.3% of squared donor-delta norm and rarely reproduced
the behavior. Rank two preserved 99.99% at the median. The result identifies
a sufficient subspace in these cases and tests what removing that
component does to the donor edit. It does not establish a universal
two-dimensional mechanism or necessity under other interventions.

## Run the two selected WASM tools

The selected witness starts with a real WASM return and model feedback of
6 in both executions. The rank-two patch changes the selected registry
slot from 0 to 1. The baseline executes the add-five Rust tool and returns
11. The patched run executes the subtract-two Rust tool and returns 4.
Both follow-ups take 6 as their argument. Each capture checks the return
and complete page against Wasmtime. The first difference in the pipeline
journal is the second request.

From `visuaml-client/voxel-replay`:

```sh
export PYTHONPATH=.:../backend
python tools/replay_routing_subspaces.py
python tools/visuaml_replay.py open release/examples/routing-subspaces/baseline-tool-report.json
python tools/visuaml_replay.py open release/examples/routing-subspaces/patched-tool-report.json
```

[Saved replay and tool reports](../examples/routing-subspaces/) include
the weights' identities, calibrated bases, logits, and the comparison.
Open `replay.json` in the browser viewer. The script checks the experiment
and tool reports against the archive. Use `--record` to regenerate it.

The subtract-two guest was compiled with Rust 1.100.0:

```sh
rustc release/examples/routing-subspaces/trim.rs --crate-type=cdylib \
  --target=wasm32-unknown-unknown -C opt-level=z -C panic=abort \
  -C link-arg=-zstack-size=0 -C link-arg=--global-base=1024 \
  -C link-arg=--initial-memory=65536 -C link-arg=--max-memory=65536 \
  -o release/examples/routing-subspaces/trim.wasm
```
