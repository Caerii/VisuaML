# Replay a learned decision from live WASM feedback

The first tool is a new WASM guest authored after all three controllers were
frozen. It increments its i32 input, stores the result at byte 32 of its fixed
64 KiB page, and loads it back. Each case runs a live and an alternate input.
The comparison cases use thresholds 8, 19, and 25, which were held out from
training for seeds 617 and 829. The original controller serves as a reference
and may have seen those thresholds in training. A separate replay preserves
its original v91 failure at threshold 7.

The host runs WASM, forwards the selected feedback, applies the experimental
activation edits, and dispatches the chosen follow-up. The controller chooses
Boost, Trim, or abstention. The host always passes the live first-tool return
as the follow-up argument. For the direct substitution, only the controller's
feedback input uses the alternate return. The two activation interventions leave
the controller's input tokens intact and transplant block-zero outputs from
the alternate run, which uses a different rule token. The recorded logit margin
targets the alternate action in all four modes.

| Controller | Cases | Natural | Direct substitution | Threshold patch | Joint patch |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 162 | 162 | 162 | 130 | 162 |
| Seed 617 | 162 | 162 | 162 | 162 | 162 |
| Seed 829 | 162 | 162 | 162 | 162 | 162 |

The grid crosses below, equal, and above-threshold returns with the three
recipient rules and their distinct donor rules. A case passes if the requested
tool and argument and the final return match the independent rule and arithmetic
expectations. Abstention requires the absence of a second call. Calls are
executed once per distinct tool and argument, then reused across cases.
The reference runner and Wasmtime agree on the return and full memory-page
digest for all 10 distinct first calls and 18 distinct follow-ups. Packed
fallback is blocked throughout.

## Open the first failure

The first failure in the fresh case grid is row 18 (zero-based) of the original
controller's replay. WASM returns 8 from input 7. The alternate argument 6
produces 7. At threshold 8, the threshold-only patch leaves Trim's margin at
−0.668, so the first wrong event is the second request, Boost(8). It returns 13.
The joint patch raises Trim's margin to +8.904 and issues Trim(8), which returns
6. Both exact executions match Wasmtime. The error occurs in the learned
choice under the intervention.

From `visuaml-client/voxel-replay` with PyTorch and Wasmtime installed:

```sh
export PYTHONPATH=.:../backend
python tools/replay_live_tool_interventions.py
for trace in first_call alternate_first_call threshold pair; do
  python tools/visuaml_replay.py open "release/examples/live-tool-interventions/holdout-$trace.json"
done
```

[WASM source, frozen binary, traces, and raw controller outputs](../examples/live-tool-interventions/)
