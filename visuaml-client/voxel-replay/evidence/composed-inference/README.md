# Composed inference

A single CPU TorchScript forward pass runs greedy language decoding, reads the emitted request, checks its arguments, selects the supplied program, runs the hard-attention interpreter, and inserts feedback into the controller context. It then decodes again. The inference host supplies tokenized prompts, admitted program images, roles, per-slot fuel, and an optional first-result intervention. It reads the returned evidence tensors after the call. Program-image admission stays on the host, as it does for the reference. The host makes no tool request, result-routing, or termination decisions during inference.

## Artifacts and interface

- `reference.pt` incorporates the original language controller. The replay matches all 15 normal, disabled, and substituted sessions against the separate-graph reference, including requests, tool register and memory traces, feedback, and answers.
- `conditional.pt` uses a separately trained controller with three named slots and a controlled-English conditional prompt. The artifacts were committed at `0eea384` before the final prompts and three tool images were added. Training withheld seed/count pairs satisfying `(3 * seed + count) % 7 == 0`. The threshold 12 and the conditional sentence structure were present in training. No broader language evaluation was performed.
- The CPU ABI accepts int64 prompt tokens, a batch of validated program matrices shaped `[slots, rows, 3]`, int64 role and fuel vectors indexed by slot, and an integer intervention value. Alpha, beta, and gamma select slots 0, 1, and 2. Role 0 requires a repeat count of 1. Other roles allow counts 1–9. The value range is 0–31. Fuel -1 disables a slot, 0 exhausts it without starting, and positive fuel budgets at most 48 interpreter cycles. There are at most two calls, three generations, six generated tokens per generation, and 64 context tokens.
- The result contains the generated token rows, per-call feedback status, raw and reported results, the published tool states, and the answer status. Each invocation initializes private tool state from the request. A rejected call never enters the executor and publishes no tool steps. The replay also checks that the supplied program tensors are unchanged.
- The branch exposes fuel as an interpreter loop bound. The same synthesized binary64 coefficients implement the admitted integer transitions. After HALT, any remaining iterations stutter and are excluded from published traces. The numeric audit compares coefficients to the original archive and runs the bounded interpreter against the original executor and scalar oracle for the 25 edge traces, including `0xfffffffe + 3`. The inherited affine bound is 148,176,371,709, below the exact-integer limit of binary64. This bound applies to the admitted integer states, not to arbitrary tensors or dtypes.
- The saved `.ir` files contain the inlined operators and graph loops. Artifact construction checks all nested blocks for Python callbacks and retained module calls. The controller and interpreter remain identifiable subcomputations within the single forward pass.

## Replay

The prompt and first tool execution are identical in both histories. The first tool returns 10. The intervention changes only the value reported to the model at event 2.

| Case | First feedback | Second request | Outcome |
| --- | --- | --- | --- |
| Normal | 10 | beta(10, 5) | 20 |
| Substituted | 14 | gamma(14, 5) | 9 |
| Exhausted | exhausted after two cycles | — | Model stops |
| Invalid repeat | 14 | beta(14, 0) | Graph rejects before execution, model stops |

Run from `visuaml-client/voxel-replay` with Python and PyTorch installed.

```sh
PYTHONPATH=.:../backend python tools/audit_attention_numeric.py
PYTHONPATH=.:../backend python tools/replay_composed_inference.py --open evidence/composed-inference/replay.json
npm ci
npm run dev
```

Open `/replay-viewer.html` and select **Open tool choice intervention**. The viewer displays saved evidence. The Python command reruns the frozen artifacts and verifies the downloaded bundle.

## Timing

`timings.json` records CPU single-thread inference samples in nanoseconds. The composed timer includes graph-side trace collection and excludes model load, input admission, and host JSON rendering. In the final five repeated samples after the initial two calls for each case, normal and substituted medians were approximately 7.7 and 7.8 ms. The first normal call took 31.2 ms, and the next took 24.8 ms. The original two-graph reference records its controller decode and tool validation/execution/trace phases separately. For the three warm valid normal reference rows, summed phase times were about 8.0–8.2 ms. The composed evidence does not instrument individual circuit times within the graph, and these samples are not a speedup benchmark.
