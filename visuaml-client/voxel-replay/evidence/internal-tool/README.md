# A learned router and an exact internal tool

A small learned attention controller chooses passthrough, call with operand order as given, or call with reversed operands. It was trained on six synthetic routing labels. It received no arithmetic labels or tool program. The exact executor is frozen recurrent hard attention with a live PC, two registers and four mutable indexed words. It selects instructions and operands using exact positional scores and commits typed memory and register updates. An in-graph conditional skips the executor for passthrough or disabled cases. The caller supplies token IDs, operand values and the program image to one TorchScript forward.

The archived circuit and controller were frozen before the held-out loop was written. The loop loads a word, adds 3 with i32 wraparound, stores it, decrements a counter, branches and halts. Nine synthetic cases verify learned routing, including six exact tool calls with independent trace checks. Ordered stores and completed outputs agree with an independently assembled WASM mirror executed by the frozen VisuaML graph and Wasmtime. Per-step comparisons use a logical instruction oracle, since the mirror uses several WASM instructions per logical operation. With the circuit disabled, the call returns the passthrough input. Adding 9 to the tool result changes each tool-dependent answer by 9 and leaves passthrough unchanged.

This is a structured-input controller on a small recurrent hard-attention machine. It uses explicit writable memory and a fixed 48-cycle allowance. The hard attention scans bounded slots and has no long-history scaling measurement. The learned interface selects positionally typed operands. It does not test language-model reasoning or free-form tool selection.

```sh
cd visuaml-client/voxel-replay
export PYTHONPATH=.:../backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python tools/replay_internal_tool.py
```

[Held-out program and freeze hash](heldout.json) · [Frozen artifact](internal-tool.pt) · [Trace and interventions](replay.json) · [WASM mirror source](reference.wat)
