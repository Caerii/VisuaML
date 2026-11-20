# Synthesized recurrent attention executor

This experiment lowers a declared affine/ReLU transition graph into fixed weights. A stage scheduler groups independent ReLU nodes by dependency depth. It emits the affine matrix and bias for each stage and a final projection back to the recurrent state. The data-only weight archive exposes every synthesized coefficient. The retained build record lists operators in the frozen TorchScript graph.

Instruction fetch, register read, and memory read each use a fixed hard-attention head. For an address `a` and position `i`, the query and key projections produce a dot product `2ia − i²`. The unique maximum is the requested position for an admitted integer address. The hardmax selection is projected over value rows. The FFNs compute next PC, halt, register updates, and memory updates. The small memory is four explicit 32-bit word values carried in the recurrent state. Stores select the next value of each word. There is no write history or KV-cache growth in this design.

The host validates and loads the program and calls the frozen forward pass. TorchScript drives execution through a fixed 48-cycle loop with halted state held stable. The controller uses an in-graph conditional to skip this loop when disabled or on passthrough. This is a recurrent hard-attention and ReLU architecture, with hardmax and mutable recurrent state specified as runtime primitives.

## Checks

The original +3 loop was already pinned in the internal-tool evidence. The first post-freeze attempt exposed a failure at PC 1 on `0xfffffffe + 3`. The first lowering multiplied the two large ReLU terms of a comparison by `2^32` before subtracting them. The binary64 intermediate exceeded exact integer precision. The corrected compiler materializes the difference as a bounded 0-or-1 activation before scaling. The failing observation and original artifact hash are retained.

The corrected artifact was then frozen before the final holdout was authored. That program copies input to word 2, adds `INT32_MAX` in a data-dependent loop, and copies the final result back. The same frozen executor runs the original loop, the failed attempt's program, and the new holdout. Nine executions match the independent scalar interpreter at every logical boundary, including PC, live status, registers and all four memory words. Thirty-six additional i32 cases cover signed immediates, wraparound and borrow boundaries. An out-of-range store is rejected during admission. The test harness changes one observed store value and finds the first mismatch after logical PC 1.

For each loop, a separate WASM lowering matches return and full page between Wasmtime and the frozen VisuaML source graph. VisuaML also matches the attention executor's ordered stores. The logical programs and their WASM mirrors have different instruction boundaries. The scalar interpreter supplies the per-logical-step oracle.

The learned router's six supervision labels describe routing only. It selects passthrough, direct tool operands, or reversed tool operands without training on the holdout program or its arithmetic. Both tool routes selected `[2, 4]` and returned `4294967294` on the final holdout. Disabling the circuit gave passthrough answers of 2 and 4 respectively. Adding 9 at the tool-result interface shifted both tool answers by 9, and the passthrough case was unchanged.

## Benchmark contract

The pinned original loop uses input word 2 and four iterations. The local benchmark records 48 alternating warm samples, cold loads, and image preparation on one CPU thread. The integrated attention path materializes its fixed-cycle trace. Python records logical states. VisuaML times its ordinary source-step graph, and Wasmtime includes writing the initial page and a warm guest call. The upstream Percepta reproduction has the same arithmetic inputs and final numeric result. It emits a byte using its own token protocol. Its generation timings use a separate trace and preparation boundary.

The median was 2.90 ms for the integrated attention path, 0.010 ms for the Python logical trace, 16.41 ms for the VisuaML source graph, and 0.102 ms for Wasmtime under those boundaries. Building the affine and projection tensors took 2.83 ms in the build run. TorchScript artifact loading and model scheduling are tracked separately from warmed execution. Wasmtime had lower measured latency in this workload. Percepta uses a separate token-trace benchmark.

## Reproduce

From `visuaml-client/voxel-replay` with the existing research environment:

```sh
export PYTHONPATH=.:../backend OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python tools/replay_synthesized_attention.py
python tools/bench_internal_tool.py --synthesized
```

The replay opens the shipped frozen artifacts. The synthesis script rebuilds the typed transition block and writes projection weights. Use `--write` on the replay script to train and export the combined controller.

[Weights](weights.json) · [Frozen executor](executor.pt) · [Holdout programs](programs.json) · [Exact replay](replay.json) · [Timing samples](benchmark.json) · [Percepta reproduction](../percepta-reproduction)
