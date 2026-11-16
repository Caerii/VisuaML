# VisuaML v24 — Verified shared model weights and cheap KV-isolated sessions

## Exact capability

V23 shares the traced full-decoder TorchScript executable across distinct
conversations, but on each new_session call it still canonicalizes the
checkpoint, clones weights repeatedly, and constructs a separate weight-bearing
Torch module. This makes additional sessions unnecessarily expensive.

V24 introduces SharedWeightsLlamaTemplate in
visuaml/compiler_lab/shared_weights_v24.py. The trusted factory:

1. Independently validates and freezes the source-bound Llama SemanticProgram
   before admitting any compiled executable.
2. Traces a single polymorphic CPU TorchScript decoder and compares its logits
   and *every* per-layer KV cache to the independent NumPy semantic oracle.
3. Retains the verified TorchScript executable, weight-bearing Torch module,
   canonical frozen plan and lower-level resident KV allocation contract.
4. For additional sessions, allocates only a fresh bounded KVSession, initializes
   new counters/transaction state, and reuses the exact graph and Torch module.
   It does not redo model checkpoint verification, freeze/reallocate model
   tensors, or rebuild the weight-bearing Torch module on the hot path.
5. Preserves the established v17 atomic cursor publication and snapshot/
   restore semantics. Every client receives private KV and cursor state.

The public benchmark preserves v23 as a control rather than replacing it;
reduced overhead is attributable to model-weight/module reuse, not a changed
checkpoint or different number of graph traces.

This is **process-local** and does not provide arbitrary code isolation.
Python code in the same interpreter could still introspect or mutate Torch
objects. The runtime supports trusted callers and strictly admitted,
hash-bound local checkpoints, not multi-tenant hostile Python programs.

## Reproducible tests

    PYTHONPATH=visuaml-client/backend python -m pytest -q \
      visuaml-client/backend/tests/test_shared_weights_v24.py \
      visuaml-client/backend/tests/test_shared_weights_bench_v24.py

The tests cover 1-, 2-, and 4-layer local Llama-shaped checkpoints, batches
one and two, multi-client graph/module identity, disjoint KV storage, unrelated
conversation histories, reset, caller-side checkpoint mutation, plan forgery,
concurrent session creation and numerical equality with an independent
NumPy reference. A monkeypatch test verifies that the fast path does not call
the Llama checkpoint lowerer or resnapshot the model.

## Real pretrained-model evidence

The pinned trained checkpoint is manojredhat/tiny-llama at immutable revision
756e9746eb801752d69771ef3e9bc433b21d853d and safetensors digest
d910edc7031721de23a7af111e355e6c09d5f1b3a2c43c0b6b0aa6a9135ffe03.

The independently completed GitHub Actions pretrained-model verification
run 43919811114 exercised four separate clients, three matched CPU trials,
full logits parity and disjoint KV cursors. Results:

| Measurement | v23 graph sharing | v24 graph + module sharing |
| --- | ---: | ---: |
| Graphs traced per trial | 1 | 1 |
| Unique weight-bearing Torch modules | 4 | 1 |
| Logical Torch module tensor bytes | 100,526,592 | 25,131,648 |
| Logical private KV tensor bytes | 7,077,888 | 7,077,888 |
| Median total cold session wall time | 531.908 ms | 255.624 ms |
| Median additional-session factory time | 242.939 ms | 11.168 ms |
| Median initial trace time | 46.076 ms | 48.850 ms |

The cold-start total is about 2.08x faster for v24. The aggregate factory
work for the three additional clients is about 21.75x smaller. Logical
model-module tensor bytes are one-quarter of the control while total KV
storage is unchanged and remains per-client private.

The source-bound JSON record, numerical witnesses, model/checkpoint hashes,
runtime metadata and per-trial graph/weight-module counts are retained at
benchmarks/pretrained_llama_shared_weights_v24.json.

The numerical and performance report can be reproduced locally:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      PYTHONPATH=visuaml-client/backend \
      python visuaml-client/backend/examples/shared_weights_bench_v24.py \
      /path/to/pinned/model --revision YOUR_40_HEX_REVISION \
      --clients 4 --trials 3 --prompt-len 4

## Important limitations

- Torch module tensor bytes are logical tensor storage, not physical peak RSS;
  the separately traced TorchScript graph may retain additional buffers.
- Cold timings are short single-process CI CPU microbenchmarks. No deployed
  throughput/latency service-level objectives, multi-process cache,
  GPU/optical hardware or power measurements have been demonstrated.
- The fast path requires a *previously verified* trusted template. It does
  not accept arbitrary model files or Python operators on each new session.
- The shared compiled executable is not written as a portable native AOT
  artifact; TorchScript CPU is still the execution target.
- V24 preserves immutable weight behavior by API contract, not OS-level
  process isolation against malicious or introspective Python callers.
- Each session still allocates private resident KV storage. There is no
  memory-paged KV allocator, prefill batching, eviction policy, or physical
  memory quota across sessions.

## Next technically important gates

1. Introduce a bounded, checkpoint-identity-keyed template registry with
   concurrency-safe acquisition and explicit logical memory budgets.
2. Measure actual process RSS and native memory under multiple sessions and
   contexts before claiming physical memory improvements.
3. Build a lifetime-checked KV page allocator or preallocated session pool
   without losing atomic rollback, alias isolation and cancellation safety.
4. Produce independently loadable native AOT artifacts with provenance checks,
   and compare against competent CPU/GPU inference engines at scale.
5. Introduce separate prefill and decode scheduling, precision variants,
   and realistic throughput workloads before extrapolating speedups.
