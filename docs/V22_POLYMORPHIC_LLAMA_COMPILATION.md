# VisuaML v22 — One compiled Llama graph for variable query and KV cursor

## The compiler milestone

V21 compiled complete decoder mathematics into a CPU TorchScript graph, but it
specialized every distinct (batch, query length, previous KV length). V22
introduces the opt-in torchscript_poly backend, with **one trace per model and
batch across admissible query lengths and past-context positions**.

The graph executes the very same canonical v20 typed SemanticProgram and
strict local pretrained checkpoint contract, while a scalar tensor-valued
cursor determines RoPE positions, KV writes and causal attention at runtime.
Consequently the graph does not freeze token positions or cursor into the
trace constructed for the first input.

KV effects remain ordered and atomic under the v17 resident KV backend.
A failed graph execution cannot commit a partly written cache cursor.

## Logical prefix optimization

The resident cache retains fixed physical capacity, but the TorchScript
graph receives only a **detached copy of the live logical prefix**. A scalar
tensor cursor and shape-polymorphic tensor operations let one graph handle
varying prefix and request lengths. The compiled graph constructs the active
prefix plus the current tokens' K/V; attention never processes unused reserved
capacity. This avoids copying full-capacity input buffers.

The graph returns **only the newly generated K/V suffix**, not a copy of
the full physical cache for every decoder layer. Each suffix is checked
for shape, dtype and numerical finiteness before the resident session
publishes it. TorchScript receives detached CPU copies of existing cache
storage, so it cannot mutate committed NumPy resident memory in place.

A report includes logical KV tensor byte counts for graph inputs and
outputs. These are not measured physical peak RSS or device transfers.
The backend is CPU TorchScript tracing, **not** standalone native AOT,
true IREE/Triton lowering, or GPU/optical execution.

## Independent correctness gates

- 1-, 2-, 4-layer checkpoints, batches of 1 and 2, nonuniform decode
  chunks and one trace across changing token and cursor lengths.
- Numerical witnesses against independent NumPy and PyTorch references,
  including intermediate KV state of every layer.
- A full admitted 64-token context, reset/reuse, snapshot/restore,
  greedy generation, malformed token inputs and injected bad graph outputs.
- A pinned real TinyStories-trained Llama checkpoint in a dedicated
  GitHub Actions test; finite numerical parity does not imply formal
  equivalence on all inputs.

## Reproduce

    PYTHONPATH=visuaml-client/backend python -m pytest -q \
      visuaml-client/backend/tests/test_torchscript_poly_v22.py \
      visuaml-client/backend/tests/test_llama_polymorphic_bench_v22.py

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      PYTHONPATH=visuaml-client/backend \
      python visuaml-client/backend/examples/llama_polymorphic_bench_v22.py \
      /path/to/verified/local/llama --revision YOUR_40_HEX_REVISION \
      --chunks 4 1 1 1 3 2 --trials 25

Report total graph construction/tracing time and warm decoding separately.
One reusable graph reduces compilation work but does not automatically
reduce steady-state latency. The optimized backend now copies only live KV
prefixes into Torch, but per-step tensor allocation and graph-boundary
overhead remain; all performance claims require matched measured evidence.

## Next acceptance gates

1. Extend the current live-prefix copying interface to an explicit alias-safe,
   owned storage model if pursuing zero-copy or reusable allocations. Verify
   rollback and adversarial failure preservation before any in-place writes.
2. Measure prefill/decode separately across longer realistic contexts,
   compilation costs, physical memory usage, throughput and p50/p95 tails.
3. Lower the same typed graph into a second independently deployable
   AOT execution backend, with signed/hardcoded artifact provenance checks.
4. Introduce measured cost-driven dispatch between dedicated prefill,
   shape-specialized decode and polymorphic decode backends.
5. Generalize the typed operator and state contracts to new model families
   only after independent real-checkpoint numerical verification.


## Source-bound pinned real-model measurements

Pinned trained checkpoint: manojredhat/tiny-llama at revision
756e9746eb801752d69771ef3e9bc433b21d853d, with safetensors
SHA-256 d910edc7031721de23a7af111e355e6c09d5f1b3a2c43c0b6b0aa6a9135ffe03.

The CI experiment verifies parity of every emitted logit chunk and all final
KV state, then alternates backend order across five warm full sequences
of six chunks (4, 1, 1, 1, 3, 2 tokens). The following values come from
the *same* GitHub Actions CPU run 43912614045:

| Observed measurement | v21 specialized | v22 dynamic live-prefix |
| --- | ---: | ---: |
| Separate traced graphs | 6 | 1 |
| Cumulative tracing time, ms | 238.175 | 39.625 |
| Warm six-chunk median, ms | 11.777 | 11.833 |
| Cumulative input KV tensor bytes | 2,654,208 | 2,654,208 |
| Cumulative output KV tensor bytes | 3,649,536 | 995,328 |

The original NumPy semantic interpreter took a 18.510 ms warm median
for the same workload. V22 is approximately 6x cheaper to trace than
v21 but offers no established steady-state speedup over v21 in this
individual run. V22 produces about 73% fewer KV-output tensor bytes.
All numbers vary with the CI runner, BLAS settings, and timing noise.

Evidence: benchmarks/pretrained_llama_polymorphic_kv_v22.json
and the independently completed workflow:
https://github.com/Caerii/VisuaML/actions/runs/43912614045

These byte totals represent tensor sizes observed by the CPU runtime,
not physical peak RSS, memory-allocator behavior, or PCIe transfers.
Current sessions still require tracing once per newly constructed
session unless the compiled graph is explicitly reused.
