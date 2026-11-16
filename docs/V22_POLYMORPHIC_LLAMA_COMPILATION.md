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

The fixed physical KV input shape supports tracing without per-position
specialization. Inside the compiled graph, only the **logical committed
prefix** is gathered with a dynamic tensor-valued narrow operation, then
concatenated with the current tokens' K/V. Attention sees only the valid
prefix, not the unused reserved capacity.

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
One reusable graph reduces compilation work but does not necessarily reduce
steady-state latency, because fixed-capacity input copies can be expensive.

## Next acceptance gates

1. Design an alias-safe, owned storage interface that avoids copying full
   reserved KV capacity into Torch at every decoding step. Verify rollback
   and adversarial failure preservation before pursuing zero-copy writes.
2. Measure prefill/decode separately across longer realistic contexts,
   compilation costs, physical memory usage, throughput and p50/p95 tails.
3. Lower the same typed graph into a second independently deployable
   AOT execution backend, with signed/hardcoded artifact provenance checks.
4. Introduce measured cost-driven dispatch between dedicated prefill,
   shape-specialized decode and polymorphic decode backends.
5. Generalize the typed operator and state contracts to new model families
   only after independent real-checkpoint numerical verification.
