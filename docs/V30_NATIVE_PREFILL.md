# v30 — Bounded batched native CPU prefill

*The test is not whether a new kernel looks faster. The test is whether it preserves a known computation while improving a measured workload.*

v29 established an independent scalar C++17 Llama decoder, a four-row matrix-vector kernel, a CPU-dispatched AVX2 variant, pinned trained-model parity, and reproducible short-workload timing. These remain small, explicitly admitted Llama programs—not general architecture coverage.

v30 adds **opt-in weight-row reuse across token rows** during multi-token prefill. It is an experiment on the existing native execution path, not a new general IR-to-native code generator.

## Semantics and execution scope

For the row-major weight tensor W and a token-row matrix X, each output is

```text
Y[token, output] = sum_{input=0..width-1}
    W[output, input] * X[token, input]
```

A multi-token call can reuse the same weight row across up to four token rows. The v30 loop keeps a **separate ordered accumulator per token/output pair**, reusing weight loads without introducing cross-token dependencies. The reference implementation remains available in the original scalar and runtime-guarded AVX2 modes.

The optimization is only eligible when **the current input chunk has more than one token** and input width is at least 16. It is opt-in through `NativeCpuLlamaCompiler(plan, enable_batched_prefill=True, enable_avx2=True)`; `reference_kernels=True` cannot be combined with the experimental prefill flag. Single-token decode falls through to the previous native `linear` kernel. When AVX2 is not supported by the machine, existing runtime dispatch falls back to scalar/four-row code.

The projection layout is reorganized into a uniform row-major batch for Q, K, V, attention output, gated feed-forward projections, down projection, and final vocabulary projection. RMSNorm, rotary embedding, the softmax attention loop, residual updates and transactional KV publication retain their existing mathematical and state contracts. The scratch storage is now sized for the admitted per-invocation row count; no mutable scratch is shared across sessions. This is **not** a claim of lower peak RSS, and the temporary geometry can increase scratch footprint.

## Evidence and verification

Run Python and native regression tests with the pinned Python environment and a trusted local C++17 compiler:

```sh
cd visuaml-client
python -m pytest -q backend/tests/test_native_cpu_v28.py backend/tests/test_native_cpu_prefill_bench_v30.py
python -m pytest -q backend/tests
```

On a trusted, local compatible checkpoint:

```sh
PYTHONPATH=backend python backend/examples/native_cpu_prefill_bench_v30.py /path/to/checkpoint --revision PINNED_40_CHAR_SHA --prompt 8 --decode 1 1 --trials 21 --output /tmp/visuaml-v30-prefill.json
```

The benchmark instantiates separate scalar, AVX2, and AVX2-plus-batched compiler binaries, checks every chunk of logits and all final K/V state against NumPy **before timing**, then reports separately: warm prefill, warm decode, and warm end-to-end single-request latency. It retains individual timing samples and source, compiler, checkpoint and binary hashes. The arithmetic work is bounded by the existing admitted model; no arbitrary uploaded native code is compiled.

The pretrained [native parity workflow](../.github/workflows/native-cpu-v28.yml) now includes one pinned v30 prefill/decode benchmark in its JSON witness. The same CI pipeline still records the v28-to-v29 paired baseline for comparison. Source changes should be compared on the **same physical CPU** and with the **same checkpoint and workload**; separate cloud runners are not a matched comparison.

## Research acceptance gates

1. All existing native C++ scalar, four-row, and AVX2 tests remain green, including the pinned real pretrained model.
2. New tests cover batch sizes 1/2, odd projection widths, prefill and decode continuation, independent NumPy logits and every K/V layer, and explicit rejection of conflicting execution flags.
3. The benchmark separates prefill and decode with source-bound evidence and retains all individual timings. The path remains opt-in if slower on the tested shapes.
4. A promotion decision requires repeated paired comparisons over more than one model geometry, token length and hardware configuration. A narrow result is labeled accordingly; no universal performance or speedup claim.
5. Representative longer contexts, meaningful memory profiling, multiple independent model families, GPU devices and physical photonic hardware remain **unverified**.

## Next candidate

A separate AVX2 batched matrix-vector/matrix-matrix microkernel may improve multi-token weight reuse while maintaining CPU-dispatch safety. That next optimization should be justified by v30 prefill measurements; it must not silently replace a faster AVX2 decode kernel on short chunks.
