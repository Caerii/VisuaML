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

A multi-token call can reuse each weight vector across **two token rows** and four output channels through independently accumulated AVX2 lanes. This changes floating-point reduction order, so the existing finite-tolerance NumPy and scalar references remain mandatory; no cross-token dependencies are introduced. The reference implementation remains available in the original scalar and runtime-guarded AVX2 modes.

The conservative research dispatch is only eligible when **the current input chunk contains at least four tokens**, the model hidden width is at least 128, the projection input width is at least 32, and the runtime CPU supports AVX2. The user must additionally opt in. This threshold is provisional and backed by a limited geometry sweep, not a universal hardware rule. It is opt-in through `NativeCpuLlamaCompiler(plan, enable_batched_prefill=True, enable_avx2=True)`; guarded AVX2 must be explicitly requested and `reference_kernels=True` cannot be combined with the experimental prefill flag. Single-token decode falls through to the previous native `linear` kernel. When AVX2 is not supported by the machine, existing runtime dispatch falls back to scalar/four-row code.

The projection layout is reorganized into a uniform row-major batch for Q, K, V, attention output, gated feed-forward projections, down projection, and final vocabulary projection. RMSNorm, rotary embedding, the softmax attention loop, residual updates and transactional KV publication retain their existing mathematical and state contracts. The scratch storage is now sized for the admitted per-invocation row count; no mutable scratch is shared across sessions. This is **not** a claim of lower peak RSS, and the temporary geometry can increase scratch footprint.

## First negative result and design correction

The initial scalar cross-token loop passed numerical verification but measured **0.132× and 0.128×** v29 AVX2 prefill speed ratios in two five-trial, shared-runner pinned-model witnesses (i.e. roughly 7.6–7.8 times slower). This result ruled out promoting that implementation. The code has now been replaced by a **CPU-feature-guarded, two-token/four-output-channel AVX2 prefill microkernel**, with fallback to the v29 verified path when the AVX2 preconditions are not met. The first [replacement-kernel pinned-model run](https://github.com/Caerii/VisuaML/actions/runs/44132428720) measured **0.822×** relative prefill latency on one GitHub CI runner, while decode was approximately neutral (**1.020×**). The replacement is substantially better than the rejected scalar loop, but it **still loses on this small model**. It must remain opt-in; no positive speedup is claimed. The original negative result is preserved as part of the experiment history.

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

## Shape-dependent crossover experiment

The separate [v30 geometry sweep](../visuaml-client/backend/scripts/sweep_native_prefill_v30.py) uses deterministic **synthetic** Llama weights with independent NumPy parity and records raw prefill/decode latencies across five bounded configurations, including a vocabulary-heavy case. Its [GitHub workflow](../.github/workflows/v30-prefill-sweep.yml) attaches full source/compilation evidence and observed ratios. This is not a substitute for pinned **trained-model** verification; both experiments must be considered before choosing a dispatch policy.

An opt-in batched microkernel should only be promoted for verified geometry classes where it wins, with a fallback to v29 AVX2 everywhere else. Unsupported x86 variants or CPUs without AVX2 must never execute AVX2 instructions. Any crossover threshold must be grounded in measurements rather than marketing optimism.

## Results and conservative shape gate

The [initial five-case shape sweep](https://github.com/Caerii/VisuaML/actions/runs/44133097934) found the first indexed-accumulator batched SIMD implementation slower on every shape, with v29/batched ratios ranging from **0.53× to 0.79×**. A second, [explicitly unrolled AVX2 study](https://github.com/Caerii/VisuaML/actions/runs/44133451387) improved the experimental kernel: **0.95× small**, **1.04× medium**, **1.19× wide**, **1.30× wide long prefill** and **1.21× vocabulary-heavy**, in each case on one shared CI runner with five trials. These observations motivate a conservative opt-in **query ≥ 4 and hidden ≥ 128** dispatch gate; they do not prove the heuristic is optimal or portable. Small/marginal cases continue using the verified v29 path.

The C++ library exports a bounded query/geometry eligibility check. The Python `NativeCpuLlamaCompiler.prefill_dispatch(query_tokens)` reports the actual native gate, including whether an eligible batched path was selected. The benchmark records these decisions beside source hashes, numerical evidence and latency samples, making it possible to distinguish a real batched execution from a silent fallback. Single-token decode always uses v29.

## Research acceptance gates

1. All existing native C++ scalar, four-row, and AVX2 tests remain green, including the pinned real pretrained model.
2. New tests cover batch sizes 1/2, odd projection widths, prefill and decode continuation, independent NumPy logits and every K/V layer, and explicit rejection of conflicting execution flags.
3. The benchmark separates prefill and decode with source-bound evidence and retains all individual timings. The path remains opt-in if slower on the tested shapes.
4. A promotion decision requires repeated paired comparisons over more than one model geometry, token length and hardware configuration. A narrow result is labeled accordingly; no universal performance or speedup claim.
5. Representative longer contexts, meaningful memory profiling, multiple independent model families, GPU devices and physical photonic hardware remain **unverified**.

## Next candidate

Broader prefill speedups require a more general batched matrix/matrix microkernel and dedicated-hardware replication. The present two-token/four-output AVX2 kernel remains experimental, with explicit fallback when its shape gate is not satisfied. That next optimization should be justified by v30 prefill measurements; it must not silently replace a faster AVX2 decode kernel on short chunks.
