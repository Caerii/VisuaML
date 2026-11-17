# v29 — Evidence-driven native CPU performance

The v28 C++17 decoder is an **independent numerical witness**, not a competitive inference engine. v29 preserves that witness while investigating narrowly scoped performance transformations. The first experiment reuses rotary position embedding (RoPE) sine/cosine values across Q and K heads for each token position instead of recomputing `pow`, `sin`, and `cos` for every head.

## First experiment: amortized RoPE angles

For a single token position, the RoPE angle depends on position, head dimension, and theta; it does **not** depend on head identity or the tensor element's Q/K source. The native decoder now computes one table for that position and shares it across Q and K heads. The mathematical order of rotation within each head is unchanged. This does **not** justify a performance claim until the same trained checkpoint, workloads, and correctness gates have been rerun.

## Reproduce and admit

From `visuaml-client`, with the pinned Python environment and a C++17 compiler installed:

```sh
python -m pytest -q backend/tests/test_native_cpu_v28.py backend/tests/test_native_cpu_bench_v28.py
python -m pytest -q backend/tests
```

Run the trained model witness described in [Verification](VERIFICATION.md), then rerun the matched benchmark with a pinned local checkpoint:

```sh
PYTHONPATH=backend python backend/examples/native_cpu_bench_v28.py /path/to/pinned/checkpoint --revision YOUR_PINNED_COMMIT_SHA --prompt 4 --chunks 1 1 2 --trials 21 --output /tmp/visuaml-v29-rope.json
```

The benchmark checks chunked logits and final K/V state against NumPy before recording warm latency. The v29 branch also records the individual warm-run samples in `raw_samples_ms` and the alternating measurement order; compare raw distributions rather than relying on a single median. Compare against the v28 baseline using the **same CPU, thread settings, checkpoint hash, input sequence, trials, and compiler flags**. Do not compare across heterogeneous CI machines as evidence of a speedup. Report median and p95 separately; record the compilation time separately from warm inference. Warm-up and interleaved order reduce but do not eliminate benchmarking biases.

## Acceptance conditions

1. C++17 compilation succeeds on the supported CI toolchain.
2. Native logits and all final K/V states agree with the independent NumPy oracle within the established tolerances, including multiple query lengths, cursor positions, and grouped-query attention configurations.
3. The pinned pretrained Llama workflow remains green.
4. Matched-device before/after performance has raw samples and hardware/compiler metadata. A speedup is reported only if supported by those measurements; a regression reverts the optimization.
5. No roadmap acceptance status is changed merely because the code exists.


## v29 implementation progress: linear and scratch work

The native C++17 decoder now includes a default four-output-row linear kernel (single input traversal with independent per-row accumulators) and a separately compiled `VISUAML_NATIVE_SCALAR_REFERENCE` variant. The latter preserves the conventional one-output-row loop so the implementation can compare actual native binaries, rather than only native execution against the NumPy oracle. These two variants share RoPE caching and scratch reuse; therefore their comparison isolates the changed linear traversal more closely than a v28-to-v29 comparison.

Per-layer tensors and attention-score scratch are now allocated once per decoder invocation and reused across layers/heads. The attention output is explicitly cleared each layer; scores are fully written over the active prefix. This reduces repeated vector-allocation calls but does **not** prove lower physical peak RSS. Scratch remains invocation-owned, not shared mutable state across sessions.

Run the comparison with:

```sh
PYTHONPATH=backend python backend/examples/native_cpu_bench_v28.py /path/to/pinned/checkpoint --revision YOUR_PINNED_COMMIT_SHA --prompt 4 --chunks 1 1 2 --trials 21 --compare-native-reference --output /tmp/visuaml-v29-compare.json
```

The [native parity GitHub Actions workflow](../.github/workflows/native-cpu-v28.yml) now includes this comparison on the pinned real pretrained checkpoint, and uploads its full JSON result as a 30-day `native-cpu-pretrained-witness` artifact. A CI runner is not stable matched hardware for longitudinal performance claims; do not compare its ratios against unrelated machines.

The output includes `native_scalar_reference` timing samples, `native_linear_variants`, separate native compilation times, and `warm_native_speedup_vs_scalar` calculated as median scalar time divided by median four-row time. It is a *ratio*, not an established speedup claim unless reproduced with controlled hardware and enough trials. A combined v28-to-v29 measurement additionally captures RoPE and allocation effects.

**Additional safeguards:** compiler flags use `-O2 -fno-fast-math`; optimized and scalar streams must match independent NumPy logits and all KV states. The existing native pretrained witness remains mandatory. Numerical agreement for tested models is not a proof of all possible inputs. Future physical-memory experiments must distinguish RSS/peak high-water measurements from logical live KV budgets.

## Physical process memory probe (Linux)

The independent `examples/native_cpu_memory_v29.py` probe executes the scalar-reference, four-row, and AVX2-requested variants in a **fresh child process**, alternating the order across repetitions. It collects Linux `VmRSS` and `VmHWM` before model loading, after loading, after native compilation, and after an initial + streaming decode. These are observed physical resident-process statistics—not the sizes of Python logical KV leases.

```sh
PYTHONPATH=backend python backend/examples/native_cpu_memory_v29.py /path/to/pinned/checkpoint --revision YOUR_PINNED_COMMIT_SHA --repetitions 3 --output /tmp/visuaml-v29-memory.json
```

The JSON includes checkpoint and C++ source hashes, compiler identity, platform, and per-process samples. Compare distributions without assuming that differences are attributable solely to C++ scratch reuse: `VmHWM` is a process high-water measure including Python, checkpoint weights and loaded shared libraries; it excludes RSS used by the external C++ compiler subprocess. This first probe is not a statistically controlled heap profiler and is not available on Windows or macOS. No physical-memory reduction should be claimed until repeated measurements support it.

## Experimental AVX2 kernel and matched revision study

The v29 branch includes an **explicitly opt-in**, runtime-guarded AVX2 four-row matrix-vector kernel. `NativeCpuLlamaCompiler(plan, enable_avx2=True)` compiles AVX2 instructions only inside a target-specific function on supported GNU/Clang x86 processors; `__builtin_cpu_supports("avx2")` guards execution. Unsupported CPUs retain the four-row fallback, and input widths below 32 also use the fallback. No global `-mavx2` is added. The returned compiler evidence identifies the **requested** and **effective** implementation, native binary hash and compiler flags.

```sh
PYTHONPATH=backend python backend/examples/native_cpu_bench_v28.py /path/to/pinned/checkpoint --revision YOUR_PINNED_COMMIT_SHA --prompt 4 --chunks 1 1 2 --trials 21 --compare-native-reference --compare-simd --output /tmp/visuaml-v29-all-kernels.json
```

On [one seven-trial pinned pretrained CI run](https://github.com/Caerii/VisuaML/actions/runs/44124083185), median warm execution was 40.078 ms (native scalar reference), 14.988 ms (four-row), **6.528 ms (AVX2)**, 20.761 ms (NumPy semantic reference) and 14.973 ms (TorchScript whole graph). The median AVX2/four-row speed ratio was **2.296×** on that runner, and this is **not** a broad production throughput or cross-hardware claim. Both native variants preserve the documented numerical and KV witnesses. These are early CI observations, not a finished benchmark campaign.

The [paired revision runner](../visuaml-client/backend/scripts/compare_native_revisions_v29.py) measures the merged v28 C++ baseline and v29 variants against the **same pinned checkpoint** on one GitHub runner, alternating process order and publishing source-bound artifacts. It checks pinned safetensors/configuration hashes and semantic identity. The original v28 script records aggregate statistics only; the current branch records raw individual latency samples. Both compare outputs against the independent NumPy program. GitHub shared-runner contention, DVFS, thermals, process startup, and limited model dimensions remain important limitations. Future dedicated-hardware experiments must test different shapes, large vocabularies, longer contexts, and energy/performance over time.

### v28-to-v29 matched-revision witness (November 17, 2026)

The [paired CI run](https://github.com/Caerii/VisuaML/actions/runs/44124932669) compiled the merged v28 commit `a2e2042` and the experimental v29 branch on the **same AMD EPYC 7763 runner**, with exact pinned checkpoint and semantic hashes, alternating process order, two pairs and seven trials per backend per pair. The two v28 native medians were **41.156 / 40.901 ms**; v29 four-row medians were **16.972 / 15.914 ms**; v29 AVX2 medians were **7.978 / 7.112 ms**. The median of the paired v28-to-v29 ratios was **2.497×** for four-row and **5.455×** for AVX2. The downloadable artifact retains per-pair source fingerprints, compiler identity, and current-branch individual latency samples.

This comparison adds a controlled revision-to-revision observation; it is **not** a multi-run, dedicated-machine, representative-model study, nor does it establish a production performance guarantee. The v28 benchmark did not retain individual raw timing samples, so the paired comparator explicitly preserves v28 aggregate statistics rather than fabricating its raw distribution. Repeating on a dedicated machine and wider shapes remains necessary.

The paired workflow additionally captures Linux whole-process RSS observations for all three v29 kernel variants. That evidence is a process-memory observation—not an allocator-isolated measurement or proof that scratch reuse lowers physical peak memory.

The native code remains architecture-specific and experimental. A fast finite witness does **not** establish semantic equivalence for every model input or competitive industry serving economics.

## Planned next kernels

Measure linear projection and vocabulary-head costs first. Evaluate safe contiguous vectorization and matrix multiplication blocking, then attention score/value accumulation and reusable scratch buffers. Keep scalar/reference and optimized paths separable for independent comparison. Investigate physically measured RSS/peak allocation and long-context behavior before claiming memory reductions. GPU and optical code generation remain separate future milestones.

No throughput or memory improvement is claimed by this document.
