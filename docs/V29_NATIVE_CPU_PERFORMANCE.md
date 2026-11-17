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

The output includes `native_scalar_reference` timing samples, `native_linear_variants`, separate native compilation times, and `warm_native_speedup_vs_scalar` calculated as median scalar time divided by median four-row time. It is a *ratio*, not an established speedup claim unless reproduced with controlled hardware and enough trials. A combined v28-to-v29 measurement additionally captures RoPE and allocation effects.

**Additional safeguards:** compiler flags use `-O2 -fno-fast-math`; optimized and scalar streams must match independent NumPy logits and all KV states. The existing native pretrained witness remains mandatory. Numerical agreement for tested models is not a proof of all possible inputs. Future physical-memory experiments must distinguish RSS/peak high-water measurements from logical live KV budgets.

## Physical process memory probe (Linux)

The independent `examples/native_cpu_memory_v29.py` probe executes each native linear variant in a **fresh child process**, alternating the order across repetitions. It collects Linux `VmRSS` and `VmHWM` before model loading, after loading, after native compilation, and after an initial + streaming decode. These are observed physical resident-process statistics—not the sizes of Python logical KV leases.

```sh
PYTHONPATH=backend python backend/examples/native_cpu_memory_v29.py /path/to/pinned/checkpoint --revision YOUR_PINNED_COMMIT_SHA --repetitions 3 --output /tmp/visuaml-v29-memory.json
```

The JSON includes checkpoint and C++ source hashes, compiler identity, platform, and per-process samples. Compare distributions without assuming that differences are attributable solely to C++ scratch reuse: `VmHWM` is a process high-water measure including Python, checkpoint weights, shared libraries, and compiler activity. This first probe is not a statistically controlled heap profiler and is not available on Windows or macOS. No physical-memory reduction should be claimed until repeated measurements support it.

## Planned next kernels

Measure linear projection and vocabulary-head costs first. Evaluate safe contiguous vectorization and matrix multiplication blocking, then attention score/value accumulation and reusable scratch buffers. Keep scalar/reference and optimized paths separable for independent comparison. Investigate physically measured RSS/peak allocation and long-context behavior before claiming memory reductions. GPU and optical code generation remain separate future milestones.

No throughput or memory improvement is claimed by this document.
