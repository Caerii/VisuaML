# v31 research: bounded online attention and context-length evidence

*The optimization is not accepted merely because it compiles or avoids one buffer.*

## Context and scope

The merged v30 CPU decoder runs a **bounded, admitted Llama-specific program** with independently checked NumPy and Hugging Face witnesses. Its reference attention uses three conceptual passes: calculate causal grouped-query scores, compute stable softmax probabilities, then apply the resulting probabilities to values. That remains the **default**.

The v31 research branch introduces an **opt-in** online softmax/value accumulator. For each query and head, it maintains the running score maximum `m`, scaled denominator `z`, and weighted value vector `a`. For a new score `s` and value `v`:

```text
m_next = max(m, s)
scale = exp(m - m_next)      # zero on the first element
weight = exp(s - m_next)
z_next = z * scale + weight
a_next = a * scale + weight * v
output = a / z              # only after all causal positions
```

The C++ code returns an error if scores or the normalization denominator become nonfinite or invalid. The Python session commits only fully verified KV suffixes. Each attention head owns its intermediate state within a call; no mutable online state is shared across sessions. The online path does not allocate the C++ `scores(capacity)` vector, though this does **not** establish a meaningful physical-RSS reduction: execution includes weights, Python, and other temporaries, and the old buffer is bounded to at most 128 floats.

Enable the variant only when explicitly requested:

```python
native = NativeCpuLlamaCompiler(
    plan,
    enable_avx2=True,
    enable_online_attention=True,
)
print(native.report()["attention_kernel"])  # "online_stable_softmax"
```

The default `NativeCpuLlamaCompiler(plan)` remains `materialized_scores`. Floating-point reduction order differs from the reference; all correctness claims are finite-tolerance witnesses, **not** a proof of all-input equivalence.

## Reproduction

From `visuaml-client` with the pinned Python dependencies and an installed C++17 compiler:

```sh
python -m pytest -q backend/tests/test_native_online_attention_v31.py backend/tests/test_native_online_attention_bench_v31.py
python -m pytest -q backend/tests
PYTHONPATH=backend python backend/examples/native_online_attention_bench_v31.py /path/to/local/checkpoint --revision YOUR_PINNED_REVISION --past 4 8 16 32 48 --trials 21 --output /tmp/online-attention-v31.json
```

For each context prefix, the benchmark runs the same token sequence and canonical semantic plan through **two different C++ binaries** and the NumPy oracle. It compares all prefill outputs, single-token continuation logits and final KV state **before timing**. The resulting record contains separate warm-prefix and warm-single-decode samples, p95/median/MAD, model/semantic/binary/compiler hashes and platform metadata. It alternates engine order between trials.

The context sweep is limited to **at most 64 tokens** to remain within the existing admitted fixed model contract. It does not simulate or establish the behavior of 4k, 32k or million-token serving workloads. Avoid reporting it as long-context performance.

The independent [trained-model workflow](../.github/workflows/native-cpu-v28.yml) also includes this experiment with a pinned safetensors checkpoint and a shorter three-trial witness. GitHub shared runners are useful for detecting failures and preliminary regressions, not for estimating dedicated-machine throughput or energy efficiency.

## Scientific acceptance criteria

1. Compiler and full tests green, including real pretrained-model Hugging Face/NumPy reference logits and transactional KV state.
2. Tests include multiple layers, grouped-query attention, batch 1/2, mixed chunk boundaries, unusual tensor dimensions, near-capacity context, reset/restore, and invalid flags.
3. Explicit baseline/online source and executable hashes and raw warm-timing samples survive in the CI artifact.
4. If online attention is slower on the measured geometry, **retain it as an experiment**; do not switch the default or change roadmap acceptance percentages.
5. If faster, rerun on dedicated stable CPU hardware and further model shapes before promoting any broader performance claim.
6. Physical memory, native allocator counts, true long-context performance, additional pretrained model families and real GPU/photonic hardware remain distinct unverified obligations.

## Next meaningful work

Expand the semantic contract beyond the current 128-token C++ admission ceiling *only* with bounded allocation reasoning, new edge-case witnesses and verified context-extension semantics. Measure stage-specific attention costs and native allocation lifetimes on trusted hardware. A general IR-to-native compiler and independent second pretrained architecture are separate milestones, not implied by this isolated optimization.
