# v36 — Independent pretrained numerical replay from a reconstructed source bundle

## Why this is scientifically consequential

The original v28 evidence established that VisuaML's independent C++17 decoder could reproduce Hugging Face and NumPy results for a **real, pinned Tiny Llama checkpoint**. v35 indexed that historical record and reported that replay was blocked because the currently merged sources were different.

v36 audited the original source manifest against actual Git history rather than simply assuming the v28 merge represented the running experiment. The audit found **no single Git commit containing the three recorded source files in their exact recorded versions**.

Each exact byte sequence does, however, exist in Git history:

| File relative to `visuaml-client/backend` | Original SHA-256 | Git commit containing exact bytes |
| --- | --- | --- |
| `examples/native_cpu_bench_v28.py` | `f4b8c517876dd18af7977a415e2f4c587ee8c02086a1e6e9b62eb3c2a7bea4d9` | `12c6070373b92f1a3628581e0d37094ea5cc82db` |
| `visuaml/compiler_lab/native_cpu_v28.cpp` | `408054895f5f017817022f2a7399980d8913fb86655fd43f15d81ab62cb20579` | `9afd8ed006ea40b2f1010c0909d510a1a89a476b` |
| `visuaml/compiler_lab/native_cpu_v28.py` | `d0836801810f42404f3c4549d8ea20cdc466acbccfee78e717ef388c4489c3be` | `ac8ad81bf22b6e8eef8e43f775e7006c2fffa39e` |

**This is a reconstructed source bundle, not an original Git checkout.** The v28 merge `a2e2042cca69028bce528bef86246ec7f387bd5b` serves only as a pinned base for other supporting Python modules and the original reference probe. The three source files are overlaid with their independently hash-verified blobs before execution. The repository does **not** assert which exact composite tree or commit the old CI runner used.

## The two-stage reproducibility contract

The dedicated [v36 workflow](../.github/workflows/pretrained-replay-v36.yml) performs two distinct stages:

**Stage 1 — source recovery.** With only Python's standard library and Git, identify all matching historical source file versions; refuse an unapproved Git source object, changed SHA-256, missing history, mismatched archive/source contract or unsupported path. Reconstruct a canonical bundle manifest with distinct file version commits, source digests, original archive hash and pinned supporting probe identity. Run adversarial fast admission tests.

**Stage 2 — fresh numerical execution.** Only after stage 1 passes, create a detached worktree at the pinned v28 base, overlay the three original exact source blobs, and execute the historical reference probe. Dependencies are pinned to the original CPU/PyTorch/Transformers/Safetensors versions. Only `config.json` and `model.safetensors` may be retrieved from the hardcoded, 40-character model revision. The code verifies the original configuration and safetensors hashes before numerical execution; `trust_remote_code=False` is used for the independent Hugging Face reference.

The freshly generated result must independently satisfy:

- HF PyTorch logits and NumPy semantic IR logits within the pinned `3e-4` tolerance.
- Full native KV-state agreement on streamed chunks, plus greedy-generation and final KV parity.
- The historical two token/witness shapes and original exact benchmark chunks/trial count.
- Correct checkpoint/configuration/semantic digests, C++ SHA-256 and all three Python/C++ source-file hashes.
- A generated native C++ executable fingerprint and compiler identity from this new run.
- Separate numerical and CPU benchmark results, with the complete fresh JSON archived as a new experimental observation.

The exact replay result is **not** compared byte-for-byte with the historical latency measurements or compiler binary: they can legitimately change across toolchain/runner environments. The C++ source and checkpoint are what are pinned; machine code is recorded as a *fresh* artifact.

## Commands

The source discovery and verification stage requires no ML framework:

```sh
python visuaml-client/backend/scripts/replay_pretrained_v36.py --discover-source
python visuaml-client/backend/scripts/replay_pretrained_v36.py --verify-source --output /tmp/recovery-v36.json
PYTHONPATH=visuaml-client/backend python -m pytest -q visuaml-client/backend/tests/test_replay_pretrained_v36.py
```

With the trusted pinned CPU environment and network access to the two exact checkpoint files:

```sh
python visuaml-client/backend/scripts/replay_pretrained_v36.py --run --output /tmp/replay-v36.json
```

This code only accepts the hardcoded historical source bundle and model revision. There is no untrusted Python model-upload path, arbitrary native compiler input, automatic production promotion or access to credentials. The repo's generated evidence files remain source-controlled, reviewed and checked for byte drift.

## Scientific limitations and next gate

- The successful first-stage source recovery does **not** establish fresh numerical replay success. The second-stage CI result and its source-bound artifact are required.
- The original v28 archival record remains immutable and separately indexable. A new successful v36 result is a **different** experiment on the reconstructed source bundle.
- Original v28 CI execution cannot be attributed to a Git commit solely by matching source file SHA-256 digests.
- The legacy historical benchmark records median/p95 summaries, not every raw per-trial sample. Consequently this stage prioritizes correct fresh numerical evidence over throughput claims.
- A finite numerical witness is not formal all-input equivalence or proof of correct behavior on every pretrained model.
- The pinned support environment is not an isolation boundary for arbitrary uploaded Python. We only run vetted checked-in code.
- New empirical evidence alone does not alter the public roadmap's approved weighted acceptance percentage.

Next, if CI produces a fresh passing witness, archive the complete result with GitHub run and source/checkpoint provenance, index it as a separate v36 node, then add direct evidence-navigation from the existing `/research/knowledge` and `/compiler` experiences. Broader architecture coverage still requires a genuinely different pinned pretrained model family, not another Tiny Llama run.
