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

## Actual independent numerical replay (November 17, 2026)

The [dedicated real-checkpoint replay run](https://github.com/Caerii/VisuaML/actions/runs/44193012326) passed both stages: exact source-byte recovery and a new detached-worktree CPU/Hugging Face/NumPy numerical run. The new execution observed:

| Verification | Fresh measured result |
| --- | ---: |
| Max absolute native logits error against independent Hugging Face | `1.519918441772461e-6` |
| Max absolute native logits error against NumPy semantic IR | `1.1920928955078125e-6` |
| Independent pinned input witnesses | 2 |
| Chunked KV and greedy-generation parity | Passed |
| Legacy native C++17 warm CPU median | 35.429234 ms |
| Native compile time | 602.79612 ms |
| Native executable SHA-256 | `15f718f94c42921e7c67774e0ea3f8faea91cd96595a76060d2a04cc234e14ef` |

The native executable fingerprint also matches the old recorded v28 executable on the tested GNU C++ toolchain. This is a strong byte-level reproducibility observation, but it does **not** imply all-input equivalence, future cross-platform code identity, or production serving performance.

The complete, newly produced 7,897-byte [v36 JSON record](../benchmarks/pretrained_llama_independent_replay_v36.json) is archived alongside its [exact CI origin manifest](../benchmarks/pretrained_llama_independent_replay_v36.origin.json). The archive retains SHA-256 of the original downloaded CI bytes `a6b06210cc37a78ded00ab52c201c2c96eea0f4f754b427437b6ded3acce48f0` and the full structured numerical record digest `0e5f11a929e9a594f10233bc7e0027af0e86fc8ab2bc487ab13cab7c802da674`. The [archival validator](../visuaml-client/backend/scripts/archive_pretrained_replay_v36.py) checks both historical source fingerprints and this fresh record. It does not treat an unsigned SHA as an authenticated human signature.

### A new knowledge graph, without rewriting old evidence

The deterministic [v36 replay knowledge graph](../visuaml-client/public/evidence/knowledge-v36.json) overlays the existing v35 graph:

- **14 distinct historical/fresh observations** across five reviewed artifacts
- **21 exact identity relationships** including old-to-new Llama semantic, checkpoint, C++ and native binary identities
- The original 13 prior study IDs, old v35 static preflight plan, and reviewed roadmap criteria are unchanged
- One **new, independently executed** real-pretrained witness; still just **one distinct pinned pretrained model family**

The read-only `/research/knowledge` experience distinguishes the recovered new numerical result from the earlier historical CI record and preflight-only plan. Source availability does not mean this frontend route is deployed at visuaml.com.

## Scientific limitations and next gate

- The fresh numerical replay is a finite **new** execution of the exact reconstructed source bundle, not a formal proof or historical CI checkout recovery.
- The original v28 archival record remains immutable and separately indexable. A new successful v36 result is a **different** experiment on the reconstructed source bundle.
- Original v28 CI execution cannot be attributed to a Git commit solely by matching source file SHA-256 digests.
- The legacy historical benchmark records median/p95 summaries, not every raw per-trial sample. Consequently this stage prioritizes correct fresh numerical evidence over throughput claims.
- A finite numerical witness is not formal all-input equivalence or proof of correct behavior on every pretrained model.
- The pinned support environment is not an isolation boundary for arbitrary uploaded Python. We only run vetted checked-in code.
- New empirical evidence alone does not alter the public roadmap's approved weighted acceptance percentage.

Next, prioritize a **genuinely distinct pinned pretrained architecture family** with independent numerical oracles and exact model/parameter provenance. Then build a systematic transformation search that reuses the reviewed computational identities, not merely raw benchmark counts. Broader architecture coverage still requires real independent evidence, not another Tiny Llama run.
