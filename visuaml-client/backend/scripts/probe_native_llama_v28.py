"""Pinned *real pretrained* Tiny Llama parity against independent native C++17.

This workflow permits one hardcoded Hugging Face model revision and safetensors
SHA-256. Python validates the local checkpoint and compiles a fixed C++ source;
the native kernel performs the mathematics without PyTorch/NumPy kernels.
An independent Hugging Face PyTorch model and the VisuaML NumPy semantic IR
are the numerical oracles.

No arbitrary Python model code, native source uploads, or download of a C++
compiler. This is a bounded research certificate, not production performance.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from examples.native_cpu_bench_v28 import benchmark
from examples.native_cpu_prefill_bench_v30 import benchmark as prefill_benchmark
from examples.native_online_attention_bench_v31 import benchmark as attention_benchmark
from visuaml.compiler_lab.hf_llama_checkpoint_v19 import load_hf_llama_local
from visuaml.compiler_lab.llama_semantic_v20 import (
    lower_llama_semantic_program,
    make_llama_semantic_program,
)
from visuaml.compiler_lab.native_cpu_v28 import NativeCpuLlamaCompiler

MODEL_ID = "manojredhat/tiny-llama"
REVISION = "756e9746eb801752d69771ef3e9bc433b21d853d"
WEIGHT_SHA = "d910edc7031721de23a7af111e355e6c09d5f1b3a2c43c0b6b0aa6a9135ffe03"
CONFIG_SHA = "b53aec3cd6cfb1be4b96cadf8ca0c5b9e0279d751100dbebd8f93ff6564c0e02"


def run() -> dict:
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoModelForCausalLM

    info = HfApi().model_info(MODEL_ID, revision=REVISION)
    if info.sha != REVISION:
        raise RuntimeError("Pinned pretrained source revision changed")

    with tempfile.TemporaryDirectory(prefix="visuaml-native-v28-") as directory:
        local = Path(snapshot_download(
            repo_id=MODEL_ID,
            revision=REVISION,
            allow_patterns=("config.json", "model.safetensors"),
            local_dir=directory,
        ))
        imported = load_hf_llama_local(
            local, declared_revision=REVISION,
            expected_weights_sha256=WEIGHT_SHA,
        )
        if imported.config_sha256 != CONFIG_SHA:
            raise RuntimeError("Pinned configuration fingerprint mismatch")
        semantic = make_llama_semantic_program(imported)
        plan = lower_llama_semantic_program(semantic, imported)
        native_compiler = NativeCpuLlamaCompiler(plan)
        batched_compiler = NativeCpuLlamaCompiler(
            plan, enable_avx2=True, enable_batched_prefill=True
        )

        # Metadata-normalize only the independent reference copy:
        # the original source bytes stay separately pinned and unchanged.
        reference = local / "hf_reference"
        reference.mkdir()
        shutil.copy2(local / "config.json", reference / "config.json")
        save_file(
            {
                name: torch.from_numpy(np.asarray(value).copy())
                for name, value in imported.arrays.items()
            },
            str(reference / "model.safetensors"),
            metadata={"format": "pt"},
        )
        framework = AutoModelForCausalLM.from_pretrained(
            reference, local_files_only=True, use_safetensors=True,
            trust_remote_code=False, torch_dtype=torch.float32,
        ).eval()

        witnesses = []
        for token_list in ((1, 2, 3, 4), (7, 3, 8, 2, 1, 9)):
            tokens = np.array([token_list], dtype=np.int64)
            native = native_compiler.new_session()
            semantic_session = plan.new_session(backend="numpy")
            with torch.no_grad():
                expected = framework(torch.from_numpy(tokens)).logits.cpu().numpy()
            actual = native.step(tokens)
            from_batched = batched_compiler.new_session().step(tokens)
            from_ir = semantic_session.step(tokens)
            np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-4)
            np.testing.assert_allclose(actual, from_ir, rtol=3e-4, atol=3e-4)
            np.testing.assert_allclose(
                from_batched, expected, rtol=3e-4, atol=3e-4
            )
            np.testing.assert_allclose(
                from_batched, from_ir, rtol=3e-4, atol=3e-4
            )
            for a, b in zip(native.snapshot().layers, semantic_session.snapshot().layers):
                np.testing.assert_allclose(a.keys, b.keys, rtol=3e-4, atol=3e-4)
                np.testing.assert_allclose(a.values, b.values, rtol=3e-4, atol=3e-4)
            witnesses.append({
                "tokens": tokens.tolist(),
                "logit_shape": list(actual.shape),
                "max_abs_error_against_HF": float(np.max(np.abs(actual - expected))),
                "max_abs_error_against_semantic_IR": float(np.max(np.abs(actual - from_ir))),
                "max_abs_error_batched_against_HF": float(
                    np.max(np.abs(from_batched - expected))
                ),
            })

        # The decoder carries a private resident KV across arbitrary chunks.
        full = np.array([[1, 2, 3, 5, 4, 7]], dtype=np.int64)
        native = native_compiler.new_session()
        batched = batched_compiler.new_session()
        oracle = plan.new_session(backend="numpy")
        offsets = ((0, 3), (3, 4), (4, 6))
        for lo, hi in offsets:
            got = native.step(full[:, lo:hi])
            got_batched = batched.step(full[:, lo:hi])
            direct = oracle.step(full[:, lo:hi])
            with torch.no_grad():
                expected = framework(torch.from_numpy(full[:, :hi])).logits.cpu().numpy()[:, lo:hi]
            np.testing.assert_allclose(got, direct, rtol=3e-4, atol=3e-4)
            np.testing.assert_allclose(got, expected, rtol=3e-4, atol=3e-4)
            np.testing.assert_allclose(
                got_batched, expected, rtol=3e-4, atol=3e-4
            )
        for a, b in zip(batched.snapshot().layers, oracle.snapshot().layers):
            np.testing.assert_allclose(a.keys, b.keys, rtol=3e-4, atol=3e-4)
            np.testing.assert_allclose(a.values, b.values, rtol=3e-4, atol=3e-4)
        for a, b in zip(native.snapshot().layers, oracle.snapshot().layers):
            np.testing.assert_allclose(a.keys, b.keys, rtol=3e-4, atol=3e-4)
            np.testing.assert_allclose(a.values, b.values, rtol=3e-4, atol=3e-4)

        generated, state = native_compiler.new_session().generate(full[:, :3], steps=3)
        baseline, baseline_state = plan.new_session(backend="numpy").generate(
            full[:, :3], steps=3
        )
        np.testing.assert_array_equal(generated, baseline)
        for a, b in zip(state.layers, baseline_state.layers):
            np.testing.assert_allclose(a.keys, b.keys, atol=3e-4, rtol=3e-4)
            np.testing.assert_allclose(a.values, b.values, atol=3e-4, rtol=3e-4)
        matched_performance = benchmark(
            imported, prompt=4, chunk_sizes=(1, 1, 2), trials=7,
            compare_native_reference=True, compare_simd=True,
        )
        prefill_performance = prefill_benchmark(
            imported, prompt=8, decode=(1, 1), trials=5,
        )
        admitted_prefixes = tuple(
            length for length in (4, 8, 16, 32)
            if length < min(imported.profile.context, 64)
        )
        attention_performance = attention_benchmark(
            imported, prefixes=admitted_prefixes, trials=3,
        )
        return {
            "schema": "visuaml.pinned-native-llama-v28",
            "matched_cpu_benchmark": matched_performance,
            "v30_prefill_decode_benchmark": prefill_performance,
            "v31_online_attention_benchmark": attention_performance,
            "source": MODEL_ID,
            "revision": REVISION,
            "checkpoint_sha256": imported.checkpoint_sha256,
            "config_sha256": imported.config_sha256,
            "semantic_sha256": plan.semantic_sha256,
            "backend": native_compiler.report(),
            "batched_backend": batched_compiler.report(),
            "witnesses": witnesses,
            "chunked_KV_and_greedy_parity": True,
            "limitations": [
                "A scalar independent C++17 CPU kernel, not GPU or photonic hardware.",
                "Numerical tolerances hold for only the pinned inputs and model.",
                "This backend is not yet optimized against production CPU inference engines.",
                "Temporary local library is not a portable independently signed native AOT package.",
            ],
        }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True), flush=True)
