"""Opt-in CI probe of a pinned, independently executed pretrained Llama checkpoint.

Downloads only config.json and model.safetensors from the allowlisted public
TinyStories checkpoint. Never uses trust_remote_code, arbitrary model paths,
pickle or server-uploaded source. This is integration verification, not a
general production Llama compiler or tokenizer validation.
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

from examples.llama_semantic_bench_v20 import benchmark
from visuaml.compiler_lab.hf_llama_checkpoint_v19 import load_hf_llama_local
from visuaml.compiler_lab.llama_semantic_v20 import (
    lower_llama_semantic_program,
    make_llama_semantic_program,
)
from visuaml.compiler_lab.sentencepiece_tokenizer_v20 import (
    load_sentencepiece_tokenizer,
)

MODEL_ID = "manojredhat/tiny-llama"
REVISION = "756e9746eb801752d69771ef3e9bc433b21d853d"
WEIGHTS_SHA256 = "d910edc7031721de23a7af111e355e6c09d5f1b3a2c43c0b6b0aa6a9135ffe03"
CONFIG_SHA256 = "b53aec3cd6cfb1be4b96cadf8ca0c5b9e0279d751100dbebd8f93ff6564c0e02"
TOKENIZER_SHA256 = "219d5967cd1bc4dbdd0d880fddcf4d61a703391f79c889dc63a0c4b0eb367823"
TOKENIZER_CONFIG_SHA256 = "3bb9c9aa2c3b5f44436dc1bbe2092a15dfcf6f1d37299bd199a496b1db0aa1e3"


def run() -> dict:
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    info = HfApi().model_info(MODEL_ID, revision=REVISION)
    revision = info.sha
    if revision != REVISION:
        raise ValueError("Pinned checkpoint revision changed unexpectedly")
    with tempfile.TemporaryDirectory(prefix="visuaml-pretrained-probe-") as directory:
        local = snapshot_download(
            repo_id=MODEL_ID,
            revision=revision,
            allow_patterns=("config.json", "model.safetensors", "tokenizer.model", "tokenizer_config.json"),
            local_dir=directory,
        )
        importer = load_hf_llama_local(
            local, declared_revision=revision,
            expected_weights_sha256=WEIGHTS_SHA256,
        )
        if importer.config_sha256 != CONFIG_SHA256:
            raise ValueError("Pinned model config digest mismatch")
        # This model omits safetensors metadata.format; Transformers 4.46.3
        # rejects that otherwise valid file before reading any tensors.
        # Produce a temporary, tensor-identical reference copy with the
        # standard metadata field. The original source hash remains recorded.
        reference_dir = Path(directory) / "hf_reference"
        reference_dir.mkdir()
        shutil.copy2(Path(local) / "config.json", reference_dir / "config.json")
        normalized_weights = {
            name: torch.from_numpy(np.array(array, copy=True))
            for name, array in importer.arrays.items()
        }
        save_file(
            normalized_weights, str(reference_dir / "model.safetensors"),
            metadata={"format": "pt"},
        )
        hf = AutoModelForCausalLM.from_pretrained(
            reference_dir, local_files_only=True,
            use_safetensors=True, trust_remote_code=False,
            torch_dtype=torch.float32,
        ).eval()
        cases = (
            np.array([[1, 2, 3, 4]], np.int64),
            np.array([[2, 11, 9, 4, 31, 8]], np.int64),
        )
        canonical_ir = make_llama_semantic_program(importer, batch=1)
        compiled = lower_llama_semantic_program(canonical_ir, importer, batch=1)
        witnessed = []
        worst_ir = 0.0
        for tokens in cases:
            with torch.no_grad():
                expected = hf(torch.from_numpy(tokens)).logits.detach().cpu().numpy()
            actual = importer.forward(tokens)
            semantic_session = compiled.new_session()
            actual_ir = semantic_session.step(tokens)
            np.testing.assert_allclose(actual, expected, rtol=5e-4, atol=5e-4)
            np.testing.assert_allclose(actual_ir, expected, rtol=5e-4, atol=5e-4)
            worst_ir = max(worst_ir, float(np.max(np.abs(actual_ir - expected))))
            witnessed.append({
                "tokens": tokens.tolist(),
                "worst_abs_error": float(np.max(np.abs(actual - expected))),
                "mean_abs_error": float(np.mean(np.abs(actual - expected))),
                "output_shape": list(actual.shape),
                "semantic_ir_worst_abs_error": float(np.max(np.abs(actual_ir - expected))),
            })
        streaming = compiled.new_session()
        full_tokens = cases[1]
        streaming.step(full_tokens[:, :3])
        incremental = streaming.step(full_tokens[:, 3:4])
        streaming_rest = streaming.step(full_tokens[:, 4:])
        with torch.no_grad():
            full_logits = hf(torch.from_numpy(full_tokens)).logits.cpu().numpy()
        np.testing.assert_allclose(
            incremental, full_logits[:, 3:4], rtol=5e-4, atol=5e-4
        )
        np.testing.assert_allclose(
            streaming_rest, full_logits[:, 4:], rtol=5e-4, atol=5e-4
        )
        whole_graph = compiled.new_session(backend="torchscript_graph")
        whole_graph_logits = whole_graph.step(cases[0])
        hybrid = compiled.new_session(backend="torchscript_hybrid")
        hybrid_logits = hybrid.step(cases[0])
        fused = compiled.new_session(backend="torchscript_fused")
        fused_logits = fused.step(cases[0])
        with torch.no_grad():
            hybrid_reference = hf(torch.from_numpy(cases[0])).logits.cpu().numpy()
        np.testing.assert_allclose(
            hybrid_logits, hybrid_reference, rtol=5e-4, atol=5e-4
        )
        np.testing.assert_allclose(
            whole_graph_logits, hybrid_reference, rtol=5e-4, atol=5e-4
        )
        np.testing.assert_allclose(
            fused_logits, hybrid_reference, rtol=5e-4, atol=5e-4
        )
        if hybrid.report()["compiled_primitive_calls"] <= 0:
            raise ValueError("Hybrid backend failed to execute a compiled primitive")
        functional_graph = compiled.new_session(backend="torchscript_graph")
        functional_graph.step(full_tokens[:, :3])
        graph_second = functional_graph.step(full_tokens[:, 3:4])
        graph_tail = functional_graph.step(full_tokens[:, 4:])
        np.testing.assert_allclose(
            graph_second, full_logits[:, 3:4], rtol=5e-4, atol=5e-4
        )
        np.testing.assert_allclose(
            graph_tail, full_logits[:, 4:], rtol=5e-4, atol=5e-4
        )
        greedy = compiled.new_session()
        generated, _ = greedy.generate(cases[0], steps=3)
        current = cases[0].copy()
        reference_ids = []
        for _ in range(3):
            with torch.no_grad():
                logits = hf(torch.from_numpy(current)).logits.cpu().numpy()
            ids = np.argmax(logits[:, -1], axis=-1).astype(np.int64)[:, None]
            reference_ids.append(ids)
            current = np.concatenate((current, ids), axis=1)
        np.testing.assert_array_equal(
            generated, np.concatenate(reference_ids, axis=1)
        )
        tokenizer = load_sentencepiece_tokenizer(
            local, declared_revision=revision,
            vocabulary=importer.profile.vocabulary,
            expected_model_sha256=TOKENIZER_SHA256,
        )
        if tokenizer.identity.config_sha256 != TOKENIZER_CONFIG_SHA256:
            raise ValueError("Pinned tokenizer configuration fingerprint mismatch")
        hf_tokenizer = AutoTokenizer.from_pretrained(
            local, use_fast=False, local_files_only=True,
            trust_remote_code=False,
        )
        text_witnesses = []
        for message in ("Once upon a time", "The little cat ran home."):
            input_ids = tokenizer.encode(message, add_bos=True)
            hf_ids = hf_tokenizer(message, add_special_tokens=True)["input_ids"]
            np.testing.assert_array_equal(
                input_ids[0], np.asarray(hf_ids, np.int64)
            )
            with torch.no_grad():
                oracle_logits = hf(torch.from_numpy(input_ids)).logits.cpu().numpy()
            compiler_logits = compiled.new_session().step(input_ids)
            np.testing.assert_allclose(
                compiler_logits, oracle_logits, rtol=5e-4, atol=5e-4
            )
            text_witnesses.append({
                "text": message,
                "token_ids": input_ids.tolist()[0],
                "logit_max_abs_error": float(np.max(np.abs(compiler_logits - oracle_logits))),
            })
        timing_evidence = benchmark(
            importer, lengths=(4, 8, 16), trials=5,
            include_hybrid=True, include_whole_graph=True,
        )
        return {
            "schema": "visuaml.pretrained-llama-parity.v20",
            "matched_cpu_benchmark": timing_evidence,
            "source": MODEL_ID,
            "declared_source_revision": revision,
            "checkpoint_sha256": importer.checkpoint_sha256,
            "config_sha256": importer.config_sha256,
            "architecture": {
                "layers": importer.profile.layers,
                "hidden": importer.profile.hidden,
                "vocab": importer.profile.vocabulary,
                "heads": importer.profile.heads,
                "kv_heads": importer.profile.kv_heads,
            },
            "framework": "transformers AutoModelForCausalLM CPU",
            "semantic_ir_sha256": compiled.semantic_sha256,
            "semantic_ir_max_abs_error": worst_ir,
            "streaming_ir_cache_tokens": streaming.cursor,
            "hybrid_compiled_primitive_calls": hybrid.report()["compiled_primitive_calls"],
            "hybrid_compiled_primitive_specializations": hybrid.report()["compiled_primitive_specializations"],
            "hybrid_max_abs_error": float(np.max(np.abs(hybrid_logits - hybrid_reference))),
            "whole_graph_max_abs_error": float(np.max(np.abs(whole_graph_logits - hybrid_reference))),
            "whole_graph_streaming_past_tokens": functional_graph.cursor,
            "whole_graph_trace_count": len(functional_graph.report()["traces"]),
            "fused_ffn_region_calls": fused.report()["fused_ffn_region_calls"],
            "fused_ffn_max_abs_error": float(np.max(np.abs(fused_logits - hybrid_reference))),
            "greedy_tokens_match_hf_recompute": True,
            "tokenizer_model_sha256": tokenizer.identity.model_sha256,
            "tokenizer_config_sha256": tokenizer.identity.config_sha256,
            "tokenizer_parity": text_witnesses,
            "greedy_generated_ids": generated.tolist(),
            "reference_weight_copy": "Tensor-identical safetensors with format=pt metadata",
            "comparison": witnessed,
            "tolerance": {"atol": 5e-4, "rtol": 5e-4},
            "limitations": [
                "Tokenizer parity is verified on only two short text witnesses.",
                "v20 uses a Llama-specific typed dialect rather than unrestricted model IR.",
                "v21 TorchScript graph compiles all decoder math for concrete shapes, not standalone AOT or polymorphic decode.",
                "No physical GPU/photonic device or throughput claim.",
            ],
        }


def main() -> None:
    result = run()
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
