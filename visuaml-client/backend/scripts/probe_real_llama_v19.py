"""Opt-in CI probe of a pinned, independently executed pretrained Llama checkpoint.

Downloads only config.json and model.safetensors from the allowlisted public
TinyStories checkpoint. Never uses trust_remote_code, arbitrary model paths,
pickle or server-uploaded source. This is integration verification, not a
general production Llama compiler or tokenizer validation.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from visuaml.compiler_lab.hf_llama_checkpoint_v19 import load_hf_llama_local

MODEL_ID = "manojredhat/tiny-llama"


def run() -> dict:
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoModelForCausalLM

    info = HfApi().model_info(MODEL_ID)
    revision = info.sha
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("Missing immutable pretrained checkpoint revision")
    with tempfile.TemporaryDirectory(prefix="visuaml-pretrained-probe-") as directory:
        local = snapshot_download(
            repo_id=MODEL_ID,
            revision=revision,
            allow_patterns=("config.json", "model.safetensors"),
            local_dir=directory,
        )
        importer = load_hf_llama_local(local, declared_revision=revision)
        # This model omits safetensors metadata.format; Transformers 4.46.3
        # rejects that otherwise valid file before reading any tensors.
        # Produce a temporary, tensor-identical reference copy with the
        # standard metadata field. The original source hash remains recorded.
        import shutil
        from safetensors.torch import save_file

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
        witnessed = []
        for tokens in cases:
            with torch.no_grad():
                expected = hf(torch.from_numpy(tokens)).logits.detach().cpu().numpy()
            actual = importer.forward(tokens)
            np.testing.assert_allclose(actual, expected, rtol=5e-4, atol=5e-4)
            witnessed.append({
                "tokens": tokens.tolist(),
                "worst_abs_error": float(np.max(np.abs(actual - expected))),
                "mean_abs_error": float(np.mean(np.abs(actual - expected))),
                "output_shape": list(actual.shape),
            })
        return {
            "schema": "visuaml.pretrained-llama-parity.v19",
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
            "reference_weight_copy": "Tensor-identical safetensors with format=pt metadata",
            "comparison": witnessed,
            "tolerance": {"atol": 5e-4, "rtol": 5e-4},
            "limitations": [
                "Full prompt logits only, no tokenizer parity or streamed KV decoding.",
                "HF Llama frontend is separate from canonical v18 Semantic IR.",
                "No physical GPU/photonic device or throughput claim.",
            ],
        }


def main() -> None:
    result = run()
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
