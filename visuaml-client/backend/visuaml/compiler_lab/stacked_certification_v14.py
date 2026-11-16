"""Executable AOT certificates for bounded 1..4-layer synthetic GQA decoders.

The receipt records content identities and finite numerical witnesses. It does
not authenticate weight provenance, certify pretrained model compatibility,
prove numerical equivalence for every input, or measure hardware performance.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .stacked_decoder_v14 import (
    CompiledStackedDecoder,
    StackedWeights,
    checkpoint_digest,
    empty_stack,
    reference_stack_step,
    verify_stacked_witness,
)

SCHEMA = "visuaml.stacked-decoder-aot-certificate.v14"
CASES = ((1, 7), (2, 19), (4, 41))


def semantic_source_sha256() -> str:
    root = Path(__file__).resolve().parent
    h = hashlib.sha256(b"visuaml-v14-compiled-stacks\x00")
    for name in (
        "stacked_decoder_v14.py",
        "stacked_certification_v14.py",
        "decoder_aot_v13.py",
        "tiny_decoder_v12.py",
        "family_blocks_v12.py",
        "aot_compile.py",
        "numpy_runtime.py",
        "ir_contract.py",
        "../compiler_spike.py",
    ):
        h.update(name.encode())
        source = (root / name).resolve().read_text(encoding="utf-8")
        h.update(
            ast.dump(
                ast.parse(source), annotate_fields=True, include_attributes=False
            ).encode()
        )
    return h.hexdigest()


def certify_stacked_decoder() -> dict[str, Any]:
    cases = []
    for layers, seed in CASES:
        weights = StackedWeights.seeded(seed, layers=layers)
        rng = np.random.default_rng(seed + 10)
        tokens = rng.integers(
            0, weights.embedding.shape[0], size=(2, 5), dtype=np.int64
        )
        witness = verify_stacked_witness(weights, tokens, (2, 2, 1))
        decoder = CompiledStackedDecoder(weights)
        generated, observed_state = decoder.generate(tokens[:, :3], steps=3)
        initial = empty_stack(weights, tokens.shape[0])
        ref, reference_state = reference_stack_step(weights, tokens[:, :3], initial)
        wanted = []
        for _ in range(3):
            ids = np.argmax(ref[:, -1, :], axis=-1).astype(np.int64)[:, None]
            wanted.append(ids)
            ref, reference_state = reference_stack_step(weights, ids, reference_state)
        np.testing.assert_array_equal(generated, np.concatenate(wanted, axis=1))
        for a, b in zip(observed_state.layers, reference_state.layers):
            np.testing.assert_allclose(a.keys, b.keys, atol=5e-5, rtol=5e-5)
            np.testing.assert_allclose(a.values, b.values, atol=5e-5, rtol=5e-5)
        cases.append(
            {
                "layers": layers,
                "seed": seed,
                "tokens_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
                "checkpoint_sha256": checkpoint_digest(weights),
                "witness": witness,
                "generated_token_sha256": hashlib.sha256(
                    generated.tobytes()
                ).hexdigest(),
                "generation_matches_reference": True,
                "all_layer_cache_matches_reference": True,
            }
        )
    report = {
        "schema": SCHEMA,
        "architecture_id": "decoder_transformer",
        "claim_level": "1-to-4-layer-synthetic-shape-specialized-numpy-aot-execution",
        "backend": "standalone NumPy CPU interpreter",
        "source_kind": "synthetic-seeded-checkpoint-fixtures",
        "source_sha256": semantic_source_sha256(),
        "cases": cases,
        "canonical_checkpoint_io": "NumPy NPZ arrays + SHA-256-verified schema; no code execution",
        "pretrained_checkpoint_certified": False,
        "arbitrary_layer_count_supported": False,
        "dynamic_shape_aot_supported": False,
        "physical_hardware_validated": False,
        "limitations": [
            "Only 1..4 GQA/RoPE/SwiGLU blocks with concrete dimensions and compatible KV heads.",
            "Every (batch,query,past) combination produces an independent AOT specialization.",
            "Canonical checkpoints have user-declared provenance, not authenticated model weights.",
            "No tokenizer, tied-weight policy, sampling distributions, external model-format adapter, or pretrained checkpoint test.",
            "The lowerer's PyTorch-free artifact executes on CPU NumPy, never photonic hardware.",
        ],
    }
    report["sha256"] = hashlib.sha256(
        json.dumps(
            report, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    return report
