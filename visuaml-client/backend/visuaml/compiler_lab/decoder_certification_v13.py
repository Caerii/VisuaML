"""Reproducible claim-limited certification of one complete synthetic decoder AOT.

Every result is derived from executing a fixed-input graph, *not* importing a
real pretrained model or executing a physical accelerator. Metadata digests
establish content identity and reproducibility, not signature authenticity.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .decoder_aot_v13 import CompiledTinyDecoder, compile_decoder_chunk, verify_decoder_artifact
from .tiny_decoder_v12 import TinyDecoderWeights, greedy_generate

SCHEMA = "visuaml.decoder-aot-certification.v13"
SEEDS = (0, 19, 41)


def certify_decoder_compiler() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for seed in SEEDS:
        weights = TinyDecoderWeights.seeded(seed=seed)
        tokens = np.random.default_rng(seed + 99).integers(
            0, weights.embedding.shape[0], size=(2, 5), dtype=np.int64
        )
        witness = verify_decoder_artifact(weights, tokens, (2, 2, 1))
        driver = CompiledTinyDecoder(weights)
        generated, state = driver.generate(tokens[:, :3], steps=3)
        expected, reference = greedy_generate(tokens[:, :3], weights, steps=3)
        if not np.array_equal(generated, expected):
            raise AssertionError(f"Compiled generation changed tokens at seed {seed}")
        np.testing.assert_allclose(state.keys, reference.keys, atol=4e-5, rtol=4e-5)
        np.testing.assert_allclose(state.values, reference.values, atol=4e-5, rtol=4e-5)
        compiled = compile_decoder_chunk(weights, batch=2, query=2, past=0)
        checks.append(
            {
                "seed": seed,
                "input_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
                "prefill_manifest_sha256": compiled.manifest["manifest_sha256"],
                "input_shape": [2, 5],
                "witness": witness,
                "generation_steps": 3,
                "generation_token_sha256": hashlib.sha256(generated.tobytes()).hexdigest(),
                "generation_matches_reference": True,
                "cache_matches_reference": True,
            }
        )
    root = Path(__file__).resolve().parent
    hashes = hashlib.sha256(b"visuaml-decoder-aot-sources-v13\x00")
    for relative in (
        "decoder_aot_v13.py", "decoder_certification_v13.py", "tiny_decoder_v12.py",
        "family_blocks_v12.py", "aot_compile.py", "numpy_runtime.py", "ir_contract.py",
        "../compiler_spike.py",
    ):
        path = (root / relative).resolve()
        hashes.update(relative.encode())
        # The AST binds executed semantics without falsely invalidating the
        # certificate because Ruff/Black changed whitespace or quote style.
        tree = ast.parse(path.read_text(encoding="utf-8"))
        hashes.update(ast.dump(tree, annotate_fields=True, include_attributes=False).encode())
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "architecture_id": "decoder_transformer",
        "claim_level": "one-synthetic-tiny-decoder-full-chunk-compiler-execution",
        "backend": "shape-specialized NumPy-AOT interpreter (CPU only)",
        "source_kind": "synthetic-seeded-weights-and-tokens",
        "source_sha256": hashes.hexdigest(),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "cases": checks,
        "family_checkpoint_certified": False,
        "arbitrary_decoder_supported": False,
        "physical_accelerator_validated": False,
        "limitations": [
            "Each (batch,query,past) specialization is a different static AOT artifact.",
            "Input token-range validation occurs in the trusted host driver, not in the raw manifest.",
            "Numerical equivalence is demonstrated on finite witnesses, not proven for all inputs.",
            "This is a single-layer synthetic decoder, not a pretrained Llama checkpoint.",
            "No kernel fusion, measured transfer-cost model, or real photonic execution.",
        ],
    }
    report["sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return report
