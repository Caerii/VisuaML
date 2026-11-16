"""Serve a trusted, bounded semantic decoder witness for the visual inspector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from visuaml.compiler_lab.memory_planner_v19 import plan_decoder_memory
from visuaml.compiler_lab.semantic_decoder_v18 import (
    lower_decoder_semantic_program,
    make_decoder_semantic_program,
)
from visuaml.compiler_lab.stacked_decoder_v14 import StackedWeights
from visuaml.compiler_lab.streaming_decoder_v16 import StreamingStackedDecoder


def inspect(*, layers: int = 2, seed: int = 19, batch: int = 1) -> dict:
    if (
        type(layers) is not int or not 1 <= layers <= 4
        or type(seed) is not int or not 0 <= seed <= 100_000
        or type(batch) is not int or not 1 <= batch <= 2
    ):
        raise ValueError("Only fixed bounded synthetic decoder cases are admitted")
    weights = StackedWeights.seeded(seed=seed, layers=layers)
    program = make_decoder_semantic_program(weights, batch=batch)
    compiled = lower_decoder_semantic_program(program, weights, batch=batch)
    engine = compiled.new_session()
    oracle = StreamingStackedDecoder(weights, batch=batch)
    tokens = (np.arange(4, dtype=np.int64)[None, :] + 1).repeat(batch, axis=0)
    observed, expected = engine.step(tokens), oracle.step(tokens)
    np.testing.assert_allclose(observed, expected, rtol=3e-5, atol=3e-5)
    worst = float(np.max(np.abs(observed - expected)))
    for new, old in zip(engine.snapshot().layers, oracle.snapshot().layers):
        np.testing.assert_allclose(new.keys, old.keys, rtol=3e-5, atol=3e-5)
        np.testing.assert_allclose(new.values, old.values, rtol=3e-5, atol=3e-5)
        worst = max(
            worst, float(np.max(np.abs(new.keys - old.keys))),
            float(np.max(np.abs(new.values - old.values))),
        )
    return {
        "schema": "visuaml.semantic-inspection.v19",
        "source": "trusted seeded synthetic checkpoint",
        "fixture": {"layers": layers, "seed": seed, "batch": batch},
        "semantic_sha256": compiled.digest,
        "checkpoint_sha256": compiled.checkpoint_sha256,
        "operations": [
            {
                "name": op.name, "opcode": op.op, "operands": list(op.operands),
                "result": op.result,
                "shape": list(op.result_type.shape) if op.result_type else None,
                "effect": op.effect, "cell": op.state_cell,
            }
            for op in program.regions[0].operations
        ],
        "memory": plan_decoder_memory(program).as_dict(),
        "verification": {
            "passed": True, "worst_abs_error": worst,
            "reference": "independent v16 resident NumPy decoder",
            "tensor_coverage": "logits and every per-layer K/V tensor",
        },
        "limitations": [
            "Not a pretrained checkpoint or tokenizer.",
            "Liveness is a logical SSA estimate, not measured physical allocation.",
            "This executor uses NumPy CPU, not native AOT, CUDA, or photonic hardware.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    document = inspect(layers=args.layers, seed=args.seed, batch=args.batch)
    wire = json.dumps(document, allow_nan=False, separators=(",", ":"))
    if len(wire.encode()) > 512 * 1024:
        parser.error("Inspection report is oversized")
    print(wire)


if __name__ == "__main__":
    main()
