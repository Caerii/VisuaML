"""Transactional NumPy interpreter for validated bounded effect programs.

No eval, imports from source strings, dynamic callable registration, or user
callbacks. All public inputs are copied before execution; a failed step cannot
mutate its caller's state. Receipts capture every completed loop iteration.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np

from .effect_ir import EffectProgram, EffectIRError, TensorType, validate_effect_program


@dataclass(frozen=True)
class IterationReceipt:
    loop: str
    iteration: int
    before_sha256: str
    after_sha256: str
    effect_count: int


@dataclass(frozen=True)
class EffectExecution:
    final_state: dict[str, np.ndarray]
    program_sha256: str
    initial_sha256: str
    final_sha256: str
    receipts: tuple[IterationReceipt, ...]

    def report(self) -> dict:
        return {
            "schema": "visuaml.effect-execution.v1",
            "program_sha256": self.program_sha256,
            "initial_sha256": self.initial_sha256,
            "final_sha256": self.final_sha256,
            "iterations": [asdict(r) for r in self.receipts],
        }


def _checked(value: np.ndarray, t: TensorType, name: str) -> np.ndarray:
    a = np.asarray(value)
    if a.shape != t.shape or a.dtype != np.dtype(t.dtype) or not np.all(np.isfinite(a)):
        raise EffectIRError(f"{name} violates shape, dtype, or finiteness contract")
    return a.copy()


def state_sha256(state: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256(b"visuaml.effect-state.v1")
    for name in sorted(state):
        arr = state[name]
        digest.update(name.encode("utf-8"))
        digest.update(repr(arr.shape).encode("utf-8"))
        digest.update(arr.dtype.str.encode("utf-8"))
        digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def _compute(op: str, args: list[np.ndarray], params: dict) -> np.ndarray:
    if op == "identity":
        out = args[0].copy()
    elif op == "add":
        out = np.add(args[0], args[1])
    elif op == "subtract":
        out = np.subtract(args[0], args[1])
    elif op == "matmul":
        out = np.matmul(args[0], args[1])
    elif op == "relu":
        out = np.maximum(args[0], np.float32(0))
    elif op == "tanh":
        out = np.tanh(args[0])
    elif op == "gather_sum":
        w, index = args
        if np.any(index < 0) or np.any(index >= w.shape[0]) or np.unique(index).size != len(index):
            raise EffectIRError("gather_sum received invalid or repeated source indices")
        out = np.sum(w[index], axis=0, dtype=np.float32)
    elif op == "topk_indices":
        (drive,) = args
        if not np.all(np.isfinite(drive)):
            raise EffectIRError("Cannot select winners from non-finite drive")
        chosen = np.lexsort((np.arange(drive.size), -drive))[:params["k"]]
        out = np.sort(chosen).astype(np.int64)
    elif op == "hebbian_update":
        w, source, chosen = args
        if (np.unique(source).size != source.size or np.unique(chosen).size != chosen.size
            or np.any(source < 0) or np.any(source >= w.shape[0])
            or np.any(chosen < 0) or np.any(chosen >= w.shape[1])):
            raise EffectIRError("Invalid Hebbian source or winner index")
        out = w.copy()
        ix = np.ix_(source, chosen)
        out[ix] *= np.float32(1.0 + params["beta"])
        np.clip(out, 0, params["ceiling"], out=out)
    elif op == "refraction_update":
        b, raw, selected = args
        if np.unique(selected).size != selected.size or np.any(selected < 0) or np.any(selected >= b.size):
            raise EffectIRError("Invalid refraction winner index")
        out = b.copy()
        out[selected] += raw[selected] * np.float32(params["strength"])
    else:
        raise EffectIRError(f"Unsupported runtime operation {op}")
    if not np.all(np.isfinite(out)):
        raise EffectIRError(f"Non-finite computation in {op}")
    return out


def execute_effect_program(program: EffectProgram,
    initial_state: Mapping[str, np.ndarray], inputs: Mapping[str, np.ndarray] | None = None,
) -> EffectExecution:
    """Execute atomically: publish a fresh state only after every loop succeeds."""
    admission = validate_effect_program(program)
    state_spec = {s.name: s.tensor for s in program.state}
    input_spec = {s.name: s.tensor for s in program.inputs}
    inputs = inputs or {}
    if set(initial_state) != set(state_spec) or set(inputs) != set(input_spec):
        raise EffectIRError("Initial state or input names do not match declared interface")
    state = {name: _checked(initial_state[name], spec, name) for name, spec in state_spec.items()}
    given = {name: _checked(inputs[name], spec, name) for name, spec in input_spec.items()}
    initial_hash = state_sha256(state)
    receipts: list[IterationReceipt] = []
    for loop in program.loops:
        for i in range(loop.count):
            before_hash = state_sha256(state)
            values: dict[str, np.ndarray] = {name: v.copy() for name,v in given.items()}
            effect_count = 0
            for ins in loop.instructions:
                if ins.op == "read":
                    values[ins.output] = state[ins.cell].copy()
                    effect_count += 1
                elif ins.op == "write":
                    state[ins.cell] = _checked(values[ins.args[0]], state_spec[ins.cell], ins.cell)
                    effect_count += 1
                else:
                    values[ins.output] = _compute(ins.op, [values[x] for x in ins.args], dict(ins.params))
            receipts.append(IterationReceipt(loop.name, i, before_hash, state_sha256(state), effect_count))
    # Copy and freeze outputs. Caller-supplied arrays have never been mutated.
    final = {k:v.copy() for k,v in state.items()}
    for v in final.values():
        v.flags.writeable = False
    return EffectExecution(final, admission.program_sha256, initial_hash,
                           state_sha256(final), tuple(receipts))
