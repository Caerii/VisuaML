"""v57 shared fixed-coefficient local-register selection and mutation.

The host supplies a canonical source-bound one-hot local index, but does not
select a particular local value or destination with Python indexing.
Works on the exact v49 bit-sliced i32 representation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exact_binary import is_exact_binary

from .bit_neural_vm_v49 import WIDTH
from .typed_microop_ir_v57 import LOCALS_WIDTH, MicroopAdmissionError

MODES = ("none", "get", "set", "tee")


@dataclass(frozen=True)
class LocalBankWeights:
    """Shared fixed gates; not selected or trained per WASM program."""

    read_gain: np.ndarray
    write_gain: np.ndarray


def construct_local_bank_weights() -> LocalBankWeights:
    return LocalBankWeights(
        np.ones(LOCALS_WIDTH, dtype=np.float64),
        np.ones(LOCALS_WIDTH, dtype=np.float64),
    )


def _binary(a: np.ndarray, shape: tuple[int, ...]) -> None:
    if (not isinstance(a, np.ndarray) or a.dtype != np.float64
            or a.shape != shape or not is_exact_binary(a)):
        raise MicroopAdmissionError("Noncanonical local-bank tensor")


def read_local(
    registers: np.ndarray,
    selector: np.ndarray,
    *,
    fixed: LocalBankWeights | None = None,
) -> np.ndarray:
    _binary(registers, (LOCALS_WIDTH, WIDTH))
    _binary(selector, (LOCALS_WIDTH,))
    if selector.sum() != 1.:
        raise MicroopAdmissionError("Local read requires one-hot source")
    fixed = construct_local_bank_weights() if fixed is None else fixed
    if not isinstance(fixed, LocalBankWeights):
        raise MicroopAdmissionError("Invalid local-bank weights")
    _binary(fixed.read_gain, (LOCALS_WIDTH,))
    selection = selector * fixed.read_gain
    if selection.sum() != 1.:
        raise MicroopAdmissionError("Local read gate ablated")
    result = selection @ registers
    _binary(result, (WIDTH,))
    return result


def write_local(
    registers: np.ndarray,
    selector: np.ndarray,
    value: np.ndarray,
    *,
    fixed: LocalBankWeights | None = None,
) -> np.ndarray:
    _binary(registers, (LOCALS_WIDTH, WIDTH))
    _binary(selector, (LOCALS_WIDTH,))
    _binary(value, (WIDTH,))
    if selector.sum() != 1.:
        raise MicroopAdmissionError("Local write requires one-hot source")
    fixed = construct_local_bank_weights() if fixed is None else fixed
    if not isinstance(fixed, LocalBankWeights):
        raise MicroopAdmissionError("Invalid local-bank weights")
    _binary(fixed.write_gain, (LOCALS_WIDTH,))
    selection = selector * fixed.write_gain
    if selection.sum() != 1.:
        raise MicroopAdmissionError("Local write gate ablated")
    result = (
        registers * (1. - selection[:, None])
        + selection[:, None] * value[None, :]
    )
    _binary(result, (LOCALS_WIDTH, WIDTH))
    return result


def apply_local_microop(
    registers: np.ndarray,
    selector: np.ndarray,
    mode: np.ndarray,
    value: np.ndarray,
    *,
    fixed: LocalBankWeights | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed one-hot mode gates own the local-bank transition and read output.

    The host may still choose how to change the WASM operand stack, but
    it cannot select whether the local bank is read or written. This is
    decided by the decoded v57 mode (none/get/set/tee) in the circuit.
    """
    _binary(mode, (len(MODES),))
    if mode.sum() != 1. or mode[0]:
        raise MicroopAdmissionError("Local effect requires get/set/tee mode")
    _binary(value, (WIDTH,))
    current = read_local(registers, selector, fixed=fixed)
    written = write_local(registers, selector, value, fixed=fixed)
    write_gate = mode[2] + mode[3]
    result_bank = (
        (1. - write_gate) * registers + write_gate * written
    )
    read_result = mode[1] * current
    _binary(result_bank, (LOCALS_WIDTH, WIDTH))
    _binary(read_result, (WIDTH,))
    return result_bank, read_result
