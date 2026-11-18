"""Bind admitted v58 data rows to the native fused data gate."""
from __future__ import annotations

import numpy as np

from .bit_neural_vm_v49 import WIDTH
from .constructed_alu_v58 import compiled_alu, compiled_coefficients
from .neural_alu_v56 import ALU_OPS, ALUWeights, NeuralALUError
from .neural_local_bank_v57 import LocalBankWeights
from .neural_microstep_v58 import DATA_OPS, MicrostepWeights
from .neural_operand_storage_v55 import ACTIONS, MAX_STACK, OperandCircuitError, OperandWeights
from .typed_microop_ir_v57 import EFFECTS, LOCALS_WIDTH, LOCAL_MODES, OPCODES, MicroopAdmissionError

_ERRORS = (
    None,
    (OperandCircuitError, "Noncanonical binary operand tensor"),
    (OperandCircuitError, "Noncanonical one-hot operand token"),
    (OperandCircuitError, "Nonzero inactive operand slots"),
    (OperandCircuitError, "Bounded operand stack underflow"),
    (OperandCircuitError, "Bounded operand stack overflow"),
    (MicroopAdmissionError, "Noncanonical v58 microstep tensor"),
    (NeuralALUError, "ALU input/output must be exact binary tensor"),
    (NeuralALUError, "Fixed ALU gate fails unique selection"),
    (MicroopAdmissionError, "Expected one-hot fixed microstep selector"),
    (MicroopAdmissionError, "Noncanonical local-bank tensor"),
    (MicroopAdmissionError, "Local read gate ablated"),
    (MicroopAdmissionError, "Local write gate ablated"),
    (MicroopAdmissionError, "Fixed microstep lost stack effect authority"),
)

# This ordering is the row and coefficient ABI in constructed_alu_v58.cpp.
_LAYOUT = (
    ("retain", "push", "pop", "unary", "binary"),
    ("none", "get", "set", "tee"),
    ("i32.add", "i32.sub", "i32.mul", "i32.lt_s", "i32.gt_s", "i32.eqz"),
    ("retain", "push", "pop", "truncate"),
    32,
)


def _tensors(weights, layout, error, message):
    values = tuple(getattr(weights, name) for name, _ in layout)
    if any(type(value) is not np.ndarray or value.dtype != np.float64 or value.shape != (size,)
           for value, (_, size) in zip(values, layout, strict=True)):
        raise error(message)
    return values


class _NativeMicrostep:
    """Internal gate for owned operand and local buffers from the prepared executor."""

    def __init__(self, rows, micro, operand, alu, bits, local):
        if (EFFECTS, LOCAL_MODES, ALU_OPS, ACTIONS, WIDTH) != _LAYOUT:
            raise RuntimeError("Incompatible v58 native data gate layout")
        if not isinstance(micro, MicrostepWeights):
            raise MicroopAdmissionError("Unknown v58 fixed coefficients")
        if not isinstance(operand, OperandWeights):
            raise OperandCircuitError("Unknown circuit weights")
        if not isinstance(alu, ALUWeights):
            raise NeuralALUError("Unsupported ALU coefficient object")
        if not isinstance(local, LocalBankWeights):
            raise MicroopAdmissionError("Invalid local-bank weights")

        arithmetic = compiled_coefficients(bits)
        if tuple(value.size for value in arithmetic) != (3, 4, 32, 2):
            raise RuntimeError("Incompatible v58 native arithmetic layout")

        fixed = (
            *_tensors(micro, (("pop_gain", 2), ("produce_gain", 3), ("write_gain", 1)),
                      MicroopAdmissionError, "Noncanonical v58 microstep tensor"),
            *_tensors(operand, (("action_gain", len(ACTIONS)),),
                      OperandCircuitError, "Noncanonical binary operand tensor"),
            *_tensors(alu, (("output_gains", len(ALU_OPS)),),
                      NeuralALUError, "ALU input/output must be exact binary tensor"),
            *arithmetic,
            *_tensors(local, (("read_gain", LOCALS_WIDTH), ("write_gain", LOCALS_WIDTH)),
                      MicroopAdmissionError, "Noncanonical local-bank tensor"),
        )
        self._fixed = np.frombuffer(np.concatenate([value.ravel() for value in fixed]).tobytes(),
                                    dtype=np.float64)
        const = OPCODES.index("i32.const")
        data = np.array([name in DATA_OPS for name in OPCODES], dtype=np.float64)
        table = []
        for row in rows:
            if row.opcode @ data != 1.:
                raise MicroopAdmissionError("Control opcode outside v58 data microstep")
            table.append(np.concatenate((
                row.effect, row.local_mode, row.alu_selector, row.immediate_bits,
                row.local_selector, row.opcode[const:const + 1],
            )))
        self._rows = np.frombuffer(np.stack(table).tobytes(), dtype=np.float64)
        self._gate = compiled_alu().data_microstep
        self._pointers = self._rows.ctypes.data, self._fixed.ctypes.data

    def apply(self, pc, buffers):
        status = self._gate(pc, MAX_STACK, LOCALS_WIDTH, *self._pointers, *buffers)
        if status:
            error, message = _ERRORS[status]
            raise error(message)
