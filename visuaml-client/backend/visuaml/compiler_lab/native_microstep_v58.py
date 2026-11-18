"""Bind admitted v58 data rows to the native fused data gate."""
from __future__ import annotations

import ctypes

import numpy as np

from .bit_neural_vm_v49 import WIDTH, _validate_i32
from .compact_routing_v58 import CompactRouter
from .constructed_alu_v58 import compiled_alu, compiled_coefficients
from .neural_alu_v56 import ALU_OPS, ALUWeights, NeuralALUError
from .neural_control_routing_v52 import TERMINAL, NeuralControlError
from .neural_local_bank_v57 import LocalBankWeights
from .neural_microstep_v58 import DATA_OPS, MicrostepWeights
from .neural_operand_storage_v55 import ACTIONS, MAX_STACK, OperandCircuitError, OperandWeights
from .typed_microop_ir_v57 import EFFECTS, LOCALS_WIDTH, LOCAL_MODES, OPCODES, MicroopAdmissionError
from .wasm_control_flow_v51 import ControlExecutionTrap, ControlFuelExhausted

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
    (NeuralControlError, "Invalid binary condition or noncanonical active PC"),
    (NeuralControlError, "Constructed route lost unique next PC"),
    (ControlExecutionTrap, "Reached terminal before function end"),
    (ControlExecutionTrap, "Final source WASM end needs exactly one i32"),
    (ControlFuelExhausted, "Explicit bounded fuel exhausted after {fuel} v58 microsteps"),
    (RuntimeError, "Native trace exceeded its straight-line capacity"),
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
        self._row_count = len(table)
        self._rows = np.frombuffer(np.stack(table).tobytes(), dtype=np.float64)
        kernels = compiled_alu()
        self._gate = kernels.data_microstep
        self._loop = kernels.prepared_loop
        self._batch = kernels.prepared_batch
        self._pointers = self._rows.ctypes.data, self._fixed.ctypes.data

    def apply(self, pc, buffers):
        status = self._gate(pc, MAX_STACK, LOCALS_WIDTH, *self._pointers, *buffers)
        if status:
            error, message = _ERRORS[status]
            raise error(message)

    def bind_loop(self, program, router):
        if (MAX_STACK, LOCALS_WIDTH) != (24, 5):
            raise RuntimeError("Incompatible v58 native loop state layout")
        opcodes = tuple(op.opcode for op in program.operations)
        visible_locals = program.local_count + 1
        if (len(opcodes) != self._row_count + 1 or opcodes[-1] != "end"
                or not 1 <= visible_locals <= LOCALS_WIDTH):
            raise MicroopAdmissionError("Invalid prepared loop metadata")
        self._opcodes = opcodes
        self._visible_locals = visible_locals
        self._router = router  # Owns the buffers borrowed by the native loop.

    def run_loop(self, argument, fuel, capture_trace):
        _validate_i32(argument)
        count = len(self._opcodes)
        visible_locals = self._visible_locals
        result = (ctypes.c_int32 * 2)()
        if capture_trace:
            capacity = min(fuel, count)
            states = np.zeros((capacity, 2 + MAX_STACK + visible_locals), dtype=np.int32)
            routes = np.empty((capacity, 5), dtype=np.float64)
            state_pointer, route_pointer = states.ctypes.data, routes.ctypes.data
        else:
            state_pointer = route_pointer = None
        status = self._loop(
            argument, fuel, count, TERMINAL, visible_locals,
            *self._pointers, *self._router._pointers, result, state_pointer, route_pointer,
        )
        if status:
            raise _loop_error(status, fuel)

        value, steps = result
        if not capture_trace:
            return value, steps, ()

        return value, steps, self._trace(steps, states, routes)

    def _trace(self, steps, states, routes):
        count = len(self._opcodes)
        records = []
        for tick, (state, route) in enumerate(zip(
            states[:steps].tolist(), routes[:steps].tolist(), strict=True,
        )):
            pc, depth = state[:2]
            next_pc, details = CompactRouter._read_route(0, route, True)
            records.append({
                "step": tick,
                "pc": pc,
                "opcode": self._opcodes[pc],
                "next_pc": count if next_pc == TERMINAL else next_pc,
                "stack_i32": state[2:2 + depth],
                "locals_i32": state[2 + MAX_STACK:],
                "control_labels": [],
                "halted": pc == count - 1,
                "neural_route": details,
            })
        return tuple(records)

    def run_many(self, arguments, fuels, capture_trace):
        """Yield outcomes for admitted inputs after one native batch call."""
        size = len(arguments)
        arguments = np.asarray(arguments, dtype=np.int32)
        limits = np.asarray(fuels, dtype=np.int32)
        results = np.zeros((size, 2), dtype=np.int32)
        statuses = np.empty(size, dtype=np.int32)
        count = len(self._opcodes)
        stride = 0
        state_pointer = route_pointer = None
        if capture_trace:
            stride = min(max(fuels), count)
            states = np.zeros((size, stride, 2 + MAX_STACK + self._visible_locals), dtype=np.int32)
            routes = np.empty((size, stride, 5), dtype=np.float64)
            state_pointer, route_pointer = states.ctypes.data, routes.ctypes.data
        self._batch(
            arguments.ctypes.data, limits.ctypes.data, size,
            count, TERMINAL, self._visible_locals, stride,
            *self._pointers, *self._router._pointers,
            results.ctypes.data, statuses.ctypes.data, state_pointer, route_pointer,
        )
        for i, (status, (value, steps)) in enumerate(zip(statuses.tolist(), results.tolist(), strict=True)):
            if status:
                yield _loop_error(status, fuels[i])
            else:
                yield value, steps, self._trace(steps, states[i], routes[i]) if capture_trace else ()


def _loop_error(status: int, fuel: int) -> Exception:
    error, message = _ERRORS[status]
    return error(message.format(fuel=fuel))
