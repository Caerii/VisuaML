"""Bind admitted v58 data rows to the native fused data gate."""
from __future__ import annotations

import ctypes
import hashlib
from dataclasses import dataclass

import numpy as np

from .bit_neural_vm_v49 import WIDTH, _validate_i32
from .compact_routing_v58 import CompactRouter
from .constructed_alu_v58 import compiled_alu, compiled_coefficients
from .exact_binary import is_exact_binary
from .neural_frame_circuit_v53 import FrameCircuitError, FrameCircuitWeights, fixed_frame_weights
from .neural_frame_events_v54 import ACTIONS as FRAME_ACTIONS
from .neural_frame_events_v54 import OPCODES as FRAME_OPCODES
from .neural_frame_events_v54 import EventCircuitError, EventWeights, fixed_event_weights
from .neural_alu_v56 import ALU_OPS, ALUWeights, NeuralALUError
from .neural_control_routing_v52 import NeuralControlError
from .neural_local_bank_v57 import LocalBankWeights
from .neural_microstep_v58 import DATA_OPS, MicrostepWeights
from .neural_operand_storage_v55 import ACTIONS, MAX_STACK, OperandCircuitError, OperandWeights
from .typed_microop_ir_v57 import EFFECTS, LOCALS_WIDTH, LOCAL_MODES, OPCODES, MicroopAdmissionError
from .wasm_control_flow_v51 import (
    MAX_NESTING, MEMORY_OPCODES, ControlExecutionTrap, ControlFuelExhausted,
    LinearMemoryTrap, _syntax, uses_memory,
)

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
    (ControlExecutionTrap, "Decoded stack effect violates source WASM"),
    (ControlExecutionTrap, "Unbalanced or invalid structured control frame"),
    (EventCircuitError, "Noncanonical one-hot frame event"),
    (FrameCircuitError, "Noncanonical one-hot frame output"),
    (ControlExecutionTrap, "Unexpected return stack or target"),
    (LinearMemoryTrap, "Out of bounds i32 linear memory access"),
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


@dataclass(frozen=True)
class MemoryWeights:
    load_gain: int = 1
    store_gain: int = 1


class _NativeMicrostep:
    """Internal gate for owned operand and local buffers from the prepared executor."""

    def __init__(self, rows, micro, operand, alu, bits, local, *, packed=False, simd=False, structured=False):
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
        self._packed = packed
        self._simd = simd
        self.control_digest = None
        self.memory_digest = None
        self._structured = structured
        powers = np.ldexp(np.ones(WIDTH), np.arange(WIDTH)) if packed else None
        table = []
        for row in rows:
            is_data = row.opcode[:len(OPCODES)] @ data
            if is_data and len(row.effect) > len(EFFECTS) and row.effect[len(EFFECTS)] != 0.:
                raise MicroopAdmissionError("Data instruction selected a memory store effect")
            if structured and is_data == 0.:
                table.append(np.zeros(len(EFFECTS) + len(LOCAL_MODES) + len(ALU_OPS) + 1 + LOCALS_WIDTH + 1))
                continue
            if is_data != 1.:
                raise MicroopAdmissionError("Control opcode outside v58 data microstep")
            literal = (row.immediate_bits @ powers,) if packed else row.immediate_bits
            table.append(np.concatenate((
                row.effect[:len(EFFECTS)], row.local_mode, row.alu_selector, literal,
                row.local_selector, row.opcode[const:const + 1],
            )))
        self._row_count = len(table)
        self._rows = np.frombuffer(np.stack(table).tobytes(), dtype=np.float64)
        kernels = compiled_alu()
        self._gate = kernels.data_microstep
        self._loop = kernels.prepared_loop
        self._batch = kernels.prepared_batch
        self._structured_batch = kernels.structured_batch
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
        if (len(opcodes) != self._row_count + (not self._structured) or opcodes[-1] != "end"
                or not 1 <= visible_locals <= LOCALS_WIDTH):
            raise MicroopAdmissionError("Invalid prepared loop metadata")
        self._opcodes = opcodes
        self._visible_locals = visible_locals
        self._terminal = router.terminal
        self._router = router  # Owns the buffers borrowed by the native loop.

    def bind_control(self, program, rows, event_fixed, frame_fixed, memory_fixed):
        opcodes = ("block", "loop", "if", "else", "end", "br", "br_if", "return", "other")
        if (FRAME_OPCODES, FRAME_ACTIONS, MAX_NESTING) != (
            opcodes, ("retain", "push", "pop", "branch", "return"), 12,
        ):
            raise RuntimeError("Incompatible v58 native frame layout")
        event = fixed_event_weights() if event_fixed is None else event_fixed
        frame = fixed_frame_weights() if frame_fixed is None else frame_fixed
        if not isinstance(event, EventWeights):
            raise EventCircuitError("Unrecognized fixed circuit")
        if not isinstance(frame, FrameCircuitWeights):
            raise FrameCircuitError("Unknown weights")

        def snapshot(value, shape, error):
            if (type(value) is not np.ndarray or value.dtype != np.float64 or value.shape != shape):
                raise error("Noncanonical frame coefficients")
            immutable = np.frombuffer(value.tobytes(), dtype=np.float64).reshape(shape)
            if not is_exact_binary(immutable):
                raise error("Noncanonical frame coefficients")
            return immutable

        defaults = snapshot(event.default, (len(opcodes), len(FRAME_ACTIONS)), EventCircuitError)
        conditional = snapshot(event.conditional_gains, (3,), EventCircuitError)
        gains = snapshot(frame.action_weights, (len(FRAME_ACTIONS),), FrameCircuitError)
        if not np.all(defaults.sum(axis=1) == 1.):
            raise EventCircuitError("Corrupt opcode action matrix")
        links = _syntax(program)
        flow = []
        names = (*opcodes, *MEMORY_OPCODES)
        if uses_memory(program):
            memory = MemoryWeights() if memory_fixed is None else memory_fixed
            if not isinstance(memory, MemoryWeights):
                raise MicroopAdmissionError("Invalid fixed memory coefficients")
            pair = memory.load_gain, memory.store_gain
            if any(type(bit) is not int or bit not in (0, 1) for bit in pair):
                raise MicroopAdmissionError("Expected binary memory gains")
            self._memory_gains = np.frombuffer(bytes(pair), dtype=np.uint8)
            self.memory_digest = hashlib.sha256(b"visuaml.v58.memory" + bytes(pair)).hexdigest()
        powers = np.ldexp(np.ones(WIDTH), np.arange(WIDTH))
        for pc, (op, row) in enumerate(zip(program.operations, rows, strict=True)):
            code = names.index(op.opcode) if op.opcode in names else opcodes.index("other")
            end, otherwise = links.get(pc, (-1, None))
            target = int(row.branch_depth.argmax()) if op.opcode in ("br", "br_if") else -1
            pops = int(row.effect[2] + row.effect[3] + 2 * row.effect[4])
            offset = 0
            if op.opcode in MEMORY_OPCODES:
                if op.opcode == "i32.load" and row.effect[3] != 1.:
                    raise MicroopAdmissionError("Load requires the admitted unary effect")
                if op.opcode == "i32.store":
                    pops = 2 * int(row.effect[5])
                offset = int(row.immediate_bits @ powers)
            flow.append((code, end, -1 if otherwise is None else otherwise, target, pops, offset))
        self._control_arrays = (
            np.frombuffer(np.array(flow, dtype=np.int64).tobytes(), dtype=np.int64),
            defaults, conditional, gains,
        )
        self._control_pointers = tuple(array.ctypes.data for array in self._control_arrays)
        self._links = links
        h = hashlib.sha256(b"visuaml.v58.packed.frame-gates")
        for array in self._control_arrays[1:]:
            h.update(array.astype("<f8", copy=False).tobytes())
        self.control_digest = h.hexdigest()

    def run_loop(self, argument, fuel, capture_trace):
        _validate_i32(argument)
        if self._structured:
            result = next(self.run_many((argument,), (fuel,), capture_trace))
            if isinstance(result, Exception):
                raise result
            return result
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
            argument, fuel, count, self._terminal, visible_locals,
            *self._pointers, *self._router._pointers, result, state_pointer, route_pointer, self._packed,
        )
        if status:
            raise _loop_error(status, fuel)

        value, steps = result
        if not capture_trace:
            return value, steps, ()

        return value, steps, self._trace(steps, states, routes)

    def run_memory(self, argument, fuel, capture_trace, page):
        result = next(self.run_many((argument,), (fuel,), capture_trace, memory=page))
        if isinstance(result, Exception):
            raise result
        return result

    def _trace(self, steps, states, routes, frames=None, memory=None):
        count = len(self._opcodes)
        records = []
        accesses = memory[:steps].tolist() if memory is not None else None
        frame_rows = frames[:steps].tolist() if frames is not None else None
        for tick, (state, route) in enumerate(zip(
            states[:steps].tolist(), routes[:steps].tolist(), strict=True,
        )):
            pc, depth = state[:2]
            next_pc, details = CompactRouter._read_route(0, route, True)
            record = {
                "step": tick,
                "pc": pc,
                "opcode": self._opcodes[pc],
                "next_pc": count if next_pc == self._terminal else next_pc,
                "stack_i32": state[2:2 + depth],
                "locals_i32": state[2 + MAX_STACK:],
                "control_labels": [],
                "halted": pc == count - 1,
                "neural_route": details,
            }
            if frame_rows is not None:
                frame = frame_rows[tick]
                active = []
                for i in range(frame[1]):
                    entry, base = frame[2 + 2 * i:4 + 2 * i]
                    end, otherwise = self._links[entry]
                    active.append({"kind": self._opcodes[entry], "start_pc": entry, "end_pc": end,
                                   "else_pc": otherwise, "stack_base": base})
                record["control_labels"] = [item["kind"] for item in active]
                record["control_frames"] = active
                record["neural_frame_event"] = frame[0]
                record["halted"] = next_pc == self._terminal and tick == steps - 1
            if accesses is not None:
                access = None
                if self._opcodes[pc] in MEMORY_OPCODES:
                    address, before, after = accesses[tick]
                    access = {"kind": self._opcodes[pc], "address": address,
                              "before_hex": before.to_bytes(4, "little").hex(),
                              "after_hex": after.to_bytes(4, "little").hex()}
                record["memory_access"] = access
            records.append(record)
        return tuple(records)

    def run_many(self, arguments, fuels, capture_trace, *, memory=None):
        """Yield outcomes for admitted inputs after one native batch call."""
        size = len(arguments)
        arguments = np.asarray(arguments, dtype=np.int32)
        limits = np.asarray(fuels, dtype=np.int32)
        results = np.zeros((size, 2), dtype=np.int32)
        statuses = np.empty(size, dtype=np.int32)
        count = len(self._opcodes)
        stride = 0
        state_pointer = route_pointer = frame_pointer = None
        frames = memory_trace = trap = None
        memory_pointer = memory_gains_pointer = access_pointer = trap_pointer = None
        if memory is not None:
            memory_buffer = np.frombuffer(memory, dtype=np.uint8)
            memory_pointer = memory_buffer.ctypes.data
            memory_gains_pointer = self._memory_gains.ctypes.data
        if capture_trace:
            stride = max(fuels) if self._structured else min(max(fuels), count)
            states = np.zeros((size, stride, 2 + MAX_STACK + self._visible_locals), dtype=np.int32)
            routes = np.empty((size, stride, 5), dtype=np.float64)
            state_pointer, route_pointer = states.ctypes.data, routes.ctypes.data
            if self._structured:
                frames = np.zeros((size, stride, 2 + 2 * MAX_NESTING), dtype=np.int32)
                frame_pointer = frames.ctypes.data
            if memory is not None:
                memory_trace = np.zeros((stride, 3), dtype=np.uint32)
                access_pointer = memory_trace.ctypes.data
                trap = np.full(3, -1, dtype=np.int64)
                trap_pointer = trap.ctypes.data
        common = (
            arguments.ctypes.data, limits.ctypes.data, size,
            count, self._terminal, self._visible_locals, stride,
            *self._pointers, *self._router._pointers,
        )
        outputs = (results.ctypes.data, statuses.ctypes.data, state_pointer, route_pointer)
        if self._structured:
            self._structured_batch(
                *common, *self._control_pointers, *outputs, frame_pointer,
                memory_pointer, memory_gains_pointer, access_pointer, trap_pointer,
            )
        else:
            self._batch(*common, *outputs, self._packed, self._simd)
        for i, (status, (value, steps)) in enumerate(zip(statuses.tolist(), results.tolist(), strict=True)):
            if status:
                error = _loop_error(status, fuels[i])
                if trap is not None and isinstance(error, (LinearMemoryTrap, ControlFuelExhausted)):
                    tick, pc, address = trap.tolist()
                    error.trace = self._trace(tick, states[i], routes[i], frames[i], memory_trace)
                    error.trap_site = {
                        "step": tick, "pc": pc,
                        "opcode": self._opcodes[pc] if isinstance(error, LinearMemoryTrap) else None,
                        "address": address if address >= 0 else None,
                    }
                yield error
            else:
                trace = ()
                if capture_trace:
                    trace = self._trace(
                        steps, states[i], routes[i], frames[i] if frames is not None else None, memory_trace,
                    )
                yield value, steps, trace


def _loop_error(status: int, fuel: int) -> Exception:
    error, message = _ERRORS[status]
    return error(message.format(fuel=fuel))
