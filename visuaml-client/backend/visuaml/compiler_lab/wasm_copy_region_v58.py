"""Source-bound lowering of a bounded i32 word-move region."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from hashlib import sha256

from .bit_neural_vm_v49 import _validate_i32
from .neural_microstep_executor_v58 import (
    MicrostepAdmissionError, PreparedMicrostepProgram, _validate_run_options,
)
from .wasm_control_flow_v51 import MEMORY_BYTES, ControlOperation, ControlProgram, MemoryArgument, encode_control


def copy_program() -> ControlProgram:
    """run(count) moves words using the source and destination stored at bytes 0 and 4."""
    def op(name, value=None):
        return ControlOperation(name, value)

    load = op("i32.load", MemoryArgument())
    store = op("i32.store", MemoryArgument())

    def address(local):
        return op("local.get", local), op("local.get", 3), op("i32.const", 4), op("i32.mul"), op("i32.add")

    transfer = (*address(2), *address(1), load, store)
    countdown = (
        op("local.get", 0), op("local.set", 3), op("block"), op("loop"),
        op("local.get", 3), op("i32.eqz"), op("br_if", 1),
        op("local.get", 3), op("i32.const", 1), op("i32.sub"), op("local.set", 3),
        *transfer, op("br", 0), op("end"), op("end"),
    )
    countup = (
        op("block"), op("loop"), op("local.get", 3), op("local.get", 0), op("i32.lt_s"),
        op("i32.eqz"), op("br_if", 1), *transfer,
        op("local.get", 3), op("i32.const", 1), op("i32.add"), op("local.set", 3),
        op("br", 0), op("end"), op("end"),
    )
    return ControlProgram(3, (
        op("local.get", 0), op("i32.const", 0), op("i32.gt_s"), op("if"),
        op("i32.const", 0), load, op("local.set", 1),
        op("i32.const", 4), load, op("local.set", 2),
        op("local.get", 1), op("local.get", 2), op("i32.lt_s"), op("if"),
        *countdown, op("else"), *countup, op("end"), op("end"), op("local.get", 0), op("end"),
    ))


@dataclass(frozen=True)
class CopyRegionResult:
    result_i32: int
    steps: int
    trace: tuple[dict, ...]
    fused: bool


class PreparedCopyRegion:
    """The exact word-move fixture with a guarded bulk lowering and source replay."""

    def __init__(self, program: ControlProgram, **overrides):
        raw = encode_control(program)
        if raw != encode_control(copy_program()):
            raise MicrostepAdmissionError("No certified copy-region lowering for this WASM source")
        self.source_wasm_sha256 = sha256(raw).hexdigest()
        self._specialize = not overrides
        self._reference = PreparedMicrostepProgram(
            program, use_native_loop=True, use_packed_scalar=True, use_structured_control=True, **overrides,
        )

    def run_memory(self, argument: int, page: bytearray, *, fuel=400, capture_trace=True) -> CopyRegionResult:
        _validate_run_options(fuel, capture_trace)
        _validate_i32(argument)
        if type(page) is not bytearray or len(page) != MEMORY_BYTES:
            raise MicrostepAdmissionError("Expected a bytearray with one WASM memory page")
        if self._specialize and not capture_trace:
            if argument <= 0 and fuel >= 6:
                return CopyRegionResult(argument, 6, (), True)
            if argument > 0:
                # Hold the buffer length fixed while reading the descriptor and
                # moving bytes. The slice expression snapshots overlapping input.
                with memoryview(page) as memory:
                    if len(memory) != MEMORY_BYTES:
                        raise MicrostepAdmissionError("Expected a bytearray with one WASM memory page")
                    src, dst = struct.unpack_from("<II", memory)
                    length = 4 * argument
                    # Both positive paths use 25 setup/exit steps.
                    # Backward iterations cost 20 steps, forward iterations 22.
                    steps = 25 + argument * (20 if src < dst else 22)
                    if (steps <= fuel and src + length <= MEMORY_BYTES and dst + length <= MEMORY_BYTES):
                        memory[dst:dst + length] = memory[src:src + length].tobytes()
                        return CopyRegionResult(argument, steps, (), True)
        reference = self._reference.run_memory(argument, page, fuel=fuel, capture_trace=capture_trace)
        return CopyRegionResult(reference.result_i32, reference.steps, reference.trace, False)
