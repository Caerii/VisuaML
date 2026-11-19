"""v58 restricted straight-line WASM executor with one shared data microstep.

The v52 fixed-weight one-hot PC routes real execution. The same v57 typed
microop tokens select v58 constructed operand/local/ALU state transitions.
Prepared packed execution optionally integrates the bounded v51
structured subset with v52 routing and packed v53-v54 frame gates.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, fields, replace

import numpy as np

from .bit_neural_vm_v49 import (
    _bits, _int, _validate_i32, BitCircuitWeights, construct_bit_weights,
)
from .compact_routing_v58 import CompactRouter
from .native_microstep_v58 import MemoryWeights, _NativeMicrostep
from .neural_frame_circuit_v53 import FrameCircuitWeights
from .neural_frame_events_v54 import EventWeights
from .neural_alu_v56 import ALUWeights, construct_alu_weights
from .neural_control_routing_v52 import (
    TERMINAL, ControlRouteTokens, ControlRouterWeights, _onehot,
    compile_route_tokens, construct_router_weights, neural_pc_step,
    validate_route_tokens,
)
from .neural_local_bank_v57 import (
    LocalBankWeights, construct_local_bank_weights,
)
from .neural_microstep_v58 import (
    MicrostepWeights, construct_microstep_weights, step_data_microop,
)
from .neural_operand_storage_v55 import (
    OperandWeights, empty_operands, fixed_operand_weights, top_word,
)
from .typed_microop_ir_v57 import (
    LOCALS_WIDTH, IndexedMicroops, MicroopWeights, ProgramMicroopTokens, SelectedMicroop,
    compile_microop_tokens, construct_microop_weights, read_microop,
    validate_microop_tokens,
)
from .wasm_control_flow_v51 import (
    MAX_FUEL, MEMORY_BYTES, SCALED_MAX_FUEL, SCALED_SCHEMA, ControlAdmissionError, ControlExecutionTrap,
    ControlFuelExhausted, ControlProgram, _syntax, encode_control, uses_memory,
)


class MicrostepAdmissionError(ControlAdmissionError):
    """Only reviewed, straight-line, original source WASM is admitted."""


@dataclass(frozen=True)
class MicrostepExecution:
    result_i32: int
    trace: tuple[dict, ...]
    steps: int
    source_wasm_sha256: str
    fixed_pc_sha256: str
    fixed_alu_sha256: str
    fixed_microop_sha256: str
    fixed_microstep_sha256: str
    source_program_tokens_sha256: str
    host_structural_end: bool = True
    data_effects_driven_by_microstep: bool = True
    structured_control_sha256: str | None = None
    memory_weights_sha256: str | None = None


@dataclass(frozen=True)
class MicrostepBatchFailure:
    """The error this input would raise in an individual run."""

    error: Exception


def _microstep_digest(weights: MicrostepWeights) -> str:
    h = hashlib.sha256(b"visuaml.v58.fixed-microstep")
    for array in (
        weights.pop_gain, weights.produce_gain, weights.write_gain,
    ):
        h.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class _ExecutionInputs:
    program: ControlProgram
    tokens: ControlRouteTokens | None
    program_tokens: ProgramMicroopTokens | None
    pc_fixed: ControlRouterWeights | None
    bit_fixed: BitCircuitWeights
    microop_fixed: MicroopWeights | None
    microstep_fixed: MicrostepWeights
    alu_fixed: ALUWeights
    local_fixed: LocalBankWeights
    operand_fixed: OperandWeights
    use_compiled_alu: bool
    use_fused_state: bool
    compact: CompactRouter | None
    indexed: IndexedMicroops | None
    rows: tuple[SelectedMicroop, ...] | None = None
    provenance: tuple[str, ...] | None = None
    native_microstep: _NativeMicrostep | None = None
    native_loop: bool = False

    def digests(self) -> tuple[str, ...]:
        assert self.pc_fixed is not None
        return (
            hashlib.sha256(encode_control(self.program)).hexdigest(),
            self.pc_fixed.digest(), self.alu_fixed.digest(),
            self.indexed.weight_digest if self.indexed else self.microop_fixed.digest(),
            _microstep_digest(self.microstep_fixed),
            self.indexed.token_digest if self.indexed else self.program_tokens.digest(),
        )


def _snapshot_weights(weights):
    def snapshot(value):
        if type(value) is np.ndarray and not value.dtype.hasobject:
            return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
        if isinstance(value, np.ndarray):
            raise MicrostepAdmissionError("Unsupported prepared coefficient array")
        return _snapshot_weights(value)

    return replace(weights, **{
        field.name: snapshot(getattr(weights, field.name)) for field in fields(weights)
    })


def _validate_run_options(fuel: int, capture_trace: bool, max_fuel: int = MAX_FUEL) -> None:
    if type(capture_trace) is not bool:
        raise MicrostepAdmissionError("capture_trace must be a boolean")
    _validate_fuel(fuel, MAX_FUEL if capture_trace else max_fuel)


def _validate_fuel(fuel: int, max_fuel: int = MAX_FUEL) -> None:
    if type(fuel) is not int or not 1 <= fuel <= max_fuel:
        raise MicrostepAdmissionError("Invalid v58 execution fuel")


def _prepare(
    program: ControlProgram, *,
    use_compiled_alu: bool = False,
    use_fused_state: bool = False,
    use_compact_routing: bool = False,
    use_indexed_pc: bool = False,
    tokens: ControlRouteTokens | None = None,
    program_tokens: ProgramMicroopTokens | None = None,
    pc_fixed: ControlRouterWeights | None = None,
    bit_fixed: BitCircuitWeights | None = None,
    microop_fixed: MicroopWeights | None = None,
    microstep_fixed: MicrostepWeights | None = None,
    alu_fixed: ALUWeights | None = None,
    local_fixed: LocalBankWeights | None = None,
    operand_fixed: OperandWeights | None = None,
    reusable: bool = False,
    structured_control: bool = False,
) -> _ExecutionInputs:
    if type(use_indexed_pc) is not bool or (use_indexed_pc and not use_compact_routing):
        raise MicrostepAdmissionError("Indexed execution requires compact routing")
    if type(use_compact_routing) is not bool:
        raise MicrostepAdmissionError("use_compact_routing must be a boolean")
    if type(use_compiled_alu) is not bool:
        raise MicrostepAdmissionError("use_compiled_alu must be a boolean")
    if type(use_fused_state) is not bool:
        raise MicrostepAdmissionError("use_fused_state must be a boolean")
    _syntax(program)
    scaled = program.schema == SCALED_SCHEMA
    if scaled and not (structured_control and use_indexed_pc):
        raise MicrostepAdmissionError("Scaled programs require packed structured execution")
    memory = uses_memory(program)
    if memory and not structured_control:
        raise MicrostepAdmissionError("Linear memory requires packed structured control")
    if not structured_control and any(
        op.opcode in ("block", "loop", "if", "else", "br", "br_if", "return")
        for op in program.operations[:-1]
    ):
        raise MicrostepAdmissionError(
            "v58 microstep executor admits straight-line WASM only"
        )

    if not scaled and tokens is None:
        tokens = compile_route_tokens(program)
    if not use_compact_routing:
        validate_route_tokens(tokens, program)
    program_tokens = (
        compile_microop_tokens(program)
        if program_tokens is None else program_tokens
    )
    if not use_indexed_pc:
        validate_microop_tokens(program_tokens, program)
    pc_fixed = construct_router_weights() if pc_fixed is None else pc_fixed
    bit_fixed = construct_bit_weights() if bit_fixed is None else bit_fixed
    microop_fixed = (
        construct_microop_weights(include_memory=memory) if microop_fixed is None
        else microop_fixed
    )
    indexed = IndexedMicroops(program, program_tokens, microop_fixed) if use_indexed_pc else None
    microstep_fixed = (
        construct_microstep_weights() if microstep_fixed is None
        else microstep_fixed
    )
    alu_fixed = construct_alu_weights() if alu_fixed is None else alu_fixed
    local_fixed = (
        construct_local_bank_weights() if local_fixed is None
        else local_fixed
    )
    operand_fixed = (
        fixed_operand_weights() if operand_fixed is None
        else operand_fixed
    )
    if reusable:
        # Snapshot coefficients shared by the router, live data gates, and
        # provenance before admitting the reusable program.
        pc_fixed = _snapshot_weights(pc_fixed)
        bit_fixed = _snapshot_weights(bit_fixed)
        microstep_fixed = _snapshot_weights(microstep_fixed)
        alu_fixed = _snapshot_weights(alu_fixed)
        local_fixed = _snapshot_weights(local_fixed)
        operand_fixed = _snapshot_weights(operand_fixed)
    compact = CompactRouter(program, tokens, pc_fixed, bit_fixed) if use_compact_routing else None
    inputs = _ExecutionInputs(
        program, tokens, program_tokens, pc_fixed, bit_fixed, microop_fixed,
        microstep_fixed, alu_fixed, local_fixed, operand_fixed, use_compiled_alu,
        use_fused_state, compact, indexed,
    )
    if reusable:
        assert indexed is not None
        row_count = len(program.operations) - (not structured_control)
        rows = tuple(indexed.read(pc) for pc in range(row_count))
        return replace(
            inputs, rows=rows, provenance=inputs.digests(),
            tokens=None, program_tokens=None, pc_fixed=None, microop_fixed=None,
        )
    return inputs


def execute_microstep_control(
    program: ControlProgram, argument: int, *, fuel: int = 400, capture_trace: bool = True,
    use_compiled_alu: bool = False,
    use_fused_state: bool = False,
    use_compact_routing: bool = False,
    use_indexed_pc: bool = False,
    tokens: ControlRouteTokens | None = None,
    program_tokens: ProgramMicroopTokens | None = None,
    pc_fixed: ControlRouterWeights | None = None,
    bit_fixed: BitCircuitWeights | None = None,
    microop_fixed: MicroopWeights | None = None,
    microstep_fixed: MicrostepWeights | None = None,
    alu_fixed: ALUWeights | None = None,
    local_fixed: LocalBankWeights | None = None,
    operand_fixed: OperandWeights | None = None,
) -> MicrostepExecution:
    _validate_run_options(fuel, capture_trace)
    inputs = _prepare(
        program, use_compiled_alu=use_compiled_alu, use_fused_state=use_fused_state,
        use_compact_routing=use_compact_routing, use_indexed_pc=use_indexed_pc,
        tokens=tokens, program_tokens=program_tokens, pc_fixed=pc_fixed, bit_fixed=bit_fixed,
        microop_fixed=microop_fixed, microstep_fixed=microstep_fixed, alu_fixed=alu_fixed,
        local_fixed=local_fixed, operand_fixed=operand_fixed,
    )
    return _run(inputs, argument, fuel, capture_trace)


class PreparedMicrostepProgram:
    """Admit and snapshot one straight-line program for repeated v58 execution.

    Preparation validates all data microop rows. Each run starts with fresh
    operands, locals, PC, fuel, and trace. The live gates use the snapshotted
    coefficients, including supported fault-injection overrides. The opt-in
    native microstep binds these same rows and coefficients to one C++ call.
    The native loop also runs live PC routing in that call for the full program.
    The packed scalar mode uses uint32 words for the native loop's binary state.
    Optional AVX2 SIMD processes groups of eight eligible untraced inputs.
    Structured control uses the packed scalar loop and live frame-event gains.
    """

    def __init__(
        self, program: ControlProgram, *,
        use_compiled_alu: bool = True,
        use_fused_state: bool = True,
        use_native_microstep: bool = False,
        use_native_loop: bool = False,
        use_packed_scalar: bool = False,
        use_packed_simd: bool = False,
        use_structured_control: bool = False,
        event_fixed: EventWeights | None = None,
        frame_fixed: FrameCircuitWeights | None = None,
        memory_fixed: MemoryWeights | None = None,
        tokens: ControlRouteTokens | None = None,
        program_tokens: ProgramMicroopTokens | None = None,
        pc_fixed: ControlRouterWeights | None = None,
        bit_fixed: BitCircuitWeights | None = None,
        microop_fixed: MicroopWeights | None = None,
        microstep_fixed: MicrostepWeights | None = None,
        alu_fixed: ALUWeights | None = None,
        local_fixed: LocalBankWeights | None = None,
        operand_fixed: OperandWeights | None = None,
    ):
        if type(use_structured_control) is not bool:
            raise MicrostepAdmissionError("use_structured_control must be a boolean")
        if use_structured_control and (not use_packed_scalar or not use_native_loop or use_packed_simd):
            raise MicrostepAdmissionError("Structured control requires the scalar packed native loop")
        if not use_structured_control and (event_fixed is not None or frame_fixed is not None):
            raise MicrostepAdmissionError("Frame coefficients require structured control")
        if type(use_packed_simd) is not bool:
            raise MicrostepAdmissionError("use_packed_simd must be a boolean")
        if use_packed_simd and not use_packed_scalar:
            raise MicrostepAdmissionError("Packed SIMD requires use_packed_scalar=True")
        if type(use_packed_scalar) is not bool:
            raise MicrostepAdmissionError("use_packed_scalar must be a boolean")
        if use_packed_scalar and not use_native_loop:
            raise MicrostepAdmissionError("Packed scalar execution requires use_native_loop=True")
        if type(use_native_loop) is not bool:
            raise MicrostepAdmissionError("use_native_loop must be a boolean")
        if type(use_native_microstep) is not bool:
            raise MicrostepAdmissionError("use_native_microstep must be a boolean")
        if (use_native_microstep or use_native_loop) and (not use_compiled_alu or not use_fused_state):
            raise MicrostepAdmissionError("Native microstep requires compiled ALU and fused state")
        inputs = _prepare(
            program, use_compiled_alu=use_compiled_alu, use_fused_state=use_fused_state,
            use_compact_routing=True, use_indexed_pc=True,
            tokens=tokens, program_tokens=program_tokens, pc_fixed=pc_fixed, bit_fixed=bit_fixed,
            microop_fixed=microop_fixed, microstep_fixed=microstep_fixed, alu_fixed=alu_fixed,
            local_fixed=local_fixed, operand_fixed=operand_fixed, reusable=True, structured_control=use_structured_control,
        )

        if memory_fixed is not None and not uses_memory(inputs.program):
            raise MicrostepAdmissionError("Memory coefficients require a memory instruction")
        if use_native_microstep or use_native_loop:
            native = _NativeMicrostep(
                inputs.rows, inputs.microstep_fixed, inputs.operand_fixed,
                inputs.alu_fixed, inputs.bit_fixed, inputs.local_fixed,
                packed=use_packed_scalar, simd=use_packed_simd, structured=use_structured_control,
            )
            if use_native_loop:
                native.bind_loop(inputs.program, inputs.compact)
            if use_structured_control:
                native.bind_control(inputs.program, inputs.rows, event_fixed, frame_fixed, memory_fixed)
            inputs = replace(inputs, native_microstep=native, rows=None, native_loop=use_native_loop)
        self._inputs = inputs
        self._max_fuel = SCALED_MAX_FUEL if program.schema == SCALED_SCHEMA else MAX_FUEL

    def run(
        self, argument: int, *, fuel: int = 400, capture_trace: bool = True,
    ) -> MicrostepExecution:
        _validate_run_options(fuel, capture_trace, self._max_fuel)
        if self._inputs.native_microstep and self._inputs.native_microstep.memory_digest:
            raise MicrostepAdmissionError("Memory programs require run_memory")
        return _run(self._inputs, argument, fuel, capture_trace)

    def run_memory(
        self, argument: int, memory: bytearray, *, fuel: int = 400, capture_trace: bool = True,
    ) -> MicrostepExecution:
        """Execute against one page, committing completed stores even when a later instruction traps."""
        _validate_run_options(fuel, capture_trace, self._max_fuel)
        native = self._inputs.native_microstep
        if native is None or native.memory_digest is None:
            raise MicrostepAdmissionError("run_memory requires a prepared memory program")
        _validate_i32(argument)
        if type(memory) is not bytearray or len(memory) != MEMORY_BYTES:
            raise MicrostepAdmissionError("Expected a bytearray with one WASM memory page")
        page = bytearray(memory)
        if len(page) != MEMORY_BYTES:
            raise MicrostepAdmissionError("Expected a bytearray with one WASM memory page")
        try:
            value, steps, trace = native.run_memory(argument, fuel, capture_trace, page)
        finally:
            memory[:] = page
        return MicrostepExecution(
            value, trace, steps, *self._inputs.provenance,
            structured_control_sha256=native.control_digest, memory_weights_sha256=native.memory_digest,
        )

    def run_many(
        self, arguments: Iterable[int], *, fuel: int | Iterable[int] = 400,
        capture_trace: bool = True,
    ) -> tuple[MicrostepExecution | MicrostepBatchFailure, ...]:
        """Run independent inputs in order with one native call for admitted entries.

        Fuel can be shared or supplied once per input. Admission failures and
        execution traps are returned in their input positions. The prepared
        program must have been created with native loop execution enabled.
        """
        inputs = self._inputs
        if not inputs.native_loop:
            raise MicrostepAdmissionError("Batch execution requires use_native_loop=True")
        if inputs.native_microstep.memory_digest:
            raise MicrostepAdmissionError("Memory programs require run_memory")
        if type(capture_trace) is not bool:
            raise MicrostepAdmissionError("capture_trace must be a boolean")
        arguments = tuple(arguments)
        if type(fuel) is int:
            fuels = (fuel,) * len(arguments)
        else:
            try:
                fuels = tuple(fuel)
            except TypeError:
                raise MicrostepAdmissionError("Expected shared fuel or one fuel value per input") from None
            if len(fuels) != len(arguments):
                raise MicrostepAdmissionError("Expected one fuel value per input")

        admitted, limits, failures = [], [], {}
        for i, (argument, limit) in enumerate(zip(arguments, fuels, strict=True)):
            try:
                _validate_fuel(limit, MAX_FUEL if capture_trace else self._max_fuel)
                _validate_i32(argument)
            except ValueError as error:
                failures[i] = MicrostepBatchFailure(error.with_traceback(None))
            else:
                admitted.append(argument)
                limits.append(limit)

        results = iter(inputs.native_microstep.run_many(admitted, limits, capture_trace)) if admitted else iter(())
        outcomes = []
        for i in range(len(arguments)):
            if i in failures:
                outcomes.append(failures[i])
                continue
            result = next(results)
            if isinstance(result, Exception):
                outcomes.append(MicrostepBatchFailure(result))
            else:
                value, steps, trace = result
                outcomes.append(MicrostepExecution(
                    value, trace, steps, *inputs.provenance,
                    structured_control_sha256=inputs.native_microstep.control_digest,
                ))
        return tuple(outcomes)


def _run(inputs: _ExecutionInputs, argument: int, fuel: int, capture_trace: bool) -> MicrostepExecution:
    program = inputs.program
    compact, indexed, rows = inputs.compact, inputs.indexed, inputs.rows
    native = inputs.native_microstep
    if inputs.native_loop:
        value, steps, trace = native.run_loop(argument, fuel, capture_trace)
        return MicrostepExecution(
            value, trace, steps, *inputs.provenance, structured_control_sha256=native.control_digest,
        )
    operands = empty_operands()
    locals_ = np.zeros((LOCALS_WIDTH, 32), dtype=np.float64)
    locals_[0] = _bits(argument)
    if native is not None:
        buffers = (operands.height.ctypes.data, operands.cells.ctypes.data, locals_.ctypes.data)
    pc_vector = None if indexed is not None else _onehot(0)
    pc = 0
    count = len(program.operations)
    records = []

    for tick in range(fuel):
        if indexed is None:
            pc = int(np.argmax(pc_vector))
        if pc >= count or pc == TERMINAL:
            raise ControlExecutionTrap("Reached terminal before function end")
        op = program.operations[pc]
        condition = (
            top_word(operands) if operands.height[0] == 0.
            else np.zeros(32, dtype=np.float64)
        )
        if indexed is not None:
            next_pc, route = compact.step_index(pc, condition, capture_trace=capture_trace)
        else:
            if compact is None:
                selected, route = neural_pc_step(
                    pc_vector, condition, inputs.tokens, inputs.pc_fixed, inputs.bit_fixed,
                )
            else:
                selected, route = compact.step(pc_vector, condition, capture_trace=capture_trace)
            next_pc = int(np.argmax(selected))
        halted = op.opcode == "end"
        if halted:
            if (pc != count - 1
                    or int(np.argmax(operands.height)) != 1
                    or next_pc != TERMINAL):
                raise ControlExecutionTrap(
                    "Final source WASM end needs exactly one i32"
                )
        elif native is not None:
            native.apply(pc, buffers)
        else:
            if rows is not None:
                microop = rows[pc]
            elif indexed is not None:
                microop = indexed.read(pc)
            else:
                microop = read_microop(pc_vector, inputs.program_tokens, inputs.microop_fixed)
            result = step_data_microop(
                operands, locals_, microop,
                micro_fixed=inputs.microstep_fixed, operand_fixed=inputs.operand_fixed,
                local_fixed=inputs.local_fixed, alu_fixed=inputs.alu_fixed,
                bit_fixed=inputs.bit_fixed, use_compiled_alu=inputs.use_compiled_alu,
                use_fused_state=inputs.use_fused_state,
            )
            operands, locals_ = result.operands, result.locals

        if capture_trace:
            depth = int(np.argmax(operands.height))
            records.append({
                "step": tick,
                "pc": pc,
                "opcode": op.opcode,
                "next_pc": count if next_pc == TERMINAL else next_pc,
                "stack_i32": [_int(operands.cells[i]) for i in range(depth)],
                "locals_i32": [
                    _int(value) for value in locals_[:program.local_count + 1]
                ],
                "control_labels": [],
                "halted": halted,
                "neural_route": route,
            })
        if indexed is None:
            pc_vector = selected
        else:
            pc = next_pc
        if halted:
            return MicrostepExecution(
                _int(top_word(operands)), tuple(records), tick + 1,
                *(inputs.provenance if inputs.provenance is not None else inputs.digests()),
            )
    raise ControlFuelExhausted(
        f"Explicit bounded fuel exhausted after {fuel} v58 microsteps"
    )
