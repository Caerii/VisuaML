"""v58 restricted straight-line WASM executor with one shared data microstep.

The v52 fixed-weight one-hot PC routes real execution. The same v57 typed
microop tokens select v58 constructed operand/local/ALU state transitions.
Only the final WASM function-end structural instruction uses host admission.
Control flow, memory and calls are NOT yet implemented by this executor.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields, replace

import numpy as np

from .bit_neural_vm_v49 import (
    _bits, _int, BitCircuitWeights, construct_bit_weights,
)
from .compact_routing_v58 import CompactRouter
from .native_microstep_v58 import _NativeMicrostep
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
    MAX_FUEL, ControlAdmissionError, ControlExecutionTrap,
    ControlFuelExhausted, ControlProgram, _syntax, encode_control,
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


def _validate_run_options(fuel: int, capture_trace: bool) -> None:
    if type(capture_trace) is not bool:
        raise MicrostepAdmissionError("capture_trace must be a boolean")
    if type(fuel) is not int or not 1 <= fuel <= MAX_FUEL:
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
    if (not program.operations or program.operations[-1].opcode != "end"
            or any(op.opcode in (
                "block", "loop", "if", "else", "br", "br_if", "return",
            ) for op in program.operations[:-1])):
        raise MicrostepAdmissionError(
            "v58 microstep executor admits straight-line WASM only"
        )

    tokens = compile_route_tokens(program) if tokens is None else tokens
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
        construct_microop_weights() if microop_fixed is None
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
        rows = tuple(indexed.read(pc) for pc in range(len(program.operations) - 1))
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
    """

    def __init__(
        self, program: ControlProgram, *,
        use_compiled_alu: bool = True,
        use_fused_state: bool = True,
        use_native_microstep: bool = False,
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
        if type(use_native_microstep) is not bool:
            raise MicrostepAdmissionError("use_native_microstep must be a boolean")
        if use_native_microstep and (not use_compiled_alu or not use_fused_state):
            raise MicrostepAdmissionError("Native microstep requires compiled ALU and fused state")
        inputs = _prepare(
            program, use_compiled_alu=use_compiled_alu, use_fused_state=use_fused_state,
            use_compact_routing=True, use_indexed_pc=True,
            tokens=tokens, program_tokens=program_tokens, pc_fixed=pc_fixed, bit_fixed=bit_fixed,
            microop_fixed=microop_fixed, microstep_fixed=microstep_fixed, alu_fixed=alu_fixed,
            local_fixed=local_fixed, operand_fixed=operand_fixed, reusable=True,
        )

        if use_native_microstep:
            native = _NativeMicrostep(
                inputs.rows, inputs.microstep_fixed, inputs.operand_fixed,
                inputs.alu_fixed, inputs.bit_fixed, inputs.local_fixed,
            )
            inputs = replace(inputs, native_microstep=native, rows=None)
        self._inputs = inputs

    def run(
        self, argument: int, *, fuel: int = 400, capture_trace: bool = True,
    ) -> MicrostepExecution:
        _validate_run_options(fuel, capture_trace)
        return _run(self._inputs, argument, fuel, capture_trace)


def _run(inputs: _ExecutionInputs, argument: int, fuel: int, capture_trace: bool) -> MicrostepExecution:
    program = inputs.program
    compact, indexed, rows = inputs.compact, inputs.indexed, inputs.rows
    operands = empty_operands()
    locals_ = np.zeros((LOCALS_WIDTH, 32), dtype=np.float64)
    locals_[0] = _bits(argument)
    native = inputs.native_microstep
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
