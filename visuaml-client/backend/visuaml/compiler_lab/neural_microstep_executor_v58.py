"""v58 restricted straight-line WASM executor with one shared data microstep.

The v52 fixed-weight one-hot PC routes real execution. The same v57 typed
microop tokens select v58 constructed operand/local/ALU state transitions.
Only the final WASM function-end structural instruction uses host admission.
Control flow, memory and calls are NOT yet implemented by this executor.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .bit_neural_vm_v49 import (
    _bits, _int, BitCircuitWeights, construct_bit_weights,
)
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
    LOCALS_WIDTH, MicroopWeights, ProgramMicroopTokens,
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


def execute_microstep_control(
    program: ControlProgram, argument: int, *, fuel: int = 400, capture_trace: bool = True,
    use_compiled_alu: bool = False,
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
    if type(use_compiled_alu) is not bool:
        raise MicrostepAdmissionError("use_compiled_alu must be a boolean")
    if type(capture_trace) is not bool:
        raise MicrostepAdmissionError("capture_trace must be a boolean")
    _syntax(program)
    if type(fuel) is not int or not 1 <= fuel <= MAX_FUEL:
        raise MicrostepAdmissionError("Invalid v58 execution fuel")
    if (not program.operations or program.operations[-1].opcode != "end"
            or any(op.opcode in (
                "block", "loop", "if", "else", "br", "br_if", "return",
            ) for op in program.operations[:-1])):
        raise MicrostepAdmissionError(
            "v58 microstep executor admits straight-line WASM only"
        )

    tokens = compile_route_tokens(program) if tokens is None else tokens
    validate_route_tokens(tokens, program)
    program_tokens = (
        compile_microop_tokens(program)
        if program_tokens is None else program_tokens
    )
    validate_microop_tokens(program_tokens, program)
    pc_fixed = construct_router_weights() if pc_fixed is None else pc_fixed
    bit_fixed = construct_bit_weights() if bit_fixed is None else bit_fixed
    microop_fixed = (
        construct_microop_weights() if microop_fixed is None
        else microop_fixed
    )
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
    operands = empty_operands()
    locals_ = np.zeros((LOCALS_WIDTH, 32), dtype=np.float64)
    locals_[0] = _bits(argument)
    pc_vector = _onehot(0)
    count = len(program.operations)
    records = []

    for tick in range(fuel):
        pc = int(np.argmax(pc_vector))
        if pc >= count or pc == TERMINAL:
            raise ControlExecutionTrap("Reached terminal before function end")
        op = program.operations[pc]
        condition = (
            top_word(operands) if operands.height[0] == 0.
            else np.zeros(32, dtype=np.float64)
        )
        selected, route = neural_pc_step(
            pc_vector, condition, tokens, pc_fixed, bit_fixed,
        )
        next_pc = int(np.argmax(selected))
        halted = op.opcode == "end"
        if halted:
            if (pc != count - 1
                    or int(np.argmax(operands.height)) != 1
                    or next_pc != TERMINAL):
                raise ControlExecutionTrap(
                    "Final source WASM end needs exactly one i32"
                )
        else:
            microop = read_microop(
                pc_vector, program_tokens, microop_fixed,
            )
            result = step_data_microop(
                operands, locals_, microop,
                micro_fixed=microstep_fixed, operand_fixed=operand_fixed,
                local_fixed=local_fixed, alu_fixed=alu_fixed,
                bit_fixed=bit_fixed, use_compiled_alu=use_compiled_alu,
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
        pc_vector = selected
        if halted:
            return MicrostepExecution(
                _int(top_word(operands)), tuple(records), tick + 1,
                hashlib.sha256(encode_control(program)).hexdigest(),
                pc_fixed.digest(), alu_fixed.digest(),
                microop_fixed.digest(), _microstep_digest(microstep_fixed),
                program_tokens.digest(),
            )
    raise ControlFuelExhausted(
        f"Explicit bounded fuel exhausted after {fuel} v58 microsteps"
    )
