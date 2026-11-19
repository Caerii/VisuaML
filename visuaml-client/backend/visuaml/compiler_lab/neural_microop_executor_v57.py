"""v57 source-bound microoperation-driven WASM execution path.

Constructed v52 one-hot PC routing remains authoritative. The v53 finite
frame-circuit output now determines live label depth and actual branch
operand-stack restoration. Host still decodes/dispatches opcode effects and maintains lexical source
metadata, but all live operands reside in v55 tensor storage. Not universal WASM or a fully neural instruction decoder.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .bit_neural_vm_v49 import (
    BitCircuitWeights,
    _bits,
    _eqz,
    _int,
    construct_bit_weights,
)
from .compiled_frame_event_v58 import CompiledFrameEvents, _CompiledFrameRetain
from .neural_alu_v56 import (
    ALUWeights,
    construct_alu_weights,
    execute_alu,
    _prepare_selected_alu,
    _PREPARED_CIRCUITS,
)
from .neural_control_routing_v52 import (
    TERMINAL,
    ControlRouterWeights,
    ControlRouteTokens,
    NeuralControlError,
    _onehot,
    _validate_router_weights,
    compile_route_coordinates,
    compile_route_tokens,
    construct_router_weights,
    neural_pc_step,
    validate_route_tokens,
)
from .neural_frame_circuit_v53 import (
    HEIGHTS,
    KINDS,
    MAX_NESTING,
    FrameCircuitWeights,
    FrameTokens,
    empty_frames,
    fixed_frame_weights,
    step_frames,
)
from .neural_frame_circuit_v53 import onehot as frame_onehot
from .neural_frame_events_v54 import (
    EventWeights,
    decode_opcode_token,
    fixed_event_weights,
    select_frame_action,
)
from .neural_local_bank_v57 import (
    LocalBankWeights,
    apply_local_microop,
    construct_local_bank_weights,
    read_local,
)
from .neural_operand_adapter_v55 import TensorOperandStack, _PreparedPopStack
from .neural_operand_storage_v55 import OperandState, OperandWeights
from .typed_microop_ir_v57 import (
    LOCALS_WIDTH,
    IndexedMicroops,
    MicroopWeights,
    ProgramMicroopTokens,
    compile_microop_tokens,
    construct_microop_weights,
    read_microop,
    validate_microop_tokens,
)
from .wasm_control_flow_v51 import (
    _BINARY,
    MAX_FUEL,
    MEMORY_BYTES,
    MEMORY_OPCODES,
    SCALED_MAX_FUEL,
    SCALED_SCHEMA,
    LinearMemoryTrap,
    MAX_STACK,
    ControlAdmissionError,
    ControlExecutionTrap,
    ControlFuelExhausted,
    ControlProgram,
    _syntax,
    encode_control, uses_memory,
)


@dataclass(frozen=True)
class FrameRoutedResult:
    result_i32: int
    trace: tuple[dict, ...]
    steps: int
    source_wasm_sha256: str
    fixed_router_weights_sha256: str
    fixed_arithmetic_weights_sha256: str
    program_route_tokens_sha256: str
    fixed_frame_weights: tuple[float, ...]
    pc_routed_by_constructed_gates: bool = True
    label_depth_driven_by_constructed_gates: bool = True
    stack_restoration_driven_by_constructed_gates: bool = True
    instruction_dispatch_host_interpreted: bool = True
    event_selection_gate_driven: bool = True
    tensor_operand_storage_drives_execution: bool = True
    neural_alu_selection_drives_execution: bool = True
    microop_token_selection_drives_execution: bool = True
    fixed_microop_weights_sha256: str = ""
    fixed_alu_weights_sha256: str = ""
    fixed_event_weight_digest: str = ""
    backend: str = "v57-structured-numpy"
    compiled_frame_event_sha256: str | None = None
    compiled_frame_retain_mode: str | None = None


def _event_digest(weights: EventWeights) -> str:
    h = hashlib.sha256(b"visuaml.v54.fixed-event-gates")
    for value in (weights.default, weights.conditional_gains):
        h.update(np.ascontiguousarray(value, dtype="<f8").tobytes())
    return h.hexdigest()


def _frame_optional(index: int | None, width: int) -> np.ndarray:
    return (np.zeros(width, dtype=np.float64)
            if index is None else frame_onehot(index, width))


def _pop(stack: list[np.ndarray]) -> np.ndarray:
    if not stack:
        raise ControlExecutionTrap("Typed operand stack underflow")
    return stack.pop()


def _array_route(targets, flags, condition, fixed, bit_fixed):
    # Contract the same v52 gate over only its two source-bound coordinates.
    normal, alternate = map(int, targets)
    zero = _eqz(condition, bit_fixed)
    nonzero = 1. - zero
    take = (flags * fixed.guard_weights) @ np.array([1., nonzero, zero])
    gains = fixed.path_weights * np.array([1. - take, take])
    if normal == alternate:
        gains = np.array([gains.sum(), 0.])
    if not np.array_equal(np.sort(gains), [0., 1.]):
        raise NeuralControlError("Constructed route lost unique next PC")
    next_pc = (normal, alternate)[int(np.argmax(gains))]
    return next_pc, {
        "nonzero_condition_gate": int(nonzero), "branch_guard_gate": int(take),
        "sequential_target": normal, "alternative_target": alternate, "next_pc": next_pc,
    }


def _prepare_const_add_regions(program, targets, guards, indexed, selected_circuits, include_local):
    """Bind single-entry, unconditional const/add pairs and optional local reads."""
    alternate_entries = {
        int(target) for pc, row in enumerate(targets) for target in row if target != pc + 1
    }

    def sequential(pc):
        return (targets[pc] == pc + 1).all() and not guards[pc].any()

    regions = {}
    local_regions = {}
    for pc, op in enumerate(program.operations[:-1]):
        finish = pc + 1
        if (op.opcode != "i32.const" or program.operations[finish].opcode != "i32.add"
                or finish in alternate_entries or finish not in selected_circuits
                or not sequential(pc) or not sequential(finish)):
            continue
        constant = indexed.read(pc)
        addition = indexed.read(finish)
        if int(np.argmax(constant.effect)) != 1 or int(np.argmax(addition.effect)) != 4:
            continue
        regions[pc] = (constant.immediate_bits, selected_circuits[finish])
        start = pc - 1
        if (include_local and start >= 0 and program.operations[start].opcode == "local.get"
                and pc not in alternate_entries and sequential(start)):
            local = indexed.read(start)
            if int(np.argmax(local.effect)) == 1 and int(np.argmax(local.local_mode)) == 1:
                local_regions[start] = (local.local_selector, *regions[pc])
    return regions, local_regions


def _run_const_add_region(stack, depth, region):
    constant, add = region
    state = stack.state
    result = add(state.cells[depth - 1], constant)
    cells = state.cells.copy()
    cells[depth - 1] = result
    # The initial and final heights are identical. Subsequent transitions
    # create new arrays, and no intermediate state is exposed.
    stack.state = OperandState(state.height, cells)


def _run_local_const_add_region(stack, region, local_bank, local_fixed):
    selector, constant, add = region
    value = read_local(local_bank, selector, fixed=local_fixed)
    stack.append(add(value, constant))


def execute_microop_routed_control(
    program: ControlProgram, argument: int, *, fuel: int = 500,
    tokens: ControlRouteTokens | None = None,
    fixed: ControlRouterWeights | None = None,
    bit_fixed: BitCircuitWeights | None = None,
    frame_fixed: FrameCircuitWeights | None = None,
    event_fixed: EventWeights | None = None,
    operand_fixed: OperandWeights | None = None,
    alu_fixed: ALUWeights | None = None,
    microop_fixed: MicroopWeights | None = None,
    microop_tokens: ProgramMicroopTokens | None = None,
    local_fixed: LocalBankWeights | None = None,
    array_memory: bytearray | None = None,
    capture_trace: bool = True,
    observe_store: Callable[[int, int, dict], None] | None = None,
    prepare_operand_pop: bool = True,
    prepare_alu: bool = True,
    prepare_mul: bool = True,
    prepare_regions: bool = True,
    prepare_local_region: bool = True,
    use_compiled_frame_event: bool = False,
    compiled_frame_retain: str = "off",
) -> FrameRoutedResult:
    """Execute with fixed-weight routing and frame operations.

    Passing array_memory opts into indexed NumPy routing and direct page
    updates. Completed stores survive traps. The optional store observer
    receives (step, source PC, access) after each write, even without tracing.
    The default array profile prepares unit pop for its private operand
    state and binds source-admitted add and signed less-than circuits.
    Custom tokens or weights use the checked stack and full-candidate ALU.
    Prepared multiply and untraced const/add regions can be disabled
    independently for differential execution. The opt-in compiled event
    gate is used only by the canonical untraced array profile. Its optional
    copy and resident retain modes keep frame slabs private to the run.

    v53 gate output owns frame depths and branch operand-stack restoration.
    Host still dispatches opcode arithmetic and carries lexical PC metadata.
    """
    if type(prepare_operand_pop) is not bool:
        raise ControlAdmissionError("prepare_operand_pop must be a boolean")
    if type(prepare_alu) is not bool:
        raise ControlAdmissionError("prepare_alu must be a boolean")
    if any(type(flag) is not bool for flag in (prepare_mul, prepare_regions, prepare_local_region)):
        raise ControlAdmissionError("Prepared multiply and region options must be booleans")
    if type(use_compiled_frame_event) is not bool:
        raise ControlAdmissionError("use_compiled_frame_event must be a boolean")
    if type(compiled_frame_retain) is not str or compiled_frame_retain not in ("off", "copy", "resident"):
        raise ControlAdmissionError("Invalid compiled_frame_retain mode")
    links = _syntax(program)
    memory = uses_memory(program)
    array_mode = array_memory is not None
    if memory and not array_mode:
        raise ControlAdmissionError("Linear memory requires explicit array_memory or packed execution")
    if array_mode and (not memory or type(array_memory) is not bytearray
                       or len(array_memory) != MEMORY_BYTES):
        raise ControlAdmissionError("Expected a memory program and one bytearray page")
    if observe_store is not None and (not array_mode or not callable(observe_store)):
        raise ControlAdmissionError("A store observer requires an array memory program")
    if type(capture_trace) is not bool:
        raise ControlAdmissionError("capture_trace must be a boolean")
    scaled = program.schema == SCALED_SCHEMA
    if scaled and not array_mode:
        raise ControlAdmissionError("Scaled NumPy execution requires array_memory")
    max_fuel = SCALED_MAX_FUEL if scaled and not capture_trace else MAX_FUEL
    if type(fuel) is not int or not 1 <= fuel <= max_fuel:
        raise ControlAdmissionError("Invalid execution fuel")
    canonical_array = array_mode and type(program) is ControlProgram and all(
        value is None for value in (tokens, fixed, bit_fixed, frame_fixed, event_fixed,
                                    operand_fixed, alu_fixed, microop_fixed, microop_tokens, local_fixed)
    )
    fixed = construct_router_weights() if fixed is None else fixed
    bit_fixed = construct_bit_weights() if bit_fixed is None else bit_fixed
    if array_mode:
        if tokens is not None:
            raise ControlAdmissionError("Array memory routes derive coordinates from source")
        _validate_router_weights(fixed)
        terminal = len(program.operations)
        targets, guards = compile_route_coordinates(program, terminal)
        route_digest = hashlib.sha256(targets.astype("<i4", copy=False).tobytes() + guards.astype("<f8", copy=False).tobytes()).hexdigest()
        page = np.frombuffer(array_memory, dtype=np.uint8)
    else:
        terminal = TERMINAL
        tokens = compile_route_tokens(program) if tokens is None else tokens
        validate_route_tokens(tokens, program)
        route_digest = tokens.digest()
    microop_tokens = (
        compile_microop_tokens(program) if microop_tokens is None
        else microop_tokens
    )
    microop_fixed = (
        construct_microop_weights(include_memory=memory) if microop_fixed is None
        else microop_fixed
    )
    if array_mode:
        indexed = IndexedMicroops(program, microop_tokens, microop_fixed)
    else:
        validate_microop_tokens(microop_tokens, program)
    local_fixed = (
        construct_local_bank_weights() if local_fixed is None
        else local_fixed
    )
    local_bank = np.zeros((LOCALS_WIDTH, 32), dtype=np.float64)
    local_bank[0] = _bits(argument)
    stack = _PreparedPopStack() if canonical_array and prepare_operand_pop else TensorOperandStack(operand_fixed)
    alu_fixed = construct_alu_weights() if alu_fixed is None else alu_fixed
    selected_circuits = {}
    if canonical_array and prepare_alu:
        for pc, op in enumerate(program.operations):
            if op.opcode in _PREPARED_CIRCUITS and (prepare_mul or op.opcode != "i32.mul"):
                selected_circuits[pc] = _prepare_selected_alu(
                    op.opcode, indexed.read(pc).alu_selector, alu_fixed, bit_fixed,
                )

    regions, local_regions = (
        _prepare_const_add_regions(program, targets, guards, indexed, selected_circuits, prepare_local_region)
        if canonical_array and prepare_alu and prepare_operand_pop and prepare_regions and not capture_trace
        else ({}, {})
    )

    frame_state = empty_frames()
    frame_meta: list[tuple[str, int, int, int | None]] = []
    pc_vector = None if array_mode else _onehot(0)
    pc = 0
    frame_fixed = fixed_frame_weights() if frame_fixed is None else frame_fixed
    event_fixed = fixed_event_weights() if event_fixed is None else event_fixed
    # Memory operations retain the current control frame, as nop does.
    opcode_tokens = tuple(decode_opcode_token("nop" if op.opcode in MEMORY_OPCODES else op.opcode)
                          for op in program.operations)
    compiled_event = None
    fused_frame = None
    if canonical_array and not capture_trace and use_compiled_frame_event:
        if compiled_frame_retain == "off":
            compiled_event = CompiledFrameEvents(opcode_tokens, event_fixed)
        else:
            fused_frame = _CompiledFrameRetain(opcode_tokens, event_fixed, frame_state,
                                             resident=compiled_frame_retain == "resident")
            compiled_event = fused_frame
            frame_state = fused_frame.state
    records = []
    count = len(program.operations)

    tick = 0
    while tick < fuel:
        local_region = local_regions.get(pc)
        if local_region is not None and fuel - tick >= 3 and len(stack) <= MAX_STACK - 2:
            _run_local_const_add_region(stack, local_region, local_bank, local_fixed)
            pc += 3
            tick += 3
            continue
        region = regions.get(pc)
        if region is not None and fuel - tick >= 2:
            depth = len(stack)
            if 0 < depth < MAX_STACK:
                _run_const_add_region(stack, depth, region)
                pc += 2
                tick += 2
                continue
        if not array_mode:
            pc = int(np.argmax(pc_vector))
        if pc == terminal or pc >= count:
            raise ControlExecutionTrap("Reached terminal without WASM return")
        op = program.operations[pc]
        kind = op.opcode
        microop = (indexed.read(pc) if array_mode else
                   read_microop(pc_vector, microop_tokens, microop_fixed))
        # Condition is read from the ACTUAL binary neural stack before
        # consuming the operand. The gate itself is independent of host
        # opcode branch selection and uses v49 fixed ReLU zero testing.
        condition = (
            stack[-1] if stack
            else np.zeros(32, dtype=np.float64)
        )
        if array_mode:
            next_pc, route = _array_route(targets[pc], guards[pc], condition, fixed, bit_fixed)
        else:
            selected, route = neural_pc_step(pc_vector, condition, tokens, fixed, bit_fixed)
            next_pc = int(np.argmax(selected))
        sequential_pc = route["sequential_target"]
        taken = bool(route["branch_guard_gate"])
        halted = False
        access = None

        entry_kind = None
        entry_base = None
        branch_depth = None
        entering_metadata = None
        # The source-bound decoder, rather than the host opcode, owns
        # how many physical operands the execution path consumes.
        effect = microop.effect
        pop_count = int(effect[2] + effect[3] + 2. * effect[4])
        if array_mode:
            pop_count += int(2 * effect[5])
        # Source semantics are a fail-closed admission oracle only:
        # they never choose the runtime pop count. A mutated one-hot
        # decoder cannot silently consume the wrong number of operands.
        expected_pops = (
            2 if kind in _BINARY or kind == "i32.store" else
            1 if kind in ("if", "br_if", "drop", "local.set", "i32.eqz", "i32.load")
            else 0
        )
        if pop_count != expected_pops:
            raise ControlExecutionTrap("Decoded stack effect violates source WASM")
        consumed = tuple(_pop(stack) for _ in range(pop_count))
        produced = None
        depth_before = int(np.argmax(frame_state.depth))
        if kind in ("block", "loop", "if"):
            end, else_pc = links[pc]
            entry_kind = ("block", "loop", "if").index(kind)
            entry_base = len(stack)
            entering_metadata = (kind, pc, end, else_pc)
        elif kind == "else":
            if (not frame_meta or frame_meta[-1][0] != "if"
                    or frame_meta[-1][3] != pc):
                raise ControlExecutionTrap("Unexpected dynamic else")
            base = int(np.argmax(frame_state.bases[depth_before - 1]))
            if len(stack) != base:
                raise ControlExecutionTrap("Incorrect empty if-arm result")
        elif kind == "end":
            if depth_before:
                if frame_meta[-1][2] != pc:
                    raise ControlExecutionTrap("Mismatched dynamic frame end")
                base = int(np.argmax(frame_state.bases[depth_before - 1]))
                if len(stack) != base:
                    raise ControlExecutionTrap("Unbalanced empty control frame")
            elif pc == count - 1 and len(stack) == 1:
                halted = True
            else:
                raise ControlExecutionTrap("Function must end with one i32")
        elif kind in ("br", "br_if"):
            branch_depth = int(np.argmax(microop.branch_depth))
        elif kind == "return":
            if len(stack) != 1 or next_pc != terminal:
                raise ControlExecutionTrap("Unexpected return stack or target")
            halted = True
        elif kind == "nop":
            pass
        elif kind == "i32.const":
            produced = microop.immediate_bits.copy()
        elif kind == "local.get":
            local_bank, fetched = apply_local_microop(
                local_bank, microop.local_selector, microop.local_mode,
                np.zeros(32, dtype=np.float64), fixed=local_fixed,
            )
            produced = fetched
        elif kind == "local.set":
            local_bank, _ = apply_local_microop(
                local_bank, microop.local_selector, microop.local_mode,
                consumed[0], fixed=local_fixed,
            )
        elif kind == "local.tee":
            if not stack:
                raise ControlExecutionTrap("local.tee stack underflow")
            local_bank, _ = apply_local_microop(
                local_bank, microop.local_selector, microop.local_mode,
                stack[-1], fixed=local_fixed,
            )
        elif kind == "drop":
            pass
        elif kind in MEMORY_OPCODES:
            base = np.packbits(consumed[-1].astype(np.uint8), bitorder="little").view("<u4")[0]
            address = np.uint64(base) + np.uint64(op.immediate.offset)
            if address > MEMORY_BYTES - 4:
                error = LinearMemoryTrap("i32 memory access exceeds the admitted page")
                if capture_trace:
                    error.trace = tuple(records)
                    error.trap_site = {"step": tick, "pc": pc, "opcode": kind, "address": int(address)}
                raise error
            word = page[address:address + np.uint64(4)]
            record_access = capture_trace or (kind == "i32.store" and observe_store is not None)
            before = word.tobytes().hex() if record_access else None
            if kind == "i32.load":
                produced = np.unpackbits(word, bitorder="little").astype(np.float64)
            else:
                word[:] = np.packbits(consumed[0].astype(np.uint8), bitorder="little")
            if record_access:
                access = {"kind": kind, "address": int(address), "before_hex": before,
                          "after_hex": word.tobytes().hex()}
                if kind == "i32.store" and observe_store is not None:
                    observe_store(tick, pc, access.copy())
        elif kind == "i32.eqz":
            left = consumed[0]
            result = execute_alu(
                microop.alu_selector, left, np.zeros(32, dtype=np.float64),
                fixed=alu_fixed, bit_fixed=bit_fixed,
            )
            produced = result
        elif kind in _BINARY:
            right, left = consumed
            circuit = selected_circuits.get(pc)
            if circuit is None:
                produced = execute_alu(
                    microop.alu_selector, left, right, fixed=alu_fixed, bit_fixed=bit_fixed,
                )
            else:
                produced = circuit(left, right)
        else:
            raise ControlExecutionTrap("Unadmitted dynamic operation")
        # The selected microop effect also determines result materialization,
        # including both ALU outputs and source-immediate/local reads.
        pushes = int(effect[1] + effect[3] + effect[4])
        if pushes:
            if produced is None:
                raise ControlExecutionTrap("Microop requested missing output")
            stack.append(produced)
        elif produced is not None:
            raise ControlExecutionTrap("Microop suppressed executable output")
        # Frame-event action is emitted exclusively by the constructed
        # opcode/guard circuit; host opcode branches above handle only
        # operand effects, lexical admission and return validation.
        sequential = next_pc == sequential_pc
        has_else = kind == "if" and links[pc][1] is not None
        active = depth_before > 0
        if compiled_event is None:
            event_vector = select_frame_action(
                opcode_tokens[pc],
                conditional_sequential=np.array([float(sequential)], dtype=np.float64),
                has_else=np.array([float(has_else)], dtype=np.float64),
                frame_active=np.array([float(active)], dtype=np.float64),
                branch_taken=np.array([float(taken)], dtype=np.float64),
                weights=event_fixed,
            )
            action = int(np.argmax(event_vector))
        else:
            action = compiled_event.select(pc, sequential, has_else, active, taken)
            event_vector = compiled_event.vector
        if action == 1 and entering_metadata is None:
            raise ControlExecutionTrap("Gate emitted unsupported frame enter")
        if action == 3 and branch_depth is None:
            raise ControlExecutionTrap("Gate emitted branch without depth token")
        if fused_frame is not None and action == 0:
            # Unit retain preserves depth and lexical metadata. The v53
            # returned height would be the live height and is unused here.
            frame_state = fused_frame.state
            depth_after = depth_before
        else:
            event_tokens = FrameTokens(
                event_vector,
                _frame_optional(entry_kind, KINDS),
                _frame_optional(entry_base, HEIGHTS),
                _frame_optional(branch_depth, MAX_NESTING),
                frame_onehot(len(stack), HEIGHTS),
            )
            frame_state, restored = step_frames(frame_state, event_tokens, frame_fixed)
            depth_after = int(np.argmax(frame_state.depth))
            if action == 3:
                restored_height = int(np.argmax(restored))
                if restored_height > len(stack):
                    raise ControlExecutionTrap("Neural restoration beyond live operands")
                # The actual data stack is trimmed using the constructed output,
                # not a host-selected dynamic branch-frame base.
                del stack[restored_height:]
            frame_meta = frame_meta[:depth_after]
            if action == 1:
                frame_meta.append(entering_metadata)
            if fused_frame is not None:
                fused_frame.publish_checked(frame_state)
                frame_state = fused_frame.state
        if len(frame_meta) != depth_after:
            raise ControlExecutionTrap("Frame metadata not aligned to circuit depth")
        if len(stack) > MAX_STACK:
            raise ControlExecutionTrap("Operand stack resource budget exceeded")
        if capture_trace:
            labels = [
                ("block", "loop", "if")[int(np.argmax(frame_state.kinds[i]))]
                for i in range(depth_after)
            ]
            records.append({
                "step": tick, "pc": pc, "opcode": kind,
                # TERMINAL is a routing sink, represented as WASM len(program)
                # in public trace for parity with the v51 host reference.
                "next_pc": count if next_pc == terminal else next_pc,
                "stack_i32": [_int(v) for v in stack],
                "locals_i32": [
                    _int(v) for v in local_bank[:program.local_count + 1]
                ],
                "control_labels": labels,
                "halted": halted,
                "neural_route": route,
                "neural_frame_event": action,
                "microop_effect": int(np.argmax(microop.effect)),
                "microop_local_mode": int(np.argmax(microop.local_mode)),
                "neural_alu_opcode": (
                    int(np.argmax(microop.alu_selector))
                    if microop.alu_selector.sum() else None
                ),
            })
            if array_mode:
                records[-1]["memory_access"] = access

        if array_mode:
            pc = next_pc
        else:
            pc_vector = selected
        if halted:
            if next_pc != terminal:
                raise ControlExecutionTrap("WASM halted at nonterminal routing PC")
            return FrameRoutedResult(
                _int(stack[-1]), tuple(records), tick + 1,
                hashlib.sha256(encode_control(program)).hexdigest(),
                fixed.digest(), bit_fixed.digest(), route_digest,
                tuple(float(x) for x in frame_fixed.action_weights),
                fixed_event_weight_digest=_event_digest(event_fixed),
                fixed_alu_weights_sha256=alu_fixed.digest(),
                fixed_microop_weights_sha256=microop_fixed.digest(),
                backend="v58-structured-numpy-memory" if array_mode else "v57-structured-numpy",
                compiled_frame_event_sha256=compiled_event.binary_sha256 if compiled_event else None,
                compiled_frame_retain_mode=compiled_frame_retain if fused_frame is not None else None,
            )
        tick += 1
    error = ControlFuelExhausted(f"Explicit bounded fuel exhausted after {fuel} neural-routed transitions")
    if array_mode and capture_trace:
        error.trace = tuple(records)
        error.trap_site = {"step": fuel, "pc": pc, "opcode": None, "address": None}
    raise error
