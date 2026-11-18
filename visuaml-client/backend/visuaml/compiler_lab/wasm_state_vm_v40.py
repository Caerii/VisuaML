"""v40 bounded, effect-explicit i32 machine and strict WASM binary bridge.

Research scope: one exported (i32)->i32 function, at most 16 i32 locals,
straight-line MVP instructions, no imports, host calls, memory, globals,
custom sections, branching, traps or alternate value types. This is a
state-transition dialect distinct from the neural-assembly stateful IR.

Canonical WASM bytes are parsed without evaluating supplied code. Wasmtime
validation/execution is a *separate*, independent CI requirement. Alpha-local
rewrites have a syntactic certificate, not a machine-checked general proof.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from .wasm_bitstream_v39 import (
    MAGIC,
    WasmBinaryError,
    _read_u32,
    _u32,
    parse_section_envelope,
)

SCHEMA = "visuaml.typed-i32-wasm-transitions.v40"
REWRITE_SCHEMA = "visuaml.wasm-i32-local-alpha-rewrite.v40"
MAX_LOCALS = 16
MAX_INSTRUCTIONS = 64
MAX_STACK = 64
I32_MIN = -(1 << 31)
I32_MAX = (1 << 31) - 1
_OPCODES = {
    0x20: "local.get",
    0x21: "local.set",
    0x22: "local.tee",
    0x41: "i32.const",
    0x6A: "i32.add",
    0x6B: "i32.sub",
    0x6C: "i32.mul",
    0x45: "i32.eqz",
    0x48: "i32.lt_s",
    0x4A: "i32.gt_s",
    0x1A: "drop",
    0x0B: "end",
}
_CODES = {v: k for k, v in _OPCODES.items()}
_LOCAL_OPS = frozenset({"local.get", "local.set", "local.tee"})
_BINARY_OPS = frozenset({"i32.add", "i32.sub", "i32.mul", "i32.lt_s", "i32.gt_s"})
_TYPE_PAYLOAD = b"\x01\x60\x01\x7f\x01\x7f"
_FUNCTION_PAYLOAD = b"\x01\x00"
_EXPORT_PAYLOAD = b"\x01\x03run\x00\x00"


class VMAdmissionError(WasmBinaryError):
    """Outside the exact typed machine, noncanonical bytes, or bad state."""


@dataclass(frozen=True)
class VMOperation:
    opcode: str
    immediate: int | None = None


@dataclass(frozen=True)
class I32Program:
    local_count: int
    operations: tuple[VMOperation, ...]
    schema: str = SCHEMA

    def digest(self) -> str:
        return _digest_json(asdict(self))


@dataclass(frozen=True)
class VMState:
    pc: int
    stack: tuple[int, ...]
    locals: tuple[int, ...]
    halted: bool = False


@dataclass(frozen=True)
class VMTransition:
    pc: int
    opcode: str
    immediate: int | None
    stack_before: tuple[int, ...]
    stack_after: tuple[int, ...]
    locals_before: tuple[int, ...]
    locals_after: tuple[int, ...]
    writes_local: int | None


@dataclass(frozen=True)
class VMResult:
    value: int
    final_state: VMState
    trace: tuple[VMTransition, ...]
    trace_sha256: str


@dataclass(frozen=True)
class ModuleBinding:
    source_sha256: str
    program: I32Program
    code_section_sha256: str


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest_json(value: object) -> str:
    return _sha(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8"))


def _signed32(value: int) -> int:
    result = value & 0xFFFFFFFF
    return result - (1 << 32) if result >= 1 << 31 else result


def _s32(value: int) -> bytes:
    if type(value) is not int or not I32_MIN <= value <= I32_MAX:
        raise VMAdmissionError("i32 immediate outside signed 32-bit range")
    encoded = bytearray()
    while True:
        digit = value & 0x7F
        value >>= 7
        last = (value == 0 and not digit & 0x40) or (
            value == -1 and bool(digit & 0x40)
        )
        encoded.append(digit if last else digit | 0x80)
        if last:
            return bytes(encoded)


def _read_s32(raw: bytes, pos: int) -> tuple[int, int]:
    begin = pos
    value = 0
    for index in range(5):
        if pos >= len(raw):
            raise VMAdmissionError("Truncated signed i32 LEB128")
        digit = raw[pos]
        pos += 1
        value |= (digit & 0x7F) << (index * 7)
        if not digit & 0x80:
            bits = (index + 1) * 7
            if digit & 0x40:
                value -= 1 << bits
            if not I32_MIN <= value <= I32_MAX or raw[begin:pos] != _s32(value):
                raise VMAdmissionError("Noncanonical/overflowing i32 LEB128")
            return value, pos
    raise VMAdmissionError("Unterminated signed i32 LEB128")


def validate_program(program: I32Program) -> None:
    if (not isinstance(program, I32Program) or program.schema != SCHEMA
            or type(program.local_count) is not int
            or not 0 <= program.local_count <= MAX_LOCALS
            or not isinstance(program.operations, tuple)
            or not 2 <= len(program.operations) <= MAX_INSTRUCTIONS):
        raise VMAdmissionError("Invalid bounded VM program schema")
    depth = 0
    for position, operation in enumerate(program.operations):
        if not isinstance(operation, VMOperation) or operation.opcode not in _CODES:
            raise VMAdmissionError("Unknown typed i32 operation")
        name, immediate = operation.opcode, operation.immediate
        if name in _LOCAL_OPS:
            if type(immediate) is not int or not 0 <= immediate <= program.local_count:
                raise VMAdmissionError("Invalid typed local index")
        elif name == "i32.const":
            if type(immediate) is not int or not I32_MIN <= immediate <= I32_MAX:
                raise VMAdmissionError("Invalid signed i32 constant")
        elif immediate is not None:
            raise VMAdmissionError("Spurious opcode immediate")
        if name == "end":
            if position != len(program.operations) - 1 or depth != 1:
                raise VMAdmissionError("Function end must return exactly one i32")
        elif position == len(program.operations) - 1:
            raise VMAdmissionError("Missing final end opcode")
        elif name in ("i32.const", "local.get"):
            depth += 1
        elif name in ("local.set", "drop"):
            depth -= 1
        elif name in _BINARY_OPS:
            depth -= 1
        elif name in ("local.tee", "i32.eqz"):
            pass
        if depth < 0 or depth > MAX_STACK:
            raise VMAdmissionError("WASM operand stack type/size violation")


def _section(kind: int, data: bytes) -> bytes:
    return bytes([kind]) + _u32(len(data)) + data


def _write_operations(program: I32Program) -> bytes:
    result = bytearray()
    for op in program.operations:
        result.append(_CODES[op.opcode])
        if op.opcode in _LOCAL_OPS:
            result.extend(_u32(op.immediate))
        elif op.opcode == "i32.const":
            result.extend(_s32(op.immediate))
    return bytes(result)


def encode_program(program: I32Program) -> bytes:
    """Create canonical module with one run(i32)->i32 function."""
    validate_program(program)
    if program.local_count:
        declaration = _u32(1) + _u32(program.local_count) + b"\x7f"
    else:
        declaration = _u32(0)
    body = declaration + _write_operations(program)
    code_payload = _u32(1) + _u32(len(body)) + body
    return (MAGIC + _section(1, _TYPE_PAYLOAD)
            + _section(3, _FUNCTION_PAYLOAD)
            + _section(7, _EXPORT_PAYLOAD)
            + _section(10, code_payload))


def decode_module(raw: bytes) -> ModuleBinding:
    """Reject anything outside a closed-world, canonical, typed WASM subset."""
    try:
        sections = parse_section_envelope(raw)
    except WasmBinaryError as exc:
        raise VMAdmissionError("Invalid WASM section envelope") from exc
    if [section.section_id for section in sections] != [1, 3, 7, 10]:
        raise VMAdmissionError("Only type/function/export/code sections admitted")
    if (sections[0].payload != _TYPE_PAYLOAD
            or sections[1].payload != _FUNCTION_PAYLOAD
            or sections[2].payload != _EXPORT_PAYLOAD):
        raise VMAdmissionError("Expected one exported run(i32)->i32 function")
    data = sections[3].payload
    try:
        functions, pos = _read_u32(data, 0)
        size, pos = _read_u32(data, pos)
    except WasmBinaryError as exc:
        raise VMAdmissionError("Invalid code section LEB128") from exc
    if functions != 1 or size != len(data) - pos:
        raise VMAdmissionError("One exact-length WASM function body required")
    try:
        groups, pos = _read_u32(data, pos)
        if groups == 0:
            count = 0
        elif groups == 1:
            count, pos = _read_u32(data, pos)
            if not 1 <= count <= MAX_LOCALS or pos >= len(data) or data[pos] != 0x7F:
                raise VMAdmissionError("Only bounded i32 locals admitted")
            pos += 1
        else:
            raise VMAdmissionError("Unsupported WASM local declaration groups")
    except WasmBinaryError as exc:
        raise VMAdmissionError("Truncated WASM local declaration") from exc
    operations: list[VMOperation] = []
    while pos < len(data):
        opcode = data[pos]
        pos += 1
        name = _OPCODES.get(opcode)
        if name is None:
            raise VMAdmissionError("Unsupported WASM instruction byte")
        immediate = None
        if name in _LOCAL_OPS:
            try:
                immediate, pos = _read_u32(data, pos)
            except WasmBinaryError as exc:
                raise VMAdmissionError("Malformed local index LEB128") from exc
        elif name == "i32.const":
            immediate, pos = _read_s32(data, pos)
        operations.append(VMOperation(name, immediate))
        if len(operations) > MAX_INSTRUCTIONS:
            raise VMAdmissionError("WASM instruction count exceeded")
        if name == "end" and pos != len(data):
            raise VMAdmissionError("Trailing instructions after function end")
    program = I32Program(count, tuple(operations))
    validate_program(program)
    if encode_program(program) != raw:
        raise VMAdmissionError("Noncanonical/restructured binary not admitted")
    return ModuleBinding(_sha(raw), program, _sha(sections[3].raw))


def initial_state(program: I32Program, argument: int) -> VMState:
    validate_program(program)
    if type(argument) is not int or not I32_MIN <= argument <= I32_MAX:
        raise VMAdmissionError("Expected one signed i32 input")
    return VMState(0, (), (argument,) + (0,) * program.local_count)


def step(program: I32Program, state: VMState) -> tuple[VMState, VMTransition]:
    """Pure, immutable single-instruction state transition, including local effects."""
    if (not isinstance(state, VMState) or state.halted
            or not 0 <= state.pc < len(program.operations)
            or len(state.locals) != program.local_count + 1):
        raise VMAdmissionError("Invalid mutable machine boundary")
    op = program.operations[state.pc]
    name, immediate = op.opcode, op.immediate
    stack = list(state.stack)
    local = list(state.locals)
    def pop() -> int:
        if not stack:
            raise VMAdmissionError("Dynamic operand stack underflow")
        return stack.pop()
    written = None
    if name == "i32.const":
        stack.append(immediate)
    elif name == "local.get":
        stack.append(local[immediate])
    elif name in ("local.set", "local.tee"):
        value = pop()
        local[immediate] = value
        written = immediate
        if name == "local.tee":
            stack.append(value)
    elif name in _BINARY_OPS:
        right, left = pop(), pop()
        if name == "i32.add":
            result = _signed32(left + right)
        elif name == "i32.sub":
            result = _signed32(left - right)
        elif name == "i32.mul":
            result = _signed32(left * right)
        elif name == "i32.lt_s":
            result = int(left < right)
        else:
            result = int(left > right)
        stack.append(result)
    elif name == "i32.eqz":
        stack.append(int(pop() == 0))
    elif name == "drop":
        pop()
    elif name != "end":
        raise VMAdmissionError("Unadmitted runtime instruction")
    next_state = VMState(state.pc + 1, tuple(stack), tuple(local), name == "end")
    transition = VMTransition(
        state.pc, name, immediate,
        state.stack, next_state.stack, state.locals, next_state.locals, written,
    )
    return next_state, transition


def execute_program(program: I32Program, argument: int,
                    *, fuel: int = MAX_INSTRUCTIONS) -> VMResult:
    validate_program(program)
    if type(fuel) is not int or not 0 <= fuel <= MAX_INSTRUCTIONS:
        raise VMAdmissionError("Unbounded or invalid VM execution fuel")
    state = initial_state(program, argument)
    trace: list[VMTransition] = []
    while not state.halted:
        if len(trace) >= fuel:
            raise VMAdmissionError("Execution fuel exhausted")
        state, item = step(program, state)
        trace.append(item)
    if len(state.stack) != 1:
        raise VMAdmissionError("Unexpected final stack size")
    return VMResult(
        state.stack[0], state, tuple(trace),
        _digest_json([asdict(x) for x in trace]),
    )


def alpha_rename_locals(source: bytes,
                        new_order: tuple[int, ...]) -> tuple[bytes, dict]:
    """Permute i32 *nonparameter* locals and rewrite every access.

    new_order[new_local_slot] = old_local_slot, both zero-based. Parameter 0
    stays untouched. No custom/name sections, imports or indirect references.
    """
    binding = decode_module(source)
    program = binding.program
    count = program.local_count
    if (not isinstance(new_order, tuple) or len(new_order) != count
            or any(type(x) is not int for x in new_order)
            or set(new_order) != set(range(count))):
        raise VMAdmissionError("Expected bijective permutation of i32 local slots")
    inverse = tuple(new_order.index(i) for i in range(count))
    def remap(item: VMOperation) -> VMOperation:
        if item.opcode not in _LOCAL_OPS or item.immediate == 0:
            return item
        return VMOperation(item.opcode, inverse[item.immediate - 1] + 1)
    transformed = I32Program(count, tuple(remap(x) for x in program.operations))
    target = encode_program(transformed)
    checked = decode_module(target)
    if (checked.program != transformed or checked.program.local_count != count
            or tuple(
                VMOperation(x.opcode, (
                    new_order[x.immediate - 1] + 1
                    if x.opcode in _LOCAL_OPS and x.immediate != 0
                    else x.immediate
                )) for x in transformed.operations
            ) != program.operations):
        raise VMAdmissionError("Local rewrite failed syntactic alpha-inverse check")
    canonical_inverse = alpha_inverse_order(new_order)
    if _reverse_alpha(target, canonical_inverse) != source:
        raise VMAdmissionError("Local rewrite failed exact binary round-trip")
    receipt = {
        "schema": REWRITE_SCHEMA,
        "source_sha256": _sha(source),
        "target_sha256": _sha(target),
        "source_semantic_sha256": program.digest(),
        "target_semantic_sha256": transformed.digest(),
        "permutation_new_slot_to_old_slot": list(new_order),
        "inverse_permutation": list(inverse),
        "changed_byte_count": sum(x != y for x, y in zip(source, target)),
        "binary_roundtrip_exact": True,
        "typed_alpha_inverse_validated": True,
        "module_typechecked_by_independent_engine": False,
        "observational_parity_tested": False,
        "claim": "Closed-world same-typed local alpha-renaming only",
    }
    return target, receipt


def alpha_inverse_order(new_order: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(new_order.index(i) for i in range(len(new_order)))


def _reverse_alpha(source: bytes, order: tuple[int, ...]) -> bytes:
    program = decode_module(source).program
    mapping = tuple(order.index(i) for i in range(program.local_count))
    operations = tuple(
        VMOperation(op.opcode, mapping[op.immediate - 1] + 1)
        if op.opcode in _LOCAL_OPS and op.immediate != 0 else op
        for op in program.operations
    )
    return encode_program(I32Program(program.local_count, operations))
