"""v51 strict structured-control WASM subset with source-bound execution traces.

Actual WASM 1.0 blocks/loops/if/else/br/br_if/return are encoded/decoded
with their real opcode bytes, not a custom instruction set. One canonical
exported run(i32)->i32, homogeneous i32 locals, empty block signatures,
and only the reviewed v49 i32 operations are admitted.

An explicit Python-host control stack selects PCs and handles fuel/traps,
while arithmetic uses v49's constructed ReLU/bit-gated i32 circuits.
This is NOT YET a wholly neural control-flow executor and no all-input
equivalence to Wasmtime is claimed from finite differential witnesses.

The custom parser validates the restricted ENVELOPE and STRUCTURAL nesting,
not every WASM stack type. CI independently validates each original module
with Wasmtime before treating it as admitted.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

from .bit_neural_vm_v49 import (
    BitCircuitWeights,
    _add,
    _bits,
    _eqz,
    _int,
    _mul,
    _signed_lt,
    _sub,
    construct_bit_weights,
)
from .wasm_bitstream_v39 import (
    MAGIC,
    WasmBinaryError,
    _read_u32,
    _u32,
    parse_section_envelope,
)
from .wasm_state_vm_v40 import (
    _EXPORT_PAYLOAD,
    _FUNCTION_PAYLOAD,
    _TYPE_PAYLOAD,
    I32Program,
    _read_s32,
    _s32,
)

SCHEMA = "visuaml.wasm-structured-control.v51"
MAX_OPERATIONS = 96
MAX_FUEL = 2000
SCALED_SCHEMA = "visuaml.wasm-structured-control.v58-scaled"
SCALED_MAX_OPERATIONS = 4096
SCALED_MAX_FUEL = 50_000_000
MAX_NESTING = 12
MAX_STACK = 24
MAX_LOCALS = 4
MEMORY_BYTES = 65536
MEMORY_OPCODES = ("i32.load", "i32.store")
_MEMORY_CODES = {0x28: MEMORY_OPCODES[0], 0x36: MEMORY_OPCODES[1]}
_MEMORY_ENCODING = {name: code for code, name in _MEMORY_CODES.items()}
_MEMORY_ENTRY = b"\x06memory\x02\x00"
_MEMORY_EXPORT = b"\x02" + _EXPORT_PAYLOAD[1:] + _MEMORY_ENTRY
_MEMORY_FIRST_EXPORT = b"\x02" + _MEMORY_ENTRY + _EXPORT_PAYLOAD[1:]

_CODES = {
    0x01: "nop",
    0x02: "block", 0x03: "loop", 0x04: "if", 0x05: "else",
    0x0B: "end", 0x0C: "br", 0x0D: "br_if", 0x0F: "return",
    0x1A: "drop", 0x20: "local.get", 0x21: "local.set",
    0x22: "local.tee", 0x41: "i32.const", 0x45: "i32.eqz",
    0x48: "i32.lt_s", 0x4A: "i32.gt_s", 0x6A: "i32.add",
    0x6B: "i32.sub", 0x6C: "i32.mul",
}
_OPCODES = {name: code for code, name in _CODES.items()}
_ADMITTED_NAMES = frozenset((*_OPCODES, *MEMORY_OPCODES))
_BLOCKS = frozenset(("block", "loop", "if"))
_LOCALS = frozenset(("local.get", "local.set", "local.tee"))
_BINARY = frozenset(("i32.add", "i32.sub", "i32.mul", "i32.lt_s", "i32.gt_s"))


class ControlAdmissionError(ValueError):
    """Malformed or unsupported restricted structured control source."""


class ControlExecutionTrap(RuntimeError):
    """Runtime stack/control error in the admitted bounded machine.

    Traced packed memory runs populate the completed prefix and trap site
    for out-of-bounds and fuel exhaustion.
    """

    trace: tuple[dict, ...] | None = None
    trap_site: dict | None = None


class LinearMemoryTrap(ControlExecutionTrap):
    """A bounded linear memory access exceeded the admitted page."""


class ControlFuelExhausted(ControlExecutionTrap):
    """Bounded nontermination witness, NOT a WASM language trap."""


@dataclass(frozen=True)
class MemoryArgument:
    offset: int = 0
    align: int = 2


@dataclass(frozen=True)
class ControlOperation:
    opcode: str
    immediate: int | MemoryArgument | None = None


@dataclass(frozen=True)
class ControlProgram:
    local_count: int
    operations: tuple[ControlOperation, ...]
    schema: str = SCHEMA
    memory_export_first: bool = False

    def digest(self) -> str:
        data = asdict(self)
        if not self.memory_export_first:
            del data["memory_export_first"]  # Preserve legacy source identities.
        return hashlib.sha256(json.dumps(
            data, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()


@dataclass(frozen=True)
class ControlFrame:
    kind: str
    start_pc: int
    end_pc: int
    else_pc: int | None
    stack_base: int


@dataclass(frozen=True)
class ControlResult:
    value: int
    trace: tuple[dict, ...]
    transitions: int
    source_sha256: str
    constructed_arithmetic_weights_sha256: str
    host_control_routing: bool = True


def uses_memory(program: ControlProgram) -> bool:
    return any(op.opcode in MEMORY_OPCODES for op in program.operations)


def _syntax(program: ControlProgram) -> dict[int, tuple[int, int | None]]:
    if (not isinstance(program, ControlProgram) or program.schema not in (SCHEMA, SCALED_SCHEMA)
            or type(program.memory_export_first) is not bool
            or type(program.local_count) is not int
            or not 0 <= program.local_count <= MAX_LOCALS
            or not isinstance(program.operations, tuple)
            or not 2 <= len(program.operations) <= (
                SCALED_MAX_OPERATIONS if program.schema == SCALED_SCHEMA else MAX_OPERATIONS
            )):
        raise ControlAdmissionError("Invalid bounded structured WASM program")
    stack: list[tuple[int, str, int | None]] = []
    links: dict[int, tuple[int, int | None]] = {}
    terminal = -1
    for pc, op in enumerate(program.operations):
        if not isinstance(op, ControlOperation) or op.opcode not in _ADMITTED_NAMES:
            raise ControlAdmissionError("Unsupported structured instruction")
        name, imm = op.opcode, op.immediate
        if name in _BLOCKS:
            if imm is not None:
                raise ControlAdmissionError("Only empty 0x40 block signatures")
            stack.append((pc, name, None))
            if len(stack) > MAX_NESTING:
                raise ControlAdmissionError("Structured control nesting too deep")
        elif name == "else":
            if not stack or stack[-1][1] != "if" or stack[-1][2] is not None:
                raise ControlAdmissionError("Unmatched or duplicate else")
            if imm is not None:
                raise ControlAdmissionError("Else cannot have immediate")
            start, kind, _ = stack.pop()
            stack.append((start, kind, pc))
        elif name == "end":
            if imm is not None:
                raise ControlAdmissionError("End cannot have immediate")
            if stack:
                start, _, else_pc = stack.pop()
                links[start] = (pc, else_pc)
            else:
                if pc != len(program.operations) - 1:
                    raise ControlAdmissionError("Function end must be terminal")
                terminal = pc
        elif name in ("br", "br_if"):
            if (type(imm) is not int or not 0 <= imm < len(stack)):
                raise ControlAdmissionError("Branch label outside lexical control stack")
        elif name in _LOCALS:
            if type(imm) is not int or not 0 <= imm <= program.local_count:
                raise ControlAdmissionError("Invalid declared local index")
        elif name == "i32.const":
            if type(imm) is not int or not -(1 << 31) <= imm < (1 << 31):
                raise ControlAdmissionError("Non-i32 immediate")
        elif name in MEMORY_OPCODES:
            if (type(imm) is not MemoryArgument or type(imm.offset) is not int
                    or not 0 <= imm.offset < 1 << 32 or type(imm.align) is not int
                    or not 0 <= imm.align <= 2):
                raise ControlAdmissionError("Invalid i32 memory alignment or offset")
        elif imm is not None:
            raise ControlAdmissionError("Unexpected instruction immediate")
    if terminal != len(program.operations) - 1 or stack:
        raise ControlAdmissionError("Unclosed or missing final WASM end")
    return links


def _memory_section(pages: int) -> bytes:
    if type(pages) is not int or not 1 <= pages <= 256:
        raise ControlAdmissionError("Fixed memory must have between 1 and 256 pages")
    limit = _u32(pages)
    return b"\x01\x01" + limit + limit


def encode_control(program: ControlProgram, *, memory_pages: int = 1) -> bytes:
    """Canonical one-function actual WASM binary with structural controls."""
    _syntax(program)
    body = bytearray()
    if program.local_count:
        body.extend(_u32(1) + _u32(program.local_count) + b"\x7f")
    else:
        body.extend(_u32(0))
    for operation in program.operations:
        if operation.opcode in MEMORY_OPCODES:
            body.append(_MEMORY_ENCODING[operation.opcode])
            body.extend(_u32(operation.immediate.align) + _u32(operation.immediate.offset))
            continue
        body.append(_OPCODES[operation.opcode])
        if operation.opcode in _BLOCKS:
            body.append(0x40)
        elif operation.opcode in (*_LOCALS, "br", "br_if"):
            body.extend(_u32(operation.immediate))
        elif operation.opcode == "i32.const":
            body.extend(_s32(operation.immediate))
    code = _u32(1) + _u32(len(body)) + bytes(body)

    def section(kind: int, data: bytes) -> bytes:
        return bytes((kind,)) + _u32(len(data)) + data

    memory = uses_memory(program)
    exports = _EXPORT_PAYLOAD
    if program.memory_export_first:
        if not memory or program.schema != SCALED_SCHEMA:
            raise ControlAdmissionError("Memory-first exports require a scaled memory program")
        exports = _MEMORY_FIRST_EXPORT
    elif memory:
        exports = _MEMORY_EXPORT
    return (MAGIC + section(1, _TYPE_PAYLOAD)
            + section(3, _FUNCTION_PAYLOAD)
            + (section(5, _memory_section(memory_pages)) if memory else b"")
            + section(7, exports)
            + section(10, code))


def decode_control(raw: bytes, *, scaled: bool = False,
                   memory_pages: int = 1) -> ControlProgram:
    """Parse the canonical opcode subset with optional packed-scale bounds."""
    if type(scaled) is not bool:
        raise ControlAdmissionError("scaled must be a boolean")
    max_operations = SCALED_MAX_OPERATIONS if scaled else MAX_OPERATIONS
    try:
        chunks = parse_section_envelope(raw)
        ids = [section.section_id for section in chunks]
        memory = ids == [1, 3, 5, 7, 10]
        if not memory and ids != [1, 3, 7, 10]:
            raise ControlAdmissionError("Only reviewed standard sections admitted")
        payloads = {section.section_id: section.payload for section in chunks}
        memory_first = scaled and memory and payloads[7] == _MEMORY_FIRST_EXPORT
        exports = _MEMORY_FIRST_EXPORT if memory_first else (_MEMORY_EXPORT if memory else _EXPORT_PAYLOAD)
        if (payloads[1] != _TYPE_PAYLOAD or payloads[3] != _FUNCTION_PAYLOAD
                or payloads[7] != exports
                or (memory and payloads[5] != _memory_section(memory_pages))):
            raise ControlAdmissionError("Expected reviewed run export and optional fixed-size memory")
        data = payloads[10]
        n, pos = _read_u32(data, 0)
        length, pos = _read_u32(data, pos)
        if n != 1 or pos + length != len(data):
            raise ControlAdmissionError("One exact-length function body required")
        count, ops = _decode_control_body(data[pos:], memory, max_operations)
        program = ControlProgram(count, tuple(ops), SCALED_SCHEMA if scaled else SCHEMA, memory_first)
        _syntax(program)
        if encode_control(program, memory_pages=memory_pages) != raw:
            raise ControlAdmissionError("Noncanonical or unreviewed WASM encoding")
        return program
    except (WasmBinaryError, IndexError, TypeError) as exc:
        raise ControlAdmissionError("Invalid canonical WASM structured encoding") from exc


def _decode_control_body(data, memory, max_operations, *, direct_calls=False):
    nlocals, pos = _read_u32(data, 0)
    if nlocals == 0:
        count = 0
    elif nlocals == 1:
        count, pos = _read_u32(data, pos)
        if (not 1 <= count <= MAX_LOCALS or pos >= len(data)
                or data[pos] != 0x7f):
            raise ControlAdmissionError("Only bounded i32 local group admitted")
        pos += 1
    else:
        raise ControlAdmissionError("Unadmitted local declaration groups")
    ops = []
    while pos < len(data):
        code = data[pos]
        pos += 1
        name = _CODES.get(code) or (_MEMORY_CODES.get(code) if memory else None)
        if code == 0x10 and direct_calls:
            name = "call"
        if name is None:
            raise ControlAdmissionError("Unknown or excluded WASM instruction byte")
        imm = None
        if name in _BLOCKS:
            if pos >= len(data) or data[pos] != 0x40:
                raise ControlAdmissionError("Unadmitted nonempty block type")
            pos += 1
        elif name == "call":
            # rust-lld leaves padded relocation immediates. The caller has
            # already validated the complete original module with Wasmtime.
            imm = 0
            for shift in range(0, 35, 7):
                byte = data[pos]
                pos += 1
                imm |= (byte & 0x7f) << shift
                if byte < 0x80:
                    break
            else:
                raise ControlAdmissionError("Unterminated direct-call index")
        elif name in (*_LOCALS, "br", "br_if"):
            imm, pos = _read_u32(data, pos)
        elif name == "i32.const":
            imm, pos = _read_s32(data, pos)
        elif name in MEMORY_OPCODES:
            align, pos = _read_u32(data, pos)
            offset, pos = _read_u32(data, pos)
            imm = MemoryArgument(offset, align)
        ops.append(ControlOperation(name, imm))
        if len(ops) > max_operations:
            raise ControlAdmissionError("Instruction count exceeded")
    return count, ops


def _pop(stack: list[np.ndarray]) -> np.ndarray:
    if not stack:
        raise ControlExecutionTrap("Typed operand stack underflow")
    return stack.pop()


def execute_control(
    program: ControlProgram, argument: int, *, fuel: int = 400,
    fixed: BitCircuitWeights | None = None,
) -> ControlResult:
    """One explicit host-routed structured PC/label stack + constructed i32 gates.

    Every runtime transition records source PC, next PC, label frames,
    complete live stack/local values, and step count. The host dispatches
    control flow; the v49 numeric gates implement all i32 arithmetic.
    """
    links = _syntax(program)
    if uses_memory(program):
        raise ControlAdmissionError("Memory execution requires the packed v58 executor")
    if type(fuel) is not int or not 1 <= fuel <= MAX_FUEL:
        raise ControlAdmissionError("Unbounded or invalid execution fuel")
    fixed = construct_bit_weights() if fixed is None else fixed
    local = [_bits(argument), *[_bits(0) for _ in range(program.local_count)]]
    stack: list[np.ndarray] = []
    frames: list[ControlFrame] = []
    trace = []
    pc = 0

    def label_jump(depth: int) -> int:
        if depth >= len(frames):
            raise ControlExecutionTrap("Dynamic branch label outside control stack")
        index = len(frames) - 1 - depth
        target = frames[index]
        del stack[target.stack_base:]
        if target.kind == "loop":
            # Keep the target loop label; remove only its nested labels.
            del frames[index + 1:]
            return target.start_pc + 1
        # A branch to block/if exits that label and all nested labels.
        del frames[index:]
        return target.end_pc + 1

    for tick in range(fuel):
        if not 0 <= pc < len(program.operations):
            raise ControlExecutionTrap("Invalid bounded control PC")
        op = program.operations[pc]
        before = pc
        name, imm = op.opcode, op.immediate
        jump = pc + 1
        halted = False
        if name in ("block", "loop", "if"):
            end, other = links[pc]
            if name == "if":
                condition = _pop(stack)
                enabled = _int(condition) != 0
            else:
                enabled = True
            if enabled:
                frames.append(ControlFrame(name, pc, end, other, len(stack)))
            elif other is not None:
                frames.append(ControlFrame(name, pc, end, other, len(stack)))
                jump = other + 1
            else:
                jump = end + 1
        elif name == "else":
            if not frames or frames[-1].kind != "if" or frames[-1].else_pc != pc:
                raise ControlExecutionTrap("Unexpected else in dynamic control")
            target = frames.pop()
            if len(stack) != target.stack_base:
                raise ControlExecutionTrap("If arm changed empty block stack")
            jump = target.end_pc + 1
        elif name == "end":
            if frames:
                frame = frames.pop()
                if frame.end_pc != pc or len(stack) != frame.stack_base:
                    raise ControlExecutionTrap("Unbalanced empty block/loop result")
            else:
                if pc != len(program.operations)-1 or len(stack) != 1:
                    raise ControlExecutionTrap("Function did not return exactly one i32")
                halted = True
        elif name == "br":
            jump = label_jump(imm)
        elif name == "br_if":
            condition = _pop(stack)
            if _int(condition):
                jump = label_jump(imm)
        elif name == "return":
            if len(stack) != 1:
                raise ControlExecutionTrap("Return must leave exactly one i32")
            frames.clear()
            halted = True
            jump = len(program.operations)
        elif name == "nop":
            pass
        elif name == "i32.const":
            stack.append(_bits(imm))
        elif name == "local.get":
            stack.append(local[imm].copy())
        elif name == "local.set":
            local[imm] = _pop(stack)
        elif name == "local.tee":
            if not stack:
                raise ControlExecutionTrap("local.tee missing operand")
            local[imm] = stack[-1].copy()
        elif name in _BINARY:
            right, left = _pop(stack), _pop(stack)
            if name == "i32.add":
                result = _add(left, right, fixed=fixed)
            elif name == "i32.sub":
                result = _sub(left, right, fixed=fixed)
            elif name == "i32.mul":
                result = _mul(left, right, fixed.carry_thresholds)
            else:
                result = np.zeros(32, dtype=np.float64)
                result[0] = (_signed_lt(left, right, fixed)
                             if name == "i32.lt_s"
                             else _signed_lt(right, left, fixed))
            stack.append(result)
        elif name == "i32.eqz":
            bits = np.zeros(32, dtype=np.float64)
            bits[0] = _eqz(_pop(stack), fixed)
            stack.append(bits)
        elif name == "drop":
            _pop(stack)
        else:
            raise ControlExecutionTrap("Unadmitted dynamic instruction")
        if len(stack) > MAX_STACK:
            raise ControlExecutionTrap("Operand stack resource bound")
        trace.append({
            "step": tick, "pc": before, "opcode": name,
            "next_pc": jump,
            "stack_i32": [_int(v) for v in stack],
            "locals_i32": [_int(v) for v in local],
            "control_labels": [f.kind for f in frames],
            "halted": halted,
        })
        if halted:
            return ControlResult(
                _int(stack[-1]), tuple(trace), len(trace),
                hashlib.sha256(encode_control(program)).hexdigest(),
                fixed.digest(),
            )
        pc = jump
    raise ControlFuelExhausted(
        f"Explicit bounded fuel exhausted after {fuel} transitions"
    )


def linear_program_to_control(program: I32Program) -> ControlProgram:
    """Wrap a reviewed v40 linear program without altering source semantics."""
    operations = tuple(ControlOperation(x.opcode, x.immediate)
                       for x in program.operations)
    return ControlProgram(program.local_count, operations)
