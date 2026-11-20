"""Legalize a small set of LLVM i32 forms into Pocket's existing opcode subset."""
import argparse
from pathlib import Path

import wasmtime

from visuaml.compiler_lab.wasm_control_flow_v51 import (
    _CODES, _MEMORY_CODES, _decode_control_body, _read_s32, _read_u32, _s32, _u32, parse_section_envelope,
)

# Equal i32 words have wrapping difference zero. Ordered comparisons invert
# the supported signed opposite comparison. Constant shl is multiply mod 2**32.
REPLACEMENTS = {
    0x46: b"\x6b\x45", 0x47: b"\x6b\x45\x45",
    0x4c: b"\x4a\x45", 0x4e: b"\x48\x45",
}


def lower_body(data, remap=None):
    groups, pos = _read_u32(data, 0)
    for _ in range(groups):
        _, pos = _read_u32(data, pos)
        pos += 1
    prefix = data[:pos]
    instructions = []
    while pos < len(data):
        start = pos
        op = data[pos]
        pos += 1
        immediate = None
        if op in (2, 3, 4):
            pos += 1
        elif op in (0x0c, 0x0d, 0x20, 0x21, 0x22):
            _, pos = _read_u32(data, pos)
        elif op == 0x10:
            target = shift = 0
            while True:
                byte = data[pos]
                pos += 1
                target |= (byte & 127) << shift
                if not byte & 128:
                    break
                shift += 7
            if remap is not None:
                instructions.append((op, None, b"\x10" + _u32(remap[target])))
                continue
        elif op == 0x41:
            immediate, pos = _read_s32(data, pos)
        elif op in _MEMORY_CODES:
            _, pos = _read_u32(data, pos)
            _, pos = _read_u32(data, pos)
        elif op == 0x74:
            if not instructions or instructions[-1][0] != 0x41:
                raise ValueError(f"Dynamic shift at byte {start}")
            shift = instructions.pop()[1] & 31
            multiplier = -(1 << 31) if shift == 31 else 1 << shift
            instructions.append((op, None, b"\x41" + _s32(multiplier) + b"\x6c"))
            continue
        elif op not in _CODES and op not in REPLACEMENTS:
            raise ValueError(f"Unadmitted opcode 0x{op:02x} at body byte {start}")
        instructions.append((op, immediate, REPLACEMENTS.get(op, data[start:pos])))
    return prefix + b"".join(encoded for _, _, encoded in instructions)


def lower(raw):
    engine = wasmtime.Engine()
    wasmtime.Module.validate(engine, raw)
    sections = {section.section_id: section.payload for section in parse_section_envelope(raw)}
    if list(sections) != [1, 3, 5, 7, 10] or sections[1] != bytes.fromhex("0160017f017f"):
        raise ValueError("Expected fixed-memory unary i32 cartridge module")
    data = sections[10]
    count, pos = _read_u32(data, 0)
    entry = count - 1
    memory, run = b"\x06memory\x02\x00", b"\x03run\x00" + _u32(entry)
    if (sections[3] != _u32(count) + bytes(count)
            or sections[7] not in (b"\x02" + memory + run, b"\x02" + run + memory)):
        raise ValueError("Unexpected cartridge exports or function signatures")
    bodies, edges = [], []
    for i in range(count):
        size, pos = _read_u32(data, pos)
        try:
            body = lower_body(data[pos:pos+size])
            _, ops = _decode_control_body(body, True, 4096, direct_calls=True)
        except ValueError as exc:
            raise ValueError(f"Function {i}: {exc}") from exc
        bodies.append(body)
        edges.append([op.immediate for op in ops if op.opcode == "call"])
        pos += size
    assert pos == len(data)
    ordered, active = [], set()

    def visit(i):
        if i in active or i == entry and active:
            raise ValueError("Pocket requires acyclic direct calls")
        if i in ordered:
            return
        active.add(i)
        for target in edges[i]:
            visit(target)
        active.remove(i)
        ordered.append(i)

    for i in range(count):
        visit(i)
    remap = {prior: i for i, prior in enumerate(ordered)}
    sections[10] = _u32(count) + b"".join(
        _u32(len(body)) + body for body in (lower_body(bodies[i], remap) for i in ordered))
    output = bytearray(raw[:8])
    for sid, payload in sections.items():
        output.extend(bytes((sid,)) + _u32(len(payload)) + payload)
    wasmtime.Module.validate(engine, bytes(output))
    return bytes(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_bytes(lower(args.input.read_bytes()))
