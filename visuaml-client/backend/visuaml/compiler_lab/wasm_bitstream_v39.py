"""Fail-closed bounded WebAssembly V1 section transformer.

This performs *only* opaque custom-section relocation. It does not parse,
rewrite or certify executable instructions, imports, exports or functions.
Core-section raw bytes remain bit-identical and in original order. Custom
section order is retained, but external tooling may interpret custom metadata.
Independent full WASM validation and runtime comparison are separate gates.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

MAGIC = b"\x00asm\x01\x00\x00\x00"
MAX_BYTES = 2 * 1024 * 1024
# Canonical ordering of noncustom sections, including DataCount before Code.
SECTION_ORDER = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 7,
                 7: 8, 8: 9, 9: 10, 10: 12, 11: 13, 12: 11}


class WasmBinaryError(ValueError):
    """Invalid, unsupported or unbounded binary WASM section envelope."""


@dataclass(frozen=True)
class Section:
    section_id: int
    offset: int
    raw: bytes
    payload: bytes


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _u32(value: int) -> bytes:
    if type(value) is not int or not 0 <= value <= 0xFFFFFFFF:
        raise WasmBinaryError("Invalid bounded u32")
    data = bytearray()
    while True:
        b = value & 0x7F
        value >>= 7
        data.append(b | (0x80 if value else 0))
        if not value:
            return bytes(data)


def _read_u32(data: bytes, position: int) -> tuple[int, int]:
    start = position
    number = 0
    for i in range(5):
        if position >= len(data):
            raise WasmBinaryError("Truncated LEB128")
        b = data[position]
        position += 1
        if i == 4 and b & 0xF0:
            raise WasmBinaryError("u32 LEB128 overflow")
        number |= (b & 0x7F) << (7 * i)
        if not b & 0x80:
            if data[start:position] != _u32(number):
                raise WasmBinaryError("Noncanonical LEB128")
            return number, position
    raise WasmBinaryError("Unterminated LEB128")


def parse_section_envelope(data: bytes) -> tuple[Section, ...]:
    if not isinstance(data, bytes) or len(data) > MAX_BYTES or not data.startswith(MAGIC):
        raise WasmBinaryError("Invalid, missing or oversized WASM v1 header")
    pos = len(MAGIC)
    seen = set()
    last = 0
    sections: list[Section] = []
    while pos < len(data):
        start = pos
        section_id = data[pos]
        pos += 1
        if section_id not in (0, *SECTION_ORDER):
            raise WasmBinaryError("Unsupported section kind")
        size, pos = _read_u32(data, pos)
        end = pos + size
        if end > len(data):
            raise WasmBinaryError("Section exceeds container bounds")
        payload = data[pos:end]
        if section_id == 0:
            # A WASM custom section begins with a length-prefixed UTF-8 name.
            name_size, body_start = _read_u32(payload, 0)
            if body_start + name_size > len(payload):
                raise WasmBinaryError("Custom-section name truncated")
            try:
                payload[body_start:body_start + name_size].decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise WasmBinaryError("Invalid custom-section UTF-8 name") from exc
        else:
            order = SECTION_ORDER[section_id]
            if section_id in seen or order <= last:
                raise WasmBinaryError("Duplicate or out-of-order core section")
            seen.add(section_id)
            last = order
        sections.append(Section(section_id, start, data[start:end], payload))
        pos = end
        if len(sections) > 128:
            raise WasmBinaryError("Too many sections")
    return tuple(sections)


def relocate_custom_sections(source: bytes, *, placement: str = "end") -> tuple[bytes, dict]:
    """Move *opaque* custom sections to front/end; never edit core bytes."""
    if placement not in ("start", "end"):
        raise WasmBinaryError("Custom-section placement must be start or end")
    sections = parse_section_envelope(source)
    custom = tuple(item for item in sections if item.section_id == 0)
    core = tuple(item for item in sections if item.section_id != 0)
    emitted = (*custom, *core) if placement == "start" else (*core, *custom)
    target = MAGIC + b"".join(item.raw for item in emitted)
    if len(target) > MAX_BYTES:
        raise WasmBinaryError("Rewritten binary too large")
    verified = parse_section_envelope(target)
    assert [x.raw for x in verified if x.section_id != 0] == [x.raw for x in core]
    assert [x.raw for x in verified if x.section_id == 0] == [x.raw for x in custom]
    receipt = {
        "schema": "visuaml.wasm-opaque-custom-relocation.v39",
        "source_sha256": _sha(source), "target_sha256": _sha(target),
        "placement": placement, "source_bytes": len(source),
        "target_bytes": len(target), "custom_sections": len(custom),
        "standard_sections": len(core),
        "standard_raw_sha256": _sha(b"".join(x.raw for x in core)),
        "custom_raw_sha256": _sha(b"".join(x.raw for x in custom)),
        "core_bytes_exactly_unchanged": True,
        "wasm_module_typecheck_or_runtime_validated": False,
        "limitations": (
            "Section-envelope reordering only. This is NOT code rewriting, "
            "full WASM module validation, or proof that custom-metadata "
            "consumers are unaffected. Use independent WASM validator/runtime."
        ),
    }
    return target, receipt
