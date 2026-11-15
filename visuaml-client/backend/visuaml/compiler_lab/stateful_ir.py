"""Effect-typed stateful assembly step IR (research subset, schema v1).

This IR makes state dependencies explicit; it does NOT prove that a graph rewrite
preserves numerical results. Numeric witness checks must accompany any accepted
pure-compute rewrite. It deliberately rejects effect reordering, state aliasing
and omitted learning writes instead of treating stateful work as pure FX.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

SCHEMA = "visuaml.stateful-assembly-ir.v1"
STATE_KEYS = frozenset({"weights", "bias", "winners"})
EFFECT_OPS = ("hebbian_write", "refraction_write", "publish_winners")


class StatefulIRError(ValueError):
    """An invalid stateful graph, state version, or effectful rewrite."""


@dataclass(frozen=True)
class Instruction:
    name: str
    op: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    reads: tuple[tuple[str, int], ...] = ()
    writes: tuple[tuple[str, int], ...] = ()
    token_in: str | None = None
    token_out: str | None = None


@dataclass(frozen=True)
class StatefulProgram:
    source_mode: Literal["external", "recurrent"]
    inputs: tuple[str, ...]
    instructions: tuple[Instruction, ...]
    schema: str = SCHEMA

    def fingerprint(self) -> str:
        wire = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(wire.encode("utf-8")).hexdigest()


def make_assembly_step_ir(
    source_mode: Literal["external", "recurrent"] = "recurrent",
) -> StatefulProgram:
    """Emit explicit read-compute-select-write barriers for one area step."""
    if source_mode not in ("external", "recurrent"):
        raise StatefulIRError("Unknown source mode")
    program = make_assembly_step_ir_unchecked(source_mode)
    validate_stateful_ir(program)
    return program


def validate_stateful_ir(program: StatefulProgram) -> None:
    """Check SSA values, current state versions, and linear effect tokens."""
    if program.schema != SCHEMA or program.source_mode not in ("external", "recurrent"):
        raise StatefulIRError("Unsupported stateful schema or source mode")
    expected_inputs = () if program.source_mode == "recurrent" else ("given_source",)
    if program.inputs != expected_inputs:
        raise StatefulIRError("External source declaration mismatch")
    if len(program.instructions) > 128:
        raise StatefulIRError("Stateful operation budget exceeded")
    versions = {name: 0 for name in STATE_KEYS}
    values = set(program.inputs)
    names: set[str] = set()
    token = "state_0"
    effects: list[str] = []
    source_reads: list[str] = []
    for instr in program.instructions:
        if not instr.name or instr.name in names or not instr.op:
            raise StatefulIRError("Duplicate or empty instruction")
        names.add(instr.name)
        if not set(instr.inputs) <= values:
            raise StatefulIRError(f"Use before definition in {instr.name}")
        if len(set(instr.outputs)) != len(instr.outputs) or any(
            x in values or not x for x in instr.outputs
        ):
            raise StatefulIRError(f"Duplicate or invalid SSA output in {instr.name}")
        reads = dict(instr.reads)
        writes = dict(instr.writes)
        if len(reads) != len(instr.reads) or len(writes) != len(instr.writes):
            raise StatefulIRError("Duplicate state access")
        if not (set(reads) | set(writes)) <= STATE_KEYS:
            raise StatefulIRError("Unknown state cell")
        for key, ver in reads.items():
            if type(ver) is not int or versions[key] != ver:
                raise StatefulIRError(f"Stale state read: {instr.name}/{key}")
        if instr.op in EFFECT_OPS:
            effects.append(instr.op)
            if (
                instr.token_in != token
                or not instr.token_out
                or instr.token_out == token
            ):
                raise StatefulIRError("Effect token does not follow the prior write")
            if len(writes) != 1 or len(reads) != 1 or set(writes) != set(reads):
                raise StatefulIRError(
                    "Each effect must read and write its own state cell"
                )
            key = next(iter(writes))
            if writes[key] != versions[key] + 1:
                raise StatefulIRError("State version must increase by exactly one")
            expected_key = {
                "hebbian_write": "weights",
                "refraction_write": "bias",
                "publish_winners": "winners",
            }[instr.op]
            if key != expected_key:
                raise StatefulIRError("Effect operation has wrong state effect")
            if instr.outputs:
                raise StatefulIRError("Effectful writes cannot masquerade as pure SSA")
            versions[key] += 1
            token = instr.token_out
        else:
            if instr.writes or instr.token_in or instr.token_out:
                raise StatefulIRError("Pure operation declares an effect")
            source_reads.append(instr.op)
            if instr.op not in {
                "snapshot_previous_winners",
                "read_external_source",
                "weighted_drive",
                "subtract_refraction",
                "stable_k_winners",
            }:
                raise StatefulIRError("Unknown pure stateful operator")
        values.update(instr.outputs)
    # This is a deliberately narrow, admitted computational dialect. A different
    # stateful algorithm must declare a NEW schema and its own semantic rules.
    expected_pure = [
        "snapshot_previous_winners"
        if program.source_mode == "recurrent"
        else "read_external_source",
        "weighted_drive",
        "subtract_refraction",
        "stable_k_winners",
    ]
    if source_reads != expected_pure or effects != list(EFFECT_OPS):
        raise StatefulIRError("Step is missing or reorders required reads/writes")
    if any(versions[key] != 1 for key in STATE_KEYS):
        raise StatefulIRError("Incomplete state transition")
    # Freeze the wiring, not only the labels: otherwise a pass could claim
    # the correct effects while rewriting their inputs to unrelated values.
    canonical = make_assembly_step_ir_unchecked(program.source_mode)
    if program.instructions != canonical.instructions:
        raise StatefulIRError(
            "Noncanonical step wiring; requires a new dialect contract"
        )


def make_assembly_step_ir_unchecked(
    source_mode: Literal["external", "recurrent"],
) -> StatefulProgram:
    """Internal canonical template, split out to avoid recursive validation."""
    source = (
        Instruction(
            "source",
            "snapshot_previous_winners",
            outputs=("source_indices",),
            reads=(("winners", 0),),
        )
        if source_mode == "recurrent"
        else Instruction(
            "source",
            "read_external_source",
            inputs=("given_source",),
            outputs=("source_indices",),
        )
    )
    return StatefulProgram(
        source_mode,
        () if source_mode == "recurrent" else ("given_source",),
        (
            source,
            Instruction(
                "drive",
                "weighted_drive",
                inputs=("source_indices",),
                outputs=("raw_drive",),
                reads=(("weights", 0),),
            ),
            Instruction(
                "inhibit",
                "subtract_refraction",
                inputs=("raw_drive",),
                outputs=("net_drive",),
                reads=(("bias", 0),),
            ),
            Instruction(
                "select",
                "stable_k_winners",
                inputs=("net_drive",),
                outputs=("selected",),
            ),
            Instruction(
                "learn",
                "hebbian_write",
                inputs=("source_indices", "selected"),
                reads=(("weights", 0),),
                writes=(("weights", 1),),
                token_in="state_0",
                token_out="state_1",
            ),
            Instruction(
                "refraction",
                "refraction_write",
                inputs=("raw_drive", "selected"),
                reads=(("bias", 0),),
                writes=(("bias", 1),),
                token_in="state_1",
                token_out="state_2",
            ),
            Instruction(
                "publish",
                "publish_winners",
                inputs=("selected",),
                reads=(("winners", 0),),
                writes=(("winners", 1),),
                token_in="state_2",
                token_out="state_3",
            ),
        ),
    )


def assert_effects_preserved(before: StatefulProgram, after: StatefulProgram) -> None:
    """Conservative pass admission: effectful transformations are never silent."""
    validate_stateful_ir(before)
    validate_stateful_ir(after)
    if before.source_mode != after.source_mode:
        raise StatefulIRError("Source mode changed")
    # v1 is intentionally closed-world: even pure rewrites need a new dialect
    # contract plus numerical witnesses before admission.
    before_effects = tuple(i for i in before.instructions if i.op in EFFECT_OPS)
    after_effects = tuple(i for i in after.instructions if i.op in EFFECT_OPS)
    if before_effects != after_effects:
        raise StatefulIRError("Compiler pass altered effect order or state access")
