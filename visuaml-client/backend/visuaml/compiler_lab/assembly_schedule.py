"""Declarative assembly-network schedules lowered into bounded state/effect IR.

Admitted subset: explicit float32 weight matrices, deterministic k-WTA, local
Hebbian updates, ordered projections and fixed source winner counts. This is
not the entire neural-assemblies runtime: no stochastic recruitment, custom
winner policies, inhibition, neuron-ID remapping or delayed plasticity.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

import numpy as np

from .effect_ir import (
    EffectIRError,
    EffectProgram,
    Instruction,
    Repeat,
    StateSlot,
    TensorType,
    validate_effect_program,
)
from .effect_runtime import EffectExecution, execute_effect_program


@dataclass(frozen=True)
class Area:
    name: str
    population: int
    winners: int


@dataclass(frozen=True)
class Fiber:
    source: str
    target: str
    beta: float
    ceiling: float = 20.0


@dataclass(frozen=True)
class Project:
    target: str
    sources: tuple[str, ...]


@dataclass(frozen=True)
class AssemblySchedule:
    areas: tuple[Area, ...]
    fibers: tuple[Fiber, ...]
    projections: tuple[Project, ...]
    rounds: int = 1


@dataclass(frozen=True)
class LoweredSchedule:
    program: EffectProgram
    schedule: AssemblySchedule

    def execute(self, initial_state: Mapping[str, np.ndarray]) -> EffectExecution:
        check_initial_state(self.schedule, initial_state)
        return execute_effect_program(self.program, initial_state)


def _named(name: str) -> bool:
    return isinstance(name, str) and bool(
        re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,31}", name)
    )


def validate_schedule(spec: AssemblySchedule) -> None:
    if not isinstance(spec, AssemblySchedule) or not 1 <= len(spec.areas) <= 16:
        raise EffectIRError("Invalid or empty assembly network")
    if type(spec.rounds) is not int or not 1 <= spec.rounds <= 128:
        raise EffectIRError("Unbounded assembly schedule")
    areas = {}
    for area in spec.areas:
        if (
            not isinstance(area, Area)
            or not _named(area.name)
            or area.name in areas
            or type(area.population) is not int
            or type(area.winners) is not int
            or not 1 <= area.winners <= area.population <= 1024
        ):
            raise EffectIRError("Invalid population, winner count, or duplicate area")
        areas[area.name] = area
    if (
        not spec.fibers
        or len(spec.fibers) > 64
        or not spec.projections
        or len(spec.projections) > 24
    ):
        raise EffectIRError("Invalid number of fibers or projections")
    fibers = set()
    for fiber in spec.fibers:
        if (
            not isinstance(fiber, Fiber)
            or fiber.source not in areas
            or fiber.target not in areas
            or (fiber.source, fiber.target) in fibers
        ):
            raise EffectIRError("Unknown or duplicate fiber")
        if (
            any(
                type(v) not in (float, int) or not isfinite(v)
                for v in (fiber.beta, fiber.ceiling)
            )
            or fiber.beta < 0
            or fiber.ceiling <= 0
        ):
            raise EffectIRError("Invalid Hebbian parameters")
        fibers.add((fiber.source, fiber.target))
    for projection in spec.projections:
        if (
            not isinstance(projection, Project)
            or projection.target not in areas
            or not 1 <= len(projection.sources) <= 8
            or len(set(projection.sources)) != len(projection.sources)
            or any(
                (source, projection.target) not in fibers
                for source in projection.sources
            )
        ):
            raise EffectIRError("Projection references missing/duplicate fibers")


def lower_assembly_schedule(spec: AssemblySchedule) -> LoweredSchedule:
    """Compile any admitted ordered multi-area, multi-fiber projection schedule.

    State changes are represented by linear tokens and per-cell SSA revisions.
    Source order is semantically significant for float32 drive accumulation.
    """
    validate_schedule(spec)
    areas = {a.name: a for a in spec.areas}
    fibers = {(f.source, f.target): f for f in spec.fibers}
    state = [
        StateSlot(f"{a.name}.winners", TensorType((a.winners,), "int64"))
        for a in spec.areas
    ]
    state += [
        StateSlot(
            f"{f.source}.{f.target}.weights",
            TensorType(
                (areas[f.source].population, areas[f.target].population), "float32"
            ),
        )
        for f in spec.fibers
    ]
    versions = {slot.name: 0 for slot in state}
    token_count = 0
    instructions: list[Instruction] = []

    def add_effect(
        name: str,
        op: str,
        cell: str,
        *,
        result: str | None = None,
        value: str | None = None,
    ) -> None:
        nonlocal token_count
        instructions.append(
            Instruction(
                name,
                op,
                args=(value,) if value is not None else (),
                output=result,
                cell=cell,
                token_in=f"e{token_count}",
                token_out=f"e{token_count + 1}",
                revision=versions[cell],
            )
        )
        token_count += 1
        if op == "write":
            versions[cell] += 1

    for pidx, step in enumerate(spec.projections):
        tag = f"p{pidx}"
        sources = []
        for j, source in enumerate(step.sources):
            fiber = fibers[(source, step.target)]
            ix, weights = f"{tag}_source{j}", f"{tag}_weights{j}"
            add_effect(f"{tag}_read_source{j}", "read", f"{source}.winners", result=ix)
            add_effect(
                f"{tag}_read_weights{j}",
                "read",
                f"{source}.{step.target}.weights",
                result=weights,
            )
            drive = f"{tag}_drive{j}"
            instructions.append(
                Instruction(
                    f"{tag}_drive_for_source{j}",
                    "gather_sum",
                    args=(weights, ix),
                    output=drive,
                )
            )
            sources.append((fiber, ix, weights, drive))
        acc = sources[0][3]
        for j, (_, _, _, drive) in enumerate(sources[1:], 1):
            next_acc = f"{tag}_combined{j}"
            instructions.append(
                Instruction(
                    f"{tag}_sum_drive{j}", "add", args=(acc, drive), output=next_acc
                )
            )
            acc = next_acc
        selected = f"{tag}_selected"
        instructions.append(
            Instruction(
                f"{tag}_select",
                "topk_indices",
                args=(acc,),
                output=selected,
                params=(("k", areas[step.target].winners),),
            )
        )
        for j, (fiber, ix, weights, _) in enumerate(sources):
            learned = f"{tag}_learned{j}"
            instructions.append(
                Instruction(
                    f"{tag}_hebbian{j}",
                    "hebbian_update",
                    args=(weights, ix, selected),
                    output=learned,
                    params=(("beta", fiber.beta), ("ceiling", fiber.ceiling)),
                )
            )
            add_effect(
                f"{tag}_write_weights{j}",
                "write",
                f"{fiber.source}.{fiber.target}.weights",
                value=learned,
            )
        add_effect(
            f"{tag}_publish_winners", "write", f"{step.target}.winners", value=selected
        )
    program = EffectProgram(
        tuple(state),
        (),
        (Repeat("network_schedule", spec.rounds, tuple(instructions)),),
    )
    validate_effect_program(program)
    return LoweredSchedule(program, spec)


def check_initial_state(
    spec: AssemblySchedule, state: Mapping[str, np.ndarray]
) -> None:
    validate_schedule(spec)
    areas = {a.name: a for a in spec.areas}
    needed = {f"{a.name}.winners" for a in spec.areas} | {
        f"{f.source}.{f.target}.weights" for f in spec.fibers
    }
    if set(state) != needed:
        raise EffectIRError("Initial state must provide exactly the declared cells")
    for area in spec.areas:
        winners = np.asarray(state[f"{area.name}.winners"])
        if (
            winners.shape != (area.winners,)
            or winners.dtype != np.int64
            or np.any(winners < 0)
            or np.any(winners >= area.population)
            or np.unique(winners).size != area.winners
        ):
            raise EffectIRError("Malformed, duplicate, or out-of-range source winners")
    for fiber in spec.fibers:
        weights = np.asarray(state[f"{fiber.source}.{fiber.target}.weights"])
        if (
            weights.shape
            != (areas[fiber.source].population, areas[fiber.target].population)
            or weights.dtype != np.float32
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0)
            or np.any(weights > fiber.ceiling)
        ):
            raise EffectIRError("Weight matrix violates explicit-fiber assumptions")
