"""Observable state and access claims for a bounded memory transformation."""
from __future__ import annotations

from dataclasses import dataclass

from .wasm_control_flow_v51 import ControlFuelExhausted, LinearMemoryTrap


@dataclass(frozen=True)
class RegionObservation:
    exit: str
    result: int | None
    page: bytes
    source_steps: int | None = None
    accesses: tuple[dict, ...] | None = None


def observe(execute, argument, initial, *, fuel=400, capture_trace=False):
    """Capture committed memory on both success and known bounded traps."""
    page = bytearray(initial)
    try:
        result = execute(argument, page, fuel=fuel, capture_trace=capture_trace)
    except LinearMemoryTrap:
        return RegionObservation("memory_oob", None, bytes(page))
    except ControlFuelExhausted:
        return RegionObservation("fuel", None, bytes(page))
    accesses = None
    if capture_trace:
        accesses = tuple(row["memory_access"] for row in result.trace if row["memory_access"] is not None)
    return RegionObservation("return", result.result_i32, bytes(page), result.steps, accesses)


def first_difference(expected, actual, *, source_steps=False, access_order=False):
    """Compare terminal effects and optionally require instruction or access claims."""
    if access_order and (expected.accesses is None or actual.accesses is None):
        raise ValueError("Access-order verification requires both traces")
    if source_steps and (expected.source_steps is None or actual.source_steps is None):
        raise ValueError("Source-step verification requires both counts")
    for field in ("exit", "result"):
        left, right = getattr(expected, field), getattr(actual, field)
        if left != right:
            return {"field": field, "expected": left, "actual": right}
    if expected.page != actual.page:
        for address, (left, right) in enumerate(zip(expected.page, actual.page)):
            if left != right:
                return {"field": "memory", "address": address, "expected": left, "actual": right}
        return {"field": "memory_length", "expected": len(expected.page), "actual": len(actual.page)}
    if source_steps and expected.source_steps != actual.source_steps:
        return {"field": "source_steps", "expected": expected.source_steps, "actual": actual.source_steps}
    if access_order:
        return first_access_difference(expected.accesses, actual.accesses)
    return None


def first_access_difference(expected, actual):
    """Locate the first unequal access independently of terminal state."""
    if expected is None or actual is None:
        raise ValueError("Access-order verification requires both traces")
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left != right:
            return {"field": "access_order", "index": index, "expected": left, "actual": right}
    if len(expected) != len(actual):
        return {"field": "access_count", "expected": len(expected), "actual": len(actual)}
    return None


def first_counterexample(reference, candidate, cases, **claims):
    """Find the first witness in the caller's declared, ordered domain."""
    for case in cases:
        difference = first_difference(reference(case), candidate(case), **claims)
        if difference is not None:
            return case, difference
    return None
