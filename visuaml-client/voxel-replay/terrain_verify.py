"""Differential replay and independent transition model for the terrain ABI."""

import base64
import hashlib
import math
import struct
from collections import Counter
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from unittest.mock import patch

import wasmtime

from visuaml.compiler_lab.neural_microop_executor_v57 import execute_microop_routed_control
from visuaml.compiler_lab.wasm_control_flow_v51 import decode_control, encode_control

BASE = 256
PAGE_BYTES = 65_536
ABI_BYTES = 1856
GUEST = Path(__file__).with_name("guest") / "terrain_tick.wasm"


class Divergence(Exception):
    def __init__(self, tick, reasons, inputs, before, actual, reference):
        super().__init__(f"First divergence at tick {tick}: {', '.join(reasons)}")
        self.report = {
            "tick": tick, "reasons": reasons, "input": inputs,
            **{label: base64.b64encode(page).decode() for label, page in
               (("page_before_base64", before), ("numpy_page_base64", actual),
                ("wasmtime_page_base64", reference))},
        }


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(page, offset, values):
    struct.pack_into(f"<{len(values)}i", page, BASE + offset, *values)


def read(page, offset, count):
    return list(struct.unpack_from(f"<{count}i", page, BASE + offset))


def initial_page(initial):
    positions, terrain = initial["positions"], initial["materials"]
    if not (1 <= len(positions) <= len(terrain) <= 64
            and len(set(positions)) == len(positions)
            and all(type(m) is int and 0 <= m <= 0x7fffffff for m in terrain)
            and all(type(p) is int and 0 <= p < len(terrain) and terrain[p] == 0 for p in positions)):
        raise ValueError("Invalid terrain scene")
    page = bytearray(PAGE_BYTES)
    write(page, 0, [4 * len(positions)])
    write(page, 16, [4 * len(terrain)])
    write(page, 64, [4 * p for p in positions])
    write(page, 576, terrain)
    return page


def resolve(positions, terrain, inputs):
    """Resolve from separate immutable pre-tick occupancy and material snapshots."""
    moves, edit = inputs["moves"], inputs["edit"]
    if (len(moves) != len(positions) or any(type(m) is not int or m not in (-1, 0, 1) for m in moves)
            or type(edit) is not bool):
        raise ValueError("Invalid terrain input")
    targets = [p + m for p, m in zip(positions, moves, strict=True)]
    edit_cell = positions[0] + 1 if edit else None
    if any(not 0 <= t < len(terrain) for t in targets) or (edit and edit_cell >= len(terrain)):
        raise ValueError("Input exceeds terrain bounds")
    claims = Counter(targets)
    occupied = set(positions)
    materials = [terrain[t] for t in targets]
    conflicts = [claims[t] - 1 for t in targets]
    after_positions = [
        t if not material and t not in occupied and not conflict else p
        for p, t, material, conflict in zip(positions, targets, materials, conflicts, strict=True)
    ]
    after_terrain = terrain.copy()
    if edit:
        after_terrain[edit_cell] = 0
    return after_positions, after_terrain, materials, conflicts


def verify(session, expected=None):
    started = perf_counter_ns()
    raw = GUEST.read_bytes()
    if session["version"] != 1 or session["mode"] != "terrain" or digest(raw) != session["binary_sha256"]:
        raise ValueError("Unexpected terrain guest or replay revision")
    if expected is not None:
        if (expected["session"]["initial"] != session["initial"]
                or expected["session"]["binary_sha256"] != session["binary_sha256"]):
            raise ValueError("Replay starts from a different state or binary")
        if len(expected["ticks"]) != len(session["inputs"]):
            raise ValueError("Replay length differs from saved proof")
    program = decode_control(raw, scaled=True)
    if encode_control(program) != raw:
        raise ValueError("Guest changed during admission")
    engine = wasmtime.Engine()
    store = wasmtime.Store(engine)
    exports = wasmtime.Instance(store, wasmtime.Module(engine, raw), []).exports(store)
    positions, terrain = session["initial"]["positions"], session["initial"]["materials"]
    page = initial_page(session["initial"])
    initial_hash = digest(page)
    exports["memory"].write(store, page)
    preparation_ns = perf_counter_ns() - started
    ticks = []
    started = perf_counter_ns()
    with patch("visuaml.compiler_lab.neural_microstep_executor_v58._NativeMicrostep",
               side_effect=AssertionError("Unexpected packed fallback")):
        for index, inputs in enumerate(session["inputs"]):
            positions, terrain, materials, conflicts = resolve(positions, terrain, inputs)
            before = bytes(page)
            write(page, 4, [int(inputs["edit"])])
            write(page, 320, [4 * d for d in inputs["moves"]])
            exports["memory"].write(store, page[BASE + 4:BASE + 8], BASE + 4)
            exports["memory"].write(store, page[BASE + 320:BASE + 320 + 4 * len(positions)], BASE + 320)
            start = perf_counter_ns()
            reference_return = exports["run"](store, BASE)
            wasmtime_ns = perf_counter_ns() - start
            start = perf_counter_ns()
            actual = execute_microop_routed_control(
                program, BASE, array_memory=page, fuel=50_000, capture_trace=False,
            )
            numpy_ns = perf_counter_ns() - start
            reference_page = exports["memory"].read(store)
            current_hash = digest(page)
            reasons = []
            if (actual.backend != "v58-structured-numpy-memory" or actual.result_i32 != reference_return
                    or page != reference_page):
                reasons.append("engine state")
            if (read(page, 64, len(positions)) != [4 * p for p in positions]
                    or read(page, 576, len(terrain)) != terrain
                    or read(page, 1344, len(materials)) != materials
                    or read(page, 1600, len(conflicts)) != conflicts
                    or len(set(positions)) != len(positions)):
                reasons.append("independent rules")
            if expected is not None:
                if inputs != expected["session"]["inputs"][index]:
                    reasons.append("recorded input")
                if current_hash != expected["ticks"][index]["page_sha256"]:
                    reasons.append("recorded page hash")
            if reasons:
                raise Divergence(index + 1, reasons, inputs, before, page, reference_page)
            ticks.append({
                "page_sha256": current_hash, "positions": positions, "terrain": terrain,
                "materials": materials, "conflict_agents": sum(n > 0 for n in conflicts),
                "steps": actual.steps, "numpy_call_ns": numpy_ns, "wasmtime_call_ns": wasmtime_ns,
            })
    return {
        "session": session, "initial_page_sha256": initial_hash, "ticks": ticks,
        "module_preparation_ns": preparation_ns, "replay_ns": perf_counter_ns() - started,
        "engines": ["VisuaML fixed-weight NumPy", "Wasmtime"], "packed_fallback": False,
    }


def scale_session(agents, crowded, ticks):
    if crowded:
        positions = [5 * (i // 2) + (1 if i % 2 == 0 else 3) for i in range(agents)]
        materials = [0] * (5 * (agents // 2))
        directions = [[1, -1], [1, -1], [-1, 1], [1, -1]]
        moves = [directions[tick % len(directions)] * (agents // 2) for tick in range(ticks)]
    else:
        positions = [4 * i + 1 for i in range(agents)]
        materials = [0] * (4 * agents)
        moves = [[1 if tick % 2 == 0 else -1] * agents for tick in range(ticks)]
    return {"version": 1, "mode": "terrain", "binary_sha256": digest(GUEST.read_bytes()),
            "initial": {"positions": positions, "materials": materials},
            "inputs": [{"moves": row, "edit": False} for row in moves]}


def summarize(proof):
    ticks = proof["ticks"]
    agents = len(proof["session"]["initial"]["positions"])

    def timings(key):
        samples = sorted(tick[key] for tick in ticks)
        return {"median_ms": median(samples) / 1e6,
                "p95_ms": samples[math.ceil(.95 * len(samples)) - 1] / 1e6,
                "agent_ticks_per_second": agents * len(samples) * 1e9 / sum(samples)}

    return {
        "agents": agents, "cells": len(proof["session"]["initial"]["materials"]),
        "completed_ticks": len(ticks), "agent_ticks_in_conflict": sum(t["conflict_agents"] for t in ticks),
        "source_steps": sum(t["steps"] for t in ticks), "guest_page_bytes": PAGE_BYTES,
        "reserved_abi_bytes": ABI_BYTES, "numpy": timings("numpy_call_ns"),
        "wasmtime": timings("wasmtime_call_ns"), "replay_ns": proof["replay_ns"],
    }
