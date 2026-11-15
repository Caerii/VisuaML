"""Resource-constrained *estimated* host/device/link schedule for static tensor DAGs.

Unlike the earlier additive graph cut, this scheduler models independent host,
device, and serialized transfer resources, optional transfer reuse, and a
conservative device-residency admission. All durations are *provided model
inputs*, not observed execution times. Single FIFO resource queues are used.
This is a deterministic feasible schedule, NOT globally optimal scheduling.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Mapping

from visuaml.compiler_spike import ProgramIR


class ScheduleError(ValueError):
    pass


@dataclass(frozen=True)
class ScheduledTask:
    name: str
    kind: str
    resource: str
    start_us: float
    finish_us: float
    bytes: int = 0


@dataclass(frozen=True)
class ScheduleReport:
    schema: str
    tasks: tuple[ScheduledTask, ...]
    makespan_us: float
    total_work_us: float
    transfers: int
    moved_bytes: int
    device_residency_upper_bound: int
    input_sha256: str
    warnings: tuple[str, ...]

    def as_dict(self) -> dict:
        return asdict(self)


def estimate_schedule(
    ir: ProgramIR,
    assignments: Mapping[str, str],
    operation_us: Mapping[str, float],
    *,
    transfer_us: float,
    us_per_byte: float,
    device_memory_budget: int = 256 * 1024 * 1024,
    cache_transfers: bool = True,
) -> ScheduleReport:
    """Produce a reproducible schedule with concurrent independent resources.

    Host/device operation lanes execute one operation at a time; the single
    bidirectional link serializes transfers. Node order is a deterministic
    topological schedule heuristic, not a globally optimal placement/schedule.
    """
    if not isinstance(ir, ProgramIR) or len(ir.nodes) > 1024:
        raise ScheduleError("Invalid or oversized input graph")
    if type(cache_transfers) is not bool or type(device_memory_budget) is not int or device_memory_budget <= 0:
        raise ScheduleError("Invalid cache/memory policy")
    if any(type(v) not in (float, int) or not isfinite(v) or v < 0 for v in (transfer_us, us_per_byte)):
        raise ScheduleError("Transfer costs must be finite nonnegative")
    names = {n.id for n in ir.nodes}
    if len(names) != len(ir.nodes) or set(assignments) != names:
        raise ScheduleError("Assignments must cover each SSA node exactly")
    compute_nodes = {n.id for n in ir.nodes if n.op not in ("input", "output", "constant")}
    if set(operation_us) != compute_nodes:
        raise ScheduleError("Compute costs must cover every computational node")
    if any(type(v) not in (float, int) or not isfinite(v) or v < 0 for v in operation_us.values()):
        raise ScheduleError("Invalid execution duration")
    size: dict[str, int] = {}
    for node in ir.nodes:
        if assignments[node.id] not in ("host", "device"):
            raise ScheduleError("Unknown resource assignment")
        if node.op in ("input", "output", "constant") and assignments[node.id] != "host":
            raise ScheduleError("External input/output/constant must be host-owned")
        if node.op != "output" and (not node.tensors or any(t.bytes is None or t.bytes < 0 for t in node.tensors)):
            raise ScheduleError(f"Unknown tensor size for {node.id}")
        size[node.id] = sum(t.bytes or 0 for t in node.tensors)
    # A deliberately conservative capacity bound. Any device-computed output or
    # host->device copied value might remain resident until the end of a graph.
    resident = sum(size[n.id] for n in ir.nodes if assignments[n.id] == "device")
    resident += sum(size[src] for src in names if assignments[src] == "host"
                    and any(assignments.get(n.id) == "device" and src in n.inputs for n in ir.nodes))
    if resident > device_memory_budget:
        raise ScheduleError("Conservative device memory-residency budget exceeded")

    clocks = {"host": 0.0, "device": 0.0, "link": 0.0}
    produced: dict[str, float] = {}
    transferred: dict[tuple[str, str], float] = {}
    records: list[ScheduledTask] = []
    for node in ir.nodes:
        ready = 0.0
        for source in node.inputs:
            if source not in produced:
                raise ScheduleError(f"Missing or forward dependency {source}->{node.id}")
            earliest = produced[source]
            source_adapter, target_adapter = assignments[source], assignments[node.id]
            if source_adapter != target_adapter:
                key = (source, target_adapter)
                if not cache_transfers or key not in transferred:
                    start = max(earliest, clocks["link"])
                    finish = start + transfer_us + us_per_byte * size[source]
                    if not isfinite(finish):
                        raise ScheduleError("Transfer schedule overflow")
                    clocks["link"] = finish
                    records.append(ScheduledTask(f"{source}->{node.id}", "copy", "link", start, finish, size[source]))
                    if cache_transfers:
                        transferred[key] = finish
                earliest = transferred[key] if cache_transfers else clocks["link"]
            ready = max(ready, earliest)
        resource = assignments[node.id]
        start = max(ready, clocks[resource])
        duration = operation_us.get(node.id, 0.0)
        finish = start + duration
        if not isfinite(finish):
            raise ScheduleError("Execution schedule overflow")
        clocks[resource] = finish
        produced[node.id] = finish
        if node.id in compute_nodes:
            records.append(ScheduledTask(node.id, node.op, resource, start, finish, size[node.id]))
    copies = [t for t in records if t.kind == "copy"]
    makespan = max(produced.values(), default=0.0)
    payload = {
        "graph": ir.as_dict(), "assignments": sorted(assignments.items()),
        "execution": sorted(operation_us.items()), "transfer_us": transfer_us,
        "us_per_byte": us_per_byte, "memory_budget": device_memory_budget,
        "cache_transfers": cache_transfers,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return ScheduleReport("visuaml.schedule-estimate.v1", tuple(records), makespan,
                          sum(operation_us.values()) + sum(t.finish_us-t.start_us for t in copies),
                          len(copies), sum(t.bytes for t in copies), resident, digest,
                          ("Durations are modeled inputs, not observed device timings.",
                           "Host and device may overlap; link is a single FIFO resource.",
                           "Device memory is a conservative all-values-resident upper bound, not a peak trace.",
                           "This is a feasible greedy schedule, not a global scheduling optimum."))
