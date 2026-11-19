"""Paired before/after measurement of the frame retain contraction."""
import hashlib
import json
import subprocess
import sys
import types
from time import perf_counter_ns
from unittest.mock import patch

from scripts.benchmark_neural_vm_v58 import environment
from scripts.stateful_scale_workloads_v58 import graph_case
from visuaml.compiler_lab import neural_frame_circuit_v53 as frames
from visuaml.compiler_lab import neural_microop_executor_v57 as vm
from visuaml.compiler_lab.wasm_control_flow_v51 import ControlFuelExhausted, decode_control

import terrain_verify
from scale_terrain import ARCHIVE

BASELINE = "2588452e6474fafc71ac2580a07362311e567d28"
FRAME_PATH = "visuaml-client/backend/visuaml/compiler_lab/neural_frame_circuit_v53.py"


def original_frame():
    name = "visuaml.compiler_lab._baseline_frame"
    module = types.ModuleType(name)
    module.__package__ = "visuaml.compiler_lab"
    sys.modules[name] = module
    raw = subprocess.check_output(["git", "show", f"{BASELINE}:{FRAME_PATH}"])
    exec(compile(raw, FRAME_PATH, "exec"), module.__dict__)
    for name in ("FrameState", "FrameTokens", "FrameCircuitWeights"):
        setattr(module, name, getattr(frames, name))
    return module.step_frames


def measure(call, samples):
    route = vm._array_route
    first = None

    def first_route(*args):
        nonlocal first
        first = perf_counter_ns()
        vm._array_route = route
        return route(*args)

    with patch.object(vm, "_array_route", first_route):
        started = perf_counter_ns()
        try:
            return call()
        finally:
            ended = perf_counter_ns()
            samples.append({"before_first_route_ns": first - started, "execution_ns": ended - first})


def main():
    original = original_frame()
    choices = {"original": original, "unit_retain": vm.step_frames}
    data = {workload: {choice: [] for choice in choices} for workload in ("world", "bfs")}
    world = terrain_verify.scale_session(8, True, 4)
    source = decode_control((ARCHIVE / "bfs.wasm").read_bytes(), scaled=True)
    argument, page, *_ = graph_case(0)
    reference_world = reference_bfs = None

    for pair in range(4):
        order = tuple(choices) if pair % 2 == 0 else tuple(reversed(choices))
        for name in order:
            with patch.object(vm, "step_frames", choices[name]):
                calls = []

                def execute(*args, **kwargs):
                    return measure(lambda: vm.execute_microop_routed_control(*args, **kwargs), calls)

                with patch.object(terrain_verify, "execute_microop_routed_control", execute):
                    proof = terrain_verify.verify(world)
                hashes = [(row["steps"], row["page_sha256"]) for row in proof["ticks"]]
                if reference_world is None:
                    reference_world = hashes
                assert hashes == reference_world
                if pair:
                    data["world"][name].append({"module_preparation_ns": proof["module_preparation_ns"],
                                                 "calls": calls})

                state = page.copy()
                calls = []
                try:
                    execute(source, argument, fuel=2000, capture_trace=False, array_memory=state)
                except ControlFuelExhausted:
                    current = hashlib.sha256(state).hexdigest()
                    if reference_bfs is None:
                        reference_bfs = current
                    assert current == reference_bfs
                else:
                    raise AssertionError("Expected BFS cutoff")
                if pair:
                    data["bfs"][name].append({"calls": calls})

    print(json.dumps({"environment": environment(), "baseline": BASELINE,
                      "world_agents": 8, "world_ticks_per_sample": 4, "bfs_steps": 2000,
                      "pairs_after_warmup": 3, "cases": data}, indent=2))


if __name__ == "__main__":
    main()
