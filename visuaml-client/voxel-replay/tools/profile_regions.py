"""Warm source-sequence counts and subsystem profiling with the prepared ALU."""
import argparse
import cProfile
import inspect
import json
import pstats
import sys
from collections import Counter, deque
from pathlib import Path
from unittest.mock import patch

from scripts.benchmark_neural_vm_v58 import environment
from visuaml.compiler_lab import neural_alu_v56 as alu
from visuaml.compiler_lab import neural_microop_executor_v57 as vm
from visuaml.compiler_lab.wasm_control_flow_v51 import ControlFuelExhausted, decode_control

import terrain_verify as terrain
from operand_pairs import workload

HOTSPOTS = {
    "fixed_add": ("bit_neural_vm_v49.py", "_add"),
    "fixed_mul": ("bit_neural_vm_v49.py", "_mul"),
    "signed_lt": ("bit_neural_vm_v49.py", "_signed_lt"),
    "driver": ("neural_microop_executor_v57.py", "execute_microop_routed_control"),
    "route": ("neural_microop_executor_v57.py", "_array_route"),
    "frame_event": ("neural_frame_events_v54.py", "select_frame_action"),
    "frame_state": ("neural_frame_circuit_v53.py", "step_frames"),
    "operand_state": ("neural_operand_storage_v55.py", "step_operands"),
    "pop": ("neural_operand_storage_v55.py", "_unit_pop"),
    "top": ("neural_operand_storage_v55.py", "top_word"),
    "selected": ("neural_alu_v56.py", "_run_prepared_alu"),
    "all_candidates": ("neural_alu_v56.py", "execute_alu"),
}


def profile(case, limit):
    if isinstance(case, dict):
        program = decode_control(terrain.GUEST.read_bytes(), scaled=True)
        argument = terrain.BASE
        initial = terrain.initial_page(case["initial"])
        terrain.write(initial, 320, [4 * d for d in case["inputs"][0]["moves"]])
        fuel = 50_000
    else:
        program, initial, argument, _, _, steps, _ = case
        fuel = min(steps, limit) if limit else steps

    def run():
        page = initial.copy()
        try:
            return vm.execute_microop_routed_control(program, argument, array_memory=page,
                                                    capture_trace=False, fuel=fuel, prepare_mul=False, prepare_regions=False).steps
        except ControlFuelExhausted:
            assert limit and fuel == limit
            return fuel

    with patch("visuaml.compiler_lab.neural_microstep_executor_v58._NativeMicrostep",
               side_effect=AssertionError("Unexpected packed fallback")):
        # A short warmup covers the same decoder, router and kernels.
        try:
            vm.execute_microop_routed_control(program, argument, array_memory=initial.copy(),
                                              capture_trace=False, fuel=min(500, fuel), prepare_mul=False, prepare_regions=False)
        except ControlFuelExhausted:
            pass

        lines, first = inspect.getsourcelines(vm.execute_microop_routed_control)
        def locate(token):
            return next(first + i for i, line in enumerate(lines) if token in line)

        opcode_line = locate("kind = op.opcode")
        memory_start = locate("base = np.packbits")
        memory_stop = locate('elif kind == "i32.eqz"')
        windows = Counter()
        unused = Counter()
        previous = deque(maxlen=3)
        last = None
        memory_ns = 0

        def tracer(frame, event, _arg):
            nonlocal memory_ns, last
            if event != "call":
                return None
            if frame.f_code is alu.execute_alu.__code__:
                unused[alu.ALU_OPS[int(frame.f_locals["opcode"].argmax())]] += 1
                return None
            if frame.f_code is not vm.execute_microop_routed_control.__code__:
                return None
            from time import perf_counter_ns
            line = frame.f_lineno
            stamp = perf_counter_ns()

            def trace(frame, _event, _arg):
                nonlocal line, stamp, memory_ns, last
                now = perf_counter_ns()
                if line == opcode_line:
                    pc, kind = frame.f_locals["pc"], frame.f_locals["kind"]
                    if last is None or pc != last + 1:
                        previous.clear()
                    last = pc
                    previous.append(kind)
                    if len(previous) >= 2:
                        windows[" ".join(tuple(previous)[-2:])] += 1
                    if len(previous) == 3:
                        windows[" ".join(previous)] += 1
                if memory_start <= line < memory_stop:
                    memory_ns += now - stamp
                line, stamp = frame.f_lineno, perf_counter_ns()
                return trace

            return trace

        sys.settrace(tracer)
        try:
            steps = run()
        finally:
            sys.settrace(None)
        profiler = cProfile.Profile()
        assert profiler.runcall(run) == steps
        stats = pstats.Stats(profiler)
    return {"steps": steps, "windows": dict(windows), "generic_alu_opcodes": dict(unused),
            "memory_body_line_trace_ns": memory_ns, "profile_seconds": stats.total_tt,
            "hotspots": {
                key: {"calls": nc, "self_seconds": own, "cumulative_seconds": end}
                for key, (file, name) in HOTSPOTS.items()
                for (path, _, label), (_, nc, own, end, _) in stats.stats.items()
                if label == name and Path(path).name == file
            }}


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("case", choices=("sparse", "crowded", "bfs", "sudoku"))
    args.add_argument("--limit", type=int)
    choice = args.parse_args()
    print(json.dumps({"case": choice.case, "environment": environment(),
                      **profile(workload(choice.case), choice.limit)}, indent=2))


if __name__ == "__main__":
    main()
