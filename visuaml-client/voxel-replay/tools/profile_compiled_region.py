"""Profile NumPy regions and surrounding gate work outside timing runs."""
import argparse
import cProfile
import inspect
import json
import pstats
import sys
from pathlib import Path
from time import perf_counter_ns
from unittest.mock import patch

from scripts.benchmark_neural_vm_v58 import environment
from operand_pairs import workload, run
from visuaml.compiler_lab import neural_microop_executor_v57 as vm

GROUPS = {
    "local_const_add": ("neural_microop_executor_v57.py", "_run_local_const_add_region"),
    "const_add": ("neural_microop_executor_v57.py", "_run_const_add_region"),
    "compiled_event": ("compiled_frame_event_v58.py", "select"),
    "numpy_event": ("neural_frame_events_v54.py", "select_frame_action"),
    "alu": ("neural_alu_v56.py", "_run_prepared_alu"),
    "adder": ("bit_neural_vm_v49.py", "_add"),
    "compare": ("bit_neural_vm_v49.py", "_signed_lt"),
    "local_read": ("neural_local_bank_v57.py", "read_local"),
    "operand_move": ("neural_operand_storage_v55.py", "step_operands"),
    "frame_event": ("neural_frame_events_v54.py", "select_frame_action"),
    "frame_update": ("neural_frame_circuit_v53.py", "step_frames"),
    "frame_equations": ("neural_frame_circuit_v53.py", "_step_frame_gates"),
    "route": ("neural_microop_executor_v57.py", "_array_route"),
    "driver": ("neural_microop_executor_v57.py", "execute_microop_routed_control"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("sparse", "crowded", "bfs", "sudoku"))
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--retain", choices=("off", "copy", "resident"), default="off")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    case = workload(args.case)
    settings = dict(prepare_operand_pop=True, prepare_alu=True, prepare_mul=True,
                    prepare_regions=True, prepare_local_region=True, use_compiled_frame_event=args.compiled,
                    compiled_frame_retain=args.retain, prepare_frame_inputs=args.compact,
                    compact_frame_inputs=args.compact)
    with patch("visuaml.compiler_lab.neural_microstep_executor_v58._NativeMicrostep",
               side_effect=AssertionError("Unexpected packed fallback")):
        expected = run(case, [], None, **settings)
        profiler = cProfile.Profile()
        profiler.runcall(run, case, [], expected, **settings)
        lines, start = inspect.getsourcelines(vm.execute_microop_routed_control)

        def line(marker):
            return start + next(i for i, text in enumerate(lines) if marker in text)

        mem_start = line("base = np.packbits")
        mem_end = line('elif kind == "i32.eqz"')
        memory_ns = 0

        def tracer(frame, event, _arg):
            if event != "call" or frame.f_code is not vm.execute_microop_routed_control.__code__:
                return None
            previous, timestamp = frame.f_lineno, perf_counter_ns()

            def trace(frame, _event, _arg):
                nonlocal previous, timestamp, memory_ns
                now = perf_counter_ns()
                if mem_start <= previous < mem_end:
                    memory_ns += now - timestamp
                previous, timestamp = frame.f_lineno, perf_counter_ns()
                return trace

            return trace

        sys.settrace(tracer)
        try:
            run(case, [], expected, **settings)
        finally:
            sys.settrace(None)
    stats = pstats.Stats(profiler)
    selected = {key: {"calls": count, "self_s": own, "cumulative_s": cumulative}
                for key, (file, name) in GROUPS.items()
                for (path, _, label), (_, count, own, cumulative, _) in stats.stats.items()
                if Path(path).name == file and label == name}
    print(json.dumps({"case": args.case, "environment": environment(), "profile_s": stats.total_tt,
                      "retain": args.retain, "compact": args.compact, "hotspots": selected, "memory_body_line_trace_ns": memory_ns}, indent=2))


if __name__ == "__main__":
    main()
