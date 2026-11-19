"""Measure checked-pop phases outside benchmark timing runs."""
import cProfile
import inspect
import json
import pstats
import sys
from collections import Counter
from time import perf_counter_ns
from unittest.mock import patch

from scripts.benchmark_neural_vm_v58 import environment
from visuaml.compiler_lab import neural_microop_executor_v57 as vm
from visuaml.compiler_lab.wasm_control_flow_v51 import ControlFuelExhausted, decode_control

from visuaml.compiler_lab.neural_operand_storage_v55 import _unit_pop

import terrain_verify as terrain
from operand_pairs import workload


def profile(program, argument, initial, fuel):
    from visuaml.compiler_lab.neural_operand_storage_v55 import step_operands

    def run(prepared=False):
        try:
            result = vm.execute_microop_routed_control(
                program, argument, array_memory=initial.copy(), fuel=fuel,
                capture_trace=False, prepare_operand_pop=prepared, prepare_alu=False,
            )
            return result.steps
        except ControlFuelExhausted:
            return fuel

    def line_of(function, marker):
        lines, first = inspect.getsourcelines(function)
        return next(first + i for i, line in enumerate(lines) if marker in line)

    executor = vm.execute_microop_routed_control
    memory_start = line_of(executor, 'base = np.packbits')
    memory_end = line_of(executor, 'elif kind == "i32.eqz"')
    opcode_line = line_of(executor, 'kind = op.opcode')
    start = line_of(step_operands, 'validate_operands(state)')
    live = line_of(step_operands, 'depth = int(np.argmax')
    eligibility = line_of(step_operands, 'push, pop = action')
    eligible_end = line_of(step_operands, 'and all(type(array)')
    post = line_of(step_operands, 'validate_operands(result)')
    copy_line = line_of(_unit_pop, 'cells = state.cells.copy()')
    gate_line = line_of(_unit_pop, 'return OperandState(state.height @ _DOWN')
    counts = Counter()
    phases = Counter()
    current_opcode = None
    clock = perf_counter_ns

    def trace(frame, event, _arg):
        nonlocal current_opcode
        if event != "call":
            return None
        driver = frame.f_code is executor.__code__
        kernel = frame.f_code is _unit_pop.__code__
        if not (driver or kernel or frame.f_code is step_operands.__code__):
            return None
        if not (driver or kernel) and frame.f_locals["action"][2] != 1.:
            return None
        previous = frame.f_lineno
        stamp = clock()

        def handle(frame, event, _arg):
            nonlocal previous, stamp, current_opcode
            now = clock()
            elapsed = now - stamp
            if driver:
                if previous == opcode_line:
                    current_opcode = frame.f_locals["kind"]
                if memory_start <= previous < memory_end:
                    phases["memory_opcode_body"] += elapsed
            elif kernel:
                if copy_line <= previous < gate_line:
                    phases["array_copy_and_selected_row"] += elapsed
                elif previous == gate_line:
                    phases["fixed_height_matrix"] += elapsed
            else:
                if previous in (start, post):
                    phases["owned_state_checks"] += elapsed
                elif start < previous < live:
                    phases["static_token_coefficient_and_default_setup"] += elapsed
                elif live <= previous < eligibility:
                    phases["depth_and_resource_guards"] += elapsed
                elif eligibility <= previous <= eligible_end:
                    phases["unit_eligibility"] += elapsed
                if event == "return":
                    counts[current_opcode] += 1
            previous = frame.f_lineno
            stamp = clock()
            return handle

        return handle

    with patch("visuaml.compiler_lab.neural_microstep_executor_v58._NativeMicrostep",
               side_effect=AssertionError("Unexpected packed fallback")):
        run()
        begin = clock()
        run()
        ordinary_ns = clock() - begin
        sys.settrace(trace)
        begin = clock()
        try:
            steps = run()
        finally:
            sys.settrace(None)
        diagnostic_ns = clock() - begin
        profiler = cProfile.Profile()
        profiler.runcall(run, True)
    profile_rows = pstats.Stats(profiler).stats
    exact_functions = {
        "driver": ("neural_microop_executor_v57.py", "execute_microop_routed_control"),
        "checked_operand_transitions": ("neural_operand_storage_v55.py", "step_operands"),
        "operand_state_validation": ("neural_operand_storage_v55.py", "validate_operands"),
        "operand_top": ("neural_operand_storage_v55.py", "top_word"),
        "fixed_alu": ("neural_alu_v56.py", "execute_alu"),
        "frame_gates": ("neural_frame_circuit_v53.py", "step_frames"),
    }
    total_self = sum(row[2] for row in profile_rows.values())
    hotspots = {label: {"calls": calls, "self_ns": int(exclusive * 1e9),
                        "cumulative_ns": int(cumulative * 1e9)}
                for label, (file, name) in exact_functions.items()
                for (path, _, function), (_, calls, exclusive, cumulative, _) in profile_rows.items()
                if path.endswith(file) and function == name}
    movable = sum(phases[k] for k in ("owned_state_checks", "static_token_coefficient_and_default_setup",
                                    "unit_eligibility"))
    return {"source_steps": steps, "checked_pops_by_opcode": dict(counts), "ordinary_ns": ordinary_ns,
            "instrumented_ns": diagnostic_ns, "instrumented_phases_ns": dict(phases),
            "instrumented_movable_share": movable / diagnostic_ns,
            "instrumented_amdahl_ceiling": diagnostic_ns / (diagnostic_ns - movable),
            "prepared_cprofile": {"total_self_ns": int(total_self * 1e9), "hotspots": hotspots}}


def main():
    results = {}
    for name in ("sparse", "crowded", "bfs", "sudoku"):
        case = workload(name)
        if isinstance(case, dict):
            program = decode_control(terrain.GUEST.read_bytes(), scaled=True)
            initial = terrain.initial_page(case["initial"])
            terrain.write(initial, 320, [4 * d for d in case["inputs"][0]["moves"]])
            argument, fuel = terrain.BASE, 5000
        else:
            program, initial, argument, _, _, fuel, _ = case
            if name == "bfs":
                fuel = 4000
        results[name] = profile(program, argument, initial, fuel)
    print(json.dumps({"environment": environment(), "cases": results}, indent=2))


if __name__ == "__main__":
    main()
