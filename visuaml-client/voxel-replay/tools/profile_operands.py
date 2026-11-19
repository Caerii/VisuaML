"""Diagnostic line profile for operand lowering and WASM memory dispatch."""
import cProfile
import inspect
import json
import pstats
import sys
from collections import Counter
from time import perf_counter_ns
from unittest.mock import patch

import numpy as np

from scripts.stateful_scale_workloads_v58 import graph_case
from visuaml.compiler_lab import neural_microop_executor_v57 as vm
from visuaml.compiler_lab import neural_operand_adapter_v55 as adapter
from visuaml.compiler_lab.neural_operand_storage_v55 import step_operands

from operand_pairs import reference_transition
from visuaml.compiler_lab.wasm_control_flow_v51 import ControlFuelExhausted, decode_control

from scale_terrain import ARCHIVE
from terrain_verify import BASE, GUEST, initial_page, scale_session, write


def marker(function, text):
    lines, first = inspect.getsourcelines(function)
    return next(first + i for i, line in enumerate(lines) if text in line)


def diagnostic(program, argument, initial, fuel):
    def run():
        try:
            vm.execute_microop_routed_control(program, argument, array_memory=initial.copy(),
                                                fuel=fuel, capture_trace=False)
        except ControlFuelExhausted:
            pass

    run()
    reference = adapter.step_operands
    executor = vm.execute_microop_routed_control
    clock = perf_counter_ns
    opcodes = Counter()
    transitions = Counter()
    costs = Counter()
    memory_start = marker(executor, 'base = np.packbits')
    memory_stop = marker(executor, 'elif kind == "i32.eqz"')
    operand_start = marker(reference, 'push_slot = state.height')
    stack_start = marker(reference, 'choices_height = np.stack')
    gate_start = marker(reference, 'gates = action')
    gate_stop = marker(reference, 'validate_operands(result)')
    execution_start = marker(executor, 'for tick in range(fuel)')
    instruction = marker(executor, 'kind = op.opcode')
    current_opcode = None

    def trace(frame, event, _arg):
        nonlocal current_opcode
        is_executor = frame.f_code is executor.__code__
        if event != "call" or not (is_executor or frame.f_code is reference.__code__):
            return None
        previous_line = frame.f_lineno
        previous_time = clock()

        def line_trace(frame, event, _arg):
            nonlocal previous_line, previous_time, current_opcode
            now = clock()
            line = frame.f_lineno
            if is_executor:
                if memory_start <= previous_line < memory_stop:
                    costs["memory_dispatch"] += now - previous_time
                # Record the source opcode after its assignment, including
                # instructions that perform no operand transition.
                if previous_line == instruction:
                    current_opcode = frame.f_locals["kind"]
                    opcodes[current_opcode] += 1
                if previous_line == execution_start and not costs["through_loop_entry"]:
                    costs["through_loop_entry"] = now - run_start
            else:
                if operand_start <= previous_line < stack_start:
                    costs["candidate_arithmetic_and_temporaries"] += now - previous_time
                elif gate_start <= previous_line < gate_stop:
                    costs["fixed_gain_contraction"] += now - previous_time
                elif stack_start <= previous_line < gate_start:
                    costs["candidate_stack_allocation_copy"] += now - previous_time
                else:
                    costs["operand_admission_and_postchecks"] += now - previous_time
                if event == "return":
                    action = frame.f_locals["action"].argmax()
                    transitions[f"{current_opcode}:{('retain', 'push', 'pop', 'truncate')[action]}"] += 1
            previous_line = line
            previous_time = clock()
            return line_trace

        return line_trace

    sys.settrace(trace)
    run_start = clock()
    try:
        run()
    finally:
        sys.settrace(None)
    total_ns = clock() - run_start
    profile = cProfile.Profile()
    profile.runcall(run)
    rows = pstats.Stats(profile).stats
    dispatch = {}
    adapter_names = ("append", "pop", "__delitem__", "__getitem__", "__bool__", "__len__")
    for key, (_, calls, exclusive, inclusive, _) in rows.items():
        path, _, name = key
        if ((path.endswith("neural_operand_adapter_v55.py") and name in adapter_names)
                or (path.endswith("neural_microop_executor_v57.py")
                    and name in ("execute_microop_routed_control", "_pop"))):
            dispatch[name] = {"calls": calls, "self_ms": exclusive * 1000, "cumulative_ms": inclusive * 1000}
    physical_work = {}
    stack = np.stack
    for name, transition in (("dense", reference), ("unit_push_pop", step_operands)):
        count = 0

        def count_stack(*args, **kwargs):
            nonlocal count
            if sys._getframe(1).f_code is transition.__code__:
                count += 1
            return stack(*args, **kwargs)

        with patch.object(adapter, "step_operands", transition), patch.object(np, "stack", count_stack):
            run()
        physical_work[name] = {"operand_candidate_stack_allocations": count}
    return {"instrumented_ns": total_ns, "instrumented_categories_ns": dict(costs),
            "cprofile_dispatch": dispatch, "physical_work": physical_work, "completed_source_opcodes": dict(opcodes), "operand_transitions": dict(transitions)}


def main():
    source = decode_control(GUEST.read_bytes(), scaled=True)
    session = scale_session(8, True, 4)
    page = initial_page(session["initial"])
    write(page, 320, [4 * d for d in session["inputs"][0]["moves"]])
    argument, graph, *_ = graph_case(0)
    graph_source = decode_control((ARCHIVE / "bfs.wasm").read_bytes(), scaled=True)
    with patch.object(adapter, "step_operands", reference_transition()), patch(
        "visuaml.compiler_lab.neural_microstep_executor_v58._NativeMicrostep",
        side_effect=AssertionError("Unexpected packed fallback"),
    ):
        report = {"terrain": diagnostic(source, BASE, page, 5000),
                  "bfs_prefix": diagnostic(graph_source, argument, graph, 2000)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
