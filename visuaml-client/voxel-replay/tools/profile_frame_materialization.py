"""Line-attributed frame materialization costs, separate from paired timings."""
import argparse
import inspect
import json
import sys
from collections import Counter
from time import perf_counter_ns
from unittest.mock import patch

from visuaml.compiler_lab import neural_microop_executor_v57 as vm
from visuaml.compiler_lab.wasm_control_flow_v51 import ControlFuelExhausted

from compiled_frame_event_pairs import SETTINGS
from operand_pairs import run, workload


def line_of(func, text, *, after=0):
    lines, start = inspect.getsourcelines(func)
    return start + next(i for i, line in enumerate(lines) if start + i > after and text in line)


def profile(case, retain="off"):
    def execute():
        if isinstance(case, dict):
            run(case, [], None, **SETTINGS, use_compiled_frame_event=True, compiled_frame_retain=retain)
        else:
            program, page, argument, _, _, steps, _ = case
            try:
                vm.execute_microop_routed_control(program, argument, array_memory=page.copy(),
                                                   capture_trace=False, fuel=min(4000, steps),
                                                   **SETTINGS, use_compiled_frame_event=True, compiled_frame_retain=retain)
            except ControlFuelExhausted:
                assert steps > 4000

    costs = Counter()
    by_action = Counter()
    checked_events = Counter()
    frame = vm.step_frames
    gates = vm._step_frame_gates
    driver = vm.execute_microop_routed_control
    state_admission = line_of(frame, "validate_state(state)")
    token_start = line_of(frame, "if not isinstance(tokens")
    gate_entry = line_of(frame, "return _step_frame_gates")
    retain_start = line_of(gates, "if action_token[0]")
    retain_copy = line_of(gates, "return FrameState(state.depth.copy")
    change_start = line_of(gates, "d = int(np.argmax")
    build_tokens = line_of(driver, "event_tokens = FrameTokens(")
    check_frame = line_of(driver, "frame_state, restored = step_frames", after=build_tokens)
    decode_depth = line_of(driver, "depth_after = int(np.argmax")
    materialize_metadata = line_of(driver, "frame_meta = frame_meta[:depth_after]")
    decode_fields_end = line_of(driver, "fused_frame.publish_checked")

    def tracer(call, event, _arg):
        if event != "call" or call.f_code not in (frame.__code__, gates.__code__, driver.__code__):
            return None
        is_frame = call.f_code is frame.__code__
        is_gates = call.f_code is gates.__code__
        if is_frame and call.f_back.f_code is driver.__code__:
            opcode = call.f_back.f_locals["kind"]
            action_id = int(call.f_locals["tokens"].action.argmax())
            event_name = ("retain", "push", "pop", "branch", "return")[action_id]
            checked_events[f"{opcode}:{event_name}"] += 1
        action_token = (call.f_locals["tokens"].action if is_frame else
                        call.f_locals["action_token"] if is_gates else None)
        action = "retain" if action_token is not None and action_token[0] else "change"
        previous, stamp = call.f_lineno, perf_counter_ns()

        def trace(f, _event, _arg):
            nonlocal previous, stamp
            stop = perf_counter_ns()
            key = None
            if is_frame:
                if previous == state_admission:
                    key = "state_admission"
                elif token_start <= previous < gate_entry:
                    key = "token_and_weight_admission"
            elif is_gates:
                if previous == retain_copy:
                    key = "retain_copies_and_result"
                elif retain_start <= previous < change_start:
                    key = "retain_eligibility"
                elif previous >= change_start:
                    key = "change_gate_and_validation"
            elif build_tokens <= previous < check_frame:
                key = "driver_frame_token_encoding"
            elif previous == decode_depth:
                key = "driver_depth_decode"
            elif materialize_metadata <= previous < decode_fields_end:
                key = "driver_metadata_update"
            if key:
                costs[key] += stop - stamp
                if is_frame or is_gates:
                    by_action[action] += stop - stamp
            previous, stamp = f.f_lineno, perf_counter_ns()
            return trace

        return trace

    with patch("visuaml.compiler_lab.neural_microstep_executor_v58._NativeMicrostep",
               side_effect=AssertionError("Unexpected packed fallback")):
        execute()
        sys.settrace(tracer)
        began = perf_counter_ns()
        try:
            execute()
        finally:
            sys.settrace(None)
        return {"instrumented_ns": perf_counter_ns() - began, "categories_ns": dict(costs), "transition_ns_by_action": dict(by_action), "checked_opcode_events": dict(checked_events)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("sparse", "crowded", "bfs", "sudoku"))
    parser.add_argument("--retain", default="off", choices=("off", "copy", "resident"))
    args = parser.parse_args()
    print(json.dumps({"case": args.case, "bfs_prefix_steps": 4000 if args.case == "bfs" else None,
                      **profile(workload(args.case), args.retain)}, indent=2))


if __name__ == "__main__":
    main()
