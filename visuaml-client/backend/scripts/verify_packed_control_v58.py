"""Verify packed structured control against the host, gate, and WASM references."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from functools import partial
from pathlib import Path

import wasmtime

from scripts.benchmark_neural_vm_v58 import environment, git_identity, measure
from visuaml.compiler_lab.constructed_alu_v58 import compiled_alu
from visuaml.compiler_lab.neural_control_routing_v52 import construct_router_weights
from visuaml.compiler_lab.neural_frame_circuit_v53 import FrameCircuitError, fixed_frame_weights
from visuaml.compiler_lab.neural_frame_events_v54 import fixed_event_weights
from visuaml.compiler_lab.neural_microop_executor_v57 import execute_microop_routed_control
from visuaml.compiler_lab.neural_microstep_executor_v58 import (
    MicrostepBatchFailure, PreparedMicrostepProgram,
)
from visuaml.compiler_lab.wasm_control_flow_v51 import (
    MAX_NESTING, ControlFuelExhausted, ControlProgram, _syntax, encode_control, execute_control,
)
from visuaml.compiler_lab.wasm_research_corpus_v57 import corpus, ifelse, op

FIELDS = ("step", "pc", "opcode", "next_pc", "stack_i32", "locals_i32", "control_labels", "halted")
GATES = (*FIELDS, "neural_route", "neural_frame_event", "control_frames")


def prepare(program, **gains):
    return PreparedMicrostepProgram(
        program, use_native_loop=True, use_packed_scalar=True, use_structured_control=True, **gains,
    )


def witnesses():
    reviewed = corpus()
    return {
        "ifelse": (ifelse(), (-(1 << 31), -1, 0, 1, (1 << 31) - 1)),
        "countdown": reviewed["countdown"],
        "gcd12": reviewed["gcd12"],
        "nested_branches": (ControlProgram(1, (
            op("i32.const", 5), op("block"), op("block"), op("i32.const", 99),
            op("local.get", 0), op("br_if", 1), op("drop"), op("br", 0),
            op("end"), op("i32.const", 8), op("local.set", 1), op("end"),
            op("local.get", 1), op("i32.add"), op("end"),
        )), (-1, 0, 1)),
        "nested_return": (ControlProgram(0, (
            op("block"), op("local.get", 0), op("if"), op("i32.const", -42),
            op("return"), op("end"), op("end"), op("local.get", 0), op("end"),
        )), (-1, 0, 1)),
        "nested_loop_target": (ControlProgram(0, (
            op("i32.const", 101), op("loop"), op("block"), op("local.get", 0),
            op("i32.const", 1), op("i32.sub"), op("local.tee", 0), op("br_if", 1),
            op("end"), op("end"), op("end"),
        )), (1, 2, 7)),
        "if_label_target": (ControlProgram(1, (
            op("local.get", 0), op("if"), op("i32.const", 10), op("local.set", 1),
            op("block"), op("local.get", 0), op("i32.const", 1), op("i32.gt_s"), op("br_if", 1),
            op("end"), op("i32.const", 8), op("local.set", 1), op("else"),
            op("i32.const", 9), op("local.set", 1), op("end"), op("local.get", 1), op("end"),
        )), (-1, 0, 1, 2)),
        "maximum_nesting": (ControlProgram(0, (
            op("local.get", 0), *(op("block"),) * MAX_NESTING,
            *(op("end"),) * MAX_NESTING, op("end"),
        )), (-1, 0, 1)),
    }


def nonterminating():
    return ControlProgram(0, (
        op("loop"), op("br", 0), op("end"), op("i32.const", 0), op("end"),
    ))


def first_divergence(expected, actual, fields=GATES):
    """Return the earliest mismatch in an ordered transition record."""
    for tick, (left, right) in enumerate(zip(expected, actual)):
        changes = {key: [left[key], right[key]] for key in fields if left[key] != right[key]}
        if changes:
            return {"step": tick, "pc": [left["pc"], right["pc"]], "changes": changes}
    if len(expected) != len(actual):
        return {"step": min(len(expected), len(actual)), "lengths": [len(expected), len(actual)]}
    return None


def frame_reference(program, records):
    """Recover frame metadata from the v57 reference event and source, for trace comparison."""
    links = _syntax(program)
    frames = []
    for record in records:
        pc, action = record["pc"], record["neural_frame_event"]
        if action == 1:
            end, otherwise = links[pc]
            frames.append({
                "kind": record["opcode"], "start_pc": pc, "end_pc": end,
                "else_pc": otherwise, "stack_base": len(record["stack_i32"]),
            })
        elif action == 2:
            frames.pop()
        elif action == 3:
            index = len(frames) - 1 - program.operations[pc].immediate
            del frames[index + (frames[index]["kind"] == "loop"):]
        elif action == 4:
            frames.clear()
        assert [frame["kind"] for frame in frames] == record["control_labels"]
        yield {**record, "control_frames": frames.copy()}


def check_fuel(action):
    try:
        action()
    except ControlFuelExhausted:
        return
    raise AssertionError("Expected bounded fuel exhaustion")


def intervention_evidence():
    program = ifelse()
    baseline = prepare(program).run(0)
    router = construct_router_weights()
    router.guard_weights[2] = 0.
    changed = prepare(program, pc_fixed=router).run(0)
    ref = execute_microop_routed_control(program, 0, fixed=router)
    assert changed.result_i32 == ref.result_i32 == 11
    assert baseline.result_i32 == 22
    assert first_divergence(tuple(frame_reference(program, ref.trace)), changed.trace) is None
    difference = first_divergence(baseline.trace, changed.trace)
    assert difference["pc"] == [3, 3] and "next_pc" in difference["changes"]

    frame = fixed_frame_weights()
    frame.action_weights[1] = 0.
    outcomes = prepare(program, frame_fixed=frame).run_many((0, 1), fuel=30)
    assert all(isinstance(out, MicrostepBatchFailure) and isinstance(out.error, FrameCircuitError)
               for out in outcomes)
    event = fixed_event_weights()
    event.conditional_gains[0] = 0.
    try:
        prepare(program, event_fixed=event).run(-1)
    except RuntimeError as exc:
        assert "frame" in str(exc)
        event_error = type(exc).__name__
    else:
        raise AssertionError("Entering an if with the event gain ablated unexpectedly succeeded")
    return {"route": difference, "frame": type(outcomes[0].error).__name__, "event": event_error}


def collect(root, repeats):
    source = git_identity(root)
    kernel = compiled_alu()
    config = wasmtime.Config()
    config.consume_fuel = True
    engine = wasmtime.Engine(config)
    rows = []
    for name, (program, arguments) in witnesses().items():
        store = wasmtime.Store(engine)
        wasm_bytes = encode_control(program)
        module = wasmtime.Module(engine, wasm_bytes)
        wasm = wasmtime.Instance(store, module, []).exports(store)["run"]
        packed = prepare(program)
        cases = []
        for argument, actual in zip(arguments, packed.run_many(arguments), strict=True):
            store.set_fuel(10000)
            assert actual == packed.run(argument)
            host = execute_control(program, argument)
            gate = execute_microop_routed_control(program, argument)
            assert actual.result_i32 == host.value == gate.result_i32 == wasm(store, argument)
            first = first_divergence(host.trace, actual.trace, FIELDS)
            assert first is None, (name, argument, first)
            first = first_divergence(tuple(frame_reference(program, gate.trace)), actual.trace)
            assert first is None, (name, argument, first)
            assert packed.run(argument, capture_trace=False) == replace(actual, trace=())
            boundary = next((record["step"] for record in actual.trace if record["opcode"] == "if"),
                            actual.steps - 1)
            limits = sorted({fuel for fuel in (boundary, boundary + 1, actual.steps - 1)
                             if 0 < fuel < actual.steps})
            for fuel in limits:
                check_fuel(partial(packed.run, argument, fuel=fuel))
                check_fuel(partial(execute_control, program, argument, fuel=fuel))
                check_fuel(partial(execute_microop_routed_control, program, argument, fuel=fuel))
            assert packed.run(argument, fuel=actual.steps) == actual
            timings = {
                "packed_trace": measure(partial(packed.run, argument), warmups=1, repeats=repeats),
                "packed_untraced": measure(partial(packed.run, argument, capture_trace=False),
                                           warmups=1, repeats=repeats),
                "v57_trace": measure(partial(execute_microop_routed_control, program, argument),
                                     warmups=1, repeats=repeats),
            }
            cases.append({"argument": argument, "result_i32": actual.result_i32, "steps": actual.steps,
                          "trace": actual.trace, "fuel_traps": limits, "timings": timings})
        rows.append({"name": name, "source_wasm_sha256": hashlib.sha256(wasm_bytes).hexdigest(),
                     "wasm_hex": wasm_bytes.hex(), "cases": cases})

    endless = nonterminating()
    for action in (partial(prepare(endless).run, 0, fuel=50),
                   partial(execute_control, endless, 0, fuel=50),
                   partial(execute_microop_routed_control, endless, 0, fuel=50)):
        check_fuel(action)
    store = wasmtime.Store(engine)
    store.set_fuel(50)
    module = wasmtime.Module(engine, encode_control(endless))
    try:
        wasmtime.Instance(store, module, []).exports(store)["run"](store, 0)
    except wasmtime.Trap as error:
        # Some Wasmtime Python bindings cannot decode the native enum
        # member for this trap. Cross-check the message and remaining fuel.
        assert "fuel" in error.message and store.get_fuel() == 0
    else:
        raise AssertionError("Nonterminating Wasmtime fixture ran to completion")
    return {"source": source, "environment": environment(), "kernel_sha256": kernel.binary_sha256,
            "rows": rows, "nontermination_fuel": 50, "interventions": intervention_evidence()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    if args.output.exists() or args.output.resolve().is_relative_to(root) or args.repeats < 1:
        parser.error("Use a fresh output path outside the checkout and positive repeats")
    result = collect(root, args.repeats)
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")


if __name__ == "__main__":
    main()
