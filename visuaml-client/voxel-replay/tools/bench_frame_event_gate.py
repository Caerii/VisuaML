"""Measure warm gate invocation separately from library build and replay."""
import json
from itertools import product
from statistics import median
from time import perf_counter_ns
from unittest.mock import patch

import numpy as np

from scripts.benchmark_neural_vm_v58 import environment
from visuaml.compiler_lab.compiled_frame_event_v58 import CompiledFrameEvents
from visuaml.compiler_lab.constructed_alu_v58 import compiled_alu
from visuaml.compiler_lab.neural_frame_events_v54 import (
    OPCODES, decode_opcode_token, fixed_event_weights, select_frame_action,
)


def main():
    begin = perf_counter_ns()
    library = compiled_alu()
    compile_ns = perf_counter_ns() - begin
    tokens = tuple(decode_opcode_token(opcode) for opcode in OPCODES)
    weights = fixed_event_weights()
    begin = perf_counter_ns()
    prepared = CompiledFrameEvents(tokens, weights)
    binding_ns = perf_counter_ns() - begin
    inputs = list(product(range(len(tokens)), product((False, True), repeat=4)))

    def reference(pc, flags):
        sequential, has_else, active, taken = (np.array([float(flag)]) for flag in flags)
        return int(select_frame_action(
            tokens[pc], conditional_sequential=sequential, has_else=has_else, frame_active=active,
            branch_taken=taken, weights=weights,
        ).argmax())

    def kernel(pc, flags):
        return prepared.select(pc, *flags)

    calls = {"numpy": reference, "kernel": kernel}
    expected = [reference(*row) for row in inputs]
    assert expected == [kernel(*row) for row in inputs]
    repeats = 40
    samples = {name: [] for name in calls}
    for pair in range(9):
        order = tuple(calls) if pair % 2 == 0 else tuple(reversed(calls))
        for name in order:
            invoke = calls[name]
            begin = perf_counter_ns()
            result = sum(invoke(*row) for _ in range(repeats) for row in inputs)
            elapsed = perf_counter_ns() - begin
            assert result == repeats * sum(expected)
            if pair:
                samples[name].append(elapsed)
    array_calls = array_ns = 0
    make_array = np.array

    def tracked_array(*args, **kwargs):
        nonlocal array_calls, array_ns
        start = perf_counter_ns()
        result = make_array(*args, **kwargs)
        array_ns += perf_counter_ns() - start
        array_calls += 1
        return result

    with patch.object(np, "array", tracked_array):
        result = sum(reference(*row) for _ in range(repeats) for row in inputs)
    assert result == repeats * sum(expected)
    assert array_calls == repeats * len(inputs) * 6
    print(json.dumps({
        "compile_and_load_ns": compile_ns, "binding_ns": binding_ns,
        "kernel_source_sha256": library.source_sha256, "kernel_binary_sha256": library.binary_sha256,
        "compiler": library.compiler, "flags": library.flags, "environment": environment(),
        "calls_per_sample": repeats * len(inputs), "samples_ns": samples,
        "separate_numpy_array_pass": {"calls": array_calls, "construction_ns": array_ns},
        "median_ns_per_call": {name: median(values) / (repeats * len(inputs))
                               for name, values in samples.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
