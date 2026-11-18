"""Separate warmed native-kernel timing from Python result assembly."""
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from scripts.benchmark_native_batch_loop_v58 import CONTROL
from scripts.benchmark_neural_vm_v58 import environment, git_identity, summarize_ns, workloads
from visuaml.compiler_lab.constructed_alu_v58 import compiled_alu
from visuaml.compiler_lab.neural_control_routing_v52 import TERMINAL
from visuaml.compiler_lab.neural_microstep_executor_v58 import PreparedMicrostepProgram

source = git_identity(Path.cwd())
kernel = compiled_alu()
assert kernel.prepared_simd_available()
corpus = sorted(
    (work for work in workloads(scaling=True)
     if not any(op.opcode in CONTROL for op in work.program.operations[:-1])),
    key=lambda work: (len(work.program.operations), work.name),
)


def measure_pair(scalar, simd, calls):
    modes = (("packed", scalar), ("simd", simd))
    samples = {name: [] for name, _ in modes}
    for _, action in modes:
        action()
    for i in range(21):
        order = modes if i % 2 == 0 else tuple(reversed(modes))
        for name, action in order:
            start = time.perf_counter_ns()
            for _ in range(calls):
                action()
            samples[name].append((time.perf_counter_ns() - start) // calls)
    return {name: summarize_ns(values) for name, values in samples.items()}


rows = []
for work in (corpus[0], corpus[len(corpus) // 2], corpus[-1]):
    program = work.program
    scalar_session = PreparedMicrostepProgram(program, use_native_loop=True, use_packed_scalar=True)
    simd_session = PreparedMicrostepProgram(program, use_native_loop=True, use_packed_scalar=True,
                                            use_packed_simd=True)
    native = simd_session._inputs.native_microstep
    for size in (8, 32, 128, 1024):
        arguments = tuple(work.arguments[i % len(work.arguments)] for i in range(size))
        inputs = np.asarray(arguments, dtype=np.int32)
        fuels = np.full(size, 400, dtype=np.int32)
        output = np.empty((size, 2), dtype=np.int32)
        statuses = np.empty(size, dtype=np.int32)
        const_args = (
            inputs.ctypes.data, fuels.ctypes.data, size, len(program.operations),
            TERMINAL, program.local_count + 1, 0, *native._pointers,
            *native._router._pointers, output.ctypes.data, statuses.ctypes.data, None, None, True,
        )

        def call_kernel(simd):
            kernel.prepared_batch(*const_args, simd)

        reference = scalar_session.run_many(arguments, capture_trace=False)
        assert simd_session.run_many(arguments, capture_trace=False) == reference
        expected = [(row.result_i32, row.steps) for row in reference]
        for simd in (False, True):
            call_kernel(simd)
            assert not statuses.any() and list(map(tuple, output.tolist())) == expected
        direct = measure_pair(lambda: call_kernel(False), lambda: call_kernel(True), max(1, 2048 // size))
        api = measure_pair(lambda: scalar_session.run_many(arguments, capture_trace=False),
                           lambda: simd_session.run_many(arguments, capture_trace=False),
                           max(1, 2048 // size))
        rows.append({
            "program": work.name, "size": size, "steps": len(program.operations),
            "kernel": direct, "api": api,
            "kernel_speedup": direct["packed"]["median_ns"] / direct["simd"]["median_ns"],
            "api_speedup": api["packed"]["median_ns"] / api["simd"]["median_ns"],
        })
with Path(sys.argv[1]).open("x") as stream:
    json.dump({
        "source": source, "environment": environment(), "kernel_sha256": kernel.binary_sha256,
        "rows": rows,
        "geomean_1024": {
            key: statistics.geometric_mean(row[f"{key}_speedup"] for row in rows if row["size"] == 1024)
            for key in ("kernel", "api")
        },
    }, stream, indent=2, sort_keys=True)
    stream.write("\n")
