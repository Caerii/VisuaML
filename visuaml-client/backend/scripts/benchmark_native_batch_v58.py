"""Compare v58 specialization and Wasmtime with in-engine batch loops."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import wasmtime

from scripts.benchmark_neural_vm_v58 import environment, git_identity, summarize_ns, workloads
from visuaml.compiler_lab.native_specialization_v58 import NativeSpecialization
from visuaml.compiler_lab.neural_microstep_v58 import DATA_OPS
from visuaml.compiler_lab.wasm_bitstream_v39 import MAGIC, _read_u32, _u32, parse_section_envelope
from visuaml.compiler_lab.wasm_control_flow_v51 import encode_control

_SIZES = (1, 32, 1024, 16384)
_WRAPPER = """(module
    (func $run (param i32) (result i32) local.get 0)
    (memory (export "memory") 1 1)
    (func (export "batch") (param $count i32) (local $offset i32)
      (block $done
        (loop $next
          (br_if $done (i32.eqz (local.get $count)))
          (i32.store (local.get $offset)
                     (call $run (i32.load (local.get $offset))))
          (local.set $offset (i32.add (local.get $offset) (i32.const 4)))
          (local.set $count (i32.sub (local.get $count) (i32.const 1)))
          (br $next)))))"""


def _wrapped_source(source: bytes) -> bytes:
    """Keep the original function body byte-for-byte and append a batch loop."""
    original = parse_section_envelope(source)[-1].payload
    _, first_body = _read_u32(original, 0)
    result = bytearray(MAGIC)
    for section in parse_section_envelope(bytes(wasmtime.wat2wasm(_WRAPPER))):
        payload = section.payload
        if section.section_id == 10:
            count, first = _read_u32(payload, 0)
            length, start = _read_u32(payload, first)
            payload = _u32(count) + original[first_body:] + payload[start + length:]
        result += bytes([section.section_id]) + _u32(len(payload)) + payload
    return bytes(result)


def _timed(action):
    start = time.perf_counter_ns()
    value = action()
    return value, time.perf_counter_ns() - start


def _sample(actions, inputs, expected, repeats):
    samples = {name: [] for name in actions}
    for trial in range(repeats + 1):
        order = tuple(actions)
        if trial % 2:
            order = order[::-1]
        for name in order:
            buffer, invoke = actions[name]
            buffer[:] = inputs
            _, elapsed = _timed(invoke)
            np.testing.assert_array_equal(buffer, expected)
            if trial:
                samples[name].append(elapsed)
    return {name: summarize_ns(times) for name, times in samples.items()}


def _sample_copied(actions, inputs, expected, repeats):
    samples = {name: [] for name in actions}
    for trial in range(repeats + 1):
        for name in (tuple(actions) if trial % 2 == 0 else tuple(reversed(actions))):
            output, elapsed = _timed(lambda: actions[name](inputs))
            np.testing.assert_array_equal(output, expected)
            if trial:
                samples[name].append(elapsed)
    return {name: summarize_ns(times) for name, times in samples.items()}


def run(root: Path, *, repeats: int, include_llvm: bool = False) -> dict:
    if include_llvm:
        from visuaml.compiler_lab.llvm_specialization_v58 import LlvmSpecialization
    source = git_identity(root)
    engine = wasmtime.Engine()
    rows = []
    with tempfile.TemporaryDirectory(prefix="visuaml-batch-bench-") as directory:
        cache = Path(directory) / "native"
        serialized = Path(directory) / "wasmtime.bin"
        for work in workloads(scaling=True):
            if any(op.opcode not in DATA_OPS for op in work.program.operations[:-1]):
                continue
            raw = encode_control(work.program)
            native, native_cold = _timed(lambda: NativeSpecialization(raw, cache_dir=cache))
            warm, native_reload = _timed(lambda: NativeSpecialization(raw, cache_dir=cache))
            if native.cache_hit or not warm.cache_hit or native.binary_sha256 != warm.binary_sha256:
                raise AssertionError("Unexpected native cache outcome")
            jit_metadata = {}
            if include_llvm:
                jit, llvm_compile = _timed(lambda: LlvmSpecialization(raw))
                jit_metadata = {
                    "llvm_compile_ns": llvm_compile, "llvm_ir_sha256": jit.ir_sha256,
                    "llvm_version": jit.llvm_version,
                }
            wrapped, wrapper_ns = _timed(lambda: _wrapped_source(raw))
            module, wasm_compile = _timed(lambda: wasmtime.Module(engine, wrapped))
            cold_store = wasmtime.Store(engine)
            _, wasm_instantiate = _timed(lambda: wasmtime.Instance(cold_store, module, []))
            _, wasm_save = _timed(lambda: serialized.write_bytes(module.serialize()))
            store = wasmtime.Store(engine)
            exports, wasm_reload = _timed(
                lambda: wasmtime.Instance(
                    store, wasmtime.Module.deserialize_file(engine, str(serialized)), []
                ).exports(store)
            )
            memory = np.ctypeslib.as_array(
                exports["memory"].data_ptr(store), shape=(65536,)
            ).view("<i4")
            original_store = wasmtime.Store(engine)
            original = wasmtime.Instance(
                original_store, wasmtime.Module(engine, raw), []
            ).exports(original_store)["run"]
            edge_inputs = np.array(
                [-(1 << 31), -1, 0, 1, (1 << 31) - 1, *work.arguments],
                dtype=np.int32,
            )
            random = np.random.default_rng(58).integers(
                -(1 << 31), 1 << 31, size=max(_SIZES) - edge_inputs.size, dtype=np.int32,
            )
            inputs = np.concatenate((edge_inputs, random))
            expected = np.fromiter(
                (original(original_store, int(value)) for value in inputs),
                dtype=np.int32,
            )
            batches = {}
            copied_batches = {}

            def native_copy(values):
                buffer = values.copy()
                warm.run_batch_inplace(buffer)
                return buffer

            def wasm_copy(values):
                buffer = memory[:values.size]
                buffer[:] = values
                exports["batch"](store, values.size)
                return buffer.copy()

            copied_actions = {"native": native_copy, "wasmtime": wasm_copy}
            if include_llvm:
                def llvm_copy(values):
                    result = values.copy()
                    jit.run_batch_inplace(result)
                    return result

                copied_actions["llvm"] = llvm_copy
                for value in edge_inputs:
                    if jit.run(int(value)) != original(original_store, int(value)):
                        raise AssertionError("LLVM scalar output disagrees with original WASM")

            for size in _SIZES:
                buffer = np.empty(size, dtype=np.int32)
                actions = {
                    "native": (buffer, lambda: warm.run_batch_inplace(buffer)),
                    "wasmtime": (memory[:size], lambda: exports["batch"](store, size)),
                }
                if include_llvm:
                    llvm_buffer = np.empty(size, dtype=np.int32)
                    actions["llvm"] = (llvm_buffer, lambda: jit.run_batch_inplace(llvm_buffer))
                batches[str(size)] = _sample(actions, inputs[:size], expected[:size], repeats)
                copied_batches[str(size)] = _sample_copied(
                    copied_actions,
                    inputs[:size], expected[:size], repeats,
                )
            rows.append({
                **jit_metadata,
                "name": work.name, "wasm_sha256": warm.source_sha256,
                "native_binary_sha256": warm.binary_sha256, "compiler": warm.compiler,
                "native_cold_ns": native_cold, "native_reload_ns": native_reload,
                "wasmtime_wrapper_ns": wrapper_ns, "wasmtime_compile_ns": wasm_compile,
                "wasmtime_instantiate_ns": wasm_instantiate,
                "wasmtime_serialize_ns": wasm_save, "wasmtime_reload_ns": wasm_reload,
                "batches": batches, "copied_batches": copied_batches,
            })
    summary = {}
    for scope in ("batches", "copied_batches"):
        summary[scope] = {}
        for size in _SIZES:
            per_engine = {
                engine_name: [
                    row[scope][str(size)][engine_name]["median_ns"] / size for row in rows
                ] for engine_name in (("native", "wasmtime", "llvm") if include_llvm else ("native", "wasmtime"))
            }
            summary[scope][str(size)] = {
                "median_ns_per_value": {name: statistics.median(values)
                                        for name, values in per_engine.items()},
                "paired_geomean_native_speedup": math.exp(statistics.mean(
                    math.log(wasm / native) for wasm, native in
                    zip(per_engine["wasmtime"], per_engine["native"], strict=True)
                )),
                "native_faster_programs": sum(native < wasm for wasm, native in
                                             zip(per_engine["wasmtime"], per_engine["native"], strict=True)),
            }
            if include_llvm:
                summary[scope][str(size)]["paired_geomean_llvm_speedup"] = math.exp(
                    statistics.mean(math.log(wasm / jit) for wasm, jit in
                                    zip(per_engine["wasmtime"], per_engine["llvm"], strict=True))
                )
    return {
        "schema": "visuaml.native-batch-benchmark.v58",
        "source": source, "environment": environment(), "repeats": repeats,
        "summary": summary, "programs": len(rows), "rows": rows,
        "scope": "Resident in-place and copied-input/owned-output trials, one engine call per batch. "
                 "Each result was checked against the original scalar Wasmtime module.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--llvm", action="store_true", help="Include the optional LLVM JIT")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    if args.repeats < 1 or output.exists() or output.is_relative_to(root):
        parser.error("Use positive repeats and a fresh output path outside the checkout")
    record = run(root, repeats=args.repeats, include_llvm=args.llvm)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(record, stream, sort_keys=True)
        stream.write("\n")
    print(json.dumps(record["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
