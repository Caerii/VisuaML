"""Replay independently written copy loops and trap diagnostics against Wasmtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import asdict
from itertools import product
from pathlib import Path

import wasmtime

from scripts.benchmark_neural_vm_v58 import git_identity
from scripts.verify_memory_recovery_v58 import replay_trace, word_values
from scripts.verify_memory_region_v58 import ADDRESSES, COUNTS, PATTERNS, Case
from scripts.verify_packed_control_v58 import prepare
from visuaml.compiler_lab.memory_region_contract_v58 import RegionObservation, first_difference, observe
from visuaml.compiler_lab.neural_microstep_executor_v58 import MemoryWeights
from visuaml.compiler_lab.wasm_control_flow_v51 import LinearMemoryTrap, decode_control, encode_control
from visuaml.compiler_lab.wasm_copy_region_v58 import PreparedCopyRegion, copy_program, recognize_copy_region

INDEPENDENT_WAT = """
(module
  (func (export "run") (param i32) (result i32)
    (local i32 i32 i32 i32)
    local.get 0 i32.const 0 i32.gt_s
    if
      i32.const 24 i32.load align=1 local.set 3
      i32.const 24 i32.load offset=4 local.set 1
      nop
      local.get 1 local.get 3 i32.gt_s
      if
        local.get 0 local.set 4
        block
          loop
            local.get 4 i32.eqz br_if 1
            local.get 4 i32.const 1 i32.sub local.set 4
            local.get 4 i32.const 4 i32.mul local.get 1 i32.add
            local.get 4 i32.const 4 i32.mul local.get 3 i32.add i32.load align=1
            i32.store align=1
            nop br 0
          end
        end
      else
        block
          loop
            local.get 4 local.get 0 i32.lt_s i32.eqz br_if 1
            local.get 4 i32.const 4 i32.mul local.get 1 i32.add
            local.get 4 i32.const 4 i32.mul local.get 3 i32.add i32.load align=1
            i32.store align=1
            local.get 4 i32.const 1 i32.add local.set 4
            br 0
          end
        end
      end
    end
    nop local.get 0
  )
  (memory (export "memory") 1 1)
)
"""


def independent_program():
    return decode_control(bytes(wasmtime.wat2wasm(INDEPENDENT_WAT)))


def fixture_page(case):
    initial = case.memory()
    struct.pack_into("<II", initial, 24, case.src, case.dst)
    return initial


def near_misses():
    return {
        "forward_only": INDEPENDENT_WAT.replace("local.get 1 local.get 3 i32.gt_s", "i32.const 0"),
        "trap_after_store": INDEPENDENT_WAT.replace(
            "i32.store align=1\n", "i32.store align=1 i32.const 65536 i32.load drop\n",
        ),
        "reversed_direction": INDEPENDENT_WAT.replace(
            "local.get 1 local.get 3 i32.gt_s", "local.get 3 local.get 1 i32.gt_s",
        ),
    }


def wasm_oracle(program):
    store = wasmtime.Store()
    wasm = wasmtime.Instance(store, wasmtime.Module(store.engine, encode_control(program)), []).exports(store)

    def run(count, initial):
        wasm["memory"].write(store, initial, 0)
        try:
            result = wasm["run"](store, count)
        except wasmtime.Trap as error:
            assert error.trap_code == wasmtime.TrapCode.MEMORY_OUT_OF_BOUNDS
            return RegionObservation("memory_oob", None, bytes(wasm["memory"].read(store)))
        return RegionObservation("return", result, bytes(wasm["memory"].read(store)))

    return run


def trap_record(session, argument, initial):
    page = bytearray(initial)
    try:
        session.run_memory(argument, page)
    except LinearMemoryTrap as error:
        replayed, _ = replay_trace(initial, error.trace)
        assert replayed == page
        return {"site": error.trap_site,
                "accesses": [row["memory_access"] for row in error.trace if row["memory_access"] is not None]}
    raise AssertionError("Expected an out-of-bounds trap")


def collect(root):
    identity = git_identity(root)
    program = independent_program()
    raw = encode_control(program)
    assert raw != encode_control(copy_program())
    lowered = PreparedCopyRegion(program)
    plan = recognize_copy_region(program)
    assert plan == lowered.plan and (plan.source_descriptor, plan.destination_descriptor) == (24, 28)
    assert plan != recognize_copy_region(copy_program())
    packed = prepare(program)
    wasm = wasm_oracle(program)
    cases = tuple(Case(*case) for case in product(COUNTS, ADDRESSES, ADDRESSES, range(len(PATTERNS))))
    fused = traps = fuel_checks = 0
    for case in cases:
        initial = fixture_page(case)
        expected = observe(packed.run_memory, case.count, initial)
        assert first_difference(wasm(case.count, initial), expected) is None, case
        actual = observe(lowered.run_memory, case.count, initial)
        success = expected.exit == "return"
        assert first_difference(expected, actual, source_steps=success) is None, case
        if success:
            step_count = expected.source_steps
            assert lowered.run_memory(case.count, bytearray(initial), fuel=step_count, capture_trace=False).fused
            before = observe(packed.run_memory, case.count, initial, fuel=step_count - 1, capture_trace=True)
            replay = observe(lowered.run_memory, case.count, initial, fuel=step_count - 1, capture_trace=True)
            assert before.exit == replay.exit == "fuel"
            assert before.trap_site["step"] == step_count - 1
            assert first_difference(before, replay, access_order=True) is None
            fused += 1
            fuel_checks += 2
        else:
            traced = observe(packed.run_memory, case.count, initial, capture_trace=True)
            replay = observe(lowered.run_memory, case.count, initial, capture_trace=True)
            assert traced.trap_site["address"] is not None
            assert first_difference(traced, replay, access_order=True) is None, case
            traps += 1

    overlap = Case(3, 8, 12)
    initial = fixture_page(overlap)
    nominal = observe(lowered.run_memory, overlap.count, initial)
    assert word_values(nominal.page[8:], 4) == (1, 1, 2, 3)
    rejected = {}
    for name, text in near_misses().items():
        variant = decode_control(bytes(wasmtime.wat2wasm(text)))
        runner = PreparedCopyRegion(variant)
        assert runner.plan is None, name
        case = Case(2, 12, 8) if name == "trap_after_store" else overlap
        initial_variant = fixture_page(case)
        result = observe(runner.run_memory, case.count, initial_variant)
        assert first_difference(wasm_oracle(variant)(case.count, initial_variant), result) is None
        normal = observe(lowered.run_memory, case.count, initial_variant)
        assert first_difference(normal, result) is not None, name
        evidence = {"case": asdict(case), "wasm_hex": encode_control(variant).hex(),
                    "difference_from_move": first_difference(normal, result)}
        if name == "trap_after_store":
            evidence["trap"] = trap_record(runner, case.count, initial_variant)
        else:
            evidence["result_words"] = word_values(result.page[8:], 4)
        rejected[name] = evidence

    late = Case(2, 65532, 8)
    fault = trap_record(lowered, late.count, fixture_page(late))
    assert fault["site"]["address"] == 65536
    assert fault["accesses"][-1] == {
        "kind": "i32.store", "address": 8, "before_hex": "01000000", "after_hex": "6d000000",
    }
    modified = PreparedCopyRegion(program, memory_fixed=MemoryWeights(store_gain=0))
    page = bytearray(initial)
    assert not modified.run_memory(overlap.count, page, capture_trace=False).fused
    assert page == initial
    return {
        "source": identity, "wasm_hex": raw.hex(), "wasm_sha256": hashlib.sha256(raw).hexdigest(),
        "plan": asdict(plan), "domain": {"counts": COUNTS, "addresses": ADDRESSES, "patterns": PATTERNS},
        "cases": len(cases), "fused": fused, "memory_traps": traps, "fuel_checks": fuel_checks,
        "near_misses": rejected, "late_trap": fault, "gain_override_replayed": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    if args.output.exists() or args.output.resolve().is_relative_to(root):
        parser.error("Use a fresh output path outside the checkout")
    evidence = collect(root)
    with args.output.open("x") as stream:
        json.dump(evidence, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
