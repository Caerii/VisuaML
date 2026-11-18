"""v51.1 immutable Euclid real-WASM and bounded-fuel research record audit.

Historical exact bytes and source commit are checked separately from any
fresh Wasmtime run. Program results and selected full control traces replay
under the current bounded host-PC/v49-gated arithmetic contract.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from visuaml.compiler_lab.wasm_control_flow_v51 import (
    ControlFuelExhausted,
    decode_control,
    execute_control,
)

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "benchmarks/wasm_control_gcd_v51_1.json"
ORIGIN = ROOT / "benchmarks/wasm_control_gcd_v51_1.origin.json"


def _data():
    data = RAW.read_bytes()
    origin = json.loads(ORIGIN.read_bytes())
    report = json.loads(data)
    assert len(data) == origin["raw_bytes"] == 115260
    assert hashlib.sha256(data).hexdigest() == origin["raw_sha256"] == (
        "7173b9dd90360cfae513daadb620a3bff31e3e5d2048499218a7d0539b776ffd"
    )
    assert report["source_commit"] == origin["source_commit"] == (
        "6ea371af0b4983896b0ecae91e469b39ec9f9824"
    )
    assert origin["run_url"] == (
        "https://github.com/Caerii/VisuaML/actions/runs/44253535787"
    )
    assert origin["source_file_sha256"] == report["source_file_sha256"]
    return report, origin


def test_original_gcd_archive_and_no_false_nontermination_claim():
    report, origin = _data()
    assert report["schema"] == "visuaml.real-wasm-gcd-control-evidence.v51.1"
    assert origin["schema"] == "visuaml.real-wasm-gcd-origin.v51.1"
    assert report["sample_count"] == len(report["witnesses"]) == 27
    assert report["original_binary_wasmtime_validated"]
    assert report["independent_mathematical_reference_used"]
    assert report["source_wasm_real_runtime_executed"]
    assert report["fuel_exhausted_on_mathematically_terminating_large_input"]
    assert not report["fuel_exhaustion_means_nontermination"]
    assert not report["neural_routing_of_branch_targets"]
    assert not report["full_wasm_supported"]
    assert not report["general_gcd_all_i32_semantics_proved"]
    assert not origin["neural_branch_routing_proved"]
    assert not origin["all_input_equivalence_proved"]
    assert origin["fuel_limited_result_is_not_nontermination"]


def test_all_gcd_witnesses_and_original_host_control_traces_replay():
    report, origin = _data()
    raw = bytes.fromhex(report["gcd_12_source_wasm_hex"])
    assert hashlib.sha256(raw).hexdigest() == (
        report["gcd_12_source_wasm_sha256"]
    ) == origin["original_source_wasm_sha256"]
    program = decode_control(raw)
    assert program.digest() == report["program_sha256"]
    assert program.local_count == 1
    total = 0
    by_input = {}
    for row in report["witnesses"]:
        x = row["input_i32"]
        expected = math.gcd(x,12) if x >= 0 else 0
        computed = execute_control(program, x, fuel=1500)
        assert row["passed"]
        assert row["gcd_reference_i32"] == expected
        assert row["wasmtime_i32"] == expected
        assert row["host_control_bit_gates_i32"] == expected
        assert computed.value == expected
        assert row["instruction_transitions"] == computed.transitions
        assert row["branch_back_edges"] == sum(
            t["next_pc"] <= t["pc"]
            for t in computed.trace if t["opcode"] == "br"
        )
        total += computed.transitions
        by_input[x] = computed
    assert total == report["total_executed_control_transitions"]
    for row in report["sampled_complete_control_traces"]:
        assert list(by_input[row["input_i32"]].trace) == (
            row["complete_machine_trace"]
        )
    assert {x["input_i32"] for x in report["sampled_complete_control_traces"]} == (
        {0,1,12,17}
    )
    with __import__("pytest").raises(ControlFuelExhausted):
        execute_control(program, (1 << 31)-1, fuel=60)
