"""v52 immutable archival audit of actual gate-selected program counters.

Do not rewrite or reattribute the original 451,450-byte CI record after
subsequent code changes. Recompute all witnessed machine states and PC-gate
trace hashes against canonical original source WASM, not summary counters.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from visuaml.compiler_lab.neural_control_routing_v52 import (
    compile_route_tokens,
    construct_router_weights,
    execute_routed_control,
    validate_route_tokens,
)
from visuaml.compiler_lab.wasm_control_flow_v51 import (
    decode_control,
    encode_control,
    execute_control,
)

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "benchmarks/neural_pc_control_v52.json"
ORIGIN = ROOT / "benchmarks/neural_pc_control_v52.origin.json"
RAW_SHA = "136333f7a7e7c9bc5da98bd9b7635fd08a72f574c83fbe506362ba32d5103a51"


def _original():
    raw = RAW.read_bytes()
    record = json.loads(raw)
    origin = json.loads(ORIGIN.read_bytes())
    assert len(raw) == origin["raw_bytes"] == 451450
    assert hashlib.sha256(raw).hexdigest() == RAW_SHA == origin["raw_sha256"]
    assert record["source_commit"] == origin["source_commit"] == (
        "80c6d4dba7439461e8732c2979777984877546a5"
    )
    assert origin["workflow_url"] == (
        "https://github.com/Caerii/VisuaML/actions/runs/44333149043"
    )
    assert origin["source_file_sha256"] == record["source_file_sha256"]
    return record, origin


def test_v52_original_ci_identity_and_exact_claim_limits():
    record, origin = _original()
    assert record["schema"] == "visuaml.original-wasm-neural-pc-router.v52"
    assert origin["schema"] == "visuaml.neural-pc-routing-origin.v52"
    assert record["program_count"] == origin["program_count"] == 6
    assert record["witness_count"] == origin["input_witness_count"] == 61
    assert record["total_per_instruction_comparisons"] == (
        origin["per_instruction_state_comparisons"]
    ) == 3405
    assert record["taken_jump_routing_events"] == 418
    assert record["conditional_instruction_events"] == 421
    assert record["fixed_router_weights_sha256"] == origin[
        "fixed_router_weights_sha256"
    ] == construct_router_weights().digest()
    assert record["fixed_i32_arithmetic_weights_sha256"] == origin[
        "fixed_arithmetic_weights_sha256"
    ]
    assert record["actual_onehot_pc_controls_next_instruction"]
    assert record["neural_program_counter_selected_by_matrix_gates"]
    assert record["same_analytic_router_weights_for_all_programs"]
    assert record["wasmtime_original_source_validated"]
    assert record["separate_v51_host_interpreter_compared_every_step"]
    assert record["branch_targets_compiled_from_source_into_program_data"]
    assert record["runtime_lexical_control_frames_still_host_managed"]
    assert record["runtime_wasm_opcode_dispatch_still_host_managed"]
    assert not record["complete_neural_wasm_interpreter_constructed"]
    assert not record["standard_transformer_softmax_executed"]
    assert not record["arbitrary_pretrained_decompilation_implemented"]
    assert not record["all_input_semantics_formally_proved"]
    assert origin["runtime_pc_routed_by_constructed_weights"]
    assert origin["runtime_control_stack_host_managed"]
    assert not origin["all_WASM_supported"]
    assert not origin["all_inputs_formally_verified"]
    assert not origin["external_signed_attestation"]


def test_replay_all_original_source_wasm_routing_and_machine_trace_hashes():
    record, _ = _original()
    programs = {}
    for entry in record["reviewed_programs"]:
        source = bytes.fromhex(entry["canonical_original_wasm_hex"])
        assert hashlib.sha256(source).hexdigest() == entry["source_wasm_sha256"]
        program = decode_control(source)
        assert encode_control(program) == source
        assert program.digest() == entry["source_program_sha256"]
        tokens = compile_route_tokens(program)
        validate_route_tokens(tokens, program)
        assert tokens.digest() == entry["program_route_tokens_sha256"]
        assert entry["shared_fixed_router_sha256"] == record[
            "fixed_router_weights_sha256"
        ]
        for i, row in enumerate(entry["reviewed_lexical_route_rows"]):
            assert row["pc"] == i
            assert row["default_pc"] == int(tokens.sequential[i].argmax())
            assert row["alternate_pc"] == int(tokens.alternate[i].argmax())
            assert row["guard_kind"] == tokens.guard_kind[i].tolist()
        programs[entry["name"]] = (program, tokens)
    assert len(programs) == 6
    total = 0
    for witness in record["witnesses"]:
        program, tokens = programs[witness["program"]]
        x = witness["argument_i32"]
        routed = execute_routed_control(
            program, x, fuel=1500, tokens=tokens,
        )
        host = execute_control(program, x, fuel=1500)
        assert witness["passed"]
        assert routed.result_i32 == host.value == witness[
            "original_wasmtime_result"
        ] == witness["v51_host_pc_result"] == witness[
            "v52_actual_neural_pc_result"
        ]
        assert routed.steps == host.transitions == witness[
            "executed_instruction_count"
        ]
        complete = [
            {
                "step": w["step"], "pc": w["pc"], "opcode": w["opcode"],
                "next_pc": w["next_pc"], "route": w["neural_route"],
                "stack_i32": w["stack_i32"], "locals_i32": w["locals_i32"],
                "control_labels": w["control_labels"], "halted": w["halted"],
            } for w in routed.trace
        ]
        encoded = json.dumps(complete, sort_keys=True,
                             separators=(",", ":")).encode()
        assert hashlib.sha256(encoded).hexdigest() == witness[
            "complete_neural_routing_trace_sha256"
        ]
        if "complete_machine_trace" in witness:
            assert complete == witness["complete_machine_trace"]
        total += routed.steps
    assert total == record["total_per_instruction_comparisons"]
