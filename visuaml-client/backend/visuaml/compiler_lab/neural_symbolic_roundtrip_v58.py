"""v58 cross-representation round-trip: neural behavior -> symbolic -> WASM.

This experiment tests an intentionally narrow *functional* round trip:
a constructed v49 neural bit-VM supplies only I/O observations to v50's
blinded CEGIS; an independent Z3 oracle compares a candidate with known
source WASM; the accepted candidate is reencoded and executed by the v58
fixed-weight microstep machine, v51, and Wasmtime.

The synthesizer has no access to source bytes, source instructions, weights,
token maps or traces. The verifier is separately allowed access to source.
This is NOT arbitrary learned-network decompilation, unique source recovery,
or a proof that all i32 inputs to the neural gate circuit equal the source.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

from .bit_neural_vm_v49 import construct_bit_weights, run_bit_vm
from .blinded_behavioral_synthesis_v50 import (
    BehavioralSynthesisError,
    Candidate,
    emit_canonical_wasm,
    synthesize_from_oracle,
    verify_against_typed_source,
)
from .neural_microstep_executor_v58 import execute_microstep_control
from .wasm_control_flow_v51 import (
    ControlOperation,
    ControlProgram,
    decode_control,
    encode_control,
    execute_control,
)
from .wasm_state_vm_v40 import (
    I32Program,
    VMOperation,
    encode_program,
    execute_program,
)

SCHEMA = "visuaml.cross-representation-verified-roundtrip.v58"
INPUTS = (
    -(1 << 31), -(1 << 31) + 1, -65536, -17, -1, 0,
    1, 2, 5, 17, 65535, (1 << 31) - 1,
)


def _op(name: str, immediate: int | None = None) -> VMOperation:
    return VMOperation(name, immediate)


@dataclass(frozen=True)
class RoundtripWorkload:
    name: str
    source: I32Program
    split: str


def source_workloads() -> tuple[RoundtripWorkload, ...]:
    """Five historical v50 families plus eight deterministically held out.

    The held-out cases use distinct source instruction graphs but remain
    inside v50's declared bounded grammar. They do NOT establish
    generalization to arbitrary learned networks or unrestricted WASM.
    """
    original = (
        RoundtripWorkload("quadratic", I32Program(0, (
            _op("local.get", 0), _op("local.get", 0), _op("i32.mul"),
            _op("local.get", 0), _op("i32.const", 3), _op("i32.mul"),
            _op("i32.sub"), _op("i32.const", 7), _op("i32.add"),
            _op("end"),
        )), "historical"),
        RoundtripWorkload("identity_via_zero", I32Program(0, (
            _op("local.get", 0), _op("i32.const", 0),
            _op("i32.add"), _op("end"),
        )), "historical"),
        RoundtripWorkload("zero_test", I32Program(0, (
            _op("local.get", 0), _op("i32.eqz"), _op("end"),
        )), "historical"),
        RoundtripWorkload("signed_negative", I32Program(0, (
            _op("local.get", 0), _op("i32.const", 0),
            _op("i32.lt_s"), _op("end"),
        )), "historical"),
        RoundtripWorkload("signed_positive", I32Program(0, (
            _op("local.get", 0), _op("i32.const", 0),
            _op("i32.gt_s"), _op("end"),
        )), "historical"),
    )
    heldout_candidates = (
        Candidate("poly", a=0, b=2, c=7),
        Candidate("poly", a=1, b=2, c=-3),
        Candidate("poly", a=-2, b=1, c=4),
        Candidate("poly", a=2, b=-1, c=0),
        Candidate("poly", a=0, b=-3, c=6),
        Candidate("lt_s", threshold=5),
        Candidate("gt_s", threshold=-4),
        Candidate("lt_s", threshold=-7),
    )
    extra = []
    for idx, hypothesis in enumerate(heldout_candidates):
        source, _ = emit_canonical_wasm(hypothesis)
        # A semantically neutral, source-distinct instruction suffix
        # requires the synthesizer to recover behavior, not original bytes.
        operations = source.operations[:-1] + (
            _op("i32.const", 0), _op("i32.add"), _op("end"),
        )
        extra.append(RoundtripWorkload(
            f"heldout_{idx:02d}", I32Program(0, operations), "heldout",
        ))
    items = original + tuple(extra)
    hashes = [hashlib.sha256(encode_program(item.source)).hexdigest()
              for item in items]
    if len(set(hashes)) != len(items):
        raise ValueError("Round-trip source programs are not byte-distinct")
    return items


def as_control(source: I32Program) -> ControlProgram:
    """Cross the historical v40/v51 typed WASM IR boundary by exact bytes."""
    control = ControlProgram(
        source.local_count,
        tuple(ControlOperation(op.opcode, op.immediate)
              for op in source.operations),
    )
    if encode_control(control) != encode_program(source):
        raise ValueError("Original v40/v51 canonical WASM bytes differ")
    if decode_control(encode_control(control)) != control:
        raise ValueError("WASM control parser failed source round trip")
    return control


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _stable_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()


def verify_roundtrip(
    workload: RoundtripWorkload, *, timeout_ms: int = 15000,
    audit: dict | None = None,
) -> dict:
    """Run the full blinded recovery cycle with separately scoped oracles."""
    import wasmtime

    source = workload.source
    raw_source = encode_program(source)
    control_source = as_control(source)
    fixed = construct_bit_weights()
    oracle_query_count = 0
    if audit is not None:
        audit.update({
            "neural_oracle_queries": 0,
            "independent_verifier_invocations": 0,
            "source_wasm_sha256": _sha256(raw_source),
            "constructed_neural_weights_sha256": fixed.digest(),
            "requested_solver_timeout_ms": timeout_ms,
        })

    def neural_oracle(argument: int) -> int:
        nonlocal oracle_query_count
        oracle_query_count += 1
        if audit is not None:
            audit["neural_oracle_queries"] = oracle_query_count
        # This is the ONLY callable the synthesizer can invoke for behavior.
        return run_bit_vm(source, argument, fixed=fixed)["result"]

    def independent_verifier(candidate: Candidate) -> dict:
        if audit is not None:
            audit["independent_verifier_invocations"] += 1
            audit["last_candidate_sha256"] = candidate.digest()
        # This closure is deliberately separate from the synthesizer.
        return verify_against_typed_source(
            source, candidate, timeout_ms=timeout_ms,
        )

    begin = time.perf_counter_ns()
    synthesis = synthesize_from_oracle(
        neural_oracle, independent_verifier,
    )
    recovery_ns = time.perf_counter_ns() - begin
    if audit is not None:
        audit["cegis_rounds"] = len(synthesis["rounds"])
        audit["verified_candidate_sha256"] = synthesis[
            "rounds"
        ][-1]["selected_hypothesis_sha256"]
    candidate = Candidate(**synthesis["accepted_candidate"])
    recovered, raw_recovered = emit_canonical_wasm(candidate)
    if raw_recovered.hex() != synthesis["accepted_wasm_hex"]:
        raise AssertionError("Candidate WASM bytes differ from CEGIS receipt")
    last = synthesis["rounds"][-1]["verification"]
    if (last["solver_outcome"] != "unsat"
            or not last["all_i32_typed_source_equivalent"]
            or last["all_i32_neural_circuit_proved"]):
        raise AssertionError("Incorrect scope or result of SMT certificate")
    control_recovered = as_control(recovered)
    engine = wasmtime.Engine()
    wasmtime.Module.validate(engine, raw_source)
    wasmtime.Module.validate(engine, raw_recovered)
    outputs = []
    total_recovered_steps = 0
    for arg in INPUTS:
        store = wasmtime.Store(engine)
        original_run = wasmtime.Instance(
            store, wasmtime.Module(engine, raw_source), []
        ).exports(store)["run"]
        recovered_run = wasmtime.Instance(
            store, wasmtime.Module(engine, raw_recovered), []
        ).exports(store)["run"]
        original = original_run(store, arg)
        recovered_native = recovered_run(store, arg)
        neural = neural_oracle(arg)
        candidate_vm = execute_microstep_control(
            control_recovered, arg, fuel=400,
        )
        recovered_reference = execute_control(
            control_recovered, arg, fuel=400,
        )
        source_reference = execute_control(
            control_source, arg, fuel=400,
        )
        if not (
            original == recovered_native == neural == candidate_vm.result_i32
            == recovered_reference.value == source_reference.value
            == execute_program(source, arg).value
        ):
            raise AssertionError(
                (workload.name, arg, "cross-representation disagreement")
            )
        for row, reference in zip(
            candidate_vm.trace, recovered_reference.trace, strict=True,
        ):
            for key in (
                "pc", "next_pc", "opcode", "stack_i32",
                "locals_i32", "control_labels", "halted",
            ):
                if row[key] != reference[key]:
                    raise AssertionError(
                        (workload.name, arg, key, "v58 trace disagreement")
                    )
        trace_digest = _sha256(_stable_bytes(candidate_vm.trace))
        outputs.append({
            "input_i32": arg,
            "result_i32": original,
            "source_vs_recovered_equal": True,
            "recovered_v58_trace_sha256": trace_digest,
            "recovered_v58_steps": candidate_vm.steps,
        })
        total_recovered_steps += candidate_vm.steps

    return {
        "schema": SCHEMA,
        "name": workload.name,
        "split": workload.split,
        "original_wasm_sha256": _sha256(raw_source),
        "original_wasm_hex": raw_source.hex(),
        "recovered_wasm_sha256": _sha256(raw_recovered),
        "recovered_wasm_hex": raw_recovered.hex(),
        "original_wasm_bytes": len(raw_source),
        "recovered_wasm_bytes": len(raw_recovered),
        "same_source_binary": raw_source == raw_recovered,
        "constructed_neural_weights_sha256": fixed.digest(),
        "oracle_queries_including_finite_replay": oracle_query_count,
        "cegis_observation_count": len(synthesis["observed_input_outputs"]),
        "cegis_refinements": len(synthesis["rounds"]) - 1,
        "cegis_recovery_and_smt_wall_ns": recovery_ns,
        "candidate_expression": synthesis["accepted_expression"],
        "candidate": synthesis["accepted_candidate"],
        "blinded_synthesis_claims": {
            "source_received": synthesis["source_program_received_by_synthesizer"],
            "weights_received": synthesis["neural_weights_received_by_synthesizer"],
            "manifest_received": synthesis[
                "construction_manifest_received_by_synthesizer"
            ],
        },
        "candidate_sha256": candidate.digest(),
        "smt_formula_sha256": last["smt_formula_sha256"],
        "z3_version": last["z3_version"],
        "all_i32_typed_source_equivalence": True,
        "all_i32_neural_weights_equivalence": False,
        "unique_original_source_recovered": False,
        "arbitrary_learned_neural_model_decompiled": False,
        "finite_replay_inputs": len(INPUTS),
        "finite_recovered_instruction_transitions": total_recovered_steps,
        "finite_results": outputs,
    }
