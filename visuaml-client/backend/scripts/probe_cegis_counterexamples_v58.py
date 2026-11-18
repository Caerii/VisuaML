"""Stress blinded recovery with compositions that agree on the initial queries."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from scripts.benchmark_neural_vm_v58 import git_identity
from visuaml.compiler_lab.bit_neural_vm_v49 import construct_bit_weights, run_bit_vm
from visuaml.compiler_lab.blinded_behavioral_synthesis_v50 import (
    INITIAL,
    BehavioralSynthesisError,
    synthesize_from_oracle,
    verify_against_typed_source,
)
from visuaml.compiler_lab.neural_symbolic_roundtrip_v58 import RoundtripWorkload, verify_roundtrip
from visuaml.compiler_lab.wasm_state_vm_v40 import I32Program, VMOperation, encode_program

_CASES = (
    ("lt", 8), ("lt", 11), ("lt", 15),
    ("gt", -8), ("gt", -11), ("gt", -15),
    ("lt", 24),
)


def _source(direction: str, threshold: int) -> I32Program:
    # Express x < k as !(x > k-1), and x > k as !(x < k+1).
    complementary = "i32.gt_s" if direction == "lt" else "i32.lt_s"
    shifted = threshold - 1 if direction == "lt" else threshold + 1
    operations = (
        VMOperation("local.get", 0),
        VMOperation("i32.const", shifted),
        VMOperation(complementary),
        VMOperation("local.set", 1),
        VMOperation("local.get", 1),
        VMOperation("i32.eqz"),
        VMOperation("end"),
    )
    return I32Program(1, operations)


def _case(direction, threshold):
    source = _source(direction, threshold)
    raw = encode_program(source)
    fixed = construct_bit_weights()
    observations = []
    receipts = []

    def oracle(argument):
        result = run_bit_vm(source, argument, fixed=fixed)["result"]
        observations.append({"input": argument, "output": result})
        return result

    def verify(candidate):
        result = verify_against_typed_source(source, candidate)
        receipts.append({"candidate": asdict(candidate), "result": result})
        return result

    synthesis = None
    rejection = None
    try:
        synthesis = synthesize_from_oracle(oracle, verify)
    except BehavioralSynthesisError as error:
        if str(error) not in (
            "No surviving candidate in admitted grammar", "CEGIS refinement fuel exhausted",
        ):
            raise
        rejection = str(error)
    if observations[:len(INITIAL)] != [{"input": value, "output": 1} for value in INITIAL]:
        raise AssertionError("The fixture must be indistinguishable from constant one initially")
    first = receipts[0]
    if first["candidate"] != {"kind": "poly", "a": 0, "b": 0, "c": 1, "threshold": 0}:
        raise AssertionError("The first hypothesis must be the misleading constant")
    if first["result"]["solver_outcome"] != "sat":
        raise AssertionError("The first hypothesis must have an independently found counterexample")
    if not any(entry["input"] == first["result"]["counterexample_i32"] and entry["output"] == 0
               for entry in observations[len(INITIAL):]):
        raise AssertionError("The constructed oracle did not observe the first counterexample")
    if synthesis and abs(threshold) > 16:
        raise AssertionError("Out-of-grammar control was accepted")

    name = f"complement_{direction}_{threshold}"
    replay = None
    if synthesis:
        replay = verify_roundtrip(RoundtripWorkload(name, source, "counterexample_stress"))
        if synthesis["accepted_wasm_sha256"] != replay["recovered_wasm_sha256"]:
            raise AssertionError("Independent recovery runs selected different source bytes")

    return {
        "name": name, "status": "recovered" if synthesis else "rejected",
        "threshold": threshold, "direction": direction, "out_of_grammar": abs(threshold) > 16,
        "source_wasm_hex": raw.hex(), "source_wasm_sha256": hashlib.sha256(raw).hexdigest(),
        "fixed_weights_sha256": fixed.digest(), "observations": observations,
        "receipts": receipts, "rejection": rejection, "roundtrip": replay,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(root):
        parser.error("Use a fresh output path outside the checkout")
    source = git_identity(root)
    cases = [_case(direction, threshold) for direction, threshold in _CASES]
    recovered = [row for row in cases if row["status"] == "recovered"]
    if not recovered:
        raise AssertionError("Experiment requires a successful counterexample-driven recovery")
    record = {
        "schema": "visuaml.cegis-counterexample-stress.v58", "source": source, "cases": cases,
        "summary": {
            "attempts": len(cases), "recovered": len(recovered), "rejected": len(cases) - len(recovered),
            "first_hypotheses_refuted": len(cases),
            "finite_roundtrip_inputs": sum(row["roundtrip"]["finite_replay_inputs"] for row in recovered),
        },
        "scope": "Threshold grammar only. The solver checks typed source versus symbolic candidate. "
                 "Constructed execution is checked on finite inputs.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(record["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
