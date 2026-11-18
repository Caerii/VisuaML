"""Offline adversarial fault-detection benchmark for constructed WASM gates.

Tests whether corruption *actually reaches* execution, changes its trace or
fails closed, against an unmodified independent baseline. A failed detection
is reported, never discarded or silently scored as success. No Actions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

from visuaml.compiler_lab.neural_alu_v56 import construct_alu_weights
from visuaml.compiler_lab.neural_control_routing_v52 import (
    construct_router_weights,
)
from visuaml.compiler_lab.neural_frame_circuit_v53 import fixed_frame_weights
from visuaml.compiler_lab.neural_frame_events_v54 import fixed_event_weights
from visuaml.compiler_lab.neural_local_bank_v57 import (
    construct_local_bank_weights,
)
from visuaml.compiler_lab.neural_microop_executor_v57 import (
    execute_microop_routed_control,
)
from visuaml.compiler_lab.neural_microstep_v58 import (
    construct_microstep_weights,
)
from visuaml.compiler_lab.neural_microstep_executor_v58 import (
    execute_microstep_control,
)
from visuaml.compiler_lab.neural_operand_storage_v55 import (
    fixed_operand_weights,
)
from visuaml.compiler_lab.typed_microop_ir_v57 import (
    EFFECTS, OPCODES, compile_microop_tokens, construct_microop_weights,
)
from visuaml.compiler_lab.wasm_research_corpus_v57 import countdown, generated
from visuaml.compiler_lab.wasm_control_flow_v51 import encode_control

SCHEMA = "visuaml.neural-vm-fault-campaign.v58"


@dataclass(frozen=True)
class Attack:
    name: str
    engine: str
    source: str
    opcode_activated: str
    mutation: str


ATTACKS = (
    Attack("pc_alternate_route", "v57", "countdown", "br_if",
           "zero alternate path coefficient"),
    Attack("frame_push", "v57", "countdown", "block",
           "zero frame push coefficient"),
    Attack("frame_conditional_guard", "v57", "countdown", "br_if",
           "zero br_if frame-event condition coefficient"),
    Attack("operand_push", "v57", "countdown", "local.get",
           "zero tensor operand push coefficient"),
    Attack("alu_add", "v57", "countdown", "i32.add",
           "zero fixed ALU add output coefficient"),
    Attack("microop_add_arity", "v57", "countdown", "i32.add",
           "misdecode add effect as retain with valid one-hot"),
    Attack("local_write", "v57", "countdown", "local.set",
           "zero local 1 write gate"),
    Attack("source_token_tamper", "v57", "countdown", "block",
           "alter source opcode token, leave source hash untouched"),
    Attack("microstep_first_pop", "v58", "generated", "i32.mul",
           "zero first tensor-pop coefficient"),
    Attack("microstep_output", "v58", "generated", "i32.mul",
           "zero ALU output materialization coefficient"),
)


def stable_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def trace_hash(result) -> str:
    return hashlib.sha256(stable_bytes(result.trace)).hexdigest()


def execute(engine: str, program, x: int, **kwargs):
    if engine == "v57":
        return execute_microop_routed_control(
            program, x, fuel=1500, **kwargs,
        )
    if engine == "v58":
        return execute_microstep_control(
            program, x, fuel=500, **kwargs,
        )
    raise ValueError(f"Unknown engine: {engine}")


def mutate(attack: Attack, program) -> dict:
    """Return intentionally corrupted construction; not executable host patch."""
    if attack.name == "pc_alternate_route":
        fixed = construct_router_weights()
        new = fixed.path_weights.copy()
        new[1] = 0.
        return {"fixed": replace(fixed, path_weights=new)}
    if attack.name == "frame_push":
        fixed = fixed_frame_weights()
        new = fixed.action_weights.copy()
        new[1] = 0.
        return {"frame_fixed": replace(fixed, action_weights=new)}
    if attack.name == "frame_conditional_guard":
        fixed = fixed_event_weights()
        new = fixed.conditional_gains.copy()
        new[2] = 0.
        return {"event_fixed": replace(fixed, conditional_gains=new)}
    if attack.name == "operand_push":
        fixed = fixed_operand_weights()
        new = fixed.action_gain.copy()
        new[1] = 0.
        return {"operand_fixed": replace(fixed, action_gain=new)}
    if attack.name == "alu_add":
        fixed = construct_alu_weights()
        new = fixed.output_gains.copy()
        new[0] = 0.
        return {"alu_fixed": replace(fixed, output_gains=new)}
    if attack.name == "microop_add_arity":
        fixed = construct_microop_weights()
        new = fixed.effects.copy()
        new[OPCODES.index("i32.add"), :] = 0.
        new[OPCODES.index("i32.add"), EFFECTS.index("retain")] = 1.
        return {"microop_fixed": replace(fixed, effects=new)}
    if attack.name == "local_write":
        fixed = construct_local_bank_weights()
        new = fixed.write_gain.copy()
        new[1] = 0.
        return {"local_fixed": replace(fixed, write_gain=new)}
    if attack.name == "source_token_tamper":
        tokens = compile_microop_tokens(program)
        new = tokens.opcode_rows.copy()
        new[0] = new[1]
        return {"microop_tokens": replace(tokens, opcode_rows=new)}
    if attack.name in ("microstep_first_pop", "microstep_output"):
        fixed = construct_microstep_weights()
        pop = fixed.pop_gain.copy()
        produce = fixed.produce_gain.copy()
        if attack.name == "microstep_first_pop":
            pop[0] = 0.
        else:
            produce[2] = 0.
        return {"microstep_fixed": replace(
            fixed, pop_gain=pop, produce_gain=produce,
        )}
    raise ValueError("Unknown mutation")


def run_attack(attack: Attack) -> dict:
    program = countdown() if attack.source == "countdown" else generated(2)
    argument = 4 if attack.source == "countdown" else -17
    baseline = execute(attack.engine, program, argument)
    if not any(
        row["opcode"] == attack.opcode_activated for row in baseline.trace
    ):
        raise AssertionError(
            f"Attack {attack.name} has no exercised target opcode"
        )
    changed = None
    error = None
    try:
        changed = execute(
            attack.engine, program, argument, **mutate(attack, program)
        )
    except (ValueError, RuntimeError, TypeError, IndexError) as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    detected = bool(
        error or (
            changed is not None and (
                changed.result_i32 != baseline.result_i32 or
                trace_hash(changed) != trace_hash(baseline)
            )
        )
    )
    return {
        "attack": attack.name,
        "mutation": attack.mutation,
        "engine": attack.engine,
        "source": attack.source,
        "input_i32": argument,
        "opcode_activated": attack.opcode_activated,
        "source_program_sha256": program.digest(),
        "source_wasm_sha256": hashlib.sha256(
            encode_control(program)
        ).hexdigest(),
        "baseline_result_i32": baseline.result_i32,
        "baseline_trace_sha256": trace_hash(baseline),
        "mutated_result_i32": (
            None if changed is None else changed.result_i32
        ),
        "mutated_trace_sha256": (
            None if changed is None else trace_hash(changed)
        ),
        "exception": error,
        "detected": detected,
    }


def campaign() -> dict:
    cases = [run_attack(attack) for attack in ATTACKS]
    return {
        "schema": SCHEMA,
        "injected_faults": len(cases),
        "detected_faults": sum(case["detected"] for case in cases),
        "undetected_faults": [
            case["attack"] for case in cases if not case["detected"]
        ],
        "measurement": "activated opcode and independent execution divergence",
        "cases": cases,
        "limitations": (
            "Hand-picked single-coefficient faults only. Detection ratio "
            "is not a probability of reliability, formal coverage, or a "
            "measure of unknown-weight interpretability."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-all-detected", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    def git(*arguments: str) -> str:
        return subprocess.check_output(
            ("git", "-C", str(root), *arguments),
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    if git("status", "--porcelain", "--untracked-files=normal"):
        parser.error("Fault evidence requires a clean Git worktree")
    report = campaign()
    report["source"] = {
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
    }
    report["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite existing artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(stable_bytes(report) + b"\n")
    print(json.dumps({
        "output": str(output),
        "injected": report["injected_faults"],
        "detected": report["detected_faults"],
        "undetected": report["undetected_faults"],
    }, sort_keys=True))
    if args.require_all_detected and report["undetected_faults"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
