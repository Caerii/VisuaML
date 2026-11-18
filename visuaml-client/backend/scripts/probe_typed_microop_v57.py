"""Offline, source-bound v57 full-trace evidence generator.

Use a clean local Git checkout. No GitHub Actions or network activity.
Run from repository root with PYTHONPATH=visuaml-client/backend.
Requires independently installed wasmtime and numpy.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import wasmtime

from visuaml.compiler_lab.neural_alu_executor_v56 import (
    execute_alu_routed_control,
)
from visuaml.compiler_lab.neural_alu_v56 import construct_alu_weights
from visuaml.compiler_lab.neural_control_routing_v52 import (
    construct_router_weights,
)
from visuaml.compiler_lab.neural_frame_circuit_v53 import (
    fixed_frame_weights,
)
from visuaml.compiler_lab.neural_frame_events_v54 import fixed_event_weights
from visuaml.compiler_lab.neural_local_bank_v57 import (
    construct_local_bank_weights,
)
from visuaml.compiler_lab.neural_microop_executor_v57 import (
    execute_microop_routed_control,
)
from visuaml.compiler_lab.neural_operand_storage_v55 import (
    fixed_operand_weights,
)
from visuaml.compiler_lab.typed_microop_ir_v57 import (
    compile_microop_tokens,
    construct_microop_weights,
    validate_microop_tokens,
)
from visuaml.compiler_lab.wasm_control_flow_v51 import (
    encode_control,
    execute_control,
)
from visuaml.compiler_lab.wasm_research_corpus_v57 import corpus

SCHEMA = "visuaml.offline-source-bound-microop-evidence.v57"
FIELDS = (
    "step", "pc", "opcode", "next_pc", "stack_i32",
    "locals_i32", "control_labels", "halted",
)


def stable_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(stable_bytes(value)).hexdigest()


def tensors_digest(*tensors: np.ndarray) -> str:
    h = hashlib.sha256(b"visuaml.v57.fixed-auxiliary-weights")
    for array in tensors:
        h.update(str(array.shape).encode())
        h.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
    return h.hexdigest()


def git_identity(root: Path) -> dict[str, str]:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ("git", "-C", str(root), *args), text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    status = git("status", "--porcelain", "--untracked-files=normal")
    if status:
        raise RuntimeError(
            "Evidence generation requires a clean tracked and untracked Git tree"
        )
    return {
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "branch": git("branch", "--show-current"),
    }


def produce(root: Path) -> dict:
    identity = git_identity(root)
    engine = wasmtime.Engine()
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "wasmtime": importlib.metadata.version("wasmtime"),
    }
    frame = fixed_frame_weights()
    event = fixed_event_weights()
    operand = fixed_operand_weights()
    local = construct_local_bank_weights()
    fixed_weights = {
        "alu": construct_alu_weights().digest(),
        "microop": construct_microop_weights().digest(),
        "pc": construct_router_weights().digest(),
        "frame_event": tensors_digest(
            event.default, event.conditional_gains,
        ),
        "frame_state": tensors_digest(frame.action_weights),
        "operand": tensors_digest(operand.action_gain),
        "local_bank": tensors_digest(local.read_gain, local.write_gain),
    }
    cases = []
    n_witnesses = 0
    n_steps = 0
    for name, (program, arguments) in corpus().items():
        source = encode_control(program)
        wasmtime.Module.validate(engine, source)
        module = wasmtime.Module(engine, source)
        store = wasmtime.Store(engine)
        run = wasmtime.Instance(store, module, []).exports(store)["run"]
        tokens = compile_microop_tokens(program)
        validate_microop_tokens(tokens, program)
        case = {
            "name": name,
            "source_wasm_hex": source.hex(),
            "source_wasm_sha256": hashlib.sha256(source).hexdigest(),
            "source_program_sha256": program.digest(),
            "program_microop_tokens_sha256": tokens.digest(),
            "witnesses": [],
        }
        for argument in arguments:
            current = execute_microop_routed_control(
                program, argument, fuel=1500,
                microop_tokens=tokens,
            )
            prior = execute_alu_routed_control(program, argument, fuel=1500)
            reference = execute_control(program, argument, fuel=1500)
            original = run(store, argument)
            if not (current.result_i32 == prior.result_i32
                    == reference.value == original):
                raise AssertionError((name, argument, "result disagreement"))
            if not (current.steps == prior.steps == reference.transitions):
                raise AssertionError((name, argument, "transition count"))
            for step, (new, old, oracle) in enumerate(zip(
                current.trace, prior.trace, reference.trace, strict=True,
            )):
                for key in FIELDS:
                    if new[key] != old[key] or new[key] != oracle[key]:
                        raise AssertionError(
                            (name, argument, step, key, "state mismatch")
                        )
            if current.source_wasm_sha256 != case["source_wasm_sha256"]:
                raise AssertionError((name, argument, "source mismatch"))
            transcript = list(current.trace)
            case["witnesses"].append({
                "input_i32": argument,
                "result_i32": current.result_i32,
                "steps": current.steps,
                "trace_sha256": digest(transcript),
                "trace": transcript,
            })
            n_witnesses += 1
            n_steps += current.steps
        cases.append(case)
    report = {
        "schema": SCHEMA,
        "git": identity,
        "versions": versions,
        "fixed_weight_digests": fixed_weights,
        "program_count": len(cases),
        "witness_count": n_witnesses,
        "instruction_state_comparisons": n_steps,
        "verification": {
            "original_source_wasmtime": True,
            "independent_v51_and_v56": True,
            "all_live_states_checked": True,
            "fuel_bound": 1500,
            "finite_input_witnesses_not_all_i32_proof": True,
        },
        "cases": cases,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    # Generate before creating output, so clean Git-tree checks are meaningful.
    report = produce(root)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to rewrite evidence: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(stable_bytes(report) + b"\n")
    print(json.dumps({
        "output": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "programs": report["program_count"],
        "witnesses": report["witness_count"],
        "state_comparisons": report["instruction_state_comparisons"],
        "source_commit": report["git"]["commit"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
