"""Compare archived v57 source bytes, results and states with independent engines."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

import wasmtime

from visuaml.compiler_lab.wasm_control_flow_v51 import decode_control, execute_control

TRACE_FIELDS = (
    "step", "pc", "opcode", "next_pc", "stack_i32", "locals_i32",
    "control_labels", "halted",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def trace_sha256(trace: list) -> str:
    return sha256(json.dumps(
        trace, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8"))


def verify(path: Path) -> dict:
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    archive = json.loads(raw)
    if archive["schema"] != "visuaml.offline-source-bound-microop-evidence.v57":
        raise ValueError("Unexpected v57 archive schema")
    commit, tree = subprocess.check_output(
        ("git", "rev-parse", "HEAD", "HEAD^{tree}"), text=True,
    ).splitlines()
    if (commit, tree) != (archive["git"]["commit"], archive["git"]["tree"]):
        raise ValueError("Check out the exact producing source before replay")
    if subprocess.check_output(("git", "status", "--porcelain"), text=True):
        raise ValueError("Replay requires a clean source checkout")

    engine = wasmtime.Engine()
    seen = set()
    witnesses = transitions = 0
    for case in archive["cases"]:
        source = bytes.fromhex(case["source_wasm_hex"])
        source_digest = sha256(source)
        if source_digest != case["source_wasm_sha256"] or source_digest in seen:
            raise ValueError(f"Invalid or duplicate source digest for {case['name']}")
        seen.add(source_digest)
        program = decode_control(source)
        if program.digest() != case["source_program_sha256"]:
            raise ValueError(f"Typed program digest changed for {case['name']}")
        wasmtime.Module.validate(engine, source)
        module = wasmtime.Module(engine, source)
        store = wasmtime.Store(engine)
        run = wasmtime.Instance(store, module, []).exports(store)["run"]
        for witness in case["witnesses"]:
            argument = witness["input_i32"]
            reference = execute_control(
                program, argument, fuel=archive["verification"]["fuel_bound"],
            )
            label = (case["name"], argument)
            if run(store, argument) != reference.value or reference.value != witness["result_i32"]:
                raise ValueError(f"Stored result disagrees with v51 or Wasmtime for {label}")
            trace = witness["trace"]
            if (trace_sha256(trace) != witness["trace_sha256"]
                    or len(trace) != witness["steps"]
                    or len(trace) != reference.transitions):
                raise ValueError(f"Stored trace digest or length changed for {label}")
            for actual, expected in zip(trace, reference.trace, strict=True):
                if any(actual[key] != expected[key] for key in TRACE_FIELDS):
                    raise ValueError(f"Stored state differs from v51 for {label} at step {expected['step']}")
            witnesses += 1
            transitions += len(trace)

    if (len(seen), witnesses, transitions) != (
        archive["program_count"], archive["witness_count"],
        archive["instruction_state_comparisons"],
    ):
        raise ValueError("Stored inventories disagree with the replay")
    return {
        "archive_sha256": sha256(raw),
        "source_commit": commit,
        "source_tree": tree,
        "programs": len(seen),
        "witnesses": witnesses,
        "v51_instruction_states_replayed": transitions,
        "original_wasmtime_outputs_replayed": witnesses,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    print(json.dumps(verify(parser.parse_args().archive), sort_keys=True))
