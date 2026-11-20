"""Run the Rust optimizer and execute its candidate on the same frozen graph."""
import argparse
import json
import struct
from unittest.mock import patch

import torch
import wasmtime

from bench_frozen_multistep import drive
from replay_frozen_memory_selection import OUT as EXECUTOR
from replay_frozen_recurrent import PACKED, sha, signed, wasmtime_run
from visuaml.compiler_lab.frozen_calls_v58 import initialize, load_call_image
from visuaml.compiler_lab.frozen_recurrent_v58 import snapshot_page

OUT = EXECUTOR.parent / "guest-rewrite"
INPUT_BASE = 512
RESULT_BASE = 256
PAGE_SIZE = 65536


def assemble(sequence):
    """Only encode the already selected instructions and a fixed test harness."""
    lines = []
    for tag, operand in sequence:
        if tag == 0:
            lines.append(f"i32.const {operand}")
        elif tag < 0:
            lines.append("i32.add")
        else:
            raise ValueError("Unknown instruction tag")
    body = "\n    ".join(lines)
    source = f"""(module
  (memory (export "memory") 1 1)
  (func (export "run") (param i32) (result i32)
    i32.const {RESULT_BASE}
    {body}
    i32.store
    i32.const {RESULT_BASE}
    i32.load))
"""
    return source, bytes(wasmtime.wat2wasm(source))


def check_constant_fold(original, emitted):
    """An independent, complete check for this constant-only rewrite rule."""
    if len(original) != 3 or [tag for tag, _ in original[:2]] != [0, 0] or original[2][0] >= 0:
        return False
    if len(emitted) != 1 or emitted[0][0] != 0:
        return False
    a, b = [operand for _, operand in original[:2]]
    expected = (a + b) % (1 << 32)
    return expected == emitted[0][1] % (1 << 32)


def invoke(graph, raw, argument, page, fuel=200):
    image = load_call_image(raw)
    state, calls = drive(graph, graph, image, initialize(image, argument, page, fuel), 64)
    status = int(state.vm.status)
    final = snapshot_page(state.vm)
    result = signed(state.vm.stack[0]) if status == 1 else None
    return final, {"return_i32": result, "status": status, "fuel_budget": fuel,
                   "unspent_fuel": int(state.vm.fuel), "source_steps": fuel - int(state.vm.fuel),
                   "stop_image_pc": int(state.vm.pc), "graph_invocations": calls,
                   "binary_sha256": image.binary_sha256, "image_sha256": image.image_sha256,
                   "input_page_sha256": sha(page), "output_page_sha256": sha(final),
                   "words_at_result": list(struct.unpack_from("<i", final, RESULT_BASE))}


def optimize(graph, raw, sequence):
    page = bytearray(PAGE_SIZE)
    struct.pack_into("<6i", page, INPUT_BASE, *(word for item in sequence for word in item))
    # Sentinels prove that the output length is written by the guest.
    struct.pack_into("<3i", page, INPUT_BASE + 24, -1, -1, -1)
    changed, evidence = invoke(graph, raw, INPUT_BASE, page)
    assert evidence["status"] == 1
    assert wasmtime_run(raw, INPUT_BASE, page) == (evidence["return_i32"], changed)
    count, tag, value = struct.unpack_from("<3i", changed, INPUT_BASE + 24)
    assert count in (0, 1) and count == evidence["return_i32"]
    assert changed[:INPUT_BASE + 24] == page[:INPUT_BASE + 24]
    assert changed[INPUT_BASE + 36:] == page[INPUT_BASE + 36:]
    if count == 0:
        assert (tag, value) == (-1, -1)
    return ([(tag, value)] if count else []), evidence


def run_program(graph, raw, expected, first_write):
    initial = bytearray(PAGE_SIZE)
    initial[RESULT_BASE:RESULT_BASE + 4] = bytes.fromhex("aabbccdd")
    page, record = invoke(graph, raw, 0, initial)
    assert record["return_i32"] == expected
    assert wasmtime_run(raw, 0, initial) == (expected, page)
    changed = initial.copy()
    struct.pack_into("<i", changed, RESULT_BASE, expected)
    assert page == changed
    cuts = []
    for fuel in range(record["source_steps"] + 1):
        after, cut = invoke(graph, raw, 0, initial, fuel)
        assert cut["source_steps"] == fuel
        assert cut["status"] == (1 if fuel == record["source_steps"] else 2)
        assert cut["unspent_fuel"] == 0
        assert after == (changed if fuel >= first_write else initial)
        cuts.append(cut)
    record["cutoffs"] = cuts
    return page, record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-binaries", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    path = EXECUTOR / "machine.pt"
    graph = torch.jit.load(str(path))
    optimizer = (OUT / "optimizer.wasm").read_bytes()
    original = [(0, 2147483647), (0, 1), (-1, 0)]
    deliberately_bad = [(0, 2147483647)]
    records = []
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        emitted, rewrite = optimize(graph, optimizer, original)
        assert check_constant_fold(original, emitted)
        assert not check_constant_fold(original, deliberately_bad)
        for name, sequence, expected in (
            ("original", original, -2147483648),
            ("rewritten", emitted, -2147483648),
            ("rejected-saturating", deliberately_bad, 2147483647),
        ):
            source, raw = assemble(sequence)
            if args.write_binaries:
                (OUT / f"{name}.wat").write_text(source)
                (OUT / f"{name}.wasm").write_bytes(raw)
            assert (OUT / f"{name}.wasm").read_bytes() == raw
            page, evidence = run_program(graph, raw, expected, first_write=2 + len(sequence))
            records.append((page, evidence))
        assert records[0][0] == records[1][0] != records[2][0]
        assert records[0][1]["source_steps"] - records[1][1]["source_steps"] == 2
        additional = []
        for left, right in ((-2147483648, -1), (-1, 1), (0, 0), (4, 7)):
            sequence = [(0, left), (0, right), (-1, 0)]
            result, evidence = optimize(graph, optimizer, sequence)
            assert check_constant_fold(sequence, result)
            additional.append({"input": sequence, "emitted": result, **evidence})
        unsupported = [(0, 4), (0, 7), (1, 0)]
        rejected, no_match = optimize(graph, optimizer, unsupported)
        assert rejected == []
    print(json.dumps({
        "graph_sha256": sha(path.read_bytes()),
        "optimizer_wasm_sha256": sha(optimizer), "record_abi_address": INPUT_BASE,
        "input_sequence": original, "emitted_sequence": emitted, "optimizer": rewrite,
        "original": records[0][1], "rewritten": records[1][1],
        "rejected_candidate": {"sequence": deliberately_bad, "checker_accepts": False,
                               "counterexample": {"argument": 0, "output_address": RESULT_BASE,
                                                  "initial_word_hex": "aabbccdd",
                                                  "expected_word_hex": "00000080",
                                                  "candidate_word_hex": "ffffff7f"},
                               "replay": records[2][1]},
        "additional_optimizer_cases": additional,
        "unsupported_pattern": {"input": unsupported, "emitted": rejected, **no_match},
    }, indent=2))


if __name__ == "__main__":
    main()
