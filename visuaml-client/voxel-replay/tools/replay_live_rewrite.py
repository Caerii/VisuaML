"""Check a guest's live-input rewrite claims and replay the emitted candidates."""
import argparse
from dataclasses import replace
import json
import struct
from unittest.mock import patch

import torch
import wasmtime
import z3

import terrain_verify as terrain
from replay_guest_rewrite import EXECUTOR, invoke
from replay_frozen_recurrent import PACKED, sha, signed, wasmtime_run
from visuaml.compiler_lab.frozen_calls_v58 import load_call_image
from visuaml.compiler_lab.wasm_control_flow_v51 import ControlOperation, decode_control, encode_control

OUT = EXECUTOR.parent / "live-rewrite"
LANES = ("x", "const", "add", "sub", "gt_s", "end")
CANDIDATES = {1: (("x", 0),), 2: (("const", 1),),
              3: (("x", 0), ("const", 1), ("sub", 0))}
BASE = 512
RESULT = 256
EMPTY = bytes(65536)


def encode_tokens(tokens):
    words = []
    for opcode, immediate in list(tokens) + [("end", 0)] * (5 - len(tokens)):
        words.extend([int(lane != opcode) for lane in LANES])
        words.append(immediate)
    assert len(words) == 35
    return words


def propose(graph, optimizer, tokens):
    words = encode_tokens(tokens)
    page = bytearray(EMPTY)
    struct.pack_into("<38i", page, BASE, *words, -1, -1, -1)
    final, evidence = invoke(graph, optimizer, BASE, page)
    assert evidence["status"] == 1
    assert wasmtime_run(optimizer, BASE, page) == (evidence["return_i32"], final)
    candidate, condition = struct.unpack_from("<2i", final, BASE + 36 * 4)
    assert candidate == evidence["return_i32"]
    assert final[:BASE + 35 * 4] == page[:BASE + 35 * 4]
    assert final[BASE + 38 * 4:] == page[BASE + 38 * 4:]
    if candidate == 0:
        return None, evidence
    return {"candidate_id": candidate, "condition_id": condition, "condition": "all i32 x",
            "input": tokens, "input_abi_words": words, "replacement": CANDIDATES[candidate]}, evidence


def expression(tokens, x):
    stack = []
    for opcode, value in tokens:
        if opcode == "x":
            stack.append(x)
        elif opcode == "const":
            stack.append(z3.BitVecVal(value, 32))
        else:
            right, left = stack.pop(), stack.pop()
            if opcode == "add":
                stack.append(left + right)
            elif opcode == "sub":
                stack.append(left - right)
            elif opcode == "gt_s":
                stack.append(z3.If(left > right, z3.BitVecVal(1, 32), z3.BitVecVal(0, 32)))
            else:
                raise ValueError(opcode)
    answer, = stack
    return answer


def prove(claim):
    assert claim["condition_id"] == 0
    x = z3.BitVec("x", 32)
    left = expression(claim["input"], x)
    right = expression(claim["replacement"], x)
    solver = z3.SolverFor("QF_BV")
    solver.add(left != right)
    status = solver.check()
    evidence = {"result": str(status), "query": solver.to_smt2()}
    if status == z3.sat:
        model = solver.model()
        evidence["counterexample"] = {key: signed(model.eval(term).as_long()) for key, term in
                                      (("x", x), ("original", left), ("candidate", right))}
    else:
        assert status == z3.unsat
    return evidence


def assemble(tokens):
    opcodes = {"x": "local.get 0", "add": "i32.add", "sub": "i32.sub", "gt_s": "i32.gt_s"}
    body = "\n    ".join(f"i32.const {value}" if op == "const" else opcodes[op]
                         for op, value in tokens)
    wat = f"""(module
  (memory (export "memory") 1 1)
  (func (export "run") (param i32) (result i32)
    i32.const {RESULT}
    {body}
    i32.store
    i32.const {RESULT}
    i32.load))
"""
    return wat, bytes(wasmtime.wat2wasm(wat))


def artifact(name, raw, write, wat=None):
    wasm = OUT / f"{name}.wasm"
    if write:
        wasm.write_bytes(raw)
        if wat is not None:
            wasm.with_suffix(".wat").write_text(wat)
    assert wasm.read_bytes() == raw


def run_with_fuel(graph, raw, x, tokens):
    page = bytearray(EMPTY)
    page[RESULT:RESULT + 4] = bytes.fromhex("aabbccdd")
    final, run = invoke(graph, raw, x, page)
    assert run["status"] == 1
    assert wasmtime_run(raw, x, page) == (run["return_i32"], final)
    expected = page.copy()
    struct.pack_into("<i", expected, RESULT, run["return_i32"])
    assert final == expected
    first_write = 2 + len(tokens)
    cutoffs = []
    for fuel in range(run["source_steps"] + 1):
        cut_page, cut = invoke(graph, raw, x, page, fuel)
        assert cut["source_steps"] == fuel
        assert cut["status"] == (1 if fuel == run["source_steps"] else 2)
        assert cut_page == (expected if fuel >= first_write else page)
        cutoffs.append(cut)
    return final, {**run, "first_write_fuel": first_write, "fuel_cuts": cutoffs}


def guest_case(graph, optimizer, name, tokens, write):
    claim, optimizer_run = propose(graph, optimizer, tokens)
    proof = prove(claim)
    variants = (tokens, claim["replacement"])
    binaries = []
    for suffix, sequence in zip(("original", "candidate"), variants, strict=True):
        wat, raw = assemble(sequence)
        artifact(f"{name}-{suffix}", raw, write, wat)
        binaries.append(raw)
    accepted = proof["result"] == "unsat"
    inputs = (-2147483648, -1, 0, 17, 2147483647) if accepted else (proof["counterexample"]["x"],)
    replays = []
    for x in inputs:
        original, candidate = (run_with_fuel(graph, binary, x, seq)
                               for binary, seq in zip(binaries, variants, strict=True))
        assert (original[0] == candidate[0]) == accepted
        if not accepted:
            assert original[1]["return_i32"] == proof["counterexample"]["original"]
            assert candidate[1]["return_i32"] == proof["counterexample"]["candidate"]
        replays.append({"x": x, "original": original[1], "candidate": candidate[1]})
    return {"claim": claim, "proof": proof, "optimizer": optimizer_run, "replays": replays}


def terrain_case(graph, optimizer, write):
    raw = terrain.GUEST.read_bytes()
    program = decode_control(raw, scaled=True)
    ops = program.operations
    names = [op.opcode for op in ops]
    assert all(names[i:i + 4] != ["i32.const", "i32.add", "i32.const", "i32.sub"]
               or ops[i].immediate != ops[i + 2].immediate for i in range(len(ops) - 3))
    sites = [i for i in range(len(ops) - 1) if ops[i] == ControlOperation("i32.const", -1)
             and ops[i + 1] == ControlOperation("i32.add")]
    assert sites == [80]
    site, = sites
    assert names[site - 1] == "i32.load" and names[site + 2] == "i32.store"
    claim, optimizer_run = propose(graph, optimizer, (("x", 0), ("const", -1), ("add", 0)))
    proof = prove(claim)
    assert proof["result"] == "unsat"
    # The load already supplied x. Encode the guest's accepted suffix in place.
    assert claim["replacement"] == CANDIDATES[3]
    suffix = tuple(ControlOperation("i32." + op, value if op == "const" else None)
                   for op, value in claim["replacement"][1:])
    candidate = encode_control(replace(program, operations=ops[:site] + suffix + ops[site + 2:]))
    artifact("terrain-candidate", candidate, write)
    histories = {}
    for fixture in ("terrain", "terrain-omitted"):
        recorded = json.loads((OUT.parents[1] / "fixtures" / f"{fixture}.proof.json").read_text())
        run = recorded["session"]
        page = terrain.initial_page(run["initial"])
        ticks = []
        for inputs, expected in zip(run["inputs"], recorded["ticks"], strict=True):
            terrain.write(page, 4, [int(inputs["edit"])])
            terrain.write(page, 320, [4 * move for move in inputs["moves"]])
            original, orig_run = invoke(graph, raw, terrain.BASE, page, fuel=50_000)
            changed, candidate_run = invoke(graph, candidate, terrain.BASE, page, fuel=50_000)
            assert original == changed
            for binary, evidence in ((raw, orig_run), (candidate, candidate_run)):
                assert evidence["status"] == 1
                assert wasmtime_run(binary, terrain.BASE, page) == (evidence["return_i32"], original)
            assert sha(original) == expected["page_sha256"]
            assert orig_run["source_steps"] == expected["steps"] == candidate_run["source_steps"]
            page = bytearray(original)
            ticks.append({"original": orig_run, "candidate": candidate_run, "source_steps_saved": 0})
        histories[fixture] = ticks
    return {"cancellation_sites": [], "site_pc": site, "claim": claim, "proof": proof,
            "optimizer": optimizer_run, "original_sha256": sha(raw), "candidate_sha256": sha(candidate),
            "static_instructions_saved": len(ops) - len(decode_control(candidate, scaled=True).operations),
            "histories": histories}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-binaries", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    path = EXECUTOR / "machine.pt"
    graph = torch.jit.load(str(path))
    optimizer = (OUT / "optimizer.wasm").read_bytes()
    live = (("x", 0), ("const", -17), ("add", 0), ("const", -17), ("sub", 0))
    invalid = (("x", 0), ("const", 1), ("add", 0), ("x", 0), ("gt_s", 0))
    mismatch = live[:3] + (("const", -16), live[-1])
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        good = guest_case(graph, optimizer, "cancel", live, args.write_binaries)
        bad = guest_case(graph, optimizer, "signed-growth", invalid, args.write_binaries)
        assert good["proof"]["result"] == "unsat"
        assert bad["proof"]["counterexample"] == {"x": 2147483647, "original": 0, "candidate": 1}
        no_match, mismatch_run = propose(graph, optimizer, mismatch)
        assert no_match is None
        terrain_result = terrain_case(graph, optimizer, args.write_binaries)
    probe = (OUT / "rust-cancellation-probe.wasm").read_bytes()
    compiled = load_call_image(probe)
    assert compiled.functions == (("local.get", "end"),)
    print(json.dumps({"rust_probe": {"binary_sha256": sha(probe), "instructions": compiled.functions[0]},
                      "graph_sha256": sha(path.read_bytes()), "optimizer_sha256": sha(optimizer),
                      "z3_version": z3.get_version_string(), "cancellation": good,
                      "rejected_claim": bad, "no_match": {"input": mismatch, "replay": mismatch_run},
                      "terrain": terrain_result}, indent=2))


if __name__ == "__main__":
    main()
