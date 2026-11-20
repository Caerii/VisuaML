"""Check router decisions, exact tool traces, and causal output interventions."""
import json
import struct
from unittest.mock import patch

import torch
import wasmtime

from internal_tool import OUT, prepare_program
from frozen_call_checkpoint import digest
from fork_divergence import first_divergence
from replay_frozen_recurrent import PACKED, wasmtime_run
from visuaml_replay import Replay, make_bundle


def oracle(program, values):
    pc = 0
    registers = [0, values[1]]
    memory = [values[0], 0, 0, 0]
    trace = []
    for _ in range(48):
        opcode, reg, arg = program[pc]
        next_pc = pc+1
        if opcode == 1:
            registers[reg] = memory[arg]
        elif opcode == 2:
            memory[arg] = registers[reg]
        elif opcode == 3:
            registers[reg] = (registers[reg]+arg) % (1 << 32)
        elif opcode == 4 and registers[reg]:
            next_pc = arg
        elif opcode == 0:
            next_pc = pc
        pc = next_pc
        trace.append([pc, int(opcode != 0), *registers, *memory])
        if opcode == 0:
            return trace
    raise AssertionError("Tool exceeded its allowance")


WAT = """(module (memory (export "memory") 1 1)
 (func (export "run") (param i32) (result i32)
  (loop
    i32.const 0
    i32.const 0 i32.load i32.const 3 i32.add
    i32.store
    local.get 0 i32.const -1 i32.add local.tee 0 br_if 0)
  i32.const 0 i32.load))"""


def main():
    torch.set_num_threads(1)
    artifact = OUT / "internal-tool.pt"
    spec = json.loads((OUT / "heldout.json").read_text())
    assert digest(artifact.read_bytes()) == spec["frozen_sha256"]
    model = torch.jit.load(str(artifact))
    program = prepare_program(spec["instructions"])
    raw = bytes(wasmtime.wat2wasm(WAT))
    replay = Replay(raw)
    records = []
    with torch.inference_mode(), patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        for mode in (0, 1, 2):
            for pair in ((2, 4), (5, 3), (4294967294, 2) if mode != 2 else (6, 1)):
                tokens = torch.tensor([4, mode])
                inputs = torch.tensor(pair)
                result, route, chosen, trace = model(tokens, inputs, program, torch.tensor(True), torch.tensor(0))
                assert int(route) == mode
                disabled, _, _, skipped = model(tokens, inputs, program, torch.tensor(False), torch.tensor(0))
                shifted, _, _, _ = model(tokens, inputs, program, torch.tensor(True), torch.tensor(9))
                assert int(disabled) == pair[0] and not skipped.numel()
                record = dict(mode=mode, operands=pair, selected=chosen.tolist(), answer=int(result),
                              disabled=int(disabled), perturbed=int(shifted))
                if mode:
                    values = list(reversed(pair)) if mode == 2 else list(pair)
                    assert chosen.tolist() == values
                    expected = oracle(spec["instructions"], values)
                    actual = trace[:len(expected)].tolist()
                    delta = first_divergence(expected, actual)
                    if delta is not None:
                        raise AssertionError(f"Logical tool divergence at boundary {delta+1}: "
                                             f"expected {expected[delta]}, got {actual[delta]}")
                    assert not trace[len(expected):, 1].any()
                    memory = bytearray(65536)
                    struct.pack_into("<I", memory, 0, values[0])
                    bundle = make_bundle(replay, values[1], memory, 1000, 0)
                    finish = replay.finish(bundle["checkpoint"])
                    value, final = wasmtime_run(raw, values[1], memory)
                    assert int(result) == expected[-1][4] == finish["returned"] == (value & 0xffffffff)
                    assert finish["page_sha256"] == digest(final)
                    assert int(shifted) == int(result)+9 and int(disabled) != int(result)
                    stores = [step["access"][3] for step in finish["trace"] if step["access"][:2] == [2, 0]]
                    assert len(stores) == values[1]
                    assert stores == [expected[i][4] for i in range(2, len(expected)-1, 5)]
                    record.update(tool_trace=expected, wasm_steps=len(finish["trace"]))
                else:
                    assert int(result) == int(shifted) == pair[0] and not trace.numel()
                records.append(record)
    (OUT / "replay.json").write_text(json.dumps(dict(frozen_sha256=spec["frozen_sha256"],
          wasm_sha256=digest(raw), cases=records), indent=2)+"\n")
    (OUT / "reference.wat").write_text(WAT+"\n")
    print("Nine router cases; six exact tool traces and causal interventions verified")


if __name__ == "__main__":
    main()
