"""Check both relocations and a one-symbol fork against two execution engines."""
import argparse
import base64
import json
import struct
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import wasmtime

from replay_frozen_recurrent import PACKED, sha
from replay_monitor_program import COMPILED, ROOT, native_rows, no_regions, register
from replay_pocket import drive
from replay_pocket_universal import BASE, FUEL
from visuaml.compiler_lab.frozen_calls_v58 import initialize, load_call_image

OUT = ROOT / "evidence/pocket-tape-fork"


def word(page, index):
    return struct.unpack_from("<i", page, BASE + 4 * index)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    fixture = json.loads((ROOT / "fixtures/pocket-tape-fork.json").read_text())
    runner = ROOT / "tools/pocket_tape_fork.mjs"
    saved = json.loads(subprocess.check_output(["node", runner, "export"]))
    exported = (json.dumps(saved["checkpoint"], indent=2)+"\n").encode()
    with TemporaryDirectory() as temp:
        path = Path(temp) / "checkpoint.json"
        path.write_bytes(exported)
        page_record = json.loads(subprocess.check_output(["node", runner, path]))
    page_record["records"][:0] = saved["records"]
    checkpoint, restored, edited = (base64.b64decode(page_record[key]) for key in ("checkpoint", "restored", "edited"))
    table_size = 3 * fixture["states"] * fixture["symbols"]
    tape_index = 8 + table_size
    tape_start = BASE + 4 * tape_index
    rules = {}
    for rule in fixture["rules"].splitlines():
        q, read, dest, write, move = rule.split()
        rules[int(q), int(read)] = int(dest), int(write), {"L": -1, "S": 0, "R": 1}[move]

    def snapshot(page, origin):
        return dict(state=word(page, 3), head=origin+word(page, 4), count=word(page, 6),
                    cells={origin+i: word(page, tape_index+i) for i in range(word(page, 2))
                           if word(page, tape_index+i)})

    def evolve(page):
        expected = bytearray(page)
        def put(index, value):
            struct.pack_into("<i", expected, BASE + index*4, value)
        head, state = word(page, 4), word(page, 3)
        rule = rules.get((state, word(page, tape_index+head)))
        if rule is None:
            put(5, 1)
        else:
            dest, symbol, move = rule
            put(7, head + move)
            if 0 <= head + move < word(page, 2):
                put(tape_index+head, symbol)
                put(3, dest)
                put(4, head + move)
                put(6, word(page, 6) + 1)
            else:
                put(5, 2)
        return expected

    torch.set_num_threads(1)
    register()
    graph_file = COMPILED / "machine.pt"
    graph = torch.jit.load(str(graph_file))
    wasm = (ROOT / "evidence/pocket-tape-compiler/erase.linked.wasm").read_bytes()
    image = native_rows(no_regions(load_call_image(wasm)))
    engine = wasmtime.Engine()
    module = wasmtime.Module(engine, wasm)
    logs = {name: [] for name in ("short", "clean", "fork", "wide")}
    inputs = {name: [] for name in logs}
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        for record in page_record["records"]:
            name, origin, status = record["name"], record["origin"], record["status"]
            before, after = (base64.b64decode(record[key]) for key in ("before", "after"))
            assert before[BASE+32:tape_start] == after[BASE+32:tape_start] == checkpoint[BASE+32:tape_start]
            assert after == evolve(before) and status == word(after, 5)
            store = wasmtime.Store(engine)
            exports = wasmtime.Instance(store, module, []).exports(store)
            exports["memory"].write(store, before)
            assert exports["run"](store, 0) == status and bytes(exports["memory"].read(store)) == after
            state = initialize(image, 0, before, FUEL)
            steps, _, result, _ = drive(graph, image.rows, state, fuel=FUEL)
            assert result == status and state.vm.page.numpy().tobytes() == after
            logical = snapshot(after, origin)
            if status == 2:
                assert logical == snapshot(before, origin)
                assert word(after, 7) in (-1, word(before, 2))
            logs[name].append(dict(status=status, logical=logical, wasm_steps=steps, page_sha256=sha(after)))
            inputs[name].append((before, after))

    assert inputs["short"][0][1] == inputs["short"][1][0] == checkpoint
    assert inputs["clean"][0][0] == restored and inputs["fork"][0][0] == edited
    assert snapshot(checkpoint, 0) == snapshot(restored, -1)
    assert [(i, a, b) for i, (a, b) in enumerate(zip(restored, edited)) if a != b] == [(tape_start+4, 1, 2)]
    assert [r["status"] for r in logs["clean"]] == [0, 0, 2, 0, 0, 1]
    assert [r["status"] for r in logs["fork"]] == [0, 0, 2, 0, 0, 1]
    for name in ("clean", "fork"):
        held = base64.b64decode(page_record["continuations"][name])
        assert held == inputs[name][2][0] == inputs[name][1][1]
        resumed = inputs[name][3][0]
        assert [(i, a, b) for i, (a, b) in enumerate(zip(held, resumed)) if a != b] == [(BASE+8, 2, 3)]
    clean = [r["logical"] for r in logs["clean"] if r["status"] != 2]
    fork = [r["logical"] for r in logs["fork"] if r["status"] != 2]
    assert clean == [r["logical"] for r in logs["wide"][1:]]
    assert inputs["clean"][3:] == inputs["wide"][3:]
    first = next(i+1 for i, (a, b) in enumerate(zip(clean, fork, strict=True)) if a != b)
    assert first == 1 and clean[0]["state"] == 1 and fork[0]["state"] == 4
    assert clean[-1]["cells"] == {-1: 1, 0: 1, 1: 2}
    assert fork[-1]["cells"] == {-1: 2, 0: 1, 1: 2}
    evidence = dict(graph_sha256=sha(graph_file.read_bytes()), wasm_sha256=sha(wasm),
                    table_sha256=sha(checkpoint[BASE+32:tape_start]), packed_fallback=False,
                    checkpoint_sha256=sha(checkpoint), fork_byte=tape_start+4, first_fork_transition=fork[0]["count"], first_fork_step_after_resume=first, runs=logs)
    assert saved["checkpoint"]["identities"]["graph"] == evidence["graph_sha256"]
    files = {"checkpoint.json": exported, "replay.json": (json.dumps(evidence, indent=2)+"\n").encode()}
    for name, data in files.items():
        path = OUT / name
        if args.write:
            path.write_bytes(data)
        else:
            assert path.read_bytes() == data
    print(f"Verified {len(page_record['records'])} attempted transitions; fork diverges at transition {first}")
    print(f"Clean: {clean[-1]}; fork: {fork[-1]}")


if __name__ == "__main__":
    main()
