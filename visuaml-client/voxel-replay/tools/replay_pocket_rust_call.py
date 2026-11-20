"""Replay both language boundaries under the unchanged frozen graph."""
import argparse
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import wasmtime

from replay_frozen_calls import run
from replay_frozen_multistep import CALLS
from replay_frozen_recurrent import PACKED, sha, wasmtime_run
from replay_monitor_program import COMPILED, ROOT, keys, register, runlog, load_call_image, native_rows, no_regions
from replay_pocket_compile import words
from replay_pocket import drive
from visuaml.compiler_lab.frozen_calls_v58 import initialize

OUT = ROOT / "evidence/pocket-rust-call"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    def save(path, data):
        if not isinstance(data, bytes):
            data = (json.dumps(data, indent=2) + "\n").encode()
        if args.write:
            path.write_bytes(data)
        else:
            assert path.read_bytes() == data, path

    torch.set_num_threads(1)
    register()
    graph_path = COMPILED / "machine.pt"
    graph = torch.jit.load(str(graph_path))
    single = torch.jit.load(str(CALLS / "machine.pt"))
    raw = (ROOT / "guest/pocket_rust_call.wasm").read_bytes()
    rust = OUT / "passable.wasm"
    assert rust.read_bytes() == (OUT / "passable.raw.wasm").read_bytes()
    linker = ROOT / "tools/link_pocket_rust_call.mjs"

    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")), TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for cell in (2, 3):
            name = f"cell-{cell}"
            value = int(cell == 3)
            buttons = [0] + keys(f"10 SET {cell}\n20 CALL\n30 PRINT\nRUN\nCOMPILE\n")
            records, _, pages, _ = runlog(graph, raw, buttons, (OUT / "monitor.raw.wasm").read_bytes(), retain=True)
            page = pages[-1]
            assert "WASM READY" in records[-1]["transcript"] and str(value) in records[-1]["transcript"]
            binary = page[24576:24576 + words(page, 400)[0]]
            source = tmp / "emitted.wasm"
            source.write_bytes(binary)
            output = tmp / "linked.wasm"
            link = json.loads(subprocess.check_output(
                ["node", linker, source, rust, output], text=True))
            linked = output.read_bytes()
            delta = link["delta"]
            image = load_call_image(linked)
            native = native_rows(no_regions(image))
            state = initialize(native, 0, bytes(65536), 2000)
            steps, _, result, _ = drive(graph, native.rows, state, fuel=2000)
            memory = state.vm.page.numpy().tobytes()
            assert (result, memory) == wasmtime_run(linked, 0, bytes(65536))
            assert (result, words(memory, 16), words(memory, 32, 3), words(memory, 64)) == (
                1, (value,), (cell, value, value), (cell if value else 0,))
            audited, trace = run(single, image, 0, bytes(65536), trace=True)
            assert len(trace) == steps and audited.vm.page.numpy().tobytes() == memory
            call = next(i for i, row in enumerate(trace) if row["opcode"] == "call")
            ret = next(i for i, row in enumerate(trace) if i > call and row["depth"] == 0)
            assert trace[call]["depth"] == 1 and trace[call]["locals"][0] == cell
            assert trace[ret]["stack"] == [value]
            assert trace[ret]["next_image_pc"] == image.starts[1] + trace[call]["pc"] + 1
            mapping = []
            for i in range(3):
                line, start, end = words(page, 24000 + 12*i, 3)
                assert linked[start+delta:end+delta] == binary[start:end]
                mapping.append(dict(line=line, emitted_start=start, emitted_end=end,
                                    linked_start=start+delta, linked_end=end+delta))
            assert binary[mapping[1]["emitted_start"]+2:][:2] == b"\x10\x00"
            meta = dict(binary_sha256=sha(linked), page_sha256=sha(memory), emitted_sha256=sha(binary),
                        rust_sha256=sha(rust.read_bytes()), printed=[value], graph_sha256=sha(graph_path.read_bytes()),
                        packed_fallback=False, source_steps=steps, source_map=mapping, call_step=call,
                        return_step=ret, trace=trace, **link)
            for tick in records:
                for phase in tick["phases"]:
                    phase.pop("elapsed_ns")
            session = dict(version=1, cartridge="rust-call-monitor", binary_sha256=sha(raw), buttons=buttons)
            for suffix, data in (("emitted.wasm", binary), ("linked.wasm", linked), ("map.json", meta)):
                save(OUT / f"{name}.{suffix}", data)
            path = ROOT / "fixtures" / f"pocket-rust-{name}.json"
            save(path, session)
            save(path.with_suffix(".proof.json"), dict(session=session, wasm_sha256=sha(raw),
                 graph_sha256=meta["graph_sha256"], packed_fallback=False, ticks=records))
            print(f"{name}: Rust returned {value}, {steps} linked source steps", flush=True)

        wrong = wasmtime.wat2wasm('(module (memory (export "memory") 1 1) '
                                  '(func (export "run") (param i64) (result i32) i32.const 0))')
        mismatch = tmp / "mismatch.wasm"
        mismatch.write_bytes(wrong)
        output.unlink()
        rejected = subprocess.run(["node", linker, source, mismatch, output], capture_output=True, text=True)
        assert rejected.returncode and "Expected unary i32" in rejected.stderr and not output.exists()
        save(OUT / "mismatch.wasm", bytes(wrong))


if __name__ == "__main__":
    main()
