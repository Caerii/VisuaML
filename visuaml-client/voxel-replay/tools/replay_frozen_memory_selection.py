"""Verify live memory selection and measure guest memory-op coverage."""
from collections import Counter
import json
import struct
from unittest.mock import patch

import torch
import wasmtime

import terrain_verify as terrain
from replay_frozen_calls import NESTED, run
from replay_frozen_multistep import CALLS, terrain_histories, witness
from replay_frozen_recurrent import ARCHIVE, EVIDENCE, PACKED, graph_case, sha
from visuaml.compiler_lab.frozen_calls_v58 import load_call_image

OUT = EVIDENCE.parent / "frozen-memory-selection"


def coverage(one):
    session = terrain.scale_session(8, False, 1)
    page = terrain.initial_page(session["initial"])
    terrain.write(page, 320, [4 * delta for delta in session["inputs"][0]["moves"]])
    bfs = (ARCHIVE / "v58-stateful-scale-7125937/bfs.wasm").read_bytes()
    arg, graph_page, *_ = graph_case(0)
    result = {}
    for name, raw, address, initial in (
        ("8-agent terrain", terrain.GUEST.read_bytes(), terrain.BASE, page),
        ("full BFS", bfs, arg, graph_page),
    ):
        image = load_call_image(raw)
        final, trace = run(one, image, address, initial, fuel=100_000, trace=True)
        assert int(final.vm.status) == 1
        counts = Counter(row["opcode"] for row in trace)
        assert len(trace) == (1650 if name == "8-agent terrain" else 41591)
        accesses = counts["i32.load"] + counts["i32.store"]
        result[name] = {"binary_sha256": sha(raw), "source_steps": len(trace),
                        "opcode_counts": dict(counts), "memory_rows": accesses,
                        "discarded_gathers_removed": len(trace) - accesses}
    return result


def main():
    torch.set_num_threads(1)
    ref = torch.jit.load(str(CALLS / "machine.pt"))
    path = OUT / "machine.pt"
    graph = torch.jit.load(str(path))
    empty = bytearray(65536)
    nested = bytes(wasmtime.wat2wasm(NESTED))
    rollback = (ARCHIVE / "v58-array-memory/rollback.wasm").read_bytes()
    undo = empty.copy()
    undo[1:5] = undo[7:11] = bytes.fromhex("01000080")
    rust = (CALLS / "passage.wasm").read_bytes()
    world = bytearray(16 << 20)
    struct.pack_into("<iii", world, len(world) - 12, 7, 1, 0)
    cases = {}
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        for name, raw, arg, page, status, fuel, pages in (
            ("nested-return", nested, 0, empty, 1, 100, 1),
            ("nested-trap", nested, 65533, empty, 3, 100, 1),
            ("rollback", rollback, 0, undo, 1, 100, 1),
            ("before-write", rollback, 0, undo, 2, 8, 1),
            ("after-write", rollback, 0, undo, 2, 9, 1),
            ("rust-helper-edit", rust, len(world) - 12, world, 1, 100, 256),
        ):
            cases[name] = witness(ref, graph, raw, arg, page, status, fuel=fuel, pages=pages,
                                  canonical_no_access=True)
        terrain_results = terrain_histories(ref, graph)
        counts = coverage(ref)
    print(json.dumps({"graph_sha256": sha(path.read_bytes()),
                      "operators": sorted(torch.jit.export_opnames(graph)),
                      "cases": cases, "terrain_histories": terrain_results,
                      "full_instruction_coverage": counts}, indent=2))


if __name__ == "__main__":
    main()
