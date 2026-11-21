"""Check compiled/reference handoffs and measure warm end-to-end continuation."""
import json
from pathlib import Path
from statistics import median
import struct
from time import perf_counter_ns
from unittest.mock import patch

import torch
import wasmtime

from replay_frozen_recurrent import ARCHIVE, PACKED, graph_case, wasmtime_run
from visuaml.replay import Replay, open_bundle
from visuaml.replay.checkpoint import digest

BASE = Path(__file__).resolve().parents[1]


def measure(replay, argument, initial, fuel, cut, repeats):
    start = perf_counter_ns()
    first = replay.start(argument, initial, fuel)
    first = replay.run(first, cut, runner="compiled")
    elapsed = perf_counter_ns() - start
    reference = replay.run(replay.start(argument, initial, fuel), cut)
    checkpoint = replay.checkpoint(reference)
    assert replay.checkpoint(first) == checkpoint
    first_compiled_prefix_ns = elapsed
    expected, page = wasmtime_run(replay.raw, argument, initial)
    samples = dict(reference=[], compiled=[])
    setup = dict(reference=[], compiled=[])
    for i in range(repeats):
        order = ("reference", "compiled") if i % 2 == 0 else ("compiled", "reference")
        for mode in order:
            start = perf_counter_ns()
            state = replay.restore(checkpoint)
            setup[mode].append(perf_counter_ns()-start)
            start = perf_counter_ns()
            result = replay.run(state, runner=mode)
            samples[mode].append(perf_counter_ns()-start)
            summary = replay.summary(result)
            assert summary["status"] == 1 and summary["returned"] == expected & 0xffffffff
            assert summary["page_sha256"] == digest(page)
            if mode == "reference":
                final_reference = result
            else:
                final_compiled = result
        assert all(torch.equal(a, b) for a, b in zip(final_reference.vm, final_compiled.vm))
        assert torch.equal(final_reference.frames, final_compiled.frames)
        assert torch.equal(final_reference.depth, final_compiled.depth)
    return dict(cut=cut, source_steps=fuel-summary["fuel"], first_compiled_prefix_ns=first_compiled_prefix_ns,
                samples_ns=samples, median_ns={mode: median(times) for mode, times in samples.items()},
                restore_ns=setup, summary=summary)


def main():
    torch.set_num_threads(1)
    document = json.loads((BASE / "release/examples/suspended-passage-report.json").read_text())
    replay, checkpoint = open_bundle(document["bundle"])
    initial = bytearray(65536)
    struct.pack_into("<6i", initial, 65512, 7, 1, 0, 0, -1, 0)
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        passage = measure(replay, 65512, initial, 2000, 14, 32)
        reference = replay.start(65512, initial, 2000)
        interior = []
        for cut in range(1, 31):
            prior_pc = int(reference.vm.pc)
            reference = replay.advance(reference)
            compiled = replay.run(replay.start(65512, initial, 2000), cut, runner="compiled")
            assert replay.checkpoint(compiled) == replay.checkpoint(reference)
            # An eligible predecessor and a live eligible successor witness
            # that a larger budget could cross this exact checkpoint.
            if replay._compiled.rows[prior_pc, 17] and replay._compiled.rows[int(reference.vm.pc), 17]:
                interior.append(cut)
        assert interior
        ordinary_cut = interior[0]
        exact = replay.run(replay.start(65512, initial, 2000), ordinary_cut, runner="compiled")
        audit = replay.finish(replay.checkpoint(exact))
        assert len(audit["trace"]) == 31-ordinary_cut
        assert replay.finish(checkpoint)["trace"][0]["pc"] == 11
        # A budget of one at zero remaining fuel marks exhaustion without a store.
        for fuel in (13, 14, 15):
            left = replay.run(replay.start(65512, initial, fuel))
            right = replay.run(replay.start(65512, initial, fuel), runner="compiled")
            assert int(left.vm.status) == 2 and replay.summary(left) == replay.summary(right)
        trap_guest = wasmtime.wat2wasm("""
            (module (memory (export "memory") 1 1)
              (func (export "run") (param i32) (result i32)
                local.get 0 i32.const 123 i32.store i32.const 0))
        """)
        trap_replay = Replay(bytes(trap_guest))
        left = trap_replay.run(trap_replay.start(65533, bytes(65536), 100))
        right = trap_replay.run(trap_replay.start(65533, bytes(65536), 100), runner="compiled")
        assert trap_replay.summary(left) == trap_replay.summary(right)
        assert int(right.vm.status) == 3 and not right.vm.page.any()
        passage["straddling_store_trapped_without_write"] = True
        passage["checked_source_boundaries"] = 30
        passage["compiled_interior_cuts"] = interior
        raw = (ARCHIVE / "v58-stateful-scale-7125937/bfs.wasm").read_bytes()
        argument, page, *_ = graph_case(0)
        bfs = measure(Replay(raw), argument, page, 100000, 1024, 3)
        assert bfs["source_steps"] == 41591
    print(json.dumps(dict(passage=passage, bfs=bfs), indent=2))


if __name__ == "__main__":
    main()
