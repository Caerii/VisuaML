"""Paired exact-frame timings for shifting the Monitor raster during scroll."""
import json
from unittest.mock import patch

import torch

from replay_frozen_recurrent import PACKED, sha
from replay_monitor_program import COMPILED, ROOT, register, runlog, stats
from replay_pocket_compile import OUT, words

torch.set_num_threads(1)
register()
graph = torch.jit.load(str(COMPILED / "machine.pt"))
buttons = json.loads((ROOT / "evidence/monitor-program/renderer.json").read_text())["buttons"]
compile_tick = len(buttons) + len("10 SET 7\n20 ADD 6\n30 PRINT\nNEXT\nRUN\nCOMPILE\n") - 1
buttons += [ord(c) if c != "\n" else 13 for c in "10 SET 7\n20 ADD 6\n30 PRINT\nNEXT\nRUN\nCOMPILE\n"]
buttons += [-1] + [ord(c) if c != "\n" else 13 for c in "HELP\n" + "MAP\n"*7]
binaries = {"dirty": (OUT / "compiler-baseline.wasm").read_bytes(),
            "shift": (ROOT / "guest/pocket_compile.wasm").read_bytes()}
runs = []
with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
    for order in (("dirty", "shift"), ("shift", "dirty"), ("dirty", "shift")):
        captures = {name: runlog(graph, binaries[name], buttons) for name in order}
        old, new = captures["dirty"], captures["shift"]
        assert old[1] == new[1]
        assert [t["transcript"] for t in old[0]] == [t["transcript"] for t in new[0]]
        assert {words(page, 432)[0] for page in new[2]} >= {0, 1, 2, 3}
        runs.append({name: captures[name][0] for name in order})
        print(f"{order}: {len(buttons)} frames match", flush=True)
summary = {}
for kind in ("printable", "backspace", "enter", "scroll"):
    summary[kind] = {}
    for name in binaries:
        samples = [tick for run in runs for tick in run[name]
                   if tick["kind"] == kind and tick["tick"] != compile_tick and tick["button"] != -1]
        combined = [sum(p["elapsed_ns"] for p in tick["phases"]) for tick in samples]
        draw = [tick["phases"][1]["elapsed_ns"] for tick in samples]
        summary[kind][name] = dict(total_ms=stats([ms/1e6 for ms in combined]),
                                   draw_ms=stats([ms/1e6 for ms in draw]),
                                   draw_steps=stats([tick["phases"][1]["source_steps"] for tick in samples]))
result = dict(buttons=buttons, both_framebuffers_exact=True, both_transcripts_exact=True,
              packed_fallback=False, wasm_sha256={name: sha(raw) for name, raw in binaries.items()},
              summary=summary, runs=runs)
(OUT / "scroll-timings.json").write_text(json.dumps(result, separators=(",", ":")) + "\n")
print(json.dumps(summary, indent=2))
