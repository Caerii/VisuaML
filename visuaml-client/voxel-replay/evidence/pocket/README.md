# VisuaML Pocket · Passage Relay

[Playthrough GIF](playing.gif) · [Edit and no-edit replays GIF](edit-vs-omitted.gif)

From the repository root:

```sh
cd visuaml-client/voxel-replay
npm ci
npm run dev -- --port 4179
```

Open `http://localhost:4179/pocket.html`. **Dig** or **X** clears the
block. **Step** or **Right Arrow** advances the next tick. **Run** and
**Pause** control the 20 Hz clock. **Reset** boots a fresh instance.
Either replay button loads a saved history, and the inspector's
timeline can jump straight to the edit or subsequent movement.
The inspector shows source PCs and writes when the current browser
page and framebuffer match the saved VisuaML replay. **Save memory**
pauses and downloads a full 64 KiB checkpoint. **Export input log**
downloads the exact button history for offline verification.

## One-page guest ABI

The [Rust source](../../guest/pocket.rs) and its original
[WASM binary](../../guest/pocket.wasm) use a single 65,536-byte page,
initially all zero. The guest function takes address 256. Memory words
are little-endian i32. The host writes the button at byte 256 and the
phase at 260. Button zero idles, one digs, and minus one restarts the
scene within an existing instance. A saved session starts with a zero
boot input. Phase zero updates the world, and phase one draws pending
changes. The world stores the block flag at byte 268 and the second
agent's position flag at 272. On boot the guest draws the rails,
the first agent, the block, and the second agent. On later ticks it
redraws only the affected passage or agent tiles. It tests movement
against the pre-edit block.

The 160 × 144 framebuffer starts at byte 4096. It uses **5,760 bytes**,
40 per scanline, with the leftmost pixel of each byte in bits 7–6 and
the following pixels in bits 5–4, 3–2, and 1–0. The four shades in
index order are `#dcebb3`, `#93ab70`, `#41624e`, and `#152d28`.
The host exposes a new frame after both phases finish and copies
exactly the packed framebuffer. It expands those bytes to RGBA and
submits them to Canvas. The input log and optional full checkpoint
are separate. Published frames are independent of later writes.

## One event, two histories

| Tick | Edit history | No-edit history |
| --- | --- | --- |
| 0 | Guest draws blocked scene | Same initial frame |
| 1 | PC 59 clears the block at 268. PC 170 starts clearing its pixel words at 6756. | Block and frame remain |
| 2 | PC 40 updates the second agent at 272. Drawing replaces the destination and erases the old tile. | Agent waits and frame remains |

The [edit fixture](../../fixtures/pocket-edit.json) uses `[0,1,0]`,
and the [counterfactual](../../fixtures/pocket-omitted.json) uses
`[0,0,0]`. The compiled VisuaML graph, single-step graph and Wasmtime
matched complete pages at all 12 phase boundaries with packed fallback
blocked. Chromium matched every complete-page SHA-256 and compared
every framebuffer byte against the archived proof. Independent pixel
expectations test the rails, first agent, blocked and opened passage,
and both second-agent positions. Tests also retain a published frame
across later writes, check the presented Canvas palette and exercise
keyboard, pause, step, reset, replay, and downloads.

The browser runs its own WASM interpreter. The VisuaML graph runs
offline with the shared compiled CPU lowering, writes the same bytes,
and records replay evidence. The browser status marks a match to that
evidence. The offline graph has its own replay command below.

## Paced playback

The passage repeat and frequent-change stream each run 240 scheduled
ticks. The browser measurement includes input writes, both guest phases,
the 5,760-byte copy, and Canvas submission. It separately records
the next animation callback as a paint opportunity. Timing includes
schedule delay and excludes replay hashing, checkpoint exports and the
inspector DOM. Browser clocks quantize very short phases near 0.1 ms.
All 480 measured browser frames match the expected replay frames.

| 240 ticks per stream | Chromium scheduled median / p95 | Paint opportunity p95 | Browser misses | Offline VisuaML scheduled median / p95 | Offline misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| Passage repeat | 1.10 / 1.70 ms | 8.40 ms | 0 | 2.32 / 5.48 ms | 1 |
| Frequent changes | 0.80 / 1.60 ms | 15.60 ms | 0 | 4.66 / 5.31 ms | 0 |

The offline run includes input, guest simulation, guest drawing,
publication, and CPU palette expansion to RGBA. For passage repeat
and frequent changes respectively, its phase p95s were:

| Phase | Passage repeat | Frequent changes |
| --- | ---: | ---: |
| Input | 0.020 ms | 0.016 ms |
| Guest simulation | 1.53 ms | 1.66 ms |
| Guest drawing | 3.38 ms | 3.42 ms |
| Publication | 0.010 ms | 0.010 ms |
| Palette expansion | 0.41 ms | 0.37 ms |

Chromium's input, simulation, drawing, and copy were at or below
0.1 ms p95. Palette conversion and Canvas submission together were
about 0.5 ms p95. The [Chromium samples](chromium-paced.json) and
[compiled VisuaML samples](visuaml-paced.json) report each phase
independently. The browser's animation callback is a display
opportunity, while the offline measurement ends at RGBA publication.

Full framebuffer copy is below 0.1 ms p95 in both configurations, so
publication uses one compact copy per tick. Guest-side tile redraw is
already selective. Full-memory checks run separately from the timer.

The [eight-agent simulation-only measurements](../compiled-ordinary-cpu/README.md)
remain a separate workload. This console run uses the two-agent
passage scene and includes guest drawing.

## Replay and rebuild

From `visuaml-client/voxel-replay`, with backend dependencies, CPU
PyTorch 2.9.1, Wasmtime and a C++17 compiler:

```sh
export PYTHONPATH=.:../backend OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python tools/replay_pocket.py
python tools/replay_pocket.py --session /path/to/pocket-session.json
python tools/replay_pocket.py --paced
npm test
npm run build
```

Rebuild the unchanged guest with `bash tools/build_pocket.sh` using
rustc 1.100.0 and its wasm32-unknown-unknown target. Browser timing
and interaction tests use `node tools/pocket_browser.mjs` against the
dev server with Playwright Core and Chromium. `PLAYWRIGHT_CORE`,
`CHROMIUM_PATH` and `POCKET_URL` can override defaults. GIFs were
captured from that browser player and slowed to make each change
readable. Reproduce them with `node tools/capture_pocket_gifs.mjs
/tmp/pocket-frames` and `python tools/pack_pocket_gifs.py
/tmp/pocket-frames evidence/pocket` with Pillow.
