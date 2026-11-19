# VisuaML Voxel Replay

A small React Three Fiber lab for two deterministic Rust WASM scenes.
The canvas displays the guest's state. The inspector shows attempted
cells, destination materials, changed bytes, and the selected tick.

## Run locally

From the VisuaML checkout, use Node 20 or newer.

```bash
cd visuaml-client/voxel-replay
npm ci
npm run dev -- --port 4179
```

Open http://localhost:4179. Localhost lets the browser use Web Crypto
to hash every full 64 KiB memory page.

- **One Agent** uses the exact previously verified `voxel_tick.wasm`.
  Press Right twice for an empty-cell move followed by an obstacle hit.
  Pause queues a move and Single step commits it.
- **Two Agents** queues A with Left/Right and B with J/L. Press Space
  or Single step to submit both intentions on the same tick. Queue
  A right and B left twice. Both move inward on the first tick. On
  the second, they claim cell 2 together and stay in cells 1 and 3.
  Positions occupied at the start of a tick are reserved for the
  entire tick, including during attempted swaps.
- **Play scene** runs the shipped input sequence from its initial state.
  **Replay inputs** reruns your current recording from the beginning.
  The inspector's timeline can show an earlier tick while the live
  world remains at the end of your current recording.
- **Matches saved VisuaML proof** means the initial state, guest hash,
  inputs, and full post-tick page hashes match a verified fixture prefix.
  Other sessions show **Browser playback** and can be exported for
  offline verification.

The Rust shared ABI starts at byte 256 and uses little-endian i32 words.
Positions and requested deltas occupy offsets 0–12. Observed materials
occupy 16 and 20. The conflict flag is at 24, comparison scratch at 28,
and the five cells occupy offsets 32–48. The host supplies distinct
positions and in-range intentions. Both destination reads and collision
decisions use the pre-tick positions. The Rust guest owns world-state
writes. The browser lays out the input words, renders snapshots, and
hashes the page.

The first shared compilation generated five locals plus `i32.eq` and
`i32.ne`. The final guest has a uniform return value and performs
comparisons via a documented volatile scratch word, which keeps the
emitted code within the existing four-local, structured i32 dialect.
The inspector shows scratch writes alongside other guest changes.

## Replay and test

From this directory, activate the VisuaML research Python environment
with Wasmtime installed. Run the verifier without Python's optimize flag
so its assertions remain enabled.

```bash
export PYTHONPATH=../backend PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python verify_replay.py fixtures/single.json --fixture
python verify_replay.py fixtures/shared.json --fixture
python verify_replay.py /path/to/exported-voxel-shared.json > replay.proof.json
npm test
npm run build
```

The verifier builds each page independently, checks the binary hash,
and compares every byte and return value against Wasmtime on every
tick. The shipped fixtures also check independent expected outcomes
and duplicate occupancy. The NumPy run disallows packed fallback.

Rebuild the shared binary with rustc 1.100.0 and its
`wasm32-unknown-unknown` target.

```bash
out=$(mktemp -d)
../../benchmarks/results/v58-voxel-tick/build_guest.sh guest/shared_tick.rs "$out/shared.wasm"
cmp "$out/shared.wasm" guest/shared_tick.wasm
```

The optional Chromium check drives both scenes by keyboard, verifies
the proof badge, exercises pause and replay, exports both logs, checks
mobile overflow, and collects timing samples. With the dev server
running, use Playwright Core and an installed Chromium.

```bash
tools_dir=$(mktemp -d)
npm --prefix "$tools_dir" install playwright-core
PLAYWRIGHT_CORE="$tools_dir/node_modules/playwright-core/index.mjs" \
  CHROMIUM_PATH=/usr/bin/chromium node tools/browser_check.mjs
```

## Measurements

| Scene | WASM steps per tick | Offline verification for both ticks | Chromium module instantiation |
| --- | --- | --- | --- |
| One agent | 22, 18 | 40.5 ms | 0.7 ms |
| Two agents | 78, 46 | 89.0 ms | 2.6 ms |

Offline verification includes both engines and page comparison. Module
preparation is measured separately in the proof files. The NumPy path
still prepares each invocation. The proofs split that invocation at the
first route and report the remainder separately.

The live shared-scene sample recorded 61 rendered frames in headless
Chromium with SwiftShader. Median CPU render submission was 0.4 ms and
the median frame interval was 16.7 ms. Submission excludes GPU
completion. Two interactive simulation measurements were below the
browser timer's resolution at the median. The separate hot-loop
benchmark times argument writes and WASM calls in ten batches of 40,000
calls per scene, after warmup. Its batch medians were about 0.016 μs
per call for one agent and 0.036 μs for two agents. These tiny repeated
calls exclude hashing, rendering, and UI updates, which are outside
the timed loop.

Signed comparison remains the next ALU candidate based on the prior
Sudoku and BFS profiles. This demo keeps the executor implementation
as-is. AI Escargot's prior verified prefix is a separate scale result.

[Raw Chromium samples](evidence/chromium-timings.json) · [One-agent proof](fixtures/single.proof.json) · [Two-agent proof](fixtures/shared.proof.json)
