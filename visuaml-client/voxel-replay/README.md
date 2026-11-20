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

## Terrain edit and counterfactual

Open **Terrain Edit** and press **Play scene**. In tick 1, B moves from
cell 3 to cell 2. A clears cell 1 after movement resolves. On tick 2,
B reads material 0 and enters the passage. Use **Replay without edit**
to remove only A's action from the input log. B takes the same actions
and remains at cell 2 because cell 1 still contains material 7.
Select **Tick 1 · Edit** to inspect the terrain write, then **Tick 2**
to see its consequence.

The guest takes a pre-tick snapshot of occupied cells and counts all
destination claims, including stay intentions. Agents can enter only a
unique destination that was empty and traversable at tick start.
Contested and preoccupied destinations are blocked. The guest commits
A's edit to the cell on its right after resolving all movement.
Everyone sees that edit in the next tick. The ABI stores lengths in
bytes and uses one i32 word per cell or agent. It reserves 1,856 bytes
of one 64 KiB guest page for up to 64 cells and agents.

The initial Rust loop compilation used a result-bearing WASM loop and
too many locals for the current VM. The working source keeps loop
indices and temporary addresses in documented volatile scratch words,
uses signed bounds, and joins the loops through an empty block.
The resulting unmodified rustc 1.100.0 binary uses 171 instructions
and three locals. Browser, Wasmtime, and VisuaML all use those bytes.

```bash
export PYTHONPATH=.:../backend PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python verify_replay.py fixtures/terrain.json --fixture
python verify_replay.py fixtures/terrain-omitted.json --fixture
python verify_replay.py /path/to/exported-voxel-terrain.json \
  --expected-proof fixtures/terrain.proof.json --divergence /tmp/voxel-divergence.json
```

Omit `--expected-proof` when verifying a new history. When a mismatch
occurs, the command stops at the first divergent tick. The divergence
file includes that input and the preceding page plus both resulting
engine pages as base64. The full input log and initial state also
allow reconstruction from tick zero. The regression flips one edit
input at tick 1 and a NumPy page byte at tick 2.

```bash
out=$(mktemp -d)
python tools/scale_terrain.py --ticks 12 --output "$out/sweep"
PLAYWRIGHT_CORE="$tools_dir/node_modules/playwright-core/index.mjs" \
  CHROMIUM_PATH=/usr/bin/chromium node tools/terrain_browser.mjs "$out/sweep"
python tools/frame_pair.py
# Optional full Escargot correctness run. Execute separately from timings.
python tools/check_escargot.py
```

### First sustained scale sweep

Each row has 12 verified ticks. The independent model checks all
positions, materials and conflict counts. NumPy and Wasmtime compare
every byte each tick. Chromium independently matches all 72 page
hashes. The reported p95 is the maximum in each 12-tick sample.

| Agents | Scene | Cells | Source steps | NumPy agent-ticks/s | NumPy p95 tick |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2 | Sparse | 8 | 5,400 | 8.5 | 254 ms |
| 2 | Crowded | 5 | 4,368 | 10.5 | 199 ms |
| 4 | Sparse | 16 | 10,200 | 9.3 | 452 ms |
| 4 | Crowded | 10 | 8,136 | 10.9 | 400 ms |
| 8 | Sparse | 32 | 19,800 | 9.4 | 903 ms |
| 8 | Crowded | 20 | 15,672 | 12.0 | 689 ms |

Half of the crowded agent-ticks have conflicting claims. The dense
rows are shorter and blocked moves skip stores, so the crowded
scenes execute fewer source steps at each agent count. The archived
Chromium batches separately time warmed guest calls and host input
writes, with rendering, hashing, and inspection excluded. The live
two-agent terrain view also measures CPU render submission and frame
interval using headless SwiftShader. The numbers in each fixture proof
include the individual NumPy and Wasmtime calls.

### Profile-driven frame change

Before the change, signed comparison took about 9% of the crowded
8-agent sample and 16% of a 4,000-instruction BFS prefix. Frame
transitions took about 24% and 20%, respectively. The frame circuit
now copies the retained rows directly when its validated one-hot
action has unit retain gain. The other candidates have zero weight
in that case. Non-unit gains and other actions still use the general
contraction.

Across three alternating pairs after warmup, the median of each
four-tick world's mean execution time fell from 796 to 676 ms per tick
(1.18×). Execution on a separate 2,000-step BFS cutoff fell from 1,466
to 1,339 ms (1.10×). Per-call time through the first route was measured
separately. Matching world hashes and exact BFS cutoff pages were
checked in both variants. Operand transitions now account for the
largest measured share of the terrain sample, at about 24%.

[Scale proofs and raw timing samples](evidence/terrain)

## Tensor assembly and learned routing

The [causal tensor assembly](evidence/causal-tensor-assembly) lowers the
bounded tool ISA to a fixed softmax decoder that appends structured records.
The [paired routing experiment](evidence/paired-routing) uses a learned
controller over a typed registry with the frozen synthesized executor.
The [original router diagnostic](evidence/frozen-routing-diagnostic) replays
the full feedback sweep, separates text order from physical order, and
forces one handle to isolate its effect. The
[Tensor Programs research note](evidence/tensor-programs-research.md)
describes a separate scaling and activation-patching study.

Run the local viewer and open `/replay-viewer.html` to inspect or download
the saved causal and paired-routing replays.
