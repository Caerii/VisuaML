# Programmable Pocket Monitor

[Watch the typing and rerun GIF](program-monitor.gif).

```sh
cd visuaml-client/voxel-replay
npm ci
npm run dev -- --port 4180
```

Open `http://localhost:4180/pocket.html?cartridge=program-monitor`.
Type each line and press Enter:

```text
10 SET 7
20 ADD 5
30 PRINT
RUN
```

The guest prints **12**. Enter `20 ADD 6` and `RUN` again to print
**13**. Enter `NEXT` to execute one occupied line and inspect the
accumulator and next line. The original Machine Monitor, Rail Crate,
Passage Relay and their replay artifacts remain available through
the same selector. The framebuffer ABI, host publication and frozen
VisuaML graph are unchanged.

## Guest language

The [guest interpreter](../../guest/monitor-language.rs) has four
line slots at 10, 20, 30 and 40. A valid numbered line replaces that
slot and resets the interpreter to accumulator zero and line 10.
Instructions in a numbered line are uppercase.

| Instruction | Behavior |
| --- | --- |
| `SET n` | Set the accumulator to `n` |
| `ADD n` | Add `n` modulo 1000 |
| `PRINT` | Emit the current unsigned accumulator in decimal |
| `HALT` | Stop execution before the next line |
| `RUN` | Start from line 10 and accumulator zero, then run to HALT or the end |
| `NEXT` | Execute one occupied line from the current interpreter state |

Literals contain one to three decimal digits and represent 0–999.
Execution advances in line order and skips empty slots. One invocation
executes at most four populated slots. A RUN with several PRINT
instructions writes their values in order in one response line.
The host Reset boots a fresh instance. The Monitor's typed RESET
continues to reset the passage world. The browser displays guest
pixels and exposes the private accumulator and next-line fields as
host readouts. The typed language runs inside the Rust guest, and
the frozen VisuaML graph executes the guest's WASM.

## Redraw comparison

A renderer-only build retains the old command behavior. It caches
the two character codes that share each aligned framebuffer word.
Within the affected text rows it redraws only changed pairs. Each
pair spans eight scanlines. This catches both cursor cells during
movement, clears erased pixels and handles scroll with the same
cache comparisons. The renderer keeps 32-bit guest memory operations
and publishes the same full 5,760-byte frame.

The old and dirty renderer produced identical transcripts and every
framebuffer byte on all 50 frames of the same key log. Chromium
independently replayed both renderer binaries. Within each offline
run the compiled VisuaML graph and Wasmtime matched full memory at
both phases on every frame. The two binaries have different private
renderer state. [Raw draw timings and audit counts](renderer.json)

| Event | Samples per renderer | Baseline → dirty p95 drawing | Actual framebuffer stores at audited witness |
| --- | ---: | ---: | ---: |
| Printable key | 37 | 128.88 → **47.27 ms** | 80 → 16 |
| Backspace | 2 | 121.64 → **41.97 ms** | 80 → 16 |
| Enter | 8 | 358.97 → **154.03 ms** | 240 → 72 |
| Scroll | 2 | 2,065.83 → **1,273.71 ms** | 1,440 → 328 |

Timings are offline compiled VisuaML draw phases on sequential
replays, with comparison and hashing outside each phase timer.
Small-sample p95 is the observed maximum for these categories.
Exact source tracing counts the stores at one witness of each
type, including stores that would have left the byte unchanged.
Printable witness source steps dropped from 5,917 to 1,985.
Enter and scroll still exceed the 50 ms reference.

## Program replay

The two logs differ at just the argument of ADD. Each loads SET,
ADD, PRINT and HALT, steps twice, then runs the program. The first
NEXT yields accumulator 7 and next line 20. At the second NEXT,
the guest WASM store at image **PC 3323** writes accumulator byte
address **348** with word 12 or 13. Both subsequently print the
corresponding number as guest-drawn pixels.

- [ADD 5 key log](../../fixtures/monitor-program-5.json) and
  [exact proof](../../fixtures/monitor-program-5.proof.json).
- [ADD 6 key log](../../fixtures/monitor-program-6.json) and
  [exact proof](../../fixtures/monitor-program-6.proof.json).

The two programs account for 100 frames and 200 phase boundaries.
Chromium checked every frame, transcript and full-page digest.
VisuaML and Wasmtime matched every byte of the full pages with
packed fallback blocked. An additional
[language regression](language-checks.json) verifies modulo wrap,
rejection of an out-of-range literal and HALT skipping a later PRINT.

The build archives the raw Rust compiler output. Its Wasmtime behavior
also matches the shipped WASM at every program phase boundary. In
addition to the previous Pocket lowering, build-time legalization
biases constant unsigned comparisons into signed comparisons and
spills the value of result-only loops into a private i32 local.
There is no executor change.

## Reproduce

From `visuaml-client/voxel-replay` with the backend dependencies,
CPU PyTorch, Wasmtime, and rustc 1.100.0 with wasm32-unknown-unknown
if rebuilding the Rust binaries:

```sh
export PYTHONPATH=.:../backend OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
bash tools/build_pocket_dirty.sh
bash tools/build_pocket_program.sh
python tools/replay_monitor_program.py
python tools/replay_monitor_program.py --checks
python tools/replay_monitor_program.py --profile
npm test
npm run build
```

Use Playwright Core and Chromium to run
`node tools/browser_monitor_program.mjs /tmp/pocket-program-captures`
against the running dev server. It verifies the replays and captures
the six browser frames used in the GIF.
