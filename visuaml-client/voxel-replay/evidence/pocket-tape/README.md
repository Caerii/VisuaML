# Pocket tape machine

Open [Tape machine](http://localhost:4180/pocket-tape.html) after running `npm ci` and `npm run dev -- --port 4180` from `visuaml-client/voxel-replay`. Pick an example or enter your own rules and initial tape. Each rule uses `state read next write L/R/S`. Step attempts one transition. Run repeats steps at a readable pace.

The browser parses the editor into tables and displays the tape. The Rust guest reads the current symbol, looks up and validates the transition, writes the new symbol, and updates the head and state. The screen labels live browser WASM. Example steps can also show a full-page hash match against saved VisuaML results.

## Memory and stopping

The guest exports `run(i32) -> i32` and one fixed 64 KiB page. The caller passes byte 256 to `run`. Eight little-endian i32 header words hold state count, symbol count, tape length, current state, head index, status, completed-transition count, and next-head scratch. Then come dense triples `(next state, write symbol, movement)`, ordered by state and read symbol, and one i32 word per tape cell. Initial state is 0. A move is −1, 0, or 1. The host encodes missing rules with next state −1. There are up to 256 states and 256 symbols, subject to one shared limit of 16,300 words for the table and tape.

The result is 0 after a completed transition, 1 on a missing rule, 2 on a move beyond the tape window, or 3 for invalid live machine data. A boundary failure leaves the tape, head, state, and completed-transition counter at their previous values. Fuel belongs to the frozen WASM executor and is separate from guest status.

## What the example establishes

The same compiled guest and frozen graph ran three machines supplied as data. For every attempted transition, the replay compares the full page across the compiled graph, single-step graph, and Wasmtime. An independent transition model checks state, head, tape, and status. The single-step audit records WASM PCs and their tape reads and writes. Packed fallback is blocked.

| Machine | Completed transitions | Final tape | Stop |
| --- | ---: | --- | --- |
| Flip bits | 3 | `0 2 1 2 0` | No matching rule |
| Grow unary | 3 | `0 1 1 1 0` | No matching rule |
| Walk off the tape | 1 | `0 1 0 0` | Boundary |

For the mathematical construction, encode any deterministic single-tape machine's finite state and symbol sets as integer indices. Put its transition function in the dense table and choose a tape window that contains the input, the starting head, and at least (n) cells on both sides of the starting head. In (n) moves the head cannot leave that window. Assume the table and window fit the model's address space and execution allowance. At the start the encoding represents the same configuration. At each step, the matching state and symbol select the same table entry, so both machines write the same symbol, move in the same direction, and enter the same state. The result follows by induction up to (n) transitions or an earlier halt.

With extensible storage, representable indices, and adequate execution allowance, that construction simulates arbitrary finite prefixes. The actual WASM guest and frozen artifact have fixed bounds. The replays establish bounded instances of the construction. The unbounded result applies to the extensible model.

## Reproduce

```sh
export PYTHONPATH=.:../backend OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
bash tools/build_pocket_tape.sh
python tools/replay_pocket_tape.py
npm test
PLAYWRIGHT_CORE=/path/to/playwright-core/index.mjs node tools/browser_pocket_tape.mjs
```

[GIF](flip-bits.gif) · [Encoded machines](../../fixtures/pocket-tape.json) · [Exact tape audit](replay.json) · [Rust guest](../../guest/pocket_tape.rs)
