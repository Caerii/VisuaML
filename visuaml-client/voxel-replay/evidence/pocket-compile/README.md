# Pocket Compiler Monitor

The Rust Monitor interprets typed programs with `RUN` and `NEXT`. With `COMPILE`, it reads the same stored source records and emits a standard WebAssembly module and a byte-range source map. The host reads the guest output, validates the module, and invokes it as a separate program. The offline replay uses the same compiled frozen graph for both executions with packed fallback blocked.

## Play

From `visuaml-client/voxel-replay`:

```sh
npm ci
npm run dev -- --port 4180
```

Open [Compiler Monitor](http://localhost:4180/pocket.html?cartridge=compile-monitor). Enter each line separately.

```text
10 SET 7
20 ADD 6
30 PRINT
40 HALT
COMPILE
```

Select `20 ADD 6` in the inspector and click **Run emitted WASM**. The browser shows the instructions at module bytes 62–88 and the accumulator changing from 7 to 13. Comparing the two archived modules
finds one changed byte at zero-based offset 65, the immediate for `i32.const`. **Download WASM** saves exactly the bytes the Monitor produced. The browser labels its own execution and separately reports matches against saved VisuaML evidence by module and output-page hashes.

## Compiler interface

All offsets are absolute byte offsets in the cartridge's 65,536-byte memory. Multi-byte fields are little-endian.

| Offset | Contents |
| --- | --- |
| 400 | Emitted module size as i32. Zero means there is no current module. |
| 404 | Number of entries in the current source map. Read only when module size is nonzero. |
| 24000 | Triples of i32 words with line number, inclusive module byte start, and exclusive module byte end. |
| 24576 | Exact binary beginning with the WASM magic and version. |

The input is four line slots (10, 20, 30, 40). `SET` and `ADD` take literals 0–999. Arithmetic stays in 0–999 and ADD wraps modulo 1000. Compilation visits slots in ascending order. `PRINT` preserves output order and `HALT` omits later slots. A source edit invalidates the previously emitted module. If the latest numbered edit failed to parse, `COMPILE` reports an error until a valid numbered line is stored.

The emitted module has no imports. It exports one fixed memory page and `run(i32) -> i32`. The argument is ignored and the accumulator starts at zero. The return is the count of prints (0–4). The outputs start at memory byte 16, one i32 per print. Bytes 32, 36, 40, and 44 hold the accumulator at the end of each executed source line. These inspection stores belong to the emitted program and are included in its source-step count. The host decodes source map offsets and disassembles the matching byte ranges.

The raw compiler cartridge is built by Rust. The repo's existing cartridge legalizer normalizes LLVM opcode forms for the frozen profile. The Monitor's *emitted modules* are loaded directly from guest memory, without running that legalizer on them. The Monitor shares the font initialization entry with its decimal output routine to fit the frozen profile's eight-function limit.

## Verify

Run from `visuaml-client/voxel-replay` using the research Python environment with Torch and Wasmtime. Rebuilding requires Rust's `wasm32-unknown-unknown` target.

```sh
export PYTHONPATH=.:../backend
bash tools/build_pocket_compile.sh
python tools/replay_pocket_compile.py
python tools/replay_pocket_compile.py --session /path/to/pocket-session.json
npm test
```

`replay_pocket_compile.py` checks exact pages after the simulation and draw phases, plus the final output pages of the emitted modules. It also checks the raw compiler cartridge against the legalized version in Wasmtime. The saved input fixtures and proofs are in [fixtures](../../fixtures). The tested keyboard histories cover the existing 12 and 13 programs, modulo overflow, malformed source after a previously successful compile, wrapped command-hash lookalikes, and an out-of-order four-line program that generates the longest instruction sequences. The browser replay independently verifies the saved phase/page/frame hashes and transcripts before running each emitted module.

```sh
PLAYWRIGHT_CORE=/path/to/playwright-core/index.mjs \
  node tools/browser_pocket_compile.mjs /tmp/pocket-compile-captures
```

| Case | Verified key ticks | Printed values | Emitted source steps |
| --- | ---: | --- | ---: |
| ADD 5 | 58 | 12 | 30 |
| ADD 6 | 58 | 13 | 30 |
| 999 + 5 | 42 | 4 | 27 |
| Malformed | 63 | Compiler rejected the edit | — |
| Four ADD 999 lines | 53 | No prints | 58 |

[Typed-program GIF](pocket-compile.gif) · [ADD 6 WASM](add-6.wasm) · [ADD 6 source map](add-6.map.json) · [Compiler phase timings](timings.json)

## Separate scrolling comparison

The dirty renderer already caches glyph pairs, but scrolling changed the correspondence between the cache and nearly every visible row. The new build moves the existing 2-bit raster and the cache upward in ascending address order, clears the newly exposed bottom, and repairs only the rows at and below the old cursor's new position. Ascending addresses are safe because each source word is above its destination. The replay covers one-, two-, and three-row shifts, a multi-line HELP response, and reset. All 128 published frames and transcripts matched in each of three paired runs. Within each build, Wasmtime and VisuaML matched all page bytes at both phase boundaries.

These measurements use the compiled frozen runner, in milliseconds. The scroll class excludes the Enter that compiles a program. Event totals combine input handling and drawing. Compiler simulation samples are in the separate phase timings.

| Event | Samples per build | Dirty draw p95 | Shift draw p95 | Dirty total p95 | Shift total p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Printable | 300 | 43.0 | 44.1 | 45.8 | 46.5 |
| Backspace | 6 | 41.3 | 43.0 | 43.6 | 45.7 |
| Enter without scroll | 45 | 251.4 | 251.7 | 266.5 | 266.3 |
| Scroll | 24 | 1288.2 | 657.0 | 1927.8 | 1314.9 |

The scroll draw p95 fell 49.0%. Draw source steps at p95 fell from 63,443 to 31,616. A scroll also moves the text model during simulation, which contributes to the larger total. The current scroll path remains above the 50 ms reference budget.

```sh
POCKET_SCROLL=0 COMPILE_RAW=/tmp/pocket-compiler-baseline.wasm \
  COMPILE_WASM=evidence/pocket-compile/compiler-baseline.wasm \
  bash tools/build_pocket_compile.sh
python tools/replay_pocket_scroll.py
```

[Raw paired scroll timings](scroll-timings.json) · [Chromium replay results](chromium.json)
