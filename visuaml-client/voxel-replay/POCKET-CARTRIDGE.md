# Pocket cartridge ABI v1

Pocket loads a synchronous WebAssembly cartridge with no imports. Export
one memory named `memory` whose initial and maximum sizes are both one
64 KiB page. Export `run(i32) -> i32`. A new instance starts with all bytes
zero. For replay in the frozen VisuaML graph, keep to its admitted unary
i32 subset. Use acyclic direct helper calls with callees before callers,
empty control block signatures, at most four additional locals per function,
and i32 loads and stores. The
[starter](guest/pocket_starter.rs) and [build script](tools/build_pocket_starter.sh)
show the complete entry point and build flags.

| Bytes | Contract |
| --- | --- |
| 256–259 | Signed i32 input code in little-endian order |
| 260–263 | Signed i32 phase in little-endian order |
| 4096–9855 | Packed 160 × 144 framebuffer |
| Other bytes | Cartridge-private memory |

The input is a command word with one code per tick. All 32 bits encode a
signed integer. Zero means idle and minus one is available for an in-guest
reset. Positive codes belong to the cartridge manifest. Pocket's **Reset**
host control boots a new instance. Logs start with one zero boot tick.
The host writes the command and phase zero, calls `run(256)`, then writes
phase one and calls it again with the same command. Phase zero updates the
world and phase one draws. Both calls return the same i32 result for the
tick. Drawing may update its private bookkeeping. A frame becomes visible
after phase one returns.

The framebuffer has 40 bytes per scanline. Its four pixel lanes run
left to right in bits 7–6, 5–4, 3–2, 1–0. The palette is
`#dcebb3`, `#93ab70`, `#41624e`, `#152d28` for indices zero through
three. An untouched zero byte displays four shade-zero pixels. The host
copies 5,760 bytes to publish a frame, converts indices to RGBA and
submits that image to Canvas. Private world fields can be displayed by the
inspector separately.

## Build and load a cartridge

A checked-in manifest in [cartridges](cartridges) identifies its WASM
filename under `guest` and the SHA-256 of those bytes. See the
[starter manifest](cartridges/starter.json). Each control supplies a label,
a keyboard key, and an integer code. Optional `keyboard: "ascii"` maps
printable ASCII bytes 32–126 plus Backspace (8) and Enter (13) to the
input word. The terminal owns its line editing and parsing.

The manifest registry populates the selector. Readout offsets are relative
to 256 and read as signed i32. Readouts, notice text, frame labels and
saved proof references are optional presentation metadata. The browser
does not perform a cartridge's simulation or drawing. The **Host · Load
local manifest and WASM** control accepts the two local files together.
Local instances expose their buttons and keyboard, plus the common replay
and export controls. Curated proofs and cartridge-specific readouts come
from the checked-in registry.

```sh
cd visuaml-client/voxel-replay
export PYTHONPATH=.:../backend
bash tools/build_pocket_starter.sh
npm ci
npm run dev -- --port 4180
```

Open `http://localhost:4180/pocket.html`. The route
`?cartridge=monitor` opens the terminal directly. Use `?cartridge=rail-crate`
for the push puzzle. Existing Passage Relay stays the default.

The starter build uses Rust 1.100.0 and the wasm32-unknown-unknown target.
The build can rewrite a small set of equivalent LLVM instruction forms
into the frozen subset. It expands equality and opposite signed comparison
forms, replaces constant left shifts with wrapping multiplication and
topologically orders unary helpers, rewriting direct call indices. The
Rust compiler originals are retained next to the evidence. Each shipped
cartridge binary is the same byte sequence used by Chromium, Wasmtime and
VisuaML. The replay verifies that the compiler original and shipped
binary agree in Wasmtime at both phase boundaries for the supplied logs.

## Verify a log

Browser playback runs browser WASM. The inspector displays a saved
VisuaML proof match for supported recorded frames. For offline execution
with backend dependencies, CPU PyTorch and Wasmtime installed:

```sh
export PYTHONPATH=.:../backend OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python tools/replay_pocket.py
python tools/replay_cartridges.py
python tools/replay_cartridges.py --session /path/to/pocket-session.json
python tools/replay_cartridges.py --session /path/to/local-session.json \
  --manifest /path/to/manifest.json --wasm /path/to/cartridge.wasm
```

Use `replay_pocket.py --session` for a shipped Passage Relay log. The host
owns cartridge loading, log export, memory checkpoints and selecting
saved replay trace points. The Machine Monitor has no guest loader or
filesystem.

For the typed language and renderer comparison, see the
[Programmable Pocket Monitor](evidence/monitor-program/README.md).

The [Compiler Monitor](evidence/pocket-compile/README.md) adds a manifest-gated
inspector for modules and source maps emitted by a guest. Open
`?cartridge=compile-monitor` to try it.

The [Rust Call Monitor](evidence/pocket-rust-call/README.md) adds a unary
cross-language call and a saved source-step trace.

The [Tape Compiler](evidence/pocket-tape-compiler/README.md) links a program
typed into Pocket to the Rust tape interpreter. Open
`?cartridge=tape-compiler` to compile rules and step the resulting tape.

[Universal Tape](evidence/pocket-universal/README.md) runs a fixed table that
interprets machines encoded on its tape. Open `pocket-universal.html` to
step either execution level.

The [tape resume experiment](evidence/pocket-tape-resume/README.md)
restores a pre-attempt checkpoint into a longer logical tape window and
compares it with a run that started with the wider window.
