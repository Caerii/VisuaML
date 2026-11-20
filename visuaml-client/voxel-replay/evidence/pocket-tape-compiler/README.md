# Compile a tape machine in Pocket

The Pocket terminal accepts a program, emits a WASM caller and table bytes, and links to the existing separately compiled Rust tape interpreter. The browser copies the emitted table verbatim into the linked module's memory and draws the resulting tape. Pocket's Rust compiler guest owns the rule decoding, table construction, and WASM emission. The frozen graph is unchanged.

## Play

From `visuaml-client/voxel-replay` run `npm ci` and `npm run dev -- --port 4180`. Open [Tape Compiler](http://localhost:4180/pocket.html?cartridge=tape-compiler) and enter each line individually.

```text
10 TAPE 112
20 RULE 112
30 RULE 202
40 STEP
COMPILE
```

Click **Link and load**, then **Step into Rust** three times. Edit line 30 to `30 RULE 222`, compile and load again. Both runs advance the head and write the same values for the first two steps. On step three, the Rust helper reads the rule from line 30 and stores to byte 332. The original writes 0 and the edit writes 2. The fourth step halts on the missing rule for symbol 0.

The browser labels live linked WASM. Shipped input histories show saved VisuaML lookup and store evidence when the hashes match. The two histories have different initial table contents, so their full pages differ even during the identical tape prefix.

## Scope and ABI

| Line | Meaning |
| --- | --- |
| `10 TAPE abc` | Three initial symbols followed by a blank, with the head at cell 0 |
| `20 RULE rwd`, `30 RULE rwd` | One rule each. The digits mean read, write, and direction. Symbols are 0–2. Directions are 0 left, 1 stay, 2 right. |
| `40 STEP` | Request one transition per invocation of the linked module |

The model has one state and a four-cell tape. Duplicate rules, out-of-alphabet digits, and incomplete declarations cause the compiler to withhold the module. The emitted caller imports `pocket.tapeStep(i32) -> i32`, passes byte 256 to the helper, and returns its status. The linker binds the import to function 0 and retains the caller's instructions. The guest publishes 84 little-endian table bytes at byte 25000 of Monitor memory and publishes one source-line number per read symbol at byte 25100. The linked page uses the existing [tape ABI](../pocket-tape/README.md).

The helper checks the header and active row at each step. The saved replay injects an invalid write symbol to exercise rejection in the helper independently of the compiler's duplicate-rule check. An out-of-bounds move returns status 2 before writing the tape, and a malformed active row returns status 3. Status 1 is a normal halt.

## Reproduce

Use the repo research environment with Rust's WASM target, Torch, Wasmtime, and Node installed.

```sh
export PYTHONPATH=.:../backend OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
POCKET_TAPE_COMPILER=1 COMPILE_RAW=evidence/pocket-tape-compiler/monitor.raw.wasm \
  COMPILE_WASM=guest/pocket_tape_compiler.wasm bash tools/build_pocket_compile.sh
bash tools/build_pocket_tape.sh
python tools/replay_pocket_tape_compiler.py
npm test
```

The replay compares every Monitor page and linked step against Wasmtime. It also checks the linked steps against the frozen single-step graph, with packed fallback blocked, and verifies the source-attributed write where the two tapes first diverge. The artifacts include key histories, guest-emitted module and table, the final linked binary and lookup/store PCs.

[Demo GIF](tape-compiler.gif) · [Replay and key histories](replay.json) · [Original table](erase.table.bin) · [Edited table](keep.table.bin) · [Emitted WASM](erase.emitted.wasm) · [Rust helper](../../guest/pocket_tape.wasm) · [Linked WASM](erase.linked.wasm)
