# Pocket calls Rust

The Monitor emits an import of `pocket.passable(i32) -> i32` and a standard WASM `call 0`. The Pocket accumulator is the argument, and the returned 0 or 1 becomes the new accumulator. SET and ADD retain the language's 0–999 values and modulo-1000 addition. RUN and NEXT use the equivalent predicate inside the interpreter. A linked compiled call executes the separate Rust implementation, including its memory effect.

The helper models a blocked cell at 2 and the open right side of a corridor at coordinates greater than 2. It returns 0 and leaves memory unchanged on the blocked path. On the open path it returns 1 and writes the input to byte 64 as a little-endian i32. PRINT writes its result at byte 16.

The host linker validates the signature, the exact import, fixed memory and exports, and direct call targets. It installs the compiler-produced Rust body as function 0 and keeps Pocket's unchanged caller body as function 1. Both bodies share the final memory page. The linker removes the import and adjusts only module structure and reported source offsets. The separately compiled artifact and its legalized copy are byte-identical.

| Input | Printed | Word at byte 64 | Linked source steps |
| --- | ---: | ---: | ---: |
| Cell 2 | 0 | 0 | 28 |
| Cell 3 | 1 | 3 | 34 |

The unchanged compiled frozen graph matched Wasmtime on the return and full memory for both linked modules, with packed fallback blocked. Its single-step reference records argument entry, Rust instructions, helper return, resumed caller PC, and the print store. The host rejects the archived i64-parameter helper before producing a linked binary. The JS regression also rejects a caller that targets itself.

## Play and replay

From `visuaml-client/voxel-replay` run `npm ci` and `npm run dev -- --port 4180`. Open [Rust Call Monitor](http://localhost:4180/pocket.html?cartridge=rust-call-monitor). Enter these lines individually.

```text
10 SET 3
20 CALL
30 PRINT
RUN
COMPILE
```

RUN prints the reference interpreter's answer on the guest terminal. Use **Run emitted WASM** for the browser's linked execution. Select **20 CALL**, then **Step verified call** to walk the saved single-step VisuaML trace. The matching module and memory hashes identify the saved run.

```sh
export PYTHONPATH=.:../backend OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
bash tools/build_pocket_rust_call.sh
python tools/replay_pocket_rust_call.py
npm test
```

[GIF](pocket-rust-call.gif) · [Cell 2 keys](../../fixtures/pocket-rust-cell-2.json) · [Cell 3 keys](../../fixtures/pocket-rust-cell-3.json) · [Emitted WASM](cell-3.emitted.wasm) · [Rust WASM](passable.raw.wasm) · [Linked WASM](cell-3.linked.wasm) · [Source map and trace](cell-3.map.json)
