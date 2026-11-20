import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { linkPassable } from './pocket-link.mjs';
import { runEmitted } from './pocket-module.mjs';

const read = file => readFile(new URL(`./evidence/pocket-rust-call/${file}`, import.meta.url));
for (const cell of [2, 3]) {
  test(`Pocket calls separately compiled Rust for cell ${cell}`, async () => {
    const caller = await read(`cell-${cell}.emitted.wasm`);
    const helper = await read('passable.raw.wasm');
    const meta = JSON.parse(await read(`cell-${cell}.map.json`));
    const { wasm, helperBody, helperStart } = linkPassable(caller, helper);
    assert.deepEqual(wasm, new Uint8Array(await read(`cell-${cell}.linked.wasm`)));
    assert.deepEqual(wasm.slice(helperStart, helperStart + helperBody.length), new Uint8Array(helperBody));
    const result = await runEmitted(wasm);
    assert.equal(result.pageHash, meta.page_sha256);
    assert.deepEqual(result.printed, [cell - 2]);
    for (const line of meta.source_map) {
      assert.deepEqual(wasm.slice(line.linked_start, line.linked_end),
        new Uint8Array(caller.slice(line.emitted_start, line.emitted_end)));
    }
  });
}

test('link rejects an i64 parameter and a recursive call before creating an artifact', async () => {
  const caller = await read('cell-2.emitted.wasm');
  const [mismatch, helper] = await Promise.all([read("mismatch.wasm"), read("passable.wasm")]);
  assert.throws(() => linkPassable(caller, mismatch), /Expected unary i32/);
  const meta = JSON.parse(await read('cell-2.map.json'));
  const recursive = new Uint8Array(caller);
  recursive[meta.source_map[1].emitted_start + 3] = 1;
  assert(WebAssembly.validate(recursive));
  assert.throws(() => linkPassable(recursive, helper), /Unexpected direct call target/);
});
