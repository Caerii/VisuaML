import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { createPocket, digest } from './pocket.mjs';
import { readTapeArtifact, startTapeArtifact } from './pocket-tape-compiler.mjs';

const read = path => readFile(new URL(path, import.meta.url));
const proof = JSON.parse(await read('./evidence/pocket-tape-compiler/replay.json'));
const monitor = await read('./guest/pocket_tape_compiler.wasm');
const helper = await read('./guest/pocket_tape.wasm');

for (const [name, record] of Object.entries(proof.cases)) {
  test(`Tape compiler reproduces ${name} from key events`, async () => {
    const pocket = await createPocket(monitor, proof.cartridge_sha256, { keyboard: 'ascii' });
    for (const [i, key] of record.buttons.entries()) {
      const { frame } = pocket.step(key);
      assert.equal(await digest(frame), record.framebuffer_hashes[i]);
      assert.equal(await digest(pocket.checkpoint()), record.monitor_pages[i]);
    }
    const output = readTapeArtifact(pocket.checkpoint());
    if (!record.ticks) {
      assert.equal(output, null);
      return;
    }
    assert.equal(await digest(output.table), record.table_sha256);
    assert.equal(await digest(output.wasm), record.module_sha256);
    assert.deepEqual(output.lines, record.rule_lines);
    const machine = await startTapeArtifact(output, helper);
    assert.equal(await digest(machine.wasm), record.linked_sha256);
    assert.equal(machine.callByte, record.call_byte);
    for (const expected of record.ticks) {
      const actual = machine.step();
      for (const key of ['tape', 'head', 'line', 'status']) assert.deepEqual(actual[key], expected[key]);
      assert.equal(await digest(machine.checkpoint()), expected.page_sha256);
    }
    if (record.malformed_table) {
      new DataView(output.table.buffer).setInt32(12 * 4, 3, true);
      const corrupt = await startTapeArtifact(output, helper);
      assert.equal(corrupt.step().status, 3);
      assert.equal(await digest(corrupt.checkpoint()), record.malformed_table.page_sha256);
    }
  });
}
