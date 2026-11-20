import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { digest } from './pocket.mjs';
import { startResumeComparison } from './pocket-tape-resume.mjs';
import { extendUniversal } from './pocket-universal.mjs';

const read = file => readFile(new URL(file, import.meta.url));
const json = async file => JSON.parse(await read(file));
const wasm = await read('./evidence/pocket-tape-compiler/erase.linked.wasm');
const table = await json('./evidence/pocket-universal/table.json');
const evidence = await json('./evidence/pocket-tape-resume/replay.json');
const fixture = await json('./fixtures/pocket-tape-resume.json');

test('pre-attempt checkpoint resumes to the wide run byte for byte', async () => {
  const comparison = await startResumeComparison(wasm, table, fixture);
  assert.equal(await digest(wasm), evidence.wasm_sha256);
  assert.equal(await digest(await read('./evidence/pocket-universal/table.json')), evidence.table_sha256);
  for (const [index, expected] of evidence.transitions.entries()) {
    let pair = comparison.advance();
    if (pair.stopped) {
      assert.equal(index, evidence.checkpoint.transition);
      assert.equal(pair.boundary, evidence.checkpoint.simulated_steps);
      const checkpoint = comparison.checkpoint();
      assert.deepEqual(new Uint8Array(checkpoint), new Uint8Array(await read('./evidence/pocket-tape-resume/checkpoint.bin')));
      const [stopped] = comparison.pages();
      assert.equal(await digest(stopped), evidence.checkpoint.stopped_sha256);
      await assert.rejects(extendUniversal(wasm, table, stopped, 3), /Cannot extend/);
      await assert.rejects(extendUniversal(wasm, table, checkpoint, 65536), /Cannot extend/);
      await comparison.resume();
      const [left, right] = comparison.pages();
      assert.equal(await digest(left), evidence.checkpoint.extended_sha256);
      assert.deepEqual(new Uint8Array(left), new Uint8Array(right));
      pair = comparison.advance();
    }
    assert.equal(pair.short.head, expected.head);
    const [left, right] = await Promise.all(comparison.pages().map(digest));
    assert.equal(left, expected.short_sha256);
    assert.equal(right, expected.wide_sha256);
  }
  const { short, wide, boundary } = comparison.current();
  assert.equal(short.phase, 'halt');
  assert.equal(boundary, evidence.boundaries.at(-1).simulated);
  assert.deepEqual(short.decoded, wide.decoded);
});
