import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { digest } from './pocket.mjs';
import { inspectTape, loadTape, TAPE_BASE } from './pocket-tape.mjs';

const read = path => readFile(new URL(path, import.meta.url));
const raw = await read('./guest/pocket_tape.wasm');
const cases = JSON.parse(await read('./fixtures/pocket-tape.json'));
const proof = JSON.parse(await read('./evidence/pocket-tape/replay.json'));

for (const example of cases) {
  test(`Tape machine reproduces ${example.name} on the same guest`, async () => {
    assert.equal(await digest(raw), proof.wasm_sha256);
    const { instance: { exports: { memory, run } } } = await WebAssembly.instantiate(raw);
    const index = loadTape(memory, example);
    for (const expected of proof.runs[example.name]) {
      assert.equal(run(TAPE_BASE), expected.status);
      assert.deepEqual(inspectTape(memory, index), { state: expected.state, head: expected.head,
        count: expected.transitions, status: expected.status, tape: expected.tape });
      assert.equal(await digest(memory.buffer), expected.page_sha256);
    }
  });
}

test('larger tape preserves a prefix and can extend it beyond the small tape boundary', async () => {
  const { instance: { exports: { memory, run } } } = await WebAssembly.instantiate(raw);
  const config = { states: 1, symbols: 2, head: 2, rules: '0 0 0 1 R' };
  const index = loadTape(memory, { ...config, tape: [0, 0, 0, 0] });
  assert.equal(run(TAPE_BASE), 0);
  const prefix = inspectTape(memory, index);
  assert.equal(run(TAPE_BASE), 2);
  assert.equal(inspectTape(memory, index).tape[3], 0);
  loadTape(memory, { ...config, tape: [0, 0, 0, 0, 0, 0] });
  assert.equal(run(TAPE_BASE), 0);
  const larger = inspectTape(memory, index);
  assert.deepEqual({ ...larger, tape: larger.tape.slice(0, 4) }, prefix);
  assert.equal(run(TAPE_BASE), 0);
  assert.equal(inspectTape(memory, index).tape[3], 1);
  assert.throws(() => loadTape(memory, { ...config, tape: [0, 0, 0], rules: config.rules + '\n0 0 0 0 S' }),
    /Duplicate rule/);
});
