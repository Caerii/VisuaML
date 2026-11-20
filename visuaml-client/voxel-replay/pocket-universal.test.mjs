import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { digest } from './pocket.mjs';
import { encode, loadUniversal } from './pocket-universal.mjs';

const read = path => readFile(new URL(path, import.meta.url));
const tableRaw = await read('./evidence/pocket-universal/table.json');
const table = JSON.parse(tableRaw);
const evidence = JSON.parse(await read('./evidence/pocket-universal/replay.json'));
const cases = JSON.parse(await read('./fixtures/pocket-universal.json'));
const wasm = await read('./evidence/pocket-tape-compiler/erase.linked.wasm');

for (const example of cases) {
  test(`Universal tape reproduces ${example.name} with fixed table and linked WASM`, async () => {
    assert.equal(await digest(tableRaw), evidence.table_sha256);
    assert.equal(await digest(wasm), evidence.linked_wasm_sha256);
    const machine = await loadUniversal(wasm, table, example.program, example.input);
    for (const expected of evidence.runs[example.name].transitions) {
      const state = machine.step();
      assert.equal(state.phase, expected.phase);
      assert.equal(state.head, expected.head);
      assert.equal(await digest(machine.page()), expected.page_sha256);
    }
  });
}

async function execute(program, input) {
  const machine = await loadUniversal(wasm, table, program, input);
  for (let i = 0; i < 1500; i++) {
    const state = machine.step();
    if (['halt', 'left_edge'].includes(state.phase)) return state;
    assert.equal(state.status, 0);
  }
  assert.fail('Machine did not stop');
}

test('universality witness supports false condition, backward jump and extending onto blank', async () => {
  const skip = await execute('J 2\nH\nM\nH', '0');
  assert.equal(skip.decoded.pc, 1);
  const loop = await execute('G 2\nH\nG 1\nH', '0');
  assert.equal(loop.decoded.pc, 1);
  const grow = await execute('R\nR\nM\nL\nH', '1');
  assert.deepEqual(grow.decoded.tape.slice(0, 3), [1, 0, 1]);
  assert.equal(grow.decoded.head, 1);
  const edge = await execute('L\nH', '1');
  assert.equal(edge.phase, 'left_edge');
  assert.equal(edge.decoded.head, 0);
  assert.throws(() => encode('J 2\nH', '0', table.symbols), /Invalid operand/);
});
