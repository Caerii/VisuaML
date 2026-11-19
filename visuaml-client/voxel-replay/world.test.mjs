import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { createWorld, GUESTS, matchesProof } from './world.mjs';

const fixture = async (name) => JSON.parse(await readFile(new URL(`./fixtures/${name}.json`, import.meta.url)));

for (const [mode, expected] of Object.entries({
  single: [{ positions: [1], materials: [0], conflict: false },
           { positions: [1], materials: [7], conflict: false }],
  shared: [{ positions: [1, 3], materials: [0, 0], conflict: false },
           { positions: [1, 3], materials: [0, 0], conflict: true }],
})) {
  test(`${mode} WASM replay matches independent outcomes and the saved NumPy page hashes`, async () => {
    const session = await fixture(mode);
    const proof = await fixture(`${mode}.proof`);
    const world = await createWorld(await readFile(GUESTS[mode]), session);
    const play = async () => {
      const rows = [];
      for (const [index, input] of session.inputs.entries()) {
        const row = world.step(input);
        const { positions, materials, conflict } = row;
        assert.deepEqual({ positions, materials, conflict }, expected[index]);
        rows.push({ input, hash: await row.pageHash });
        assert.ok(matchesProof(session, rows, proof));
      }
      return rows;
    };
    const original = await play();
    world.reset();
    assert.deepEqual(await play(), original);
    assert.equal(matchesProof(session, [{ ...original[0], hash: 'bad hash' }], proof), false);
    assert.equal(matchesProof(session, [{ ...original[0], input: [0] }], proof), false);
  });
}

test('invalid boundary input preserves the starting page, and an occupied cell blocks a swap', async () => {
  const session = await fixture('shared');
  session.initial.positions = [1, 2];
  const world = await createWorld(await readFile(GUESTS.shared), session);
  assert.throws(() => world.step([1, 2]), /leave the board/);
  const outcome = world.step([1, -1]);
  assert.deepEqual(outcome.positions, [1, 2]);
  assert.equal(outcome.conflict, false);
  await outcome.pageHash;
});


for (const [name, moves, material] of [
  ['terrain', [0, 1], 0],
  ['terrain-omitted', [0, 2], 7],
]) {
  test(`${name} reproduces the causal history on the original guest`, async () => {
    const session = await fixture(name);
    const proof = await fixture(`${name}.proof`);
    const world = await createWorld(await readFile(GUESTS.terrain), session);
    const first = world.step(session.inputs[0]);
    assert.deepEqual(first.positions, [0, 2]);
    assert.equal(first.terrain[1], material);
    const next = world.step(session.inputs[1]);
    assert.deepEqual(next.positions, moves);
    assert.equal(next.materials[1], material);
    assert.ok(matchesProof(session, [
      { input: first.input, hash: await first.pageHash },
      { input: next.input, hash: await next.pageHash },
    ], proof));
  });
}
