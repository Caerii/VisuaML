import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { denseTick, passage, SparseWorld } from './cellular-automata.mjs';

const proof = JSON.parse(await readFile(new URL('./evidence/pocket-cellular-automata/replay.json', import.meta.url)));

test('both Pocket histories match each archived scalar/tensor tick', () => {
  const {width, height, walls, signal, gate} = passage();
  assert.deepEqual([...walls], proof.scene.walls);
  assert.deepEqual([...signal], proof.scene.signal);
  for (const name of ['dig', 'omitted']) {
    const sparse = new SparseWorld(walls, signal, width, height);
    let dense = signal.slice();
    if (name === 'dig') sparse.open(gate);
    for (let tick = 1; tick <= 16; tick++) {
      dense = denseTick(sparse.walls, dense, width, height);
      sparse.tick();
      assert.deepEqual([...sparse.signal], proof.frames[name][tick]);
      assert.deepEqual(dense, sparse.signal);
    }
    assert.equal(sparse.tick().evaluated, 0);
  }
});

test('an edit after quiescence reactivates the frontier and never wraps a row', () => {
  const {width, height, walls, signal, gate} = passage();
  const sparse = new SparseWorld(walls, signal, width, height);
  for (let tick = 0; tick < 16; tick++) sparse.tick();
  assert.equal(sparse.tick().evaluated, 0);
  sparse.open(gate);
  assert.ok(sparse.frontier.size);
  const dense = denseTick(sparse.walls, sparse.signal, width, height);
  sparse.tick();
  assert.deepEqual(sparse.signal, dense);
  assert.equal(sparse.signal[gate], 1);
  assert.deepEqual([...denseTick(new Uint8Array(6), Uint8Array.of(0,0,1,0,0,0), 3, 2)], [0,1,1,0,0,1]);
});
