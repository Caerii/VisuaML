import assert from 'node:assert/strict';
import { writeFile } from 'node:fs/promises';
import { performance } from 'node:perf_hooks';
import { denseTick, SparseWorld } from '../cellular-automata.mjs';

function initialGrid(size, seeds) {
  const walls = new Uint8Array(size*size);
  const initial = new Uint8Array(walls.length);
  // Seeds expand into a sealed, otherwise stationary world.
  for (let i = 0; i < seeds; i++) {
    const x = 8+(i % 8)*14;
    const y = 8+Math.floor(i/8)*14;
    initial[y*size+x] = 1;
  }
  return { walls, initial };
}

function trial(size, seeds, sparse) {
  const { walls, initial } = initialGrid(size, seeds);
  const world = sparse ? new SparseWorld(walls, initial, size, size) : null;
  let signal = initial, evaluated = 0;
  const start = performance.now();
  for (let tick = 0; tick < 12; tick++) {
    if (sparse) evaluated += world.tick().evaluated;
    else {
      signal = denseTick(walls, signal, size, size);
      evaluated += size*size;
    }
  }
  return { signal: world?.signal ?? signal, evaluated, ms: performance.now()-start };
}
function verifyEveryTick(size, seeds) {
  const {walls, initial} = initialGrid(size, seeds);
  const sources = [];
  initial.forEach((value, i) => { if (value) sources.push([i % size, Math.floor(i / size)]); });
  const distance = initial.map((_, i) => {
    const x = i % size, y = Math.floor(i / size);
    let closest = 255;
    for (const [a,b] of sources) closest = Math.min(closest, Math.abs(a-x)+Math.abs(b-y));
    return closest;
  });
  const sparse = new SparseWorld(walls, initial, size, size);
  let dense = initial;
  for (let tick = 1; tick <= 12; tick++) {
    const expected = distance.map(d => Number(d <= tick));
    dense = denseTick(walls,dense,size,size);
    sparse.tick();
    assert.deepEqual(dense,expected);
    assert.deepEqual(sparse.signal,expected);
  }
}

const rows = [];
for (const [size,seeds] of [[32,1],[128,1],[384,1],[128,16],[128,64]]) {
  verifyEveryTick(size,seeds);
  trial(size,seeds,false);
  trial(size,seeds,true);
  const dense = [], sparse = [];
  for (let i = 0; i < 7; i++) {
    const first = trial(size,seeds,i % 2 === 0);
    const second = trial(size,seeds,i % 2 !== 0);
    const [a,b] = i % 2 === 0 ? [second,first] : [first,second];
    assert.deepEqual(a.signal, b.signal);
    dense.push({ ms:a.ms, evaluated:a.evaluated });
    sparse.push({ ms:b.ms, evaluated:b.evaluated });
  }
  rows.push({ size,seeds,ticks:12,dense,sparse });
}
await writeFile(new URL('../evidence/pocket-cellular-automata/timings.json', import.meta.url),
                JSON.stringify(rows,null,2)+'\n');
