import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';

const raw = await readFile(new URL('./voxel_tick.wasm', import.meta.url));
const evidence = JSON.parse(await readFile(new URL('./guest-evidence.json', import.meta.url), 'utf8'));
const sha256 = (value) => createHash('sha256').update(value).digest('hex');
assert.equal(sha256(raw), evidence.guest_sha256);

const { instance } = await WebAssembly.instantiate(raw);
const { run, memory } = instance.exports;
const page = new Uint8Array(memory.buffer);
const state = new DataView(memory.buffer);
for (const row of evidence.cases) {
  page.fill(0);
  const base = evidence.base;
  state.setInt32(base, row.initial_position_bytes, true);
  state.setInt32(base + 4, row.delta_bytes, true);
  state.setInt32(base + 8, -1, true);
  state.setInt32(base + 16 + 2 * 4, 7, true);
  assert.equal(run(base), row.final_position_bytes, row.case);
  assert.equal(sha256(page), row.page_sha256, row.case);
}
console.log(`${process.version}: ${evidence.cases.length} JavaScript WebAssembly cases matched`);
