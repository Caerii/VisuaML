import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { tapeSession } from './pocket-tape-fork.mjs';

const read = path => readFile(new URL(path, import.meta.url));
const json = async path => JSON.parse(await read(path));
const fixture = await json('./fixtures/pocket-tape-fork.json');
const checkpoint = await json('./evidence/pocket-tape-fork/checkpoint.json');
const evidence = await json('./evidence/pocket-tape-fork/replay.json');
const wasm = await read('./evidence/pocket-tape-compiler/erase.linked.wasm');
const session = await tapeSession(wasm, fixture, evidence.graph_sha256);

test('restored copies remain independent and diverge on the pending transition', async () => {
  const clean = await session.importCheckpoint(checkpoint, { left: 1 });
  const edited = await session.importCheckpoint(checkpoint, { left: 1, edit: { coordinate: 0, symbol: 2 } });
  const prior = clean.page();
  assert.equal(edited.step().state, 4);
  assert.deepEqual(clean.page(), prior);
  assert.equal(clean.step().state, 1);
  const fresh = await session.importCheckpoint(checkpoint);
  assert.equal(fresh.current().count, 1);
  assert.deepEqual(fresh.current().tape, [1]);
});

test('overlapping left relocation preserves ordered symbols and logical head', async () => {
  const wide = session.fresh(-1, [2, 1, 2]);
  const exported = await session.exportCheckpoint(wide);
  const relocated = await session.importCheckpoint(exported, { left: 2, right: 1 });
  assert.deepEqual(relocated.current().tape, [0, 0, 2, 1, 2, 0]);
  assert.equal(relocated.current().head, 0);
  assert.deepEqual(wide.current().tape, [2, 1, 2]);
});

test('import rejects incompatible identities and altered page or head', async () => {
  for (const key of ['graph', 'table', 'wasm', 'program']) {
    await assert.rejects(session.importCheckpoint({ ...checkpoint, identities: { ...checkpoint.identities, [key]: '' } }),
      new RegExp(`Checkpoint ${key} mismatch`));
  }
  await assert.rejects(session.importCheckpoint({ ...checkpoint, page: 'B' + checkpoint.page.slice(1) }), /page hash/);
  await assert.rejects(session.importCheckpoint({ ...checkpoint, logical: { ...checkpoint.logical, head: 10 } }), /logical metadata/);
});

test('import checks actual table bytes and binary, independently of page integrity', async () => {
  const { digest } = await import('./pocket.mjs');
  const page = Buffer.from(checkpoint.page, 'base64');
  page[288] ^= 1;
  await assert.rejects(session.importCheckpoint({ ...checkpoint, page: page.toString('base64'),
    pageHash: await digest(page) }), /table bytes mismatch/);
  const changed = Buffer.concat([wasm, Buffer.from([0, 2, 1, 120])]); // Valid custom section named x.
  const other = await tapeSession(changed, fixture, evidence.graph_sha256);
  await assert.rejects(other.importCheckpoint(checkpoint), /wasm mismatch/);
});
