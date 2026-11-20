import { readFile } from 'node:fs/promises';
import { tapeSession } from '../pocket-tape-fork.mjs';

const read = path => readFile(new URL(path, import.meta.url));
const fixture = JSON.parse(await read('../fixtures/pocket-tape-fork.json'));
const wasm = await read('../evidence/pocket-tape-compiler/erase.linked.wasm');
const { graph_sha256 } = JSON.parse(await read('../evidence/pocket-universal/replay.json'));
const session = await tapeSession(wasm, fixture, graph_sha256);
const records = [];
const pack = page => Buffer.from(page).toString('base64');

function step(name, live) {
  const before = pack(live.page());
  const state = live.step();
  records.push({ name, origin: live.origin, before, after: pack(live.page()), status: state.status });
  return state;
}

if (process.argv[2] === 'export') {
  const live = session.fresh();
  if (step("short", live).count !== 1) throw Error("Expected committed setup transition");
  const checkpoint = await session.exportCheckpoint(live);
  if (step('short', live).status !== 2) throw Error('Expected left boundary');
  console.log(JSON.stringify({ checkpoint, records }));
} else {
  const checkpoint = JSON.parse(await readFile(process.argv[2], 'utf8'));
  const continuations = {};
  const initial = {};
  for (const [name, edit] of [['clean', undefined], ['fork', { coordinate: 0, symbol: fixture.forkSymbol }]]) {
    let live = await session.importCheckpoint(checkpoint, { left: 1, edit });
    initial[name] = pack(live.page());
    let growth;
    for (let attempt = 0; attempt < 8; attempt++) {
      const prior = await session.exportCheckpoint(live);
      const result = step(name, live);
      if (result.status === 1) break;
      if (result.status === 2) {
        if (growth) throw Error('Unexpected second right stop');
        growth = prior.page;
        live = await session.importCheckpoint(prior, { right: 1 });
      } else if (result.status !== 0) throw Error('Unexpected guest status');
    }
    if (!growth) throw Error('Expected right boundary');
    continuations[name] = growth;
  }
  const wide = session.fresh(-1, [0, 1, 0]);
  for (let i = 0; i < 6; i++) step('wide', wide);
  for (const key of Object.keys(session.identities)) {
    const wrong = { ...checkpoint, identities: { ...checkpoint.identities, [key]: 'wrong' } };
    try {
      await session.importCheckpoint(wrong);
      throw Error('Mismatched identity was accepted');
    } catch (error) {
      if (error.message !== `Checkpoint ${key} mismatch`) throw error;
    }
  }
  console.log(JSON.stringify({ checkpoint: checkpoint.page, restored: initial.clean,
    edited: initial.fork, continuations, records }));
}
