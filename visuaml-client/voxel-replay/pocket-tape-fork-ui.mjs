import { tapeSession } from './pocket-tape-fork.mjs';
import { digest } from './pocket.mjs';
import fixture from './fixtures/pocket-tape-fork.json';
import evidence from './evidence/pocket-tape-fork/replay.json';
import wasmUrl from './evidence/pocket-tape-compiler/erase.linked.wasm?url';

const get = id => document.getElementById(id);
const note = (id, text) => { get(id).textContent = text; };

async function boot() {
  const response = await fetch(wasmUrl);
  if (!response.ok) throw Error('Unable to load linked WASM');
  const wasm = await response.arrayBuffer();
  if (await digest(wasm) !== evidence.wasm_sha256) throw Error('Unexpected linked WASM hash');
  const session = await tapeSession(wasm, fixture, evidence.graph_sha256);
  note('status', 'Browser linked WASM · shipped checkpoint has a VisuaML replay');
  const save = get('export');
  save.disabled = false;
  save.onclick = async () => {
    const live = session.fresh();
    live.step();
    const checkpoint = await session.exportCheckpoint(live);
    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([JSON.stringify(checkpoint, null, 2)], { type: 'application/json' }));
    link.download = 'pocket-checkpoint-v1.json';
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    note('status', 'Checkpoint exported. Import it into both panels after reopening.');
  };
  for (const name of ['clean', 'fork']) setupCopy(session, name);
}

function setupCopy(session, name) {
  const file = get(`${name}-file`);
  const run = get(`${name}-run`);
  const status = text => note(`${name}-status`, text);
  const trace = get(`${name}-trace`);
  let machine, stamp = 0;
  const append = text => {
    const row = document.createElement('li');
    row.textContent = text;
    trace.append(row);
  };
  file.disabled = false;
  file.onchange = async () => {
    const token = ++stamp;
    run.disabled = true;
    trace.replaceChildren();
    const selected = file.files[0];
    if (!selected) return;
    try {
      if (selected.size > 100000) throw Error('Checkpoint file is too large');
      const checkpoint = JSON.parse(await selected.text());
      const edit = name === 'fork' ? { coordinate: 0, symbol: fixture.forkSymbol } : undefined;
      const next = await session.importCheckpoint(checkpoint, { left: 1, edit });
      if (token !== stamp) return;
      machine = next;
      status('Imported into fresh instance · left blank added');
      run.disabled = false;
    } catch (error) { if (token === stamp) status(error.message); }
  };
  run.onclick = async () => {
    const token = ++stamp;
    run.disabled = file.disabled = true;
    try {
      let matches = true;
      for (let attempt = 0; attempt < 64; attempt++) {
        const expected = evidence.runs[name][attempt];
        const prior = await session.exportCheckpoint(machine);
        const next = machine.step();
        matches &&= await digest(machine.page()) === expected?.page_sha256;
        if (next.status === 2) {
          machine = await session.importCheckpoint(prior, { right: 1 });
          append('Right edge · restored pre-attempt page and added a blank');
        } else {
          append(`Step ${next.count} · state ${next.state} · head ${next.head} · tape ${next.tape.join('')}${next.status === 1 ? ' · halt' : ''}`);
        }
        if (next.status === 1) break;
        if (next.status !== 0 && next.status !== 2) throw Error(`Guest status ${next.status}`);
      }
      if (token !== stamp) return;
      const final = machine.current();
      if (final.status !== 1) throw Error('Transition allowance exhausted');
      status(`Tape ${final.tape.join('')} · ${final.count} committed transitions · ${matches ? 'Matches saved VisuaML' : 'Custom run'}`);
      if (name === 'fork' && matches) append(`First divergence: committed transition ${evidence.first_fork_transition}`);
    } catch (error) { if (token === stamp) status(error.message); }
    finally { file.disabled = false; }
  };
}

void boot().catch(error => note('status', error.message));
