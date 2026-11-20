import { createPocket, createPresenter, digest, POCKET_WASM, PERIOD_MS } from './pocket.mjs';
import editProof from './fixtures/pocket-edit.proof.json';
import omitProof from './fixtures/pocket-omitted.proof.json';

const get = id => document.getElementById(id);
const present = createPresenter(get('display'));
const actionIds = ['dig', 'play', 'step', 'reset', 'edit', 'omit', 'export', 'checkpoint'];
const proofTable = { edit: editProof, omit: omitProof };
const hex = word => '0x' + word.toString(16).padStart(8, '0');
let raw, pocket, log = [], proof = null, queued = 0, playing = false;
let generation = 0, clock = 0, serial = 0, resetting = false;

function pause() {
  playing = false;
  clearTimeout(clock);
  get('play').textContent = 'Run';
}

function timeline() {
  get('timeline').replaceChildren();
  if (!proof) return;
  proof.ticks.forEach((_, index) => {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.textContent = ['Boot', proof.session.buttons[1] ? 'Edit tick' : 'Edit omitted', 'Next tick'][index];
    chip.setAttribute('aria-pressed', index === log.length - 1);
    chip.onclick = () => void safely(() => replayTo(proof, index));
    get('timeline').append(chip);
  });
}

function inspect(result, button) {
  const snapshot = ++serial;
  const index = log.length - 1;
  const record = proof?.ticks[index];
  get('tick').textContent = String(index);
  get('terrain').textContent = result.block ? 'Blocked' : 'Open';
  get('agent').textContent = result.crossed ? 'Crossed' : 'Waiting';
  get('event').textContent = index === 0 || button === -1 ? 'Scene drawn'
    : result.block !== result.before.block ? 'Block cleared · passage pixels changed'
    : result.crossed !== result.before.crossed ? 'Agent crossed · two tiles changed'
    : 'No terrain or agent change';
  get('writes').replaceChildren();
  get('verification').textContent = record ? 'Checking saved replay…' : 'Live browser WASM · export to verify offline';
  if (record) {
    const page = pocket.checkpoint();
    void Promise.all([digest(page), digest(result.frame)]).then(([pageHash, frameHash]) => {
      if (snapshot !== serial) return;
      const matches = pageHash === record.page_sha256 && frameHash === record.frame_sha256;
      get('verification').textContent = matches ? 'Matches saved VisuaML replay' : 'Replay hash mismatch';
      if (!matches) return;
      for (const [stage, writes] of Object.entries(record.events)) {
        for (const { pc, address, before, after } of writes.slice(0, 4)) {
          const row = document.createElement('li');
          row.textContent = `${stage} · PC ${pc} · [${address}] ${hex(before)} → ${hex(after)}`;
          get('writes').append(row);
        }
        if (writes.length > 4) {
          const row = document.createElement('li');
          row.textContent = `+${writes.length - 4} ${stage} writes in saved proof`;
          get('writes').append(row);
        }
      }
    }).catch(showError);
  }
  timeline();
}

function tick(button) {
  log.push(button);
  inspect(pocket.step(button, present), button);
}

async function reset(nextProof = null) {
  pause();
  const token = ++generation;
  resetting = true;
  const instance = await createPocket(raw, editProof.wasm_sha256);
  if (token !== generation) return false;
  pocket = instance;
  proof = nextProof;
  queued = 0;
  log = [];
  resetting = false;
  tick(0);
  return true;
}

async function replayTo(nextProof, index) {
  if (!await reset(nextProof)) return;
  for (const button of nextProof.session.buttons.slice(1, index + 1)) tick(button);
}

function step() {
  if (resetting || !pocket) return;
  if (proof) {
    if (log.length >= proof.session.buttons.length) { pause(); return; }
    tick(proof.session.buttons[log.length]);
    if (log.length >= proof.session.buttons.length) pause();
  } else {
    const button = queued;
    queued = 0;
    tick(button);
  }
}

function play() {
  if (resetting || !pocket) return;
  if (playing) { pause(); return; }
  if (proof && log.length >= proof.session.buttons.length) {
    void safely(() => startReplay(proof));
    return;
  }
  playing = true;
  get('play').textContent = 'Pause';
  let due = performance.now();
  function pulse() {
    if (!playing) return;
    try {
      step();
      due += PERIOD_MS;
      if (playing) clock = setTimeout(pulse, Math.max(0, Math.ceil(due - performance.now())));
    } catch (error) { showError(error); }
  }
  pulse();
}

async function startReplay(nextProof) {
  if (await reset(nextProof)) play();
}

function dig() {
  if (resetting || !pocket) return;
  proof = null;
  queued = 1;
  if (!playing) step();
}

function showError(error) {
  pause();
  resetting = false;
  get('runtime').textContent = String(error);
}

async function safely(fn) {
  try { await fn(); } catch (error) { showError(error); }
}

get('dig').onclick = dig;
get('play').onclick = play;
get('step').onclick = () => { pause(); step(); };
get('reset').onclick = () => void safely(() => reset());
get('edit').onclick = () => void safely(() => startReplay(proofTable.edit));
get('omit').onclick = () => void safely(() => startReplay(proofTable.omit));
function download(name, blob) {
  const link = document.createElement('a');
  link.download = name;
  link.href = URL.createObjectURL(blob);
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 100);
}
get('export').onclick = () => {
  if (!pocket) return;
  download('pocket-session.json', new Blob([JSON.stringify({
    version: 1, binary_sha256: pocket.hash, buttons: log,
  }, null, 2)], { type: 'application/json' }));
};
get('checkpoint').onclick = () => {
  if (!pocket) return;
  pause();
  download(`pocket-tick-${log.length - 1}.bin`, new Blob([pocket.checkpoint()]));
};
document.addEventListener('keydown', event => {
  if (event.repeat || event.altKey || event.ctrlKey || event.metaKey) return;
  if (event.key.toLowerCase() === 'x') dig();
  if (event.key === 'ArrowRight') { event.preventDefault(); pause(); step(); }
});
void safely(async () => {
  const response = await fetch(POCKET_WASM);
  if (!response.ok) throw Error('Unable to load the Pocket guest');
  raw = await response.arrayBuffer();
  await reset();
  get('runtime').textContent = 'Browser WASM playback';
  actionIds.forEach(id => { get(id).disabled = false; });
});
