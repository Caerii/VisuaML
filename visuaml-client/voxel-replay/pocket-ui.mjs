import { createPocket, createPresenter, digest, PERIOD_MS } from './pocket.mjs';
const manifests = Object.values(import.meta.glob('./cartridges/*.json', { eager: true, import: 'default' }));
const binaries = import.meta.glob('./guest/*.wasm', { eager: true, query: '?url', import: 'default' });
const proofs = import.meta.glob('./fixtures/*.proof.json', { eager: true, import: 'default' });
const clips = import.meta.glob('./evidence/{pocket,cartridges,monitor-program}/*.gif', { eager: true, query: '?url', import: 'default' });
const get = id => document.getElementById(id);
const present = createPresenter(get('display'));
const actionIds = ['play', 'step', 'reset', 'export', 'checkpoint'];
let cartridge = null, history = null;
const localBinaries = new Map();
const hex = word => '0x' + word.toString(16).padStart(8, '0');
let raw, pocket, log = [], proof = null, queued = [], playing = false;
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
    chip.textContent = history?.frames?.[index] ?? (index === 0 ? 'Boot' : `Key ${index}`);
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
  for (const item of cartridge.readouts) {
    const value = result.fields[item.id];
    get(item.id).textContent = item.values?.[value] ?? String(value);
  }
  const notice = cartridge.notices.find(item =>
    item.changed?.some(field => result.fields[field] !== result.before[field]) || item.commands?.includes(button));
  get('event').textContent = index === 0 || button === -1 ? 'Scene drawn' : notice?.text ?? cartridge.idleNotice;
  get('writes').replaceChildren();
  get('verification').textContent = record ? 'Checking saved replay…' : 'Live browser WASM · export to verify offline';
  if (record) {
    const page = pocket.checkpoint();
    void Promise.all([digest(page), digest(result.frame)]).then(([pageHash, frameHash]) => {
      if (snapshot !== serial) return;
      const matches = pageHash === record.page_sha256 && frameHash === record.frame_sha256;
      get('verification').textContent = matches ? 'Matches saved VisuaML replay' : 'Replay hash mismatch';
      if (!matches) return;
      for (const stage of record.phases) {
        const row = document.createElement('li');
        row.textContent = `${stage.stage} · ${stage.source_steps} source steps`;
        get('writes').append(row);
      }
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
  const instance = await createPocket(raw, cartridge.binary_sha256, cartridge);
  if (token !== generation) return false;
  pocket = instance;
  proof = nextProof;
  queued = [];
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
    const button = queued.shift() ?? 0;
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

function input(code) {
  if (resetting || !pocket) return;
  proof = null;
  queued.push(code);
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

get('play').onclick = play;
get('step').onclick = () => { pause(); step(); };
get('reset').onclick = () => void safely(() => reset());

function button(parent, id, label, handler) {
  const control = document.createElement('button');
  control.id = id;
  control.type = 'button';
  control.textContent = label;
  control.onclick = handler;
  parent.append(control);
  return control;
}

async function loadCartridge(next) {
  pause();
  const token = ++generation;
  resetting = true;
  get('runtime').textContent = 'Loading browser WASM…';
  const binary = binaries[`./guest/${next.binary}`];
  let loaded = localBinaries.get(next.id);
  if (!loaded) {
    const response = await fetch(binary);
    if (!response.ok) throw Error('Unable to load cartridge');
    loaded = await response.arrayBuffer();
  }
  if (token !== generation) return;
  raw = loaded;
  cartridge = next;
  history = null;
  get('screen-title').textContent = next.name;
  get('guide').textContent = next.guide;
  get('game-controls').replaceChildren();
  if (next.keyboard === 'ascii') {
    const label = document.createElement('label');
    label.textContent = 'Keyboard ';
    const field = document.createElement('input');
    field.id = 'terminal-keys';
    field.type = 'text';
    field.autocomplete = 'off';
    field.placeholder = 'Type a command';
    field.addEventListener('beforeinput', event => {
      event.preventDefault();
      if (event.inputType === 'deleteContentBackward') input(8);
      if (event.inputType === 'insertLineBreak') input(13);
      for (const character of event.data ?? '') {
        const code = character.charCodeAt(0);
        if (pocket?.accepts(code)) input(code);
      }
    });
    label.append(field);
    get('game-controls').append(label);
  }
  next.controls.forEach(control => {
    const node = button(get('game-controls'), control.id, control.label, () => input(control.code));
    const key = document.createElement('kbd');
    key.textContent = control.keyLabel ?? control.key;
    node.append(' ', key);
    if (control.primary) node.classList.add('primary');
  });
  document.querySelector('.scenes').replaceChildren();
  next.replays.forEach(replay => button(document.querySelector('.scenes'), replay.id, replay.label, () => {
    history = replay;
    const saved = proofs[`./fixtures/${replay.proof}`];
    void safely(() => startReplay(saved));
  }));
  const tickRow = get('tick').parentElement;
  document.querySelector('.readout').replaceChildren(tickRow);
  for (const item of next.readouts) {
    const div = document.createElement('div');
    const name = document.createElement('dt');
    name.textContent = item.label;
    const value = document.createElement('dd');
    value.id = item.id;
    div.append(name, value);
    document.querySelector('.readout').append(div);
  }
  await reset();
  if (cartridge !== next || resetting) return;
  get('runtime').textContent = 'Browser WASM playback';
  actionIds.forEach(id => { get(id).disabled = false; });
  if (next.keyboard === 'ascii') get('terminal-keys')?.focus();
}
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
    version: 1, cartridge: cartridge.id, binary_sha256: pocket.hash, buttons: log,
  }, null, 2)], { type: 'application/json' }));
};
get('checkpoint').onclick = () => {
  if (!pocket) return;
  pause();
  download(`pocket-tick-${log.length - 1}.bin`, new Blob([pocket.checkpoint()]));
};
document.addEventListener('keydown', event => {
  if (event.repeat || event.altKey || event.ctrlKey || event.metaKey || !pocket || resetting) return;
  const key = event.key;
  if (cartridge.keyboard === 'ascii' && (event.target.id === 'terminal-keys' || event.target === document.body)) {
    const code = key === 'Enter' ? 13 : key === 'Backspace' ? 8 : key.length === 1 ? key.charCodeAt(0) : -2;
    if (code > 0 && pocket.accepts(code)) { event.preventDefault(); input(code); }
    return;
  }
  const control = cartridge.controls.find(item => item.key.toLowerCase() === key.toLowerCase());
  if (control) { event.preventDefault(); input(control.code); }
  else if (key === 'ArrowRight') { event.preventDefault(); pause(); step(); }
});

const sorted = manifests.sort((a, b) => a.order - b.order);
for (const item of sorted) {
  get('cartridge').append(new Option(item.name, item.id));
  for (const [label, filename] of item.gallery) {
    const path = Object.keys(clips).find(p => p.endsWith('/' + filename));
    if (!path) continue;
    const link = document.createElement('a');
    link.href = clips[path];
    const img = document.createElement('img');
    img.src = clips[path];
    img.alt = `${item.name} · ${label}`;
    img.loading = 'lazy';
    const text = document.createElement('span');
    text.textContent = `${item.name} · ${label}`;
    link.append(img, text);
    get('gallery').append(link);
  }
}
get('cartridge').onchange = () => {
  const next = sorted.find(item => item.id === get('cartridge').value);
  if (next) void safely(() => loadCartridge(next));
};
get('host-cartridge-files').onchange = event => void safely(async () => {
  const files = [...event.target.files];
  const manifestFile = files.find(file => file.name.endsWith('.json'));
  const wasmFile = files.find(file => file.name.endsWith('.wasm'));
  if (files.length !== 2 || !manifestFile || !wasmFile) throw Error('Select one manifest and its WASM file');
  const data = JSON.parse(await manifestFile.text());
  if (typeof data.name !== 'string' || typeof data.guide !== 'string'
      || !/^[a-f0-9]{64}$/.test(data.binary_sha256) || data.binary !== wasmFile.name
      || !Array.isArray(data.controls)
      || data.controls.some(control => !Number.isInteger(control.code) || control.code < 1
        || control.code > 0x7fffffff || typeof control.key !== 'string' || typeof control.label !== 'string')
      || (data.keyboard != null && data.keyboard !== 'ascii')) throw Error('Invalid local cartridge manifest');
  const raw = await wasmFile.arrayBuffer();
  const id = `local-${data.binary_sha256.slice(0, 12)}`;
  if (await digest(raw) !== data.binary_sha256) throw Error('Pocket binary hash mismatch');
  const entry = { id, name: data.name, binary: data.binary, binary_sha256: data.binary_sha256,
    guide: data.guide, keyboard: data.keyboard, controls: data.controls.map((control, i) =>
      ({ ...control, id: `${id}-control-${i}` })), readouts: [], notices: [], replays: [],
    idleNotice: 'Local cartridge' };
  const previous = sorted.findIndex(record => record.id === id);
  if (previous < 0) {
    sorted.push(entry);
    get('cartridge').append(new Option(entry.name, id));
  } else sorted[previous] = entry;
  localBinaries.set(id, raw);
  get('cartridge').value = id;
  await loadCartridge(entry);
});
const requested = new URLSearchParams(location.search).get('cartridge');
const initial = sorted.find(item => item.id === requested) ?? sorted[0];
get('cartridge').value = initial.id;
void safely(() => loadCartridge(initial));
