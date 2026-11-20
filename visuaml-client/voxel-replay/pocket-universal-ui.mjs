import { digest } from './pocket.mjs';
import { loadUniversal } from './pocket-universal.mjs';
import cases from './fixtures/pocket-universal.json';
import table from './evidence/pocket-universal/table.json';
import evidence from './evidence/pocket-universal/replay.json';
import wasmUrl from './evidence/pocket-tape-compiler/erase.linked.wasm?url';

const get = id => document.getElementById(id);
const [micro, macro] = ['micro', 'macro'].map(get);
let machine, wasm, saved, last, boundary, serial = 0;
const done = state => state.status !== 0 || ['halt', 'left_edge'].includes(state.phase)
  || state.count >= evidence.universal_transition_limit;

function row(id, symbols, head, width) {
  const start = Math.min(Math.max(0, head - (width >> 1)), Math.max(0, symbols.length - width));
  get(id).replaceChildren(...symbols.slice(start, start + width).map((symbol, i) => {
    const cell = document.createElement('span');
    cell.textContent = symbol;
    if (start + i === head) cell.setAttribute('aria-current', 'true');
    return cell;
  }));
}

function render(state) {
  get('phase').textContent = `Universal transition ${state.count} · ${state.phase} · head ${state.head}`
    + (state.status ? ` · guest status ${state.status}` : '')
    + (state.count >= evidence.universal_transition_limit ? ' · transition allowance reached' : '');
  row('physical', state.encoded, state.head, 39);
  row('simulated', last.tape, last.head, 20);
  get('boundary').textContent = `Simulated boundary ${boundary} · instruction ${last.pc} · head ${last.head}`;
  micro.disabled = macro.disabled = done(state);
}

async function load() {
  const stamp = ++serial;
  try {
    const program = get('program').value.trim(), bits = get('bits').value.trim();
    const next = await loadUniversal(wasm, table, program, bits);
    if (stamp !== serial) return;
    machine = next;
    saved = evidence.runs[cases.find(c => c.program === program && c.input === bits)?.name];
    boundary = 0;
    const state = machine.current();
    last = state.decoded;
    get('error').textContent = '';
    get('verify').textContent = saved ? 'Example has a saved VisuaML replay' : 'Custom encoding · browser WASM';
    render(state);
  } catch (error) { get('error').textContent = error.message; }
}

async function step(simulated) {
  let state = machine.current();
  if (done(state)) return;
  const stamp = ++serial;
  do {
    state = machine.step();
    if (state.phase === 'boundary' || state.phase === 'halt') {
      last = state.decoded;
      boundary++;
      break;
    }
  } while (simulated && !done(state));
  render(state);
  const prior = saved?.transitions[state.count - 1];
  if (prior) {
    const hash = await digest(machine.page());
    if (stamp !== serial) return;
    get('verify').textContent = hash === prior.page_sha256
      ? `Matches saved VisuaML page · transition ${state.count}` : 'Saved VisuaML page differs';
  }
}

function preset() {
  const example = cases[Number(get('preset').value)];
  get('program').value = example.program;
  get('bits').value = example.input;
  void load();
}

async function boot() {
  try {
    const response = await fetch(wasmUrl);
    if (!response.ok) throw Error('Unable to load linked WASM');
    wasm = await response.arrayBuffer();
    if (await digest(wasm) !== evidence.linked_wasm_sha256) throw Error('Linked WASM hash differs');
    get('preset').append(...cases.map((c, i) => new Option(c.name, i)));
    get('preset').onchange = preset;
    get('load').onclick = load;
    get('load').disabled = false;
    micro.onclick = () => void step(false);
    macro.onclick = () => void step(true);
    preset();
  } catch (error) { get('phase').textContent = error.message; }
}

void boot();
