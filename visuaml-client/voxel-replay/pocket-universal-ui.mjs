import { startResumeComparison } from './pocket-tape-resume.mjs';
import resumeFixture from './fixtures/pocket-tape-resume.json';
import resumeProof from './evidence/pocket-tape-resume/replay.json';
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

function setupResume() {
  const start = get('resume-start'), advance = get('resume-advance'), extend = get('resume-extend');
  let comparison, stamp = 0;
  const status = message => { get('resume-status').textContent = message; };
  const display = ({ short, wide }) => {
    for (const [id, state] of [['resume-short', short], ['resume-wide', wide]]) {
      get(id).textContent = `Tape ${state.decoded.tape.join('')} · simulated head ${state.decoded.head} · ${state.count} universal transitions`;
    }
  };
  start.disabled = false;
  start.onclick = async () => {
    const token = ++stamp;
    advance.disabled = extend.disabled = true;
    const next = await startResumeComparison(wasm, table, resumeFixture);
    if (token !== stamp) return;
    comparison = next;
    display(comparison.current());
    advance.disabled = false;
    status('Simulated steps completed: 0');
  };
  advance.onclick = async () => {
    const token = ++stamp;
    advance.disabled = true;
    try {
      let result;
      do {
        result = comparison.advance();
      } while (!result.stopped && !done(result.short) && result.short.phase !== 'boundary');
      display(result);
      if (result.stopped) {
        const hash = await digest(comparison.checkpoint());
        if (token !== stamp) return;
        extend.disabled = false;
        status(hash === resumeProof.checkpoint.checkpoint_sha256
          ? 'Right edge reached. Pre-attempt checkpoint matches VisuaML.'
          : 'Right edge reached. Checkpoint hash differs.');
        return;
      }
      const actual = await Promise.all(comparison.pages().map(digest));
      if (token !== stamp) return;
      const expected = resumeProof.transitions[result.short.count - 1];
      const matches = actual[0] === expected.short_sha256 && actual[1] === expected.wide_sha256;
      status(`Simulated steps completed: ${result.boundary} · ${matches ? 'Both pages match saved VisuaML' : 'Replay differs'}`);
      advance.disabled = done(result.short);
    } catch (error) { if (token === stamp) status(error.message); }
  };
  extend.onclick = async () => {
    const token = ++stamp;
    extend.disabled = true;
    await comparison.resume();
    if (token !== stamp) return;
    display(comparison.current());
    advance.disabled = false;
    status('Restored checkpoint and appended 3 blanks. The guest will retry the move.');
  };
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
    setupResume();
  } catch (error) { get('phase').textContent = error.message; }
}

void boot();
