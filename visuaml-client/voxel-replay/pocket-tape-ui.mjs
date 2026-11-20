import examples from './fixtures/pocket-tape.json';
import evidence from './evidence/pocket-tape/replay.json';
import wasmUrl from './guest/pocket_tape.wasm?url';
import { digest } from './pocket.mjs';
import { inspectTape, loadTape, TAPE_BASE } from './pocket-tape.mjs';

const get = id => document.getElementById(id);
const [runButton, stepButton, reset] = ['tape-run', 'tape-step', 'load-tape'].map(get);
let memory, run, tapeIndex, timer, tick, preset, serial = 0;
const states = ['Ready', 'Halted · no matching rule', 'Tape boundary · write not committed', 'Invalid machine'];
const stop = () => {
  clearInterval(timer);
  timer = undefined;
  runButton.textContent = 'Run';
};

function render(snapshot) {
  const { tape, head, state, count, status } = snapshot;
  get('tape-state').textContent = `State ${state} · head ${head} · ${count} transitions · ${states[status]}`;
  const start = Math.min(Math.max(0, head - 8), Math.max(0, tape.length - 17));
  get('tape').replaceChildren(...tape.slice(start, start + 17).map((symbol, index) => {
    const cell = document.createElement('div');
    cell.className = 'cell';
    const position = start + index;
    if (position === head) cell.setAttribute('aria-current', 'true');
    const label = document.createElement('small');
    label.textContent = position;
    cell.append(label, String(symbol));
    return cell;
  }));
  runButton.disabled = stepButton.disabled = status !== 0;
  if (status !== 0) stop();
}

async function step() {
  run(TAPE_BASE);
  render(inspectTape(memory, tapeIndex));
  const expected = preset && evidence.runs[preset]?.[tick++];
  const stamp = ++serial;
  if (!expected) return;
  const hash = await digest(memory.buffer.slice(0));
  if (serial === stamp) get('tape-proof').textContent = hash === expected.page_sha256
    ? 'Full page matches saved VisuaML step' : 'Saved VisuaML step differs';
}

function load(example) {
  stop();
  ++serial;
  try {
    const config = example ?? {
      states: Number(get('states').value), symbols: Number(get('symbols').value),
      head: Number(get('head').value), rules: get('rules').value,
      tape: get('initial-tape').value.split(',').map(value => value.trim() ? Number(value) : NaN),
    };
    tapeIndex = loadTape(memory, config);
    tick = 0;
    preset = example?.name;
    get('tape-proof').textContent = preset ? 'Example has a saved VisuaML replay' : 'Custom rules · live browser WASM';
    get('tape-error').textContent = '';
    render(inspectTape(memory, tapeIndex));
  } catch (error) {
    get('tape-error').textContent = error.message;
  }
}

function selectExample() {
  const example = examples[Number(get('example').value)];
  for (const key of ['states', 'symbols', 'head', 'rules']) get(key).value = example[key];
  get('initial-tape').value = example.tape.join(', ');
  load(example);
}

try {
  const bytes = await (await fetch(wasmUrl)).arrayBuffer();
  if (await digest(bytes) !== evidence.wasm_sha256) throw Error('Tape guest hash differs from saved evidence');
  const { instance } = await WebAssembly.instantiate(bytes);
  ({ memory, run } = instance.exports);
  get('example').append(...examples.map((example, i) => new Option(example.name, i)));
  get('example').onchange = selectExample;
  reset.onclick = () => load();
  stepButton.onclick = () => { stop(); void step(); };
  runButton.onclick = () => {
    if (timer) stop();
    else {
      runButton.textContent = 'Pause';
      timer = setInterval(step, 280);
      void step();
    }
  };
  reset.disabled = false;
  selectExample();
} catch (error) {
  get('tape-state').textContent = error.message;
}
