import proof from './evidence/voxel-call-checkpoint/replay.json';
import checkpointUrl from './evidence/voxel-call-checkpoint/checkpoint.json?url';

const get = id => document.getElementById(id);
const text = (id, value) => { get(id).textContent = value; };
const start = proof.prefix.at(-1);
const cell = proof.cell_address;
let cursor, record, state, action;
get('download').href = checkpointUrl;

function render() {
  get('near').toggleAttribute('aria-current', record[2] === 0);
  get('far').toggleAttribute('aria-current', record[2] === 1);
  text('cell', record[0] ? 'Block' : 'Opening');
  text('world', `Material ${record[0]} · helper result ${record[4] < 0 ? 'pending' : record[4]} · caller commits ${record[5]}`);
  text('where', `Next PC ${state.next_pc} · call depth ${state.depth} · fuel ${state.fuel}`);
  text('stack', `Operand stack [${state.stack.join(', ')}] · locals [${state.locals.join(', ')}]`);
  text('effect', action);
  const ready = cursor === proof.suffix.length;
  get('step').disabled = get('finish').disabled = ready;
  get('wait').disabled = get('walk').disabled = !ready;
}

function rewind() {
  cursor = 0;
  record = [0, 1, 0, 0, -1, 0];
  state = start;
  action = 'Checkpoint after the terrain store. The helper still holds the original material, 7.';
  render();
}

function step() {
  state = proof.suffix[cursor++];
  const [kind, address, before, after] = state.access;
  if (kind === 2 && address >= cell) record[(address - cell) / 4] = after;
  action = `Executed ${state.opcode} at PC ${state.pc}`;
  if (kind === 2) action += ` · byte ${address}: ${before} → ${after}`;
  if (state.function === 0 && state.depth === 0) action += ' · Rust returned material 7 to its caller';
  render();
}

function branch(move) {
  const outcome = proof.copies[move].next_tick;
  record = outcome.record;
  state = outcome.trace.at(-1);
  action = move ? 'Next-tick input: walk. The agent crossed the open passage.'
    : 'Next-tick input: wait. The passage stayed open and the agent stayed put.';
  render();
}

get('rewind').onclick = rewind;
get('step').onclick = step;
get('finish').onclick = () => { while (cursor < proof.suffix.length) step(); };
get('wait').onclick = () => branch(0);
get('walk').onclick = () => branch(1);
rewind();
