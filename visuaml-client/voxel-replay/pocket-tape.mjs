export const TAPE_BASE = 256;
export const MAX_WORDS = 16300;

function bounded(value, min, max, label) {
  if (!Number.isInteger(value) || value < min || value > max) {
    throw Error(`${label} must be an integer from ${min} to ${max}`);
  }
  return value;
}

export function loadTape(memory, { states, symbols, head, tape, rules }) {
  bounded(states, 1, 256, 'States');
  bounded(symbols, 1, 256, 'Symbols');
  bounded(tape.length, 1, MAX_WORDS - 3 * states * symbols, 'Tape length');
  bounded(head, 0, tape.length - 1, 'Head');
  const words = new Int32Array(8 + 3 * states * symbols + tape.length);
  words.set([states, symbols, tape.length, 0, head, 0, 0, 0]);
  for (let i = 0; i < states * symbols; i++) words[8 + 3 * i] = -1;
  for (const line of rules.split('\n').map(row => row.trim()).filter(Boolean)) {
    const fields = line.split(/\s+/);
    if (fields.length !== 5 || !fields.slice(0, 4).every(field => /^[0-9]+$/.test(field))
      || !['L', 'R', 'S'].includes(fields[4])) throw Error(`Invalid transition: ${line}`);
    const [state, read, next, write] = fields.slice(0, 4).map(Number);
    bounded(state, 0, states - 1, 'State');
    bounded(read, 0, symbols - 1, 'Read symbol');
    bounded(next, 0, states - 1, 'Next state');
    bounded(write, 0, symbols - 1, 'Write symbol');
    const at = 8 + 3 * (state * symbols + read);
    if (words[at] !== -1) throw Error(`Duplicate rule for state ${state}, symbol ${read}`);
    words.set([next, write, { L: -1, S: 0, R: 1 }[fields[4]]], at);
  }
  const tapeIndex = 8 + 3 * states * symbols;
  tape.forEach((symbol, i) => { words[tapeIndex + i] = bounded(symbol, 0, symbols - 1, 'Tape symbol'); });
  new Uint8Array(memory.buffer).fill(0);
  const page = new DataView(memory.buffer);
  words.forEach((value, i) => page.setInt32(TAPE_BASE + 4 * i, value, true));
  return tapeIndex;
}

export function inspectTape(memory, tapeIndex) {
  const page = new DataView(memory.buffer);
  const word = i => page.getInt32(TAPE_BASE + 4 * i, true);
  const [state, head, status, count] = [3, 4, 5, 6].map(word);
  return { state, head, status, count, tape: Array.from({ length: word(2) }, (_, i) => word(tapeIndex + i)) };
}
