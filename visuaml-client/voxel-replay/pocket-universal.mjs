import { inspectTape, loadTape, TAPE_BASE } from './pocket-tape.mjs';

export function encode(program, input, symbols) {
  const lines = program.trim().split(/\r?\n/).map(line => line.trim().split(/\s+/));
  if (lines.length > 64 || lines.at(-1)?.[0] !== 'H') throw Error('End the program with H (up to 64 instructions).');
  if (!/^[01]{1,128}$/.test(input)) throw Error('Enter 1–128 input bits.');
  const records = lines.map(([op, arg, extra], i) => {
    if (!/^[LRMEJGH]$/.test(op) || extra !== undefined) throw Error(`Invalid instruction ${i}`);
    const jump = op === 'J' || op === 'G';
    if (jump ? !/^\d+$/.test(arg ?? '') || Number(arg) >= lines.length : arg !== undefined) {
      throw Error(`Invalid operand at instruction ${i}`);
    }
    return `;${i === 0 ? '*' : '-'}${op}${jump ? 'u'.repeat(Number(arg)) : ''}`;
  });
  return [...`#${records.join('')}|${input[0] === '1' ? 'b' : 'a'}${input.slice(1)}${'_'.repeat(24)}`]
    .map(symbol => symbols.indexOf(symbol));
}

export function decode(symbols) {
  const end = symbols.indexOf('|');
  const mark = symbols.indexOf('*');
  const data = symbols.slice(end + 1);
  const head = data.findIndex(symbol => symbol === 'a' || symbol === 'b');
  return { pc: mark < 0 ? null : symbols.slice(0, mark).filter(s => s === ';').length - 1,
    head, tape: data.map(s => Number(s === 'b' || s === '1')) };
}

export async function loadUniversal(wasm, table, program, input) {
  const { instance: { exports: { memory, run } } } = await WebAssembly.instantiate(wasm);
  const tape = encode(program, input, table.symbols);
  const index = loadTape(memory, { tape, states: table.states.length, symbols: table.symbols.length,
    head: 0, rules: table.rules });
  const current = () => {
    const state = inspectTape(memory, index);
    const encoded = state.tape.map(i => table.symbols[i]);
    return { ...state, encoded, decoded: decode(encoded), phase: table.states[state.state] };
  };
  return { current, step() { run(TAPE_BASE); return current(); }, page: () => memory.buffer.slice(0) };
}
