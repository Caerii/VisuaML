import { relocateTapePage } from './pocket-tape-checkpoint.mjs';
import { inspectTape, loadTape, TAPE_BASE } from './pocket-tape.mjs';

export function encode(program, input, symbols, blankCells = 24) {
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
  return [...`#${records.join('')}|${input[0] === '1' ? 'b' : 'a'}${input.slice(1)}${'_'.repeat(blankCells)}`]
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

export async function loadUniversal(wasm, table, program, input, blankCells = 24) {
  const { instance: { exports: { memory, run } } } = await WebAssembly.instantiate(wasm);
  const tape = encode(program, input, table.symbols, blankCells);
  const index = loadTape(memory, { tape, states: table.states.length, symbols: table.symbols.length,
    head: 0, rules: table.rules });
  return universalInstance(memory, run, table, index);
}

function universalInstance(memory, run, table, index) {
  const current = () => {
    const state = inspectTape(memory, index);
    const encoded = state.tape.map(i => table.symbols[i]);
    return { ...state, encoded, decoded: decode(encoded), phase: table.states[state.state] };
  };
  return { current, step() { run(TAPE_BASE); return current(); }, page: () => memory.buffer.slice(0) };
}


// Restore a clean instruction-boundary checkpoint and extend its logical window.
export async function extendUniversal(wasm, table, checkpoint, extra) {
  if (checkpoint.byteLength !== 65536) throw Error("Cannot extend tape checkpoint");
  const page = new DataView(checkpoint);
  const index = 8 + 3 * table.states.length * table.symbols.length;
  if (page.getInt32(TAPE_BASE, true) !== table.states.length
      || page.getInt32(TAPE_BASE + 4, true) !== table.symbols.length
      || table.symbols[0] !== '_') throw Error('Cannot extend tape checkpoint');
  const extended = relocateTapePage(checkpoint, { right: extra });
  const { instance: { exports: { memory, run } } } = await WebAssembly.instantiate(wasm);
  new Uint8Array(memory.buffer).set(new Uint8Array(extended));
  return universalInstance(memory, run, table, index);
}
