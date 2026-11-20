import { digest } from './pocket.mjs';

const ops = new Map([
  [0x20, ['local.get', 1]], [0x21, ['local.set', 1]], [0x22, ['local.tee', 1]],
  [0x10, ['call', 1]], [0x41, ['i32.const', 1]], [0x36, ['i32.store', 2]],
  [0x6a, ['i32.add', 0]], [0x6c, ['i32.mul', 0]], [0x4a, ['i32.gt_s', 0]],
]);

export function inspectModule(page, bodyStart = 51) {
  const word = new DataView(page.buffer, page.byteOffset, page.byteLength);
  const length = word.getInt32(400, true);
  if (length === 0) return null;
  const count = word.getInt32(404, true);
  if (length < 54 || length > 174 + (bodyStart - 51) || count < 0 || count > 4) throw Error('Invalid guest compiler output');
  const wasm = page.slice(24576, 24576 + length);
  const rows = [];
  let next = bodyStart, previousLine = 0;
  for (let i = 0; i < count; i++) {
    const offset = 24000 + 12 * i;
    const line = word.getInt32(offset, true);
    const start = word.getInt32(offset + 4, true);
    const end = word.getInt32(offset + 8, true);
    if (![10, 20, 30, 40].includes(line) || line <= previousLine || start !== next
        || start >= end || end > length - 3) throw Error('Invalid guest source map');
    const instructions = [];
    let pc = start;
    const leb = signed => {
      let value = 0, shift = 0;
      while (pc < end && shift < 35) {
        const byte = wasm[pc++];
        value |= (byte & 127) << shift;
        shift += 7;
        if (!(byte & 128)) return signed && shift < 32 && (byte & 64) ? value | (-1 << shift) : value;
      }
      throw Error('Invalid instruction immediate');
    };
    while (pc < end) {
      const byte = pc;
      const encoding = wasm[pc++];
      const op = ops.get(encoding);
      if (!op) throw Error('Unsupported instruction in source map');
      const args = Array.from({ length: op[1] }, () => leb(encoding === 0x41));
      instructions.push({ byte, name: op[0], args });
    }
    const names = instructions.map(item => item.name);
    const source = names.includes('call') ? 'CALL' : names.includes('i32.add') ? `ADD ${instructions[1].args[0]}`
      : names.includes('local.set') ? `SET ${instructions[0].args[0]}`
        : names.filter(name => name === 'i32.store').length === 2 ? 'PRINT' : 'HALT';
    rows.push({ line, start, end, source, instructions });
    next = end;
    previousLine = line;
  }
  if (next !== length - 3) throw Error('Incomplete guest source map');
  return { wasm, rows };
}

export async function runEmitted(wasm) {
  const module = await WebAssembly.compile(wasm);
  if (WebAssembly.Module.imports(module).length) throw Error('Emitted WASM must be self-contained');
  const { exports: { run, memory } } = await WebAssembly.instantiate(module);
  if (!(memory instanceof WebAssembly.Memory) || memory.buffer.byteLength !== 65536 || typeof run !== 'function') {
    throw Error('Unexpected emitted module ABI');
  }
  const count = run(0);
  if (!Number.isInteger(count) || count < 0 || count > 4 || memory.buffer.byteLength !== 65536) {
    throw Error('Invalid emitted output');
  }
  const data = new DataView(memory.buffer);
  return {
    printed: Array.from({ length: count }, (_, i) => data.getInt32(16 + 4 * i, true)),
    accumulators: Array.from({ length: 4 }, (_, i) => data.getInt32(32 + 4 * i, true)),
    binaryHash: await digest(wasm),
    pageHash: await digest(memory.buffer),
  };
}
