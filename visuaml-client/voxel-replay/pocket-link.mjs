// Link the single Pocket import to a separately built, unary leaf.
const type = [1, 96, 1, 127, 1, 127];
const memory = [1, 1, 1, 1];
const imported = [1, 6, ...Array.from('pocket', c => c.charCodeAt(0)),
  8, ...Array.from('passable', c => c.charCodeAt(0)), 0, 0];
const exportsFor = index => [2, 6, 109, 101, 109, 111, 114, 121, 2, 0, 3, 114, 117, 110, 0, index];
const same = (a, b) => a?.length === b.length && b.every((value, i) => a[i] === value);
const u32 = value => {
  const bytes = [];
  while (value >= 128) { bytes.push((value & 127) | 128); value >>>= 7; }
  return [...bytes, value];
};
function readWord(bytes, start) {
  let value = 0;
  for (let i = 0; i < 5 && start + i < bytes.length; i++) {
    const byte = bytes[start + i];
    value += (byte & 127) * 2 ** (7 * i);
    if (byte < 128 && value <= 0xffffffff) return [value, start + i + 1];
  }
  throw Error('Invalid WASM length');
}
const section = (id, payload) => [id, ...u32(payload.length), ...payload];
function parse(wasm, expectedIds, index) {
  if (!WebAssembly.validate(wasm)) throw Error('Invalid WASM artifact');
  const sections = new Map();
  for (let pos = 8; pos < wasm.length;) {
    const id = wasm[pos++];
    const [size, start] = readWord(wasm, pos);
    pos = start + size;
    if (pos > wasm.length || sections.has(id)) throw Error('Invalid WASM section');
    sections.set(id, wasm.slice(start, pos));
  }
  if (!same([...sections.keys()], expectedIds) || !same(sections.get(1), type)) {
    throw Error('Expected unary i32 → i32 signature and bounded module shape');
  }
  if (!same(sections.get(3), [1, 0]) || !same(sections.get(5), memory)
      || !same(sections.get(7), exportsFor(index))) throw Error('Unexpected function index, exports, or memory');
  const code = sections.get(10);
  const [size, start] = readWord(code, 1);
  if (code[0] !== 1 || start + size !== code.length) throw Error('Unexpected function body count');
  return { sections, body: code.slice(start), bodyStart: wasm.length - size };
}

// Walk just the bounded integer bodies emitted by these two toolchains.
function checkCalls(body, allowImport) {
  let [groups, pc] = readWord(body, 0);
  while (groups-- > 0) {
    [, pc] = readWord(body, pc);
    if (body[pc++] !== 0x7f) throw Error('Expected i32 locals');
  }
  while (pc < body.length) {
    const op = body[pc++];
    if (op === 0x10) {
      let target;
      [target, pc] = readWord(body, pc);
      if (!allowImport || target !== 0) throw Error('Unexpected direct call target');
    } else if ([2, 3, 4].includes(op)) {
      if (body[pc++] !== 0x40) throw Error('Expected empty control signature');
    } else if (op === 0x41) {
      // Only the length matters here. Browser validation checks the signed value.
      while (body[pc++] & 128) { /* continuation */ }
    } else if ([0x0c, 0x0d, 0x20, 0x21, 0x22].includes(op)) {
      [, pc] = readWord(body, pc);
    } else if (op === 0x36) {
      [, pc] = readWord(body, pc);
      [, pc] = readWord(body, pc);
    } else if (![5, 11, 15, 0x45, 0x48, 0x4a, 0x6a, 0x6b, 0x6c].includes(op)) {
      throw Error('Unexpected instruction in link input');
    }
  }
}

export function linkPassable(guest, helper) {
  const caller = parse(guest, [1, 2, 3, 5, 7, 10], 1);
  if (!same(caller.sections.get(2), imported)) throw Error('Expected declaration pocket.passable with type 0');
  const rust = parse(helper, [1, 3, 5, 7, 10], 0);
  checkCalls(caller.body, true);
  checkCalls(rust.body, false);
  const prefix = [...guest.slice(0, 8), ...section(1, type), ...section(3, [2, 0, 0]),
    ...section(5, memory), ...section(7, exportsFor(1))];
  const wasm = Uint8Array.from([...prefix, ...section(10, [2, ...u32(rust.body.length), ...rust.body,
    ...u32(caller.body.length), ...caller.body])]);
  if (!WebAssembly.validate(wasm)) throw Error('Linked module does not validate');
  const delta = wasm.length - caller.body.length - caller.bodyStart;
  return { wasm, delta, helperBody: rust.body, helperStart: wasm.length - caller.body.length
    - u32(caller.body.length).length - rust.body.length };
}
