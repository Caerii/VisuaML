import { linkUnary } from './pocket-link.mjs';
import { inspectTape, TAPE_BASE } from './pocket-tape.mjs';

export function readTapeArtifact(page) {
  const data = new DataView(page.buffer, page.byteOffset, page.byteLength);
  const size = data.getInt32(400, true);
  if (!size) return null;
  if (size !== 76) throw Error('Unexpected tape module length');
  return {
    wasm: page.slice(24576, 24576 + size),
    table: page.slice(25000, 25084),
    lines: [25100, 25104, 25108].map(offset => data.getInt32(offset, true)),
  };
}

export async function startTapeArtifact(artifact, helper) {
  const { wasm, delta } = linkUnary(artifact.wasm, helper, 'tapeStep');
  const { instance: { exports: { memory, run } } } = await WebAssembly.instantiate(wasm);
  new Uint8Array(memory.buffer).set(artifact.table, TAPE_BASE);
  const read = () => inspectTape(memory, 17);
  return {
    wasm, callByte: 73 + delta, read,
    checkpoint: () => new Uint8Array(memory.buffer).slice(),
    step() {
      const before = read();
      run(0);
      return { ...read(), line: artifact.lines[before.tape[before.head]], previousHead: before.head };
    },
  };
}
