export const WIDTH = 160;
export const HEIGHT = 144;
export const FRAME_BYTES = 5760;
export const FRAME_OFFSET = 4096;
export const BASE = 256;
export const PERIOD_MS = 50;
export const POCKET_WASM = new URL('./guest/pocket.wasm', import.meta.url).href;
export const PALETTE = [
  [220, 235, 179], [147, 171, 112], [65, 98, 78], [21, 45, 40],
];

export async function digest(data) {
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', data)),
    value => value.toString(16).padStart(2, '0')).join('');
}

export function shade(frame, x, y) {
  return (frame[y * 40 + (x >> 2)] >>> (6 - ((x & 3) << 1))) & 3;
}

export function createPresenter(canvas) {
  const context = canvas.getContext('2d', { alpha: false });
  const image = context.createImageData(WIDTH, HEIGHT);
  return frame => {
    let out = 0;
    for (const byte of frame) {
      for (let shift = 6; shift >= 0; shift -= 2) {
        const [r, g, b] = PALETTE[(byte >>> shift) & 3];
        image.data[out++] = r;
        image.data[out++] = g;
        image.data[out++] = b;
        image.data[out++] = 255;
      }
    }
    context.putImageData(image, 0, 0);
  };
}

export async function createPocket(raw, expectedHash) {
  const hash = await digest(raw);
  if (hash !== expectedHash) throw Error('Pocket binary hash mismatch');
  const { instance } = await WebAssembly.instantiate(raw);
  const { memory, run } = instance.exports;
  if (memory.buffer.byteLength !== 65536) throw Error('Pocket requires one memory page');
  const page = new Uint8Array(memory.buffer);
  const state = new DataView(memory.buffer);
  const get = offset => state.getInt32(BASE + offset, true);
  const put = (offset, value) => state.setInt32(BASE + offset, value, true);
  return {
    hash,
    checkpoint: () => page.slice(),
    step(button, present, audit = false) {
      if (![-1, 0, 1].includes(button)) throw Error('Invalid Pocket button');
      const began = performance.now();
      const before = { block: get(12), crossed: get(16) };
      put(0, button);
      put(4, 0);
      const updateStart = performance.now();
      const update = run(BASE);
      const updateEnd = performance.now();
      const simulationPage = audit ? page.slice() : null;
      put(4, 1);
      const drawStart = performance.now();
      const draw = run(BASE);
      const drawEnd = performance.now();
      const frame = page.slice(FRAME_OFFSET, FRAME_OFFSET + FRAME_BYTES);
      const publicationEnd = performance.now();
      present?.(frame);
      const presented = performance.now();
      if (update !== draw) throw Error('Drawing changed the scene state');
      return { frame, before, block: get(12), crossed: get(16), simulationPage,
        timing: { input_ms: updateStart - began + drawStart - updateEnd,
          simulation_ms: updateEnd - updateStart, drawing_ms: drawEnd - drawStart,
          publication_ms: publicationEnd - drawEnd, presentation_ms: presented - publicationEnd } };
    },
  };
}
