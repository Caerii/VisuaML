export const BASE = 256;
export const GUESTS = {
  single: new URL('../../benchmarks/results/v58-voxel-tick/voxel_tick.wasm', import.meta.url),
  shared: new URL('./guest/shared_tick.wasm', import.meta.url),
  terrain: new URL('./guest/terrain_tick.wasm', import.meta.url),
};

export async function sha256(bytes) {
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
}

export async function createWorld(binary, session) {
  const { mode, initial } = session;
  const shared = mode === 'shared';
  const terrain = mode === 'terrain';
  const count = terrain ? initial.positions.length : (shared ? 2 : 1);
  const cells = terrain ? initial.materials.length : (shared ? 5 : 4);
  const materialOffset = terrain ? 576 : (shared ? 32 : 16);
  const positionsOffset = terrain ? 64 : 0;
  const inputOffset = terrain ? 320 : (shared ? 8 : 4);
  if (!Object.hasOwn(GUESTS, mode) || session.version !== 1 || initial.positions.length !== count
      || count < 1 || count > 64 || cells < count || cells > 64 || initial.materials.length !== cells
      || initial.materials.some((value) => !Number.isInteger(value) || value < 0 || value > 0x7fffffff)
      || initial.positions.some((value) => !Number.isInteger(value) || value < 0 || value >= cells
        || initial.materials[value] !== 0)
      || new Set(initial.positions).size !== count) throw Error('Invalid initial world');
  if (await sha256(binary) !== session.binary_sha256) throw Error('Guest binary hash mismatch');
  const { instance } = await WebAssembly.instantiate(binary);
  const { memory, run } = instance.exports;
  const page = new Uint8Array(memory.buffer);
  const state = new DataView(memory.buffer);
  const read = (offset) => state.getInt32(BASE + offset, true);
  const write = (offset, value) => state.setInt32(BASE + offset, value, true);
  const positions = () => Array.from({ length: count }, (_, index) => read(positionsOffset + 4 * index) / 4);
  const reset = () => {
    page.fill(0);
    if (terrain) {
      write(0, count * 4);
      write(16, cells * 4);
    }
    initial.positions.forEach((value, index) => write(positionsOffset + 4 * index, 4 * value));
    initial.materials.forEach((value, index) => write(materialOffset + 4 * index, value));
  };
  reset();

  return {
    reset,
    step(input) {
      const previous = positions();
      const moves = terrain ? input.moves : input;
      const edit = terrain && input.edit;
      const editCell = edit ? previous[0] + 1 : null;
      if (terrain && (typeof edit !== 'boolean' || editCell >= cells)) throw Error('Invalid terrain edit');
      const targets = previous.map((position, index) => position + moves[index]);
      if (moves.length !== count || moves.some((delta) => ![-1, 0, 1].includes(delta))
        || targets.some((target) => target < 0 || target >= cells)) throw Error('Move would leave the board');
      const before = page.slice();
      const start = performance.now();
      if (terrain) write(4, Number(edit));
      moves.forEach((delta, index) => write(inputOffset + 4 * index, 4 * delta));
      const returned = run(BASE);
      const simulationMs = performance.now() - start;
      const after = page.slice();
      const changes = [];
      for (let address = 0; address < after.length; address++) {
        if (after[address] === before[address]) continue;
        const offset = address - BASE;
        changes.push({
          address, before: before[address], after: after[address],
          owner: (terrain && offset >= 4 && offset < 8)
            || (offset >= inputOffset && offset < inputOffset + count * 4) ? 'Input' : 'Guest',
        });
      }
      const next = positions();
      if (new Set(next).size !== count) throw Error('Agent occupancy invariant violated');
      return {
        input: terrain ? { moves: [...moves], edit } : [...moves],
        moves, editCell, targets, positions: next, returned, simulationMs, changes,
        materials: terrain ? next.map((_, i) => read(1344 + 4 * i))
          : (shared ? [read(16), read(20)] : [read(8)]),
        terrain: terrain ? initial.materials.map((_, i) => read(576 + 4 * i)) : initial.materials,
        conflict: terrain ? next.some((_, i) => read(1600 + 4 * i) > 0) : (shared && read(24) === 1),
        pageHash: sha256(after),
      };
    },
  };
}

export function matchesProof(session, rows, proof) {
  if (session.mode !== proof.session.mode || session.version !== proof.session.version
    || session.binary_sha256 !== proof.session.binary_sha256 || !rows.length
    || JSON.stringify(session.initial.positions) !== JSON.stringify(proof.session.initial.positions)
    || JSON.stringify(session.initial.materials) !== JSON.stringify(proof.session.initial.materials)
    || rows.length > proof.ticks.length) return false;
  return rows.every((row, index) =>
    JSON.stringify(row.input) === JSON.stringify(proof.session.inputs[index])
    && row.hash === proof.ticks[index].page_sha256);
}
