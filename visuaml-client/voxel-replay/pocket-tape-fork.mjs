import { digest } from './pocket.mjs';
import { loadTape, inspectTape, MAX_WORDS, TAPE_BASE } from './pocket-tape.mjs';
import { relocateTapePage } from './pocket-tape-checkpoint.mjs';

const PAGE_BYTES = 65536;
const text = new TextEncoder();
const encodePage = page => btoa(String.fromCharCode(...new Uint8Array(page)));
const decodePage = value => Uint8Array.from(atob(value), c => c.charCodeAt(0)).buffer;
const word = (page, slot) => new DataView(page).getInt32(TAPE_BASE + 4 * slot, true);
const tableStart = TAPE_BASE + 32;

export async function tapeSession(wasm, fixture, graphHash) {
  const module = await WebAssembly.compile(wasm);
  const index = 8 + 3 * fixture.states * fixture.symbols;
  const tapeStart = TAPE_BASE + 4 * index;
  const boot = () => new WebAssembly.Instance(module).exports;
  const baseline = boot();
  loadTape(baseline.memory, { ...fixture, tape: [1], head: 0 });
  const identities = {
    wasm: await digest(wasm), graph: graphHash,
    program: await digest(text.encode(JSON.stringify(fixture))),
    table: await digest(baseline.memory.buffer.slice(tableStart, tapeStart)),
  };

  function describe(page, origin) {
    return { origin, length: word(page, 2), head: word(page, 4) + origin,
      state: word(page, 3), transitions: word(page, 6) };
  }

  function live(exports, origin) {
    return {
      origin,
      page: () => exports.memory.buffer.slice(0),
      current() {
        const state = inspectTape(exports.memory, index);
        return { ...state, head: origin + state.head };
      },
      step() {
        exports.run(0);
        return this.current();
      },
    };
  }

  return {
    identities,
    fresh(origin = 0, tape = [1]) {
      const exports = boot();
      loadTape(exports.memory, { ...fixture, tape, head: -origin });
      return live(exports, origin);
    },
    async exportCheckpoint(machine) {
      const page = machine.page();
      if (word(page, 5) !== 0) throw Error('Checkpoint requires a pre-attempt running page');
      return { version: 1, identities, logical: describe(page, machine.origin),
        pageHash: await digest(page), page: encodePage(page) };
    },
    async importCheckpoint(blob, { left = 0, right = 0, edit } = {}) {
      if (blob?.version !== 1) throw Error('Unsupported checkpoint version');
      for (const [key, expected] of Object.entries(identities)) {
        if (blob.identities?.[key] !== expected) throw Error(`Checkpoint ${key} mismatch`);
      }
      if (typeof blob.page !== 'string' || blob.page.length !== Math.ceil(PAGE_BYTES / 3) * 4) {
        throw Error('Invalid checkpoint page length');
      }
      let page = decodePage(blob.page);
      if (page.byteLength !== PAGE_BYTES || await digest(page) !== blob.pageHash) throw Error('Checkpoint page hash mismatch');
      const origin = blob.logical?.origin;
      const logical = describe(page, origin);
      if (!Number.isSafeInteger(origin) || !Number.isSafeInteger(origin + word(page, 2))
          || Object.entries(logical).some(([key, val]) => blob.logical?.[key] !== val)
          || word(page, 0) !== fixture.states || word(page, 1) !== fixture.symbols
          || logical.length < 1 || index + logical.length > MAX_WORDS + 8
          || logical.state < 0 || logical.state >= fixture.states
          || word(page, 4) < 0 || word(page, 4) >= logical.length || logical.transitions < 0
          || word(page, 5) !== 0) throw Error('Checkpoint logical metadata mismatch');
      if (await digest(page.slice(tableStart, tapeStart)) !== identities.table) throw Error('Checkpoint table bytes mismatch');
      if (left || right) page = relocateTapePage(page, { left, right });
      const newOrigin = origin - left;
      if (!Number.isSafeInteger(newOrigin) || !Number.isSafeInteger(newOrigin + word(page, 2))) {
        throw Error("Checkpoint origin exceeds safe coordinates");
      }
      if (edit) {
        const offset = edit.coordinate - newOrigin;
        if (!Number.isSafeInteger(offset) || offset < 0 || offset >= word(page, 2)
            || !Number.isInteger(edit.symbol) || edit.symbol < 0 || edit.symbol >= fixture.symbols) {
          throw Error('Checkpoint fork edit is out of bounds');
        }
        new DataView(page).setInt32(tapeStart + offset * 4, edit.symbol, true);
      }
      const exports = boot();
      new Uint8Array(exports.memory.buffer).set(new Uint8Array(page));
      return live(exports, newOrigin);
    },
  };
}
