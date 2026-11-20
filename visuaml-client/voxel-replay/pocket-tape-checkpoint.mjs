import { MAX_WORDS, TAPE_BASE } from './pocket-tape.mjs';

const PAGE_BYTES = 65536;

// Origin is host metadata. The guest sees the same head through a rebased index.
export function relocateTapePage(checkpoint, { left = 0, right = 0 } = {}) {
  const invalid = () => Error('Cannot extend tape checkpoint');
  if (checkpoint.byteLength !== PAGE_BYTES) throw invalid();
  const page = new DataView(checkpoint);
  const word = i => page.getInt32(TAPE_BASE + 4 * i, true);
  const [states, symbols, length, state, head, status] = Array.from({ length: 6 }, (_, i) => word(i));
  const tableWords = 3 * states * symbols;
  if (![left, right].every(x => Number.isSafeInteger(x) && x >= 0)
      || left + right < 1 || states < 1 || states > 256 || symbols < 1 || symbols > 256
      || length < 1 || state < 0 || state >= states || head < 0 || head >= length
      || status !== 0 || tableWords + length + left + right > MAX_WORDS) throw invalid();
  const start = TAPE_BASE + 4 * (8 + tableWords);
  const relocated = checkpoint.slice(0);
  const bytes = new Uint8Array(relocated);
  bytes.copyWithin(start + 4 * left, start, start + 4 * length);
  bytes.fill(0, start, start + 4 * left);
  bytes.fill(0, start + 4 * (left + length), start + 4 * (left + length + right));
  const live = new DataView(relocated);
  live.setInt32(TAPE_BASE + 8, length + left + right, true);
  live.setInt32(TAPE_BASE + 16, head + left, true);
  // Scratch is the previous transition's next-head index. The guest overwrites it before using it.
  if (left) live.setInt32(TAPE_BASE + 28, word(7) + left, true);
  return relocated;
}
