import { extendUniversal, loadUniversal } from './pocket-universal.mjs';

export async function startResumeComparison(wasm, table, fixture) {
  const args = [wasm, table, fixture.program, fixture.input];
  let short = await loadUniversal(...args, fixture.shortBlanks);
  const wide = await loadUniversal(...args, fixture.longBlanks);
  let checkpoint;
  let boundary = 0;
  let expanded = false;
  return {
    current: () => ({ short: short.current(), wide: wide.current(), boundary, expanded }),
    pages: () => [short.page(), wide.page()],
    checkpoint: () => checkpoint?.slice(0),
    advance() {
      const prior = short.page();
      const left = short.step();
      if (left.status === 2 && !expanded) {
        checkpoint = prior;
        return { stopped: true, short: left, wide: wide.current(), boundary };
      }
      const right = wide.step();
      if (left.count !== right.count || left.phase !== right.phase || left.status !== right.status
          || left.head !== right.head || left.encoded.some((symbol, i) => symbol !== right.encoded[i])) {
        throw Error('The two tape runs diverged');
      }
      if (left.phase === 'boundary' || left.phase === 'halt') boundary++;
      return { stopped: false, short: left, wide: right, boundary };
    },
    async resume() {
      if (!checkpoint || expanded) throw Error('No held checkpoint');
      short = await extendUniversal(wasm, table, checkpoint, fixture.longBlanks - fixture.shortBlanks);
      expanded = true;
    },
  };
}
