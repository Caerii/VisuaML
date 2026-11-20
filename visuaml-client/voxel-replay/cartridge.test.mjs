import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { createPocket, digest, shade } from './pocket.mjs';
import { inspectModule, runEmitted } from './pocket-module.mjs';

const readJson = async path => JSON.parse(await readFile(new URL(path, import.meta.url)));
for (const name of ['rail-crate-play', 'rail-crate-omitted', 'monitor-play', 'monitor-omitted', 'monitor-program-5', 'monitor-program-6', 'pocket-compile-add-5', 'pocket-compile-add-6',
  'pocket-compile-overflow', 'pocket-compile-malformed', 'pocket-compile-max-size']) {
  test(`${name} reproduces full memory and all framebuffer bytes`, async () => {
    const proof = await readJson(`./fixtures/${name}.proof.json`);
    const manifest = await readJson(`./cartridges/${proof.session.cartridge}.json`);
    const raw = await readFile(new URL(`./guest/${manifest.binary}`, import.meta.url));
    const machine = await createPocket(raw, manifest.binary_sha256, manifest);
    let final;
    for (const tick of proof.ticks) {
      final = machine.step(tick.button, undefined, true);
      assert.equal(await digest(final.simulationPage), tick.phases[0].page_sha256);
      const page = machine.checkpoint();
      assert.equal(await digest(page), tick.page_sha256);
      assert.deepEqual(final.frame, new Uint8Array(Buffer.from(tick.frame_base64, 'base64')));
      if (tick.transcript) {
        const words = new DataView(page.buffer);
        const rows = Array.from({ length: 18 }, (_, row) =>
          Array.from({ length: 20 }, (_, col) =>
            String.fromCharCode(words.getInt32(512 + (row * 20 + col) * 4, true) || 32)).join('').trimEnd());
        assert.deepEqual(rows, tick.transcript);
      }
    }
    if (name.startsWith('pocket-compile-')) {
      const emitted = inspectModule(machine.checkpoint());
      if (proof.module === null) assert.equal(emitted, null);
      else {
        const result = await runEmitted(emitted.wasm);
        assert.equal(result.binaryHash, proof.module.binary_sha256);
        assert.equal(result.pageHash, proof.module.page_sha256);
        assert.deepEqual(result.printed, proof.module.printed);
        assert.deepEqual(emitted.rows.map(({ line, start, end }) => ({ line, start, end })),
          proof.module.source_map.map(({ line, start, end }) => ({ line, start, end })));
        if (name === 'pocket-compile-add-6') {
          assert.equal(emitted.rows[1].source, 'ADD 6');
          assert.deepEqual(result.accumulators.slice(0, 2), [7, 13]);
        }
      }
    }
    if (name === 'rail-crate-play') assert.equal(shade(final.frame, 98, 70), 3);
    if (name === 'rail-crate-omitted') assert.equal(shade(final.frame, 82, 70), 3);
  });
}

test('overlapping raster moves preserve every published pixel and terminal row', async () => {
  const profile = await readJson('./evidence/pocket-compile/scroll-timings.json');
  const manifest = await readJson('./cartridges/compile-monitor.json');
  const paths = ['evidence/pocket-compile/compiler-baseline.wasm', `guest/${manifest.binary}`];
  const [baseline, shifted] = await Promise.all(paths.map(async (path, index) =>
    createPocket(await readFile(new URL(path, import.meta.url)),
      profile.wasm_sha256[index ? 'shift' : 'dirty'], manifest)));
  for (const key of profile.buttons) {
    assert.deepEqual(baseline.step(key).frame, shifted.step(key).frame);
    assert.deepEqual(baseline.checkpoint().slice(512, 512 + 1440),
      shifted.checkpoint().slice(512, 512 + 1440));
  }
});
