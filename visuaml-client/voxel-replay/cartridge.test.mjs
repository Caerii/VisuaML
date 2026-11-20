import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { createPocket, digest, shade } from './pocket.mjs';

const readJson = async path => JSON.parse(await readFile(new URL(path, import.meta.url)));
for (const name of ['rail-crate-play', 'rail-crate-omitted', 'monitor-play', 'monitor-omitted']) {
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
    if (name === 'rail-crate-play') assert.equal(shade(final.frame, 98, 70), 3);
    if (name === 'rail-crate-omitted') assert.equal(shade(final.frame, 82, 70), 3);
  });
}
