import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { createPocket, digest, FRAME_BYTES, shade } from './pocket.mjs';

const raw = await readFile(new URL('./guest/pocket.wasm', import.meta.url));
const readProof = async name => JSON.parse(await readFile(new URL(`./fixtures/pocket-${name}.proof.json`, import.meta.url)));

for (const name of ['edit', 'omitted']) {
  test(`Pocket ${name} agrees with frozen replay at both phase boundaries`, async () => {
    const proof = await readProof(name);
    const player = await createPocket(raw, proof.wasm_sha256);
    let previous;
    for (const tick of proof.ticks) {
      const result = player.step(tick.button, undefined, true);
      assert.equal(await digest(result.simulationPage), tick.phases[0].page_sha256);
      assert.equal(await digest(player.checkpoint()), tick.page_sha256);
      assert.deepEqual(result.frame, new Uint8Array(Buffer.from(tick.frame_base64, 'base64')));
      assert.equal(result.frame.byteLength, FRAME_BYTES);
      assert.equal(shade(result.frame, 34, 70), 1);
      assert.equal(shade(result.frame, 81, 59), 1);
      assert.equal(shade(result.frame, 159, 143), 0);
      assert.equal(shade(result.frame, 82, 70), name === 'omitted' || tick.tick === 0 ? 3 : tick.tick === 1 ? 0 : 2);
      assert.equal(shade(result.frame, 114, 70), name === 'edit' && tick.tick === 2 ? 0 : 2);
      if (previous) assert.equal(await digest(previous.frame), previous.digest);
      previous = { frame: result.frame, digest: tick.frame_sha256 };
    }
  });
}

test('Pocket reset redraws the blocked scene and ignores unknown buttons', async () => {
  const proof = await readProof('edit');
  const player = await createPocket(raw, proof.wasm_sha256);
  const first = player.step(0).frame;
  player.step(1);
  player.step(0);
  assert.throws(() => player.step(2), /Invalid Pocket button/);
  assert.deepEqual(player.step(-1).frame, first);
});
