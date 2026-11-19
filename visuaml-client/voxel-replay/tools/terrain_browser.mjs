import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const { chromium } = await import(process.env.PLAYWRIGHT_CORE ?? 'playwright-core');
const runDirectory = process.argv[2];
if (!runDirectory) throw Error('Pass the scale proof directory');
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH ?? '/usr/bin/chromium', headless: true,
  args: ['--no-sandbox', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
});
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const errors = [];
  page.on('pageerror', (error) => errors.push(String(error)));
  const click = (name) => page.getByRole('button', { name, exact: true }).click();
  const see = (text) => page.getByText(text, { exact: true }).waitFor();
  await page.goto(process.env.VOXEL_URL ?? 'http://localhost:4179');
  await click('Terrain Edit');
  await click('Play scene');
  await see('B · Cell 1');
  await see('Matches saved VisuaML proof');
  await click('Tick 1 · Edit');
  await see('Terrain edit · cell 1 cleared after movement');
  await see('B · Cell 2');
  await click('Tick 2');
  await see('B · Cell 1');
  await click('Replay without edit');
  await see('Blocked move · agent stayed');
  await see('material 7');
  await see('B · Cell 2');
  await see('Matches saved VisuaML proof');
  await click('Replay with edit');
  await see('B · Cell 1');
  await see('Matches saved VisuaML proof');
  await click('Sample timings');
  const metricsDownload = page.waitForEvent('download');
  await click('Export timings');
  await (await metricsDownload).saveAs(join(runDirectory, 'render.json'));
  await page.screenshot({ path: join(runDirectory, 'terrain.png') });
  const rendering = JSON.parse(await readFile(join(runDirectory, 'render.json'), 'utf8'));

  const proofs = [];
  for (const agents of [2, 4, 8]) {
    for (const density of ['sparse', 'crowded']) {
      const name = `${agents}-${density}`;
      proofs.push([name, JSON.parse(await readFile(join(runDirectory, `${name}.json`), 'utf8'))]);
    }
  }
  const cases = await page.evaluate(async (proofs) => {
    const { BASE, GUESTS, createWorld } = await import('/world.mjs');
    const raw = await (await fetch(GUESTS.terrain)).arrayBuffer();
    const report = {};

    for (const [name, proof] of proofs) {
      const world = await createWorld(raw, proof.session);
      for (const [index, input] of proof.session.inputs.entries()) {
        const result = world.step(input);
        if (await result.pageHash !== proof.ticks[index].page_sha256) {
          throw Error(`${name} diverged at tick ${index + 1}`);
        }
      }
      // Time the guest and host input writes in batches. Page hashing,
      // state inspection, and rendering are outside the timed interval.
      const setup = performance.now();
      const { instance } = await WebAssembly.instantiate(raw);
      const instantiateMs = performance.now() - setup;
      const { run, memory } = instance.exports;
      const view = new DataView(memory.buffer);
      const put = (offset, value) => view.setInt32(BASE + offset, value, true);
      const { initial, inputs } = proof.session;
      const count = initial.positions.length;
      put(0, count * 4);
      put(16, initial.materials.length * 4);
      initial.materials.forEach((material, i) => put(576 + i * 4, material));
      function batch(repeats) {
        const start = performance.now();
        for (let repetition = 0; repetition < repeats; repetition++) {
          initial.positions.forEach((position, i) => put(64 + i * 4, position * 4));
          for (const input of inputs) {
            put(4, Number(input.edit));
            input.moves.forEach((direction, i) => put(320 + i * 4, direction * 4));
            run(BASE);
          }
        }
        return (performance.now() - start) / (inputs.length * repeats);
      }
      batch(300);
      const batchMeanMs = Array.from({ length: 10 }, () => batch(1000));
      report[name] = { verifiedTicks: inputs.length, instantiateMs, warmTicks: inputs.length * 10_000,
                       batchMeanMs };
    }
    return report;
  }, proofs);
  assert.deepEqual(errors, []);
  await writeFile(join(runDirectory, 'chromium.json'), JSON.stringify({
    browser: browser.version(), renderer: 'headless SwiftShader', rendering, cases,
  }, null, 2) + '\n');
  console.log('Chromium verified both terrain histories and every scaling tick');
} finally {
  await browser.close();
}
