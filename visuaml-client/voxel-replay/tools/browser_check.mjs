import { join } from 'node:path';
import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';

const { chromium } = await import(process.env.PLAYWRIGHT_CORE ?? 'playwright-core');
const output = process.env.VOXEL_TEST_OUTPUT ?? '/tmp/visuaml-voxel-browser';
await mkdir(output, { recursive: true });
const artifact = (name) => join(output, name);
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH ?? '/usr/bin/chromium', headless: true,
  args: ['--no-sandbox', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  const waitText = (text) => page.getByText(text, { exact: true }).waitFor();
  const click = (name) => page.getByRole('button', { name, exact: true }).click();
  await page.goto(process.env.VOXEL_URL ?? 'http://localhost:4179');
  await page.locator('canvas').waitFor();
  await page.getByRole('button', { name: 'Play scene' }).waitFor();
  await page.waitForFunction(() => ![...document.querySelectorAll('button')].find(x => x.textContent === 'Play scene').disabled);
  await page.keyboard.press('ArrowRight');
  await waitText('A · Cell 1');
  await waitText('Matches saved VisuaML proof');
  await page.keyboard.press('ArrowRight');
  await waitText('Blocked move · agent stayed');
  await waitText('material 7');
  await waitText('Matches saved VisuaML proof');
  await click('Start');
  await waitText('A · Cell 0');
  await click('Tick 2');
  await waitText('A · Cell 1');
  const singleDownload = page.waitForEvent('download');
  await click('Export inputs');
  await (await singleDownload).saveAs(artifact('single-export.json'));
  await click('Two Agents');
  await page.waitForFunction(() => ![...document.querySelectorAll('button')].find(x => x.textContent === 'Play scene').disabled);
  await page.keyboard.press('ArrowRight');
  await page.keyboard.press('j');
  await click('Single step');
  await waitText('B · Cell 3');
  await page.keyboard.press('ArrowRight');
  await page.keyboard.press('j');
  await click('Single step');
  await waitText('Collision · both agents stayed');
  await waitText('A · Cell 1');
  await waitText('B · Cell 3');
  await waitText('Matches saved VisuaML proof');
  const outgoing = page.waitForEvent('download');
  await click('Export inputs');
  await (await outgoing).saveAs(artifact('shared-export.json'));
  await click('Replay inputs');
  await click('Pause');
  assert.equal(await page.getByText('TICK 00', { exact: true }).count(), 1);
  await click('Single step');
  await waitText('TICK 01');
  await click('Resume');
  await waitText('Collision · both agents stayed');
  await waitText('Matches saved VisuaML proof');
  await click('Sample timings');
  const perf = page.waitForEvent('download');
  await click('Export timings');
  await (await perf).saveAs(artifact('render.json'));
  await page.screenshot({path:artifact('voxel.png')});
  const samples = await page.evaluate(async () => {
    const { BASE, GUESTS } = await import('/world.mjs');
    const result = {};
    for (const mode of ['single', 'shared']) {
      const session = await (await fetch(`/fixtures/${mode}.json`)).json();
      const proof = await (await fetch(`/fixtures/${mode}.proof.json`)).json();
      const raw = await (await fetch(GUESTS[mode])).arrayBuffer();
      const start = performance.now();
      const { instance } = await WebAssembly.instantiate(raw);
      const instantiationMs = performance.now() - start;
      const { run, memory } = instance.exports;
      const shared = mode === 'shared';
      const state = new DataView(memory.buffer);
      const set = (offset, value) => state.setInt32(BASE + offset, value, true);
      session.initial.materials.forEach((value, index) => set((shared ? 32 : 16) + index * 4, value));
      const previous = [session.initial.positions, proof.ticks[0].positions];
      function batch(repeats) {
        const start = performance.now();
        for (let i = 0; i < repeats; i++) {
          for (let tick = 0; tick < 2; tick++) {
            set(0, previous[tick][0] * 4);
            set(shared ? 8 : 4, session.inputs[tick][0] * 4);
            if (shared) {
              set(4, previous[tick][1] * 4);
              set(12, session.inputs[tick][1] * 4);
            }
            run(BASE);
          }
        }
        return (performance.now() - start) / (repeats * 2);
      }
      batch(10_000);
      const samples = Array.from({ length: 10 }, () => batch(20_000));
      result[mode] = { instantiationMs, warmCalls: 400_000, batchMeanSimulationMs: samples };
    }
    return result;
  });
  const metrics = JSON.parse(await readFile(artifact('render.json'), 'utf8'));
  await writeFile(artifact('timings.json'), JSON.stringify({
    browser: browser.version(), renderer: 'headless SwiftShader', rendering: metrics, simulation: samples,
  }, null, 2) + '\n');
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({path:artifact('mobile.png'),fullPage:true});
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth), 390);
  await click('Reset');
  await click('Single step');
  await waitText('TICK 01');
  await waitText('Browser playback');
  assert.deepEqual(errors, []);
  console.log('Chromium matched both proofs, verified collision controls, exported sessions, and checked mobile overflow');
} finally {
  await browser.close();
}
