import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const { chromium } = await import(process.env.PLAYWRIGHT_CORE ?? 'playwright-core');
const out = process.argv[2] ?? 'evidence/pocket';
await mkdir(out, { recursive: true });
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH ?? '/usr/bin/chromium',
  headless: true, args: ['--no-sandbox', '--use-angle=swiftshader'],
});
try {
  const page = await browser.newPage({ viewport: { width: 1180, height: 850 } });
  const failures = [];
  page.on('pageerror', error => failures.push(String(error)));
  const url = process.env.POCKET_URL ?? 'http://localhost:4179/pocket.html';
  await page.goto(url);
  const see = text => page.getByText(text, { exact: true }).waitFor();
  await see('Browser WASM playback');
  await page.locator('#dig').click();
  await see('Block cleared · passage pixels changed');
  assert.equal(await page.locator('#agent').innerText(), 'Waiting');
  await page.locator('#step').click();
  await see('Agent crossed · two tiles changed');
  await page.locator('#edit').click();
  await see('Matches saved VisuaML replay');
  await page.getByRole('button', { name: 'Edit tick', exact: true }).click();
  await see('Block cleared · passage pixels changed');
  await see('Matches saved VisuaML replay');
  assert.match(await page.locator('#writes').innerText(), /simulation · PC.*\[268\]/);
  await page.screenshot({ path: join(out, 'edit.png') });
  await page.getByRole('button', { name: 'Next tick', exact: true }).click();
  await see('Agent crossed · two tiles changed');
  await see('Matches saved VisuaML replay');
  await page.screenshot({ path: join(out, 'crossed.png') });
  await page.locator('#omit').click();
  await page.getByRole('button', { name: 'Next tick', exact: true }).click();
  await see('Matches saved VisuaML replay');
  assert.equal(await page.locator('#terrain').innerText(), 'Blocked');
  assert.equal(await page.locator('#agent').innerText(), 'Waiting');
  await page.screenshot({ path: join(out, 'omitted.png') });

  // Run/pause and keyboard controls take the same live path.
  await page.locator('#reset').click();
  await see('Scene drawn');
  await page.keyboard.press('x');
  await see('Block cleared · passage pixels changed');
  await page.keyboard.press('ArrowRight');
  await see('Agent crossed · two tiles changed');
  await page.locator('#play').click();
  await page.getByRole('button', { name: 'Pause', exact: true }).click();
  const pausedTick = await page.locator('#tick').innerText();
  await page.waitForTimeout(110);
  assert.equal(await page.locator('#tick').innerText(), pausedTick);
  const logDownload = page.waitForEvent('download');
  await page.locator('#export').click();
  const exported = JSON.parse(await readFile(await (await logDownload).path()));
  assert.deepEqual(exported.buttons.slice(0, 3), [0, 1, 0]);

  const requestedDownload = page.waitForEvent('download');
  await page.locator('#checkpoint').click();
  const checkpoint = await requestedDownload;
  assert.equal((await readFile(await checkpoint.path())).length, 65536);

  const proofPaths = ['edit', 'omitted'].map(key => new URL(`../fixtures/pocket-${key}.proof.json`, import.meta.url));
  const proofs = await Promise.all(proofPaths.map(async path => JSON.parse(await readFile(path))));
  const measured = await page.evaluate(async proofs => {
    const { createPocket, createPresenter, digest, shade, PALETTE, POCKET_WASM, PERIOD_MS } = await import('/pocket.mjs');
    const raw = await (await fetch(POCKET_WASM)).arrayBuffer();
    const canvas = document.getElementById('display');
    const present = createPresenter(canvas);
    const report = {};
    for (const proof of proofs) {
      const player = await createPocket(raw, proof.wasm_sha256);
      let retained;
      for (const tick of proof.ticks) {
        const result = player.step(tick.button, present, true);
        const [update, page, frame] = await Promise.all([
          digest(result.simulationPage), digest(player.checkpoint()), digest(result.frame),
        ]);
        if (update !== tick.phases[0].page_sha256 || page !== tick.page_sha256 || frame !== tick.frame_sha256)
          throw Error(`Full-memory replay divergence on tick ${tick.tick}`);
        const expected = Uint8Array.from(atob(tick.frame_base64), char => char.charCodeAt(0));
        if (expected.some((byte, i) => byte !== result.frame[i])) throw Error('Framebuffer divergence');
        const painted = canvas.getContext('2d').getImageData(82, 70, 1, 1).data;
        const color = PALETTE[shade(expected, 82, 70)];
        if (color.some((component, i) => component !== painted[i])) throw Error('Canvas differs from guest bytes');
        if (retained && await digest(retained.frame) !== retained.hash) throw Error('Published frame aliases memory');
        retained = { frame: result.frame, hash: frame };
      }
    }
    const patterns = {
      passage_repeat: Array.from({ length: 12 }, () => [-1, 0, 1, 0, ...Array(16).fill(0)]).flat(),
      frequent_changes: Array.from({ length: 80 }, () => [-1, 1, 0]).flat(),
    };
    for (const [name, stream] of Object.entries(patterns)) {
      const player = await createPocket(raw, proofs[0].wasm_sha256);
      player.step(0, present);
      const warm = [];
      const frames = [];
      const epoch = performance.now();
      for (const [index, button] of stream.entries()) {
        const due = epoch + index * PERIOD_MS;
        let gap = due - performance.now();
        while (gap > 0) {
          await new Promise(resolve => setTimeout(resolve, Math.ceil(gap)));
          gap = due - performance.now();
        }
        // This is the same input, WASM and Canvas submission path as the player.
        const began = performance.now();
        const { timing, frame } = player.step(button, present);
        const submitted = performance.now();
        await new Promise(requestAnimationFrame);
        const opportunity = performance.now();
        warm.push({ ...timing, start_delay_ms: began - due, scheduled_ms: submitted - due,
          paint_opportunity_ms: opportunity - due });
        frames.push(frame);
      }
      // Compare every retained frame outside the paced timing loop.
      const expected = proofs[0].ticks.map(tick =>
        Uint8Array.from(atob(tick.frame_base64), char => char.charCodeAt(0)));
      let scene = 0;
      for (const [index, button] of stream.entries()) {
        if (button === -1) scene = 0;
        else if (scene === 1) scene = 2;
        else if (scene === 0 && button === 1) scene = 1;
        if (expected[scene].some((value, offset) => value !== frames[index][offset]))
          throw Error(`Paced framebuffer divergence at ${name} tick ${index}`);
      }
      const distribution = values => {
        const sorted = [...values].sort((a, b) => a - b);
        const middle = Math.floor(sorted.length / 2);
        return { median: (sorted[middle] + sorted[Math.ceil(sorted.length / 2) - 1]) / 2,
          p95: sorted[Math.ceil(.95 * sorted.length) - 1] };
      };
      report[name] = {
        samples: warm.length, verified_frames: frames.length, deadline_misses: warm.filter(row => row.scheduled_ms > PERIOD_MS).length,
        paint_opportunity_misses: warm.filter(row => row.paint_opportunity_ms > PERIOD_MS).length,
        phases_ms: Object.fromEntries(Object.keys(warm[0]).map(key => [key, distribution(warm.map(row => row[key]))])),
        raw: warm,
      };
    }
    return report;
  }, proofs);
  assert.deepEqual(failures, []);
  await writeFile(join(out, 'chromium-paced.json'), JSON.stringify({
    browser: browser.version(), verified_ticks: 6, verified_phase_boundaries: 12, ...measured,
  }, null, 2) + '\n');
  console.log('Chromium verified both Pocket histories and completed the paced canvas replay');
} finally {
  await browser.close();
}
