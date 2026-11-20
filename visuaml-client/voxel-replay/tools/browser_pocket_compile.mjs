import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const { chromium } = await import(process.env.PLAYWRIGHT_CORE ?? 'playwright-core');
const out = process.argv[2] ?? '/tmp/pocket-compile-captures';
await mkdir(out, { recursive: true });
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH ?? '/usr/bin/chromium', headless: true, args: ['--no-sandbox'],
});
const read = async path => JSON.parse(await readFile(new URL(path, import.meta.url)));
try {
  const page = await browser.newPage({ viewport: { width: 1100, height: 1120 } });
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  await page.goto(process.env.POCKET_URL ?? 'http://localhost:4180/pocket.html?cartridge=compile-monitor');
  await page.getByText('Browser WASM playback', { exact: true }).waitFor();
  const shot = async i => page.screenshot({ path: join(out, `compile-${String(i).padStart(2, '0')}.png`) });
  let frame = 0;
  await shot(frame++);
  for (const line of ['10 SET 7', '20 ADD 6', '30 PRINT', '40 HALT', 'COMPILE']) {
    for (const char of line) {
      await page.keyboard.type(char);
      await shot(frame++);
    }
    await page.keyboard.press('Enter');
    await shot(frame++);
  }
  await page.locator('#module-line').selectOption({ label: '20 ADD 6' });
  await shot(frame++);
  await page.locator('#module-run').click();
  await page.getByText(/13 · matches saved VisuaML execution/).waitFor();
  assert.equal(await page.locator('#module-effect').innerText(), 'Accumulator 7 → 13');
  assert.match(await page.locator('#module-instructions').innerText(), /i32.const 6/);
  const downloaded = page.waitForEvent('download');
  await page.locator('#module-save').click();
  const download = await downloaded;
  assert.equal(download.suggestedFilename(), 'pocket-program.wasm');
  await shot(frame++);

  const names = ['add-5', 'add-6', 'overflow', 'malformed', 'max-size'];
  const proofs = await Promise.all(names.map(name => read(`../fixtures/pocket-compile-${name}.proof.json`)));
  const manifest = await read('../cartridges/compile-monitor.json');
  const verified = await page.evaluate(async ({ proofs, manifest }) => {
    const { createPocket, digest } = await import('/pocket.mjs');
    const { inspectModule, runEmitted } = await import('/pocket-module.mjs');
    const raw = await (await fetch(`/guest/${manifest.binary}`)).arrayBuffer();
    const text = page => {
      const cells = new DataView(page.buffer);
      return Array.from({length: 18}, (_, y) => Array.from({length: 20}, (_, x) =>
        String.fromCharCode(cells.getInt32(512 + 4*(20*y+x), true) || 32)).join('').trimEnd());
    };
    const cases = [];
    for (const proof of proofs) {
      const cart = await createPocket(raw, manifest.binary_sha256, manifest);
      for (const tick of proof.ticks) {
        const { frame, simulationPage } = cart.step(tick.button, undefined, true);
        const memory = cart.checkpoint();
        if (await digest(memory) !== tick.page_sha256
            || await digest(simulationPage) !== tick.phases[0].page_sha256
            || await digest(frame) !== tick.frame_sha256
            || JSON.stringify(text(memory)) !== JSON.stringify(tick.transcript)) throw Error('Browser replay diverged');
      }
      const emitted = inspectModule(cart.checkpoint());
      const result = emitted && await runEmitted(emitted.wasm);
      if (proof.module ? result?.binaryHash !== proof.module.binary_sha256
          || result?.pageHash !== proof.module.page_sha256 : result !== null) throw Error('Emitted module diverged');
      cases.push(proof.ticks.length);
    }
    return { program_ticks: cases };
  }, { proofs, manifest });
  await page.locator('#compile-6').click();
  await page.getByText(/^Guest emitted /).waitFor();
  await page.getByText('Matches saved VisuaML replay', { exact: true }).waitFor();
  await page.locator('#module-line').selectOption({ label: '20 ADD 6' });
  await page.locator('#module-run').click();
  await page.getByText(/13 · matches saved VisuaML execution/).waitFor();
  assert.deepEqual(errors, []);
  await writeFile(join(out, 'chromium.json'), JSON.stringify({ browser: browser.version(), cases: names, ...verified }, null, 2)+'\n');
} finally {
  await browser.close();
}
