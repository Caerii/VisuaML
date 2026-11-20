import assert from 'node:assert/strict';
import { mkdir, writeFile, readFile } from 'node:fs/promises';
import { join } from 'node:path';

const { chromium } = await import(process.env.PLAYWRIGHT_CORE ?? 'playwright-core');
const out = process.argv[2] ?? '/tmp/pocket-rust-captures';
await mkdir(out, { recursive: true });
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH ?? '/usr/bin/chromium',
  headless: true, args: ['--no-sandbox'] });
try {
  const page = await browser.newPage({ viewport: { width: 1100, height: 1160 } });
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  const cases = [];
  let i = 0;
  for (const cell of [2, 3]) {
    await page.goto('http://localhost:4180/pocket.html?cartridge=rust-call-monitor');
    await page.getByText('Browser WASM playback', { exact: true }).waitFor();
    const shot = () => page.screenshot({ path: join(out, `call-${String(i++).padStart(2, '0')}.png`) });
    if (cell === 3) await shot();
    for (const line of [`10 SET ${cell}`, '20 CALL', '30 PRINT', 'RUN', 'COMPILE']) {
      for (const char of line) {
        await page.keyboard.type(char);
        if (cell === 3) await shot();
      }
      await page.keyboard.press('Enter');
      if (cell === 3) await shot();
    }
    await page.locator('#module-line').selectOption({ label: '20 CALL' });
    await page.locator('#module-run').click();
    await page.getByText(`Browser linked WASM · ${cell - 2} · matches saved VisuaML execution`).waitFor();
    if (cell === 3) await shot();
    const messages = [];
    for (let step = 0; step < 30; step++) {
      await page.locator('#call-step').click();
      const message = await page.locator('#call-trace').innerText();
      messages.push(message);
      if (cell === 3 && /call 0|wrote .* byte 64|returned|PRINT →/.test(message)) await shot();
      if (message.includes('PRINT →')) break;
    }
    assert(messages.some(message => message.includes(`Rust passable(${cell})`)));
    assert(messages.some(message => message.includes(`returned ${cell - 2}`)));
    assert.equal(messages.some(message => message.includes('wrote 3 at byte 64')), cell === 3);
    assert(messages.at(-1).includes(`PRINT → ${cell - 2}`));
    const proof = JSON.parse(await readFile(new URL(`../fixtures/pocket-rust-cell-${cell}.proof.json`, import.meta.url)));
    await page.evaluate(async proof => {
      const { createPocket, digest } = await import('/pocket.mjs');
      const raw = await (await fetch('/guest/pocket_rust_call.wasm')).arrayBuffer();
      const cartridge = await createPocket(raw, proof.wasm_sha256, { keyboard: 'ascii' });
      for (const tick of proof.ticks) {
        const result = cartridge.step(tick.button, undefined, true);
        if (await digest(result.frame) !== tick.frame_sha256
            || await digest(result.simulationPage) !== tick.phases[0].page_sha256
            || await digest(cartridge.checkpoint()) !== tick.page_sha256) {
          throw Error(`Replay diverged at tick ${tick.tick}`);
        }
      }
    }, proof);
    cases.push({ cell, keys: proof.ticks.length, messages });
  }
  assert.deepEqual(errors, []);
  await writeFile(join(out, 'chromium.json'), JSON.stringify({ browser: browser.version(), cases }, null, 2) + '\n');
} finally {
  await browser.close();
}
