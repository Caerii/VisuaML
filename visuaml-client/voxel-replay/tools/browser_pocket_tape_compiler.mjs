import assert from 'node:assert/strict';
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';

const { chromium } = await import(process.env.PLAYWRIGHT_CORE ?? 'playwright-core');
const out = process.argv[2] ?? '/tmp/tape-compiler-captures';
await mkdir(out, { recursive: true });
const proof = JSON.parse(await readFile(new URL('../evidence/pocket-tape-compiler/replay.json', import.meta.url)));
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH ?? '/usr/bin/chromium',
  headless: true, args: ['--no-sandbox'] });
try {
  const page = await browser.newPage({ viewport: { width: 1100, height: 1060 } });
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  let i = 0;
  for (const [name, record] of Object.entries(proof.cases)) {
    await page.goto('http://localhost:4180/pocket.html?cartridge=tape-compiler');
    await page.getByText('Browser WASM playback', { exact: true }).waitFor();
    const shot = () => page.screenshot({ path: join(out, `tape-${String(i++).padStart(2, '0')}.png`) });
    for (const source of record.program) {
      await page.keyboard.type(source);
      await page.keyboard.press('Enter');
      if (name === 'erase') await shot();
    }
    if (!record.ticks) {
      assert(await page.locator('#compiled-tape-load').isDisabled());
      continue;
    }
    await page.locator('#compiled-tape-load').click();
    await page.getByText(/Line 40 STEP · call 0/).waitFor();
    if (name === 'erase') await shot();
    for (const tick of record.ticks) {
      await page.locator('#compiled-tape-step').click();
      const cells = tick.tape.map((symbol, index) => index === tick.head ? `[${symbol}]` : symbol).join('  ');
      await page.waitForFunction(cells => document.querySelector('#compiled-tape-cells').textContent === cells, cells);
      const expected = tick.stores[0];
      const message = expected ? `wrote byte ${expected.address}: ${expected.before} → ${expected.after} · returned ${tick.status}`
        : `returned ${tick.status}`;
      await page.getByText(new RegExp(`${message} · Matches saved VisuaML step`)).waitFor();
      if (name === 'erase' || name === 'keep' && expected?.address === 332) await shot();
    }
    assert(await page.locator('#compiled-tape-step').isDisabled());
  }
  assert.deepEqual(errors, []);
} finally {
  await browser.close();
}
