import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const { chromium } = await import(process.env.PLAYWRIGHT_CORE ?? 'playwright-core');
const out = process.argv[2] ?? '/tmp/pocket-tape-captures';
await mkdir(out, { recursive: true });
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH ?? '/usr/bin/chromium',
  headless: true, args: ['--no-sandbox'] });
try {
  const page = await browser.newPage({ viewport: { width: 1040, height: 720 } });
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  await page.goto('http://localhost:4180/pocket-tape.html');
  await page.getByText('Example has a saved VisuaML replay').waitFor();
  const expected = ['Halted', 'Halted', 'Tape boundary'];
  const ticks = [];
  let frame = 0;
  const shot = () => page.screenshot({ path: join(out, `tape-${frame++}.png`) });
  for (const [index, end] of expected.entries()) {
    await page.locator('#example').selectOption(String(index));
    if (index === 0) await shot();
    let steps = 0;
    while (await page.locator('#tape-step').isEnabled()) {
      assert(steps++ < 8);
      await page.locator('#tape-step').click();
      await page.getByText('Full page matches saved VisuaML step').waitFor();
      if (index === 0) await shot();
    }
    assert((await page.locator('#tape-state').innerText()).includes(end));
    ticks.push(steps);
  }
  await page.locator('#example').selectOption('0');
  await page.locator('#rules').fill('0 1 0 1 L');
  await page.locator('#load-tape').click();
  await page.locator('#tape-run').click();
  await page.getByText(/head 0.*Halted/).waitFor();
  await page.getByText('Custom rules · live browser WASM').waitFor();
  assert.deepEqual(errors, []);
  await writeFile(join(out,'chromium.json'), JSON.stringify({ browser: browser.version(), ticks, custom_rules: true }, null, 2)+'\n');
} finally {
  await browser.close();
}
