import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const { chromium } = await import(process.env.PLAYWRIGHT_CORE ?? 'playwright-core');
const out = process.argv[2] ?? '/tmp/pocket-universal-captures';
await mkdir(out, { recursive: true });
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH ?? '/usr/bin/chromium',
  headless: true, args: ['--no-sandbox'] });
try {
  const page = await browser.newPage({ viewport: { width: 1050, height: 850 } });
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  await page.goto('http://localhost:4180/pocket-universal.html');
  await page.getByText('Example has a saved VisuaML replay').waitFor();
  let index = 0;
  const shot = () => page.screenshot({ path: join(out, `universal-${index++}.png`) });
  await shot();
  await page.getByRole('button', { name: 'Step universal machine' }).click();
  await page.getByText('Matches saved VisuaML page · transition 1', { exact: true }).waitFor();
  await shot();
  for (let boundary = 1; boundary <= 3; boundary++) {
    await page.getByRole('button', { name: 'Step simulated machine' }).click();
    await page.getByText(new RegExp(`Simulated boundary ${boundary} ·`)).waitFor();
    await page.getByText(/Matches saved VisuaML page/).waitFor();
    await shot();
  }
  assert.equal(await page.locator('#simulated span').nth(1).innerText(), '1');
  await page.locator('#preset').selectOption('1');
  await page.getByText('Example has a saved VisuaML replay').waitFor();
  for (let boundary = 1; boundary <= 3; boundary++) {
    await page.getByRole('button', { name: 'Step simulated machine' }).click();
    await page.getByText(new RegExp(`Simulated boundary ${boundary} ·`)).waitFor();
  }
  assert.equal(await page.locator('#simulated span').nth(1).innerText(), '0');
  await page.getByText(/Matches saved VisuaML page/).waitFor();
  await shot();
  await page.locator('#program').fill('R\nH');
  await page.getByRole('button', { name: 'Encode and reset' }).click();
  await page.getByText('Custom encoding · browser WASM').waitFor();
  await page.getByRole('button', { name: 'Step simulated machine' }).click();
  await page.getByText('Simulated boundary 1 · instruction 1 · head 1').waitFor();
  assert.deepEqual(errors, []);
  await writeFile(join(out, 'chromium.json'), JSON.stringify({ browser: browser.version(),
    micro_step: true, simulated_steps: true, both_tape_outcomes: true, custom_encoding: true }, null, 2)+'\n');
} finally {
  await browser.close();
}
