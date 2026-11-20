import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const { chromium } = await import(process.env.PLAYWRIGHT_CORE ?? 'playwright-core');
const out = process.argv[2] ?? '/tmp/pocket-resume-captures';
await mkdir(out, { recursive: true });
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH ?? '/usr/bin/chromium',
  headless: true, args: ['--no-sandbox'] });
try {
  const page = await browser.newPage({ viewport: { width: 1050, height: 850 } });
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  await page.goto('http://localhost:4180/pocket-universal.html');
  const status = page.locator('#resume-status');
  const advance = page.getByRole('button', { name: 'Advance both runs' });
  const shots = page.locator('section[aria-labelledby="resume-title"]');
  const shot = i => shots.screenshot({ path: join(out, `resume-${i}.png`) });
  await page.getByRole('button', { name: 'Start tape comparison' }).click();
  await status.getByText(/completed: 0/).waitFor();
  await shot(0);
  await advance.click();
  await status.getByText(/completed: 1.*Both pages match/).waitFor();
  await shot(1);
  await advance.click();
  await status.getByText(/Right edge reached.*matches VisuaML/).waitFor();
  await shot(2);
  await page.getByRole('button', { name: 'Extend and resume' }).click();
  await status.getByText(/Restored checkpoint and appended 3 blanks/).waitFor();
  await shot(3);
  for (let step = 2; step <= 4; step++) {
    await advance.click();
    await status.getByText(new RegExp(`completed: ${step}.*Both pages match`)).waitFor();
  }
  await shot(4);
  assert(await advance.isDisabled());
  assert.equal(await page.locator('#resume-short').innerText(), await page.locator('#resume-wide').innerText());
  assert.match(await page.locator('#resume-short').innerText(), /Tape 10100 · simulated head 2 · 131/);
  assert.deepEqual(errors, []);
  await writeFile(join(out, 'chromium.json'), JSON.stringify({ browser: browser.version(),
    checkpoint_match: true, compared_boundaries: 4, final_tape: '10100' }, null, 2)+'\n');
} finally {
  await browser.close();
}
