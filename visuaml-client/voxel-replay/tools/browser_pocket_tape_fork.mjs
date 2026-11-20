import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';

const { chromium } = await import(process.env.PLAYWRIGHT_CORE ?? 'playwright-core');
const out = process.argv[2] ?? '/tmp/pocket-fork-captures';
await mkdir(out, { recursive: true });
const browser = await chromium.launch({ executablePath: '/usr/bin/chromium', headless: true, args: ['--no-sandbox'] });
try {
  const url = 'http://localhost:4180/pocket-tape-fork.html';
  let page = await browser.newPage({ viewport: { width: 1050, height: 700 } });
  const errors = [];
  const shot = i => page.screenshot({ path: join(out, `fork-${i}.png`) });
  await page.goto(url);
  const downloaded = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export checkpoint' }).click();
  const file = join(out, 'checkpoint.json');
  await (await downloaded).saveAs(file);
  await page.getByText(/Checkpoint exported/).waitFor();
  await shot(0);
  await page.close();

  page = await browser.newPage({ viewport: { width: 1050, height: 700 } });
  page.on('pageerror', error => errors.push(String(error)));
  await page.goto(url);
  await page.getByText(/Browser linked WASM/).waitFor();
  await shot(1);
  await page.locator('#clean-file').setInputFiles(file);
  await page.locator('#fork-file').setInputFiles(file);
  await page.getByText(/Imported into fresh instance/).last().waitFor();
  await shot(2);
  await page.getByRole('button', { name: 'Run clean copy' }).click();
  await page.getByText(/Tape 112 · 5 committed transitions · Matches saved VisuaML/).waitFor();
  await shot(3);
  await page.getByRole('button', { name: 'Run edited copy' }).click();
  await page.getByText(/Tape 212 · 5 committed transitions · Matches saved VisuaML/).waitFor();
  await page.getByText('First divergence: committed transition 2').waitFor();
  await shot(4);
  assert.deepEqual(errors, []);
} finally { await browser.close(); }
