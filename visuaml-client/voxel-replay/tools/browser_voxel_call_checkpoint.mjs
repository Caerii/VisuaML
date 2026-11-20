import assert from 'node:assert/strict';
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';

const { chromium } = await import(process.env.PLAYWRIGHT_CORE ?? 'playwright-core');
const out = process.argv[2] ?? '/tmp/voxel-call-checkpoint-captures';
await mkdir(out, { recursive: true });
const browser = await chromium.launch({ executablePath: '/usr/bin/chromium', headless: true, args: ['--no-sandbox'] });
try {
  const view = await browser.newPage({ viewport: { width: 1050, height: 690 } });
  const errors = [];
  view.on('pageerror', error => errors.push(String(error)));
  await view.goto('http://localhost:4180/voxel-call-checkpoint.html');
  const shot = i => view.screenshot({ path: join(out, `checkpoint-${i}.png`) });
  await view.getByText(/call depth 1/).waitFor();
  await view.getByText(/caller commits 0/).waitFor();
  await shot(0);
  for (let i = 0; i < 3; i++) await view.getByRole('button', { name: 'Step source instruction' }).click();
  await view.getByText(/Rust returned material 7/).waitFor();
  await shot(1);
  await view.getByRole('button', { name: 'Finish suspended tick' }).click();
  await view.getByText(/helper result 7 · caller commits 1/).waitFor();
  await shot(2);
  await view.getByRole('button', { name: 'Next tick · Walk' }).click();
  await view.getByText(/agent crossed/).waitFor();
  await shot(3);
  await view.getByRole('button', { name: 'Next tick · Wait' }).click();
  await view.getByText(/agent stayed put/).waitFor();
  await shot(4);
  const downloaded = view.waitForEvent('download');
  await view.getByRole('link', { name: 'Download checkpoint' }).click();
  const checkpoint = join(out, 'download.json');
  await (await downloaded).saveAs(checkpoint);
  assert.deepEqual(await readFile(checkpoint),
    await readFile(new URL('../evidence/voxel-call-checkpoint/checkpoint.json', import.meta.url)));
  assert.deepEqual(errors, []);
} finally {
  await browser.close();
}
