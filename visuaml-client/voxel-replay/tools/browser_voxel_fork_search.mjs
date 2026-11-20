import assert from 'node:assert/strict';
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';

const { chromium } = await import(process.env.PLAYWRIGHT_CORE ?? 'playwright-core');
const out = process.argv[2] ?? '/tmp/voxel-fork-search-captures';
await mkdir(out, { recursive: true });
const browser = await chromium.launch({ executablePath: '/usr/bin/chromium', headless: true, args: ['--no-sandbox'] });
try {
  const page = await browser.newPage({ viewport: { width: 1050, height: 850 } });
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  await page.goto('http://localhost:4180/voxel-call-checkpoint.html');
  const shot = i => page.screenshot({ path: join(out, `fork-search-${i}.png`) });
  await shot(0);
  await page.getByRole('button', { name: 'Find a way through' }).click();
  await page.getByText(/Live VisuaML search · 32 branches · 62 shared-prefix ticks/).waitFor();
  await shot(1);
  await page.getByRole('button', { name: 'Open winning write' }).click();
  await page.getByText(/Winning tick 1 · i32.store at PC 28 · byte 65520: 0 → 1/).waitFor();
  assert.equal(await page.locator('#far').getAttribute('aria-current'), '');
  await shot(2);
  const download = page.waitForEvent('download');
  await page.getByRole('link', { name: 'Download winning session' }).click();
  await (await download).saveAs(join(out, 'winning-session.json'));
  const read = async path => JSON.parse(await readFile(path, 'utf8'));
  assert.deepEqual(await read(join(out, 'winning-session.json')),
    await read(new URL('../evidence/voxel-fork-search/winning-session.json', import.meta.url)));
  await page.getByRole('button', { name: 'Open first compiler difference' }).click();
  await page.getByText('First difference after local.get at PC 160 · reference stack [256] · rejected plan [0]').waitFor();
  await shot(3);
  await page.getByRole('button', { name: 'Compare region exit' }).click();
  await page.getByText('Region exits agree after local.get at PC 161 · reference stack [256, 0] · rejected plan [256, 0]').waitFor();
  await shot(4);
  const artifact = page.waitForEvent('download');
  await page.getByRole('link', { name: 'Download compiler session' }).click();
  await (await artifact).saveAs(join(out, 'compiler-session.json'));
  assert.deepEqual(await read(join(out, 'compiler-session.json')),
    await read(new URL('../evidence/shared-fork-search/compiler-session.json', import.meta.url)));
  assert.deepEqual(errors, []);
} finally {
  await browser.close();
}
