import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';

const { chromium } = await import(process.env.PLAYWRIGHT_CORE ?? 'playwright-core');
const out = process.argv[2];
if (!out) throw Error('Provide a capture directory');
await mkdir(out, { recursive: true });
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH ?? '/usr/bin/chromium',
  headless: true, args: ['--no-sandbox'],
});
try {
  const page = await browser.newPage({ viewport: { width: 1120, height: 890 } });
  await page.goto(process.env.POCKET_URL ?? 'http://localhost:4179/pocket.html');
  const see = text => page.getByText(text, { exact: true }).waitFor();
  const shot = name => page.screenshot({ path: join(out, name + '.png'), fullPage: true });
  await see('Browser WASM playback');
  await shot('live-0');
  await page.locator('#dig').click();
  await see('Block cleared · passage pixels changed');
  await shot('live-1');
  await page.locator('#step').click();
  await see('Agent crossed · two tiles changed');
  await shot('live-2');

  await page.locator('#edit').click();
  await page.getByRole('button', { name: 'Boot', exact: true }).click();
  await see('Matches saved VisuaML replay');
  await shot('replay-0');
  await page.getByRole('button', { name: 'Edit tick', exact: true }).click();
  await see('Matches saved VisuaML replay');
  await shot('replay-1');
  await page.getByRole('button', { name: 'Next tick', exact: true }).click();
  await see('Matches saved VisuaML replay');
  await shot('replay-2');
  await page.locator('#omit').click();
  await page.getByRole('button', { name: 'Edit omitted', exact: true }).click();
  await see('Matches saved VisuaML replay');
  await shot('replay-3');
  await page.getByRole('button', { name: 'Next tick', exact: true }).click();
  await see('Matches saved VisuaML replay');
  await shot('replay-4');
} finally {
  await browser.close();
}
