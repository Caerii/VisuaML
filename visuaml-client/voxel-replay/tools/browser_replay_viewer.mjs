import assert from 'node:assert/strict';
const { chromium } = await import(process.env.PLAYWRIGHT_CORE ?? 'playwright-core');
const browser = await chromium.launch({ executablePath: '/usr/bin/chromium', headless: true, args: ['--no-sandbox'] });
try {
  const page = await browser.newPage();
  await page.goto('http://localhost:4180/replay-viewer.html');
  await page.getByText(/First difference after PC 160/).waitFor();
  await page.getByRole('button', { name: 'Open Pocket checkpoint' }).click();
  await page.getByText(/call depth 1/).waitFor();
  await page.locator('#file').setInputFiles('release/examples/terrain-comparison.json');
  await page.getByText(/First difference after PC 160/).waitFor();
  await page.route('**/api/compiler-counterexample', route => route.abort());
  await page.getByRole('button', { name: 'Search terrain inputs' }).click();
  await page.getByText(/Saved search.*PC 100.*\[256\] → \[0\]/).waitFor();
  const downloaded = page.waitForEvent('download');
  await page.getByRole('link', { name: 'Download replayable counterexample' }).click();
  assert.equal((await downloaded).suggestedFilename(), 'terrain-counterexample.json');
} finally { await browser.close(); }
