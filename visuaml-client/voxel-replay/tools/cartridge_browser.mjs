import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const { chromium } = await import(process.env.PLAYWRIGHT_CORE ?? 'playwright-core');
const output = process.argv[2] ?? '/tmp/pocket-cartridge-captures';
await mkdir(output, { recursive: true });
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH ?? '/usr/bin/chromium', headless: true, args: ['--no-sandbox'],
});
try {
  const page = await browser.newPage({ viewport: { width: 1100, height: 825 } });
  const failures = [];
  page.on('pageerror', error => failures.push(String(error)));
  const root = process.env.POCKET_URL ?? 'http://localhost:4180/pocket.html';
  const open = async cartridge => {
    await page.goto(`${root}?cartridge=${cartridge}`);
    await page.getByText('Browser WASM playback', { exact: true }).waitFor();
  };
  const shot = name => page.screenshot({ path: join(output, name + '.png') });
  await open('passage');
  await page.locator('#dig').click();
  assert.equal(await page.locator('#terrain').innerText(), 'Open');
  await page.keyboard.press('ArrowRight');
  assert.equal(await page.locator('#agent').innerText(), 'Crossed');
  await page.locator('#cartridge').selectOption('rail-crate');
  await page.getByText('Browser WASM playback', { exact: true }).waitFor();
  await shot('rail-0');
  await page.keyboard.press('ArrowRight');
  await shot('rail-1');
  await page.keyboard.press('ArrowRight');
  await shot('rail-2');
  await page.keyboard.press('ArrowRight');
  await page.getByText('Blocked move · positions held', { exact: true }).waitFor();
  assert.equal(await page.locator('#crate').innerText(), '5');
  await shot('rail-3');
  await page.locator('#skip-push').click();
  await page.getByText('Matches saved VisuaML replay', { exact: true }).waitFor();

  await open('monitor');
  await shot('monitor-0');
  let i = 1;
  for (const command of ['STATUS', 'DIG', 'STEP', 'MAP']) {
    await page.keyboard.type(command);
    await shot(`monitor-${i++}`);
    await page.keyboard.press('Enter');
    await shot(`monitor-${i++}`);
  }
  assert.equal(await page.locator('#world-block').innerText(), 'Open');
  assert.equal(await page.locator('#world-agent').innerText(), 'Crossed');
  const samples = await page.evaluate(async () => {
    const field = document.querySelector('#terminal-keys');
    const rows = [];
    field.focus();
    for (const command of ['STATUS','MAP','STATUS','MAP','STATUS']) {
      for (const key of [...command, 'Enter']) {
        const started = performance.now();
        field.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }));
        const frameReadyMs = performance.now() - started;
        await new Promise(requestAnimationFrame);
        rows.push({ key, frameReadyMs, paintOpportunityMs: performance.now() - started });
      }
    }
    return rows;
  });
  const readJson = async path => JSON.parse(await readFile(new URL(path, import.meta.url)));
  const scripts = await Promise.all(['rail-crate-play','rail-crate-omitted','monitor-play','monitor-omitted'].map(async name => {
    const proof = await readJson(`../fixtures/${name}.proof.json`);
    const manifest = await readJson(`../cartridges/${proof.session.cartridge}.json`);
    return { manifest, proof, name };
  }));
  const verified = await page.evaluate(async scripts => {
    const { createPocket, digest } = await import('/pocket.mjs');
    const results = {};
    for (const { name, manifest, proof } of scripts) {
      const raw = await (await fetch('/guest/' + manifest.binary)).arrayBuffer();
      const machine = await createPocket(raw, manifest.binary_sha256, manifest);
      for (const tick of proof.ticks) {
        const output = machine.step(tick.button, undefined, true);
        if (await digest(output.simulationPage) !== tick.phases[0].page_sha256
            || await digest(machine.checkpoint()) !== tick.page_sha256
            || await digest(output.frame) !== tick.frame_sha256) throw Error('Replay mismatch ' + name);
        const expected = Uint8Array.from(atob(tick.frame_base64), c => c.charCodeAt(0));
        if (expected.some((value, i) => value !== output.frame[i])) throw Error('Frame byte mismatch ' + name);
        if (tick.transcript) {
          const mem = new DataView(machine.checkpoint().buffer);
          const rows = Array.from({length:18}, (_, row) => Array.from({length:20}, (_, col) =>
            String.fromCharCode(mem.getInt32(512 + 4*(20*row+col), true) || 32)).join('').trimEnd());
          if (JSON.stringify(rows) !== JSON.stringify(tick.transcript)) throw Error('Transcript mismatch');
        }
      }
      results[name] = proof.ticks.length;
    }
    return results;
  }, scripts);
  assert.deepEqual(failures, []);
  const summarize = key => {
    const values = samples.map(row => row[key]).sort((a,b) => a-b);
    return { median: values[Math.floor(values.length/2)], p95: values[Math.ceil(values.length*.95)-1] };
  };
  await writeFile(join(output, 'browser.json'), JSON.stringify({
    chromium: browser.version(), verified, keyboard: { frame_ready_ms: summarize('frameReadyMs'),
      paint_opportunity_ms: summarize('paintOpportunityMs'), raw: samples },
  }, null, 2) + '\n');
} finally {
  await browser.close();
}
