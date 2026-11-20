import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const { chromium } = await import(process.env.PLAYWRIGHT_CORE ?? 'playwright-core');
const out = process.argv[2] ?? '/tmp/pocket-program-captures';
await mkdir(out, { recursive: true });
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH ?? '/usr/bin/chromium', headless: true, args: ['--no-sandbox'],
});
try {
  const page = await browser.newPage({ viewport: { width: 1100, height: 850 } });
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  await page.goto(process.env.POCKET_URL ?? 'http://localhost:4180/pocket.html?cartridge=program-monitor');
  await page.getByText('Browser WASM playback', { exact: true }).waitFor();
  const shot = i => page.screenshot({ path: join(out, `program-${i}.png`) });
  const type = async line => {
    await page.keyboard.type(line);
    await page.keyboard.press('Enter');
  };
  await shot(0);
  await type('10 SET 7');
  await page.keyboard.type('20 ADD 5');
  await shot(1);
  await page.keyboard.press('Enter');
  await type('30 PRINT');
  await shot(2);
  await type('RUN');
  assert.equal(await page.locator('#accumulator').innerText(), '12');
  await shot(3);
  await page.keyboard.type('20 ADD 6');
  await shot(4);
  await page.keyboard.press('Enter');
  await type('RUN');
  assert.equal(await page.locator('#accumulator').innerText(), '13');
  await shot(5);

  const read = async path => JSON.parse(await readFile(new URL(path, import.meta.url)));
  const proofs = await Promise.all([5, 6].map(n => read(`../fixtures/monitor-program-${n}.proof.json`)));
  const manifest = await read('../cartridges/program-monitor.json');
  const profile = await read('../evidence/monitor-program/renderer.json');
  const verified = await page.evaluate(async ({ proofs, manifest, profile }) => {
    const { createPocket, digest } = await import('/pocket.mjs');
    const wasm = await (await fetch(`/guest/${manifest.binary}`)).arrayBuffer();
    const equal = (a, b) => a.length === b.length && a.every((value, i) => value === b[i]);
    const text = page => {
      const cells = new DataView(page.buffer);
      return Array.from({length: 18}, (_, y) => Array.from({length: 20}, (_, x) =>
        String.fromCharCode(cells.getInt32(512 + 4*(20*y+x), true) || 32)).join('').trimEnd());
    };
    const cases = [];
    for (const proof of proofs) {
      const machine = await createPocket(wasm, manifest.binary_sha256, manifest);
      for (const r of proof.ticks) {
        const output = machine.step(r.button, undefined, true);
        const memory = machine.checkpoint();
        if (await digest(memory) !== r.page_sha256
            || await digest(output.simulationPage) !== r.phases[0].page_sha256
            || !equal(output.frame, Uint8Array.from(atob(r.frame_base64), c => c.charCodeAt(0)))
            || !equal(text(memory), r.transcript)) throw Error('Typed program replay diverged');
      }
      cases.push(proof.ticks.length);
    }
    const variants = await Promise.all(['baseline','dirty'].map(async label =>
      createPocket(await (await fetch(`/evidence/monitor-program/${label}.wasm`)).arrayBuffer(),
        profile.wasm_sha256[label], manifest)));
    for (const button of profile.buttons) {
      const [a,b] = variants.map(engine => engine.step(button));
      if (!equal(a.frame,b.frame) || !equal(text(variants[0].checkpoint()), text(variants[1].checkpoint())))
        throw Error('Renderer-only replay diverged');
    }
    return { program_ticks: cases, renderer_comparison_ticks: profile.buttons.length };
  }, {proofs, manifest, profile});
  await page.locator('#program-5').click();
  await page.getByText('Matches saved VisuaML replay',{exact:true}).waitFor();
  assert.deepEqual(errors, []);
  await writeFile(join(out, 'chromium.json'), JSON.stringify({ browser: browser.version(), ...verified }, null, 2)+'\n');
} finally {
  await browser.close();
}
