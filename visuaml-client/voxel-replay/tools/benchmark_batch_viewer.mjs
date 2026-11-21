/** Measure an already-loaded viewer importing a saved batch and selecting its next result. */
import { readFile, mkdtemp, rm } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { extname, resolve, sep } from 'node:path';
import { tmpdir } from 'node:os';
import { setTimeout as pause } from 'node:timers/promises';

const root = resolve('dist');
const directory = await mkdtemp(resolve(tmpdir(), 'visuaml-viewer-'));
const raw = JSON.parse(await readFile(process.argv[2], 'utf8'));
const payload = JSON.stringify(raw).replaceAll('<', '\\u003c');
const harness = `<script>
window.addEventListener('load', async () => {
  const data = ${payload};
  const file = document.querySelector('#file');
  const picker = document.querySelector('#candidate');
  const status = document.querySelector('#status');
  const samples = [];
  for (let i=0; i<21; i++) {
    const transfer = new DataTransfer();
    transfer.items.add(new File([JSON.stringify(data)], 'batch.json', {type:'application/json'}));
    file.files = transfer.files;
    const start = performance.now();
    await file.onchange({target: file});
    const loaded = performance.now();
    picker.value = 1;
    picker.onchange();
    if (!status.textContent.startsWith('First difference')) throw Error(status.textContent);
    const switched = performance.now();
    if (i) samples.push({import_ms: loaded-start, select_ms: switched-loaded});
  }
  const output = document.createElement('pre');
  output.id = 'viewer-benchmark';
  output.textContent = JSON.stringify(samples);
  document.body.append(output);
});
</script>`;
const html = (await readFile(resolve(root, 'replay-viewer.html'), 'utf8')).replace('</body>', harness+'</body>');
const mime = {'.js':'text/javascript', '.mjs':'text/javascript', '.css':'text/css', '.html':'text/html', '.json':'application/json'};
const server = createServer(async (request, response) => {
  const path = new URL(request.url, 'http://127.0.0.1').pathname;
  const resolved = resolve(root, '.'+path);
  try {
    if (!resolved.startsWith(root+sep)) throw Error('Outside site root');
    response.setHeader('Content-Type', mime[extname(resolved)] ?? 'application/octet-stream');
    response.end(path === '/replay-viewer.html' ? html : await readFile(resolved));
  } catch {
    response.statusCode = 404;
    response.end();
  }
});
let chrome;
try {
  await new Promise(resolveListen => server.listen(0, '127.0.0.1', resolveListen));
  const url = `http://127.0.0.1:${server.address().port}/replay-viewer.html`;
  const args = ['--headless', '--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu',
    `--user-data-dir=${directory}`, '--remote-debugging-port=0', '--no-proxy-server', url];
  chrome = spawn(process.env.CHROMIUM ?? 'chromium', args, {stdio:'ignore'});
  const deadline = Date.now()+15000;
  let tab = null;
  while (!tab && Date.now() < deadline) {
    try {
      const port = (await readFile(resolve(directory, 'DevToolsActivePort'), 'utf8')).split('\n')[0];
      const tabs = await fetch(`http://127.0.0.1:${port}/json/list`).then(response => response.json());
      tab = tabs.find(item => item.url === url);
    } catch {
      // Chromium can take a moment to publish its local debugging endpoint.
    }
    if (!tab) await pause(50);
  }
  if (!tab) throw Error('Could not open viewer in Chromium');
  await pause(1000);
  const socket = new WebSocket(tab.webSocketDebuggerUrl);
  try {
    await new Promise((ready, failed) => {
      socket.addEventListener('open', ready, {once:true});
      socket.addEventListener('error', failed, {once:true});
    });
    const result = await new Promise((done, failed) => {
      const timer = setTimeout(() => failed(Error('Viewer did not produce a measurement')), 10000);
      socket.addEventListener('message', event => {
        const message = JSON.parse(event.data);
        if (message.id !== 1) return;
        clearTimeout(timer);
        const value = message.result?.result?.value;
        if (typeof value === 'string') done(value);
        else failed(Error(JSON.stringify(message.result?.exceptionDetails ?? message)));
      });
      const expression = `new Promise(resolve => {
        const timer = setInterval(() => {
          const output = document.querySelector('#viewer-benchmark');
          if (!output) return;
          clearInterval(timer);
          resolve(output.textContent);
        }, 20);
      })`;
      socket.send(JSON.stringify({id:1, method:'Runtime.evaluate',
        params:{expression, awaitPromise:true, returnByValue:true}}));
    });
    console.log(result);
  } finally {
    socket.close();
  }
} finally {
  if (chrome && chrome.exitCode === null && chrome.signalCode === null) {
    const exited = new Promise(resolveExit => chrome.once('exit', resolveExit));
    if (chrome.kill()) await exited;
  }
  server.close();
  await rm(directory, {recursive:true, force:true, maxRetries:3});
}
