import { passage, SparseWorld } from './cellular-automata.mjs';
import evidenceUrl from './evidence/pocket-cellular-automata/replay.json?url';

const get = id => document.getElementById(id);
const scene = passage();
const makeWorld = () => new SparseWorld(scene.walls, scene.signal, scene.width, scene.height);
const palette = ['#c8d9a3', '#263f35', '#608c5c'];
let live, closed, tick, first, timer, queued;
get('download').href = evidenceUrl;

function draw(id, world) {
  const ctx = get(id).getContext('2d');
  world.walls.forEach((wall, i) => {
    const gate = i === scene.gate && wall;
    ctx.fillStyle = gate ? '#92b17b' : palette[wall ? 1 : world.signal[i] ? 2 : 0];
    ctx.fillRect((i % scene.width)*8, Math.floor(i/scene.width)*8, 8, 8);
  });
}

function show() {
  draw('live', live);
  draw('closed', closed);
  get('status').textContent = `Tick ${tick}${queued ? ' · gate edit queued' : ''}${first
    ? ` · first signal difference at tick ${first.tick}, cell (${first.cell % scene.width}, ${Math.floor(first.cell/scene.width)})`
    : ' · signal histories agree'}`;
}

function step() {
  if (queued) live.open(scene.gate);
  queued = false;
  live.tick();
  closed.tick();
  tick++;
  const cell = live.signal.findIndex((value, i) => value !== closed.signal[i]);
  if (!first && cell !== -1) first = { tick, cell };
  show();
}

function stop() {
  clearInterval(timer);
  timer = null;
  get('play').textContent = 'Play';
}

function reset() {
  stop();
  live = makeWorld();
  closed = makeWorld();
  tick = 0;
  first = null;
  queued = false;
  show();
}
get('dig').onclick = () => { queued = true; show(); };
get('step').onclick = step;
get('reset').onclick = reset;
get('play').onclick = () => {
  if (timer) stop();
  else {
    timer = setInterval(step, 220);
    get('play').textContent = 'Pause';
  }
};
window.addEventListener('keydown', event => {
  if (event.target.tagName === 'BUTTON') return;
  if (event.key.toLowerCase() === 'd') get('dig').click();
  if (event.key === 'ArrowRight') step();
});
reset();
