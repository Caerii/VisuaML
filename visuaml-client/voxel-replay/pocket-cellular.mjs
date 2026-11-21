import { passage, SparseWorld } from './cellular-automata.mjs';
import regionSixUrl from './evidence/pocket-cellular-automata/region-halo-6.json?url';
import regionFiveUrl from './evidence/pocket-cellular-automata/region-halo-5.json?url';
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

get('region-six').href = regionSixUrl;
get('region-five').href = regionFiveUrl;
get('halo').onclick = async () => {
  try {
    const response = await fetch(evidenceUrl);
    if (!response.ok) throw Error('Could not load the saved halo comparison');
    const { halo } = await response.json();
    const six = halo.cropped['6'], five = halo.cropped['5'];
    const first = five.first_different_tick;
    get('halo-status').textContent =
      `Target (${halo.target.join(', ')}) first differs at tick ${first}. Six-cell halo: ${six.target_with_edit[first]}. Five-cell halo: ${five.target_with_edit[first]}. Gate left closed: ${six.target_without_edit[first]}.`;
    get('dependency').replaceChildren(...halo.dependency.map(({cell, tick}) => {
      const row = document.createElement('li');
      row.textContent = `Tick ${tick} · cell (${cell.join(', ')})${tick === 0 ? ' · outside the five-cell halo' : ''}`;
      return row;
    }));
  } catch (error) {
    get('halo-status').textContent = error.message;
  }
};
