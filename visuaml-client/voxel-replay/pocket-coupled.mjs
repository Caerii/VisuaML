import replayUrl from './evidence/pocket-coupled-checkpoint/replay.json?url';
import digUrl from './evidence/pocket-coupled-checkpoint/checkpoint-dig.json?url';
import omittedUrl from './evidence/pocket-coupled-checkpoint/checkpoint-omitted.json?url';
import shortUrl from './evidence/pocket-coupled-checkpoint/checkpoint-short_halo.json?url';
import outsideUrl from './evidence/pocket-coupled-checkpoint/checkpoint-outside_read.json?url';

const get = id => document.getElementById(id);
const checkpointUrls = { dig: digUrl, omitted: omittedUrl, short_halo: shortUrl, outside_read: outsideUrl };
let tick = -1;
get('download').href = replayUrl;
get('step').disabled = true;
get('reset').disabled = true;

function draw(region, signal, open) {
  const ctx = get('grid').getContext('2d');
  const [ox, oy] = region.origin;
  ctx.fillStyle = '#263f35';
  ctx.fillRect(0, 0, 160, 144);
  region.walls.forEach((wall, i) => {
    const x = ox + i % region.width, y = oy + Math.floor(i / region.width);
    const gate = x === region.gate[0] && y === region.gate[1];
    ctx.fillStyle = wall && !(gate && open) ? (gate ? '#92b17b' : '#263f35')
      : signal[i] ? '#608c5c' : '#c8d9a3';
    ctx.fillRect(x*8, y*8, 8, 8);
  });
  ctx.strokeStyle = '#dc9236';
  ctx.strokeRect(region.target[0]*8+.5, region.target[1]*8+.5, 7, 7);
}

async function loadReplay() {
  const response = await fetch(replayUrl);
  if (!response.ok) throw Error('Could not load the verified replay');
  const archive = await response.json();

  function show() {
    const name = get('case').value;
    const { region, regional, prefix } = archive.cases[name];
    const edit = regional.gate_store;
    const observation = regional.observations[tick-1];
    const path = archive.signal_path[tick-1];
    const maxTick = regional.frames.length;
    get('checkpoint').href = checkpointUrls[name];
    get('checkpoint').download = `checkpoint-${name}.json`;
    get('step').disabled = tick >= maxTick;
    draw(region, regional.frames[tick-1] ?? region.signal, tick >= 0 && edit);
    let status, detail;
    if (tick < 0) {
      status = 'Suspended inside the Rust gate helper';
      detail = `Next PC ${prefix.at(-1).next_pc}. The first automaton tick has yet to run.`;
    } else if (tick === 0) {
      status = edit ? `Rust gate store at PC ${edit.pc}` : 'Rust left the gate closed';
      detail = `Observation request (${regional.observation_request.join(', ')}) written at PC ${regional.request_store.pc}.`;
    } else if (regional.violation?.tick === tick) {
      status = `Tick ${tick} · observation rejected`;
      detail = `Rust requested (${regional.violation.coordinate.join(', ')}), outside the retained region. No value was delivered.`;
    } else {
      status = `Tick ${tick} · target ${observation.value} · Rust action ${observation.action}`;
      detail = [
        ...(path && name === 'dig' ? [`Signal reaches (${path.cell.join(', ')}).`] : []),
        `Rust LOAD at PC ${observation.load.pc}, branch at PC ${observation.branch.pc} → ${observation.branch.next_pc}.`,
        observation.action_store ? `Action store at PC ${observation.action_store.pc}.` : 'Action remains 0.',
        ...(name === 'short_halo' && tick === 6 ? ['Full-world target was 1 at this tick.'] : []),
      ].join('\n');
    }
    get('status').textContent = status;
    get('instruction').textContent = detail;
  }

  function reset() { tick = -1; show(); }
  get('reset').disabled = false;
  get('reset').onclick = reset;
  get('case').onchange = reset;
  get('step').onclick = () => { tick++; show(); };
  reset();
}

loadReplay().catch(error => {
  get('status').textContent = error.message;
});
