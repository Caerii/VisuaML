import onDemandUrl from './evidence/pocket-on-demand-trace/replay.json?url';
import pendingTraceUrl from './evidence/pocket-on-demand-trace/pending.json?url';
import sharedReplayUrl from './evidence/pocket-shared-recovery/replay.json?url';
import demandReplayUrl from './evidence/pocket-demand-region/replay.json?url';
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

get('demand-download').href = demandReplayUrl;
get('demand-load').onclick = async () => {
  try {
    const response = await fetch(demandReplayUrl);
    if (!response.ok) throw Error('Could not load the saved recovery replay');
    const archive = await response.json();
    const mode = get('demand-case').value;
    const failure = archive.unavailable[mode];
    if (failure) {
      const { coordinate, tick, source_pc: pc } = failure.pending_read;
      get('demand-status').textContent = `Pending PC ${pc} · (${coordinate.join(', ')}) at tick ${tick}`;
      get('demand-detail').textContent = failure.reason.replaceAll('_', ' ');
      return;
    }
    const row = archive.cases[mode].observations[mode === 'delayed' ? 4 : 0];
    const { coordinate, tick, source_pc: pc } = row.request;
    get('demand-status').textContent =
      `Resumed PC ${pc} · (${coordinate.join(', ')}) at tick ${tick} · action ${row.action}`;
    const recovery = row.recovery;
    get('demand-detail').textContent = recovery
      ? [`Recovered from tick 0 · ${recovery.source_cells} source cells · ${recovery.cells_recomputed} recomputed cell-ticks.`,
         ...recovery.applied_edits.map(edit => `Replayed gate write at PC ${edit.pc} before tick ${edit.tick}.`),
         ...(mode === 'delayed' ? ['Using tick 4 would return 0. The requested tick 5 returns 1.'] : [])].join('\n')
      : 'Observation was covered by the shared regional checkpoint.';
  } catch (error) {
    get('demand-status').textContent = error.message;
  }
};

get('shared-download').href = sharedReplayUrl;
get('shared-load').onclick = async () => {
  try {
    const response = await fetch(sharedReplayUrl);
    if (!response.ok) throw Error('Could not load the saved shared recovery replay');
    const archive = await response.json();
    const mode = get('shared-case').value;
    const fork = archive.strategies.dependencies[mode];
    const observations = fork.consumer.observations;
    const hits = observations.filter(row => row.recovery.hit).length;
    get('shared-status').textContent = `${hits} of ${observations.length} reads shared in this fork`;
    get('shared-detail').textContent = [
      ...observations.map(({ request, recovery, value }) => {
        const events = recovery.proof.relevant_events.length;
        return `Tick ${request.tick} · ${recovery.hit ? 'reuse' : 'recompute'} · value ${value} · ${events} relevant edit${events === 1 ? '' : 's'}`;
      }),
      ...(mode === 'shortcut_on' ? ['Rust read (2, 8) at tick 1 and opened the gate before tick 2. The materialized edit changed the cache key.'] : []),
    ].join('\n');
  } catch (error) {
    get('shared-status').textContent = error.message;
  }
};

get('trace-pending').href = pendingTraceUrl;
get('open-instructions').onclick = async () => {
  get('trace-status').textContent = 'Opening trace…';
  try {
    const saved = await fetch(onDemandUrl);
    if (!saved.ok) throw Error('Could not open the saved journal');
    const archive = await saved.json();
    let details = archive.reconstructed;
    let label = 'Saved reconstructed trace';
    try {
      const response = await fetch('/api/pocket-reconstruct', { method: 'POST' });
      if (response.ok && response.headers.get('content-type')?.includes('application/json')) {
        details = await response.json();
        label = 'Reconstructed on demand';
      }
    } catch {
      // Production bundles can display the saved reconstruction.
    }
    const destination = archive.consumer.observations.at(-1);
    get('trace-status').textContent = `${label} · remote LOAD at PC ${details.trace[0].pc}`;
    get('trace-detail').textContent = [
      ...details.trace.map(row => `PC ${row.pc} · ${row.opcode} → ${row.next_pc} · stack [${row.stack.join(', ')}]`),
      `Recorded action opened gate (${archive.gate.coordinate.join(', ')}). No world effects delivered by reconstruction.`,
      `Consumer tick ${destination.request.tick} · LOAD ${destination.value} · action ${destination.action}.`,
      'Saved reconstruction of the tick-6 consumer:',
      ...archive.consumer_reconstructed.trace.map(row =>
        `PC ${row.pc} · ${row.opcode} → ${row.next_pc} · stack [${row.stack.join(', ')}]`),
    ].join('\n');
  } catch (error) {
    get('trace-status').textContent = error.message;
  }
};
