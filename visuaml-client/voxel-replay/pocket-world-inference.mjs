import replayUrl from './evidence/pocket-world-inference/replay.json?url';
import fixturesUrl from './evidence/pocket-world-inference/holdout.json?url';

const get = id => document.getElementById(id);
get('download').href = replayUrl;
get('checkpoint').href = fixturesUrl;

async function read(url, options) {
  const response = await fetch(url, options);
  if (!response.ok || !response.headers.get('content-type')?.includes('application/json')) {
    throw Error('Replay data unavailable');
  }
  return response.json();
}

function paint(packet, signal, open) {
  const ctx = get('world').getContext('2d');
  ctx.fillStyle = '#263f35';
  ctx.fillRect(0,0,160,144);
  const [ox,oy] = packet.origin;
  packet.walls.forEach((wall,i) => {
    const x=ox+i%packet.width, y=oy+Math.floor(i/packet.width);
    const gate=x===packet.gate[0]&&y===packet.gate[1];
    ctx.fillStyle = wall && !(gate&&open) ? '#263f35' : signal[i] ? '#608c5c' : '#c8d9a3';
    ctx.fillRect(x*8,y*8,8,8);
  });
  ctx.strokeStyle='#dc9236';
  ctx.strokeRect(packet.target[0]*8+.5,packet.target[1]*8+.5,7,7);
}

async function load() {
  const [archive,fixtures] = await Promise.all([read(replayUrl),read(fixturesUrl)]);
  let frame, history, caption;
  function draw() {
    const run = archive.cases[get('run').value];
    const packet = fixtures.worlds[run.scene].checkpoint;
    paint(packet,history[frame-1] ?? packet.signal,frame>0);
    const result = run.computed < 0 ? 'unavailable' : String(run.computed);
    const feedback = run.reported < 0 ? 'unavailable' : String(run.reported);
    get('prompt').textContent = archive.prompt;
    get('status').textContent = `${caption} · tick ${frame} of ${history.length}`;
    get('details').textContent = [
      'Recorded model requests:',
      ...run.requests.filter(row=>row.length).map(row=>row.join(' ')),
      `Forecast: ${result} · feedback to model: ${feedback}`,
      ...(run.first_affected ? [`First changed signal: (${run.first_affected.cell.join(', ')}) at tick ${run.first_affected.tick}.`] : []),
    ].join('\n');
    get('step').disabled = frame>=history.length;
    get('open').disabled = !run.world_frames[0].length;
  }
  function reset() {
    const run = archive.cases[get('run').value];
    frame = 0;
    caption = 'Recorded forecast';
    history = run.world_frames[0];
    draw();
  }
  get('run').onchange=reset;
  get('step').onclick=()=>{ frame++; draw(); };
  get('open').onclick=async()=>{
    const selected=get('run').value;
    const run=archive.cases[selected];
    try {
      const result=await read(`/api/pocket-world-reconstruct/${run.scene}`,{method:'POST'});
      if (selected!==get('run').value) return;
      history=result.world_frames;
      caption='Reconstructed forecast';
    } catch {
      if (selected!==get('run').value) return;
      history=run.world_frames[0];
      caption='Saved forecast';
    }
    frame=history.length;
    draw();
  };
  reset();
}
load().catch(error=>{ get('status').textContent=error.message; });
