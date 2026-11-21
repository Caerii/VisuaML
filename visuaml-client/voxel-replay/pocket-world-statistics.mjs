import { denseTick } from './cellular-automata.mjs';
import plot from './evidence/pocket-world-statistics/decisions.svg?url';
import switchUrl from './evidence/pocket-world-statistics/switch.json?url';
import irrelevantUrl from './evidence/pocket-world-statistics/irrelevant.json?url';
import failureUrl from './evidence/pocket-world-statistics/failure.json?url';
import outcomes from './evidence/pocket-world-statistics/outcomes.json?url';

const get = id => document.getElementById(id);
const examples = { switch:switchUrl, irrelevant:irrelevantUrl, failure:failureUrl };
get('plot').src = plot;
get('outcomes').href = outcomes;
let selectedExample;

async function show() {
  const name = get('example').value;
  get('forecast').textContent = '';
  get('reconstruct').disabled = true;
  const response = await fetch(examples[name]);
  if (!response.ok) throw Error('Saved example unavailable');
  const example = await response.json();
  if (name !== get('example').value) return;
  selectedExample = example;
  get('example-link').href = examples[name];
  get('example-link').download = `world-${name}.json`;
  if (name === 'failure') {
    const entry = example.record.runs[example.mode][example.bit];
    get('status').textContent = `First wrong event ${entry.first_wrong.event} · input vocabulary`;
    get('detail').textContent = `${entry.prompt}\nUnsupported tokens: ${entry.first_wrong.tokens.join(' ')}`;
    return;
  }
  const conditions = name === 'switch' ? ['actual','swapped'] : ['actual','irrelevant'];
  get('status').textContent = `${example.record.family} · recorded model decisions`;
  get('detail').textContent = [
    example.record.runs.actual['0'].prompt,
    ...conditions.flatMap(mode=>Object.entries(example.record.runs[mode]).map(([bit,record])=>
      `${mode} · world ${bit} · feedback ${record.feedback[0][2]} · ${record.requests[1].join(' ')}`)),
  ].join('\n');
  get('reconstruct').disabled = false;
}
get('reconstruct').onclick = () => {
  const entry = selectedExample.fixture;
  const comparison = entry.worlds.map((scene,index)=>{
    const walls=Uint8Array.from(scene.walls);
    if (get('example').value === 'irrelevant') walls[0]^=1;
    walls[scene.gate]=0;
    let signal=Uint8Array.from(scene.signal);
    const target=scene.target[1]*scene.width+scene.target[0];
    const bits=[];
    for (let i=0;i<entry.horizon;i++) {
      signal=denseTick(walls,signal,scene.width,scene.height);
      bits.push(signal[target]);
    }
    const prior=selectedExample.record.runs[get('example').value==='irrelevant'?'irrelevant':'actual'][String(index)].forecast;
    const status=JSON.stringify(bits)===JSON.stringify(prior)?'matches':'differs';
    return `World ${index}: ${bits.join(' → ')} (${status})`;
  });
  get('forecast').textContent = `Reconstructed scalar forecasts · ${comparison.join(' · ')}`;
};

const load = () => show().catch(error=>{ get('status').textContent=error.message; });
get('example').onchange=load;
load();
