import discovered from './evidence/counterexample-search/terrain-session.json';
import counterexampleUrl from './evidence/counterexample-search/terrain-session.json?url';
import voxel from './release/examples/terrain-comparison.json';
import pocket from './release/examples/pocket-checkpoint.json';

const get = id => document.getElementById(id);
const profile = 'visuaml-direct-call-i32-v1';
function show(data) {
  if (data?.format_version !== 1 || data.profile !== profile) throw Error('Unsupported replay version');
  let status, detail;
  if (data.checkpoint?.registers) {
    const cp = data.checkpoint;
    const regs = cp.registers;
    status = `Checkpoint after source step ${data.retired_source_steps} · next PC ${regs.pc} · call depth ${cp.depth} · fuel ${regs.fuel}`;
    detail = { checkpoint_sha256: cp.checkpoint_sha256, identities: cp.identities,
      stack: regs.stack.slice(0, regs.height), locals: regs.locals };
  } else if ('first_difference' in data && data.reference && data.candidate) {
    const first = data.first_difference;
    status = first ? `First difference after PC ${first.reference.pc} · source step ${first.retired_source_steps}`
      : 'No difference in saved trace';
    detail = first ? { reference_stack: first.reference.stack, candidate_stack: first.candidate.stack,
      next_pc: first.reference.next_pc, fuel: first.reference.fuel,
      final_page_agrees: data.reference.page_sha256 === data.candidate.page_sha256 }
      : { compared_boundaries: data.reference.trace.length, final_page: data.reference.page_sha256 };
  } else throw Error('Unrecognized replay document');
  get('title').textContent = 'Saved VisuaML record';
  get('status').textContent = status;
  get('detail').textContent = JSON.stringify(detail, null, 2);
}
get('terrain').onclick = () => show(voxel);
get('pocket').onclick = () => show(pocket);
get('file').onchange = async event => {
  const file = event.target.files[0];
  if (!file) return;
  try { show(JSON.parse(await file.text())); }
  catch (error) {
    get('status').textContent = error.message;
    get('detail').textContent = '';
  }
};
show(voxel);

get('counterexample-link').href = counterexampleUrl;
get('find-counterexample').onclick = async () => {
  const button = get('find-counterexample');
  button.disabled = true;
  get('counterexample').textContent = 'Checking input prefixes…';
  let result = discovered.bad, label = 'Saved search';
  try {
    const response = await fetch('/api/compiler-counterexample', { method: 'POST' });
    if (response.ok && response.headers.get('content-type')?.includes('application/json')) {
      result = await response.json();
      label = 'Live search';
    }
  } catch {
    // Static viewers can display the saved counterexample.
  }
  const winner = result.winner, first = winner.first_difference;
  get('counterexample').textContent = `${label} · ${result.executed_prefixes.length} prefixes · input Right · PC ${first.pc} · [${first.reference.stack}] → [${first.candidate.stack}]`;
  button.disabled = false;
};
