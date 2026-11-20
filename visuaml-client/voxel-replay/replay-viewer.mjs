import toolFailure from './evidence/two-call-controller/failure.json';
import toolFailureUrl from './evidence/two-call-controller/failure.json?url';
import discovered from './evidence/counterexample-search/terrain-session.json';
import counterexampleUrl from './evidence/counterexample-search/terrain-session.json?url';
import voxel from './release/examples/terrain-comparison.json';
import pocket from './release/examples/pocket-checkpoint.json';

const get = id => document.getElementById(id);
const profile = 'visuaml-direct-call-i32-v1';
function show(data) {
  if (data?.format_version === 1 && data.replay?.first_substitution_difference) {
    const { runs, first_substitution_difference: first } = data.replay;
    const left = first.expected[0], right = first.actual[0];
    get('title').textContent = 'Saved two-call replay';
    get('status').textContent = `First difference at event ${first.event} · ${left.feedback} → ${right.feedback}`;
    get('detail').textContent = [
      `Prompt: ${data.replay.prompt}`,
      `First tool: ${left.slot}(${left.arguments.join(', ')}) returned ${left.computed}`,
      `Second call: ${runs.normal.events[2].tokens.join(' ')} → ${runs.substitute.events[2].tokens.join(' ')}`,
      `Answer: ${runs.normal.answer} → ${runs.substitute.answer}`,
    ].join('\n');
    return;
  }
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

get('tool-failure').onclick = () => show(toolFailure);
get('tool-failure-download').href = toolFailureUrl;
