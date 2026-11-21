import wordDecoder from './evidence/word-decoder/replay.json';
import wordDecoderUrl from './evidence/word-decoder/replay.json?url';
import executionLoop from './evidence/decoder-execution-loop/replay.json';
import executionLoopUrl from './evidence/decoder-execution-loop/replay.json?url';
import vocabulary from './evidence/vocabulary-tensor-assembly/replay.json';
import vocabularyUrl from './evidence/vocabulary-tensor-assembly/replay.json?url';
import paired from './evidence/paired-routing/replay.json';
import pairedUrl from './evidence/paired-routing/replay.json?url';
import causal from './evidence/causal-tensor-assembly/replay.json';
import causalUrl from './evidence/causal-tensor-assembly/replay.json?url';
import acquisition from './evidence/tool-acquisition/replay.json';
import acquisitionUrl from './evidence/tool-acquisition/replay.json?url';
import composed from './evidence/composed-inference/replay.json';
import composedUrl from './evidence/composed-inference/replay.json?url';
import toolFailure from './evidence/two-call-controller/failure.json';
import toolFailureUrl from './evidence/two-call-controller/failure.json?url';
import discovered from './evidence/counterexample-search/terrain-session.json';
import counterexampleUrl from './evidence/counterexample-search/terrain-session.json?url';
import voxel from './release/examples/terrain-comparison.json';
import pocket from './release/examples/pocket-checkpoint.json';

const get = id => document.getElementById(id);
const profile = 'visuaml-direct-call-i32-v1';
function show(data) {
  if (data?.profile === 'visuaml-word-decoder-v1') {
    const { cases, intervention } = data;
    const before = cases.countdown;
    get('title').textContent = 'Saved word decoder replay';
    get('status').textContent = `One payload byte changes countdown from ${before.states.length} to ${intervention.states.length} instructions`;
    get('detail').textContent = [
      `0xfffffffe + 3 → ${cases['wrap-one'].writes[0].value} · branch PC ${cases['wrap-one'].states[4][0]}`,
      `0xffffffff + 1 → ${cases['wrap-zero'].writes[0].value} · branch PC ${cases['wrap-zero'].states[4][0]}`,
      `Token ${intervention.changed_token}: first committed decrement ${before.writes[0].value} → ${intervention.writes[0].value}`,
      `Next add: ${before.states[5][2]} → ${intervention.states[5][2]}`,
      `Next branch PC: ${before.states[7][0]} → ${intervention.states[7][0]}`,
      `Checked all ${data.partial_store_lengths} partial store lengths.`,
    ].join('\n');
    return;
  }
  if (data?.profile === 'visuaml-decoder-execution-loop-v1') {
    const { taken, fallthrough } = data.cases;
    const changed = data.intervention;
    const write = taken.writes.find(event => event.step === data.holdout.intervention.source_step);
    get('title').textContent = 'Saved decoder execution replay';
    get('status').textContent = `Generated store commits at token ${write.token}`;
    get('detail').textContent = [
      `Store tokens: ${taken.tokens.slice(write.token - 6, write.token).join(' ')}`,
      `Change payload token ${changed.changed_token}: load ${taken.states[4][2]} → ${changed.states[4][2]}`,
      `Branch chooses PC ${taken.states[5][0]} → ${changed.next_pc}`,
      `Intervention then writes address ${changed.writes.at(-1).address} and halts.`,
      `Taken: ${taken.states.length} instructions · ${taken.tokens.length} tokens`,
      `Fallthrough: ${fallthrough.states.length} instructions · ${fallthrough.tokens.length} tokens`,
    ].join('\n');
    return;
  }
  if (data?.profile === 'visuaml-vocabulary-tensor-assembly-v1') {
    const run = data.cases.find(entry => entry.write_audit);
    const { store_tokens: [start, end], load_tokens: [loadStart, loadEnd],
      committed_read: read, replaced_write_read: replaced } = run.write_audit;
    const word = value => `0x${value.toString(16).padStart(8, '0')}`;
    get('title').textContent = 'Saved vocabulary-token replay';
    get('status').textContent = `Store commits at emitted token ${end}`;
    get('detail').textContent = [
      `Store tokens ${start + 1}–${end}: ${run.emitted_tokens.slice(start, end).join(' ')}`,
      `Following load tokens ${loadStart + 1}–${loadEnd}: ${run.emitted_tokens.slice(loadStart, loadEnd).join(' ')}`,
      `Committed store read: ${word(read)}. Replacing its payload changes the read to ${word(replaced)}.`,
      'All 20 partial store payload lengths remained invisible.',
      ...data.cases.map(entry => `${entry.name} · ${entry.source_steps} source steps · ${entry.emitted_tokens.length} generated tokens`),
    ].join('\n');
    return;
  }
  if (data?.profile === 'visuaml-paired-routing-v1' && data.cases?.substitute_at_boundary && data.sweep) {
    const { original, substitute_at_boundary: switched } = data.cases;
    const first = original.events[0].call;
    const next = run => {
      const [handle, value, count, , result] = run.events[1].call;
      return `handle ${handle}(${value}, ${count}) → ${result}`;
    };
    get('title').textContent = 'Saved paired routing replay';
    get('status').textContent = `Feedback ${first[5]} → ${switched.events[0].call[5]} switched the second handle`;
    get('detail').textContent = [
      `First tool computed ${first[4]} in both runs.`,
      `${next(original)} · ${next(switched)}`,
      ...Object.entries(data.cases).map(([name, run]) => run.first_failure
        ? `${name} · event ${run.first_failure.event} · ${run.first_failure.category}`
        : `${name} · matched expected replay`),
    ].join('\n');
    return;
  }
  if (typeof data?.decoder_sha256 === 'string' && Array.isArray(data.cases)) {
    get('title').textContent = 'Saved causal tensor assembly replay';
    const boundaryCount = data.cases.reduce((sum, run) => sum + run.generated_records, 0);
    get('status').textContent = `${data.cases.length} cases · ${boundaryCount} compared source boundaries`;
    get('detail').textContent = data.cases.map(run => [
      `${run.name} · ${run.generated_records} emitted records`,
      ...run.emitted.map(({ next_pc, live, write }, index) =>
        `  Step ${index + 1} · next PC ${next_pc}${live ? '' : ' · halt'}${write
          ? ` · M[${write[0]}] = 0x${write[1].toString(16).padStart(8, '0')}` : ''}`),
    ].join('\n')).join('\n\n');
    return;
  }
  if (data?.format_version === 1 && data.profile === 'visuaml-tool-acquisition-v1') {
    const normal = data.cases.original.events;
    const substitute = data.cases.substitution.events;
    get('title').textContent = 'Saved tool acquisition replay';
    get('status').textContent = `First intervention difference at event ${data.first_intervention_difference}`;
    get('detail').textContent = [
      `First result: ${normal[1]?.reported ?? "—"} → ${substitute[1]?.reported ?? "—"}`,
      `Second request: ${normal[2]?.tokens?.join(' ') ?? "—"} → ${substitute[2]?.tokens?.join(' ') ?? "—"}`,
      ...Object.entries(data.cases).map(([name, run]) => run.failure
        ? `${name} · event ${run.failure.event} · ${run.failure.category} · ${JSON.stringify(run.failure)}`
        : `${name} · matched expected replay`),
    ].join('\n');
    return;
  }
  if (data?.format_version === 1 && data.conditional?.substitute && data.first_difference) {
    const { conditional: runs, first_difference: first } = data;
    const left = first.expected[0], right = first.actual[0];
    get('title').textContent = 'Saved composed inference replay';
    get('status').textContent = `Feedback intervention at event ${first.event} · ${left.feedback} → ${right.feedback}`;
    get('detail').textContent = [
      data.holdout.cases.normal.prompt,
      `Executor returned ${left.computed} in both runs. The intervention changed the reported value.`,
      `Next request: ${runs.normal.events[2].tokens.join(' ')} → ${runs.substitute.events[2].tokens.join(' ')}`,
      `Answer: ${runs.normal.answer} → ${runs.substitute.answer}`,
      `Zero repeat: ${runs.invalid.events[3].feedback} before execution`,
      `Fuel cutoff: ${runs.exhausted.events[1].trace.length} executor steps, then ${runs.exhausted.events[1].feedback}`,
    ].join('\n');
    return;
  }
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

get('composed-failure').onclick = () => show(composed);
get('composed-download').href = composedUrl;

get('acquisition').onclick = () => show(acquisition);
get('acquisition-download').href = acquisitionUrl;

get('causal').onclick = () => show(causal);
get('causal-download').href = causalUrl;

get('paired-routing').onclick = () => show(paired);
get('paired-routing-download').href = pairedUrl;

get('vocabulary').onclick = () => show(vocabulary);
get('vocabulary-download').href = vocabularyUrl;

get('execution-loop').onclick = () => show(executionLoop);
get('execution-loop-download').href = executionLoopUrl;

get('word-decoder').onclick = () => show(wordDecoder);
get('word-decoder-download').href = wordDecoderUrl;
