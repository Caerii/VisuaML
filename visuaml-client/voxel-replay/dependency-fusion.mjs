import campaignEvidence from "./evidence/incremental-optimization-v67/campaigns.json";
import campaignWall from "./evidence/incremental-optimization-v67/timings.json";
import migration from "./evidence/incremental-optimization-v67/migration.json";
import campaignProofUrl from "./evidence/incremental-optimization-v67/proofs.json?url";
import migrationUrl from "./evidence/incremental-optimization-v67/migration.json?url";
import structural from "./evidence/structural-decomposition-v66/inference.json";
import boundaryProof from "./evidence/structural-decomposition-v66/checkpoints.json";
import inferenceUrl from "./evidence/structural-decomposition-v66/inference.json?url";
import checkpointUrl from "./evidence/structural-decomposition-v66/checkpoints.json?url";
import evidence from "./evidence/dependent-synthesis-v65/outcomes.json";
import proofUrl from "./evidence/dependent-synthesis-v65/proofs.json?url";

const get = id => document.getElementById(id);
const layout = get("layout");
const variant = get("variant");
const graph = get("graph");
let lane = 1;

get("proofs").href = proofUrl;
for (const [i, item] of evidence.examples.entries()) {
  layout.add(new Option(`${item.task.replace("_", " ")} · ${item.keys.length} lanes · case ${item.case + 1}`, i));
}

function render() {
  const item = evidence.examples[Number(layout.value)];
  const mode = variant.value;
  const invalid = mode === "invalid";
  const one = item.checks.find(check => check.n === 1);
  const count = mode === "staged" ? one.steps[0] : one.steps[1];
  get("result").textContent = invalid ? `Rejected. The first changed output byte is ${item.incorrect_first_byte}.` :
    `Verified on the saved replay. One record retires ${count} source instructions.`;
  const nodes = [];
  item.keys.forEach((coefficient, index) => {
    if (index) {
      const arrow = document.createElement("span");
      arrow.textContent = "→";
      arrow.setAttribute("aria-hidden", "true");
      nodes.push(arrow);
    }
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `Lane ${index + 1}`;
    button.setAttribute("aria-pressed", String(index === lane));
    button.addEventListener("click", () => { lane = index; render(); graph.children[index * 2].focus(); });
    nodes.push(button);
  });
  graph.replaceChildren(...nodes);
  const a = item.keys[lane];
  const b = item.keys[(lane + item.keys.length - 1) % item.keys.length];
  const left = lane ? `lane ${lane}’s ${invalid ? "updated output" : "original value"}` :
    `the preceding record’s final lane’s ${invalid ? "updated output" : "original value"}, or zero on the first record`;
  get("effect").textContent = `Lane ${lane + 1} adds ${a} × its original value and ${b} × ${left}. It also saves the original value in scratch.`;
  get("scope").textContent = `The proof covers returns and the full page at completion. At fuel ${item.fuel_witness.fuel}, the two valid versions already differ at byte ${item.fuel_witness.first_byte}.`;
}

layout.addEventListener("change", () => { lane = 1; render(); });
variant.addEventListener("change", render);
render();


get("inference-download").href = inferenceUrl;
get("checkpoint-download").href = checkpointUrl;
const treeSelect = get("tree-case");
const edit = get("factor-edit");
for (const [i, tree] of structural.trees.entries()) {
  treeSelect.add(new Option(`${tree.layout} · ${tree.nodes} nodes`, i));
}

function showEdit() {
  const tree = structural.trees[Number(treeSelect.value)];
  edit.max = tree.events.length - 1;
  const position = Number(edit.value);
  const event = tree.events[position];
  const [a, b] = event.edge;
  get("message-summary").textContent =
    `Edit ${position + 1} of ${tree.events.length} changes the factor between nodes ${a} and ${b}. ` +
    `A local query at node ${event.queried} rebuilt ${event.cached_work} messages. ` +
    `Reading every marginal rebuilt ${event.all_work}. The exact space has ${event.count} assignments.`;
  get("message-list").replaceChildren(...event.invalidated.map(([src, dst]) => {
    const item = document.createElement("code");
    item.textContent = `${src} → ${dst}`;
    return item;
  }));
}

treeSelect.addEventListener("change", () => { edit.value = 0; showEdit(); });
edit.addEventListener("input", showEdit);
showEdit();

const boundary = get("record-boundary");
const records = boundaryProof.result.boundaries;
for (const [i, record] of records.entries()) {
  boundary.add(new Option(`After record ${record.record}`, i));
}
function showBoundary() {
  const record = records[Number(boundary.value)];
  get("checkpoint-result").textContent =
    `After record ${record.record}, the original carry is ${record.carry}. ` +
    `The staged and fused source-fuel cuts above have different pages from one another.`;
  get("checkpoint-page").textContent = `Full page SHA-256 ${record.page}`;
}
boundary.addEventListener("change", showBoundary);
showBoundary();


const cycle = get("cycle-case");
const method = get("cycle-method");
for (const [i, item] of structural.cycles.entries()) {
  cycle.add(new Option(`${item.family.replace("_", " ")} · ${item.nodes} nodes · case ${item.case + 1}`, i));
}
function showBags() {
  const item = structural.cycles[Number(cycle.value)];
  const plan = item.methods[method.value];
  get("cycle-result").textContent = plan.count === null
    ? `Induced width ${plan.width} exceeds the width-3 cap. Inference was refused.`
    : `Induced width ${plan.width}. Exact count ${plan.count}.`;
  get("cycle-bags").replaceChildren(...plan.bags.map((nodes, i) => {
    const li = document.createElement("li");
    const parent = plan.parents[i];
    li.textContent = `Eliminate ${nodes[0]}. Bag [${nodes.join(", ")}]. ${parent === null ? "Root bag." : `Parent bag ${parent + 1}.`}`;
    return li;
  }));
}
function cycleChanged() {
  const current = structural.cycles[Number(cycle.value)];
  method.replaceChildren(...Object.keys(current.methods).map(key => new Option(key.replace("_", " "),key)));
  method.value = "min_fill";
  showBags();
}
cycle.addEventListener("change", cycleChanged);
method.addEventListener("change", showBags);
cycleChanged();


get("campaign-proof").href = campaignProofUrl;
get("migration-download").href = migrationUrl;
const family = get("campaign");
const revision = get("revision");
const groups = [...new Set(campaignEvidence.revisions.map(row => `${row.family} /${row.width}`))];
for (const key of groups) {
  const [name, width] = key.split("/");
  family.add(new Option(`${name.trim() === "local" ? "Local edits" : "Global edits"} · ${width} lanes`, key));
}

function showRevision() {
  const [group, width] = family.value.split("/");
  const index = campaignEvidence.revisions.findIndex(row => row.family === group.trim() &&
    row.width === Number(width) && row.revision === Number(revision.value));
  const row = campaignEvidence.revisions[index];
  const wall = campaignWall[index];
  const reuse = row.analysis.incremental;
  const full = row.analysis.fresh;
  get("revision-summary").textContent = `Revision ${row.revision + 1}. Verified ${row.mode} lowering. ` +
    `${reuse.recomputed} dependence slots rebuilt, ${reuse.fresh} new checks and ${reuse.reused} reused. ` +
    `Full regeneration checked all ${full.fresh} obligations.`;
  const changed = reuse.invalidated.filter(([kind]) => kind === "source").map(([, i]) => i + 1);
  get("revision-slots").textContent = changed.length ? `Edited lanes ${changed.join(", ")}.` : "Initial revision.";
  const extra = wall.native ? (wall.native.break_even === null
    ? " No finite break-even was observed in this sample."
    : ` Sampled native break-even estimate ${wall.native.break_even.toLocaleString()} calls.`) : "";
  get("revision-runtime").textContent = `Measured proof pipeline ${(wall.timing.incremental.total_ns / 1e6).toFixed(2)} ms incremental and ${(wall.timing.fresh.total_ns / 1e6).toFixed(2)} ms fresh.` + extra;
}
family.addEventListener("change", () => { revision.value = 0; showRevision(); });
revision.addEventListener("input", showRevision);
showRevision();

const saved = migration.saved;
get("migration-summary").textContent = `The saved process captured record ${saved.state.record + 1} before committing outputs. ` +
  `A fresh process switched to reverse-order writes using the saved original carry and ${saved.state.fuel} remaining logical charges. ` +
  `Both schedules completed with page SHA-256 ${migration.expected[1]}.`;
