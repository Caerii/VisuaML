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
