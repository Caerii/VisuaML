import replays from "./evidence/pocket-goal-planning/replays.json";
import examplesUrl from "./evidence/pocket-goal-planning/replays.json?url";
import outcomesUrl from "./evidence/pocket-goal-planning/outcomes.json?url";

const get = id => document.getElementById(id);
const actionName = ["Wait", "Open", "Neither action"];
const mode = get("mode");
const next = get("next");
const events = get("events");
let journal = [];
let position = 0;

get("examples").href = examplesUrl;
get("outcomes").href = outcomesUrl;

function advance() {
  if (position === journal.length) return;
  const event = document.createElement("li");
  event.textContent = journal[position++];
  events.append(event);
  next.disabled = position === journal.length;
}

function reset() {
  const { record } = replays.examples[mode.value];
  const run = record.runs[mode.value];
  get("prompt").textContent = `Get the agent through by tick ${record.horizon}. Prefer Wait if it succeeds.`;
  journal = run.calls.flatMap(call => {
    const request = `Controller requests ${actionName[call.action]} through tick ${call.horizon}.`;
    if (call.status !== 15) return [request, "Forecast unavailable. The graph supplies no value."];
    const cells = call.first_updated_signal_cells.map(([x, y]) => `(${x}, ${y})`).join(", ");
    return [request,
      `Verified forecast: target at ticks 1–${call.steps} = ${call.target_history.join(", ")}. Tick-one cell changes: ${cells || "none"}.`,
      call.raw === call.shown ? `Controller receives ${call.shown}.` :
        `Intervention gives the controller ${call.shown} in place of the computed ${call.raw}.`];
  });
  journal.push(run.outcome === 1 ? "Controller stops without a recommendation." :
    `Controller recommends ${actionName[run.action]}.`);
  position = 0;
  events.replaceChildren();
  next.disabled = false;
  advance();
}

mode.addEventListener("change", reset);
get("reset").addEventListener("click", reset);
next.addEventListener("click", advance);
reset();
