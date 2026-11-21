import bundle from "./evidence/pocket-paired-goals/replays.json";
import bundleUrl from "./evidence/pocket-paired-goals/replays.json?url";

const get = id => document.getElementById(id);
const budget = get("budget");
const override = get("override");
const name = ["Wait", "Open", "Neither action"];
const checkpoint = "shortcut-00";

get("download").href = bundleUrl;
get("world").textContent = `Saved inference replays from the same checkpoint, with a deadline of tick ${bundle.worlds[checkpoint].deadline}. Open clears the shortcut. Wait leaves the detour available.`;

function render() {
  override.disabled = budget.value !== "2";
  const mode = override.checked && !override.disabled ? "intervention" : "actual";
  const sections = ["edits", "speed"].map(objective => {
    const { run } = bundle.examples.find(row => row.id === checkpoint && row.objective === objective &&
      row.budget === Number(budget.value) && row.mode === mode);
    const section = document.createElement("section");
    const title = document.createElement("h2");
    title.textContent = objective === "edits" ? "Avoid editing if possible" : "Arrive as early as possible";
    const list = document.createElement("ol");
    const events = run.calls.flatMap(call => {
      const explain = tick => tick === 0 ? "no arrival by the deadline" : `arrival at tick ${tick}`;
      return [
        `Controller requests ${name[call.action]} through tick ${call.horizon}.`,
        `Verified forecast: ${explain(call.raw)}.`,
        call.raw === call.shown ? `Controller receives ${explain(call.shown)}.` :
          `Intervention replaces feedback with ${explain(call.shown)}.`,
      ];
    });
    events.push(run.state === 1 ? "Controller stops with insufficient evidence." :
      `Controller recommends ${name[run.action]}.`);
    const next = document.createElement("button");
    next.textContent = "Next event";
    next.type = "button";
    let position = 0;
    next.addEventListener("click", () => {
      const event = document.createElement("li");
      event.textContent = events[position++];
      list.append(event);
      next.disabled = position === events.length;
    });
    section.append(title, next, list);
    next.click();
    return section;
  });
  get("runs").replaceChildren(...sections);
}

budget.addEventListener("change", render);
override.addEventListener("change", render);
render();
