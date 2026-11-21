# Pocket goal planning

The controller receives a deadline and a preference for avoiding edits. It emits forecast requests and a recommendation. The host passes in the prompt and a bounded checkpoint. The composed TorchScript graph generates request tokens, validates the request, runs the unchanged forecast operation from the published inference graph, and feeds its result back to the controller. It allows two calls. The inspected artifact contains no Python callbacks. The recommendation is a prediction from a private future.

## Results

The planner was frozen at `5bc6d78` before the holdout was authored. The controller trained on synthetic demonstration traces with outcome bits, including both call orders, and saw zero world layouts. Wait-first made up three quarters of the training demonstrations. The wording and deadline combination “kindly” at tick 6 was omitted during training. The trained greedy policy chose Wait first on all observed tasks. Its second request was conditional on feedback. This tests bounded two-action planning under a stated objective with direct supervision of the search policy.

- **192/192** unmodified recommendations matched the exhaustive oracle across 64 related triads in four existing layout families. Each triad tested Wait suffices, Open is necessary, and neither action succeeds. Deadline and prompt were fixed within each triad. Layout draws used a fresh seed and deadlines from 5 through 8.
- The controller stopped after a successful Wait forecast in **64/64** cases and also tested Open in **128/128** cases where Wait failed. There were zero unsupported recommendations across the tested conditions.
- In Open-required worlds, substituting a successful Wait forecast produced Wait in **64/64** cases. Substituting a failed Open forecast produced Neither in **64/64** cases. These are scores against received feedback. The actual worlds still require Open.
- **832/832** executed forecasts agreed with the independent scalar reference. The harness compares each regional frame with the scalar regional simulation, and the whole-world scalar simulation checks target values and the remaining causal cone at every forecast tick.
- The precomputed-reply and irrelevant-edit conditions each matched all **192** recommendations. Disabled, missing, and insufficient checkpoint conditions each stopped without a recommendation in all **192** cases.
- A separate template adapter recovers the deadline and both explicit policy branches from all **200** previously published paraphrases. Direct admission was **0/200**. Adapter admission, interpretation, and full task success were each **200/200**. The adapter recognizes three published sentence patterns and rejects unsupported wording. It supplies canonical text to the original controller, which still chooses the tools. These examples evaluate recovery of known wording and do not measure general paraphrase understanding.

One seeded cell represents the agent's initial position. The persistent local automaton marks cells reachable by the deadline under cardinal movement with a stay action. Open removes the gate before propagation, and Wait leaves the terrain as supplied. The simulator receives the scene as data. The prompt contains no scene clues.

## Reproduce

From `visuaml-client/voxel-replay`, use the research Python environment with PyTorch and the same dependencies as the explicit-policy baseline.

```sh
PYTHONPATH=.:../backend python tools/evaluate_goal_planning.py
PYTHONPATH=.:../backend python tools/evaluate_goal_planning.py --replay evidence/pocket-goal-planning/replays.json
PYTHONPATH=.:../backend python tools/evaluate_pocket_language_adapter.py
npm ci
npm run dev
```

Open `/pocket-goal-planning.html` to step through a saved run or an intervention. The browser presents the recorded trace. The Python replay command reruns the frozen inference graph against the included world and checks the journal. The full outcomes archive records assigned tasks and first failures by stage. The language adapter has its own archive and original frozen controller identity.
