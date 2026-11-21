# Use one Pocket forecast to decide whether to run another

One frozen TorchScript graph runs the exact Pocket Signal Relay update,
extracts the first target arrival, and uses the frozen v101 controller
to decide whether to issue a second forecast. The graph always starts
with Wait for eight ticks. The tested policy stops if Wait reaches the
target by the goal deadline. Otherwise it checks Open at one tick less
than the reported Wait arrival. For a missing arrival within eight ticks,
the fallback horizon is the goal deadline. The late-arrival horizon tests
an improvement over Wait and can exceed the original deadline.

The graph encodes the arrival as a comparison score and computes the
candidate horizon. The learned router selects stop or Open, and the
learned tagged reader emits the horizon byte. The goal and encoding are
structured. The model has a fixed first query and a supplied horizon
rule. The host loads the regional checkpoint and coverage and collects
events. Both simulator calls, result framing, learned inference, and
dispatch occur in the graph. Its inlined IR has no Python callback or
remaining method or function dispatch.

The 24 matched pairs reuse the Pocket shortcut layout generator. Opening
one gate in the initial checkpoint changes Wait from a detour to the
direct route. Each pair shares a deadline. Across 48 states, all natural,
swapped, alternate-feedback, and insufficient-coverage conditions matched
the declared policy. The scalar regional replay checks every executed
frame. An independent full-world shortest-path oracle checks arrival,
and the full-world scalar replay checks the relevant causal region.

In the first pair, the deadline is tick 5. Wait arrives at 6 with the
gate closed and 4 with it open. Natural execution calls Open at horizon
5 only in the closed world. Swapping the feedback suppresses that call
in the closed world and creates it in the open world, with each original
Wait forecast unchanged. In the closed world, supplying 7 changes the
Open horizon from 5 to 6. The alternative arrival is a controlled
intervention, not a new world execution. The coverage control stops
before any forecast.

Event fields are outcome, raw Wait arrival, shown arrival, encoded score,
learned choice, candidate horizon byte, dispatched action, dispatched
horizon, and Open arrival. A value of -1 marks an absent event. Outcome
0 is success, 1 rejects inadequate coverage or invalid input, and 2
rejects the emitted request. The archive includes source checkpoints and
target timelines. The simulator is the existing exact tensor cellular
automaton. The v103 WASM graph is verified separately.

From `visuaml-client/voxel-replay` in the research Python environment:

```sh
export PYTHONPATH=.:../backend
python tools/replay_pocket_forecast_handoff.py --example
python tools/replay_composed_delay.py
```

[Executable graph, inlined IR, paired worlds, and replay](../examples/pocket-forecast-handoff/)
