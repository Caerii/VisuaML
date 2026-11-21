# Run the computed-delay handoff inside one graph

One frozen TorchScript invocation reads tagged ticks, builds the subtraction
guest's input page, runs the pinned source-step WASM graph, frames the return,
decodes and validates the learned request, and runs the selected follow-up.
The subtraction stage is scheduled by the graph. The controller chooses
Boost, Trim, or abstention. The host loads the tagged records, configuration
and validated one-page program images and collects the event record and pages.
The initial readings guest remains upstream of this graph.

The graph takes two tagged byte records, a six-element configuration tensor,
three validated program images and their entry PCs. Configuration holds the
threshold, rule, optional feedback override, distractor and two fuel budgets,
each bounded at 100. Tensor shapes and validated program images are part of the
host ABI. The graph checks tag identity, byte range, rule and budget bounds.
It allocates private memory for each guest. Each bounded loop runs the same
pinned exact tensor step. The reader and router reuse the frozen v101 weights.

The first ten event fields are outcome, current, arrival, subtraction status,
subtraction result, forwarded feedback, tool ID, argument byte, follow-up status,
and return. The final four hold the exact framed feedback bytes. An unset
field is -1. Outcome 0 means success or explicit abstention, 1 malformed input,
2 failed subtraction, 3 invalid feedback, 4 invalid emitted request and 5
failed follow-up.

All 360 v102 conditions match at the operand bytes, raw delay, full delay page,
exact feedback record, request, return and full follow-up page. The v102 replay
independently checks the same guests against Wasmtime. Packed fallback is
blocked in verification. The inlined IR contains no Python callback, remaining
method call or function call. It includes both bounded guest loops. No host
callback runs between the delay and the follow-up.

The duplicate-tag case stops before either guest runs. The delay-exhaustion
case retires three instructions, leaves the result slot untouched and emits no
follow-up request. The follow-up-exhaustion case records a decoded request and
no completed return, with its page unchanged.

From `visuaml-client/voxel-replay` in the research Python environment:

```sh
export PYTHONPATH=.:../backend
python tools/replay_composed_delay.py
python tools/replay_composed_delay.py --case delay_exhaustion
python tools/replay_computed_argument.py
```

[Executable graph, inspectable IR and replay bundle](../examples/composed-delay/)
