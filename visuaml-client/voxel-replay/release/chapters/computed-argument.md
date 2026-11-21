# Feed a WASM-computed delay into the next request

The frozen v101 reader receives tagged current and arrival ticks in either
order from the existing WASM guest. It emits the two operand bytes. The host
schedules a new subtraction guest and writes those operands to its input
page. The guest computes arrival minus current, stores it at byte 32, and
returns it. The resulting delay differs from both original operands in every
natural case. The host frames the return as a tagged record. The frozen
controller chooses Boost, Trim, or abstention and emits the argument byte
from that record. The host dispatches that exact byte.

| Condition | Correct requests and results |
| --- | ---: |
| Natural computed result | 72/72 |
| Substitute a delay computed with a different current tick | 72/72 |
| Substitute a delay computed with a different arrival tick | 72/72 |
| Substitute a delay across the routing threshold | 72/72 |
| Change an irrelevant tagged record | 72/72 |

Each substitution preserves the original execution and changes the feedback
forwarded to the controller. The replacement bytes also come from verified
WASM executions. Current-only and arrival-only substitutions preserve the
expected action and change the emitted byte. Crossing the threshold changes
the expected action and its argument if it calls. The irrelevant record
test preserves both. Cases where a controller correctly abstains produce no
follow-up call, and the emitted byte is still checked.

For the archived example, current 2 and arrival 7 produce delay 5. The
controller requests Trim(5) and gets 3. Changing current alone gives
Trim(4) and changing arrival alone gives Trim(6). A substituted delay of 8
crosses the threshold, and the controller requests Boost(8) and gets 13.

All 24 distinct readings, 48 subtractions, and 40 follow-ups match
Wasmtime on returns and whole-page digests with packed fallback blocked.
For the subtraction, an independent expected page also checks that only
the result slot changes. This uses the existing host-scheduled harness
and arithmetic follow-ups. The host chooses which computation to schedule
for the delay. Signal Relay simulation and graph-internal scheduling are
separate milestones.

From `visuaml-client/voxel-replay` in the research Python environment:

```sh
export PYTHONPATH=.:../backend
python tools/replay_computed_argument.py
python tools/visuaml_replay.py open release/examples/computed-argument/delay.json
```

[Guest, raw conditions, and exact traces](../examples/computed-argument/)
