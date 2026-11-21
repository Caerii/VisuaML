# Rust → Signal → Rust

A compiled Rust WASM guest opens the Pocket Signal Relay gate. The frozen tensor rule advances the signal to a distant cell. The guest then reads that cell through an explicit mailbox and branches. The browser route steps saved exact evidence. The executable replay runs the frozen VisuaML call graph and exact automaton, with packed fallback blocked.

## Tick order and interface

The guest receives a pointer to eight signed i32 mailbox words at byte 65472. They hold gate material, Dig input, old material, observation request, observed value, action, phase, and the outside-read fixture input. The guest enters its gate helper and loads the old material 7. The checkpoint suspends after source step 10, at helper PC 2 with call depth one, before the gate store. The host records the VM continuation, a pre-edit region, and cursor 0, meaning finish the helper before tick one.

1. Resume the helper. If Dig is set, guest PC 10 writes material 0. The host publishes that edit to the automaton once the Rust invocation completes. With Dig omitted the wall remains closed.
2. The guest writes its requested world cell into the mailbox before leaving initialization. The regular run requests (14, 8). The test input can request (2, 8).
3. Each tick advances the radius-one, sealed, synchronous automaton from the same previous state. The host checks the requested coordinate and copies the new signal bit into the next invocation's mailbox.
4. The guest loads that bit at PC 41. Branch PC 43 goes to PC 44 when the signal is on and PC 48 when it is off. The on path stores action 1 at PC 46. The action is an observable output in this fixture.

The guest can modify the automaton once through the gate store. After initialization its invocations receive one signal value and publish an action. There are no later terrain edits in this profile.

## Exact continuation and broken guarantees

The saved checkpoint binds the VM and its program and graph, the exact automaton artifact, the region data, and the event cursor. Dig and omitted runs independently resume the exported file in fresh processes. Both match the uninterrupted full-world run at every guest instruction after import, including source access records, branch PCs, output stores, and full guest-page hashes. Every completed guest invocation also matches Wasmtime. The tensor grid matches a scalar rule at each tick. Full and regional grids are compared within the remaining target cone.

With Dig, the signal reaches successive points from (9, 8) at tick one to (14, 8) at tick six. The sixth Rust observation changes the action to 1. With Dig omitted, all six observations and actions remain zero.

The six-cell halo contains every radius-one dependency of (14, 8) for six ticks. Its guarantee also depends on the declared guest observation and edit schedule. The five-cell region omits (8, 8), so the replay rejects it as a certified checkpoint. An explicit diagnostic bypass locates the first incorrect guest observation at tick six: LOAD at PC 41 receives 0 rather than 1, branch PC 43 goes to PC 48, and the action stays 0.

The outside-read fixture requests (2, 8) at PC 33. In the full-world run that cell is already on and the guest acts at tick one. The six-cell replay detects the unauthorized dependency at the first observation and stops before supplying a mailbox value or executing the observer. This shows that a guest read can affect the action earlier than the signal can reach the usual target. A wider contract would have to include the backward cones of additional reads, and any world changes caused by those reads.

## Open the chain

From `visuaml-client/voxel-replay` using the existing research Python and Node environments:

```sh
PYTHONPATH=.:../backend python tools/replay_coupled_signal.py
PYTHONPATH=.:../backend python tools/replay_coupled_signal.py --resume evidence/pocket-coupled-checkpoint/checkpoint-dig.json
npm run dev -- --port 4182
```

Open `/pocket-coupled.html` and use **Step replay**. The viewer starts inside the Rust helper, shows the gate write, advances each signal frame, and displays the observation instruction, branch, and action store. Select the omitted, short-halo, or outside-read replay to inspect the controls. The selected combined checkpoint can be downloaded. The learned-rule intervention remains in the separate automaton replay.

Rebuild the guest with a Rust toolchain that has the `wasm32-unknown-unknown` target using `bash tools/build_signal_relay_call.sh`. [Rust source](../../guest/signal_relay_call.rs) · [Exact tape](replay.json) · [Dig checkpoint](checkpoint-dig.json) · [Closed checkpoint](checkpoint-omitted.json) · [Automaton and learned controls](../pocket-cellular-automata)
