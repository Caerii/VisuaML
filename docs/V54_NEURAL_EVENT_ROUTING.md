# v54: fixed-weight WASM opcode-to-frame-event routing

**Status:** component prototype. This is not yet a merged, end-to-end neural event decoder.

## Architectural boundary

In v53 the fixed-weight frame circuit drives live label-stack depth and
physical operand-stack restoration, but the Python host still supplies the
**frame action** (retain/push/pop/branch/return), based on decoded WASM
instruction names. This leaves a material host-control authority path.

v54 introduces a constructed one-hot opcode-event selector. The program
provides a source-bound, program-data opcode one-hot; the same constant
coefficients run across every admitted WASM program. Live `branch_taken`
must originate from the v52 neural program-counter gate. The dynamic frame
active flag must originate from the v53 frame-depth state; lexical else
presence is verified source program data.

## Truth-table contract

Actions are one-hot `retain, push, pop, branch, return`.

- block and loop => push
- if => push if live sequential arm or lexical else, otherwise retain
- else => pop
- end => pop when there is a live dynamic frame, otherwise retain
- br => branch
- br_if => branch iff the live v52 guard gate fired, else retain
- return => return
- other admitted arithmetic/stack opcode => retain

v54 computes the default event from a constant matrix, and three guarded
one-hot shifts using fixed coefficients and an exact ReLU binary OR. No
runtime Python opcode conditional selects the output action.

The dedicated test probes 9 token classes x all 16 combinations of the four
binary inputs = **144** candidate contexts, plus malformed gates, malformed
tokens, and weight ablations. It tests the static event circuit only, not
full WASM execution.

## Remaining acceptance gates

1. Integrate `select_frame_action` into a new experimental executable
   variant, feeding its output into `step_frames` rather than the host's
   action integer.
2. Independently compare every instruction across v51 host interpreter,
   v52 gate-PC executor, v53 gate-frame executor and Wasmtime source execution.
3. Ensure mutated event weights measurably alter executable behavior or
   fail closed. Never silently substitute an expected host event.
4. Run the full regression on exact head; preserve original v53 evidence
   records and keep new v54 CI history distinct.
5. Declare remaining host responsibilities: source token decoding,
   opcode data-stack effects, lexical preprocessing, stack-slot movement
   and full WASM validation are not yet neuralized.

This circuit is not a learned transformer or a proof of universal WASM
conformance. It is an analytically constructed conditional router for a
strict, bounded instruction subset.
