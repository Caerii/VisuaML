# v53: Fixed-weight control-frame transition machine

**Status: design contract / implementation milestone, NOT a verified neural-frame executor.**
Base: fused v52 routing + alternate typed-KAN v45 research. Do not modify the original v50–v52 experiment archives.

## Why this is the next verification boundary

v52 uses an actual constructed one-hot PC router, but the host executes
`frames.append`, `frames.pop`, `del frames[index:]`, and
`del stack[frame.stack_base:]`. The complete next PC depends on gate outputs,
yet the *control-frame state transition* remains host logic. v53 must replace
that logic, not rename Python list operations to neural computation.

The new architecture must implement a **finite, fixed-capacity state transition
circuit**, using the same fixed coefficients across each admitted WASM program.
Source-specific lexical links, opcode tokens, branch depth, and frame metadata
are **data**, never parameters learned or specialized to an individual program.

## Canonical finite state

Use `MAX_NESTING=12`, `MAX_STACK=24` and the v51/v52 instruction bound.
Store:

- A one-hot frame depth `D[0..12]`; `D[0]=1` at reset.
- For each slot `j<12`, a valid bit, one-hot kind (block/loop/if), and
  one-hot start/end/else PC and operand-stack base height.
- A one-hot operand stack height `H[0..24]`, with 32-bit slot contents using
  the v49 bit-vector encoding.
- A one-hot instruction and dynamic branch depth supplied as program tokens.
- An event one-hot over `none, enter, close, branch, return`.
- A branch-taken bit coming *only* from the v49 bit gate evaluated on the
  live operand-stack condition.

Every output must be one-hot/binary/canonical. Fail closed if the circuit
produces fractional weights, multiple active entries, invalid depths, or
inconsistent stack bases.

## Pure fixed-weight transitions

A fixed architecture consumes the old state plus instruction data and emits
the complete next frame state and restored stack height, without a host
conditional selecting a transition.

1. **Enter:** write the input frame into the first vacant slot and increment
   depth one step. An `if` with no else and false guard does **not** push;
   an `if` with else and false guard does push (to close at the matching
   `end`).
2. **Close:** pop exactly one live top frame and retain the current operand
   height only when the empty block signature is satisfied.
3. **Branch:** compute the dynamic target slot by subtracting the one-hot
   immediate depth from the active frame pointer. The target is retained for
   `loop` and removed for `block`/`if`; all deeper frames are invalidated.
   Restore operand height to the target frame's saved base. The selected
   PC remains the separate v52 routing output; check their correspondence.
4. **Return:** empty frame stack, preserve the single result value, and
   require a terminal v52 PC.
5. **No-op:** retain state exactly, including inactive slot canonicalization.

**Critical invariant:** zero branch condition must not mutate frame or stack
state except for consuming the `br_if` condition operand. Both zero and
nonzero are interpreted as full 32-bit signed i32 bit vectors.

## Separation of concerns

The independent reference oracle remains v51, and Wasmtime validates the
exact original source WASM. The new circuit owns its state transition, not
its own reference oracle. A host can validate outputs and raise traps, but
cannot repair an incorrect neural result or silently fall back to v51.

A finite circuit with fixed coefficients is **not** an ordinary transformer,
a learned program, full WASM validation, or universal source recovery.
The production host still fetches opcodes and holds the physical state until
each part has independently passed its own replacement gate.

## Verification matrix required before the v53 claim

- All six v52 programs and 61 historical input witnesses, per instruction:
  PC, next PC, full frame stack, complete 32-bit locals and operand stack,
  operand height, and halt status against v51.
- Deep nested loops with `br 0`, `br 1`, `br_if`, early `return`, true/
  false `if` with/without `else`, and depth-12 boundary.
- Corrupt fixed coefficients, frame validity, one-hot masks, stack base, and
  program-data target encodings. Verify failures are observable; do not
  normalize malformed tensors into success.
- Compare bounded fuel exhaustion separately against Wasmtime, never
  rebrand it as a WASM trap.
- No opcode-specific handwritten bypass for GCD.
- Independent whole-trace replay with SHA-bound source/WASM, code version,
  exact fixed coefficients and explicit provenance.
- Add an executable regression that **fails if Python host branch/pop/unwind
  dispatch is reintroduced into the claimed circuit execution path**.

## Acceptance criteria

The milestone is complete only when the new executor actually feeds its
constructed frame-state output into the next step and the per-instruction
differential gate passes. Prior v52 records remain byte-for-byte unchanged.
Publish a distinct v53 source-bound archive and conformance-ledger update.
If only the circuit is unit-tested but not connected to execution, report
it as a **component prototype**, not v53 completion.

## Following gates

v54: constructed operand-stack pointer movement and bounded store/fetch.
v55: restricted linear-memory loads/stores with out-of-bounds traps.
v56: direct calls and call-frame stacks; distinguish tail calls, imports
and indirect table calls as separate admitted features.
v57: differential official WASM conformance corpus at specific spec version,
reporting exact supported/unsupported subtests without an arbitrary coverage
percentage. Beyond this, study unknown-weight mechanism identification
separately from the analytically constructed executor.
