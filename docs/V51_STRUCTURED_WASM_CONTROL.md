# v51 — Real structured WebAssembly control-flow execution

**Research status:** A deliberately bounded feature slice of the actual WebAssembly binary format. v51 does not support the complete WebAssembly instruction language or implement branch selection as neural Q/K/V attention. Its *arithmetic* is constructed from the v49 exact i32 bit-gated circuits; its control stack, PC and branch dispatch are implemented by an explicit host interpreter.

## Why this is an important next gate

v48 established fixed weights executing multiple straight-line WASM functions. v49 added exact-width signed i32 wraparound and comparisons. v50 recovered small symbolic programs from black-box neural outputs, with an independent SMT equivalence oracle.

Those results do not establish a general-purpose virtual machine, because an input-dependent program requires **structured control flow and a real execution model**. v51 adds the first real source-WASM control contracts: `block`, `loop`, `if`, `else`, `end`, `br`, `br_if`, `return`, and `nop`.

The binary parser and encoder use *actual canonical WASM 1.0 opcodes*, not a private branch pseudo-ISA. All modules expose one `run(i32)->i32` function, with zero to four additional i32 locals and at most 96 instructions. Blocks and loops have empty `0x40` result signatures. Only a bounded subset of other already reviewed v40/v49 i32 arithmetic, local-variable and stack instructions is admitted.

Every original source module is independently validated by a pinned Wasmtime engine. The in-house structural parser does **not** implement general WebAssembly stack-polymorphic typing or all proposal extensions, and those omissions are explicit.

## Control semantics

The host interpreter tracks an explicit list of control frames, each containing its lexical kind (block/loop/if), opening and end PCs, optional else PC, and stack height on entry. A branch depth is relative to the innermost active frame:

- A branch to a **loop** restarts at the instruction immediately following the loop opcode, preserving the target loop frame while discarding nested frames and excess stack operands.
- A branch to a **block** or **if** exits that construct, drops the target frame and its nested frames, trims the operand stack to the entry height, and resumes after its matching end.
- `br_if` always consumes its condition; it otherwise preserves the current stack and proceeds sequentially.
- An `if` consumes a condition, executes the chosen branch, and checks the empty block signature on exit. `else` skips the unchosen body.
- `return` exits the function early with one i32 result.
- `end` closes one active frame or halts the function with its required return value.

The linked lexical control table is constructed during parsing. Every arithmetic operation is evaluated using the independently tested v49 bit-gated ReLU/ReGLU circuits; the PC and control frame logic remain host-implemented.

**Important limitation:** a program may be structurally admitted while failing full WASM type validation. CI independently calls `wasmtime.Module.validate` on every original replay case. No arbitrary binary is executed without this external validation in the dedicated experiment.

## Fuel is not a WASM language trap

The explicit `fuel` budget bounds how many instructions the host interpreter executes. When it expires, v51 raises **ControlFuelExhausted**, distinct from an actual runtime language trap such as memory bounds or integer division by zero. The independent original Wasmtime runtime is configured with its own fuel limit for an intentionally nonterminating real WASM loop. Both are expected to stop for lack of fuel, but their instruction/fuel accounting is **not asserted identical**.

This distinction matters for later learned/constructed interpreters: a finite experiment must never declare nontermination or equivalence merely because a timeout occurred.

## Independent evidence protocol

The first program set includes:

- Count-down sum `sum(1..n)`, using a real `block`, nested `loop`, `br_if 1` to leave the block and `br 0` to repeat the loop. Its accumulator and counter provide observable mutable-local state on every iteration.
- Signed `if/else` selection.
- Conditional branch out of a block.
- Early return from a structured conditional.
- Signed i32 modular overflow under a plain return.
- A separate nonterminating loop with explicit fuel exhaustion.

For each terminating program, original source bytes are validated by Wasmtime, then executed through both Wasmtime and the host-controlled, bit-neural-arithmetic interpreter. The complete host machine trace preserves each executed PC, next PC, branch frame names, live i32 locals, live operand stack, and halt flag. The independently observed Wasmtime *final* result is compared to the host result; Wasmtime does not expose a machine-internal per-instruction trace in this workflow.

Targeted unit tests also inspect individual back-edge PCs, stack effects and local updates. The complete raw program binary and source-commit hashes are retained in the evidence artifact.

## What v51 does NOT prove

It does **not** implement all WebAssembly block signatures, stack-polymorphic typing, br_table, call/call_indirect, host imports, globals, memory, i64, f32/f64, SIMD, GC, exception handling, WASI or all traps.

It does **not** have a neural Q/K/V selection head that independently implements branch target resolution. Its arithmetic is constructed neural logic, its control dispatch is a conventional Python state machine.

It does **not** prove all-input equivalence for arbitrary loops. Program termination and self-modifying memory require additional explicit semantics.

## Next gate: neural control routing, not inflated opcode counts

To move from this first structured-control layer toward a genuine universal neural interpreter, the next implementation must:

1. Compile lexical branch target and nested-frame semantics into data and fixed routing weights rather than host-only branch dispatch.
2. Verify the complete machine state against an independent reference after every branch and loop iteration.
3. Implement bounded mutable memory and correct traps separately, including little-endian load/store and out-of-bounds detection.
4. Add multi-function calls, typed index spaces and module-instantiation contracts.
5. Extend numeric types and proposal-specific conformance; benchmark against the official WebAssembly spec tests.
6. For decompilation, synthesize **equivalent observable machines with explicit side effects** rather than claiming unique historical source recovery.

This is a concrete step, but an honest one: **programmable control flow in source WASM plus constructed exact arithmetic is not yet a fully neural control-flow machine.**

### Reproduce

```bash
PYTHONPATH=visuaml-client/backend python -m pytest -q \
  visuaml-client/backend/tests/test_wasm_control_flow_v51.py

PYTHONPATH=visuaml-client/backend python \
  visuaml-client/backend/scripts/probe_wasm_control_flow_v51.py \
    --output /tmp/structured-control-v51.json
```

CI workflow `.github/workflows/wasm-structured-control-v51.yml`. Archive original complete evidence and a separate CI-origin manifest after all checks pass.

## v51.1 — Euclid's GCD through actual branch/loop structure

The follow-on research fixture is **a real WASM implementation of Euclid's repeated-subtraction algorithm**, computing `gcd(n,12)` for nonnegative `n`. For negative inputs, the fixture deliberately returns a documented sentinel 0 rather than claiming a mathematical GCD contract. Input zero returns 12.

This program adds the following stressors simultaneously: two guarded early returns, an outer block with a nested loop, signed comparisons, an `if/else` body that chooses which mutable local to decrement, a `br 0` back-edge, and a `br_if 1` escape when the two local variables become equal.

The dedicated regression independently validates the exact canonical WASM binary in Wasmtime and compares each execution with v51's host-controlled bit-neural arithmetic. A separate **mathematical `math.gcd` oracle** supplies expected answers for each reviewed nonnegative test. The extended evidence script `scripts/probe_wasm_gcd_v51_1.py` records source WASM bytes and SHA-256, numeric witnesses, source identities and detailed control traces for selected inputs. Its results are archived separately as `benchmarks/wasm_control_gcd_v51_1.json` and `.origin.json`; the original v51 transcript is never overwritten.

An adversarial very-large-positive input can exhaust bounded execution fuel even though the subtraction algorithm is **mathematically terminating**. This is deliberately recorded as *resource exhaustion*, not a claim that the input constitutes a nonterminating WASM program. It is important to distinguish termination of the mathematical function from practical execution under an explicit cost policy.

The machine's arithmetic continues to use the same fixed v49 bit circuits, while control routing remains Python-host dispatch. This is not neural execution of loops by softmax attention layers; that remains a separate future milestone.

## Conformance ledger

The machine-readable [WASM semantic capability ledger](WASM_CONFORMANCE_LEDGER_V51.json) explicitly distinguishes restricted supported features from unsupported instruction, effect and runtime families. It deliberately contains **no universal-WASM percentage**, since opcode counts alone cannot measure validation, traps, module instantiation, memory, calls, numerical corner cases or host-interface support.
