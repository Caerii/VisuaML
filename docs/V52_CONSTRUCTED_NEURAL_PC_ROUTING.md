# v52 — Constructed neural program-counter routing for structured WASM

**Status:** A bounded research implementation of an actual gate-driven, one-hot program counter. It is **not** a full transformer, a fully neural WASM interpreter, a general validator, or an arbitrary decompiler. The v49 bit-gated arithmetic and v51 host-managed control-label stack remain in place.

## What changed

v51 executes real source WASM `block`, `loop`, `if`, `else`, `br`, `br_if` and `return` instructions correctly on the tested cases. But the host interpreter directly selects `next_pc`. v52 moves that **runtime next-PC decision** into the constructed routing circuit.

The new executor feeds the **actual output one-hot PC vector** back into the next instruction lookup, instead of merely predicting what the host interpreter would do. It computes two candidate locations for every current instruction and chooses by binary/ReLU-gated arithmetic. The CPU still performs opcode dispatch, stack/local effects and dynamic label-stack maintenance, but it no longer supplies the branch target or selects next-PC under the normal execution path.

## Fixed weights vs program data

The distinction is critical:

- **Fixed interpreter weights:** one shared set of gate coefficients for always-taken, branch-if-nonzero, branch-if-zero and two one-hot path selectors, constructed analytically and never trained or specialized by a particular program.
- **Per-program token data:** the canonical source's lexical jump targets are compiled into two `97×97` routing matrices (normal successor and alternate) and a three-component conditional guard matrix. The tables are *data supplied to the interpreter*, not a separately trained interpreter for each program.
- **Dynamic condition:** the actual top 32-bit stack register is tested for zero using the already verified v49 constructed bit gates; its result is used to choose the control target.

The small fixed-weight selector implements:

```
z       = ReLU-based 32-bit EQZ(stack_top)
nonzero = 1 - z
take    = w0 * always + w1 * branch_if_nonzero * nonzero
          + w2 * branch_if_zero * z

default = onehot_pc @ default_target_table
branch  = onehot_pc @ alternate_target_table

next_pc = keep_weight * (1 - take) * default
          + jump_weight * take * branch
```

All fixed weights have explicit numerical values and a SHA-256 identity. The gate outputs must be *binary* and the next-PC vector must contain **exactly one active position**. Corrupted fixed path weights are expected to fail this admission condition; altering conditional weights in a reviewed source case changes which instruction the machine executes. This specifically tests against merely including decorative "neural weights" that do not influence behavior.

## Actual WASM source and lexical target preprocessing

The original canonical WASM 1.0 binary is parsed and structurally admitted by v51. The source is independently validated by Wasmtime in CI. CPU preprocessing matches lexical `block/loop/if/else/end` pairs and computes target tables:

| Source instruction | Gate condition | Alternate target |
| --- | --- | --- |
| ordinary arithmetic, local, nop, end | always sequential | same as sequential |
| `block` / `loop` | sequential | same as sequential |
| `if` | stack condition equals zero | first instruction of else arm, or after matching end |
| `else` | always taken | instruction after corresponding end |
| `br` | always taken | loop body start for loop labels, after matching end for blocks/ifs |
| `br_if` | nonzero binary stack condition | same label target as `br` |
| `return` | always taken | machine terminal sink |

Every table is finite, bounded and bound to the original typed source and WASM SHA-256. A modified alternate PC, ambiguous guard encoding or mismatched source digest must fail closed. Values after the last admitted source instruction are zero-filled; the fixed terminal slot is a sink.

**This is not a neural parser that can discover arbitrary branch structures by itself.** Compilation of source-dependent route tables is host preprocessing, and the dynamic control-label/operand stack remains conventional host-managed code. The router's *runtime branch decision and PC feedback* alone are neural/gated.

## Evidence and falsification

Tests cover six different WASM programs reusing identical fixed weights: countdown summation, signed if/else, if without else, conditional exit from a block, early return, and Euclid's GCD by repeated subtraction.

The independent v51 interpreter computes full host-machine states **separately**. The v52 routed executor's actual next-PC output chooses the subsequent instruction. At each step the experiment compares the selected source PC, next PC, opcode, live stack/locals, active control labels and halt state. Independent Wasmtime compiles and executes the original source binaries for final-output comparisons on every reviewed input.

The tests include falsification controls:

- Mutating fixed route coefficients must change an observed branch behavior or violate the one-hot target invariant.
- Corrupting an alternate target, conditional guard or source identity must be rejected before execution.
- A signed negative, nonzero bit-vector condition must route correctly (not merely test the signless low bit).
- An infinite loop and a *mathematically terminating but fuel-limited* large GCD computation must produce bounded **resource exhaustion**, not false language-trap or nontermination claims.
- Every program-token route digest is distinct, while the interpreter's fixed route-weight digest remains identical across all six programs.

## Acceptance and provenance

The dedicated `.github/workflows/analytic-neural-control-v52.yml` runs pinned NumPy, Wasmtime, Pytest and Ruff. The complete probe output records the original WASM bytes, canonical source SHA-256, reviewed lexical targets, fixed gate SHA-256, full machine-state traces for selected cases, a deterministic trace digest for every witness, the independent v51 comparison, and the original Wasmtime result.

The first successful raw CI output should be permanently archived with a separate source-commit origin manifest. Old v51/v51.1 provenance and evidence must remain untouched. This branch remains a draft until dedicated and full repository CI pass.

## What remains toward a true neural VM

v52 does *not* yet execute lexical label-stack operations via neural layers; it does not provide an actual standard attention Q/K/V/softmax transformer block, memory/trap execution, calls/imports, arbitrary WASM module validation, or full Core specification conformance.

The next critical architectural step is to encode dynamic control frames and stack-base resets into analytically constructed **neural state-transition operations**, rather than leaving them in host code. That would remove another host-controlled semantic dependency. After that, independently verified memory pages, traps, calls and typed multi-function modules become useful targets.

For arbitrary decompilation, v50's bounded black-box CEGIS is only a starting point. Recovery from unknown learned weights requires an independent source-free model/feature extraction protocol and explicit representational ambiguity and causal verification; copying the reviewed branch table is not decompilation.

**This is a concrete improvement to the generality of the constructed neural execution substrate, not an invented percentage of universal WASM coverage.**
