# v56 — Fixed-weight neural ALU as a shared execution datapath

**Status: candidate pending exact-head focused and full CI.**

## Why this advances generality

v55 has a gate-authored program counter, structured frame state, frame event
selector, and a real tensor-backed operand store. However, the Python host
still selected which of the v49 primitive arithmetic functions to call.
In v56, the host retains the program's restricted opcode/arity admission and
physical stack effects, but it **no longer selects the arithmetic result**.

The fixed ALU computes six candidate full-width bit-gate outputs:
`i32.add`, `i32.sub`, `i32.mul`, `i32.lt_s`, `i32.gt_s`,
and `i32.eqz`. A one-hot opcode token (program data) and fixed output
coefficients bilinearly select one 32-bit output. This is the *same* ALU
weights object for every admitted WASM source. Signed comparisons emit a
zero-extended one-bit boolean. Arithmetic is exactly modular on 32 bits.

**Performance warning:** the construction currently evaluates *all* six
candidate circuits on each arithmetic instruction. It favors a closed
and independently verifiable control/data contract over performance.
Benchmarks must compare it against the old host-selection path before
claiming speed. Avoid presenting this as a performance improvement.

## Independent evidence contract

- Boundary test every admitted opcode across INT_MIN, INT_MAX, negative,
  zero, and selected adversarial operands.
- Random 32-bit witnesses, with Python integer reference used ONLY
  for independent test validation, never executing the neural circuit.
- Corrupted selection gains fail closed; malformed or mixed one-hot
  opcode tokens fail.
- Real WASM programs compiled from v51's canonical source encoding:
  differential every PC, next PC, stack, locals, frame labels and halt state
  against the independent v51 host interpreter and v55 executable.
- Real Wasmtime module validation and execution are **mandatory**, not
  optional in dedicated CI; fuel exhaustion is a resource policy.
- Do not modify historical archives; add a new source-bound v56 evidence
  manifest and whole-trace digest only after final-head verification.

## Remaining host/control authority

The host still decodes program opcodes, controls stack arity and local
operations, supplies immediate/local indices, manages lexical metadata,
and provisions circuit arrays and fuel. The opcode-to-ALU output gate is
fixed, but this is *not* a learned transformer, full neural ISA dispatch,
all-input formal proof of every gate, universal WebAssembly implementation,
or recovery of arbitrary unknown neural mechanisms.

## Compounding follow-up

The most valuable next step is a **typed microoperation IR**, shared
across arithmetic, operand stack, locals and future linear memory. Each
instruction is represented as source-bound data plus fixed operation-class,
operand-index and destination selectors. Integrate it into the live
executor and falsify host fallback using gate ablation. Then create bounded
linear-memory gates with real WASM bounds/traps, followed by calls and
call-frames. Only then consider broader WASM feature conformance.

## v56.1 generated-program differential corpus

`tests/test_generated_wasm_alu_v56.py` builds **16 seeded, distinct,
canonical original WASM binaries** using four-step expression chains that
combine exact i32 arithmetic, signed comparisons, `eqz`, and a mutable
`local.tee`/`local.get` data dependency. These are *new source programs*,
not new inputs to the six historical hand-built programs. Each is checked
at seven adversarial i32 inputs (112 generated source/input executions),
and the test compares:

- Independent Wasmtime module admission and result
- The v51 original host interpreter
- The v55 tensor operand executor
- The v56 constructed ALU executor, including every executed state step

This expands compositional evidence while preserving the distinction
between tested finite witnesses and all-input formal equivalence.
The test is deterministic and repeats exactly under the same source hash.
A passing CI run is required before these counts can be reported as
verified executed witnesses.
