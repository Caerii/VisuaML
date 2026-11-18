# Neural WASM research: compositional verification gates

This ledger is an **engineering contract**, not a claim of universal WASM
conformance. The intended object is a reusable finite machine: *source
program data* determines which computation executes while a fixed set of
analytically constructed gates determines the execution mechanics.

## Ownership layers

| Layer | Program-dependent data | Fixed circuit authority | Remaining host authority |
| --- | --- | --- | --- |
| Full i32 arithmetic, v49 | i32 bit-vector operands | analytic ReLU/ReGLU arithmetic | numerical array evaluation and admission |
| WASM source, v51 | canonical typed opcode/control structure | none at first | decoder, frames and all routing |
| Next PC, v52 | compiled lexical jump tables | one-hot PC route | opcode fetch and frame semantics |
| Frames, v53 | source label metadata | bounded frame-state and branch restoration | lexical checks and event decoding |
| Events, v54 | source opcode tokens, live guards | frame push/pop/branch/return choice | control metadata, stack effects |
| Operand data, v55 | i32 values and stack-height tokens | bit-sliced tensor push/pop/truncate/top read | instruction-level stack-effect request |
| Arithmetic selection, v56 | opcode one-hot | fixed six-way ALU selection | stack arity, immediate/local lookup |
| Typed microops, v57 research | **one canonical source-bound token table** | opcode/effect/mode/ALU/local/branch selection | remaining host opcode classes and validation |

The v57 row is a **research implementation awaiting offline verification**.
Do not mark it merged or confirmed by CI. GitHub Actions are intentionally
disabled on this repository.

## Minimum independent proof obligations for every new mechanism

1. **Source identity:** SHA-256 of canonical original WASM and digest of
   the typed program; token admission compares complete source bytes, not
   merely a program name or file path.
2. **Shared weights:** one fixed coefficient set is reused across distinct
   source programs, with a digest recorded independently from program data.
3. **Live authority:** the claimed gate output actually changes the next
   execution state, not just a shadow trace; injected weight damage changes
   behavior or fails closed.
4. **Independent oracle:** compare original source bytes in Wasmtime and
   complete per-step states with an independently implemented older machine.
5. **Adversarial boundaries:** full signed-i32 edges, false conditions,
   nested label targets, mutation of legal-shaped tokens and fixed weights,
   stack overflow/underflow, and bounded fuel exhaustion.
6. **Immutable evidence:** never alter archived historical transcripts;
   commit a new record with producing source commit/tree and raw hash only
   after a verified local/CI run.
7. **Explicit negative claims:** source-aware execution and bounded
   black-box synthesis are different experiments; neither identifies
   mechanisms inside arbitrary learned networks.

## v57 acceptance

The offline evidence runner has a deterministic, closed-world corpus
of **19 source programs and 138 planned input witnesses**. Each result
must agree across four independently executed paths: v57 microop
executor, v56 ALU executor, v51 host interpreter and original Wasmtime.
Compare every recorded PC, next PC, i32 stack/local, live frame label and
halt state, not just the final integer.

The generated JSON must record full source WASM bytes, full v57 trace,
SHA-256 transcript digest, fixed-weight digests and exact Git identity.
It must be produced from a clean checkout, and must never overwrite
an existing record. Until it runs successfully, numbers remain *planned*,
not *passed*. See [v57 protocol](V57_SOURCE_BOUND_TYPED_MICROOPERATIONS.md).

## Next architectural bottlenecks

A truly compositional VM next requires uniform tensor-controlled opcode
effects, including locals and stack data movement, without a Python
opcode branch selecting each microstep. Only after this is falsified
against the source reference should bounded linear memory, traps and
calls be admitted. Conventional host execution of NumPy and resource
allocation remains permitted but should be enumerated, not hidden.

No arbitrary opcode-support percentage will be reported: conformance
should be versioned by features and *actual official test subsets*.
