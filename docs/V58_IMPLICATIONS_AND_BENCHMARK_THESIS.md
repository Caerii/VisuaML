# VisuaML research thesis: a compiler that makes computation inspectable

**Research note, November 2026.** This document distinguishes an
implemented mechanism, a reproducible evaluation protocol, and a hypothesis.
It is not a benchmark result, a universal-computation theorem, or a claim
that an arbitrary pretrained neural network contains a recoverable program.

## What matters beyond another working demo

The important development in VisuaML is not that a neural construction can
perform an addition or simulate a single WASM program. Those are individual
functions. The more interesting claim is **reuse of a computational mechanism
across different programs**, with a clear distinction between the program
and the machine that executes it.

A conventional virtual machine is programmable because its decoder,
registers, stack and transitions have stable meanings, even when the source
program changes. VisuaML is progressively constructing corresponding
mechanisms out of exact binary, ReLU and bilinear operations. Program-counter
routing, frame transitions, frame-event choice, operand storage, arithmetic
selection and source-bound microoperation fetching have each been made
explicit. A fixed circuit can therefore perform more than one useful
computation without training new weights for every program.

That is a meaningful architectural achievement inside the **reviewed,
bounded WASM subset**. But it is not yet a demonstration of an unbounded
universal computer. The state spaces are finite, source admission is
restricted, host operations remain, and the machine is implemented by CPU
code evaluating constructed mathematical circuits.

## Four scientific hypotheses worth testing

### H1 — Program/machine separation

A stable set of gate coefficients can execute genuinely different source
programs when those programs are provided as one-hot opcode, immediate,
branch and address tokens. Replacing the source program should change the
trace without changing the interpreter's fixed weights.

**Falsifier:** a new program requires rewriting gate coefficients, adding
hidden program-name conditionals, or bypassing gate outputs with host
selected answers. Evidence: hash the weights once per suite, hash each
distinct canonical WASM source, and record live per-instruction traces.

### H2 — Compositionality across state domains

Arithmetic, operand movement, local registers, and control flow should
compose with the same decoder and the same typed state contract. The
hard case is not an isolated gate's truth table but consecutive transitions
that simultaneously change several state domains.

**Falsifier:** an instruction's correct result depends on a program-specific
host patch, an inconsistent local/stack ordering convention, a stale
inactive operand slot, or an unobserved host modification to the PC.
Evidence: differential state transitions, depth sweeps, and adversarial
source programs generated independently of the executor.

### H3 — Mechanism falsifiability

If a particular gate owns execution, corrupting its output should be
observable as an explicit error or a change in the machine's trace.
This is stronger than merely recording a hash of the gate weights.

**Falsifier:** a corrupted coefficient produces a passing trace because
the host recomputes the expected state, or the test never executes the
instruction controlled by the mutated gate. Evidence: activated fault
campaigns with preserved undetected cases.

### H4 — Costs and optimization opportunities

Compositionality has a price. The current shared ALU calculates every
candidate computation before selecting one. Consequently, it may spend
more CPU work than a conventional opcode switch. The correct research
question is whether the fixed architecture offers opportunities for
batching, sparsity, compilation, specialized hardware, amortization or
proof-producing transformations—not whether it is already faster.

**Falsifier of a performance claim:** per-witness wall/CPU timing,
memory, compile cost or instruction-normalized throughput fails to improve
under a controlled comparison. A correctness gain is still scientifically
useful even if performance regresses.

## What the v58 benchmark suite needs to answer

| Dimension | Primary measurement | Avoid confusing it with |
| --- | --- | --- |
| Semantic equivalence | Original WASM and per-step state parity | Final-output-only agreement |
| Generality | Distinct validated programs, held-out source generators | Universal WASM coverage percentage |
| Compositionality | Exact state parity across multiple sequential effects | Individual gate truth tables |
| Runtime | Paired latency / CPU time; normalized per instruction | Neural hardware acceleration |
| Setup and representation | Tokenization cost, program token bytes, circuit bytes | Inference runtime alone |
| Resource scaling | Operand depth, frame depth, program length, mutation sensitivity | Asymptotic complexity proof |
| Fault sensitivity | Detected / activated corruptions, including undetected list | Formal robustness guarantees |
| Provenance | Git tree, source hashes, immutable raw evidence | Passing CI without matching source |
| Mechanistic interpretability | Effect of targeted interventions on actual state | Unique identification of arbitrary learned weights |

A benchmark should be able to embarrass the machine. An honest regression
is useful information; a hidden fallback is a methodological failure.

## Three comparison regimes

**A. Same semantics, different implementation:** run original Wasmtime,
v51 host control, v56 selected ALU, v57 microoperation engine, and v58
unified microstep where admitted. Do not assert a speedup until they
produce identical results at a matched set of source/input cases. Compare
the same engine across two commits and also compare engines on the same
host. Report p50, p95, worst cases, program preparation, and median time
per executed instruction.

**B. Same interpreter, different programs:** change source WASM while
holding the fixed coefficient digests constant. Vary arithmetic mixes,
branching, nesting, locals and program length. Hold out entire programs,
not just fresh inputs to known source. A 100% success rate on a tiny
reviewed subset does not establish generality outside it.

**C. Same program, corrupted mechanism:** deliberately change fixed
weights, legal-shaped token encodings, top-slot addressing, branch
restoration and event selection. Ensure each fault is activated by an
executed instruction. Score explicit rejection, silent wrong answers,
undetected equivalent perturbations and test-harness errors separately.

## Engineering consequences

The most compounding architectural unit is a *single typed state
transition*, not a list of individual opcodes. A new instruction should
declare its operand arity, immediate type, local/memory effect, trap
conditions, destination semantics and control effect. The compiler can
then lower it into the same canonical machine state, and independent
oracles can verify the result. This is the direction of v58.

The second compounding unit is an *evidence interface*. Every experiment
should be reproducible locally, tied to exact original source bytes and
Git provenance, and compared through the same evaluation dimensions.
That gives future contributions a stable admission gate instead of a
new, manually written success story.

The third is an *honest coverage ledger*. Count tested feature subsets,
independent binaries, instructions and failures by pinned WASM spec.
Do not turn an arbitrary list of selected examples into a claim about
overall standards coverage.

## Boundaries that are deliberately left open

- A finite constructed bit-gate virtual machine does not by itself prove
  Turing universality, bounded program synthesis beyond the admitted
  grammar, or general learned-network decompilation.
- WASM itself has memory, calls, tables, traps, imports and numerous
  numeric instructions absent from the current admitted subset.
- The present gates run on NumPy and Python; this is not evidence of
  native tensor-core, optical or photonic acceleration.
- The pretrained-network intervention studies and the constructed
  neural-WASM emulator answer different questions. Their findings should
  cross-pollinate through common typed IR and independent verification,
  but must not be conflated into evidence for the same claim.
- No new numerical performance or v58 correctness results have been
  reported until a real offline run produces source-bound evidence.

**The near-term research criterion:** can we keep increasing the number
of independently validated programs and state effects *while reducing
the amount of execution semantics encoded as host-specific dispatch*?
If so, the library becomes a stronger foundation for compiler research
and mechanistic analysis, regardless of whether its first interpreter
wins a CPU throughput race.
