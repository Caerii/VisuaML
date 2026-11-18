# Neural computation, decompilation, and verified WebAssembly rewriting

*Research design note for VisuaML. These directions are not claimed as existing product features or proved compiler equivalences.*

## Thesis

The same computation can have several representations: source code, WASM binary, a typed control/dataflow graph, a purely numerical circuit, constructed transformer weights, and an execution trace. VisuaML's increasingly typed and provenance-bound Semantic IR is the prospective interchange point between them.

There are two critically different scientific tasks:

1. **Compilation into a neural machine**: start from known machine semantics, generate a network whose forward passes realize the instruction transitions, and record a replayable proof obligation. Analytically synthesized transformer weights are constructive examples.
2. **Decompilation from a neural machine**: infer a smaller inspectable algorithm or interpreter from tensors/weights/activation traces. A *constructively generated* network with a known template can often be structurally reversed; an *arbitrarily learned* network typically admits many nonequivalent descriptions outside a finite probe set. We must not promise recovery of unique original source.

A third capability, **validated binary rewriting**, transforms real WASM bytes while preserving or explicitly changing behavior under a declared observational contract.

## What the cited projects actually implement

- [NarendraPatwardhan/vm-in-transformer](https://github.com/NarendraPatwardhan/vm-in-transformer): 12 layers for each instruction cycle, analytically generated attention/MLP matrices, 16-opcode custom stack VM with a 64-cell stack, 128 memory slots and 256 program slots. Its instructions are *WASM-inspired*, not a complete WebAssembly binary parser/runtime. Mathematical constructions are bounded by numerical saturation and integer range assumptions.
- [Percepta-Core/transformer-vm](https://github.com/Percepta-Core/transformer-vm): a richer WASM-subset frontend with a binary decoder, lowering of operations such as multiply and shifts, a programmable transformer graph, analytical weight materialization, specialization and a C++ autoregressive engine with a convex-hull attention optimization. Its exposed 35 opcodes, reference traces and tests are more relevant than an unqualified assertion that all WebAssembly modules execute.
- [Giannou et al., *Looped Transformers as Programmable Computers* (ICML 2023)](https://proceedings.mlr.press/v202/giannou23a.html): foundational constructive computability of looped fixed-weight transformer architectures.
- [WebAssembly Core Specification](https://www.w3.org/TR/2026/CRD-wasm-core-2-20261012/): the normative binary layout, validation and execution semantics against which our parser, transformations and proofs must be checked.

Their performance reports are author-reported and cannot be substituted for independent third-party benchmarking.

## Four levels of neural decompilation

### Level 0 — inspect and round-trip the known construction

Given a compiler-generated weight package **and its provenance manifest**, reconstruct the same virtual-machine IR from the original declared schedule and template. Compare weight tensor shapes/hashes, instruction encoding, state layout and lossless serializations. This is reversible **representation recovery**, not learned-network reverse engineering.

### Level 1 — recognize analytically constructed circuits

Given weights *without* the manifest, detect exact or bounded-tolerance patterns (opcode decoders, finite-position lookup attention, bit tests, pointer writeback, compiled ReGLU circuits). Propose a candidate VM control graph; independently execute it on adversarial traces and compare observable transitions. Refuse on ambiguous/mixed circuits.

### Level 2 — recover computations from observed traces

For a network with unknown weights, capture activation causal graphs, test interventions, infer control/data dependencies, and synthesize a candidate typed executable DSL. Verification is bounded to explored domains unless formal proof discharges equivalence. Return counterexamples and confidence/coverage, never a single magically reconstructed 'original program'.

### Level 3 — formal extraction or synthesis

For a bounded rational/integer circuit, encode observable behavior with SAT/SMT/finite-state model checking and synthesize an equivalent symbolic machine where solvable. Arbitrary unrestricted behavior equivalence and unique program recovery are **not** generally decidable or well-defined.

## Direct weight reordering — an early mathematical exactness target

There is a tractable family of genuinely useful 'neural bitstream' transformations that does not require guessing semantics. For any permutation matrix `P` and a pointwise activation `σ` commuting with permutation:

```text
y = W₂ σ(W₁ x + b₁)
W₁' = P W₁
b₁' = P b₁
W₂' = W₂ P⁻¹
then W₂' σ(W₁' x + b₁') = y  (exact real arithmetic)
```

The *weight bytes and internal unit order change*, but the represented mathematical function is identical. With properly rewritten connected layers, analogous channel/head permutations are candidates for convolution and attention. Arbitrary bit shuffles are **not** covered. Real floating-point execution can show changed roundoff order, so receipts must distinguish exact real-number algebra from finite numerical checks.

Proposed first artifact: a deterministic `permute_hidden_units` transformation emitting a paired inverse-channel map, source/target weight hashes, typed shape proof, 100+ adversarial random/boundary witnesses, and independent exact arithmetic check for small integer weights. Reject invalid permutations, missing bias remaps, residual aliasing, state-carrying layers and uncontrolled fanout.

## Real WASM binary rewriting — a stricter problem

WASM is **not** a free-form bitstream. Section IDs/order, LEB128 integer encodings, function/type/index namespaces, table/data/global references, control-flow labels, import/export signatures, instruction encodings and code-body lengths all carry semantics.

The W3C format permits *custom sections at different positions*, but noncustom sections have prescribed ordering. Even moving custom sections is only semantically transparent to the core VM, not necessarily to third-party tooling that interprets custom metadata.

### Recommended typed transformation pipeline

```text
.wasm bytes
  ↓ cryptographic source identity + bounded parser
validated module with explicit imports/exports/types/memories/call sites
  ↓ typed stack-to-SSA lowering
WasmSemanticProgram + side-effect/memory/capability contracts
  ↓ verified/guarded rewrite
candidate new semantic module + transformation certificate
  ↓ correct binary serialization (LEB, indices, sections, sizes)
new .wasm bytes
  ↓ independent wasm validator + 2 different runtime traces
pass / reject / bounded-evidence-only with counterexamples
```

Concrete early transformations in ascending risk:

1. **Lossless round-trip**: decode and re-encode known WASM MVP modules, proving the parsed AST is unchanged; byte identity where canonical encoding policy requires it.
2. **Custom-section relocation**: move opaque custom sections without interpreting them, prove standard-section byte equality and record metadata-order changes.
3. **Scoped local renumbering**: reorder function-local indexes together with every `local.get/set/tee`, preserve parameter and type contracts, and compare traces.
4. **Block-level algebraic normalization**: only explicit arithmetic identities under WASM integer/trap/wrap rules, followed by independent runtime execution and SMT for bounded cases.
5. **Function/code reordering**: update all index-bearing references (imports, exports, start, calls, tables, elements, names, relocations where in scope). Never byte-swap function bodies and call it semantics preserving.
6. **Neural VM target**: lower the admitted WASM subset into a declared state-transition DSL; compile that DSL into analytically constructed transformer weights; verify step-by-step against WASM reference traces.

### Necessary evidence

For each rewrite, record original/new raw bytes SHA-256, parser/validator/toolchain SHA, typed AST/IR digests, before/after byte lengths, index remapping table, exact changed spans, differential test inputs, all outputs/traps/memory state and fuel usage, negative cases, floating-point caveats, and at least one independent WebAssembly engine. For side-effecting programs, observational equivalence includes memories, host calls and traps—not only return values.

The **neural VM is another backend**, not permission to ignore WASM semantics. Each new opcode should have a pure or effectful state-transition contract and a small exhaustive model checker where practical.

## Research priority order

- **P0:** complete the v38 real CNN through the existing graph-generated native compiler, preserving old Llama and v37 evidence.
- **P1:** add a proof-carrying hidden-neuron permutation pass, bounded exact witnesses and tensor-layout receipts; this first demonstrates nontrivial, semantics-preserving *binary weight reordering*.
- **P2:** add a fail-closed WASM parser and validator binding with lossless round-trip and safe custom-section relocation; preserve original and transformed bytes.
- **P3:** implement a typed i32 stack-machine subset and a small interpreter trace oracle; generate and verify one constructed neural VM backend.
- **P4:** instrument and structurally reverse our own constructed neural VM, then challenge it with manifest-free recognition and controlled counterexamples.
- **P5:** only then explore neural program synthesis for unknown trained networks, multi-backend optimization and architecture-search feedback.

## Acceptance discipline

None of these goals can be marked 'achieved' from a repo README, a synthetic fixture, a hand-constructed trained-weight claim, a single green run, or a website deployment. Verify each independently against a declared observable contract. An exact finite model-checking result and a set of numerical witnesses are different evidence classes.
