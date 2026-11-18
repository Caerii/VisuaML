# v48 — A single constructed neural-state machine executing multiple real WASM programs

**Scientific milestone and exact claim:** seven distinct real, canonical WebAssembly modules are represented by separate input **program tokens**, while their execution shares one analytically constructed fixed set of one-hot routing and ReLU/ReGLU-gated state-transition matrices. A separate original typed i32 interpreter checks the *complete state after every instruction*; independent Wasmtime executes the original source WASM bytes. This is a **bounded constructed neural instruction machine**, not yet a standard softmax transformer or a full WebAssembly runtime.

## Why this is a distinct advance beyond v41–v47

v41 constructed an analytic ReLU circuit for a **single** finite function. v43 recovered a typed symbolic polynomial and proved its equivalence to the original source i32 program using Z3, while preserving the neural-domain counterexample. v44 synthesized a different two-unit ReGLU circuit for the same symbolic function. v45 introduced typed KAN/MultKAN edge computations. v46–v47 advanced pretrained activation causality and spatial counterfactual patching.

A true neural virtual machine cannot require rebuilding weights for each program. v48 explicitly separates *program as data* from *fixed interpreter parameters*. This is necessary for an analytically constructed universal interpreter. It is not sufficient for universal WASM.

## Exact v48 architecture

- **Program tokens:** canonical typed v40 WASM is decoded to 33 token slots (32 admitted source instructions plus one terminal slot). Each instruction has a 12-category one-hot opcode, optional signed integer constant, and one-hot local selector. There are no model-supplied code strings or dynamic Python eval operations.
- **Neural state:** program counter (33 one-hot positions); stack depth (17 one-hot positions); 16 scalar stack cells; five local slots (the argument plus up to four declared locals). All values are integer-valued float64 under an explicit magnitude budget.
- **Shared fixed matrices:** one-hot program counter shift; stack-depth increment/decrement; program fetch by matrix multiplication with the opcode/immediate/local token tables; current and second stack-element read heads; output/write target selection. These are reproducibly constructed from their semantic definitions, not optimized against outputs.
- **ReLU/ReGLU transitions:** all arithmetic candidates are evaluated in parallel, with one-hot instruction flags routing only the selected result and stack/local writeback. Multiplication uses the exact-real bilinear identity `ReLU(a)*b - ReLU(-a)*b = a*b`. Integer comparisons are implemented by bounded ReLU step functions. Machine and local updates use bilinear gates on values and one-hot selectors.
- **One executable neural step:** fetch, candidate arithmetic, stack writes/clears, local writes, depth movement and PC movement; no Python opcode switch inside `neural_step` dispatching the selected instruction. The exterior driver controls the number of steps and independently inspects states.

Every step is compared with the *separately implemented* v40 `step()` interpreter. This does not constitute a neural forward pass through standard multi-head **softmax attention** or conventional transformer residual blocks. Its selection is exact one-hot matrix routing. Neural inference speed has not been compared to Wasmtime.

## Explicitly admitted ISA

A single canonical `run(i32)->i32` function, straight-line, 2–32 instructions, up to four nonparameter i32 locals, stack depth no greater than 16, no module imports, globals, memory, tables, exceptions or control flow. Instructions:

`i32.const`, `local.get`, `local.set`, `local.tee`, `i32.add`, `i32.sub`, `i32.mul`, `i32.eqz`, `i32.lt_s`, `i32.gt_s`, `drop`, `end`.

Input interval: all 256 signed eight-bit values [-128,127]. Immediate magnitudes [-256,256]. All intermediate values must remain integral and within ±2²⁴. Any violation fails closed rather than silently applying float64 operations and claiming i32 modular wrap. In particular, **WASM's 32-bit modular overflow semantics are not implemented by the neural machine.**

Seven canonical source programs represent identity, affine addition, quadratic multiplication/subtraction, local get/set/tee, zero test, signed less-than, and signed greater-than with drop. For every admitted input, the same neural interpreter executes each program with unchanged weights. An independent Wasmtime engine executes all original source WASM modules as a separate output oracle.

## What is decompiled, and what cannot be

`detokenize_program()` reverses a **reviewed, known, canonical token format** back into v40 typed operations. This is token disassembly, not extraction of arbitrary algorithms from unknown trained transformer weights.

In fact, the v48 Z3 equivalence control proves that two different canonical programs—`local.get 0; end` and `local.get 0; i32.const 0; i32.add; end`—have the same observable return value over **all 2³² i32 inputs** under independently encoded straight-line bitvector semantics. This demonstrates that no algorithm can uniquely recover an original source program from its input-output function in general. A meaningful neural decompiler must target *equivalence classes of behavior and a chosen minimal representation*, not promise identification of a unique historical algorithm.

The SMT result applies to these typed source programs; it does **not** prove neural execution for all 32-bit inputs.

## Current WebAssembly standard and coverage frontier

The [WebAssembly Core 3.0 standard](https://www.w3.org/TR/wasm-core/) defines substantially more than a scalar stack calculator, including structured control flow, multi-function modules, calls, tables, linear memory, integer and floating types, SIMD and reference values. Actual runtime behavior includes validation, traps, initialization and imports; embedding interfaces such as WASI are additional contracts.

Do not calculate a misleading "percentage of WASM completed" from the number of admitted opcodes. Support must be tracked **by semantic feature, accepted value types, control flow, effects, corner cases and independent conformance witnesses**.

| Capability | v48 status | Next independent acceptance gate |
| --- | --- | --- |
| Canonical WASM 1.0 envelope and typed one-function binary | Bounded v40 support | Allow multiple functions/import declarations with index-safe typed validation |
| Straight-line i32 stack/local arithmetic | Fixed neural routing, seven programs; finite exact integer values | Full i32 modular carry/overflow with Wasmtime and SMT at boundaries |
| Structured block/loop/if/branch/call/return | **Not supported** | Typed control stack, fuel, branch target validation, loops/recursion and trap evidence |
| Linear memory with load/store/grow and bytes | **Not supported** | Exactly modeled endianness, OOB traps, memory pages and mutable effects |
| Globals/tables/call_indirect | **Not supported** | Typed runtime and index/table faults; independent reference oracle |
| i64/f32/f64, conversions, IEEE/NaN | **Not supported** | Format-aware exact bit patterns, NaN, signed zero, conversions and traps |
| SIMD, references/GC, threads, exceptions, memory64, tail calls, multi-memory | **Not supported** | Versioned feature-specific independent conformance suites |
| Standard transformer-layer synthesis | **Not supported** | Construct actual Q/K/V, softmax/ReGLU/residual layers and compare every intermediate to the typed state reference |
| Arbitrary pretrained weight-to-program decompilation | **Not supported** | Representational invariance, blinded recovery hypotheses, causal intervention and equivalence-class certificates |

The v48 fixed-weight microarchitecture is an intentionally small stepping stone toward the analytically constructed transformer VMs in [Percepta-Core/transformer-vm](https://github.com/Percepta-Core/transformer-vm) (Apache 2.0) and [NarendraPatwardhan/vm-in-transformer](https://github.com/NarendraPatwardhan/vm-in-transformer). Its code is independently written for VisuaML and does **not** copy code from the latter repository, for which no LICENSE was visible in this check. The Percepta repository's own README says its decoder implements 35 chosen opcode types, lowering some other arithmetic operations; it should not be automatically interpreted as 100% coverage of the WebAssembly 3.0 specification.

## Scientific acceptance protocol

1. Pin and archive canonical raw original WASM bytes and every program-token digest.
2. Record the fixed routing-matrix digest; assert it remains identical across every distinct program.
3. For every instruction, compare program counter, stack height/content, declared local variables and halt state to the independent v40 typed interpreter. Disallow compiler and oracle code sharing except immutable type contracts.
4. Execute original WASM bytes in pinned independent Wasmtime; cover all 256 signed-eight-bit inputs per program and additional adversarial checks. Treat NaNs, overflows and out-of-domain executions as hard failures/unsupported—not approximate success.
5. Keep raw per-input witnesses, selected full instruction traces, original binary bytes, source fingerprints, Z3 nonuniqueness result and environmental versions.
6. Archive the original CI transcript plus separate source-commit/byte-bound provenance. Never relabel an older experiment after changes.

### Reproduction

```bash
PYTHONPATH=visuaml-client/backend python -m pytest -q \
  visuaml-client/backend/tests/test_fixed_neural_token_vm_v48.py
PYTHONPATH=visuaml-client/backend python \
  visuaml-client/backend/scripts/probe_neural_token_vm_v48.py \
    --output /tmp/neural-multivm-v48.json
```

The dedicated GitHub workflow pins Python 3.12, NumPy 2.2.6, Wasmtime 28.0.0 and Z3 4.13.4.0. Research status stays **draft** until both dedicated and full-repo CI are green and the original evidence is permanently archived.

## Next best iteration

1. **v49 exact i32 modulo arithmetic:** bit-sliced registers with carry/borrow and explicit bounded bitwise multiplication; fail closed on all other types.
2. **v50 programmable control flow:** typed block/loop stack, jumps, fuel, trap representations, and independent instruction-trace conformance on GCD/Fibonacci/branching programs.
3. **v51 memory semantics:** 64-KiB page semantics in bounded slices; address, alignment, endianness, load/store and out-of-bounds trap witnesses.
4. **v52 transformer lowering:** generate a fixed set of Q/K/V softmax attention, ReGLU FFN and residual matrices implementing the tested micro-VM, and verify numerical precision separately from the idealized one-hot semantics.
5. **Decompilation science:** recover a *minimal equivalent* typed program from constructed weights and observed traces without the construction manifest; then challenge the approach on known-but-blinded and genuinely pretrained models. Report ambiguity and disprove false source-recovery claims.

The compounding result we're looking for is **a shared machine semantics contract across physical CPUs, WASM runtimes, constructed neural interpreters, and symbolic/KAN reconstructions**—not a magic universal percentage.
