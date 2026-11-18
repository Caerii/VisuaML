# v49 — Fixed-weight bit-sliced i32 execution: modular arithmetic, exact width, bounded WASM

## Scientific claim

v48 demonstrated **programmability**: seven distinct source WebAssembly programs supplied as token data were run with the **same analytic neural routing matrices**, without retraining or recompiling interpreter weights. But v48 represented values as scalar float64 and rejected computations exceeding a finite safe-integer budget. This meant it could not implement the essential WASM i32 two's-complement modular arithmetic contract over its complete signed value range.

v49 replaces scalar-number registers with **32-bit binary activation vectors**. Each bit is a float64 value exactly 0 or 1. One-hot program-counter, stack-depth and local-variable routing still shares fixed analytically generated weights across distinct programs.

The goal is not to quietly emulate WASM through Python integers. **Arithmetic is performed through actual ReLU threshold and bilinear gates on binary activation tensors**, with no integer addition, subtraction, multiplication, shifts or modulo executed in the neural instruction transition itself. Integer conversion occurs at the *input/output ABI* and in the independent verification oracle only.

This is **not a standard transformer** with learned or constructed softmax Q/K/V layers, nor a general WebAssembly runtime. It is a bounded and inspectable neural computation circuit approaching the machine semantics required by a later transformer lowering.

## Mathematical constructions

All admitted numeric activations are binary, and the gate weights are fixed:

- Boolean AND: (a\land b=\operatorname{ReLU}(a+b-1)).
- Boolean XOR: (a\oplus b=\operatorname{ReLU}(a-b)+\operatorname{ReLU}(b-a)).
- Boolean OR: (a\lor b=a+b-(a\land b)).
- Full-adder at bit i: (n=a_i+b_i+c_i); (c_{i+1}=\operatorname{ReLU}(n-1)-\operatorname{ReLU}(n-2)), (s_i=n-2c_{i+1}). All intermediates are exact small integers.
- Subtraction: 32-bit ripple addition with a complemented operand and carry-in 1.
- Multiplication: (p_{ij}=\operatorname{ReLU}(a_i)b_j) produces a 32×32 bilinear partial-product grid (ReGLU-style). For each bit column k, sum (n=c_k+\sum_{i+j=k}p_{ij}) and calculate (c_{k+1}=\lfloor n/2\rfloor) using a **fixed ReLU staircase** over even thresholds 2,4,…,64. The emitted bit is (s_k=n-2c_{k+1}). Discard the carry after bit 31 for modular multiplication.
- Signed i32 comparisons: lexicographic unsigned comparison of bit vectors, corrected by the sign bits; i32.eqz uses the product of 32 zero-bit predicates.

The 0/1 invariant and the explicitly bounded carry values make the numerical gate operations exact under binary64; no billion-sized float64 integer accumulations are used. However, testing an implementation's arithmetic formula on finite inputs is not the same as establishing a **machine-checked all-input correctness theorem**.

## Program and execution boundary

The same fixed interpreter processes the reviewed canonical v40 WASM format:

- One exported run(i32)->i32 function.
- Up to 32 straight-line instructions; up to four nonparameter i32 locals and stack depth up to 16.
- Instructions: i32.const, local.get/set/tee, add/sub/mul, signed lt/gt, eqz, drop and end.
- Any **signed 32-bit input**, including `INT_MIN` and `INT_MAX`. Unlike v48, i32.const may carry any signed-32-bit value.
- Opcode/immediate/local tokens are supplied as separate program matrices. Fixed routing weights and ReLU gate coefficients do not depend on the chosen program.
- The VM's raw bit-vector stack and local registers are inspected after every step, compared to the separately implemented typed v40 reference.

The underlying WASM language still excludes linear memory, control-flow branches/blocks/loops, calls, globals, imports, tables, host I/O, traps, i64, floating-point arithmetic, SIMD, GC, references, threads, WASI, and other extensions. **Full 32-bit arithmetic on a narrow ISA does not imply full WASM compliance.**

## Independent evidence and negative controls

The dedicated workflow validates 11 distinct canonical source programs with an original real Wasmtime engine: identities, overflow addition, borrow subtraction, signed full-width constants, full-width multiplication, comparisons, zero tests, local assignments/tee and stack drop. It checks adversarial edge values such as `INT_MIN`, `INT_MAX`, -1 and 0, plus seeded random full-width i32s. Every neural step is checked for PC, stack depth, all live stack values, all declared locals and halt state against the original v40 interpreter.

The Boolean gate truth tables are exhaustively tested, including overflows and signed comparison boundaries. The **full 2³² input space is not enumerated**, and this phase does **not** claim all-input SMT equivalence of the entire neural circuit.

The complete CI report binds canonical original binary bytes, every program token digest, one common fixed-weight digest, environment versions, full output witnesses, boundary-case per-instruction state traces and the SHA-256 of the exact executing source files. A passing report must be stored permanently as the original raw bytes with a distinct CI-origin manifest, never silently updated.

## Implications for decompilation

The fixed-weight neural interpreter can be analyzed without any training data. The program is disassembled from a **known token encoding** and checked against its original source hash. That is a structural decoding of an admitted format—not general model decompilation.

V48 separately used SMT to demonstrate that two different WASM programs can compute exactly the same function for every signed 32-bit input. Therefore unique original-program reconstruction from external behavior is impossible in general; the correct target for a decompiler is **a minimal typed representation within a declared behavioral equivalence class**, with explicit counterexample and ambiguity records.

## Next frontier

The next highest-leverage WASM semantics are not simply increasing the opcode count:

1. **Control-flow semantics:** typed block/loop/if/br/br_if/br_table, an explicit control stack and PCs for branch targets, fuel for nonterminating programs, validator-informed stack polymorphism, and independently verified traps.
2. **Memory:** memory pages, little-endian load/store, bounds/traps, offsets, memory.grow, data segments and side effects.
3. **Calls and imports:** module functions, call stacks, index spaces, indirect calls, tables, host-interface and effect boundaries.
4. **Additional values and extensions:** i64 bit slicing, IEEE 754 f32/f64 exact bit semantics, conversions, sign extension, SIMD, reference types, GC, exception handling, threads/atomics, memory64, and standard versioned conformance.
5. **True transformer construction:** synthesize Q/K/V, ReGLU feedforward and residual matrices representing the verified machine-state rules; compare layer intermediates to the typed oracle. Fixed one-hot routing alone does not satisfy this criterion.
6. **Blinded semantic decompilation:** remove the construction manifest, expose weights/activations alone, recover and minimize candidate state-transition programs, and check them against independent oracle traces and interventions. Unknown learned networks require separate identifiability and causal tests.

The key asymptotic scaling discipline is to **reuse the same independently verified machine-state contract across native WASM, the bit-gated VM, future transformer circuits and symbolic/KAN components**, so every new supported behavior yields reusable evidence rather than another ungrounded demo.

## Reproduce

```bash
PYTHONPATH=visuaml-client/backend python -m pytest -q \
  visuaml-client/backend/tests/test_bit_neural_vm_v49.py

PYTHONPATH=visuaml-client/backend python \
  visuaml-client/backend/scripts/probe_bit_neural_vm_v49.py \
  --output /tmp/bit-neural-i32.json
```

CI workflow: `.github/workflows/bit-neural-vm-v49.yml`. This is a stacked draft research branch; `main` is unchanged.
