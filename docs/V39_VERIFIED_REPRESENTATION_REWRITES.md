# v39 — First verified representation rewrites: WASM section bytes and neural hidden-unit layouts

**Status:** bounded research slice executed on CI. Not a general WebAssembly optimizer, neural decompiler, or universal neural VM.

## What was independently executed

The [dedicated GitHub Actions run #44215559927](https://github.com/Caerii/VisuaML/actions/runs/44215559927) passed **27 tests** and Ruff. It used pinned `numpy==2.2.6`, `pytest==8.4.2`, `wasmtime==28.0.0`, `ruff==0.17.5`, and a CPU PyTorch dependency required by the existing package import boundary.

### A. Direct binary WASM custom-section reordering

`visuaml/compiler_lab/wasm_bitstream_v39.py` supports strict bounded WASM binary v1 envelope parsing. It requires exact WASM magic/version, bounded canonical u32 LEB128 section lengths, at most 128 sections, known core section identifiers, unique and ordered noncustom sections, valid UTF-8 custom section names, and no truncated sections. It can deterministically relocate the **entire opaque custom sections** to the front or back, preserving their internal bytes and relative custom-section order.

The transformed executable core section bytes are **bit-for-bit unchanged** and their original order is preserved; the output record contains source and target byte hashes, exact sizes and concatenated standard/custom section SHA-256 fingerprints. Known malformed encodings, duplicate sections, out-of-order sections, invalid custom section UTF-8, oversized lengths, bad magic and noncanonical LEBs are rejected.

A separate pinned **Wasmtime** runtime executes a minimal actual exported WASM function `answer() -> i32` before and after both reorders and obtains the same `42` result. This is independent execution for *that fixture*, not a proof that every WASM program is valid or that unknown custom-metadata consumers are unaffected. The binary-transform receipt intentionally reports `wasm_module_typecheck_or_runtime_validated: false` because the standalone library does not itself run a validator. The CI test executes Wasmtime separately.

### B. Invertible two-linear-layer ReLU hidden-unit reordering

`visuaml/compiler_lab/reversible_weight_layout_v39.py` implements this exact-real arithmetic identity:

```text
f(x) = W2 ReLU(W1 x + b)
W1' = P W1
b'  = P b
W2' = W2 P^-1
W2' ReLU(W1' x + b') = f(x)
```

The transformation requires bounded finite float32 dense matrices with one connected hidden axis, exact permutation of all hidden units, and no untracked aliases/side effects. The implementation guards every shape, dtype, duplicate/out-of-range/bool permutation entry and nonfinite value before changing weights. It computes full source/target SHA-256s, permutation and inverse maps and proves **byte-exact weight recovery** by applying the inverse.

The independent test suite executes **20 seeded exact-integer circuits** (integer-valued weights and a separately implemented integer activation/reduction) to establish that two representations yield exactly equal outputs in those cases. It then checks finite float32 equivalence on more seeded random inputs with declared tolerances, never claiming bitwise identical floating-point output. Unsupported bias remappings, fanout/residuals, shared weights, attention cache state, quantization and arbitrary model structures are deliberately not admitted.

## Why this is not yet neural decompilation

The new neural reordering function starts with a known semantic circuit and produces another representation with a recorded inverse. That is a **constructive and reversible transformation**, not recovery of a program from unknown trained weights.

A future restricted neural decompiler should start with the exact analytically constructed transformer VM structure, recover its instruction decoder, state slots and address/pointer patterns by inspecting known-form weights, propose a typed state-transition machine, and test it against the original VM over adversarial programs. An arbitrary learned transformer generally does not determine a unique original algorithm or source program.

## Next proofs to pursue

1. Replace envelope-only WASM parsing with independent module validation and richer sections/import/type validation before **any executable instruction** rewrite.
2. Add typed local-index renumbering with correct use-site updates, new serialization and external Wasmtime differential traces (output, state, traps, host effects).
3. Add source-bound exact/tolerance certificate support for the hidden-unit permutation pass inside the shared Semantic IR transformation history.
4. Validate reverse-extraction of analytically constructed fixed-weight VM circuits from the two cited research systems, explicitly separating manifest-assisted from manifest-free extraction.
5. Study wasm-to-analytic-neural VM compilation and neural-to-WASM decompilation only under supported finite state/ISA assumptions; preserve counterexamples.

## Reproduction

```bash
cd visuaml-client
python -m pytest -q backend/tests/test_verified_representation_rewrites_v39.py
# To execute Wasmtime differential validation, install wasmtime==28.0.0.
```

The [GitHub issue #23](https://github.com/Caerii/VisuaML/issues/23) tracks the broader research. The canonical normative semantic reference is the [W3C WebAssembly Core Specification](https://www.w3.org/TR/2026/CRD-wasm-core-2-20261012/).

None of the synthetic v39 results increases VisuaML's pretrained-family count or completed roadmap acceptance percentage. These are two new bounded **representation-transform** evidence classes, not benchmarks of whole-model serving performance.
