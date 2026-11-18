# v40 — Typed WASM i32 transition IR and executable binary rewriting

Research status: a bounded, independent experiment. This is not a universal WebAssembly interpreter, a neural VM or a formal proof of arbitrary binary equivalence.

## Relationship to prior work

v39 safely relocated opaque custom sections, without changing any executable bytes. v40 introduces an actual change to the body of a WebAssembly function, under a much narrower admission contract. It creates a source-bound stack/local-state representation and an independent execution trace. The existing VisuaML neural-assembly stateful IR is intentionally algorithm-specific; this new machine dialect does not pretend that its weights/winners effects are WASM locals.

## Supported subset

- Canonical WASM 1.0 binary, exactly one exported run(i32)->i32 function
- Exactly type, function, export, code sections; no custom, imports, tables, globals or memory
- 0–16 nonparameter i32 locals, a single homogeneous declaration group
- No loops, branches, calls, host IO, floats, exceptions or extensions
- At most 64 typed instructions and 64 stack values
- i32.const, local.get/set/tee, add/sub/mul (wrapping i32), eqz, lt_s, gt_s, drop and end
- Canonical signed and unsigned LEB128 immediates; fail closed on malformed input

The parser typechecks stack height, local slots and the single i32 result before accepting the binary. Execution records every instruction's stack and local snapshot, any local write, and a deterministic trace digest.

## The alpha-rewrite

A permutation covers only the nonparameter i32 locals. The parameter remains local 0. Each get/set/tee access uses the inverse permutation's new slot, while local declarations and all other instructions are preserved. The result is new canonical executable WASM bytes. Applying the inverse yields exactly the original bytes. The typed checker re-admits both programs.

This is sound by a conventional alpha-renaming argument for the declared closed world: no instruction can reflect on a local's index and all permitted renamed locals have the same type/zero initialization. The receipt does not claim a machine-checked general WASM proof or external runtime validation.

## Independent evidence

The dedicated v40 CI executes source and target binaries via Wasmtime 28.0.0 and checks both against the separately implemented immutable typed interpreter for **all 256 signed eight-bit inputs and four boundary i32 inputs**. The evidence JSON includes the complete original/modified module bytes in hex, source/program digests, original/new outputs, full per-witness transition trace digests, tool version and source checkout identity.

The exhaustive 8-bit domain is not exhaustive for 32-bit input space. Failure to preserve any output invalidates the witness. There is no neural-network execution, WASM memory effect or arbitrary binary mutation in this phase.

## Reproduction

Install pytest 8.4.2, wasmtime 28.0.0, and ruff 0.17.5. From repository root:

    PYTHONPATH=visuaml-client/backend python -m pytest -q visuaml-client/backend/tests/test_wasm_state_vm_v40.py

    PYTHONPATH=visuaml-client/backend python visuaml-client/backend/scripts/probe_wasm_state_vm_v40.py --output /tmp/typed-wasm-v40.json

GitHub CI workflow: .github/workflows/typed-wasm-state-v40.yml.

## Scientific continuation

Independent verification and immutable archiving precede expansion. Next, lower the *same typed state transition rules* into a bounded analytically synthesized neural circuit, compare instruction-by-instruction traces with Wasmtime, and attempt reconstruction of the instruction rules without reading the construction manifest. Even a successful constructed-circuit recovery would be restricted neural decompilation, not recovery of unique original programs from arbitrary learned weights.
