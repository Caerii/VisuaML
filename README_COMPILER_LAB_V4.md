# VisuaML Compiler Lab v4 — real React/Vite integration overlay

This overlay extends [Caerii/VisuaML](https://github.com/Caerii/VisuaML), building on the v1–v3 compiler-lab prototypes. It does **not** include the whole main repository, and the source has **not** been pushed into GitHub.

## Added in v4

- New `/compiler` route inside VisuaML, using the project's existing React, ReactFlow, and Dagre dependencies, with stage navigation, graph node inspector, hypothetical cost sliders, and verification receipts.
- `/api/compiler-lab/analyze` Fastify endpoint gated by `VISUAML_COMPILER_LAB_ENABLED=1`.
- Safe allowlist of four local model fixtures (`attention`, `mlp`, `convolution`, `assembly`); no submitted arbitrary Python code is executed by the new endpoint.
- Executable cross-backend verification from the v3 backend: PyTorch/NumPy-AOT and a PyTorch `.pt2` roundtrip.
- An optional, **skipped when unavailable**, test for winner-set parity against the actual `Caerii/assemblies` `NumpyExactEngine._exact_topk` selection method. This is **not** parity for the stateful Brain/Area implementation.
- End-to-end endpoint contract tests and independent tie-robust selection tests.

## Install against an existing VisuaML checkout

From the top-level repository root, extract the ZIP. The archive adds new source files plus documentation, tests, and a small installer. It deliberately does **not overwrite** `src/App.tsx`, `src/MainApp.tsx`, or `server/index.ts`.

```bash
python apply_compiler_lab_v4.py --check
python apply_compiler_lab_v4.py
```

The installer checks all the original files and insertion anchors **before** making any changes and is idempotent. Commit to a new feature branch before applying. Review the changes using `git diff`.

Then run the existing setup (`pnpm install` in repository root; Python requirements as defined in the repository). Start your API with `VISUAML_COMPILER_LAB_ENABLED=1` and `VISUAML_PYTHON` pointing to its PyTorch-enabled Python interpreter. Start Vite as normal. Open **`/compiler`** or follow the new **Compiler Lab** link in `/app`.

For a standalone API check without JavaScript dependencies:

```bash
python visuaml-client/backend/scripts/compiler_lab_service.py --scenario attention --seed 19 > report.json
```

For Python regression:

```bash
PYTHONPATH=visuaml-client/backend python -m pytest -q \
  visuaml-client/backend/tests/test_compiler_spike.py \
  visuaml-client/backend/tests/test_compiler_lab.py \
  visuaml-client/backend/tests/test_compiler_lab_v3.py \
  visuaml-client/backend/tests/test_compiler_lab_v4.py
```

To enable comparison with the **actual** `neural-assemblies` package (separate repository), install that package into the same Python environment, then rerun the optional test; it is a *single selection-operation* parity test, not full network dynamics parity.

## Verification performed in authoring environment

- 74 Python tests passed, 1 optional actual-engine parity test skipped because `neural_assemblies` was not installed.
- Fixed scenario API checks succeeded for attention, MLP, assembly selection, and convolution. These produce real CPU execution witness results.
- New TypeScript/TSX files passed TypeScript syntax-transpilation checks; **full dependency-resolved `pnpm build` was not run**, because Node package dependencies were not installed in the authoring environment.
- Feature installer was tested for dry-run checks, idempotence, and guarded edits on representative source anchors. The existing GitHub repository was not modified.

## Claims and limitations

- Compiler input is a subset of *captured tensor programs*, not all Python programs or arbitrary algorithms.
- Reference results and pass receipts are sampled checks, **not formal equivalence proofs**.
- Device placement uses *invented illustrative abstract cost units*; it is not actual Ayo hardware compilation, power, or latency.
- The feature-gated HTTP endpoint has no login, rate-limiting, or production isolation. Do not expose it to an untrusted public network.
- The new `/compiler` view has its own ReactFlow and local state. It does not yet share graphs into the existing collaborative Yjs canvas. Such mutation requires a separate, explicit verified-import workflow.
- The model and Python runner execute on the server. No sandbox for arbitrary uploaded models has been implemented.

**Next gates:** dependency-resolved JS build and browser e2e on a real checkout; actual-engine parity against a checked-out neural-assemblies package; user-reviewed safe export of compiler stages into the collaborative canvas; physically measured device cost calibration.
