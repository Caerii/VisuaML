# Compiler Lab v4 — opt-in React integration

The `/compiler` route is a new page in the existing React/Vite VisuaML app. It consumes an authenticated **by-capability, fixture-only** backend endpoint, not arbitrary uploaded Python code. The page provides transform stages, an interactive Dagre-laid-out ReactFlow graph, operation metadata, per-pass numerical receipts, and a hypothetical host/device cost sweep.

## Apply

1. Copy the contents of this overlay into your VisuaML checkout (file paths are repo-relative). Keep the existing `App.tsx`, `MainApp.tsx`, `TopBar.tsx`, and `server/index.ts` intact; the installer changes these through checked, small insertions.
2. Run the idempotent integration installer from repository root: `python apply_compiler_lab_v4.py --check` followed by `python apply_compiler_lab_v4.py`. It makes three narrowly anchored edits to existing files and refuses unexpected layouts. No existing file is overwritten by the source overlay.
3. Install the project's existing JS and Python dependencies. The compilation layer uses PyTorch and NumPy, as in v3; the UI uses existing `dagre` and `@xyflow/react` dependencies.
4. Run API server with `VISUAML_COMPILER_LAB_ENABLED=1` and valid `VISUAML_PYTHON` for your environment. Then run Vite dev server and open `/compiler` (or click `Compiler Lab` within `/app`).
5. Run Python verification from repo root: `PYTHONPATH=visuaml-client/backend python -m pytest visuaml-client/backend/tests/test_compiler_lab*.py`.

## Boundaries

- The default workflow permits only four locally defined fixture scenarios; this is intentional until model upload has isolation, input sanitization, bounded execution, and review.
- `NumPy AOT` and `.pt2` are **actual software artifacts**, but hypothetical device assignment is **not executable hardware lowering**.
- The sparse-neural selection test captures a single k-WTA step; it does not establish parity with the recurrent `neural-assemblies` Brain runtime.
- The API endpoint must be explicitly enabled and is not production-hardened for untrusted internet exposure. It does not authenticate users. Restrict network access until authentication and rate-limiting are added.
- Existing collaborative canvas data is not overwritten. Compiler analysis is a separate route with its own state; future work can connect validated stage graphs to Yjs on explicit user action.
