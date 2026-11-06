# VisuaML Compiler Lab v3 — patch release

Experimental and opt-in compiler research implementation for `Caerii/VisuaML`. **This is an overlay**, not a complete Git checkout, and it does not modify the production VisuaML app unless you manually integrate it.

Includes all v2 features plus: strict portable IR / resource validation, exact binary min-cut host/accelerator placement under *hypothetical* cost profiles, independent 2-D convolution execution, restricted ATen convolution lowering, 25 additional v3 tests, interactive cost/placement explorer, and documentation.

**Read:** [`docs/COMPILER_LAB_V3.md`](docs/COMPILER_LAB_V3.md) and [`docs/COMPILER_LAB_V2.md`](docs/COMPILER_LAB_V2.md).

**Test** from repository root:

```bash
PYTHONPATH=visuaml-client/backend python -m pytest -q \
  visuaml-client/backend/tests/test_compiler_spike.py \
  visuaml-client/backend/tests/test_compiler_lab.py \
  visuaml-client/backend/tests/test_compiler_lab_v3.py
```

**Demo:** `PYTHONPATH=visuaml-client/backend python visuaml-client/backend/examples/compiler_lab_demo.py --scenario convolution --out ./out/convolution`

**Important:** The NumPy standalone deployment now needs both `numpy_runtime.py` and `ir_contract.py`. Moving only the old runtime file will fail. Hashes are integrity fingerprints, not signatures; execution is not an untrusted-code sandbox. The hardware cost optimizer does **not** create a physical or photonic executable.

GitHub branch from the earlier v1 work is not synchronized with this local v3 patch. The source of truth for this release is this ZIP plus SHA-256 file hashes.
