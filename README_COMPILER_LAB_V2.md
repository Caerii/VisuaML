# VisuaML Compiler Lab v2 — patch release

This archive overlays onto the existing `Caerii/VisuaML` repository; it is **not** a full checkout and does not replace VisuaML's app. Existing files are left alone except for the updated opt-in `compiler_spike.py` from v1.

**Inside:** typed FX analysis, strict portable NumPy AOT, torch.export `.pt2` output, attention and softmax decomposition, linear decomposition/fusion, semantic checks, backend contracts, small canonical assembly-selection bridge, detailed JSON evidence, and an offline graph inspector.

**Documentation:** [`docs/COMPILER_LAB_V2.md`](docs/COMPILER_LAB_V2.md).

**Tests:** `PYTHONPATH=visuaml-client/backend python -m pytest -q visuaml-client/backend/tests/test_compiler_spike.py visuaml-client/backend/tests/test_compiler_lab.py`

**Demo:** `PYTHONPATH=visuaml-client/backend python visuaml-client/backend/examples/compiler_lab_demo.py --scenario attention --out compiler-lab-output/attention`

The ZIP is a **locally built and tested artifact**. The linked GitHub feature branch from the previous turn contains documentation only; this v2 code has not been pushed to that branch. Do not mistake the branch for a completed PR.
