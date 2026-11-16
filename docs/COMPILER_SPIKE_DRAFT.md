# Compiler Lab status: v4 integration

The earlier compiler-spike proposal has now been expanded into the opt-in v4 implementation on this branch.

Source:
- `visuaml-client/backend/visuaml/compiler_spike.py`
- `visuaml-client/backend/visuaml/compiler_lab/`
- `visuaml-client/backend/scripts/compiler_lab_service.py`
- `visuaml-client/server/compilerLab.ts`
- `visuaml-client/src/ui/CompilerLab/`

The feature flag `VISUAML_COMPILER_LAB_ENABLED=1` enables fixed-scenario analysis. The `/compiler` React route can be opened even without the backend, but analysis requires the flag, Python, and PyTorch.

Validation on the extracted v4 overlay: 74 Python tests passed and one optional cross-repository selection-parity test skipped. The full Node dependency-resolved build and actual stateful neural-assemblies parity remain unverified.

The code contains no Ayo hardware backend. All placement costs are hypothetical analytical units; PyTorch/NumPy output comparisons are sampled numerical witnesses, not proofs.

See `docs/COMPILER_LAB_V4.md` and `visuaml-client/COMPILER_LAB_INTEGRATION.md`.
