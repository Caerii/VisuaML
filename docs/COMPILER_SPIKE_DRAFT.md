# Compiler IR / backend spike (draft)

This branch records an opt-in experimental direction for VisuaML: trace model computations, preserve operator/tensor metadata, decompose supported operations, plan execution by backend capabilities, and compare outputs against PyTorch.

A locally tested prototype patch was prepared alongside this branch. It currently lives in the downloadable `VisuaML_Compiler_Spike_v1.zip` artifact attached to the corresponding ChatGPT conversation; the code files were **not yet committed** to this branch. Do not merge this branch as a completed implementation.

## Prototype contents

- `visuaml-client/backend/visuaml/compiler_spike.py` — typed FX graph analysis, linear decomposition/refusion, PyTorch/NumPy interpreter, quantization/noise *illustrative* optical surrogate, pluggable backend registration and capability planning, canvas-schema projection.
- `visuaml-client/backend/tests/test_compiler_spike.py` — 10 local tests passed with torch 2.10 CPU and pytest 9.
- `visuaml-client/backend/examples/compiler_spike_demo.py` — runnable comparison demo.
- `docs/COMPILER_SPIKE.md` — setup and limitations.

To apply the prototype, unzip its files into the repository root. Do not conflate the optical surrogate with Ayo hardware: it does not model a real device.

Next: publish the code to this branch, connect a read-only analysis endpoint, compare with the neural-assemblies reference engine, then add physically calibrated backends and larger-model import support.
