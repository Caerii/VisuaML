# VisuaML Compiler Lab v2 — executable research substrate

**Status:** experimental, opt-in, sampled correctness verified; **not** an arbitrary-Python compiler or a photonic hardware SDK.

This layer develops the existing VisuaML PyTorch FX visualization and open-hypergraph work into a more systematic pipeline: capture → typed graph inspection → explicit rewrite passes → numerical witness validation → capability planning → portable artifact generation → UI projection.

## Working capabilities (verified in this release)

| Subsystem | Implementation | Evidence |
|---|---|---|
| FX frontend | trace supported `nn.Module` computation, capture shapes and dtypes | `test_compiler_spike.py` and `test_compiler_lab.py` |
| torch.export frontend | import a limited ATen set (including linear/relu and selected elementwise ops), reify **fixed** input guards | `test_torch_export_frontend_normalizes_aten` |
| Neural IR | graph node identity, inputs, operator name, target, tensors, approximate FLOPs | `compiler_spike.py`, `analysis.py` |
| Static analysis | content-based graph SHA-256, FX invariants, activation liveness approximation, output/parameter bytes | `analysis.py` |
| Operator rewrites | linear decomposition/refusion, restricted SDPA decomposition, stable softmax expansion | `passes.py`, `compiler_spike.py` |
| Verification | fail-closed, multi-witness checks after **every pass**, dtype and shape validation, exact integer comparison | `verification.py` |
| Capability planning | versioned backend declarations, native/fallback/unsupported, static crossing-size estimate | `backends.py` |
| NumPy AOT | versioned JSON data manifest + `.npz` constants; executable in a **torch-free** runtime | `aot_compile.py`, `numpy_runtime.py` |
| torch.export AOT | reloadable `.pt2` program requiring a PyTorch runtime | `torch_target.py` |
| Sparse neural bridge | fixed-connectome, one-step selection with canonical lowest-index tie handling | `assembly_bridge.py` |
| VisuaML visual adapter | existing `nodes`/`edges` canvas schema, per-node operator/shape/placement metadata | `compiler_spike.to_visuaml_graph` |
| Offline interactive inspector | click operations and switch among graph transformation stages | `examples/compiler_lab_inspector.py` |

## Run from the existing VisuaML repository root

Overlay the files in the release zip onto your working copy. Install the repository's Python backend dependencies. The implementation was exercised with **Python 3.13, torch 2.10.0+cpu, NumPy 2.3.5, and pytest 9.0.2**; the repository's pinned torch 2.9.1 environment has **not yet** been validated.

```bash
PYTHONPATH=visuaml-client/backend python -m pytest -q \
  visuaml-client/backend/tests/test_compiler_spike.py \
  visuaml-client/backend/tests/test_compiler_lab.py

PYTHONPATH=visuaml-client/backend python \
  visuaml-client/backend/examples/compiler_lab_demo.py \
  --scenario attention --out compiler-lab-output/attention

PYTHONPATH=visuaml-client/backend python \
  visuaml-client/backend/examples/compiler_lab_inspector.py \
  compiler-lab-output/attention/analysis.json \
  compiler-lab-output/attention/inspector.html
```

The other demo scenarios are `mlp` and `assembly`. Open `inspector.html` in a normal browser; it does not need a development server or external CDN. PowerShell users can set `$env:PYTHONPATH='visuaml-client/backend'` before invoking Python.

## Programmatic example

```python
import torch
from torch import nn
from visuaml.compiler_spike import capture
from visuaml.compiler_lab.backends import default_registry
from visuaml.compiler_lab.verification import apply_verified_passes, verify_outputs
from visuaml.compiler_lab.numpy_runtime import run_bundle

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(4, 3)
    def forward(self, x):
        return torch.relu(self.dense(x))

x = torch.randn(5, 4)
source = capture(Net(), x)
verified = apply_verified_passes(source, [(x,), (x * 0.7,)], ["decompose_linear"])
aot = default_registry().compile(verified.captured, "numpy-aot")
aot.artifact.write("/tmp/visuaml_aot")
result = run_bundle("/tmp/visuaml_aot", x.numpy())
assert verify_outputs(source.graph_module(x), result).passed
```

## Demonstration outputs

For each scenario, the demo writes:

- `analysis.json`: source and transformed graph metadata; stage-by-stage canvas diagrams; static costs; numerical witness evidence and fingerprints.
- `numpy-aot/manifest.json` and `constants.npz`: portable, PyTorch-free model artifact, with integrity checks for the manifest/constant table and exact input shape/dtype validation.
- `torch-export/model.pt2`: separately reloadable torch.export program.
- `inspector.html`: standalone, interactive explanation of the graph transformations and checks.

### Backend semantics and scientific integrity

The NumPy runtime contains a closed operation set: matmul/linear, arithmetic, activations, shape/view/transpose/permute, softmax/reductions, layer norm/GELU, stable argsort, sparse top-k/scatter, and structural selectors. Unsupported graph targets cause a compile error; missing state, modified bundles, and wrong input specs cause runtime errors.

**Numerical witnesses are examples, not proofs.** Floating point associativity and implementation-specific rounding can change output. A successful witness is not evidence of equivalence for all shapes/inputs, and the `graph_digest` is a reproducibility fingerprint, *not* a digital signature, proof receipt, or attestation of source provenance.

The ordinary `torch.topk` path is rejected when input values tie, because its tied-index ordering is not portable. The separate `CanonicalAssemblySelection` adapter uses a stable descending argsort that prefers lower neuron IDs. That is consistent with one neural-assemblies tie policy, but **does not import or establish parity against the full neural-assemblies Brain reference engine**. In particular, it does not model the library's Hebbian updates, winner trajectories, candidate generation, refraction, readout protocol, inhibitory dynamics, or GPU hash substrate.

Cost metrics are deliberately incomplete: the FLOP counter doesn't recognize every operation, and the peak activation estimate is a static liveness approximation. Neither number is a measured latency, SRAM allocation, optical energy, or memory-bandwidth result.

No Ayo-specific backends are implemented. The old optically *inspired* noise/quantization mode remains an illustrative, nonphysical surrogate in the v1 spike; this v2 release **does not certify any optical gain, transfer function, ADC/DAC path, precision, device yield or energy savings**.

## A principled next research roadmap

1. **Canonical SSA / typed dialect**: separate executable semantics from React Flow display metadata, attach operator semantic revisions, alias/effect types, symbolic guards and resource constraints. Current IR is an intermediate step, not the final universal dialect.
2. **Reference corpus and differential testing**: corpus of models with gradients, edge cases, multiple dtypes, nested output trees, symbolic dimensions, failure categories, and randomized test generation. Golden receipts must track exact toolchain versions.
3. **General algorithm lowering**: `torch.export` operators, bounded control flow/loops, automatic decomposition recipes, symbolic tensor algebra, source-language-independent canonicalization, and lowerings for external IRs such as MLIR/StableHLO/ONNX when justified.
4. **Cost-aware partitioning**: verified hardware profile schema, transfer/nonlinear costs, quantization and stochasticity contracts, memory residency, compilation-time search with calibrated models, and measured Pareto fronts (quality, throughput, energy, dollars).
5. **Neural-assemblies true integration**: import its actual reference and hashed runtime semantics, register explicit fibers/bias/plasticity, capture finite-step schedules into an effectful IR, and compare full winner sets, weight-state transitions and recorded evidence on identical seeds.
6. **Real photonic backend**: obtain permitted Ayo device capabilities and measurements; calibrate differentiable transfer functions, optical/electronic boundaries, losses and noise; validate against lab observations before making hardware advantage claims.
7. **Native VisuaML UI integration**: add a read-only Fastify analysis endpoint and graph-stage inspector using existing collaboration stores; do not automatically execute arbitrary uploaded Python on shared infrastructure.
8. **Formal rewrite contracts**: op law registry with preconditions, automatic counterexample search, translation validation, and selective formal verification. Initial numeric witnesses are the starting line, not the finish.

The design goal is a research library whose claims, executable semantics, transformation history, experimental evidence, and physical assumptions stay tied together. That's more important than a gigantic collection of nominal backend names.
