# VisuaML Compiler Lab v3 — validated semantic admission and cost-aware placement

**Status:** experimental *compiler research slice*, opt-in. No claim of universal Python or arbitrary accelerator compilation. An optical hardware backend still requires hardware specifications, measurements, a numerical transfer model, and actual lowering/runtime interfaces.

## Concrete improvements over v2

### 1. Portable IR is now validated prior to execution

The portable manifest remains schema `visuaml.numpy-aot.v1` for compatibility, but `compiler_lab/ir_contract.py` now provides a stricter admission layer:

- **Ordered SSA-like values:** every value is introduced once; every dependency must refer to an earlier value, rejecting cycles/forward references/duplicate definitions.
- **Closed operator vocabulary:** only registered NumPy runtime operations are admissible; malformed tagged arguments fail closed.
- **Typed input contracts:** fixed nonnegative shapes, dtypes and element-count limits. Constants must be accounted for exactly.
- **Resource budgets:** input count, node count, rank, declared tensor elements, serialized archive size, constant bytes, and retained-result budget.
- **Integrity verification:** manifest and each constant are SHA-256-checked before execution; hash checks are NOT cryptographic signatures and are replaceable by someone who can edit both manifest and hashes.

**Security boundary:** This reduces accidental invalid execution and some denial-of-resource cases, but it does not sandbox Python/NumPy or make arbitrary untrusted NPZ bundles safe. Executing unknown models requires process-level isolation, quotas, trusted model loaders, and assessment of NumPy decoder behavior.

The runtime still depends only on **NumPy + Python standard library** at deployment time. Copy `numpy_runtime.py` and `ir_contract.py` together with the compiled `manifest.json` and `constants.npz`.

### 2. Exact binary min-cut placement

`compiler_lab/placement_optimizer.py` implements an exact graph cut for this specific objective:

\[
\text{cost}(z) = \sum_i c_i(z_i) + \sum_{(i,j)\in E} \mathbf{1}[z_i\ne z_j] t_{ij}
\]

where `z_i` is either **host** or **device**. The user provides abstract operator costs, backend native op/type constraints, and transfer-cost parameters. IO and constants remain on the host. The solver reduces binary placements to max-flow/min-cut. `NumpyAOTBackend` still does real execution; the optimized hybrid plan is **analytical only** until a hybrid compiler and device runtime are implemented.

The placement is globally optimal under this declared objective, not under real hardware performance. Device memory constraints, inter-operator kernel fusion, caching, repeated transfers, conversion overhead, thermal/power behavior, control flow, multi-device scheduling, and stochastic device errors remain open research.

Validation includes exhaustive brute-force comparisons for hand-constructed and twelve randomized small DAGs, plus no-device and high-transfer-cost cases.

### 3. Spatial neural computation / grouped convolution

The restricted operator set now contains **2-D convolution** for NCHW inputs and OIHW weights, including:

- strides, integer zero-padding, dilation;
- grouped and depthwise convolution;
- optional bias;
- FX modules, functional PyTorch convolution, and a restricted `torch.export` ATen `conv2d` / non-transposed `convolution` lowering;
- computational count estimates (not measured kernel time).

The independent NumPy runtime computes cross-correlations directly, and golden examples compare against PyTorch within documented numerical tolerances. Reflect padding, transposed convolution, dynamic shapes, and very large CNNs are not supported. The implementation is correctness-oriented and not competitive with optimized convolution kernels.

### 4. Interactive what-if research explorer

`examples/compiler_lab_placement_explorer.py` produces a **self-contained HTML** visualization across three workload examples (MLP, attention, grouped convolution). It permits switching among prerecorded transfer-cost scenarios and inspecting where operators were assigned, which transfers arise, and the assumed abstract cost. It is a UI research artifact, not the existing production React application.

## Validation commands

From the VisuaML repository root, after installing the project Python requirements:

```bash
PYTHONPATH=visuaml-client/backend python -m pytest -q \
  visuaml-client/backend/tests/test_compiler_spike.py \
  visuaml-client/backend/tests/test_compiler_lab.py \
  visuaml-client/backend/tests/test_compiler_lab_v3.py

PYTHONPATH=visuaml-client/backend python \
  visuaml-client/backend/examples/compiler_lab_demo.py \
  --scenario convolution --out compiler-lab-output/convolution

python visuaml-client/backend/examples/compiler_lab_placement_explorer.py \
  --analysis compiler-lab-output/convolution/analysis.json \
  --out compiler-lab-output/placement.html
```

You can supply multiple `--analysis` files to the explorer. The `convolution`, `mlp`, `attention`, and `assembly` scenarios use fixed seeded models. The `assembly` example demonstrates one selection step only and is **not** the stateful neural-assemblies reference engine.

The reported test count describes tests collected and passing in the release environment, not distinct theorem-proven laws. Local tested stack: Python 3.13, PyTorch 2.10 CPU, NumPy 2.3.5, pytest 9. Pinned project dependencies and GPU backends have not been independently qualified.

## Scientific status and next work

| Research area | Present status | Next verification gate |
|---|---|---|
| Capture / tensor tracing | FX + restricted export | Broader ATen and symbolic guards |
| Effect-free typed computation | SSA-like portable manifest + checks | Canonical independent IR with effect/alias system |
| Program transformations | Linear/attention/softmax verified on witnesses | Rule registry with preconditions, counterexamples and selective proofs |
| Numerical backends | Independent NumPy-AOT; PyTorch export | More ONNX/MLIR-compatible lowerings + gradient equivalence |
| Multi-device planning | Exact binary additive cost-cut | Measured profiling, memory-aware model, hybrid execution implementation |
| Sparse neural assemblies | One canonical winner-selection step | Exact reference engine traces of updates and stateful dynamics |
| Physical photonic runtime | NOT implemented | Device specifications, characterization and calibration |
| Visual integration | Offline interactive inspectors | Read-only API and React canvas integration with CSP and sandboxing |

The most important future engineering task is to make the **backend contract executable**: if a plan says an operation runs on accelerator X, there must be a real code generator, launch interface, fallback, and measured results bound to the graph, weights, calibration conditions, and hardware firmware version.
