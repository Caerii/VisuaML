# VisuaML v28 — An independent C++17 CPU execution target

*A new backend is meaningful when it independently performs the mathematics, preserves the model's state, and survives comparison with a separate reference implementation.*

## What was accomplished

Earlier VisuaML execution paths used either NumPy for operator-level mathematics or CPU TorchScript to trace a larger admissible Llama computational region. Those were valuable experiments in representation, fusion and compilation boundaries, but a TorchScript trace is still an execution path within the PyTorch ecosystem.

V28 introduces a **separate C++17 implementation** of the admitted conventional Llama forward path. It performs embedding lookup, weighted linear projections, RMSNorm, rotate-half RoPE, causal grouped-query attention, stable softmax, SwiGLU and final vocabulary logits in C++. It also returns per-layer K/V suffixes that the existing transactional resident-state runtime validates before committing.

The implementation is a **statically maintained native C++ kernel with a strict ABI for the canonical Llama semantic program**. It is *not yet a general IR-to-C++ source code generator*. This distinction matters. The compiler validates that the SemanticProgram and checkpoint agree with the single admitted mathematical contract, then invokes the independently compiled backend. General operation selection, arbitrary graph rewrites and native code generation remain future work.

Source:
- [C++17 mathematics](../visuaml-client/backend/visuaml/compiler_lab/native_cpu_v28.cpp)
- [Python native compiler, ABI and session](../visuaml-client/backend/visuaml/compiler_lab/native_cpu_v28.py)
- [Differential tests](../visuaml-client/backend/tests/test_native_cpu_v28.py)
- [Pinned trained-model probe](../visuaml-client/backend/scripts/probe_native_llama_v28.py)
- [Historical native verification workflow](https://github.com/Caerii/VisuaML/blob/f942299fc2655e4190e962bc8731ac6bbf0226e3/.github/workflows/native-cpu-v28.yml)
- [Pinned trained-model evidence](../benchmarks/pretrained_llama_native_cpu_v28.json)

## The execution boundary

~~~text
Validated Llama-style checkpoint and config
                  |
Canonical typed SemanticProgram + source hash
                  |
       Independent C++17 ABI admission
                  |
      Local trusted C++ compiler (-O2)
                  |
    Temporary native CPU shared library
                  |
    Full decoder mathematics in C++17
                  |
      Emitted K/V suffixes and logits
                  |
  Python finite/shape checks and atomic KV commit
                  |
       NumPy + Hugging Face comparisons
~~~

All weight arrays and inputs come from the existing immutable-checkpoint admission. The executable is compiled from **repository-owned, fixed C++ code**, not model-supplied source. Python checks the set of weight names, dtypes, head geometry, context bounds and token IDs before passing raw pointers across the ABI.

The native function uses no PyTorch or NumPy arithmetic kernels. It performs its own scalar matrix products and attention. A nonzero error code, nonfinite outputs, invalid input or failed model witness prevents the runtime from publishing a new KV cursor.

The C++ routine is compiled locally into a temporary process-owned shared library and checked by SHA-256. That digest provides content identity, **not publisher authentication**. The result is not a signed portable offline artifact or a hardened isolation boundary for hostile C++ callers.

## What the numerical evidence establishes

The pinned trained model remains **manojredhat/tiny-llama** at source revision:

~~~text
756e9746eb801752d69771ef3e9bc433b21d853d
~~~

The safetensors hash is:

~~~text
d910edc7031721de23a7af111e355e6c09d5f1b3a2c43c0b6b0aa6a9135ffe03
~~~

The native verification workflow downloads that revision, checks hashes, loads the trusted configuration, and compares the C++ backend against independently executed Hugging Face/PyTorch and VisuaML NumPy Semantic IR references.

The tests include full-prompt vocabulary logits, irregular chunk boundaries, per-layer key/value state, greedy generation, forbidden graph modifications and failure-atomic cache behavior. Numerical tolerances are explicit and apply only to the admitted model, shapes and tested inputs. They are **finite numerical witnesses**, not formal proof on every possible input.

Reproduction requires a trusted system C++17 compiler, the pinned Python backend environment and the local checked model:

~~~bash
cd visuaml-client
python -m pytest -q backend/tests/test_native_cpu_v28.py
python -m pytest -q backend/tests/test_native_cpu_bench_v28.py
~~~

The historical online pinned-model procedure is in [native-cpu-v28.yml](https://github.com/Caerii/VisuaML/blob/f942299fc2655e4190e962bc8731ac6bbf0226e3/.github/workflows/native-cpu-v28.yml). The compiler does not fetch native toolchains or arbitrary model code.

## How to measure the implementation

The parity-gated [benchmark harness](../visuaml-client/backend/examples/native_cpu_bench_v28.py) compares:

- The NumPy interpreter of the admitted semantic graph.
- CPU TorchScript's polymorphic whole decoder.
- The independently compiled scalar C++17 model.

Every output must match the numerical reference before any timing is reported. The harness records warm median/p95 latency, compiler construction time, source digests, model identity and limits. Compilation must not be added to a wall-clock duration that already contains it, and short CPU experiments must not be extrapolated into real production token throughput.

**The independent C++ baseline is scalar and prioritizes transparent numerical implementation over peak performance.** Faster production CPU kernels require vectorization, cache-aware GEMM, fused kernels, parallelism and carefully benchmarked reference competitors. The appearance of native C++ code does not itself imply acceleration.

## What remains

1. Generalize the compiler from one fixed Llama ABI to a source-generated native backend from the typed IR's actual operations and effects.
2. Build separately reproducible, loadable native artifacts with ABI/version compatibility and independent provenance validation.
3. Introduce efficient SIMD/BLAS or other CPU backends, with measured cache/layout optimization and physical memory profiling.
4. Add a native GPU target with independent full-model numerical and KV witnesses.
5. Expand from bounded Tiny Llama to additional real pretrained architectures and long-context workloads without weakening admission rules.
6. Calibrate modeled optical/hardware mappings against physical silicon before claiming actual photonic acceleration.

The [public research roadmap](ROADMAP.md) records these unverified gates explicitly. This work is a genuine second execution implementation, but not yet a general or production-competitive compiler.
