# v58 native specialization

The constructed v58 executor remains the reference for neural gate
experiments. This backend specializes admitted straight-line source WASM
into a direct CPU function. It resolves the operand stack and locals into
SSA value references, then generates C++17 with exact unsigned 32-bit
arithmetic. Signed comparison flips the sign bit before comparing. The
warm function contains no runtime WASM dispatch or trace materialization.
A separate instantiation of the same generated function materializes
every original instruction's operand stack and locals for diagnosis.

The backend admits only canonical `run(i32) -> i32` source through the
existing v51 decoder. It checks v58's stack cap and final result shape
before compiling and checks fuel and argument admission at each call.
The supported source consists of straight-line i32 operations in one bounded
function. Compiler and dynamic library errors propagate. The output is a
process-local shared library. Compilation is part of the cold cost.

Callers can compile once and reuse the warm entry:

```python
from pathlib import Path

from visuaml.compiler_lab.native_specialization_v58 import NativeSpecialization

program = NativeSpecialization(Path("program.wasm").read_bytes())
result = program.run(17)
states = program.trace(17)
```


## Replay

Use the pinned research Python environment and a C++17 compiler. From
the repository root on the producing code revision, run:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=visuaml-client/backend
out=$(mktemp -d)
python visuaml-client/backend/scripts/benchmark_native_specialization_v58.py \
  --scaling --repeats 5 --output "$out/benchmark.json"
python visuaml-client/backend/scripts/verify_native_specialization_v58.py \
  --output "$out/proofs.json"
```

The benchmark compares output with original Wasmtime, v51, and neural v58,
and compares every recorded stack, local, and source-control field with
v51 and neural v58. It measures compilation and module setup separately,
then records wall and process CPU time for both warm calls and the native
trace entry. The neural reference also materializes its route diagnostics.
The native diagnostic trace covers semantic state and does not reproduce
neural gate diagnostics.

The validator lifts the emitted warm result expression into bitvectors
and checks against the independent v43 typed stack encoding. It rejects
unrecognized emitted constructs and verifies a deliberately incorrect
specialization is detected. The proof concerns the emitted arithmetic
expression and typed source result across all i32 inputs for the recorded
programs. Finite traces exercise the actual compiled machine code.

## Research grounding

- MLIR's progressive lowering suggests retaining a typed transition
  representation for verification while allowing target-specific
  specialization. The v58 compiler uses a small SSA lowering for this
  bounded subset. [MLIR rationale](https://mlir.llvm.org/docs/Rationale/Rationale/)
- Cranelift and Wasmtime precompilation provide a strong comparison for
  the next iteration. Compare cached or precompiled artifacts and
  batched calls before replacing the simple C++ backend.
  [Cranelift](https://github.com/bytecodealliance/wasmtime/tree/main/cranelift)
  · [Wasmtime precompilation](https://docs.wasmtime.dev/examples-pre-compiling-wasm.html)
- Alive2 is a model for validating each generated program independently. The current Z3 check validates the
  emitted expressions. Validating optimized LLVM IR is a distinct
  follow-up. [Alive2](https://github.com/AliveToolkit/alive2)

## Measured results

Two runs of the same scaling corpus each checked 35 programs, 197 inputs
and 3,387 instruction states. Forty-one structured-control inputs were
outside this path's admission scope. Each measurement used one warmup and
five samples. The benchmark code revision was `1d42366`. The translation
validator at `fba35d7` proved the emitted result expression for 33
programs. Two long arithmetic chains exceeded the independent encoder's
64-operation limit and have finite checks from both runs. The deliberately
incorrect specialization produced a Z3 counterexample. The full backend
suite finished with 1,184 passed and 20 skipped. The original v58
fault campaign detected all ten injected faults.

| Median of per-case warm medians | First run | Second run |
| --- | ---: | ---: |
| Neural v58 with trace | 44.60 ms | 41.38 ms |
| Native with semantic state trace | 17.56 μs | 17.81 μs |
| Native result only | 1.63 μs | 1.65 μs |
| Wasmtime Python call | 20.25 μs | 20.23 μs |

The paired geometric-mean speedup for warm native result-only execution
was 27,151–30,592× over traced Python v58 and 12.67–12.77× over warm
Wasmtime at the Python call boundary. With native semantic tracing, the
paired improvement over traced v58 was 2,347–2,643×. These measurements
include different Python bindings and apply to these CPU workloads.

Median native setup took 77.7–100.6 ms per program. Wasmtime setup took
1.77–2.09 ms with the engine already created. Using each program's measured
setup and median warm timings, the estimated median crossover against
Wasmtime was 4,039–5,260 calls. The exact crossover will depend on workload,
compiler cache, and runtime reuse. The stronger next comparison should
measure a cached native artifact and Wasmtime with calls batched inside
both runtimes.

For one illustrative 34-instruction source, the generated x86-64 warm
entry folded the chain into `lea eax, 136[rdi]` followed by `ret`.
The two benchmark records, proof receipts, reference fault results and
example entry are in the [artifact manifest](../benchmarks/results/v58-native-1d42366/manifest.json).
