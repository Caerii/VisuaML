# v58 in-process LLVM specialization

The optional LLVM backend emits a direct scalar entry and an in-place
batch loop from the same bounded SSA lowering as the C++ specializer.
It compiles in process through llvmlite and keeps the generated code
alive with the loaded object. The C++ diagnostic trace API, persistent
C++ artifact cache, and neural reference remain available.

```python
from pathlib import Path

import numpy as np
from visuaml.compiler_lab.llvm_specialization_v58 import LlvmSpecialization

program = LlvmSpecialization(Path("program.wasm").read_bytes())
values = np.array([0, 1, 42], dtype=np.int32)
program.run_batch_inplace(values)
```


## Measured tradeoffs

Two runs covered the 35 admitted straight-line programs, using the same
resident and copied-buffer comparison against Wasmtime and C++. Each
run checked 573,440 scalar results from the original Wasmtime modules
against the largest batches, plus the smaller prefixes. Each batch
configuration used 15 timed samples and one warmup with alternating
engine order.

| Per-program setup | Range of run medians |
| --- | ---: |
| C++ cold compile and load | 87–95 ms |
| In-process LLVM compile | 13–15 ms |
| Cached C++ reload | 1.21–1.32 ms |
| Wasmtime wrapper, compile and instantiate, sum of stage medians | 1.86–2.01 ms |
| Wasmtime serialized-artifact reload | 0.19–0.20 ms |

The first LLVM compile in each fresh process included importing and
initializing LLVM. It took 34–73 ms. The table reports per-program
medians after the same process handled the corpus, with the first setup
also retained in the raw records. LLVM artifacts are process-local.

At a batch size of 16,384, LLVM measured **0.46–0.47 ns/value** with
resident buffers and **0.59–0.60 ns/value** with copies. Paired
geometric-mean speedups over Wasmtime were **7.72–8.07×** resident and
**6.71–6.86×** with copies. LLVM and C++ had similar warm throughput.
The estimated median LLVM crossover against cold Wasmtime moved to
**169–211 copied batches** of 16,384 values. That estimate uses each
program's setup and warm latency difference. LLVM targets programs that
will be repeatedly evaluated without an existing C++ cache entry.
Wasmtime had the lowest recorded setup cost, and cached C++ was the
fastest way tested to reload a native specialization.

## Verification

The LLVM scalar verifier performed 33 full-i32 comparisons with the
independent typed-source Z3 encoder and produced a counterexample for a
deliberately corrupted result. Two longer source programs have finite
machine-code coverage from both timed runs. The proof concerns the
emitted scalar IR before LLVM optimization. Actual scalar and batch
machine code were checked against original Wasmtime execution. The
existing C++ scalar expressions were byte-identical to their previous
receipts, and the neural reference retained all ten fault detections.

## Reproduce

Use the research environment with the optional package
`llvmlite==0.46.0`. The benchmark needs a C++17 compiler for the
comparison. Run these commands from the relevant clean source commits.
The manifest identifies the producing revision of each record.

```bash
export PYTHONPATH=visuaml-client/backend PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=$(mktemp -d)
python visuaml-client/backend/scripts/benchmark_native_batch_v58.py --llvm --output "$out/batches.json"
python visuaml-client/backend/scripts/verify_llvm_specialization_v58.py --output "$out/proofs.json"
```

The [artifact manifest](../benchmarks/results/v58-llvm-b6cb20c/manifest.json)
records source revisions, checksums, batch measurements and crossover
estimates. The next useful comparison is a lower-optimization LLVM tier
and held-out workload mixes before choosing an automatic dispatch policy.
