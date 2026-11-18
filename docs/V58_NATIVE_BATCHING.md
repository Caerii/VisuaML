# v58 native batching and compilation cache

The native specializer can compile once, reload a cached shared library,
and transform a contiguous NumPy `int32` vector in place. The original
scalar and trace APIs remain available.

```python
from pathlib import Path

import numpy as np
from visuaml.compiler_lab.native_specialization_v58 import NativeSpecialization

program = NativeSpecialization(
    Path("program.wasm").read_bytes(),
    cache_dir=Path("/path/to/private-cache"),
)
values = np.array([0, 1, 42], dtype=np.int32)
program.run_batch_inplace(values)
```

Give each caller a writable, contiguous, aligned vector in native byte
order. The call checks fuel per source execution, then crosses the
Python/native boundary once for the entire vector. A failed admission
check leaves the vector intact.

Use a private cache under the application's control. Cache entries are
executable code. The key binds the original source, emitted translation
unit, compiler identity, target information and flags. The file digest
detects accidental corruption. Cache trust depends on the directory's
trusted writers. Concurrent compilers can publish by atomic replacement.
They may duplicate compilation on a simultaneous miss. Reuse the loaded
object to avoid repeated deserialization in the same process.

## Verified comparisons

Two runs each used the 35 admitted straight-line programs from the
scaling corpus. The Wasmtime wrapper adds a fixed-memory batch loop and
embeds the original scalar WASM body byte-for-byte. All timed outputs
were compared with the original scalar Wasmtime module on deterministic
signed edge cases and pseudorandom inputs. Each run checked 573,440
scalar oracle results against the largest batch, plus the smaller
prefixes. Each configuration used one warmup and 15 samples with
alternating engine order.

Resident measurements exclude loading the inputs into the buffers.
Copied measurements begin with the same host array and return a
separately owned result. Native copies once and transforms its copy in
place. Wasmtime copies into linear memory and copies the result back.
These are CPU API timings, with a Python call per batch.

| Batch of 16,384 | Native | Wasmtime | Paired speedup |
| --- | ---: | ---: | ---: |
| Resident median ns/value | 0.51–0.53 | 3.30–3.31 | 6.50–6.61× |
| Copied median ns/value | 0.655–0.656 | 3.63–3.69 | 5.48–5.73× |

Native was faster for all 35 programs in each batch-size and buffer-mode
comparison. Native cold setup took a median 93–97 ms, and cache reload
took 1.18–1.38 ms. Wasmtime reload took 0.19–0.21 ms after serialization,
with the engine already created. The median estimated crossover at
16,384 copied values was 19–22 batches if both artifacts were cached and
1,782–1,812 batches from cold setup. The estimates use per-program setup
and warm latency differences.

The scalar translation validator again covered 33 result expressions
over all i32 inputs and detected its deliberately altered expression.
Two long programs retain finite replay coverage. Batch-loop correctness
is covered by comparisons against the original WASM. The measurements
exclude the structured-control programs outside the specialization
contract.

These results favor Wasmtime for one-shot work and loaded native
specialization for sustained batches on this corpus. The next candidate
should target native cache reload and cold compile latency. Any automatic
dispatch policy should use measured workload reuse and be retested on a
held-out workload mix.

## Reproduce

From the producing code revision `6226288`, use the pinned research
environment and a C++17 compiler:

```bash
export PYTHONPATH=visuaml-client/backend PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=$(mktemp -d)
python visuaml-client/backend/scripts/benchmark_native_batch_v58.py --output "$out/batch.json"
python visuaml-client/backend/scripts/verify_native_specialization_v58.py --output "$out/proofs.json"
```

The [artifact manifest](../benchmarks/results/v58-batch-6226288/manifest.json)
records source revisions, checksums, both runs and crossover estimates.
