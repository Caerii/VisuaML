# v58 indexed PC and microoperation reads

The opt-in `use_indexed_pc=True` mode carries the coordinate selected by
the compact router directly into the next instruction and microoperation
read. It uses the same native zero-test and guard/path gain calculation as
compact routing. The live next coordinate comes from the admitted gate
result. This mode requires `use_compact_routing=True`.

At admission, the indexed decoder snapshots the plain float64 source
tokens and decoder matrices into immutable buffers and validates their
identity, shape and binary values. Rows use the admitted active PC
coordinate. A selected opcode reads its effect, local-mode and ALU rows
from the supplied decoder matrices. The first visit to an opcode checks
its selected one-hot effects and ALU presence. The immutable snapshot
makes that result reusable on later visits within the execution.
Malformed source tokens and decoder weights are rejected. Valid altered
decoder rows still reach the datapath. Existing dense and compact modes
remain available.

## Component measurements

Times below are medians per warm read or route. The decoder setup includes
the immutable snapshots, canonical source checks and digests.

| Program | Compact route | Indexed route | Dense microop read | Warm indexed read | Indexed decoder setup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Countdown | 8.83 μs | 4.46 μs | 38.67 μs | 1.49 μs | 420 μs |
| GCD | 8.39 μs | 5.69 μs | 39.94 μs | 1.46 μs | 575 μs |

Component benchmarks use isolated routes and rows from the two structured
fixtures. Full executor coverage uses v58's straight-line corpus.

## End-to-end measurements

Both runs timed 197 inputs with the compiled ALU and fused state active
in both arms. They include per-execution token and decoder preparation
and use a preloaded shared library. Each arm ran three timed calls per
case. Timing order alternated across cases.

| Run and mode | Compact case median | Indexed case median | Paired geometric mean | Faster cases | Worst slowdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| runtime-1, traced | 7.14 ms | 6.57 ms | 1.08× | 140/197 | 2.21× |
| runtime-1, no trace | 5.98 ms | 5.43 ms | 1.12× | 147/197 | 1.66× |
| runtime-2, traced | 7.17 ms | 6.59 ms | 1.09× | 147/197 | 1.81× |
| runtime-2, no trace | 5.94 ms | 5.32 ms | 1.11× | 143/197 | 1.82× |

The two largest observed slowdowns were rechecked with 15 alternating
paired samples. The traced case ran 1.05× faster in the recheck.
The untraced case remained about 4% slower. The archive retains the
original timings and the targeted diagnostic. Indexed execution
remains opt-in.

## Verification

Full trace hashes covering all 3,387 states, results, instruction
counts, and token and coefficient identities matched the archived
compact-routing baseline. The benchmark also checked compact and
indexed traces against v51, v56, and v57, with result parity against
the original WASM in Wasmtime. Focused tests exercised every active
row of the reviewed public corpus, malformed source rows and
coordinates, immutable snapshots, altered live decoder rows and
ablated routing gains. All ten established injections were detected.
The backend suite finished with 1,212 passed and 20 skipped.

The next setup target is to reuse the admitted snapshots across
invocations of the same program. The raw component measurements tie
the warm savings to setup cost and provide a baseline for that work.

## Reproduce

Use a C++17 compiler and the pinned research environment. The artifact
manifest records the producing commit.

```bash
export PYTHONPATH=visuaml-client/backend PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=$(mktemp -d)
python visuaml-client/backend/scripts/benchmark_binary_routing_v58.py --output "$out/micro.json"
python visuaml-client/backend/scripts/benchmark_constructed_runtime_v58.py --compiled-alu --fused-state --compact-routing --indexed-pc --profile --output "$out/run-1.json"
python visuaml-client/backend/scripts/benchmark_constructed_runtime_v58.py --compiled-alu --fused-state --compact-routing --indexed-pc --output "$out/run-2.json"
python visuaml-client/backend/scripts/benchmark_faults_v58.py --compiled-alu --fused-state --compact-routing --indexed-pc --require-all-detected --output "$out/faults.json"
```

[Artifact manifest](../benchmarks/results/v58-indexed-gates-7160363/manifest.json)
