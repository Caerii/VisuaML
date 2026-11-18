# v58 packed scalar gate execution

`PreparedMicrostepProgram(program, use_native_loop=True, use_packed_scalar=True)`
opts into a scalar native loop that stores each binary i32 word in a
`uint32_t`. A stack or local word takes 4 bytes in place of 32 float64
lanes (256 bytes). The packed and float64 modes share the same routing,
data-effect, gain-admission, trap, fuel, and trace control. Immediate bits
are packed once at preparation. Both `run` and `run_many` use the selected
representation.

The arithmetic lowering applies only after canonical bit coefficients pass
the existing admission check. It computes the six candidate values with
modulo-2^32 arithmetic, sign-bit-biased ordering, and a zero test. Live
operand, local, microstep, ALU output, and routing gains still control
selection and state effects. Packed source masks detect overlapping
nonzero lanes before the arity check, preserving the float64 diagnostic
order for malformed effects.

## Performance

Two serial runs compared both representations in the same library.
The workloads had 4, 14, and 82 steps. Each cell summarizes five timed
samples per program, mode, and arm after warmup. Single inputs use
`run`, and larger sizes use `run_many`. Prep-inclusive timings include
one new prepared program per call. The library is already loaded.

| Inputs | Warm untraced | Warm traced | With preparation, untraced | With preparation, traced |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 6.65–7.71× | 2.50–2.72× | 0.94–1.03× | 0.81–0.98× |
| 32 | 14.53–16.88× | 2.72–2.73× | 1.91–2.02× | 1.65–1.66× |
| 1,024 | 15.39–16.27× | 1.59–1.82× | 11.46–11.82× | 1.75–1.79× |

Median warm packed latency was **5.9–6.2 µs** for one untraced input. At
1,024 untraced inputs the median program reached **178,968–294,191 inputs per second**.
Preparation dominated single-invocation time in both modes. A direct, separate
comparison on all 197 inputs measured **682.9× untraced**
and **176.2× traced** over prepared Python.

The already-instantiated Wasmtime Python binding returned one result in 19.8–20.1 µs
and 1,024 results through repeated scalar calls in 22.4–25.9 ms, using the
same workloads. The v58 calls return execution records and share one boundary
for a batch. These are API-level timings. Both exclude setup, and the Wasmtime
measurement has no trace mode.

## Verification

Both runs matched all **3,387 trace states** across **197 inputs and
35 programs**, comparing packed single and batch calls to the float64
loop, the earlier trace archive, and original WASM outputs in Wasmtime.
The coefficient replay matched all **312** outcomes, including
traps and changed traces. The fuel replay matched **1,576** outcomes
across both trace modes. Focused tests exercised all six ALU instructions,
signed i32 boundaries, random test inputs, and concurrent packed batches.

The arithmetic check establishes all-input bitvector equivalence for
add, subtract, signed less, signed greater, and zero for the admitted
coefficients. For multiply it checks all 64 possible bounded column
totals, the carry invariant, and the 528 partial-product monomials that
contribute to the low 32 bits. Each column partitions its total into the
output bit and twice the next carry. Induction therefore leaves the
weighted partial-product sum unchanged, with the final carry and high
terms vanishing modulo 2^32. The script checks this mathematical lowering.
Executable trace and result comparisons check the C++ implementation.

The backend suite finished with **1,249 passed and 20 skipped**.

## Profile

The instrumented native profile replayed 2,400 untraced batches of
1,024 inputs for `heldout_133`. The packed loop dominates the samples.
Inlining puts the packed arithmetic and state gates in the same symbol, so
the profile gives no separate ALU share. The warm traced measurements include
native state snapshots and Python trace assembly. The shared routing helper
accounted for 10.54% of sampled self time. In a separate Python profile of
197 traced single calls, trace assembly accounted for 50% of inclusive time.

## Reproduce

Use the manifest's producing source revision, pinned research
environment, and a C++17 compiler.

```bash
export PYTHONPATH=visuaml-client/backend PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=$(mktemp -d)
python -m scripts.verify_packed_alu_v58 --output "$out/proof.json"
for run in 1 2; do
  python -m scripts.benchmark_native_batch_loop_v58 --compare-packed --output "$out/runtime-$run.json"
done
```

The archive contains the summary, coefficient and fuel replay, and native
profile scripts.

[Artifact manifest](../benchmarks/results/v58-packed-scalar-2777c35/manifest.json)
