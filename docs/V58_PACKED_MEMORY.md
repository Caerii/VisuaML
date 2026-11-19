# v58 packed linear memory and prefix-sum recovery

The prepared packed structured executor supports `i32.load` and
`i32.store` against one fixed 64 KiB WASM page. The source uses real
WASM memory opcodes and a memory export alongside `run(i32) -> i32`.
The typed memory instruction carries a canonical alignment exponent
and an unsigned 32-bit offset. The packed executor widens the effective
address before checking all four bytes, then applies little-endian reads
or writes. Unaligned access uses the same byte semantics.

```python
program = decode_control(original_wasm)
prepared = PreparedMicrostepProgram(
    program, use_native_loop=True, use_packed_scalar=True, use_structured_control=True,
)
page = bytearray(65536)
# Initialize the page and pass the program's i32 argument.
result = prepared.run_memory(argument, page)
```

The call borrows a caller-owned page and executes on a private snapshot.
Completed stores are written back even when a later instruction traps.
Each failing out-of-bounds store leaves its target bytes intact. A
trace records each successful access with its address and the bytes
before and after. Fixed load and store gains are snapshotted and
digest-bound to the execution result. Stack depth, routing and control
reuse the packed structured loop. The memory API operates on one page
per call.

## Recovery and replay

The original WASM visits each word in order, accumulates its prefix sum,
writes it in place, and reads back the write for the next iteration.
For **[3, -1, 4]** it writes **[3, 2, 6]** and returns **6**. A trace replay
checks the prior bytes at each of nine memory accesses and reconstructs
the complete final page.

The grammar searches three fold operations. An overwrite hypothesis
fits the initial singleton evidence. The input `[-2, -2]` refutes it
because the second output word is `-4` in the original and `-2` in the
hypothesis. The next candidate is a modulo-2^32 running sum. Two
deterministic runs verify this candidate across **781 inputs**, using
every length from zero to four and every element from -2 through 2.
They compare the return value and all 65,536 bytes against original
WASM in Wasmtime and the packed executor. The evidence establishes
equivalence on this declared finite domain for this grammar and fixture.

With the live store gain set to zero, the first divergent access occurs
at step 41, byte address 4. The original writes `02 00 00 00`, and the
ablated run retains `ff ff ff ff`. The archive carries the original
WASM, counterexample, rendered recovered algorithm, baseline trace,
and first-divergence record.

## Verification and latency

The focused tests cover unaligned access, the last legal four-byte
access, out-of-bounds loads and stores, unsigned effective addresses
that exceed 32 bits, i32 overflow, and load and store interventions.
The full backend suite finished with **1,283 passed and 20 skipped**.

Warm untraced measurements below include restoring the initial page
and returning a full-page readback in both arms. Program preparation,
module instantiation, and shared-library compilation precede the
timing. These are API-level latencies on the archived host.

| Input words | Packed v58 | Wasmtime Python API |
| ---: | ---: | ---: |
| 0 | 20.5 µs | 30.0 µs |
| 1 | 22.8 µs | 30.0 µs |
| 4 | 27.4 µs | 31.4 µs |
| 16 | 45.3 µs | 31.3 µs |
| 64 | 155.8 µs | 31.4 µs |

## Reproduce

Use the producing source revision, pinned research environment, and
a C++17 compiler.

```bash
export PYTHONPATH=visuaml-client/backend PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=$(mktemp -d)
python -m scripts.verify_memory_recovery_v58 --output "$out/recovery.json"
```

[Artifact manifest](../benchmarks/results/v58-packed-memory-3835e7f/manifest.json)
