# v58 memory-region contract for recovery and copy fusion

The exercise starts with original WASM that moves `count` four-byte words
inside one fixed 64 KiB page. The source and destination byte addresses
are little-endian i32 values stored at bytes 0 and 4. The exported
function returns the count. It selects its traversal direction before
making any writes and uses the existing structured control and memory
instructions.

## Claims and assumptions

- The terminal comparator checks return or trap category and every byte
  of the resulting page, including writes that committed before a trap.
- The optional source-step claim compares exact fuel consumption on a
  successful run. The access-order claim compares the recorded address,
  operation, and before/after bytes in sequence. It requires a trace
  from each arm.
- The lowering is bound to the exact WASM fixture. It fuses only
  untraced invocations with default coefficients, in-bounds regions, and
  enough fuel for the source steps. The bulk operation snapshots the
  source range before writing the destination. Other invocations use
  the packed source executor, including traced runs, interventions,
  fuel exhaustion, and partial memory traps. Callers own the page
  exclusively during a call.
- The SMT interval lemmas cover unread-byte preservation and address
  wrapping under the guard. Eight SMT array obligations also prove
  final-byte equality for one to four words in the guarded model with
  arbitrary initial bytes. Executable equivalence is checked over the
  declared replay domain. General source-region
  discovery and pre-trap access-history replay remain open.

```python
program = decode_control(original_wasm)
prepared = PreparedCopyRegion(program)
result = prepared.run_memory(count, page, capture_trace=False)
```

## Shared counterexamples

A single disjoint example admits both traversal directions. The same
ordered domain rejects the forward recovery proposal and the
forward-specialized lowering on the same descriptor-overlap case.
The readable aligned case starts with **[1, 2, 3, 4]** and moves three
words one position right. Original WASM gives **[1, 1, 2, 3]**. The
forward proposals give **[1, 1, 1, 1]**.
A second counterexample rejects always copying backward. The recovered
rule chooses direction from the signed address comparison in the source.

A disjoint case also demonstrates why access claims matter. Both
directions return the same value and page. The access-order check
locates the first different load after the two descriptor reads.
In the late-trap case, the first word **109** remains committed at
byte 8 when the next read goes out of bounds. An eager bounds check
would have left the old byte there. Disabling the live store gain
forces source replay and produces the first different store at **step
36, byte address 20**.

## Verification

Two deterministic runs agree across **1,200 inputs**, with **822
fused executions**, **378 memory traps**, and **1,644 exact-fuel
boundary comparisons**. The domain includes overlapping and unaligned
ranges, descriptor aliasing, page-edge addresses, and two initial
word patterns. Original Wasmtime and packed execution match on results,
trap category, and complete pages. The same terminal verifier accepts
the recovered candidate and guarded lowering. Successful diagnostic
replay checked **822 access sequences** against the recovered
candidate and independently replayed their effects to the final page.

The backend suite finished with **1,287 passed and 20 skipped**.

## Warm timing

Both directions of overlap were timed with full-page reset and readback.
Compilation and preparation precede timing. Per-row ranges reflect the two
directions. The fused region uses the public Python API and a bulk byte
operation on the caller's page. The comparison is specific to this
source-bound optimization.

| Moved words | Packed source | Guarded fusion | Wasmtime Python API |
| ---: | ---: | ---: | ---: |
| 1 | 23.3–23.5 µs | 4.2 µs | 28.5–30.1 µs |
| 4 | 26.1–26.7 µs | 4.1 µs | 29.1–30.7 µs |
| 16 | 39.5–42.4 µs | 4.1 µs | 29.4–29.9 µs |
| 64 | 101.1–109.5 µs | 4.2 µs | 29.7–30.8 µs |

## Reproduce

Use the pinned research environment at the producing revision in the
manifest.

```bash
export PYTHONPATH=visuaml-client/backend PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=$(mktemp -d)
python -m scripts.verify_memory_region_v58 --output "$out/evidence.json"
```

The archive contains original WASM, the ordered contract and
counterexamples, replay scripts, proof records, and latency samples.

[Artifact manifest](../benchmarks/results/v58-memory-region-36bb96e/manifest.json)
