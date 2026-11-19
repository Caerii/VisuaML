# v58 copy-loop recognition and trap replay

`PreparedCopyRegion` now derives a fusion plan from admitted decoded WASM.
It recognizes a bounded word-move grammar using descriptor load expressions,
consistent local roles, one forward and one backward loop, and a signed
direction test. Normalization accepts the reviewed commutations of integer
address arithmetic and comparisons, alignment hints, and `nop` padding.
The route graph supplies the actual setup, iteration, and exit step counts.

The independent WAT fixture changes the source and destination descriptor
offsets, local roles, expression order, and step schedule. It produces a
different valid WASM binary. The recognized schedule uses **7 steps** for a
nonpositive count. Positive calls use **27 + 22 × count** steps forward or
**27 + 21 × count** backward. Execution still checks page bounds and source
fuel before fusing. Traced calls and coefficient overrides use source
replay. An unrecognized memory program also uses its original execution.

## Evidence

Two deterministic runs agreed on **1,200 inputs**, including **822
fused executions**, **378 memory traps**, and **1,644 fuel-boundary
checks**. Both result and committed page bytes were checked against
Wasmtime. Successful exact source counts and fault prefixes were checked
against the packed source executor. The separate grammar sweep recognized
all **24 local assignments** with varying nop positions and replayed
**96 additional cases** through packed execution and Wasmtime.

The recognizer declined three near-misses: a forward-only copy, a reversed
direction test, and a copy that writes a word before trapping. The verifier
confirmed each near-miss's behavior through the original executor. The
forward-only program gives `[1, 1, 1, 1]` for the three-word right overlap
that produces `[1, 1, 2, 3]` in the recognized move.

## Trap replay

Traced packed memory exceptions expose `trace` and `trap_site`.
The trace is the completed instruction prefix. On an out-of-bounds trap,
the site includes the attempted opcode, PC, step and widened address.
On fuel exhaustion, it includes the next PC and the consumed step count.
The page still contains every completed write.

In the late-trap witness, the last completed store commits **109 at
byte 8**. The next memory access attempts an `i32.load` at **65,536** and
traps at **step 54, PC 60**. Replaying the completed access records from
the initial page reconstructs the committed page exactly. The verifier can
now compare access order and trap location as part of one diagnostic claim.

The backend suite finished with **1,296 passed and 20 skipped**.

## Warm latency

Both directions of overlap were timed on the independent fixture, with
page reset and full readback in each arm. The table reports the range
of per-direction medians over 21 measured calls. Preparation precedes
timing. Median recognition alone took **633.5 µs** once per
prepared program.

| Moved words | Packed source | Recognized fusion | Wasmtime Python API |
| ---: | ---: | ---: | ---: |
| 1 | 22.1–22.3 µs | 4.3–4.4 µs | 29.1–31.4 µs |
| 4 | 25.9–26.6 µs | 4.2–4.4 µs | 29.6–30.7 µs |
| 16 | 41.3 µs | 4.3 µs | 29.9–32.5 µs |
| 64 | 99.9–108.3 µs | 4.4–4.6 µs | 29.7–30.0 µs |

## Reproduce

Use the source revision in the manifest, the pinned research environment,
and a C++17 compiler.

```bash
export PYTHONPATH=visuaml-client/backend PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=$(mktemp -d)
python -m scripts.verify_copy_recognition_v58 --output "$out/evidence.json"
```

The archive contains source WAT, original WASM, the near-miss evidence,
the grammar-sweep script, and timings.

[Artifact manifest](../benchmarks/results/v58-copy-recognition-77bd340/manifest.json)
