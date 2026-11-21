# Resident resume for a pending world observation

The frozen Rust guest and indexed observation recovery are unchanged. The runner stops immediately before LOAD at PC 38. The resident broker retains the private VM page, live registers, operand stack and call frames. A verified answer writes the existing four-byte observation mailbox and resumes at that PC. The portable reference exports and reconstructs the same state before delivering the same answer.

The timed resident success path skips the explicit checkpoint export as well as restore. The debugger already records a state digest for the just-completed instruction, which the runner uses to identify the pending boundary. Per-instruction audit hashing remains enabled for both paths. A stopped request can still be exported.

## Binding and ownership

The request binds the branch, per-branch sequence, source PC, world coordinates, tick, digest of the sealed event ledger through that tick, and pending checkpoint identity. The scheduler or checked observation cache is the trusted source of answers. A response produced by that path becomes an in-process authorization for one waiting request. Validation precedes any mailbox write. A consumed wait is removed before the guest runs, so repeating that response cannot execute another load. This design assumes one scheduler thread and an in-process resolver.

The interleaving control suspends two forks at the same PC with byte-identical VM checkpoints and different world histories. They have separate writable pages. It rejects wrong branch, tick, PC, coordinate and history, plus a fabricated response with identical fields and a duplicate response for each fork. Each rejected response preserves the suspended VM. Delivering in reverse order yields the two correct values and exactly one LOAD per fork. The missing-ledger control retains the live wait and emits a portable snapshot without supplying a value.

One resident continuation is exported and replayed from its portable envelope in a fresh process. Every source boundary and the final guest-page digest match. The portable evidence is a fixed fixture carrying its recorded answer, while the resident authorization is process-local.

## Changed histories

The distant edit still reuses all six consumer answers. Removing the remote wall changes the producer's LOAD, the resulting Rust action schedules a newly discovered gate edit, and the dependency index selects the tick-six consumer read. The first five answers remain reusable. Every producer and consumer trace and page agrees with full-world execution. Packed fallback remains blocked.

## Cost of waiting

Four alternating CPU pairs per workload, one thread. Medians are per-trial totals. Each path runs the same 6 reads per fork and compares every guest boundary with the full-world run.

| Workload and path | Restore | Mailbox delivery | Total pending | Resumed traced guest |
| --- | ---: | ---: | ---: | ---: |
| 18 forks / 108 reads, portable | 49.01 ms | 1.20 ms | 62.20 ms | 539.50 ms |
| 18 forks / 108 reads, resident | 0 | 1.15 ms | 15.41 ms | 523.05 ms |
| 36 forks / 216 reads, portable | 97.01 ms | 2.25 ms | 114.95 ms | 1,081.98 ms |
| 36 forks / 216 reads, resident | 0 | 2.19 ms | 23.75 ms | 1,065.71 ms |

In the 108-read case, portable restore spent 24.17 ms on canonical serialization and integrity validation, 10.19 ms decoding the page, 6.43 ms creating the VM and copying the page, and 7.54 ms restoring and validating register and frame tensors. The measurements compare pending time through load delivery. The suffix timer includes per-instruction audit hashing and excludes the independent Wasmtime comparison.

The 18-fork paths both have 84 hits, read 11,130 source bytes, and recompute 1,553 cell-ticks. The 36-fork paths both have 192 hits, the same source bytes, and the same recomputation count. The larger fixture repeats the same overlapping dependency workload across more forks. Traced guest execution remains much larger than the pending interval.

Each waiting VM retains 67,544 bytes of logical tensors, including its 64 KiB page. This excludes Python and allocator overhead, shared graph weights and trace storage. The portable VM envelope takes 88,366 bytes. Two explicit exports took 0.283 and 0.330 ms; dropping the waiting-state reference after export took 3.6 and 3.7 µs. Eviction measures releasing the broker reference and makes no claim about allocator or OS reclamation. The per-read redundant portable exports are measured separately from pending time.

## Reproduce

From `visuaml-client/voxel-replay` in the existing Python environment:

```sh
export PYTHONPATH=.:../backend
python tools/replay_resident_resume.py
python tools/replay_resident_resume.py --resume evidence/pocket-resident-resume/pending-portable.json
```

The verifier runs one pair for each workload. Use `--write` for four alternating timing pairs. [Exact replay](replay.json) · [Portable pending read](pending-portable.json) · [Raw phase samples](timings.json)
