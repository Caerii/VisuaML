# v58 blinded counterexample stress experiment

Seven composed threshold programs used a complementary signed
comparison, a local store and read, and an equality-to-zero gate.
Every program returned one on the seven initial neural oracle queries.
The blinded synthesizer therefore proposed constant one each time.
The typed-source Z3 verifier refuted all seven proposals. Querying
the constructed v49 executor at the solver's first witness produced
zero in every case.

All six in-grammar programs recovered after two to five
counterexample-guided refinements. Each accepted symbolic candidate
passed typed-source full-i32 verification. The existing round-trip
verifier checked 72 finite inputs across the six accepted programs,
including original and recovered Wasmtime and constructed execution.
The threshold-24 control was outside the configured hypothesis grammar.
It ended with no surviving candidate, and the archive retains both
solver receipts and all observations for that rejection.

Two independent invocations of the experiment produced identical
records after excluding recovery timing. The evidence concerns
deliberately ambiguous compositions in the existing threshold
grammar, with a negative control at its boundary. Proof scope is
typed source versus the accepted symbolic candidate. Neural execution
has finite replay coverage.

From the producing source in the manifest and the pinned research
environment, run the offline probe.

```bash
export PYTHONPATH=visuaml-client/backend PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=$(mktemp -d)
python visuaml-client/backend/scripts/probe_cegis_counterexamples_v58.py --output "$out/recovery.json"
python visuaml-client/backend/scripts/probe_cegis_counterexamples_v58.py --output "$out/replay.json"
```

[Raw observations, SMT receipts and checksums](../benchmarks/results/v58-cegis-stress-db0fa96/manifest.json)
