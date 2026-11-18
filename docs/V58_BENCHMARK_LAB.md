# VisuaML v58 — the Neural Machine Benchmark Laboratory

**Research implementation, not yet a measured result.** GitHub Actions
remain disabled. The v57 base is an unverified draft and this v58 work is
stacked on it. No new numerical claim should be marked verified until
an actual clean local run succeeds and its source-bound output is reviewed.

This system intentionally tests *correctness*, *programmability*,
*performance*, *resource scaling*, and *falsifiability* as separate axes.
A weighted overall score would obscure critical failures, so it is not
provided.

## A reusable experimental matrix

| Suite | Question | Workload | Outcome |
| --- | --- | --- | --- |
| S0: smoke | Can the offline harness run, finish and record honest measurements? | Two known programs; four inputs | Source/result parity and correctly formatted timing |
| S1: semantic | Does v58 execute its admitted microsteps correctly? | 16 compositional straight-line programs × six i32 edge inputs | Every PC, next PC, local, stack and halt state identical to v51 and v56; original Wasmtime agrees |
| S2: generality | Are weights reusable for previously unseen programs? | 19 public + eight generated held-out sources | 27 distinct source binaries; 194 finite input witnesses; report full source identities |
| S3: control/stack scaling | What grows with instruction count, operand depth and nesting? | 26 scaling programs, 44 input witnesses | ALU length 2–40, operand depth 2–24, frame nesting 1–12, branch unwind 1–12 |
| S4: performance | Which engine is faster or slower, and on which program families? | Matched inputs across Wasmtime, v51, v56, v57 and v58 where supported | Paired latency p50/p95, CPU time, engine coverage; no unmeasured speedup |
| S5: mechanism intervention | Can we falsify claims of gate authority? | Ten deliberately corrupted gates/program tokens | Activated effect; detected errors/changed traces; retain undetected controls |
| S6: setup/representation | How expensive is tokenizing a source and preparing its machine data? | Same source suite | Tokenization latency and exact source hashes, reported separate from warm execution |
| S7: memory | Does state representation incur disproportionate allocation costs? | Opt-in same-program Python trace allocation | Python-traced peak allocations, not claimed as total RSS |
| S8: reproducibility | Can we compare two commits without silently changing the task? | Two immutable benchmark JSON files | Exact source/input/semantics check; environment-policy warning or rejection |

The optional 26-program scaling suite brings the planned total to
**53 distinct source programs and 238 input witnesses**. These counts are
theoretical workload inventory, *not* measured passing executions.

The actual v58 microstep machine admits **straight-line data instructions
only**. The structured-control cases remain in the v51/v56/v57 comparisons,
not silently attributed to v58. Missing engines are reported, never
imputed as zero timings or successful runs.

## Benchmark rules that prevent attractive nonsense

1. **Correctness first.** Run original canonical WASM in independent
   Wasmtime, compare v51/v56/v57 and, where admitted, v58 state transitions.
   A mismatch aborts before timing.
2. **Correctness during timing.** Every timed invocation also checks its
   result, preventing a fast but broken path from winning the benchmark.
3. **Source identity.** Hash the original canonical WASM, exact typed program,
   microoperation token table and source Git tree. Refuse dirty worktrees
   and refuse to overwrite prior evidence.
4. **No hidden warmup.** Specify warmup/repeat counts. Separate source
   compilation and tokenization from warm execution.
5. **Same-source comparisons.** Compare matched inputs, instructions and
   outputs. Report environment and thread policy. Strict mode rejects
   incompatible measurement conditions.
6. **No performance alchemy.** Wasmtime's Python binding overhead,
   GC, CPU frequency and NumPy implementation choices affect the
   measured latency. This is a CPU software prototype, not native neural
   hardware or proof of asymptotic improvement.
7. **Hold out programs.** Seeds 128–135 are disjoint from public generated
   seeds 0–15. They are independent source compositions, not just new
   inputs to a familiar program.
8. **Preserve regressions.** Report worst slowdown, fraction of paired
   witnesses slower, incomplete engine coverage and every undetected fault.

## Offline execution — no GitHub Actions

From a clean checkout of the v58 research branch, with Python 3.12 and
the backend dependencies installed, use a fresh **outside-the-repo**
output folder. The examples below are Unix shell commands; on Windows,
replace the path with an unused path under `$env:TEMP`.

```bash
python -m pip install -r visuaml-client/backend/requirements.txt pytest wasmtime ruff
python visuaml-client/backend/scripts/verify_offline_v58.py \
  --output-dir /tmp/visuaml-v58-smoke --repeats 2
```

This first runs focused tests, then records four source/input pairs with
real timing, and produces scorecard JSON plus CSV. The small sample is a
**harness preflight**, not a performance benchmark.

Run full backend regression and benchmark evidence only after smoke
passes:

```bash
python visuaml-client/backend/scripts/verify_offline_v58.py \
  --output-dir /tmp/visuaml-v58-full --full --benchmark \
  --scaling --faults --memory --warmups 2 --repeats 10
```

Output files are intentionally non-overwriting. The evidence includes
source hashes, environment, software versions, timing distributions,
instruction counts, per-family measurements and fixed-weight identities.

One can compare the same engine across two actual records:

```bash
python visuaml-client/backend/scripts/compare_neural_benchmarks_v58.py \
  --baseline /tmp/visuaml-v58-baseline/neural-v58-benchmark.json \
  --candidate /tmp/visuaml-v58-candidate/neural-v58-benchmark.json \
  --engine microop_v57 --require-compatible \
  --output /tmp/visuaml-v58-comparison.json
```

The comparison report gives paired geometric-mean speedup, p10/p90,
worst slowdown, and the fraction of slower cases by workload family.
The scorecard compares engines **within the same measured run**. A
reported ratio greater than 1 means the candidate is faster in the
specified paired latency; it does not imply hardware acceleration.

## Microstep architectural claim

`neural_microstep_v58.py` combines a v57 source-bound typed microop,
v56 fixed six-candidate ALU, v55 bit-sliced operand-state array, and
v57 local-register bank into a **single microstep function**.

The admitted data instruction classes are `nop`, `i32.const`,
`drop`, `local.get/set/tee`, `i32.eqz`, and the full-width
`i32.add/sub/mul/lt_s/gt_s` operations. For each instruction the
microstep computes candidate operand pops, values, local updates, and
result insertion; fixed one-hot masks determine the output state.

The separate v58 executor uses the v52 PC circuit to choose the next
source instruction and applies these microstep outputs to the next
state. It uses a host-validated final `end` instruction. It does not
pretend to neuralize dynamic frame control and memory simultaneously:
v53/v54 and the older restricted interpreter remain separate
evidence-backed components. Future fusion requires a strict whole-state
transition contract before an all-control-plus-datapath claim.

**Performance warning:** evaluating candidate circuits and validating
each tensor may make v58 slower than v56 or Wasmtime. That is not a
failure of the benchmark; it is precisely why the suite is necessary.


## S9 — Blinded neural-to-symbolic-to-neural round trips

This experiment directly connects the formerly separate v50 and v58
research results, rather than treating construction and recovery as
unrelated achievements.

1. Construct a bounded typed WASM source program and execute it using
   v49's **fixed, analytically constructed 32-bit neural bit gates**.
2. Give v50's synthesizer only a callable integer-input/output oracle.
   Its closure cannot inspect source instructions, neural weights, program
   token matrices, construction metadata or execution traces.
3. Pass each proposed symbolic candidate to an **independent Z3 source
   verifier**, which checks inequivalence over the full typed i32 domain
   and returns a counterexample on `sat`, or equivalence on `unsat`.
4. Re-encode the accepted symbolic candidate as canonical original WASM.
   Execute the original and recovered binaries in independent Wasmtime
   and the recovered program in the v58 fixed-weight microstep executor.
5. Check final outputs plus **every recovered v58 instruction-state
   transition** against the independent v51 reference. Record exact
   source and candidate byte strings, SHA-256 hashes, solver formula
   digest, candidate digest, fixed v49 weight hash, query counts, CEGIS
   refinements and finite trace digests.

The current benchmark declares 13 distinct source programs: the five
historical v50 examples and eight distinct held-out source variants.
Each has 12 adversarial finite execution replay inputs. Therefore
**156 source/input replays are planned**, not yet reported as executed.

Two proof scopes must never be conflated:

- `Z3 unsat` establishes equivalence of **typed original WASM and
  accepted candidate semantics** over all signed i32 inputs.
- The constructed neural v49 interpreter and v58 microstep execution
  are checked with **finite execution witnesses only**. This does not
  prove all-input equivalence of a neural circuit or recover the unique
  historical source of an arbitrary learned network.

The benchmark fails rather than silently accepting a source/candidate
SMT mismatch or any incorrect finite neural execution.

### Run the round-trip benchmark locally

After installing the repository's pinned Python dependencies, plus
`pytest`, `ruff`, `wasmtime` and `z3-solver`, run a new output
directory outside your Git worktree:

```bash
python -m pip install -r visuaml-client/backend/requirements.txt pytest ruff wasmtime z3-solver
python visuaml-client/backend/scripts/verify_offline_v58.py \
  --output-dir /tmp/visuaml-v58-roundtrip \
  --roundtrip --roundtrip-holdout
```

The workflow executes focused tests, smoke benchmarking, the blinded
CEGIS+SMT round trip and **an independent replay of the resulting proof
archive**. It then emits a separate measured-only recovery scorecard.
It uses no GitHub Actions or remote API.

### S10 — Proof-gated representation profitability

Only after the round-trip proof archive independently replays can we
compare the *original* and *recovered* WASM binary representations.

A dedicated benchmark compares original/recovered binaries on identical
inputs within **the same Wasmtime runtime**, then within **the same
unchanged v58 neural microstep interpreter**, alternating AB/BA invocation
order. It measures source-byte size, warm median time, differences and
the recovery-plus-proof cost required to amortize an improvement. It
reports regressions and does not infer a speedup if none was measured.

This is **symbolic representation change**, *not* partial evaluation
of the neural interpreter into specialized weights. The lower-bound
amortization estimate also excludes compilation and deployment overhead
and must not be interpreted as a realized deployment payback.

```bash
python visuaml-client/backend/scripts/verify_offline_v58.py \
  --output-dir /tmp/visuaml-v58-representation \
  --roundtrip --roundtrip-holdout --representation-cost \
  --warmups 2 --repeats 7
```

The independent receipt verifier is also callable directly:

```bash
PYTHONPATH=visuaml-client/backend python \
  visuaml-client/backend/scripts/verify_neural_roundtrip_evidence_v58.py \
  --input /tmp/visuaml-v58-roundtrip/neural-v58-roundtrips.json
```

**No numerical outcome from S9 or S10 should be displayed as verified
until these local runs finish and their source-bound artifacts are
reviewed.**

## Next compounding research priorities

**v58.1:** finish offline smoke, full regression and source-bound
benchmark evidence; address all real failures and publish the
unmodified transcripts and their Git source identities.

**v58.2:** combine v58 data microsteps with v54 control microsteps into
one canonical whole-state update rather than two handwritten dispatch
paths; differential-check nested loops and GCD.

**v59:** add bounded WASM linear memory with real address arithmetic,
unaligned loads/stores and out-of-bounds traps, then extend the same
benchmark axes to memory capacity and locality.

**v60:** add direct calls and call frames before tables/indirect calls.
Only assert actual tested official WASM conformance families with pinned
spec versions.

For the scientific interpretation, see
[V58 implications and benchmark thesis](V58_IMPLICATIONS_AND_BENCHMARK_THESIS.md).


## S11 — Failure-complete recovery and audit

A strict `all cases must pass` proof receipt is appropriate for accepting
a transformation, but insufficient to measure *synthesis reliability*:
one unsuccessful recovery can abort before any inventory appears. v58
therefore includes a separate **failure-complete recovery campaign**.

- Five original v50 source behaviors, eight held-out source graphs when
  enabled, and a cubic negative control outside the admitted quadratic
  grammar are each recorded as **attempted** cases.
- A valid grammar rejection or exhausted synthesis/SMT budget is recorded
  as `rejected`, not omitted. Candidate shape, source WASM bytes, immutable
  source hash, elapsed time and partial neural-oracle/SMT query counts are
  preserved even when no candidate is accepted.
- An incorrect execution trace or invalid verifier protocol is marked
  `invalid`, not mistakenly scored as a legitimate failure to recover a
  symbolic algorithm. The reporting schema distinguishes *all attempts*
  from *scientifically valid attempts*.
- Valid recovery yields `verified` only when the strict v50→v58 bridge
  returns a typed all-i32 source/candidate proof and independent finite
  original-WASM/neural-microstep replay.
- An independent archive auditor reconstructs each original source program,
  checks every source digest and inventory entry, recomputes the
  success/rejected/invalid denominator, requires that the cubic negative
  control reject, and reruns Z3/Wasmtime/v58 evidence for each accepted
  candidate. It will reject tampered or partial reports.

The historical-plus-heldout campaign defines **14 planned attempts**:
5 historical, 8 held-out, and 1 cubic negative. These are benchmark
definitions, *not* 14 observed outcomes. The denominator is fixed before
the CEGIS experiment begins.

Run independently from the repository root with dependencies installed:

```bash
python visuaml-client/backend/scripts/verify_offline_v58.py \
  --output-dir /tmp/visuaml-v58-recovery \
  --recovery-campaign --recovery-campaign-holdout
```

The campaign JSON cannot be overwritten and must live outside the clean
Git checkout. Its independent replay uses the same existing source-bound
proof mechanism without promoting a finite neural execution witness
into a universal proof of the constructed weights.

This is scientifically essential: **recovery rate, proof soundness,
mechanistic fidelity and execution performance are independent metrics**.
One successful CEGIS experiment proves none of the other three.
