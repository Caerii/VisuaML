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
