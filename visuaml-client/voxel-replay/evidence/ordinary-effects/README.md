# Prepared ordinary effects in the frozen portfolio

This experiment starts from `3a645e7` and uses the published rule
portfolio as the primary baseline. Terrain and BFS WASM and their
active region plans are byte-for-byte unchanged. The composed-load
runner remains a separate control.

Image preparation attaches three int64 descriptors to each validated
source row: stack pops, stack pushes, and whether the instruction can
enter call-frame handling. The live graph PC fetches this data. The
same two frozen graphs run every admitted guest, so the prepared
effects are program data. Branch conditions, frame state, operands,
memory, traps, fuel and stopping are evaluated inside the graph.

- [Descriptors](descriptor.pt) replaces the ordinary opcode one-hot
  vector and its reductions with the prepared stack effects and scalar
  opcode family selection. It commits only the affected fields.
  Instruction routing still forms the original predicate, and this
  variant retains eager operand and local gathers for comparison.
- [Lazy fields](lazy.pt) also puts the left operand, right operand, and
  local reads behind real graph conditionals. Ordinary routing tests
  the live operand only for an `if` or `br_if`; other instructions
  take the decoded route. It reuses the same selected integer ALU.
  Unchanged stack and locals are carried through. A push scatters only
  its result, and a shrink clears the canonical stack tail.

The [descriptor IR](descriptor.ir) and [lazy IR](lazy.ir) show the
graph loop and the conditional gather branches. The graph has no
Python callback. Call and nested return handling still uses the
existing transition. Load and store semantics retain the existing
little-endian byte lanes, bounds and event payload, including the
checked trap event. The page stays invocation-private.

## Exact replay

Both variants matched the single-step graph at all 1,650 source
boundaries of the audited terrain tick, including ordinary-to-region
and region-to-ordinary control. Both Terrain Edit histories matched
Wasmtime's return and complete page, the archived NumPy digests and
source counts, and the portfolio's host invocation counts.
Each timed full BFS run matched Wasmtime's return and complete page
and consumed the same source steps and host calls.

The prefix witnesses check nested calls and returns, a trap inside
a helper, and fuel cuts immediately before and after a rollback
store. The [new guest](after-freeze.wat) was authored after the
two graphs were frozen at `feadbbc`. It uses the same descriptor
preparation and exercises live if/else control, stack restoration
at a branch, `local.tee`, signed multiplication and memory
effects. Both paths match the single-step graph and Wasmtime.
The composed-load seam, fuel and trap fixture also passed separately.
Packed fallback is blocked in these runs. The suite passed 93
focused Python tests and five browser replay tests.

## Paired timing and profiles

Timing order rotates on identical initial pages and inputs. The timed
interval includes graph execution and the host budget loop, with
preparation, page publication, verification and profiling outside it.
Terrain has 24 warm samples per path. BFS has five full runs per path
after an untimed warm run.

| Warm workload | Published portfolio | Descriptors | Descriptors and lazy fields |
| --- | ---: | ---: | ---: |
| Eight-agent terrain median | 114.51 ms | 73.32 ms | **58.25 ms** |
| Terrain nearest-rank p95 | 125.87 ms | 79.45 ms | **60.41 ms** |
| Full BFS median | 2.946 s | 1.803 s | **1.416 s** |

Terrain keeps 1,650 source steps, 26 host calls and 390 fusions per
tick. BFS keeps 41,591 steps, 650 calls and 8,656 fusions. Published
budget-64 coverage leaves 870 ordinary source steps in terrain
(52.7%) and 24,279 in BFS (58.4%). The separate segmented runs
attribute 860 terrain steps and 24,001 BFS steps to ordinary
execution, since using the smaller budgets changes which fusions
fit. Full ordinary-opcode counts and diagnostic per-opcode elapsed
time, including each host crossing, are archived. The most frequent
ordinary terrain instruction is store (180). In BFS it is local.get
(8,605), followed by load (2,270).

The previous segmented diagnostic estimated an ordinary-path upper
bound of roughly 79% of terrain time and 83% of BFS time. Applied to
the paired baseline medians, removing the entire ordinary portion
would represent about 90 ms and 2.45 s of headroom. This is an
illustrative ceiling from an instrumented profile. The actual
measured reductions are **49.1%** for terrain and **51.9%** for BFS.
The best variant removes 56.26 ms of terrain median, about **87%**
of the gap from this paired baseline to 50 ms. It remains 8.25 ms
above the target.

The descriptor pass took 0.123 ms and appended 4,128 bytes for
terrain. For BFS it took 0.078 ms and appended 2,040 bytes.
These are increments on top of existing validated-image and
portfolio preparation. Freezing the two artifacts took 66 and
39 ms in the separate build. The separate composed-load control
had no eligible BFS sites. Its two full runs per path measured
3.021 s versus 2.935 s for the published portfolio, suggesting
overhead with limited precision.

### Runtime operator accounting

The following is measured per sampled ordinary invocation. It
counts CPU runtime operators, including bounded tensor copies,
rather than estimating allocator calls or bytes by graph nodes.
One row fetch remains in every column.

| Operator | Terrain baseline / descriptors / lazy | BFS baseline / descriptors / lazy |
| --- | ---: | ---: |
| select | 24.42 / 16.77 / 11.75 | 24.26 / 16.81 / 11.63 |
| eq | 11.63 / 3.00 / 3.00 | 11.99 / 3.00 / 3.15 |
| where | 11.84 / 1.42 / 0.42 | 12.04 / 1.35 / 0.35 |
| gather | 3.21 / 3.21 / 1.26 | 3.13 / 3.13 / 1.11 |
| functional scatter | 0 / 0.42 / 0.42 | 0 / 0.62 / 0.62 |
| in-place scatter | 0.21 / 0.21 / 0.21 | 0.03 / 0.03 / 0.03 |
| copy_ | 4.63 / 4.05 / 2.05 | 4.33 / 3.87 / 1.95 |

Tensor scalar extraction and row-field selection remain visible.
The lazy sample still shows roughly ten scalar extraction events
and twelve selections per ordinary invocation. One scoped compiled
follow-up is a CPU graph operator for the ordinary non-memory
transition. It would read the immutable descriptor and scalar
PC/fuel/height once, dispatch the live effect, and return the
updated scalars and at most one bounded operand or local vector.
An exact empty-memory event and a sticky stop would remain part of
the contract. Retain the existing call-frame and byte-memory
transitions and compare each source prefix against the audit
graph. This scope targets the observed small-operator overhead
without changing the region rules.

## Reproduce

From `visuaml-client/voxel-replay` with the backend dependencies,
CPU PyTorch and Wasmtime:

```sh
export PYTHONPATH=.:../backend OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python tools/replay_ordinary_effects.py
python tools/replay_ordinary_effects.py --witnesses
python tools/replay_ordinary_effects.py --profile-only
python tools/replay_composed_loads.py --transfer-only
```

Use `python tools/freeze_ordinary_effects.py` to regenerate the graphs.
The [raw evidence](replay.json) includes graph and image hashes,
whole-run samples, full segmented counts, sampled operators, replay
and prefix witnesses.
