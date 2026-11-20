# v58 prepared frame inputs

Fused copy at 0d8da4f already binds opcode coefficients by source PC
and skips NumPy on retain. The lexical else bit still needed lookup at
dispatch. Checked state changes built a FrameTokens object, four
optional or height arrays, and admitted the resulting arrays again.

- **Prepared** binds the lexical else bit to each read-only source row.
  C++ loads that bit and calls the existing event and retain graph.
  The live inputs are sequential selection, active-frame status, and
  branch outcome. The invocation reuses read-only kind and height
  rows, and the already admitted branch depth from indexed micro-ops.
  State changes still construct a token and take the checked v53 entry.
- **Compact** sends the same fields to the same v53 contraction without
  constructing a token object or re-admitting the static rows. The C++
  event circuit admits the one-hot action. Live stack heights are checked
  before indexing, live frame state is admitted at entry, and the
  contraction still checks frame resources, branch targets, output
  state and restored height. Python retains source dispatch, traps,
  lexical metadata, guest memory and fuel.

Both variants used fused copy. They stay opt-in. Detailed tracing and
custom or unsupported profiles use the checked reference.

## Cost and coverage

On four sparse ticks, frame token encoding was 7.3 ms and token and
weight admission 27.3 ms, out of 2.09 s in a line-instrumented profile.
On four crowded ticks those phases cost 6.4 and 22.5 ms out of 1.84 s.
Removing only those phases suggested a ceiling of about 1.02× in the
probe. Probe times include instrumentation. The diagnostic BFS line
profile is a 4,000-step prefix. The paired BFS runs complete all
41,591 steps.

The prior action profile for the same logs shows branches are the
majority of surviving checked work, with 388 of 672 sparse transitions,
308 of 528 crowded, and 2,113 of 2,559 in full BFS. An opcode-attributed repeat found 196 conditional
and 192 unconditional branch events in four sparse ticks, 164 and
144 on crowded terrain, and 2 and 189 in the BFS prefix. The
[full fused-copy profile](../fused-frame-retain/copy-profile/bfs.json)
provides the corresponding full-invocation breakdown. The compact path
preserves those computations. In its four-tick crowded profile,
routing takes 315 ms, operand transitions 295 ms, and retained frame
equations 76 ms. Routing and operands have more remaining coverage.

| Replay counter sample | Native gates, all arms | Token objects, copy → prepared → compact | Checked entries, copy → prepared → compact | Frame contractions, all arms |
| --- | ---: | ---: | ---: | ---: |
| Sparse terrain, four ticks | 5,888 | 672 → 672 → 0 | 672 → 672 → 0 | 672 |
| Crowded terrain, four ticks | 4,656 | 528 → 528 → 0 | 528 → 528 → 0 | 528 |
| Full BFS | 35,196 | 2,559 → 2,559 → 0 | 2,559 → 2,559 → 0 | 2,559 |
| One-blank Sudoku | 728 | 40 → 40 → 0 | 40 → 40 → 0 | 40 |

Prepared avoids three optional token constructors and a fresh
operand-height array per changing event. Both prepared variants
removed 7,677 optional constructor calls in full BFS. Compact
additionally avoids checked token and weight admission and separately
rechecks live state. These counters exclude implicit NumPy temporaries.

On a validated 4,096-instruction module, the row payload grew from
262,144 to 294,912 bytes. Lookup arrays add 5,296 bytes, for an extra
38,064 bytes of numeric payload. The pointer table and Python integer
sizes were unchanged. Setup medians through the first route were
36.06 and 36.74 ms over eight alternating samples per path. The
separate cold compiler and library load took 1.07 s.

## Paired warmed execution

| Workload | Samples per arm | Fused copy → prepared → compact median |
| --- | ---: | ---: |
| Eight sparse agents | 32 ticks | 266 → 267 → 260 ms |
| Eight crowded agents | 32 ticks | 225 → 225 → 219 ms |
| Full BFS, 41,591 steps | 2 runs | 8.884 → 8.975 → 8.867 s |
| One-blank Sudoku | 8 runs | 175 → 173 → 174 ms |

Each set alternates order after warmup, using the same original and
mirrored logs. Terrain median setup through first route increased
about 0.07 ms with compact inputs. Sparse throughput was 28.77 →
29.13 → 30.46 agent-ticks/s, and crowded 35.80 → 35.36 → 36.19.
Observed p95 was 321 → 321 → 275 ms sparse and 260 → 270 → 245 ms
crowded. The compact tails remain 5.5× and 4.9× the 50 ms budget.
The sparse tail result should not be generalized beyond these
32-tick samples. Two BFS runs establish exact replay and provide
too little data for a meaningful latency or tail claim.

## Exactness and reproduction

The event graph matches all 144 canonical opcode and guard combinations.
The tests alternate live guards at the same prepared rows and check
disjoint invocation buffers. A source WASM fixture takes both if/else
arms and revisits one branch on a loop backedge. Both saved terrain
histories match Wasmtime pages and source steps. Traces, fuel cutoffs,
bypass, custom configurations, page-straddling traps, and injected
first-divergence checks preserve the reference behavior. Packed
execution is blocked during differential verification.

From the voxel-replay directory, with the research Python environment:

    export PYTHONPATH=.:../backend PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python tools/prepared_frame_input_pairs.py sparse --pairs 8
    python tools/prepared_frame_input_pairs.py crowded --pairs 8
    python tools/prepared_frame_input_pairs.py bfs --pairs 2
    python tools/prepared_frame_input_pairs.py sudoku --pairs 8
    python tools/prepared_frame_footprint.py
    python tools/profile_frame_materialization.py bfs --retain copy
    python tools/profile_compiled_region.py crowded --compiled --retain copy --compact
    python -m pytest -q tools/test_prepared_frame_inputs.py tools/test_terrain_verify.py
    python tools/check_escargot.py --prepared-frame-inputs

Run the last command separately from timing.

[Warm pairs and counters](pairs) · [Profiles](profile) · [Scaled footprint](footprint.json)
