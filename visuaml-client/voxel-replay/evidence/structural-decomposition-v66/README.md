# v66 Structural decomposition and interruption scopes

The v64 and v65 deterministic suites replayed without modification, including fresh imports of 60 and 48 roots. This experiment keeps those artifacts intact. The two structural rankers were frozen at 02cdce0 before the v66 fixture generator was added.

## Incremental exact inference

Each cached directed message depends on its edge factor and on all incoming messages except the reverse direction. Editing an edge invalidates its two orientations and their cached nonbacktracking descendants. Lazy recomputation rebuilds only the arrows needed for the requested marginals. No-op edits leave the cache intact.

The benchmark performs 240 edits and reversals over chains, balanced trees, stars and random trees. Every complete marginal vector and count agrees with the full v65 implementation and an independent oracle. Six-node cases enumerate assignments directly. Larger cases use one incremental SMT context with toggled factor assumptions, and enumerate satisfying models to get exact counts and per-node marginal weights after each edit. The larger diagnostic deliberately starts with informative edge-sum observations, so these oracle counts stay small enough to enumerate. It does not time SMT enumeration on an unconstrained 72-node space.

| 72-node topology | Cached local median | Fresh local median | Cached all marginal median | Full v65 rebuild median |
| --- | ---: | ---: | ---: | ---: |
| Chain | 43 µs | 869 µs | 882 µs | 2,968 µs |
| Balanced | 42 µs | 783 µs | 919 µs | 3,605 µs |
| Star | 89 µs | 921 µs | 2,880 µs | 42,895 µs |
| Random | 48 µs | 765 µs | 919 µs | 4,048 µs |

A local query rebuilt one directed message at the median in each 72-node topology. Reading all marginals rebuilt 72 messages, out of 142 directed messages. The two outputs and their baselines are reported separately. Times include invalidation and lookup. Cache initialization and independent model enumeration are outside the warm query timings.

## Cyclic structure and learned proposals

An elimination permutation yields explicit bags and parent separators. The runtime checks the separator relation and admits tables only up to induced width three. Exact counting multiplies and sums finite factors, and can bind a variable to recover its marginal. Each count agrees with flat enumeration or the SMT oracle. Min-fill marginals are also checked against the oracle on each admitted fixture.

There are 75 cases, with 15 each for plain cycles, ladders, triangulated chains, chorded cycles and a guard family containing a five-clique. All policies return width two on 45 cases and width three on 15. All refuse width four on the 15 clique guards. Optimal residual DP was checked on all 30 cases of up to eight nodes. The learned rankers use the same structural inputs as the heuristics and train on small random trees with extra edges. The two learned seeds achieve the same admitted widths as min-fill and min-degree. Median planning time across all sizes was 1,035 and 807 µs for the learned rankers, 219 µs for min-fill, and 95 µs for min-degree. The held-out families show no structural or economic advantage for learned ordering. General large-treewidth inference remains outside the supported exact path.

## Transformation and checkpoint scopes

The v65 completion certificate still applies to its staged and fused WASM. At fuel 45 they expose different pages, as reproduced in the original and fresh native fixtures.

The new contract uses a **logical host record boundary**. Record schedules either fuse the writes, split a record into snapshot and output phases, or reorder output stores after capturing the record. The proof checks i32 output, scratch writes and the next original carry by a symbolic induction step. The bound and distinct half-pages establish the untouched-memory frame. It relies on the reviewed host schedule binding. Arbitrary retired instructions and trap states are outside this boundary contract.

The saved artifact includes the full page, cursor and original carry after record six. A fresh process imports the evidence nodes, reconstructs the expected prefix from independently supplied initial data, then resumes with the reordered schedule. All 14 page hashes and carries match. A corrupted carry, an aliasing assumption, and an attempt to promote a completion root to record-boundary authority are rejected. The proof binds the layout and the checkpoint binds the exact initial page. It is not a checkpoint migration certificate for the continuous WASM binaries.

## Measured deployment costs

Two v65 stencil layouts were rerun using v66 incremental entropy search. The economics include search, proof/lowering and publication, a newly opened on-disk proof check, and fused Wasmtime deployment. Native measurements use warmed, preinstantiated Wasmtime/Cranelift and optimized host C, with randomized order over 24 paired batches of 64 calls. Clang was unavailable, so the native host baseline used cc -O3.

| Full-capacity layout | Staged Wasmtime | Fused Wasmtime | Native C | Observed break-even |
| --- | ---: | ---: | ---: | ---: |
| Chain | 20.51 µs | 18.27 µs | 3.36 µs | 9,429 calls |
| Wide shifted | 19.36 µs | 18.23 µs | 11.47 µs | 48,137 calls |

The paired mean savings intervals were 1.85–2.65 µs and 0.55–1.86 µs. These are two layouts and a one-page workload. Complete costs were about 21.2 and 56.4 ms, with break-even estimated from paired medians. The separate logical checkpoint measurements used 24 repeats, about 0.30 ms median export and 0.51–0.55 ms median restore for a roughly 131 KB portable hex encoding. Those charges apply when requesting a logical checkpoint; they are excluded from uninterrupted WASM calls. The existing exact CPU tensor executor agrees with the recorded WASM replay with packed fallback blocked. The host schedules run outside that graph.

## Local replay

From visuaml-client/voxel-replay in the research Python environment:

    PYTHONPATH=.:../backend python tools/evaluate_structural_v66.py
    PYTHONPATH=.:../backend python tools/checkpoint_dependence_v66.py
    PYTHONPATH=.:../backend python tools/bench_structural_v66.py
    npm run dev

Open /dependency-fusion.html for the saved fuel counterexample, factor edit and invalidated arrows, induced bags and width refusals, and logical record checkpoints. The viewer displays archived evidence. The Python commands independently reproduce it. Use --write to regenerate raw timing files and deterministic archives.
