# v65 Dependency-aware synthesis and fusion

The v64 replay and fresh import of its 60 roots passed unchanged. The four original v65 policy artifacts froze at 3065604 before the search holdout was installed. A full-history control was added using only the same teacher trajectories, without holdout updates.

## Exact factors for interacting lanes

Each task chooses one integer coefficient per lane. An edge probe reveals a*x + b*y for two neighboring coefficients under a chosen input pair. A single observation can constrain the pair even when both endpoint values remain individually possible. Independent products of exact unary projections overcount after the first interacting constraint in the measured scaling fixtures.

The exact factor state is an immutable sorted list of allowed pairs per edge. On trees, cavity messages count assignments on either side of each edge, multiply branch counts and sum over matching values. Queries use exact outcome counts and only admit affordable splits. These predicates and costs belong to the runtime. The model ranks eligible queries. It cannot modify the version space. Chain and branched layouts have treewidth one. Programs with wider or cyclic interaction graphs are outside this implementation. An independent incremental SMT encoding checks consistency after every observation and verifies uniqueness at the end. The small flat enumerator independently checks every exact count.

| Shape | Initial assignments | Factor containers | Flat containers |
| --- | ---: | ---: | ---: |
| 5-node chain, library 5 | 3,125 | 1,968 bytes | 276,040 bytes |
| 7-node tree, library 6 | 279,936 | 2,816 bytes | 29,186,328 bytes |
| 20-node chain, library 9 | 9^20 | 6,632 bytes | Capped |
| 32-node tree, library 12 | 12^32 | 11,528 bytes | Capped |

Measured factor-update time for the seven-node tree was 1.59 ms versus 13.81 ms for flat filtering. At 32 nodes the Python implementation recomputes full messages after each observation. Across its 32 updates it took 166 ms, versus 9.81 ms for incremental SMT satisfiability in this diagnostic. SMT checks consistency, while tree propagation also computes exact counts and marginal weights. Container sizes are for initial Python state, excluding runtime and caches.

## Recurrent controls

The learned policies train on complete ordered teacher trajectories of up to 12 steps. Full backpropagation carries state through all steps, truncation detaches it every three steps, reset clears the same full model at inference, and the full-history comparator attends over explicit earlier query/result tokens. Both full-backpropagation seeds were evaluated on the same 48 post-freeze cases. The training-time action shortlist contains 12 proposals, while deployment scores all eligible actions. This remains a distribution shift despite matching state carry.

All methods solved every held-out case. On the 16 widest shifted cases, coverage used 1,382 credits, entropy used 1,345, the two retained-state seeds used 1,366 and 1,362, and full history used 1,363. Median total searches were 45.85 ms for coverage, 48.69 ms for entropy and 57.11 ms for retained seed 651. All paired approval intervals are zero-width because the methods solved the same cases. No learned configuration improved correctness-adjusted total cost over the best symbolic policy.

## Shared-memory proof

A reference copies the original data to the scratch half-page and then computes a nearest-neighbor stencil from that copy. The fused version saves the same scratch bytes while carrying the original left value in a local. The bad version carries the just-written output. Full-page observation includes scratch and untouched bytes. Values use i32 wraparound. Both implementations reject negative or oversized record counts before touching memory. The loop count is bounded by the input half-page and the output stores are disjoint from the scratch half-page.

The independent checker validates each ordered edge contract, rechecks an SMT induction step and binds the canonical WASM lowering. The base carry is zero and the step preserves output, the scratch write, and the original value for the next iteration. The read-from-written-left variant violates that last obligation. At the archived concrete input, every audited layout diverges at byte 4. Rehashed false aliasing, stale edges, the noncommuting carry, and a wider loop bound were rejected. A changed source-fuel contract was also rejected. With fuel 45 the valid staged and fused executions differ at byte 0 before completion. The certificate covers returns and full memory at completion with sufficient fuel. It authorizes only the reviewed two-pass-to-rolling transformation. It does not serve as a general loop optimizer.

Independent import rechecked 48 composite certificates with 98 reusable edge obligations. A dependency traversal locates ten affected roots for the selected changed edge. Wasmtime and the frozen CPU tensor executor matched both certified implementations on sampled inputs and full pages with packed fallback blocked. Wasmtime and optimized native C also matched at full capacity. The frozen tensor graph is host-driven and retains its existing typed WASM interface.

## Runtime and amortization

The native comparison preinstantiates Wasmtime, runs the host C baseline with cc -O3, checks full-capacity results, and randomizes method order within each of 24 warmed batches of 64 full-capacity calls. This machine used cc, with no Clang executable available. Cranelift optimizes both WASM variants. On the first chain layout, staged WASM took 20.44 µs median, fused WASM 18.78 µs, and native C 3.34 µs. Five of six paired intervals favor fusion; one spans zero. Modeled break-even from measured search, proof publication, independent recheck, and fused deployment ranges from about 4,800 to 68,100 calls by paired median saving. The interval crossing zero leaves one estimate inconclusive. The speedup comes from deterministic dependency-aware pass fusion.

## Reproduce

From visuaml-client/voxel-replay, use the research environment with PyTorch, Z3, Wasmtime, and a C compiler:

    PYTHONPATH=.:../backend python tools/evaluate_dependency_v65.py
    PYTHONPATH=.:../backend python tools/evaluate_dependency_v65.py --import-bundle
    npm ci
    npm run dev

Open /dependency-fusion.html to select a layout, inspect a lane's dependency, compare the rejected and valid executions, and download the proof bundle. The page presents saved replay evidence. The Python replay independently performs the checks. The three bundled WASM binaries correspond to the first audited layout. Use --write to regenerate raw timings and archives.
