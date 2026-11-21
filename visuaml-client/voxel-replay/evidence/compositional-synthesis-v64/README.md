# v64 Compositional verified lane synthesis

The v63 trajectories and independent import of 114 certificates replayed unchanged. The v62 recovery remains a protocol result. The v64 controllers were frozen at 7f8e497 before the scaling holdout was written.

## Factorized version space

A program selects a transform for each independent memory lane. A transform is the typed triple (a,b,g), which maps a signed i32 value x to a*x + b + g*[x<0] with WebAssembly wraparound. The reference presents behavior through exact requested examples. The query policy sees the ambiguity, response partitions and costs. It does not see the opaque component assignment. A probe filters exactly one immutable leaf. Other lanes keep their existing domain identities. Canonical bucket memoization shares repeated partitions. The Cartesian product cardinality is computed without materializing programs. Dispatch admits only affordable queries that can eliminate hypotheses.

Small cases compare every update against flat enumeration. All sizes use an independent finite constraint encoding, and the SMT solver checks that excluding the factorized solution leaves no candidate. At six lanes and eight components, the flat state held 262,144 candidates in about 25.4 MB of Python containers. The factorized top-level tuple and unique leaf tuple occupied 192 bytes. The total measured filter time for that one probe sequence was 86.4 ms flat and 0.027 ms factorized. These numbers exclude the Python interpreter, caches, shared integers and allocator footprint. At 32 lanes and 32 components, the product represents 2^160 candidates with a 592-byte initial container representation. The independent SMT encoding found one solution after 32 observations, with exact uniqueness checked. Flat enumeration was capped at 300,000 candidates. The scale cases use the easiest measured probe pricing, and each scale point is one diagnostic run.

This grammar uses independent lanes and shared local component definitions. There are no cross-lane dependencies or unrestricted memory pointers. A finite-domain SMT encoding is the constraint comparator; equality saturation was outside this product grammar experiment.

## Query policy comparison

The rankers trained on two to five lanes and at most 18 library entries. Both recurrent seeds use one-step supervision with zero initial memory during training; carrying history across queries is a held-out deployment test. The controls include MLP, resetting the same recurrent weights, entropy per credit, coverage per credit, and exact worst-case local query planning. Each family has 20 tasks. The wide split has 24 lanes, 32 candidates per lane, and shifted query costs.

| Policy | Deep identified / 20 | Wide identified / 20 | Wide credits spent |
| --- | ---: | ---: | ---: |
| Coverage | 20 | 20 | 2,597 |
| Per-lane minimax | 20 | 20 | 2,781 |
| Entropy | 20 | 18 | 2,848 |
| MLP | 20 | 18 | 2,858 |
| Recurrent 641 | 18 | 0 | 3,287 |
| Recurrent 642 | 20 | 0 | 3,287 |
| Reset 641 | 20 | 18 | 2,835 |

Coverage required 8.41 ms median on wide tasks, compared with 15.39 ms for the MLP and 23.45 ms for recurrent seed 641. These totals include symbolic query construction and filtering. The recurrent policies have no advantage here. Their failed trajectories, resource charges and paired approval intervals remain in the archive. The result does not distinguish recurrence as an architecture from mismatch between one-step training and carried-state deployment.

## Independent proof reuse

Reusable leaf contracts establish the same i32 result for all x under staged and fused forms. At the composition boundary the verifier checks lane order, exact module bytes, page limits and component identities. For every accepted n, cell addresses 4*(lane + depth*row) are aligned, bounded and distinct. Applying both passes to each cell therefore yields the fused result regardless of the order between independent cells. The original sign determines the conditional term. Local temporaries preserve that original x. Within the declared ABI, the loop is bounded and writes only the selected prefix. Invalid signed lengths return −1 before any writes. The certificate covers completion with sufficient fuel and identical full memory. Intermediate writes and source-level fuel may differ. The verifier regenerates the reviewed lowering; the claim is scoped to that lowering and does not authorize arbitrary wasm binaries.

A fresh process independently imported 60 composite certificates with 32 unique leaf proofs. Rechecking each composite with a fresh cache ran 611 leaf checks. A verifier-owned cache reduced that to 32. Median complete composite validation, including regeneration of both binaries, changed from 0.401 ms to 0.244 ms. A changed component invalidated 29 nodes in the selected dependency closure. A rehashed forged leaf, a changed lane assignment, and a shorter layout were rejected. Verifier-owned memoization never accepts evidence supplied as a bare cache entry.

## Memory-pass performance

Six representative layouts ran on Wasmtime and the frozen CPU tensor VM with packed fallback blocked. The CPU replay checked invalid bounds, zero, one and three records, full pages and exact returns. Wasmtime and the native C reference additionally checked the full-capacity input. This is one-page, bounded looping with local variables and in-place loads/stores. It does not imply a transformer-native interpreter.

The native benchmark uses preinstantiated modules and warmed, alternating measurement order, with 20 batches of 64 calls and two footprint sizes. Cranelift optimizes both the staged and fused wasm. The cc -O3 loop is an independently generated host-native reference rather than a wasm deployment equivalent. In the first full-page case, staged wasm took 25.62 µs median, fused wasm took 20.66 µs, and optimized C took 3.01 µs. Three of six full-page paired mean intervals favor fusion; three span zero. Observed median-based amortization estimates range from 300 to 13,731 calls and include entropy search, compilation, proof recheck and fused deployment. Positive point estimates with intervals spanning zero do not establish a reliable break-even. Gains are attributable to the deterministic pass fusion and verified dependency structure, rather than the learned ranker.

## Local replay

From visuaml-client/voxel-replay in the research Python environment with PyTorch, Wasmtime, Z3 and a C compiler:

    PYTHONPATH=.:../backend python tools/evaluate_compositional_v64.py
    PYTHONPATH=.:../backend python tools/evaluate_compositional_v64.py --import-bundle

The first command checks complete deterministic outcomes. The second independently imports the published DAG. The staged and fused binaries are the first full-page pair. Use --write to capture fresh raw benchmark timings.
