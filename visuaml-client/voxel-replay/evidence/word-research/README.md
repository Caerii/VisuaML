# Composition, learning, and context

This benchmark follows the word decoder at `764def4`. It uses that frozen decoder for exact execution and the existing frozen synthesized tool executor for learned routing. The saved evidence includes the evaluated programs and states, matched control conditions, checkpoints, and raw timings.

## Composition of exact instructions

Forty generated programs rearrange LOAD, ADD, STORE, and optional forward branches. They match the scalar oracle at all **246** source boundaries, including after overwrites and branches that skip work. These programs were generated after freezing the decoder. The generator seed and per-case tokens are archived.

The relevant compositional property is a refinement invariant. At every committed boundary, the generated token history represents the same PC, register, memory, and halt state as the scalar program. Initialization and CODE encoding establish the relation. At EXEC, code/cache selection and the bit circuit compute the successor. Subsequent tokens retrieve that successor from the EXEC position. COMMIT publishes it and advances the invariant. At a partial cut, no address key was published yet. Induction extends the relation to any admitted finite sequence under the context and attention recovery assumptions of the word decoder. The sweep tests that contract; the numerical guarantee remains conditional on recovery margins. [Word decoder proof scope](../word-decoder)

A checkpoint control stops the countdown after the fourth completed instruction, rebuilds the input prefix from its PC, register and four memory words, and emits the exact remaining nine-instruction suffix. It drops 48 historical tokens. Under the same invariant, an instruction-boundary checkpoint can omit older writes when no future load can observe them. This fixture tests one snapshot and one continuation, with the same code and a fresh cache.

## Distance and distractors

The sweep keeps one three-instruction LOAD/branch/HALT program and its output fixed. It moves the authoritative write from **49 to 397 tokens** before the first EXEC. Each distance has a padding and a distractor variant. The latter inserts same-address uncommitted payloads and commits to unrelated addresses. All cases also contain a stale committed value for the target address. The attention score margin is 192 in every case because the nearest competing score is the write 12 positions earlier. All eight cached and full-prefix streams match the reference. The longest case reaches 504 tokens. It uses 2,156,864 logical KV bytes, with five-sample warm prefill medians of 38.87–40.75 ms and decode medians of 20.77–21.44 ms. This measures near-limit behavior within the existing 512-token envelope.

## Learned selection and interventions

Six structured routers use widths 8, 16, and 32 and two seeds per width. They see 12,410 training rows with four entire thresholds withheld. The data supervises first and second tool choices separately. Checkpoints at steps 0, 50, 200, and 600 are downloadable. The 32 paired evaluation contexts vary registry order, polarity, choice order, and held-out thresholds. A separate pass uses held-out description phrases.

By step 200, familiar paired accuracy ranges from 0 to 4 out of 32 across the six models. At step 600 every model selects both sides correctly in all 32 familiar pairs. Held-out description accuracy ranges from 20 to 32. Both width-32 seeds score 32, with two seeds per width the result is descriptive.

In a matched familiar pair, the patch replaces only the condition MLP's hidden activation from the below-boundary run with the corresponding at-boundary donor. Registry and descriptions stay fixed. All **192/192 eligible decisions switch** as predicted. A norm-matched random offset switches 10/192, and a same-side donor switches 0/192. The first choice is unchanged as a structural control because it has no path through the patched MLP. These results establish mediation by this explicit learned hidden bank on the tested pairs. They leave narrower learned subspaces and unrestricted language routing open.

The host also chains the trained selector through a third call with live results from the frozen exact executor. All six models choose the full expected call sequence and return the expected answer on all 52 three-call cases. The host fixes the length and the structure of this test. The model chooses the handle at each stage; this tests reuse of a learned decision rule on longer tool chains.

## Reproduce

From `visuaml-client/voxel-replay` with the existing research dependencies:

```sh
PYTHONPATH=.:../backend python tools/audit_word_research.py
PYTHONPATH=.:../backend python tools/study_router_learning.py
```

The first command recomputes its cases and verifies the saved traces. Runtime timings are archived separately from equality assertions. The second loads archived checkpoints and regenerates the learning and intervention evidence. [Execution bundle](execution.json) · [Learning bundle](learning.json)
