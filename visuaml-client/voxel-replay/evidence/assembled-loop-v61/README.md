# Assembled recurrent controller

This experiment adds a typed execution loop to the v60 evidence system. The controller receives the next recipe row, a module registry, the remaining call budget, and a small summary of the last exact result. It proposes a module and two register indices. The host executes this typed request against symbolic i32 registers and returns an exact result. The shared-weight attention controller also receives its previous hidden state. The same weights are used on every step.

The recipe is structured input and prescribes which calculation to perform. The learned policy maps that input into executable module requests. The benchmark tests composition and length transfer. It does not test when to invent an algorithm, adaptively spend extra calls, or lower the full loop into a causal decoder. Python drives execution and verification, and the frozen TorchScript artifact owns the learned proposal.

## Loop semantics

Each proposal costs one unit, including halt. With no remaining fuel, the loop marks the state exhausted before asking for another proposal. The state contains the recipe cursor, register expressions, recurrent vector, remaining budget, status, and journal. A checkpoint binds the task, model artifact, and semantics version. Import recomputes the prefix before accepting it. A fresh subprocess restored a checkpoint after two calls and matched the entire uninterrupted continuation. Modified fuel, task input and model identity were rejected. An exhausted state also round-tripped.

## Equal-budget comparison

The three learned models were frozen at fa5fa2f before evaluation. Training recipes have one to four arithmetic operations. The 100 held-out recipes have five to seven. The deterministic baseline maps the recipe directly into requests. The external-boundary control uses the stateless attention policy and serializes the request state as JSON between operations in the same process. That timing excludes network and server latency.

| Controller | Exact results | Calls | Median loop time |
| --- | ---: | ---: | ---: |
| Recurrent attention | 100/100 | 694 | 1.41 ms |
| Stateless attention | 100/100 | 694 | 1.39 ms |
| MLP | 100/100 | 694 | 1.01 ms |
| Deterministic | 100/100 | 694 | 0.023 ms |
| JSON boundary | 100/100 | 694 | 1.97 ms |

The timing loop includes proposal and typed assembly. Final independent compilation and proof are outside the timing loop. The recurrent model showed no accuracy or call efficiency advantage on this workload. Its frozen artifact is 32,797 bytes and the demonstrated serialized checkpoint is approximately 1.1 KB. Peak working memory and energy were not measured.

## Verified specialization witness

For 12 held-out recipes, the learned requests assembled a symbolic program. A deterministic compiler then emitted an affine normal form, which can differ from the original module composition. An independent SMT check against the reference recipe and the v60 algebraic verifier approved every compiled artifact. The proofs cover return and unchanged memory at successful completion for all i32 inputs. Source-step counts can change. The exact tensor executor and Wasmtime matched completion of both binaries for each recorded input, with packed fallback blocked.

Five of 12 specializations retired fewer source instructions. The largest reduction was 12 to 2. The supplied binary pair is that case. The recorded pooled medians are 1.24 to 1.08 ms for the frozen replay API and 46.5 to 45.7 µs for preinstantiated Wasmtime. There are five retained repeats per binary and 12 pairs. These timings refer to the execution API and exclude training or proof time. They do not establish a native speedup. The persistent bundle contains the 12 proof roots and the dependencies needed to revalidate them. A modified candidate was rejected. At input zero the exact VM returned 5 for the reference and 6 for that candidate.

The specialized program is emitted by a deterministic compiler from the learned assembly. The measurement establishes a bounded assembly-and-verification loop alongside a specialization witness. It leaves learned discovery and budget-sensitive query strategy for a task where the controller has real choices about the algorithm.

## Local replay

From visuaml-client/voxel-replay in the research Python environment:

    PYTHONPATH=.:../backend python tools/evaluate_assembled_loop_v61.py
    PYTHONPATH=.:../backend python tools/persistent_evidence_v60.py --bundle evidence/assembled-loop-v61/verified-assemblies.json

The first command reproduces the stored outcomes. The second independently imports each proof root. Timings are archived separately from the deterministic results.
