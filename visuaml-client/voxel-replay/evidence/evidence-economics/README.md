# When is another check worth its cost?

A frozen learned policy chooses whether to stop, run a cheap negative screen, or request stronger evidence. The same policy runs in two domains. It sees the estimated costs, expected gross payoff, remaining budget, public mixture priors, and whether the cheap screen has already run. It does not see the result of either check before requesting it. The shared exact checker from Checked Decisions still governs adoption. A positive screen leaves the candidate unresolved. The runtime checks affordability using the actual charge and records any rejected request before it touches evidence.

The model was trained by imitation of a two-stage expected-value policy on synthetic cost tuples and frozen at 307154f before the evaluation scenarios were built. The deterministic policy compares the expected net value of proof-first, screen-then-proof, and keeping the incumbent without spending. The fixed baselines ask unconditionally in their declared order until the evidence settles the choice or the next charge exceeds the cap.

## Evidence from two domains

- In the compiler, a targeted replay can refute a proposal when the first source boundary has a different value. A matching sample cannot authorize the proposal. The strong check verifies the universal selector equality for the reviewed ordered local-read template. The experiment binds that check to actual Passage and Terrain sites and the previous SMT and Wasmtime archive. It prices an evidence requirement under the reviewed template rather than rerunning a general SMT solver for each accounting scenario.
- In Pocket, a complete wall cut between the only signal source and the target rules out arrival. A hole in the wall leaves the screen inconclusive, even when it is too far away to use before the deadline. A full forecast in the unchanged simulator settles arrival by the deadline. An independent full-world shortest-path oracle checks each result. The reference option is prequalified by the task contract. The candidate has the modeled payoff only if it meets the stricter arrival requirement.

## Results

There are 80 base scenarios per domain across four regimes. All are paired across policies and across accurate, underpriced, and overpriced quotes. The compiler trial uses two actual sites under repeated economic schedules. The world trial varies the geometry and deadline. Mixture priors are supplied per regime and reveal no individual outcome. The scenario weights describe this experiment's workload mix.

| Accurate quotes | Compiler net | Pocket net | Combined verification cost |
| --- | ---: | ---: | ---: |
| Learned | 646 | 664 | 893 |
| Expected value | 646 | 664 | 893 |
| Prove first | 208 | 229 | 1,873 |
| Cheap first | 299 | 317 | 1,694 |

With accurate quotes the learned policy screened first in 20, proved first in 20, and skipped verification in 40 scenarios per domain. Its 44 adoptions all paid back their verification charges. The fixed policies each adopted 60 plans, with 16 adoptions failing to pay back. Underpriced proof quotes led the learned policy to request ten proofs the hard cap refused. Overpriced proof and pessimistic payoff quotes caused it to miss seven individually profitable, affordable plans. There were zero unsupported learned adoption requests. A forced adoption control was rejected before spending or executing either check.

All values use **modeled cost units**. Gross payoff is per-use savings times workload size. Net payoff subtracts incurred verification charges and gives the prequalified reference a zero incremental payoff. Verification charges and per-use savings are controlled inputs. These figures measure policy behavior under a cost model. They do not estimate wall-clock proof or simulation performance. The learned policy matches the deterministic expected-value policy on accurate quotes, consistent with its training objective. The distorted-quote results report sensitivity to estimates, with no claim of a general advantage over deterministic allocation.

## Reproduce

From visuaml-client/voxel-replay in the research Python environment:

    PYTHONPATH=.:../backend python tools/evaluate_evidence_economics.py

The evaluator reproduces all rows and checks exact evidence and budget invariants against the archive. Each journal row records the raw proposal, remaining budget, charged cost, evidence result, certified outcome, and budget-denial flag. The archive includes actual and quoted costs, payoff inputs, graph and reference hashes, and the independent proof archive identity.
