# v59 Cost-aware metareasoning and reusable evidence

A frozen learned policy chooses among seven proposals: stop, screen one of three candidates, or request stronger evidence for one of three candidates. The network sees costs, potential returns, priors, and the current evidence status. The scheduler checks the remaining budget and dispatches the exact evidence producer. Only a positive strong check makes a candidate eligible. A negative screen can rule it out. The final selector compares the accrued certified payoffs against the incumbent. The proposal network cannot set evidence status or adopt a plan. This runs with the existing exact execution references and a Python scheduler.

The network was frozen at 64767a2 before the evaluation was written. It learned from 9,000 synthetic decision states and has no world layouts, WASM, or test traces in training. The published checked-decision, query-policy, and single-candidate economics suites were replayed first.

## Decision model

A candidate has integer screen cost c, strong-check cost d, per-use saving s, uses N, validity probability p, and detectable failure probability r. A positive screen only rules out that detectable failure. The model updates validity to p/(1−r) after an inconclusive screen. This calculation assumes the declared evidence model and independent candidate outcomes. The exact validity gate does not depend on these probabilities.

Let G(E) be the maximum gross payoff N·s among certified candidates, or zero for the incumbent. With evidence state E and budget B, the modeled terminal value is G(E). For each affordable query, compare its expected continuation value minus its charge with that terminal value. The exact ex-ante dynamic program evaluates this recurrence over at most six calls. The learned policy imitates the maximizing proposal from synthetic states. The greedy baseline values one proof or one screen-then-proof pair against the incumbent. The other baselines use proof-first, replay-first, and fixed candidate order. A separate hindsight oracle sees the outcomes and finds the best affordable direct proof or stops.

The evaluation contains 256 three-candidate scenarios, 64 per domain and split. The shifted split expands cost, horizon of use, and budget ranges, and reduces actual validity frequency below the public priors. The world cases use the frozen cardinal forecast and independent shortest-path expectations. The compiler cases use actual Passage and Terrain sites from the reviewed ordered local-read template. The negative screen compares saved values from a targeted source-boundary replay, including matched-value controls. The strong check reruns the template's SMT obligations. This scope covers two compiler sites under multiple economic schedules.

The raw calibration contains 24 measurements per query and domain. Evidence charges are integer tariffs derived from the observed median ratio, with controlled scenario multipliers. Benefits and caps are modeled. Strong compiler timings include constructing and solving the SMT obligations. Cheap compiler timings measure reading and comparing the already captured preview operands. The timing archive also separates proposal time, evidence calls, repeated evidence requests, and complete decisions. These timings do not enforce a wall-clock deadline.

## Results

The exact ex-ante baseline used every possible first proposal on this suite, including stop. The table reports realized net modeled value, summed over 64 paired tasks per row. The hindsight oracle has access to hidden outcomes.

| Domain and split | Learned | Greedy value | Ex-ante DP | Proof first | Replay first | Fixed order | Hindsight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Compiler in range | 998 | 1,012 | 1,009 | −511 | −295 | −301 | 1,239 |
| Pocket in range | 1,046 | 1,058 | 1,058 | −840 | −504 | −513 | 1,358 |
| Compiler shifted | 236 | −31 | −126 | −3,127 | −2,467 | −2,467 | 815 |
| Pocket shifted | 662 | 740 | 738 | −2,303 | −1,796 | −1,861 | 1,435 |

The learned policy authorized 94 candidates across 256 cases, made 147 abstentions justified under the quoted decision model, and missed 33 individually profitable and affordable opportunities. Those categories can overlap because the individual missed-opportunity metric uses hindsight. The budget gate denied one learned query. Zero unsupported decisions were accepted under any policy in these cases. The learned model beat the fixed-order baselines on each split. It led the value-aware baselines only on the shifted compiler split. That split supplies intentionally optimistic validity priors, so its realized advantage does not establish general superiority over a calibrated deterministic allocator.

Median compiler SMT time was 63.26 ms and median exact world forecast time was 108.3 µs in the calibration. In the same process, median proposal time was 78.7 µs for the learned network, 7.0 µs for greedy value, and 213.7 µs for the exact ex-ante search. Median whole-decision time across both domains and splits was 0.85, 0.68, and 1.39 ms respectively. These summaries mix queries and abstentions. Raw timing samples and call counts are separate. The budgeted policy comparison isolates candidate queries from reuse, which is measured separately below.

## Scope and reuse

Evidence identities hash the domain, semantics identity, subject, assumptions, and query. The compiler scope binds to the reviewed archive, binary hash, site, plan, and template. A sampled preview also binds to the two live values it compared. Changing the candidate preview invalidates the prior proof. The world forecast scope binds to the rule, width, height, target, horizon, and initial terrain and signal within its Manhattan dependency cone. A distant edit reuses the forecast. Blocking the target invalidates it. A semantics-version change also forces recomputation. The store accepts results produced by this invocation's exact adapters. The JSON exports identify evidence and support replay, while trusted import across processes is outside this experiment.

The related-world certificate uses a checked relationship between two terrain masks. For the same start, target, horizon and cardinal updates, any path through the more obstructed terrain is also a path through the less obstructed terrain. If B can arrive within the horizon and A's walls are a subset of B's, A arrives no later. Tie order then establishes A as the winner. Treating the arrivals as independent intervals leaves both winners possible. Blocking A's target violates inclusion and invalidates the certificate. The frozen forecast confirms the counterfactual.

Over 16 repeated requests, the reuse control executed the compiler proof once and made 15 scope hits. Total proof-service time was 1,133.65 ms across 16 cold executions and 68.49 ms with reuse, including the initial proof and lookup overhead. The world control moved from 2.14 ms to 0.49 ms. These paired totals time the evidence store after the scope is constructed. They do not include computing the scope or importing a trusted artifact.

## Reproduction

From visuaml-client/voxel-replay in the research Python environment:

    PYTHONPATH=.:../backend python tools/evaluate_reusable_evidence_v59.py

The command uses the published calibration and checks the complete result against outcomes.json. Add both --calibrate and --write to generate a new timing cohort and scenario archive. The archive keeps raw proposals, accepted evidence, stable scope identities, and selected options separate. No CI jobs are required.
