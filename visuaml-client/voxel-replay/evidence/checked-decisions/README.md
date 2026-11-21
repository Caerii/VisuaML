# Checked decisions

One frozen TorchScript graph uses the same exact stopping checker in two domains. A learned proposal network sees normalized cost intervals, previous queries, and remaining budget. It proposes a query, an action, Neither, or abstention. The graph records the proposal alongside the checker result and applies only accepted recommendations. The controller trained on synthetic interval states and query transcripts. The artifact was frozen at 7303e3e before the holdout fixtures were written.

## Stopping contract

Each option has an integer cost interval. The highest cost denotes infeasibility. Lower cost wins, with action index breaking ties. The checker accepts a feasible option when its worst cost beats every competitor's best cost, including the tie break. It accepts Neither when every option is known to be infeasible. Overlap can leave the result unresolved. The evaluator exhaustively checks 1,100 interval boxes of two or three options over a four-value domain. The rule is exact for independent interval completions. Dependencies between options may justify stronger bounds.

## World holdout

The simulator is unchanged. With exactly one starting cell, cardinal movement, and edits that only clear terrain, Manhattan distance is a lower bound on arrival time. For the edit objective, each edit adds nine to the cost, so any Wait arrival by tick eight wins over an edit. A forecast fixes one option's interval to its observed cost. The graph scopes replies to the current invocation, action, and horizon. It relies on the existing host contract for certified regional coverage and refuses forecasts beyond it.

| Objective | Call budget | Learned certified | Fixed-order certified |
| --- | ---: | ---: | ---: |
| Earliest arrival | 1 | 0/64 | 16/64 |
| Prefer no edit | 1 | 16/64 | 16/64 |
| Earliest arrival | 2 | 32/64 | 32/64 |
| Prefer no edit | 2 | 32/64 | 32/64 |

The fixed-order policy queries the first unresolved action and stops using the same checker. The exhaustive oracle resolves every layout by evaluating all three actions with independent full-world shortest-path search. Every graph forecast is also checked against scalar evolution. At two calls the learned and fixed-order policies each use 112 forecasts per objective, versus 192 exhaustive forecasts. The model proposed to abstain in 16 one-call speed cases where the checker already had a certificate. There were no unsupported winner proposals in the 256 learned-policy runs.

In the bound replay, Wait fails and Edit A arrives at tick four. That meets the distance bound, and the tie break settles the choice without querying B. In the invalid-shortcut replay, A arrives at eight. B could beat it, so the same two calls leave the decision unresolved. The exhaustive oracle finds that B arrives at four. The outcome archive also recomputes certificates with the distance bound removed and includes a refused forecast from insufficient checkpoint coverage.

## Execution-plan witness

The compiler adapter uses the real ordered local.get pair in the Passage guest. It compares the reference and the reviewed fused template using two versus one transition dispatches as their cost. The evaluator validates the source pair, site entry, stack capacity, fuel, exit selectors and fuel debits. Under that fixed template contract, matching the source preview selector establishes equality for all live i32 values. Separate SMT queries check both source boundaries. Actual frozen VM and Wasmtime replays are archived with the proof results. The graph ranks these plans within the reviewed template; arbitrary plan code is outside the interface.

With one call the model requested equivalence evidence. With two calls it requested a targeted replay first. A mismatch rejected the bad candidate immediately. A matching replay left the candidate unresolved, and the model requested equivalence evidence before adopting it. An equal-locals control hid the wrong preview from the targeted replay, but the universal selector check still ruled it out. Both plans have the same final stack in the wrong-preview example.

## Reproduce

From visuaml-client/voxel-replay in the research Python environment, run:

    PYTHONPATH=.:../backend python tools/evaluate_checked_decisions.py

The command verifies the full archived result. The outcome file includes the artifact and simulator hashes, raw proposals, cost intervals, verdicts, world fixtures, SMT queries and witness replays. The previous paired-goal implementation remains intact.
