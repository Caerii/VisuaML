# Decision query policy

The original checked-decision graph missed one-call world certificates by proposing to abstain after a decisive forecast. The adapted policy is a single learned proposal network trained on 2,262 synthetic interval states, with training samples balanced by proposal class. It also learns to request universal evidence first for the reviewed compiler template. The previous graph's stopping checker, evidence representation, simulator, compiler adapter, and call budgets are reused directly. The controller was frozen at 9248847 before these holdout fixtures were authored.

## Pocket

The holdout has 96 new coordinate and terrain fixtures across four corridor outcome classes. The exact world and proposed effects remain hidden from the controller. It sees the same evidence inputs as the earlier network.

| Objective | Call cap | Original certificates | Adapted | Fixed order | Calls per policy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Earliest arrival | 1 | 0/96 | 24/96 | 24/96 | 96 |
| Prefer no edit | 1 | 24/96 | 24/96 | 24/96 | 96 |
| Earliest arrival | 2 | 48/96 | 48/96 | 48/96 | 168 |
| Prefer no edit | 2 | 48/96 | 48/96 | 48/96 | 168 |

The 24 recovered certificates use the same first forecast as the original policy. The change is the model's raw stopping proposal. The remaining 72 one-call runs per objective were appropriate abstentions. Under two calls, each objective had 48 appropriate abstentions. The adapted model had zero unsupported recommendations. Full exhaustive search resolves 96/96 with three queries each and supplies the independent full-world oracle. The separate budgeted oracle tries every allowed query order against the same checker, using actual outcomes offline to identify which first queries could lead to a certificate.

The world interface presents the same initial geometric arrival bound for every action, with a greater cost for edits when requested. It offers no action-specific layout clue that would justify forecasting one unseen edit before another. For worlds certified in two calls, the minimal pair can be asked in either order. This experiment recovers the certificates available under those observations and does not establish a world query-order advantage over fixed order.

## Compiler

The evaluation uses two actual ordered local.get sites in Passage and Terrain. For each site it checks both a valid plan and a wrong-preview candidate, using live locals and an equal-locals control. Equal locals can hide the wrong preview from a targeted replay. They cannot establish universal equivalence. SMT checks both original source boundaries, and the replay runs the unchanged guest through the frozen reference and Wasmtime.

All three policies certify the right plan in all eight cases at either call cap. At cap two, the original policy starts with targeted replay and spends 14 calls. The adapted and fixed policies start with universal equivalence and spend eight. The exhaustive comparison performs both checks for 16 calls. Both checks cost one budget unit here. The measured reduction is calls spent under that accounting.

One network chooses the Pocket forecast and the compiler equivalence query. It matches the deterministic policy on these bounded tasks. The evidence reports raw proposals and checker verdicts separately, including the old policy's missed certificates and the cases where successful compiler replays leave equivalence unresolved.

## Reproduce

From visuaml-client/voxel-replay in the research Python environment:

    PYTHONPATH=.:../backend python tools/evaluate_decision_query_policy.py

The command verifies the archived fixtures, proposal journals, oracle results, SMT checks, and actual replay evidence in outcomes.json. The paired comparison pins both graph hashes. The previous checked-decision benchmark remains runnable independently.
