# Compiler Lab v5 — placement sensitivity under uncertain costs

The compiler's existing binary min-cut planner optimizes a **declared additive, hypothetical** cost function for one profile. V5 adds `placement_robustness.py`, which evaluates a set of profiles and studies how often a candidate placement changes.

## Capabilities

- Independent minimum-cut optimum and all-host baseline for each named cost profile.
- Per-operation counts of how many supplied scenarios put that operation on a hypothetical device (these are not probabilities).
- An auditable cost evaluator for **any** legal binary assignment, including transfer edges.
- For up to 12 device-eligible operations by default, exhaustive search for the assignment with the **minimum worst-case additive regret**, relative to the per-scenario optimum.
- Larger programs: candidate search among per-scenario optima and all-host; explicitly labeled **non-exact**.
- Closed-world admission for profiles, missing tensor transfer sizes, nonfinite objective values, and illegal device placements.

## Demo integration

`examples/compiler_lab_demo.py` includes a `hypothetical_placement_robustness` object alongside the existing `hypothetical_partition_sweep` report. The `/compiler` React route renders a sensitivity summary. CPU witness validation and the original backends remain unchanged.

Run the focused tests:

```sh
cd visuaml-client
python -m pytest -q backend/tests/test_placement_robustness.py
```

Run an end-to-end sampled report:

```sh
PYTHONPATH=backend python backend/examples/compiler_lab_demo.py --scenario mlp --out /tmp/visuaml-v5
```

## Scientific and engineering limitations

This is **not** measured Ayo performance, and it is not a photonic lowering backend. Exact minimax regret holds only for the stated additive binary cost model and the admitted number of eligible nodes. Profiles are illustrative *scenarios*, not posterior samples or confidence intervals. Real placement decisions will need calibration, memory and concurrency constraints, transfer aggregation, error budgets, kernel availability, and actual hardware execution.

To make the scientific workflow useful, preserve the scenario definitions and versioned cost profiles with every report; do not describe sample-to-sample placement frequency as a probability of hardware feasibility.
