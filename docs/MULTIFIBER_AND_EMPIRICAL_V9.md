# Compiler Lab v9: multi-fiber effects and empirical-residual placement

This is an opt-in research slice on `feat/compiler-ir-backends-assembly-spike`.
It does not change existing VisuaML model upload or browser behavior.

## Multi-fiber state semantics

`multifiber_bridge.py` lowers a three-area A+B→C projection into typed, bounded
`EffectProgram` operations. A and B are immutable source winner sets for this
projection; C receives their **ordered** float32 drive contributions. One
winner set is selected, then two separately parameterized Hebbian writes are
committed, followed by the C winner-set write.

This supports only a narrow explicit-reference regime: dense float32 fibers,
fixed finite weights, one target, deterministic lowest-ID ties, bounded Hebbian
multipliers/ceiling, no refraction, inhibition, connectivity growth, per-column
normalization or hardware-specific effects. `NumpyExplicitEngine` parity is
an optional test against a pinned upstream dependency, not part of ordinary
CPU unit-test claims. The effect IR is not yet a general graph lowering from
arbitrary Python algorithms.

## Empirical-residual sensitivity

`empirical_robustness.py` layers on top of v8's holdout-admitted latency and
per-node placement. It constructs **five explicitly named cost scenarios**:
nominal, device-favorable, host-favorable, joint-upper, joint-lower. Each uses
max(train residual, holdout residual) for a supplied operator or transfer fit.
The existing minimax-regret solver evaluates these profiles; small graphs use
exhaustive assignments, larger ones clearly report restricted candidates.

These errors are *sample residual envelopes*, **not** statistical confidence
intervals, future guarantees, independently authenticated hardware traces,
or observed scenario probabilities. Transfer error is modeled per graph edge,
and overlap, caching, collective movement, scheduling, photonic effects, and
actual device execution remain outside the cost model.

In particular, passing synthetic fixtures does not justify speedup claims
about Ayo or any other physical processor.

## Reproducibility

From repository root, with the backend dependencies installed:

```bash
cd visuaml-client
PYTHONPATH=backend python -m pytest -q \
  backend/tests/test_multifiber_bridge_v9.py \
  backend/tests/test_empirical_robustness_v9.py
```

The main CI runs these tests alongside the existing suite. The pinned reference
workflow executes the three-area parity test when its dependency is available.
