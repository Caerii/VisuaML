# Stateful assembly IR v6 — narrow, executable and effect-safe

This experimental slice adds two modules to the existing VisuaML compiler lab:

- `visuaml.compiler_lab.stateful_ir`: a closed-world, state-versioned SSA-like step IR, with effect tokens and a fail-closed effect-preservation validator.
- `visuaml.compiler_lab.assembly_state`: a deterministic one-area recurrent or externally driven model, representing winner-set snapshots, explicit dense weights, Hebbian synapse updates, and proportional refraction bias updates.

## Scientific boundary

This is **not** a full port or proven equivalence to [`Caerii/assemblies`](https://github.com/Caerii/assemblies). Its semantics are informed by that project's `numpy_explicit.project_into`, `NumpyExactEngine._exact_topk`, and `refraction_increment` contracts as reviewed against upstream commit `7b9a4d1f9e0401db44b0cfd12997bd6b9468051c`.

Specifically, the model uses explicit, preloaded float32 weights; sorted unique source neurons; summed synaptic drives; deterministic lowest-index k-WTA; Hebbian multiplicative updates with clipping; and learning-gated proportional refraction charged against **raw pre-update** drive.

An **optional** integration test compares repeated winner and connection-weight updates against the installed `NumpyExplicitEngine` on a zero-refraction, one-area recurrent subset. It is skipped in environments where `neural_assemblies` is absent; a skipped gate is not evidence of parity.

**Not implemented:** connectivity initialization/generation, multi-fiber/area schedules, norm-init, LRI, column scaling, actual GPU acceleration, plasticity source scopes, dynamic neuron recruitment, or biological fidelity. This narrow oracle is *not* enough to claim stateful neural-assemblies engine parity.

## Effect contract

Read the previous winner snapshot **before** computing drive. All tensor operations producing the selected winners happen before modifying weights or bias. The only legal effect chain in this admitted step dialect is:

1. `hebbian_write`: weights version 0 -> 1
2. `refraction_write`: bias version 0 -> 1
3. `publish_winners`: winners version 0 -> 1

Effect tokens `state_0` through `state_3` make the order explicit. A rewrite that drops, moves, duplicates, or disguises these effects is rejected even if an example happens to return the same winner mask. Numeric witnesses are still required for pure-computation rewrites; the static effect contract alone never proves equivalence.

## Run

From `visuaml-client`:

```sh
python -m pytest -q backend/tests/test_stateful_assembly_v6.py
```

This is a programmatic research bridge, not yet integrated into the React `/compiler` UI. Next acceptance gate: compare detailed state traces against a pinned installed `neural-assemblies` engine with controlled connectomes, learning, and refraction, then surface its receipts in the app. Hardware calibration remains separate until real device data are available.
