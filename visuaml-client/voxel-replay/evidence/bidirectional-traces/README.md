# Bidirectional trace proposals and exact replay

This experiment tests masked trace generation, corrupted-trace repair, and goal-conditioned input proposals. Exact transition replay scores the model proposals. The existing v71 causal WASM backend remains independently reproducible. The causal-mask control in this experiment is a newly trained model with the same architecture and data generator as the bidirectional model. It is not the v71 engine and its batch timings are not comparable to v71's WASM timing.

## Forward generation and endpoint conditioning

The two attention models train on the same seeded stream of synthetic 16-step traces. Each input contains a 16-entry transition table, an initial word, and a trace with randomly masked fields. Evaluation uses a separate seed with 96 generated instances per family and length. At 24 steps, later absolute position embeddings are outside the training length. Countdown reuses a fixed transition rule and possible initial states from training. The random table family tests previously unsampled transition tables from the same generator.

The decoder fills one confidence-ranked block of masked tokens per refinement round. Revealed tokens stay fixed. This is an absorbing-mask diffusion-style schedule without continuous Gaussian diffusion or remasking. Each round evaluates the whole sequence. Results include full exact trace counts, field accuracy, ordered first-divergence counts, representative failed traces, and single wall-time observations for 96 items per batch. Scores compare the proposals directly with exact scalar replay.

| Family and horizon | Bidirectional from initial only, 1 / 2 / 4 / 8 rounds | Bidirectional with endpoint, 1 / 2 / 4 / 8 rounds | Causal-mask from initial only, 1 / 2 / 4 / 8 rounds |
| --- | --- | --- | --- |
| Countdown, 16 | 89 / 96 / 96 / 96 | 89 / 96 / 96 / 96 | 96 / 96 / 96 / 96 |
| Countdown, 24 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| Pointer table, 16 | 3 / 5 / 6 / 5 | 3 / 5 / 5 / 4 | 1 / 2 / 2 / 2 |
| Pointer table, 24 | 2 / 5 / 7 / 8 | 2 / 5 / 7 / 8 | 2 / 2 / 3 / 4 |

The endpoint intervention changes only the final visible word. A deterministic table and initial word fix the legitimate ending, so all forced-wrong endings are rejected. Interior changes are counted separately to measure propagation, rather than treating rejection of an intentionally wrong ending as a learned result.

At horizon 24, serial exact replay uses 24 dependent lookups. Naive bulk-synchronous propagation uses 24 rounds over 24 lanes. Exact table doubling uses four rounds to compose powers and five lookup rounds, accounting for 118 table/application lookups and a table of 80 entries. That control exploits the complete static 16-entry transition function. It does not establish the same result for arbitrary WASM memory and control state.

## Trace repair and real VM boundaries

Each synthetic corruption changes one interior word. The final word matches. For each possible interior or final suspect, the model scores the supplied word with that slot masked. It then replaces the least likely supplied word with the argmax prediction for that slot. At pointer horizon 16 it localized 14/96 and repaired 6/96 bidirectionally, versus 15/96 and 7/96 with a causal mask. The ground truth indexes are independently checked through ordered first-divergence comparison.

The original compiler checkpoint demonstrates the same trap at the WASM boundary. The reference and rejected plan both exit with stack `[256, 0]`, but after PC 160 the rejected plan exposes `[0]` instead of `[256]`. The archived VM witness is located by exact trace comparison. The learned repairs in this experiment act on the synthetic word traces.

## Goal-conditioned schedules

A separate bidirectional model learns masked action completion on a small passage abstraction. It receives the first tick on which the gate permits crossing, the deadline, and the horizon. It chooses an action suffix. Training horizons omit five. On all 30 horizon-five gate/deadline combinations it matches the exhaustive oracle in 29 cases. It selects the optimal schedule in 14 of 15 feasible cases and produces the intended all-Wait abstention in all 15 infeasible cases.

Given the original suspended Rust checkpoint's open-passage availability, the learned model proposed Walk then four Waits. A fresh fork verified all resulting pages through Wasmtime with packed fallback blocked and matched the archived exhaustive 32-schedule optimum. Wait-only remained blocked and the delayed Walk crossed at tick five. The model receives the availability as a supplied fact. The transfer is one real checkpoint with a simple abstraction, and establishes neither inference of hidden terrain nor policy generalization to additional WASM guests.

## Run

From `visuaml-client/voxel-replay`, using the research Python environment:

```sh
export PYTHONPATH=.:../backend
python tools/experiment_bidirectional_traces.py
python tools/replay_compiler_fork.py
```

The first command loads the committed weights, reruns the holdouts and world fork, and checks archived results except wall time. Use `--train` to retrain all three models and regenerate the archive, or `--record` to regenerate with committed weights. The study uses a single training seed per task and a single CPU measurement per batch configuration.
