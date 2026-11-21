# Pocket Signal Relay

Open `/pocket-cellular.html` from the Pocket launch page. Use **Dig gate** and **Step** to commit one edit and advance the two histories side by side. **Play** and **Reset** are also available. The 160 × 144 display draws a 20 × 18 grid. The live scene runs the verified browser sparse update. The frozen tensor rule and independent scalar oracle generate the downloadable replay. The scene is a Pocket browser route.

## Neighborhood operation

The tensor operation declares `neighborhood_any`, offsets for the cell itself and the four orthogonal neighbors, sealed edges, wall masking, and synchronous commit. The lowering generates a convolution kernel from the declared offsets. It sums signal bits, clamps to one, and clears walls. Binary32 exactly represents the admitted local sums from zero through five. A gate edit changes the wall mask before the next signal tick. All output cells read the same previous signal state.

Both histories match the scalar oracle at all 32 ticks. Running them in a single tensor batch preserves each grid exactly. The browser's sparse and dense paths are also checked against every saved grid. The first signal difference is at cell (9, 8) on tick one, and each later difference lies within the forward radius-one cone.

### The first wrong target

The checkpoint is taken before the gate edit. Updates use the previous tick's state, sealed boundaries, and a radius-one orthogonal neighborhood including the center. Over six synchronous ticks, the target's dependencies can travel at most six cells in Manhattan distance. The six-cell rectangle contains that full backward cone.

The full run and the six-cell crop agree at target **(14, 8)** for ticks zero through six. The five-cell crop first differs at **tick six**, when the full run and the six-cell crop read 1 and the five-cell crop reads 0. The omitted seed is at **(8, 8)** on tick zero. Its signal enters the edited gate at **(9, 8)** on tick one and reaches successive cells on row 8 at one cell per tick. The six-cell checkpoint can also be opened in a fresh process with the gate edit omitted. It then produces 0 at the horizon.

The portable checkpoint carries just the region's wall and signal arrays, its world origin, the target and gate coordinates, the rule version, and a six-tick horizon. The replay needs no reference to the full grid. [Six-cell checkpoint](region-halo-6.json) · [Five-cell control](region-halo-5.json)

## Frontier scaling

A cell with unchanged inputs retains its previous output after the initial update. The sparse implementation schedules changed cells and their neighbors. A queued wall edit also schedules its local neighborhood. Updates are staged before committing. The rule has no evolving global state or timers.

Seven alternating paired Node.js timings per scene, each covering twelve synchronous ticks. Setup and result comparison are outside the timer. Both paths are JavaScript rule evaluators; the timings compare traversal strategies. Before timing each configuration, both paths match an independent Manhattan-distance oracle on every tick. The timing worlds are obstacle-free and use sealed edges.

| Grid and seeds | Dense cell evaluations | Sparse cell evaluations | Dense median | Sparse median |
| --- | ---: | ---: | ---: | ---: |
| 32 × 32, 1 seed | 12,288 | 740 | 1.17 ms | 0.08 ms |
| 128 × 128, 1 seed | 196,608 | 740 | 18.83 ms | 0.10 ms |
| 384 × 384, 1 seed | 1,769,472 | 740 | 170.83 ms | 0.08 ms |
| 128 × 128, 16 seeds | 196,608 | 9,684 | 16.12 ms | 0.66 ms |
| 128 × 128, 64 seeds | 196,608 | 36,384 | 15.22 ms | 2.69 ms |

Single-seed sparse work stays constant because the wave stays inside the same radius during these twelve ticks. At fixed size, more seeds increase frontier work. Initial grid allocation and seeding still depend on grid dimensions. Rendering cost is outside this comparison. [Raw timing samples](timings.json)

## Learned local proposal

Two seeds at widths four and twelve train on five-bit neighborhoods. The learned proposal is gated by the exact wall, persistence, and local reachability constraints. Each guarded model matches all 24 held-out layouts through twelve ticks. Layouts range from 19 × 31 to 33 × 67 and have randomized walls and seeds.

The raw proposal is scored separately on all 32 local patterns. Both width-four models score 31 of 32 and wrongly activate the empty neighborhood. Both width-twelve models score 32 of 32.

Counts below are per seed over 24 layouts and 12 ticks, with 426,528 cell updates per model. Both seeds at a given width had identical counts. The four categories partition the raw and committed binary outputs.

| Width | Accepted on | Accepted off | Changed off to on | Rejected on |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 181,575 | 0 | 0 | 244,953 |
| 12 | 181,575 | 177,249 | 0 | 67,704 |

The intervention patches only the gate cell's hidden activation on the first tick. The empty-neighborhood donor should suppress the proposal when the model represents the difference, followed by a committed difference at the open gate on tick one and at (14, 8) on tick six. Both width-twelve seeds show this chain. At the closed gate, the same patch still changes the raw proposal from 1 to 0, while the committed outcome stays 0. The two width-four seeds show neither proposal nor state change under the donor patch.

A same-side reachable donor changes neither proposal nor commit across all four models. None of the 64 norm-matched random directions, 16 per model, changes either. The evidence includes raw and committed gate outputs, per-tick target values, and spatial difference counts for every variant. Two seeds at each of two widths make these results descriptive.

## Reproduce

From `visuaml-client/voxel-replay` with the research Python environment and Node installed:

```sh
python tools/replay_cellular_automata.py
python tools/replay_cellular_automata.py --region evidence/pocket-cellular-automata/region-halo-6.json
python tools/replay_cellular_automata.py --region evidence/pocket-cellular-automata/region-halo-6.json --omit-edit
node tools/measure_cellular_automata.mjs
npm test
npm run dev -- --port 4182
```

`--write` rebuilds the frozen tensor rule and the four learned artifacts with their seeded training runs. [Replay, operation declaration, and patch results](replay.json)
