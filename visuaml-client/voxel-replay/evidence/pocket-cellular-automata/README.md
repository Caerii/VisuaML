# Pocket Signal Relay

Open `/pocket-cellular.html` from the Pocket launch page. Use **Dig gate** and **Step** to commit one edit and advance the two histories side by side. **Play** and **Reset** are also available. The 160 × 144 display draws a 20 × 18 grid. The live scene runs the verified browser sparse update. The frozen tensor rule and independent scalar oracle generate the downloadable replay. The scene is a Pocket browser route.

## Neighborhood operation

The tensor operation declares `neighborhood_any`, offsets for the cell itself and the four orthogonal neighbors, sealed edges, wall masking, and synchronous commit. The lowering generates a convolution kernel from the declared offsets. It sums signal bits, clamps to one, and clears walls. Binary32 exactly represents the admitted local sums from zero through five. A gate edit changes the wall mask before the next signal tick. All output cells read the same previous signal state.

Both histories match the scalar oracle at all 32 ticks. Running them in a single tensor batch preserves each grid exactly. The browser's sparse and dense paths are also checked against every saved grid. The first signal difference is at cell (9, 8) on tick one, and each later difference lies within the forward radius-one cone.

For a checkpoint before the edit, the signal at cell (14, 8) on tick six matches the full grid with a rectangular radius-six halo. Shrinking to radius five excludes the nearest initial signal and gives zero instead of one. This is one concrete failure at the causal boundary.

## Frontier scaling

A cell with unchanged inputs retains its previous output after the initial update. The sparse implementation schedules changed cells and their neighbors. A queued wall edit also schedules its local neighborhood. Updates are staged before committing. The rule has no evolving global state or timers.

Seven alternating paired Node.js timings per scene, each covering twelve synchronous ticks. Setup and result comparison are outside the timer. Both paths are JavaScript rule evaluators; the timings compare traversal strategies.

| Grid and seeds | Dense cell evaluations | Sparse cell evaluations | Dense median | Sparse median |
| --- | ---: | ---: | ---: | ---: |
| 32 × 32, 1 seed | 12,288 | 740 | 1.16 ms | 0.15 ms |
| 128 × 128, 1 seed | 196,608 | 740 | 19.04 ms | 0.08 ms |
| 384 × 384, 1 seed | 1,769,472 | 740 | 169.91 ms | 0.08 ms |
| 128 × 128, 16 seeds | 196,608 | 9,684 | 16.46 ms | 0.66 ms |
| 128 × 128, 64 seeds | 196,608 | 36,384 | 15.63 ms | 2.72 ms |

Single-seed sparse work stays constant because the wave stays inside the same radius during these twelve ticks. At fixed size, more seeds increase frontier work. Initial grid allocation and seeding still depend on grid dimensions. Rendering cost is outside this comparison. [Raw timing samples](timings.json)

## Learned local proposal

Two seeds at widths four and twelve train on five-bit neighborhoods. The learned proposal is gated by the exact wall, persistence, and local reachability constraints. Each guarded model matches all 24 held-out layouts through twelve ticks. Layouts range from 19 × 31 to 33 × 67 and have randomized walls and seeds.

The raw proposal is scored separately on all 32 local patterns. Both width-four models score 31 of 32 and wrongly activate the empty neighborhood. Both width-twelve models score 32 of 32. Before the gate update in the patch case, the gate cell's hidden activation is replaced with the activation from an empty neighborhood. The prediction is that the first signal difference will be the gate itself. Both width-twelve models meet that prediction and subsequent differences stay within its future cone. Neither width-four model changes under the patch, consistent with its active empty-neighborhood prediction. This is a small, controlled study of a designated activation bank.

## Reproduce

From `visuaml-client/voxel-replay` with the research Python environment and Node installed:

```sh
python tools/replay_cellular_automata.py
node tools/measure_cellular_automata.mjs
npm test
npm run dev -- --port 4182
```

`--write` rebuilds the frozen tensor rule and the four learned artifacts with their seeded training runs. [Replay, operation declaration, and patch results](replay.json)
