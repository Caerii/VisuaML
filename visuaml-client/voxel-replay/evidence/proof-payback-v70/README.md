# VisuaML v70 proof payback across forks

This experiment follows v69 at `04e9f7f`. It measures whether a checked optimization is worth deploying when a related revision can reuse its evidence. The baseline is the v68 reload stencil. The candidate is the v68 byte-pointer lowering. Each executes 1,024 records per call on a one-page WASM memory. Proofs authorize successful-completion full-page equivalence for the supported stencil profile.

## Design

Three pilot layouts from seed 7001 determine the batch size by estimating the midpoint between cold and shared break-even, clamped to 256–16,000 calls. This run selected 5,717 calls. The 20 held-out layouts from seed 7002 use identical page inputs and call counts across treatments within each layout. Each original and edited layout has its own matching reload baseline. Each trial creates a revision, edits its first coefficient on a fork, and creates a distinct descendant restoring the original coefficients.

| Treatment | Evidence available at the start |
| --- | --- |
| Cold | Empty store and empty checker memo |
| Shared | Matching root and checker memo populated by the donor proof |
| Edited cold | Empty store and empty checker memo |
| Stale repaired | Prior root is rejected for the edited layout. Compatible leaf results remain available for repair. |
| Edited shared | Matching edited root and checker memo populated by the repair |

Shared treatments load and admit the root, regenerate its scoped binary, and deploy a new Wasmtime instance. Original and edited treatments use identical respective binaries. The edited shared case runs after repair. Other treatment order is randomized per layout, and baseline/candidate call order alternates. Deployment includes compilation, instantiation, page initialization, and a first run checked against the independent full-page oracle. Timed calls use persistent instances. The preexisting reload setup is outside the adoption cost; the reload call batch is timed for each case.

Net time saved is reload execution minus candidate execution, proof/retrieval, stale rejection where applicable, and candidate deployment. The donor proof is charged separately in the campaign contrast. Intervals are percentile bootstrap intervals for medians, resampling held-out layouts. They describe variability within this run on one CPU and Wasmtime 36.0.0 with Cranelift optimization set to `none`.

## Held-out results

| Treatment | Median proof / admission | Median rejection | Median net savings | 95% interval | Positive layouts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cold | 21.02 ms | — | −9.27 ms | [−10.88, −7.89] ms | 1/20 |
| Shared | 3.57 ms | — | +9.42 ms | [+8.00, +10.94] ms | 20/20 |
| Edited cold | 20.95 ms | — | −7.61 ms | [−9.19, −5.60] ms | 1/20 |
| Stale repaired | 8.81 ms | 0.36 ms | +4.22 ms | [+3.47, +4.74] ms | 17/20 |
| Edited shared | 3.59 ms | — | +9.51 ms | [+9.08, +10.53] ms | 20/20 |

The paired shared-minus-cold median is +16.83 ms [14.69, 22.37]. The sign reverses in 19 of 20 layouts. Repair issues two new leaf checks per edit and improves median net savings over an empty library by 10.16 ms [8.96, 13.60]. Charging the donor proof against one shared batch costs 10.70 ms median [−12.95, −9.72]. Dividing the donor median by the shared savings median gives about 2.2 equivalent revisits under a constant-cost approximation. The experiment executes one revisit per case; that extrapolation needs a sequential campaign to confirm.

The shared timings include a warm validation memo. The portable archive takes the full independent admission path with an empty validation memo. It demonstrates validity after transport; the reported shared speed applies to a resident checked library. The v69 optimization-level controls still govern the scope of any performance claim. There is one candidate family, one machine, and one batch size selected from disjoint pilot layouts.

## Reproduce locally

From `visuaml-client/voxel-replay` in the existing research Python environment:

```sh
export PYTHONPATH=.:../backend
python tools/measure_payback_v70.py
python tools/summarize_payback_v70.py
```

The first command regenerates input pages, verifies the revision chain, re-admits each archived certificate with a fresh verifier cache, regenerates binaries and verifies their hashes, checks full-page outputs against the independent oracle, and requires rejection of the original root on the edited layout. Use `--measure` to rerun timing and replace `samples.json` and `proofs.json`. The summary command prints derived statistics for the current samples.
