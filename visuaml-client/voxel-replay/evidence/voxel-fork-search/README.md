# Find a way through

The debugger searches from the suspended Rust helper in the passage scene. Its baseline is five future Wait actions. The search enumerates all 32 schedules over Wait and Walk, including the baseline and all five one-action changes. It ranks legal crossings by number of edits, then tick of first crossing, then input order for deterministic ties.

The winner changes tick 1 to Walk. Zero edits never cross within the horizon, and all five one-edit schedules cross; tick 1 wins the tie. The first changed input and first world effect both occur at tick 1. The world comparison excludes the input word and aligns the terrain, position, observed material and commit count by tick. Ordered comparison finds the first difference even if a later state matches. The detailed replay shows the caller's `i32.store` at prepared PC 28 updating byte 65520 from 0 to 1. The selected session carries the input edit, checkpoint hash, aligned results, and both source traces around that tick.

## Continuation contract

The input-log checkpoint wraps the versioned direct-call VM state with the execution-semantics identifier, consumed-event cursor, input log and hash, retired source count, and buffered tick outputs. The VM payload still holds the identities, next PC, operand stack, locals, caller frames, fuel, and full page. This witness has no random stream or external effects. The buffered outputs are data and produce no sound, network traffic, or file changes.

`export_state(run, after_source_step)` steps the ordinary source path to the requested boundary in the live invocation. `restore` allocates independent memory and restores VM state and the input cursor. `fork` does the same and applies overrides only to future events. The suspended tick finishes before any override is delivered. Guest fuel survives restore. The original in-call verifier checks 17 remaining boundaries in fresh processes. The search verifier also checks a later checkpoint with an already buffered output. It tests corrupt input, a consumed-event edit, and independent restored pages. The baseline, winner, and all-Walk branch match Wasmtime at completed ticks, and the detailed winner and baseline match the saved suspended suffix with packed fallback blocked.

## Measurements

Each branch uses a private **65,536-byte** linear page. The search processes branches serially. Saved timing samples cover checkpoint export, restore, isolated page clone, fork including restore, and all execution per branch. The branch throughput denominator includes restore, the suspended tick suffix and future ticks. It excludes the additional verification replays and process startup. This measures the search workload independently of interactive latency.

This run completed 160 future branch-ticks at 72.7 future branch-ticks/s. Median export was 0.53 ms, restore 0.66 ms, fork 0.74 ms, and branch execution 66.1 ms. The isolated 64 KiB page clone took 5.0 µs at the median. Execution dominated this experiment. Keep the 64 KiB page copy as the baseline. The measurement excludes copy-on-write and larger memory capacities.

## Play and replay

Run `npm ci` and `npm run dev -- --port 4180` from `visuaml-client/voxel-replay`. Start the loopback backend in a separate terminal for live graph execution:

```sh
export PYTHONPATH=.:../backend OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python tools/serve_voxel_forks.py
```

Open [Suspended Passage](http://localhost:4180/voxel-call-checkpoint.html), press **Find a way through**, then **Open winning write**. The UI labels live search and saved results separately. The source view walks verified records. The backend runs only the fixed local witness.

To replay a downloaded checkpoint and winning session in a fresh process:

```sh
python tools/search_voxel_forks.py --checkpoint evidence/voxel-fork-search/checkpoint.json \
  --session evidence/voxel-fork-search/winning-session.json
```

Run `python tools/search_voxel_forks.py` to reproduce the full enumeration and compare the archived results. Run `python tools/replay_voxel_call_checkpoint.py` for the original fresh-process continuation checks.

[Watch the GIF](find-a-way-through.gif) · [Checkpoint](checkpoint.json) · [Winning session](winning-session.json) · [All branch records and raw timings](search.json)
