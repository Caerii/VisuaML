# Stop on the compiled runner and inspect on the reference runner

Run this executable chapter from `visuaml-client/voxel-replay` on Linux with
the [release dependencies](../README.md) and a C++17 compiler.

```sh
PYTHONPATH=../backend python -m doctest release/chapters/runner-handoff.md
```

The two runners share one portable checkpoint format. The ordinary CPU
runner uses bounded batches. The reference runner can replay any original
source step from a checkpoint and explain its stack and memory effects.

```pycon
>>> import json, struct
>>> from pathlib import Path
>>> import torch
>>> from visuaml.replay import check_report, make_bundle, open_bundle
>>> torch.set_num_threads(1)
>>> report = json.loads(Path("release/examples/suspended-passage-report.json").read_text())
>>> replay, checkpoint = open_bundle(report["bundle"])
>>> initial = bytearray(65536)
>>> struct.pack_into("<6i", initial, 65512, 7, 1, 0, 0, -1, 0)
>>> compiled = make_bundle(replay, 65512, initial, 2000, 14, runner="compiled")
>>> compiled == report["bundle"]
True
>>> compiled["checkpoint"]["depth"]
1
>>> traced = check_report(dict(report, bundle=compiled))
>>> len(traced["trace"])
17
>>> next(row["stack"] for row in traced["trace"] if row["depth"] == 0)
[65512, 7]
>>> any(row["access"][:2] == [2, 65512] for row in traced["trace"])
False

```

Import the original reference checkpoint into the compiled runner. The
restored state owns a private page. At completion, status, remaining
fuel, return value, and page digest agree with the reference report.

```pycon
>>> final = replay.run(replay.restore(checkpoint), runner="compiled")
>>> result = replay.summary(final)
>>> result == {key: traced[key] for key in result}
True

```

## Measured handoff

Run the paired verification and timing suite from the same directory.

```sh
PYTHONPATH=.:../backend python tools/verify_runner_handoff.py
```

| Warm suffix | Source steps after capture | Reference median | Compiled median | Pairs |
| --- | ---: | ---: | ---: | ---: |
| Passage, after the store | 17 | 6.28 ms | 0.74 ms | 32 |
| Full BFS workload, resumed after source step 1,024 | 40,567 | 15.15 s | 0.31 s | 3 |

These measurements time untraced execution after import. Median restore
times were 0.45 ms for Passage and 0.55–0.60 ms for BFS. The first compiled
prefix in this process included loading and registering an existing local
CPU library. It took 208 ms for Passage. Separate cold compiler time and
full trace capture are outside the warm measurements. These are local CPU
results compared against the source-step release runner. The compiled path
uses the existing ordinary CPU graph without optional per-program fusions.
The portable report continues to use the reference graph identity.

The test compares all 30 active Passage boundaries, including 11 cuts
between eligible ordinary instructions. It also checks fuel cutoffs and
an out-of-range store that traps without writing. Both backends execute
the full BFS task including the captured prefix and agree with Wasmtime on
the return and all 64 KiB of memory. [Raw measurements](runner-handoff-timings.json)
