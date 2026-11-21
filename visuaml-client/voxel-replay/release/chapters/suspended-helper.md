# The helper returns 7 after it writes 0

This executable chapter uses the same library as the CLI. Run it from
`visuaml-client/voxel-replay`.

```sh
PYTHONPATH=../backend python -m doctest release/chapters/suspended-helper.md
```

## Inspect the boundary

The Rust helper first reads the block material into a local and then clears
the cell. The caller is still suspended. A durable continuation therefore
includes the empty cell, the live local, and the caller's frame.

```pycon
>>> import json, struct
>>> from pathlib import Path
>>> import torch
>>> from visuaml.replay import check_report, first_divergence, open_bundle
>>> torch.set_num_threads(1)
>>> report = json.loads(Path("release/examples/suspended-passage-report.json").read_text())
>>> replay, checkpoint = open_bundle(report["bundle"])
>>> state = replay.restore(checkpoint)
>>> cell = 65512
>>> memory = int.from_bytes(bytes(state.vm.page[cell:cell + 4].tolist()), "little")
>>> memory, int(state.vm.locals[1]), int(state.depth), int(state.vm.pc)
(0, 7, 1, 11)

```

## Resume and check the trace

The next transition uses the restored state. The reference graph owns the
return and the caller's resumption. The report checks the complete ordered
suffix and final page identity against the recorded result.

```pycon
>>> result = check_report(report)
>>> trace = result["trace"]
>>> first_caller = next(row for row in trace if row["depth"] == 0)
>>> (len(trace), first_caller["stack"], result["returned"])
(17, [65512, 7], 0)
>>> sum(row["access"][:2] == [2, cell] for row in trace)
0
>>> replay.checkpoint(state) == checkpoint
True

```

## Omit the live material on purpose

Make an independent continuation and set the material local to zero. This
represents an exporter that preserved the visible world but lost part of
the computation. Produce a valid serialized checkpoint of the altered
state so the experiment tests continuation behavior after integrity checks.

```pycon
>>> altered = replay.restore(checkpoint)
>>> altered.vm.locals[1] = 0
>>> candidate = replay.finish(replay.checkpoint(altered))["trace"]
>>> mismatch = first_divergence(trace, candidate)
>>> (mismatch, trace[mismatch]["pc"])
(0, 11)
>>> resume_return = next(row for row in candidate if row["depth"] == 0)
>>> first_caller["stack"], resume_return["stack"]
([65512, 7], [65512, 0])
>>> int(state.vm.locals[1])
7

```

The state digest detects the lost local at the first resumed boundary.
Later, the caller sees the wrong return value. An imported run owns its
mutable state, so the original continuation remains intact.

## Capture at the same boundary

A caller who wants a checkpoint can advance to the requested instruction
without building a trace for the prefix. Both paths use the same graph
transition. Only the observed path serializes the entire machine state at
each step.

```pycon
>>> page = bytearray(65536)
>>> struct.pack_into("<6i", page, cell, 7, 1, 0, 0, -1, 0)
>>> running = replay.start(cell, bytes(page), 2000)
>>> for _ in range(14):
...     running = replay.advance(running)
>>> replay.checkpoint(running) == checkpoint
True

```
