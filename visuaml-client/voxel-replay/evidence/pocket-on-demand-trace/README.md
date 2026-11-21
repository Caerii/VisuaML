# Compact journals and on-demand instruction replay

The frozen WASM guest, graph, resident broker, and indexed observation cache are unchanged. Full audit records the original checkpoint digest, stack and locals on each instruction. Compact execution uses the same graph transition with thin, transient PC and access records. Its retained journal records branch and sealed-history identity, the bound observation and value, retired counts, committed stores, and the ending guest-page digest. Checkpoint anchors capture the waiting VM before the LOAD. Anchors use compressed portable envelopes.

Two cadences retain either the first pending checkpoint per fork or every pending checkpoint. The first cadence reconstructs a later selected interval by privately replaying the earlier recorded replies and checking the resulting page and next pending checkpoint identities. It keeps the detailed trace only for the selected interval. The compact journal establishes recorded boundary and output equivalence. Intermediate stacks reconstructed afterward describe the replay. A compiler change that preserves final outputs could still change an unrecorded stack. The separate full-audit compiler test detects the original wrong intermediate stack `[256]` versus `[0]` despite the same region exit.

## The interval that matters

The remote fork with the wall removed receives 1 at PC 38. The archived anchor and bound reply reconstruct its LOAD, subsequent branch, and action store. The reconstructed trace agrees, instruction for instruction, with the corresponding suffix from the fully traced run. The original scheduled action produces the gate edit. Reconstruction restores a private page and has no world-effect sink, so opening the trace repeatedly delivers zero additional effects.

The consumer reuses its first five answers. At tick 6 it observes the effect of opening the gate and reads 1. Starting from the consumer's first journal anchor and replaying the intervening replies reconstructs the tick-6 suffix, which also matches a full-world trace. The browser viewer offers the live producer reconstruction through a fixed local endpoint and labels the saved downstream reconstruction separately. If the endpoint is unavailable the viewer labels the saved producer trace.

## Measured workload

Three alternating single-thread CPU trials per mode, with identical world requests. Whole workload timing spans the fork runner, including the admitted VM, recovery, checkpoint and journal capture, guest execution and the existing completed-invocation verification. Independent full-world comparison and final JSON serialization are outside this timing. Guest step time covers source dispatch and the frozen graph call. Instruction capture measures conversion of the pulse to trace evidence. Discrete boundary checkpoint work stays in total runtime.

| Forks / reads | Mode | Total median | Instruction capture | Extra journal capture | Retained JSON | Anchor payload |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 18 / 108 | Full | 1,629 ms | 673 ms | 0 | 591,186 B | 0 |
| 18 / 108 | Compact, first | 903 ms | 10 ms | 10 ms | 205,716 B | 16,204 B |
| 18 / 108 | Compact, every | 951 ms | 10 ms | 47 ms | 287,394 B | 97,252 B |
| 36 / 216 | Full | 3,220 ms | 1,356 ms | 0 | 1,178,875 B | 0 |
| 36 / 216 | Compact, first | 1,898 ms | 21 ms | 21 ms | 410,398 B | 32,412 B |
| 36 / 216 | Compact, every | 1,872 ms | 21 ms | 93 ms | 573,826 B | 194,580 B |

The 18-fork modes all produce 84 hits, read 11,130 source bytes, and recompute 1,553 cell-ticks. At 36 forks the hit count becomes 192 and read and recomputation work stays the same. Compact first cuts retained JSON by about 65%. These byte counts describe serialized evidence, not live allocator usage. Compact first stores 18 or 36 compressed anchors; compact every stores 108 or 216. The extra anchor work is visible even though the aggregate 36-fork measurements are close. The samples also include compressed whole-trial size for comparison.

Opening the selected remote trace takes **8.70 ms median** over 12 warm samples. This includes private restore, reply delivery, traced guest execution and the Wasmtime check. It excludes server startup, model loading, and browser transport. [Raw measurements](timings.json)

## Open the replay

From `visuaml-client/voxel-replay` in the research environment:

```sh
export PYTHONPATH=.:../backend
python tools/replay_on_demand_trace.py
python tools/replay_on_demand_trace.py --reconstruct evidence/pocket-on-demand-trace/pending.json
python tools/replay_compiler_fork.py
```

Start `PYTHONPATH=.:../backend python tools/serve_voxel_forks.py` and `npm run dev -- --port 4182` in separate terminals. Open `/pocket-coupled.html` and select **Open remote load trace**. [Exact replay and compact journal](replay.json) · [Downloadable pending LOAD](pending.json)
