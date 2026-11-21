# Will opening this gate help?

A learned language controller and the exact Pocket Signal Relay rule run inside one frozen TorchScript graph. The host supplies tokenized text and a regional checkpoint tensor with its admitted horizon, gate and target offsets. The graph generates and validates calls, runs the bounded local world, formats the exact result as feedback, and generates the next request. It contains no Python callback or retained module call at the tool boundary.

The controller uses the same tiny causal Transformer architecture and generation protocol as the conditional composed-inference witness. This controller was trained on 378 synthetic text and bit-response transcripts for 700 updates. Training used no world layouts. It withheld the evaluation wording combination. The controller and composed graph were committed at `856ce40` before the paired scenes and their intervention predictions were authored. The prompt grammar is controlled English.

## Tool semantics

- **Alpha** forecasts the target after 1–8 synchronous ticks under a proposed open or wait action. It clones the wall tensor before editing. The rule uses the previous signal state, sealed boundaries, and the center plus four orthogonal neighbors.
- **Beta** forecasts the same horizon with the gate left closed. Its first argument carries the model's reported result.
- **Gamma** checks whether the straight row between the gate and target is passable after the proposed edit. Its first argument carries the model's reported result. This fixture checks corridor clearance rather than moving an agent.

The six-cell checkpoint certifies the target's six-tick dependence cone. The verifier compares every regional grid against the same rule implemented independently as scalar code. It also checks the target and shrinking dependence cone against the full-world replay at each tick. Cells near the crop edge can legitimately differ from the full world outside that cone. No simulated edit is published to a live world.

## Causal switch

The same prompt and first request, `alpha(1, 6)`, run on both unseen checkpoints.

| Condition | Graph's computed target bit | Feedback seen | Next request | Answer |
| --- | ---: | ---: | --- | ---: |
| Near | 1 | 1 | `gamma(1, 1)` | 1 |
| Delayed | 0 | 0 | `beta(0, 1)` | 0 |
| Near with feedback substituted | 1 | 0 | `beta(0, 1)` | 0 |
| Delayed with feedback substituted | 0 | 1 | `gamma(1, 1)` | 1 |

Opening the gate first changes regional signal at (9, 8) on tick one in the near case. The delayed case omits the near seed column, so the full-world signal reaches the gate at tick two and the target stays off at tick six. The regional crop lacks the delayed source but reproduces every observation within the declared target horizon.

Both precomputed-reply controls emit the same calls and answers as the integrated runs with no first forecast executed. A disabled tool, an absent checkpoint, and an actual five-cell crop declaring only five ticks of coverage all return unavailable and the model emits stop. All 12 fixture/control runs pass their separate checks for first tool selection, argument binding, exact world execution or refusal, next selection, next binding, and result use. The available executions also check the second tool against its independent reference.

Eight warm single-thread samples per case yield medians of 2.08 ms near and 2.15 ms delayed, versus 1.98 ms and 2.05 ms for their precomputed controls. Both timers include the full model interaction. These small-scene timings are separate from the fork journaling measurements.

## Inspect or reconstruct

From `visuaml-client/voxel-replay` in the research environment:

```sh
export PYTHONPATH=.:../backend
python tools/replay_pocket_world_inference.py
python tools/replay_pocket_world_inference.py --reconstruct near
```

Start `PYTHONPATH=.:../backend python tools/serve_voxel_forks.py` and `npm run dev -- --port 4182` in separate terminals. Open `/pocket-world-inference.html`. Use **Reconstruct forecast** to run the archived request on the frozen forecast entry point. Model requests and feedback remain labeled as recorded. The viewer labels the freshly reconstructed frames, or shows the saved forecast when the local endpoint is unavailable. Reconstruction checks the checkpoint identity and matches every recorded regional frame.

[Checkpoint pairs](holdout.json) · [Recorded journal](replay.json) · [Frozen graph IR](inference.ir) · [Raw timings](timings.json)
