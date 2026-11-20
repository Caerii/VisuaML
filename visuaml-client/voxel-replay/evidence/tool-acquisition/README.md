# Frozen tool acquisition

The registry-aware controller and its composed CPU TorchScript graph were frozen in commit `ecb8682` before the evaluation tool images and prompts were written. All evaluation cases use the same artifacts. The three programs are first seen at evaluation. The names, interface, and semantic families were present in training. See [the interface](INTERFACE.md) for the training distribution and the execution contract.

## Results

| Evaluation | Count |
| --- | --- |
| Planned controlled-English cases with correct final answer or stop | 6 / 9 |
| Correct tool selection in those cases | 12 / 16 requested calls |
| Correct argument binding in those cases | 16 / 16 requested calls |
| Exact executor traces against the scalar oracle | 14 / 14 executed calls |
| Paraphrase restricted to known words | 0 / 1, malformed request |
| Free paraphrase | 0 / 1, rejected at tokenization |
| Exploratory reverse intervention | 0 / 1, wrong second tool |

The original task used the fresh offset and boost programs to reach 20. A separate runtime input selected the fresh trim program and reached 7. One swapped-description case correctly moved the boost request to a new name. Renaming all three tools or reordering the textual descriptions produced incorrect second requests in the respective tests. These results demonstrate use of new executable programs within familiar interfaces, with unreliable routing across descriptions and feedback.

For the planned intervention, the first program computed 10 in both executions. Before running the trial, the expected second calls were `epsilon(10, 5)` and `zeta(15, 5)`. The replay first differs at event 2, when the reported result becomes 15. At event 3 the model emits `epsilon(15, 5)`. The executor correctly runs that wrong request and returns 25. The independently expected answer is 5. An exploratory reverse intervention from 13 to 8 also changes the argument while retaining the same tool.

Disabling the first tool and giving it two cycles both cause the model to stop after the corresponding feedback. On the invalid-repeat prompt the model emits the wrong second name but binds the intended zero repeat. The graph rejects it before execution and the model stops. These two outcomes are scored separately in the replay.

## Reproduce

From `visuaml-client/voxel-replay` with Python and PyTorch installed:

```sh
python evidence/tool-acquisition/make_holdout.py
PYTHONPATH=.:../backend python tools/replay_tool_acquisition.py --open evidence/tool-acquisition/replay.json
npm run dev
```

Open `/replay-viewer.html` and select **Open tool acquisition replay**. The viewer shows saved data and lets you download and reopen it. The CLI reruns the artifact and checks the saved record, including failures. The replay includes artifact SHA-256 hashes, expected calls, program instructions, exact execution traces, and each case’s first wrong event.

The host tokenizes prompts, supplies role and fuel tensors, and admits program images. Decode, name lookup, argument checks, program execution, and feedback run inside the composed graph. The inlined artifact is inspectable in `inference.ir`. The replay checks all nested blocks for retained module calls or Python callbacks and verifies the program-image tensors remain unchanged. The original beta/gamma replay and inherited numeric audit also pass. Conventional causal-transformer lowering remains a separate experiment.

`timings.json` records six calls per tokenizable case. The original run took about 8.0 ms median over the last four calls on one CPU thread, including graph-side trace capture. Its first two calls took 61.4 and 25.5 ms. Input preparation, loading, and JSON serialization are outside the timer.
