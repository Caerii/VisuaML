# Paired-feedback registry routing

One frozen learned router chose the expected tool at every value from 0 to 31 in a controlled feedback sweep. The task supplies a structured numeric condition and two description fields. This scope differs from the earlier autoregressive language controller, for which the same kind of sweep left the chosen tool unchanged. This experiment changes both the training distribution and the interface, so it does not identify which change explains the gain.

The composed graph and its interface were frozen at `5ab23af` before authoring the final tool images and evaluation inputs. The same synthesized VM evaluates all admitted calls. The learned relation network receives raw numeric inputs, and the learned description scorer selects a registry handle. Row-schema validation uses only identity, argument bounds, and fuel.

## Intervention

The first tool computed 10 in both runs. Only the reported feedback changed:

| Case | Reported first value | Learned second request | Final result |
| --- | --- | --- | --- |
| Original | 10 | handle 70, value 10, repeat 2 | 16 |
| At the switch boundary | 12 | handle 860, value 12, repeat 2 | 8 |

The replay verifies that the first execution traces are identical. At the second selection, the score leader switches from handle 70 to 860. Physical row permutation of the original run preserved its full event evidence. A case with fresh handles, a new threshold and choice order, and a reversed relation each matched its independent expected request. A separate forced-correct probe bypasses only learned handle selection and exercises the same validator and executor. A forced unknown handle is rejected, as is a zero repeat. Disabled and exhausted first tools stop the bounded graph with the corresponding status.

## Results

| Evaluation | Selection | Argument binding | Exact execution | Correct final outcome |
| --- | --- | --- | --- | --- |
| Six familiar-description positive cases, with new programs | 12 / 12 | 12 / 12 | 12 / 12 | 6 / 6 |
| Two novel-description composition cases | 2 / 4 | 3 / 4 | 4 / 4 | 1 / 2 |
| All 13 structured cases, including diagnostic and failure-path cases | 22 / 24 | 23 / 24 | 21 / 21 | 12 / 13 |
| Feedback sweep, one fixed task across 32 values | 64 / 64 | 64 / 64 | 64 / 64 | 32 / 32 |

Both novel-description cases selected the wrong first tool at event 1. One still reached the expected final outcome because the diagnostic override controlled the value consumed by the second call. This makes a final-only score insufficient. Free paraphrase is marked inadmissible because this structured controller has no open-language parser. Arguments are formed deterministically from typed inputs and feedback. The binding metric captures those observed values, including consequences of a wrong previous choice, and does not measure a learned copying ability.

Exact execution counts only calls that ran at least one interpreter cycle. It compares each observed trace with the scalar oracle for the request the router actually made. An exhausted call can therefore pass exact execution. The host only encodes input, admits program images and displays observations. The inlined graph has no Python callbacks or module dispatch at the tool boundary.

## Replay

From `visuaml-client/voxel-replay` with Python and PyTorch installed:

```sh
python evidence/paired-routing/make_holdout.py
PYTHONPATH=.:../backend python tools/replay_paired_routing.py --open evidence/paired-routing/replay.json
```

The downloadable JSON includes exact requests, states, candidate scores and first failure events. Graph SHA-256 is `5550d44c8b4f3a43cad5aa7b7e065b1c5338aaafb32ca9f14cf1b00ee3b24910`. `timings.json` records six single-thread CPU calls per structured case. The original and feedback-substituted cases took about 5.86 and 5.79 ms median over the last four calls, including graph-side trace collection. The original first invocation was 55.1 ms. Program preparation, input tokenization and JSON publication are outside those timers.

## Description-only probe

The frozen graph scores the first registry selection before validating its
arguments. A separate probe supplies schemas that reject every entry, so
only that learned choice is observed. No second call, live feedback, or
tool execution occurs. Both existing novel-description cases again select
handle 860 instead of the seed-transform handle 1441. A new held-out
meaning-preserving paraphrase retains handle 1441. Exchanging the seed
and boost descriptions moves the prediction to handle 70. This small
probe records logits and exact wording for a future width or activation
study. [Description-only replay](description-only.json)

```sh
PYTHONPATH=.:../backend python tools/probe_routing_descriptions.py
```
