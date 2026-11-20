# A natural-language controller using the frozen executor twice

The controller is a two-layer causal Transformer language model trained on 3,936 short synthetic transcripts. It greedily emits each word or number token of a tool call. The prompt supplies the slot-to-role mapping in text, such as `alpha does offset . beta does boost .`, followed by a request in controlled natural language. Slot assignments vary. Training gives independently sampled result tokens as feedback, so the model learns to use a result token without learning a predetermined arithmetic formula.

The controller and executor hashes were recorded before the final tool implementations and holdout prompts were authored. The freeze is commit `ab65b2c`. The executor is byte-identical to the existing synthesized-attention artifact. The two program images are supplied as data. Offset copies through word 3 and adds 4. Boost loops over additions of 1 and 2, with a runtime repeat count. Holdout seed/repeat pairs are excluded from training by the declared modulo-7 split. The evaluation tests new combinations in a small vocabulary and controlled sentence templates.

The scheduler parses the generated protocol, binds the generated slot to the declared program image, validates typed arguments, starts a fresh executor inference, and inserts the numeric result or status token into the language-model context. It computes neither the second call's arguments nor the answer. The language model and executor are separate frozen graphs, called by the host in sequence. Natural-language generation uses the model directly without a host grammar forcing its choices.

## Numeric contract

The synthesized executor operates on CPU binary64. Program length is 1–48, registers are two unsigned views of i32 words, and memory is four unsigned i32 words. ADDI takes a signed i32 immediate. LOAD, STORE and branch operands are validated indices. HALT uses the canonical zero immediate in this witness. Initial operands are i32 bit patterns. The language controller's narrower interface admits value tokens 0–31 and repeat counts 1–9, with offset requiring exactly one. Executor status is sampled after 48 cycles, so a still-live invocation yields exhausted feedback.

For the corrected block under these invariants, the audit bounds each affine row by the absolute bias plus the sum of the maximum magnitudes of its terms. It uses the algebraic fact `ReLU(ReLU(x+1)−ReLU(x)) ∈ {0,1}` for integral x. The largest conservative affine bound is **148,176,371,709**, below `2^53`, and the attention score bound is smaller. Under exact binary64 integer operations within that range, this bounds each product and partial sum in the generated affine rows. The wrap expression adds or subtracts one modulus as needed, which suffices for an unsigned word plus a signed i32 immediate. Exact hardmax uses a unique positional maximum at an admitted address. The proof is conditional on the listed invariants and binary64 operations, with CPU execution checked here.

The numeric test also rebuilt and compared the coefficient tensors with the archived weight data. It ran 25 independent **46-step** traces using repeated overwrites, min/max signed immediates, high unsigned values, and `0xfffffffe + 3`. It checks the live flag for a loop exceeding 48 cycles. The existing synthesized-attention replay supplies the earlier regression and independent WASM mirror checks.

## The second call is the dependency

The first normal prompt emitted `call alpha 2 1 .`. The actual result was 6, which the model used in `call beta 6 1 .` to answer 9. Substituting **8** for the feedback from that same first execution changed the generated second call to `call beta 8 1 .`, producing 11. Across the four valid prompts the original answers were 9, 22, 14 and 19, and every substituted answer increased by 2. Disabling the first tool produced unavailable feedback and the model stopped without a second call. A separate prompt elicited a zero repeat count on the second call. The validator rejected it before execution and the model stopped on invalid feedback.

The trace separates tool choice, argument binding, circuit transitions, and the numeric feedback used by the next response. All executed tool traces match the independent scalar interpreter. The downloadable controlled-failure replay identifies event 2 as the first difference, while the circuit's actual result and trace still agree. Event 3 records the changed second call.

The evaluation archives raw load, image preparation, language decoding, and tool validation plus trace materialization times. The first inference includes lazy runtime overhead. Timings are observations for this small CPU witness. The separate Percepta and conventional causal-executor experiments retain their own measurement contracts.

## Open and verify

From `visuaml-client/voxel-replay` in the research environment:

```sh
export PYTHONPATH=.:../backend OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python tools/audit_attention_numeric.py
python tools/replay_two_call_controller.py
python tools/replay_two_call_controller.py --open-failure evidence/two-call-controller/failure.json
```

With Vite running, open [WASM Replay](http://localhost:4180/replay-viewer.html) and select **Open two-call failure**. The browser opens the saved observation. The CLI reruns the same prompt, program images, and intervention from the frozen artifacts.

[Download failure](failure.json) · [Holdout and hashes](holdout.json) · [Replay](replay.json) · [Numeric audit](numeric.json) · [Raw timings](timings.json)
