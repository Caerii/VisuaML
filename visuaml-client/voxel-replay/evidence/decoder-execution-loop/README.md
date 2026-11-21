# Decoder-only execution loop

A fixed causal decoder fetches and executes a bounded subset of the tensor assembly. The same artifact handles both control-flow outcomes. The host supplies the validated program image as tokens, appends each greedy vocabulary token, retains the four layer caches, and honors the terminal token or token budget. Source-state reconstruction is offline. The compiled graph contains no Python callback or call to the structured transition. Decoder weights were frozen at `998a851` before the new program was authored.

## Instruction and token contract

This profile admits up to 16 instructions, one one-bit register, four one-bit memory cells, and LOAD, STORE, branch-if-nonzero, and HALT. The full i32 arithmetic and two-register profile remain in the structured reference. The admission adapter writes program and initial state into six-token frames. Token IDs 0–15 encode PCs and addresses, 16–20 encode opcodes, 21–22 encode bit values, 23 means no store, 24 starts an execution frame, 25 marks a code frame, 26 commits, 27 halts, and 28 pads initial frames.

Each emitted instruction takes six tokens, in this order.

```
EXEC  next-PC  register  store-address-or-none  store-bit  COMMIT-or-HALT
```

Four causal attention layers specialize the operations already used by the cached memory witness.

1. Relative-position heads gather tokens from the five preceding slots.
2. A CODE position publishes its instruction key and fields. At EXEC the preceding committed PC queries the instruction cache.
3. A COMMIT position publishes a memory address key and payload. At EXEC the fetched operand queries the most recent matching write. Feed-forward gates select the register, write fields, and successor PC from the opcode, current state, and loaded bit.
4. The EXEC position publishes those computed fields in that layer's value cache. The remaining positions retrieve the fields from that EXEC position and select the next vocabulary symbol using fixed phase projections.

The instruction's live inputs are present at EXEC. Only EXEC publishes the computed successor fields. The fixed layers still evaluate their dense operators at every token, with a gate discarding completion values at other positions. This shares the instruction result across its output tokens. The hand-wired block uses dot-product softmax and ReLU gates, with explicit positional features and no layer normalization. Its only attention mask is the triangular causal mask.

## Correctness envelope

The decoder accepts up to 512 positions. Exact-offset attention has a minimum logit margin of 24 at relevant fields. For code and memory, matching one-hot keys receive 8,192, while recency increases scores by 16 per position. At most 511 older positions can compete, so a match beats a nonmatch by at least 16 and the newest matching commit beats any older match by at least 16. Address initialization guarantees a matching memory cell. The score arithmetic uses exactly representable integers in binary64 under the admitted domain. For bit recovery, real-arithmetic nonwinner mass is at most `511*exp(-16)` for address matching and `511*exp(-24)` for positional heads. The ReLU recovery thresholds are 1/4 and 3/4. Each weighted result must stay within that margin after kernel rounding. CPU binary64 replay checks the observed traces, including the full context, and does not establish a kernel-independent error bound.

At a source boundary, the most recent committed frame represents PC and register and the latest address-bearing COMMIT represents each memory word. The initialized image establishes that relation. At EXEC, fetch and memory attention recover exactly the operands of the reference transition, and the gates produce its successor. The fourth cache carries those results until the frame commits. Every earlier position in the frame has a zero memory address key and payload, so cutting the budget before COMMIT leaves the old word globally visible. HALT ends the trace.

## Replay and intervention

The two holdouts agree with the scalar, structured, previous vocabulary, and cached LOAD hybrid references at 15 total source boundaries. Independent full-prefix generation and incremental decoding emit identical streams on both branches and the intervention. Five cuts verify all partial lengths of the second generated store.

In the taken run, token 23 contains bit 1 and token 24 commits it to address 2. The later LOAD recovers 1, the branch selects PC 8, and the next instruction halts. The intervention replaces only token 23 before it enters the cache. The same generated COMMIT publishes 0. The later LOAD becomes 0, the branch chooses PC 6, and the next instruction emits a store to address 3. These expected consequences are in the holdout manifest.

The 84-token initial image emits 42 or 48 tokens before halt. The final cache retains 125 or 131 positions, excluding the terminal HALT output that no subsequent forward consumes. Logical KV storage is 1,800 bytes per consumed position, or 225,000 and 235,800 bytes respectively. The separate looping program saturates the full 512-position context with 77 complete source steps and two partial tokens. Its scalar trace matches and a full-prefix forward predicts every cached output token on the same tape.

## Cost

Eight alternating warm CPU samples on one intra-op thread. Timers include generation and exclude model loading, image admission, and trace auditing.

| Equal source work | Taken, 7 instructions | Fallthrough, 8 instructions |
| --- | ---: | ---: |
| New decoder with cache | 13.12 ms | 15.16 ms |
| New decoder, full-prefix generation | 43.12 ms | 51.32 ms |
| Prior vocabulary decoder | 32.29 ms | 37.12 ms |
| Prior cached LOAD hybrid | 12.74 ms | 15.06 ms |
| Structured reference | 1.55 ms | 1.78 ms |

Cached prefill medians were 0.77 and 0.75 ms. Cached decode medians after prefill were 12.29 and 14.35 ms.

From `visuaml-client/voxel-replay` with the existing Python research dependencies:

```sh
PYTHONPATH=.:../backend python tools/replay_decoder_execution_loop.py --open evidence/decoder-execution-loop/replay.json
npm run dev -- --port 4180
```

Open `/replay-viewer.html` and select **Open decoder execution replay** for the saved causal chain. [Replay](replay.json) · [Raw samples](timings.json) · [Frozen IR](decoder.ir)
