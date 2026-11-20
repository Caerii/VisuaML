# Vocabulary-token tensor assembly

This is a fixed byte-token lowering of the bounded softmax causal assembly decoder. The host loads an admitted program and initial state as tokens, calls the decoder once per token, appends the greedy choice, and stops at the requested token budget or the halt marker. It derives no guest PC, instruction, memory address, or output value. TorchScript inlines the unchanged structured transition alongside token embedding, framing, and vocabulary logits. The frozen graph predates the two new-address holdouts at `0617009`.

## Token and commit contract

The vocabulary has 258 symbols. IDs 0–255 are literal bytes, 256 commits a frame, and 257 commits a frame and stops. Each frame contains twenty packed bytes representing the 160 bit lanes, in little-endian bit order within each byte, then one marker. The initial prefix uses complete frames for immutable instructions, four zero-or-initialized memory words, and the initial state. Admission uses the same five-opcode ISA and address bounds as the structured reference.

At each forward pass the graph embeds only complete frames. It rejects a complete frame whose last token is a byte. From those committed records it recomputes the next record, selects the current byte by the token position, and produces vocabulary logits. While emitting a record the committed prefix stays the same. By induction its 20 bytes equal the structured successor, and the marker publishes that successor as one unit. A generated halt publishes its last frame before stopping. This proof assumes an admitted initial prefix, greedy decoding, the fixed format, and the structured transition's numeric assumptions.

The maximum final context is 1,344 tokens or 64 frames. The two attention searches still see at most 64 frame records. Their score margin is at least 12 on admitted records, so real-arithmetic nonwinner mass is bounded by 63 exp(-12). Bit recovery uses thresholds at 1/4 and 3/4. Byte projection is exact integer packing of recovered bits. The CPU binary64 tests establish exact agreement on the observed traces; other kernels and dtypes need numerical validation. This implementation is a specialized causal graph with attention over embedded byte groups. It leaves standard decoder-only transformer blocks and token-level KV attention as further lowering targets.

## Exact replay

The seven fixtures matched the scalar oracle, frozen recurrent executor, and structured causal backend at all 93 source boundaries. They cover repeated overwrites, wrapping through 0xffffffff, both branch outcomes, halt, fuel cuts around a store, and the context limit. The compared wire records also match the structured records lane for lane. No divergence occurred in these cases; comparison errors report the first mismatched source boundary and its emitted tokens.

In the first new fixture, emitted tokens 64–84 encode a store to word 2. The last token commits it. Each of the 20 partial lengths was tested with both the original and deliberately altered payload. Neither changed the committed prefix or the next prediction at that phase. A full frame with a missing commit marker was rejected. An offline intervention replacing only the committed payload with the prior write changed the next LOAD from `0xfffffffe` to `0xffffffff`. The bundle includes the emitted tokens for both loads.

## Cost at equal source work

On one CPU thread, the 46-instruction full-context run emitted 966 tokens. Across 12 alternating warm samples the token decoder took **219.86 ms median** or **4.78 ms per source instruction**. The structured backend on the same input took **9.27 ms median** or **0.20 ms per source instruction**. The previous independently measured structured reference was 9.21 ms. Both new timers include generation and prefix growth and exclude loading, image admission, and offline audit.

The token prefix occupies 10,752 bytes as int64 IDs at the limit and grows by 168 bytes per retired instruction. Persistent KV cache is zero bytes. Each call reconstructs the visible records and reruns the transition. The marked cost is a measurement of this lowering, including its recomputation for each byte.

From `visuaml-client/voxel-replay` with the existing Python research dependencies:

```sh
PYTHONPATH=.:../backend python tools/replay_vocabulary_tensor_assembly.py --open evidence/vocabulary-tensor-assembly/replay.json
PYTHONPATH=.:../backend python tools/replay_paired_routing.py
PYTHONPATH=.:../backend python tools/probe_routing_descriptions.py
npm run dev -- --port 4180
```

Open `/replay-viewer.html` and select **Open vocabulary-token replay** to inspect or download the saved evidence.
