# Cached causal memory

The 512-position witness uses two layers of causal dot-product softmax attention with fixed projections and ReLU feed-forward gates. The decoder uses position features and token embeddings with no layer normalization. The host supplies ordinary tokens and retains each layer's K/V tensors. The sole attention mask is the triangular causal mask. The model is frozen in `6b2d133`, before the tape fixture was authored.

## Token and cache semantics

The vocabulary contains four address symbols (0–3), two bit symbols (4–5), COMMIT (6), READ (7), and a separator (8). A write consists of an address and 32 low-to-high payload bits followed by COMMIT. A READ immediately follows its address. The first layer retrieves fixed relative offsets through attention. At COMMIT it retrieves the address 33 positions back and all 32 intervening payload bits. A feed-forward gate gives a nonzero address key and payload value to the second attention layer only at COMMIT. The second cache therefore stores a computed address and word at the token position where the write committed. A READ query uses the address from its preceding token and the second attention layer selects the most recent matching commit.

The first-layer dot product is `-24*(j-(i-offset))²` up to a query-only constant. At valid gather positions the desired token has a margin of at least 24. In the second layer a matching commit gains 8,192 and recency gains 16 per position. With at most 512 positions a matching commit beats a nonmatch by at least 16 and beats an older matching commit by at least 16. Inputs are assumed to follow the grammar and each READ must have a prior matching commit; the hybrid initializes all addresses. The feed-forward bit recovery ramps between 1/4 and 3/4. The nonwinner mass bounds in real arithmetic are `511*exp(-24)` and `511*exp(-16)`; exact bit recovery holds if each layer's error in its weighted bit is less than the distance from its bound to 1/4. The integer score arithmetic is representable in binary64. CPU binary64 tests check the actual kernels at the observed positions, including the full 512-token context.

## Replay

Full-prefix evaluation, token-by-token cache append, and a prefetched cache with a multi-token suffix agree at the audited READs. The tape first reads `0xfffffffd`, then `0xfffffffe` despite a later uncommitted payload, and reads `0xa5a5a5a5` at the interleaved address. Replacing the second committed payload with 17 changes only its dependent read. The long-distance fixture recovers the only matching commit from near the start of a full context.

The hybrid runs the vocabulary holdout's taken and fallthrough instruction streams and replaces the structured latest-write lookup at LOAD with this primitive. The scripted graph produces READ and COMMIT tape tokens from live instruction and state data. The host passes the admitted image and fuel. The traces match the scalar, structured, and archived vocabulary replays at all 20 retired instructions. Instruction fetch, arithmetic, and branch control are still the structured transition. This hybrid groups token ingestion by memory operation, while the memory tape witness also verifies one-token incremental execution.

## Measurements

Eight alternating warm CPU samples on one thread, using the same nine retired source instructions:

| Execution | Median |
| --- | ---: |
| Structured reference | 1.89 ms |
| Structured execution with cached token memory | 10.75 ms |
| Original vocabulary decoder | 42.34 ms |
| Repeating the structured transition 21 times per source step on prebuilt histories | 32.67 ms |

The last row estimates the impact of recomputation with token embedding, reconstruction, and prefix appends removed together. The remaining difference also includes dispatch and output packing; it is not an isolated reconstruction cost.

| Prefill size | Prefill median | Append one token median | Logical cache bytes |
| ---: | ---: | ---: | ---: |
| 32 | 0.45 ms | 0.117 ms | 70,400 |
| 128 | 3.23 ms | 0.154 ms | 281,600 |
| 256 | 20.55 ms | 0.194 ms | 563,200 |
| 511 | 104.37 ms | 0.260 ms | 1,124,200 |

Logical K/V size grows by 2,200 bytes per token. At prefill, first-layer heads share broadcast views; the physical storage is 352 bytes per token before appending. This PyTorch implementation concatenates the cache on append, materializing those views. Prefill and append measurements include their respective cache construction. Timings exclude model load and trace auditing.

Run from `visuaml-client/voxel-replay` with the research Python environment:

```sh
PYTHONPATH=.:../backend python tools/replay_causal_kv_memory.py --open evidence/causal-kv-memory/replay.json
```

[Raw timings](timings.json) · [Replay tape and traces](replay.json) · [Frozen decoder IR](decoder.ir)
