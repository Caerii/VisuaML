# Word-level causal decoder

The frozen decoder lowers the `i32.add` and `i32.eqz` operation declarations to a bit circuit. Together with LOAD, STORE, branch-if-nonzero, and HALT, it executes a bounded subset of the tensor assembly. The ABI admits up to 16 instructions, one 32-bit register, four 32-bit memory cells, and signed 32-bit ADD immediates. A zero result falls through the branch. The prior one-bit weights and structured executor remain available. This decoder was frozen at `91b40c0` before these programs were authored.

The scripted graph owns opcode fetch, carry evaluation, zero detection, memory lookup, and PC selection. The host admits the image, keeps the four layer caches, and appends each selected token. The compiled IR contains no Python callback or structured executor call. The EXEC position holds the completed transition fields in its value cache. Subsequent positions attend to it and emit one byte or marker each. The dense graph still evaluates its layers at each position; only EXEC publishes the computed successor.

## Bits, arithmetic, and commit

Twelve tokens carry EXEC, next PC, four little-endian register bytes, address or NONE, four little-endian store bytes, and COMMIT or HALT. Bytes occupy vocabulary IDs 0–255. Other positions carry typed control or marker symbols. Each COMMIT publishes the selected address and store bits to memory attention. An incomplete store has no matching address key or payload in the memory layer. Offline parsing independently reconstructs the full guest state.

The adder starts with per-bit generate and propagate. Five parallel-prefix stages compose adjacent carry functions at distances 1, 2, 4, 8, and 16. In each composition the generate and propagation cases are disjoint, so the sum used in the circuit is an exact Boolean OR. The carry into a bit comes from its lower prefix; the last carry is discarded. The zero test is `ReLU(1 - sum(bits))`. All admitted bit operations and sums are exactly representable in binary64. An independent integer oracle checks 2,112 pairs, including signed extremes, all-ones, zeros, and seeded random values.

Attention retains the 512-position envelope. Relative gathers use a quadratic score with margin at least 24. Matching code and memory keys gain 8,192 and relevance increases by 16 per position, so a required match leads any competitor by at least 16. Relevant score arithmetic is exact in binary64. In real arithmetic, total nonwinner softmax mass is bounded by `511*exp(-16)`. A weighted-bit error that keeps the result outside the 1/4–3/4 recovery interval preserves the bit. The CPU binary64 executions verify the tested kernels and contexts; the analytical margin is a sufficient conditional bound, not a proof about every implementation of softmax.

## Replay and intervention

- `0xfffffffe + 3` wraps to 1. It is stored, loaded, and selects PC 7.
- `0xffffffff + 1` wraps to 0. It is stored, loaded, and falls through to PC 5.
- The post-freeze countdown starting at 3 retires 13 instructions. Replacing a single byte in the first generated store commits 1 instead of 2. As predicted in the holdout manifest, the next addition produces 0 and the corresponding branch selects exit PC 4 instead of loop PC 0. The intervention halts after nine instructions.

The three normal traces agree at all 26 source boundaries with the scalar, structured, and prior vocabulary references. Full-prefix and incremental generation emit identical streams for each and the intervention. At all eleven incomplete-store cuts the previous word stays visible. A longer countdown from 8 fills the 512-token context with 32 retired instructions and eight incomplete tokens. Its committed state agrees with the scalar reference and the full-prefix logits predict its generated stream.

## Token and timing budget

Eight warm alternating samples on one CPU thread. Total includes generation and prefill. Load, admission, and audit are outside these measurements.

| Same source work | Steps / generated tokens | Cached prefill / decode | Cached total | Full-prefix | Structured | Prior vocabulary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wrap to one | 6 / 72 | 3.84 / 33.58 ms | 37.23 ms | 409.01 ms | 1.40 ms | 29.45 ms |
| Wrap to zero | 7 / 84 | 4.00 / 38.92 ms | 42.97 ms | 497.52 ms | 1.56 ms | 32.87 ms |
| Countdown | 13 / 156 | 2.45 / 72.93 ms | 75.43 ms | 795.15 ms | 2.72 ms | 61.02 ms |

The 156 countdown tokens break down into 64 live word bytes, 40 unused non-store byte slots, 26 PC/address fields, and 26 frame markers. Logical cache growth is 4,288 bytes per consumed position. The ordinary countdown uses 120 initial tokens and retains 275 positions at halt, or 1,179,200 logical KV bytes. The output HALT does not require a further forward pass. The context-limit probe retains 2,195,456 logical KV bytes.

From `visuaml-client/voxel-replay` with the existing research dependencies:

```sh
PYTHONPATH=.:../backend python tools/replay_word_decoder.py
npm run dev -- --port 4180
```

Open `/replay-viewer.html` and select **Open word decoder replay**. The viewer displays the saved evidence and supports download and reopen. [Token and source replay](replay.json) · [Raw timing samples](timings.json) · [Frozen IR](decoder.ir)
