# Bounded causal tensor assembly

This experiment uses the five-opcode synthesized VisuaML tool ISA as a tensor assembly. An attended instruction selects HALT, LOAD, STORE, i32 ADD with wraparound, or branch on a nonzero register. The unchanged recurrent hard-attention executor and a new softmax causal decoder replay the same admitted programs. The causal decoder is a fixed tensor graph with a structured 160-lane record interface. A conventional vocabulary-token transformer is a distinct target.

## Reproduce

From `visuaml-client/voxel-replay` with the existing research Python dependencies installed:

```sh
PYTHONPATH=.:../backend python tools/causal_tensor_assembly.py --verify
```

The decoder source and artifact were committed at `8b5b2c2` before these holdouts were written. The saved replay records both artifact hashes, every source state, and each emitted write and successor PC. The holdout compares the softmax decoder with the original frozen recurrent artifact and its independently implemented scalar oracle at every completed source instruction. The five holdouts cover 73 source boundaries. A separate differential audit compares all three backends on 192 generated images and 2,262 additional source boundaries. Its deterministic corpus and results are in `tools/audit_causal_assembly.py` and `differential-audit.json`.

## Record ABI and recurrence

The prefix contains read-only program records, four initialized memory records, and one initial state record. Each decode consumes only that prefix and emits one new source record. The driver appends that record unchanged and respects fuel or the emitted halt flag. It never resolves an address, computes an instruction, or reconstructs a register during inference. The audit decoder reconstructs memory after the run.

Each record has one-hot key lanes for the program PC or memory address and bounded payload lanes for the instruction or write. The latest state record carries the next PC, two 32-bit registers, and the live flag. An emitted store is a committed write. A later load reads the newest matching record. Instruction fetch cannot match data writes because they have separate key lanes. Program validation ensures 1–16 instructions with an in-range static branch target and a final HALT. There are four memory words, a 64-record context, and no partial instructions, memory traps, or continuation beyond the context limit. Fuel limits source records appended. A full-context witness ends with 64 records.

Define decoded memory as the latest written value at each address, including initial memory records. Induct on the number of source records. At the base, the initial state and program rows decode to the reference state. For each inductive step:

- Instruction attention recovers the matching immutable program row for the graph-owned PC.
- LOAD reads the newest committed initialized word at the selected address, then changes only the selected register.
- STORE appends a matching memory address and a copy of the selected register. That record becomes its address's newest matching write.
- ADD uses 32 Boolean lanes and a five-stage generate/propagate prefix. For a bit with (P=a\mathbin{\oplus}b) and (G=a\land b), carry-out is (G\lor(P\land c)). A propagate and generate term are disjoint. Concatenating consecutive spans composes their carry functions, so five stages suffice for all 32 positions. The top carry is absent from the emitted word.
- The branch checks whether any bit in the selected register is set and emits either its target PC or the sequential PC. HALT holds the PC and clears the live flag. A fuel cutoff appends no later records.

For the attention selection, scores are (768\langle q,k\rangle+12t), where the query and matching key are one-hot and visible positions satisfy (0\le t\le63). A later match leads an older match by at least 12. Every match leads any nonmatch by at least (768-12\cdot63=12). Score computation uses small exact integer-valued binary64 operations for admitted records. With at most 63 competitors, the wrong-record softmax mass is bounded in real arithmetic by (63e^{-12}<0.000388). For bit payloads, the weighted result therefore falls near 0 or 1. The clipped ReLU ramp between (1/4) and (3/4) recovers the exact bit in real arithmetic. A total kernel error smaller than the distance to the two thresholds preserves the decoding bins. The replay tests CPU binary64 and asserts exact emitted bit lanes at each observed step. This argument does not bound rounding of every PyTorch softmax or matrix multiplication implementation. Different dtypes, kernels, or longer context need their own analysis.

## Holdouts

The paired overwrite fixtures commit two distinct values at address 1, then reload it and add 3. One reload gives `0xfffffffe`, wraps to 1, and takes the branch. The other gives `0xfffffffd`, wraps to 0, and falls through. Fuel cuts immediately before and after the second store distinguish the two possible published histories. A nine-iteration overwrite loop begins at `0xfffffffe`, revisits the same address, and reaches the 64-record limit after 46 retired instructions.

The architecture uses two dot-product softmax selections per source record, fixed bit circuits, and the last causal state record. The implementation evaluates one structured record per Python call and keeps an explicit history tensor. A classical decoder with position-wise standard blocks, ordinary vocabulary logits, and a KV cache must replace whole-record emission with token phases and account for intermediate visibility, normalization, logit margins, and generated tokens per source instruction.

## Cost of a structured replay

Each retired instruction emits one 160-lane float64 record, or 1,280
bytes. The decoder re-reads the visible history on each call and has no
KV cache. The 64-record witness holds 81,920 bytes of raw history and took
9.21 ms median over 12 warm CPU samples on one intra-op thread. The two
overwrite-and-branch paths took 1.76 and 2.16 ms median for nine and eleven
emitted records. These timings include prefix growth and exclude image
admission, model load, and offline comparison. They describe structured
record throughput.

```sh
PYTHONPATH=.:../backend python tools/audit_causal_assembly.py --benchmark
```

[Raw timing samples](timings.json)
