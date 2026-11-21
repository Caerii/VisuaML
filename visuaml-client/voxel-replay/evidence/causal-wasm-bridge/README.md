# Rust WASM through the frozen causal word decoder

This experiment uses the existing sparse-checkpoint TorchScript artifact byte for byte. A static adapter validates and lowers two compiler-generated Rust guests into program tokens for that artifact. Runtime inputs determine loads, writes, branches, and halt. The host submits tokens, retains the KV cache, appends the decoder's selected tokens, and requests a model-generated checkpoint at a committed frame. On restart it validates framing and concatenates the immutable code prefix with the emitted checkpoint. The reference interpreter and the token auditor only check outcomes.

## Executable scope

The bridge uses the canonical single-function unary i32 WASM loader. It currently accepts a narrow profile of stack-balanced source groups with fixed aligned i32 memory addresses, wrapping addition by an immediate, block and loop routing, conditional branches, and a final return load. There are 16 word slots, of which one is a reserved immutable zero. There are at most 16 decoder code rows. The archived original passage helper uses two functions and locals and is rejected by this profile.

A source group can become more than one decoder operation. Control delimiters may emit no decoder operation. The replay checks the order and linear memory at completed source groups against the frozen WASM runner, with packed fallback blocked, and compares every decoder operation against a scalar tensor-assembly reference. Wasmtime and the frozen WASM runner also agree on returns and the complete 64 KiB page. This experiment's exactness scope is the admitted groups and observed runs. WASM fuel cuts inside grouped instructions, general locals and call frames, and arbitrary dynamic memory addresses are outside this profile.

## Passage and counterfactual

The input words are edit intent at offset 4, material at offset 8, and agent position at offset 12. With material 7, the no-edit run leaves the agent at 0. The edit run stores 0 at offset 8 and immediately checkpoints. The emitted sparse checkpoint omits that default-zero word. After restart, the next load reads 0, the branch falls through, and the guest writes position 1. Restoring just material 7 in the emitted checkpoint changes that resumed load and takes the other branch. The exporter also rejects a request immediately before the store's COMMIT. Full-prefix and cached decoding agree on both passage histories.

## Resource sweep

The countdown is a second Rust module using the same frozen artifact. It decrements one volatile word on each iteration. Each row is one measured execution. Extra live words remain untouched, and extra code rows are unreachable padding, so those axes hold retired work constant. The 256-token cap is a host policy within the same artifact's 512-token envelope.

| Countdown | Initial live words | Program rows | Context cap | WASM steps | Generated tokens | Restarts | Peak cached tokens |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1 | 7 | 512 | 41 | 264 | 0 | 371 |
| 32 | 1 | 7 | 512 | 293 | 1,944 | 5 | 480 |
| 128 | 1 | 7 | 512 | 1,157 | 7,704 | 20 | 480 |
| 32 | 3 | 7 | 512 | 293 | 1,944 | 5 | 456 |
| 32 | 7 | 7 | 512 | 293 | 1,944 | 8 | 408 |
| 32 | 1 | 11 | 512 | 293 | 1,944 | 5 | 480 |
| 32 | 1 | 16 | 512 | 293 | 1,944 | 7 | 480 |
| 32 | 1 | 7 | 256 | 293 | 1,944 | 16 | 228 |

Cached positions exclude the terminal HALT token, which needs no further model call. In the longest run, 642 decoder operations produced 7,704 tokens, about 6.66 generated tokens per retired WASM instruction. Peak KV tensor storage was 2,154,240 bytes. Raw per-segment wall times for prefill (including restart KV rebuild), decode, checkpoint emission, and framing are in `replay.json`. Those are descriptive CPU samples; they provide phase attribution without a latency distribution. More live state leaves less room for execution because the runner reserves capacity for the whole checkpoint. Larger code prefixes consume context at every restart.

## Reproduce

From `visuaml-client/voxel-replay` in the research Python environment with Torch and Wasmtime installed:

```sh
export PYTHONPATH=.:../backend
python tools/replay_causal_wasm_bridge.py
```

Use `--record` to regenerate timing and replay evidence. The default command rechecks the archived streams and state without requiring identical wall times. The committed modules were built with Rust 1.100.0 and the `wasm32-unknown-unknown` target using `tools/build_causal_wasm_passage.sh`.
