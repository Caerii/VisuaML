# Inspect a model export or a WASM guest

The examples use the installed `visuaml-replay` CLI. From
`visuaml-client/voxel-replay`, the equivalent source-tree command is
`PYTHONPATH=../backend python tools/visuaml_replay.py`.

## Follow an activation mismatch

```sh
python release/examples/imports/make_exports.py /tmp/visuaml-exports
visuaml-replay describe /tmp/visuaml-exports/reference.pt2
visuaml-replay compare-exports /tmp/visuaml-exports/reference.pt2 \
  /tmp/visuaml-exports/candidate.pt2 /tmp/visuaml-exports/input.npy
```

Use trusted PyTorch exports. The example reuses one weight and changes one
element of an attention mask. The first recorded value difference is the mask
at `[1, 1]`, from `-2` to `0`. The first activation difference is
`aten.add.Tensor` at that index, from `3` to `5`.
[Saved comparison](../examples/imports/export-comparison.json)

The comparison uses PyTorch and matching graph structure, flat tensor
arguments, and one tensor output. It compares the observed tensor nodes,
including bound buffers, in graph order. Use explicit `--rtol` and `--atol`
for floating-point comparisons when exact equality is inappropriate. Inspect
each graph separately if their structures differ.

## Inspect ONNX metadata

Install the optional extra from the repository root with
`python -m pip install '.[onnx]'`. Then run `visuaml-replay describe MODEL.onnx`
to see opsets, operators, tensor ports, symbolic dimensions, and references
to external initializers. Custom operators appear as graph nodes. This path
reads graph metadata without loading external files. The report labels
execution and validation as unavailable on this inspection path.

## Bring an unchanged Rust WASM binary

```sh
visuaml-replay describe release/examples/imports/static-material.wasm
visuaml-replay open release/examples/imports/static-material-report.json
visuaml-replay capture release/examples/imports/static-material.wasm \
  checkpoint.json --argument 35 --after-step 2
```

The Rust binary reads static material `7` from byte 1,024 and adds its argument.
The captured state holds the loaded `7` on its operand stack. The remainder
returns `42` for input `35` and capture compares the full final page against
Wasmtime. Omitting `--memory` uses the module's initialized memory. Passing
`--memory` still replaces the entire page with the supplied image.

The example was compiled with Rust 1.100.0. The following command preserves
linker metadata and produces the archived binary when run with that toolchain.

```sh
rustc release/examples/imports/static-material.rs --crate-name data \
  --crate-type=cdylib --target=wasm32-unknown-unknown \
  -C opt-level=z -C panic=abort -C link-arg=-zstack-size=0 \
  -C link-arg=--global-base=1024 -C link-arg=--initial-memory=65536 \
  -C link-arg=--max-memory=65536 -o /tmp/data.wasm
```

| WASM capability | Direct-call replay profile |
| --- | --- |
| Canonical unary i32 code and bounded direct calls | Supported |
| Named custom sections | Preserved in the binary and listed as opaque metadata |
| Padded memory offsets from a validated linker output | Supported |
| Active memory-zero data with an i32 constant offset | Initialized in segment order during fresh invocation |
| Passive data, imported functions, globals, tables, memory growth, floating point | Outside this replay profile |

`describe` separates Wasmtime validation from replay admission. A valid WASM
file outside the profile gets an admission reason. The frozen execution
artifact is identical to the earlier release. Active data initialization is
part of loading the program, and restored checkpoints supply their own pages.
