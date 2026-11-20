# Percepta Transformer VM reproduction

Source pinned to `f13a48f90aa8655f1bcc18c0dcc584f5665702d1`. Built and ran the actual analytically generated transformer weights using the upstream Python runner with the standard KV cache, on one CPU thread. The authored WASM loop initializes word zero to 2, adds 3 four times through load/store and a data-dependent branch, emits byte 14, and halts. An independent Wasmtime run emitted byte 14 and confirmed word zero was 14.

The universal weights matched all 378 tokens of the upstream reference, including the 112-token prefix, across three invocations. The specialized model emitted `out(0e)` and halted. The strict comparison of its execution suffix against the universal reference finds a first raw difference at token index 1, `00'` versus `00`. The test does not discard primed carry tokens. Exact specialized trace equivalence remains an open comparison. The timings use different representations and trace boundaries from the local toy tool benchmark.

## Commands

Use a separate Python 3.12 environment and put the pinned upstream checkout on `PYTHONPATH`. This CPU run used PyTorch 2.10.0+cpu, NumPy 2.5.5, HiGHS 1.13.1, PuLP 3.3.0, PyYAML 6.0.3, ninja 1.13.2 and pybind11 3.1.0. PuLP 4.3.1 was incompatible with the upstream scheduler API. The last two solver pins match the upstream lock.

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m transformer_vm.build --save-weights=/tmp/percepta-model.bin
python -m transformer_vm.compilation.compile_wasm /path/to/loop.wasm -o /tmp/percepta-loop
python -m transformer_vm.wasm.reference /tmp/percepta-loop.txt
python /path/to/VisuaML/visuaml-client/voxel-replay/tools/verify_percepta_loop.py \
  /tmp/percepta-model.bin /tmp/percepta-loop
```

The verifier calls the upstream model runner programmatically. The stock CLI bootstrap attempts to generate the default C corpus and requires WASM-target Clang, which was unavailable here.

[Upstream commit](https://github.com/Percepta-Core/transformer-vm/tree/f13a48f90aa8655f1bcc18c0dcc584f5665702d1) · [Results and weights hash](results.json) · [WASM source](loop.wat)
