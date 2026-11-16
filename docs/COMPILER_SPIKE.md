# VisuaML compiler spike — executable technical slice

This deliberately small and opt-in prototype sits next to VisuaML's existing FX export and visualization paths. It does **not** replace the production exporter or claim universal program compilation.

## Demonstrated

1. Trace supported `torch.nn.Module` tensor computations through `torch.fx` and `ShapeProp` into a typed, JSON-serializable `ProgramIR` (operator, sources, tensor shapes/dtypes/bytes, approximate FLOPs).
2. Decompose concrete `nn.Linear` calls to `get_attr(weight)` → `transpose` → `matmul` → `add(bias)` inside the executable FX graph. Outputs are compared numerically, not considered formally proved.
3. Re-fuse matched `add(matmul(x, transpose(weight)), bias)` patterns back into executable `torch.nn.functional.linear` calls, with strict structural conditions and numerical comparison.
4. Produce an explicit native / fallback / unsupported operator placement plan.
5. Execute supported models on torch and on an independent NumPy interpreter, then compare outputs.
6. Run a strictly labeled **illustrative** analog linear surrogate (quantization plus seeded noise, with digital fallback). It is *not* a model of Ayo hardware, and does not estimate energy, throughput, optical gain or propagation.
7. Project the typed IR and backend assignments back into VisuaML's existing `{nodes, edges}` canvas format, without conflating visual graph layout with executable program semantics.
8. Demonstrate a neural-assemblies-inspired sparse top-k winner-take-all step as an algorithmic workload. This does not automatically import the separate `neural-assemblies` library or validate its research results.

## Run

From the VisuaML repository root, after installing `torch`, `numpy`, and `pytest`:

```bash
PYTHONPATH=visuaml-client/backend python -m pytest -q visuaml-client/backend/tests/test_compiler_spike.py
PYTHONPATH=visuaml-client/backend python visuaml-client/backend/examples/compiler_spike_demo.py
```

On Windows PowerShell, use `$env:PYTHONPATH='visuaml-client/backend'` before the commands.

## Known limitations and next steps

- FX handles a finite subset of Python/Torch control flow. Add `torch.export` ingestion, ONNX import, symbolic guards, and nested structured output handling before calling it a general compiler.
- Operation semantics must be versioned and covered by reference tests. Unsupported operators fail closed; kernels with different numerical semantics require explicit tolerances.
- FLOP and byte counters are approximate. The output-byte count is **not** a measure of total memory traffic or peak allocation. No benchmark or performance guarantee is made.
- The optical surrogate is a teaching/demonstration layer, not physical validation. A real backend requires measured device transfer functions, noise models, control/ADC/DAC overhead and calibrated costs.
- `nn.Linear` decomposition is a graph rewrite, not a reduction of FLOPs or a proof of benefit. Matmul packing/fusion and mixed precision require subsequent passes and benchmarks.
- The `to_visuaml_graph()` adapter produces the existing canvas JSON shape, including placement annotations. **It is not yet connected to the Fastify API or displayed in the browser**; next integration point is a read-only compilation endpoint plus frontend inspector with diagnostic overlays.
- To ingest the **actual** `neural-assemblies` implementation, add an adapter that recognizes its chosen tensorized operations and compare a registered test corpus to the assembly reference engine.
- Backend plugins must declare supported operators, numerical requirements, data layouts, and a `run` implementation. Here, the custom-backend interface is minimal proof of extensibility, **not** a portable hardware compiler SDK.
