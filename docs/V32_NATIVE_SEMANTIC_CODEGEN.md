# v32 — First graph-generated native Semantic IR pipeline

*The critical distinction is between executing a fixed decoder and compiling the operators of an admitted graph.*

## Achievement and scope

The existing v28–v31 native C++ decoder implements a **fixed Llama mathematical program**, with scalar, four-row, AVX2, shape-gated prefill and optional online-attention variants. The v32 experiment introduces a new, separate compiler, `native_semantic_codegen_v32.py`, which reads the established `SemanticProgram` envelope, validates every operation and its concrete tensor geometry, **generates C++17 from the graph itself**, compiles locally with a trusted installed C++ compiler, and runs it through a checked ctypes ABI.

The first closed-world dialect supports exactly four pure float32 operations:

| Operator | Accepted semantics |
| --- | --- |
| `matmul` | Rank-2 matrix multiplication, no broadcasting |
| `add` | Identical-shape elementwise addition, no broadcasting |
| `relu` | Shape-preserving elementwise ReLU |
| `conv2d` | NCHW input × OIHW kernel, zero padding, bounded integer stride/padding, no dilation/groups/bias |

The admitted program has one region, one output, concrete positive dimensions, at most 16 inputs and 64 SSA operations. Tensor elements, retained source/intermediate bytes and arithmetic work have independent hard limits. **No state effects, symbolic shapes, mutable aliases, untrusted uploads, arbitrary native source or Python code are accepted**. A program is rejected before compilation if its SSA definitions, geometry, effect annotations or resource budgets violate these rules.

The producer is the existing typed `SemanticProgram`, not a novel unrelated runtime representation. Generated C++ identifiers use internal integer slots; original graph names and free-form operation strings never become executable C++ tokens.

## Two structurally different numerical witnesses

The first tests generate native code from two **synthetic graph fixtures**, both compiled by the same emitter:

1. **Residual MLP:** `matmul → relu → matmul → add`. Dynamic input and parameter buffers are explicit, rank-2 float32 inputs.
2. **Residual convolutional network:** `conv2d → relu → add` in NCHW/OIHW geometry.

Each native result is independently compared against a NumPy computation and eager PyTorch; the Conv2D NumPy oracle is an intentionally separate scalar implementation. The tests include multiple random seeds, noncontiguous inputs copied explicitly to the ABI's contiguous representation, varied values, input immutability and negative cases.

**Important boundary:** These are two structurally distinct *synthetic computation graphs*, **not two independently admitted pretrained model families**. This completes only the initial native code-generation vertical slice of [v32 issue #12](https://github.com/Caerii/VisuaML/issues/12). Importing actual checkpoint-bound FX graphs, preserving parameter/constant provenance, running two named pretrained architectures and robust graph-level effect lowering remain future acceptance gates.

## Trusted PyTorch FX-to-native admission

The second v32 integration layer is the narrowly scoped `native_fx_frontend_v32.py`. For a trusted, local, eval-mode `torch.nn.Module` and a finite CPU float32 example tensor, it uses `torch.fx.symbolic_trace` and `ShapeProp` to derive **real module graph structure and concrete tensor metadata**. It accepts only bias-free `nn.Linear`, bias-free/ungrouped/undilated zero-padded `nn.Conv2d`, non-inplace `nn.ReLU`, and pure residual `operator.add`. Unknown operations, module effects, unsupported convolution attributes, multiple outputs, dynamic/symbolic shapes and dtype mismatches fail admission.

Linear weights are explicitly transposed into the rank-2 `matmul` contract, while Conv2D weights use OIHW layout; each is a separate named runtime input. The importer records SHA-256 fingerprints of the weight snapshots in the source provenance and exposes `binding.validated_inputs()` to reject later mutations before execution. The native emitter remains **exactly the same one** used for hand-declared typed fixtures—there is no MLP or CNN template hidden inside native code generation.

```python
# For a trusted local model only: tracing executes its Python forward.
binding = lower_trusted_fx(model.eval(), example, trusted=True)
compiler = NativeSemanticCompiler(binding.program)
output, receipt = compiler.run_with_receipt(**binding.validated_inputs())
```

`trusted=True` is explicit acknowledgement, **not a sandbox**. `symbolic_trace` and `ShapeProp` execute the trusted module; never pass untrusted uploaded Python or unknown model-hub code. Shape propagation uses a detached, contiguous **clone** to avoid mutating the caller's original example tensor even when rejecting an inplace operation. Only the bound snapshot hashes are certified; a generic native compilation accepts independent runtime tensors, and per-run receipts describe their actual bytes.

These are actual PyTorch FX graphs with random/synthetic weights, separately verified against eager PyTorch and independent NumPy. They still do **not** constitute two pretrained architecture-family certifications.

## Provenance and ABI

The generated native symbol is `visuaml_semantic_run(const float* const* inputs, float* output)`, compiled as a temporary process-owned shared library with `-O2 -fno-fast-math`. The Python wrapper checks exact input names, float32 dtype, shapes, finite values and contiguity before calling. It allocates a private output array, and the C++ code copies the output only after validating every intermediate for finiteness. A failure raises a Python error rather than returning partially published output.

The compiler report and per-run receipt bind the semantic IR digest, original source declaration hash, generated C++ SHA-256, binary SHA-256, compiler identity and flags, input/output geometry, input tensor byte hashes, output byte hash, compilation milliseconds and bounded operation/byte budgets. The weights are runtime inputs, not silently embedded in source, so input hashes must be recorded **per execution**.

The trusted ctypes ABI is **not a sandbox for arbitrary hostile native callers**. A Python caller must provide the exact checked tensor lengths; native pointers alone cannot prove that an external caller has allocated sufficient backing memory. There is no portable signed AOT, independent ABI manifest loader, formal equivalence proof or GPU/optical backend in this slice.

## Reproduce

With the pinned Python dependencies and trusted C++17 compiler installed:

```bash
cd visuaml-client
python -m pytest -q backend/tests/test_native_semantic_codegen_v32.py backend/tests/test_native_fx_frontend_v32.py
python -m pytest -q backend/tests
PYTHONPATH=backend python backend/examples/native_semantic_codegen_demo_v32.py --trials 9 --output /tmp/native-semantic-v32.json
PYTHONPATH=backend python backend/examples/native_fx_codegen_demo_v32.py --seed 32 --output /tmp/native-fx-v32.json
```

The [dedicated GitHub workflow](../.github/workflows/native-semantic-v32.yml) runs the independent fixture checks and publishes a time-bounded, source-bound benchmark artifact. The usual full repository CI remains mandatory. No warm-latency speedup is claimed: this prototype prioritizes semantic admission and inspectable native generation over optimization.

## Independently inspectable evidence integrity

The dedicated CI workflow publishes full generated C++ source for each manual/FX fixture, its SHA-256, the native library hash and exact source-branch commit. A separate [evidence integrity checker](../visuaml-client/backend/scripts/verify_native_semantic_evidence_v32.py) verifies the checked-out source-file fingerprints, generated source SHA, declared runtime tensor fingerprints, finite numerical witness tolerances, and the equivalence of generated C++ source/binary for the same graph geometry whether it originated from an explicit Semantic IR fixture or the trusted FX frontend. The artifacts contain a third integrity summary JSON.

```bash
python backend/scripts/verify_native_semantic_evidence_v32.py \
  --declared /tmp/native-semantic-v32.json \
  --fx /tmp/native-fx-v32.json \
  --source-commit YOUR_EXACT_GIT_COMMIT
```

The verifier is deliberately an **integrity and consistency check**, not cryptographic authentication: an attacker who controls the source and all hashes can rehash them. It is not an independent numerical rerun or proof of correctness for arbitrary inputs. Reproduce the two model runs on a trusted local machine and compare their receipts before accepting broader claims.

## Next substantial work

Expand the first restricted trusted FX frontend to carry richer operator contracts and externally auditable parameter/constant provenance; independently admit a second **pretrained** model architecture; generate a versioned offline ABI manifest and load it in a separate process; introduce explicit liveness-aware scratch allocation, input/output alias checks, better numerical error instrumentation and standardized performance suites.

The public roadmap's acceptance percentages remain unchanged until these criteria pass review. Website deployment at `visuaml.com/roadmap` is separately tracked in [issue #8](https://github.com/Caerii/VisuaML/issues/8).
