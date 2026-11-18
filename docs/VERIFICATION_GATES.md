# Research verification gates

These gates apply to the offline VisuaML research workflow. The
[neural WASM research gates](NEURAL_WASM_RESEARCH_GATES.md) describe the
individual machine layers. The [execution ledger](RESEARCH_EXECUTION_LEDGER.md)
records observed outcomes by source revision.

| Gate | Evidence required |
| --- | --- |
| Source identity | Clean producing checkout, full Git commit and tree, original WASM bytes and hashes, canonical token identity, fixed-weight digests, and runtime versions. |
| Execution parity | Validate and execute the original bytes in independent Wasmtime. Compare complete recorded state transitions against the independent v51 interpreter and applicable earlier engines. Include resource and boundary cases. |
| Circuit authority | Identify which gate outputs update live state. Inject faults into the enabled path and require either a detected divergence or fail-closed rejection. Record remaining host control and each engine's admitted subset. |
| Recovery | Preserve the blinded I/O boundary, repeat source-versus-candidate Z3 checks and finite neural/Wasmtime replays independently, and account for every success, rejection, timeout, and invalid attempt. |
| Performance | Measure paired workloads with explicit engine coverage. Record wall and CPU time, repeats, dispersion, worst regressions, representation costs, and the scope of memory measurements. Keep the reference implementation available and validate parity before accepting an optimization. |
| Handoff | Preserve historical archives, record hashes for newly generated artifacts, and make the producing source and replay commands available for review. |

Mark work **verified**, **failing**, **blocked**, or **untested**. A result
applies to the recorded source, input domain, and evidence. Solver results
establish typed source-versus-candidate equivalence for the admitted
symbolic fragment. Neural execution has a separate finite witness scope.
