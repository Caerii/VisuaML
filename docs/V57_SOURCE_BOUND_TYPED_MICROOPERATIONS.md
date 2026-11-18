# v57 — Source-bound typed microoperations and offline replay

**Status:** implemented on research branch, **not independently executed in
this environment**. GitHub Actions have been disabled at the owner's request.
No numerical replay, passing test count, or universal ISA coverage is claimed
until the offline procedure below is run and its evidence inspected.

## Why a single microoperation representation compounds

A machine does not become a general computer merely by accumulating
individually constructed opcode gates. It needs a reusable mapping from
program data into all of the state transitions those gates implement.

Versions v49–v56 progressively made exact i32 arithmetic, a one-hot
program counter, structured control-frame state, frame events, operand
storage, and a six-operation ALU participate in actual execution.
However, each of those components had independently authored token tables
and the v56 host still supplied instruction immediates, local indices,
and operand consumption arity by directly interpreting Python opcode names.

v57 begins converging those contracts into one fixed-weight typed
microoperation IR. The program supplies the token bytes and source-bound
operand data; a constructed decoder with the same coefficient matrices
for every admitted program reads the live one-hot PC and emits:

- One-hot opcode class and stack-effect class
- One-hot local mode (none/get/set/tee)
- ALU operation selector compatible with v56's six candidate circuits
- Exact bit-sliced i32 immediate
- Local register selector
- Lexical branch-depth selector

Both canonical source-WASM SHA-256 and typed source-program SHA-256 bind
the admitted token table. Any changed opcode, immediate, local selector
or branch depth is rejected at execution admission, even if its tensor
shape remains valid.

## Live execution integration

`neural_microop_executor_v57.py` is a separate source-bound executor
that retains the v52 neural PC, v53 constructed frame operations, v54
event selection, v55 tensor operand bank and v56 six-way ALU.

Compared with v56, it adds concrete state-authority transfers:

1. **Instruction immediates** are fetched through one-hot PC × canonical
   immediate-token table, rather than reconstructed from the host opcode.
2. **ALU selection** is read from the unified fixed microoperation matrix,
   rather than a second per-program opcode-to-ALU token tuple.
3. **Local register addressing** uses fixed one-hot weighted reads/writes
   over a [5,32] register bank, not Python indexing by the instruction
   immediate.
4. **Local operation mode** selects the actual bank transition:
   the get/set/tee branch provides stack operands, but the circuit's
   one-hot mode decides whether the bank is modified and whether a value
   is returned.
5. **Operand consumption and output materialization** are selected from
   the constructed stack-effect token, not the host's computed pop count.
   Host source semantics remain a fail-closed *validator* for a corrupted
   decoder, never a fallback execution path.
6. **Branch depth** comes from PC-selected one-hot source program data
   before the v53 frame-branch circuit applies its stack restoration.

All admitted effects still operate within the strict v51 canonical
WASM subset: one exported `run(i32)->i32`, homogeneous i32 locals,
empty block signatures, bounded frame/operand capacity and fuel.

## Independent validation — run locally, no Actions

From a **clean, checked-out v57 branch** with the existing backend
dependencies and `pytest` and `wasmtime` installed:

```bash
python -m pip install pytest wasmtime
python visuaml-client/backend/scripts/verify_offline_v57.py \
  --output /tmp/visuaml-v57-evidence.json --full
```

On Windows, choose an unused absolute output path. The entrypoint runs
all backend Python tests with the backend on `PYTHONPATH`, then the
source-bound evidence generator. It does not dispatch GitHub workflows
or make remote requests. The evidence generator refuses to run from a
dirty tracked/untracked Git tree and refuses to overwrite a prior JSON.

The deterministic evidence corpus contains three control-flow cases
(countdown, conditional, Euclid GCD) and 16 generated arithmetic/local
programs. Its finite input inventory contains **138 planned executions**:
10 + 5 + 11 + 16×7. These are planned witnesses, not passed results.

For each execution, the runner validates the **original canonical WASM**
in an independently installed Wasmtime, compares its result to v51 and
v56, checks every executed state field, and records source bytes, trace,
trace SHA-256, source/program digests, fixed-weight digests, environment
versions, Git commit/tree and fuel policy. The report is written only
after all comparisons complete.

A new immutable source-bound research artifact can be committed under
`benchmarks/` after review. Do not edit v50–v56 original archives.

## Architectural limits that must remain visible

The host still fetches and identifies many opcode classes, applies
structural validation, manages lexical metadata, selects some physical
stack effects and local stack operands, manages fuel, and hosts the
NumPy implementation of the constructed gates. The v57 backend is not
a stock softmax Q/K/V transformer, not learned from examples, not a
proof of arbitrary unknown-weight decompilation, and not unrestricted
WebAssembly compatibility.

The construction evaluates every candidate ALU at an arithmetic step,
so its theoretical execution work is greater than conventional host
opcode selection. An optimization must preserve fixed-weight execution
semantics and the same verified source-bound witness contract.

## The next highly compounding gates

- **v58 — unified tensor microstep:** remove the remaining host
  instruction-specific stack-effect and source metadata dispatch, using
  fixed token-class masks and a fully canonical machine-state tensor.
- **v59 — bounded linear memory:** explicit i32 address arithmetic,
  page bounds, aligned/unaligned accesses, loads/stores and traps;
  validate actual binary memory sections and independent Wasmtime.
- **v60 — function calls:** direct calls, nested call frames, returns
  and result arity, followed separately by tables/indirect calls.
- **Conformance program:** select a pinned WASM specification and
  publish an exact feature ledger with test-subset counts, unsupported
  families, negative cases and provenance. Never invent an overall
  percentage of universal coverage.
