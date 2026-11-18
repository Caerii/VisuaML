# v55: exact bit-sliced, finite neural operand storage

**Status:** isolated component prototype, not an integrated WASM stack executor.

## Architectural motivation

v54 makes the constructed event circuit authoritative over the control
frame action. v53 controls the dynamic frame depth and branch operand
truncation height. But the physical operand values remain a host Python
list. This is a real, independent neuralization boundary.

v55 experiments with a fixed-capacity representation of **24 exact
32-bit operands**, with a one-hot stack height over **25 legal levels**.
Each active slot stores a 32-component binary representation of its i32
value. Inactive slots must be all zero and are checked after each
transition. Fixed binary address matrices map one-hot height to the next
push position, previous live slot, and the mask for a truncation target.

## Construction

The standalone `neural_operand_storage_v55.py` component computes four
candidate states on each step: retain, push, pop and truncate. A one-hot
action input and fixed bilinear coefficients select the entire next
state. The top read and pop return use one-hot slot selection, not a
host Python index to choose the operand word. Truncation uses a
one-hot target height and fixed prefix mask. All gate states remain
binary exact float64 (0 or 1), without floating-value approximation
of i32 arithmetic.

The host currently **supplies** action tokens, immediates, pop
operations and truncation heights. The component is not a neural
decoder or a conventional Q/K/V transformer.

## Hard verification obligations

The dedicated component tests cover:

- Signed INT_MIN/INT_MAX and negative values through push/pop/top reads
- All 24 capacity levels, overflow/underflow and nonaliasing
- Truncation clears dead slots and never resurrects stale values
- Malformed action one-hot, inactive-slot corruption and fixed-weight
  ablation fail closed
- Retain is bytewise stable and produces no popped value

## Acceptance gate before claiming v55 executable integration

1. Add a new executor that uses this **actual state tensor** for every
   operand read/write instead of the old Python operand list.
2. Feed v54 constructed action outputs and v53 constructed restoration
   heights into the new storage path. Derive all stack operand effects
   from admitted program tokens, not hand-coded per-program patches.
3. Revalidate all six v54 canonical programs and original source Wasmtime
   binaries, comparing every instruction's PC, next PC, live operands,
   locals, control labels, halt and bounded fuel against independent
   v51/v52/v53/v54 implementations.
4. Falsify it by corrupting fixed slot gates, action weights, one-hot
   height and live frame restoration. No silent host fallback.
5. Produce an independent source-bound v55 evidence archive and origin
   record with the exact commit/tree, WASM source hash, weights and
   per-instruction trace digests. Preserve v50–v54 archives unchanged.

**No universal WASM, memory semantics, arbitrary learned-weight
decompilation, or full neural virtual computer is claimed.** These
need separate verified boundaries and full stack integration.

## Experimental executable integration

The additional `neural_operand_adapter_v55.py` exposes only top-of-stack,
push, pop, tail truncation, length and trace iteration. It owns exactly
one `OperandState`, not an auxiliary host list of i32 values. Each physical
push/pop/truncation delegates to `step_operands` and its fixed-weight
one-hot selector; top reads are matrix-selected from the tensor.

The separate `neural_operand_executor_v55.py` takes the v54 program
counter, neural frame-event, and neural frame-transition circuit, replacing
the Python operand list with that tensor adapter. Source instruction
dispatch, program metadata, locals and frame lexical validation remain
host-side. The explicit v55 integration tests compare per-instruction
state with the independently verified v54 executor and v51 interpreter,
and original source WASM with independent Wasmtime.

**Source-head verification must pass before claiming implementation
complete.** This is a restricted constructed neural operand *storage*
mechanism and not yet a fully neural WASM decoder or unrestricted memory
subsystem. Host opcodes still determine which operand operation to request.
The next gate must replace host operand-operation decoding with
constructed opcode/data selectors, with provenance-bound execution archives.
