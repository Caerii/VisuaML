# A frozen recurrent WASM machine

One CPU TorchScript artifact accepts the validated program image and current VM
state as tensors. Each invocation fetches the source instruction by PC, computes
operands and the next PC, updates the stack and locals, and admits the memory
write, fuel decrement, halt or trap inside the graph. Python validates WASM,
resolves lexical routing and branch restore heights, initializes the state,
drives recurrence and records traces. The host supplies the validated image
unchanged throughout execution.

[The frozen artifact](machine.pt) and its [IR](machine.ir) were committed in
`29d55e7` before the [second witness](after_freeze.wat) was authored. The artifact's
SHA-256 is `9e9bb7c459760b780cc2000400b8bb1155d4bb9778a6c807db6f2e87579b3422`.
The [replay record](replay.json) lists the script's exported operators, guest
binary and image hashes, source steps and page digests.

Instruction fetch uses exact integer `index_select`. Opcode equality produces
one-hot gates, and the graph selects from integer arithmetic and comparison
candidates. Addressing uses exact `gather` and functional `scatter`, with four
little-endian byte lanes. Operands use unsigned i32 bit patterns held in int64
tensors, with signed conversion for comparisons and masking for wrapping
arithmetic. The scope is the project's admitted structured single-function
i32 dialect and one 64 KiB page. The graph uses explicit tensor operators and
constant buffers frozen into the IR. There is no training stage, softmax
attention or call-frame support.

## Verification

The eight traced cases compare PCs, operand stack, locals, memory effects and
the completed prefix with the checked VisuaML NumPy executor. Its packed
fallback is blocked. Each final page is compared byte for byte, with page
digests checked at every completed step. Wasmtime additionally checks completed
values and pages and the memory trap. The fuel case uses VisuaML's source-step
fuel semantics. The 31-step rollback run restores a prior write. Fuel stops it
after nine completed instructions, with the intermediate write still present.
The straddling store traps without altering the page. Every terminal state is
checked for idempotence. Both branches of the later-authored binary and both
the open and blocked paths of the unchanged Rust voxel guest are covered.

Full BFS and one-blank Sudoku are checked against Wasmtime for return value and
full page, using the recorded source-step budgets of 41,591 and 893. The same
artifact also replays the shipped terrain histories, using their NumPy source
step counts and page digests and comparing every completed page against
Wasmtime.

## Indexed memory cost

Single-thread CPU, PyTorch 2.9.1. The medians below come from 31 alternating
samples per operation with 64 four-byte accesses per sample. These measure
the isolated Torch primitives, with indices and values prepared, and include
the Python-to-Torch call. Only the 64 KiB size is an admitted guest page.

| Address space | Gather | Functional scatter |
| --- | ---: | ---: |
| 4 KiB | 1.26 µs | 1.90 µs |
| 64 KiB | 1.26 µs | 3.05 µs |
| 1 MiB | 1.27 µs | 26.69 µs |
| 16 MiB | 1.50 µs | 703.12 µs |

Gather reads four indexed bytes. Functional scatter produces a fresh full
page even for a four-byte write. Copying that page dominates writes as it
grows. The [raw samples](memory.json) are for one page per access.

## Reproduce

From `visuaml-client/voxel-replay` with CPU PyTorch 2.9.1, Wasmtime and the
VisuaML backend dependencies installed:

```sh
export PYTHONPATH=.:../backend OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python tools/replay_frozen_recurrent.py > replay.json
python tools/bench_recurrent_memory.py > memory.json
npm test
```

Use `npm ci` followed by `npm run dev` to open the existing Terrain Edit scene.
