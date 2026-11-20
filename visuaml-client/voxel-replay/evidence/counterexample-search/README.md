# Discover a reachable counterexample

The eight-agent sparse terrain guest starts at entry before either input. Each tick offers Wait or Right for agent 0 while the other agents stay put. The candidate is fixed before the search at prepared PC 100. The guest planner supplies the valid pair. The deliberately faulty harness substitutes local 1 for the first preview, leaving the declared exit pair intact.

The search executes ordinary input prefixes once, restores fresh copies at the candidate region, and prunes descendants once their prefix has a verified failure. It minimizes changed inputs, then ticks, then lexical action history. For terrain the all-Wait two-tick history never reaches the region. The minimum failure is one Right at tick 1. After source step 1185 at PC 100, the expected stack is `[256]` and the faulty preview is `[0]`. The exit reconverges.

The separate passage guest reuses the same rule machinery at its caller entry. Its baseline Wait already fails, demonstrating that a minimum counterexample can require zero edits. The corrected region plans have no disagreement across all six nonempty prefixes of a depth-two binary tree for each guest. The fused correct exit runs on the frozen region graph. The rejected intermediate preview is a harness observation. Every completed reference tick matches Wasmtime's result and full memory, and Z3 rejects the faulty preview while proving the corrected plan's two boundaries. Packed fallback is blocked.

From `visuaml-client/voxel-replay`:

```sh
export PYTHONPATH=.:../backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python tools/search_compiler_counterexample.py
python tools/search_compiler_counterexample.py --session evidence/counterexample-search/terrain-session.json
python tools/search_compiler_counterexample.py --session evidence/counterexample-search/passage-session.json
```

The [replay viewer](http://localhost:4180/replay-viewer.html) opens the saved result and downloads the input suffix, checkpoint and plan. With `python tools/serve_voxel_forks.py` running under the research environment, its search button runs the fixed local witness live.

[Terrain session](terrain-session.json) · [Passage session](passage-session.json)
