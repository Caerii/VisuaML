# v4 integration and research boundaries

The previous self-contained v3 HTML explorer has now been mapped onto a first-party React route. `CompilerLabPage` renders actual `compiler_lab_demo.run()` stage graphs, not a static mockup. The Fastify route passes only known scenario IDs and seeds to the Python worker; each request is a bounded child process.

## API / state flow

`/compiler` React page → POST `/api/compiler-lab/analyze` → `compiler_lab_service.py` → `compiler_lab_demo.run()` → typed FX graph and verified passes → NumPy-AOT + PyTorch `.pt2` CPU roundtrip → JSON stage graph / receipts / illustrative placement sweep.

The main collaborative application remains separate. An explicit user-approved commit-to-canvas action may later store a validated stage graph in Yjs; v4 does *not* write to the shared Yjs document.

## Reference method

The neural-assemblies repository defines deterministic winner-set selection, including lowest-ID tie-breaking, for its `numpy_exact` path. The v4 bridge checks that operation against an independent threshold/tie formulation and can call the actual library implementation when installed. It does not model firing-history, recurrent connectomes, synaptic plasticity, refraction, LRI, or recording policies. Ordered winner indices can legitimately differ from a binary winner mask. Full engine parity requires explicit experiment parameters and state snapshots and should be implemented as its own test corpus.

## Device planner caveat

The cost optimizer solves a graph cut for its declared additive objective, not a realistic photonics system objective. Missing physical evidence includes transfer overhead, converter costs, calibration, laser budget, input encoding, memory staging, thermal effects, fabrication variation, and throughput/energy benchmarking. Never present v4's abstract cost values as device results.
