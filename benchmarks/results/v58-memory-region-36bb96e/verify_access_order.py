"""Replay access order separately on the successful declared copy cases."""
import hashlib
import json
import sys
from dataclasses import asdict
from itertools import product
from pathlib import Path

from scripts.benchmark_neural_vm_v58 import git_identity
from scripts.verify_memory_recovery_v58 import replay_trace
from scripts.verify_memory_region_v58 import ADDRESSES, COUNTS, PATTERNS, Case, symbolic
from visuaml.compiler_lab.memory_region_contract_v58 import first_difference, observe
from visuaml.compiler_lab.wasm_copy_region_v58 import PreparedCopyRegion, copy_program

source = git_identity(Path.cwd())
lowered = PreparedCopyRegion(copy_program())
rows = []
for parts in product(COUNTS, ADDRESSES, ADDRESSES, range(len(PATTERNS))):
    case = Case(*parts)
    if not case.fits():
        continue
    initial = case.memory()
    result = lowered.run_memory(case.count, bytearray(initial))
    assert not result.fused
    expected = observe(lowered.run_memory, case.count, initial, capture_trace=True)
    candidate = symbolic(case, "directional")
    assert first_difference(expected, candidate, access_order=True) is None, case
    replayed, accesses = replay_trace(initial, result.trace)
    assert replayed == expected.page
    rows.append({"case": asdict(case), "accesses": accesses,
                 "access_sha256": hashlib.sha256(json.dumps(expected.accesses, sort_keys=True).encode()).hexdigest()})
with Path(sys.argv[1]).open("x") as stream:
    json.dump({"source": source, "cases": len(rows), "rows": rows}, stream, indent=2, sort_keys=True)
    stream.write("\n")
