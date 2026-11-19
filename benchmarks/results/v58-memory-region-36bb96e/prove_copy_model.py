"""Prove bytewise guarded copies against an arbitrary initial array for counts 1–4."""
import json
import sys
from pathlib import Path

import z3

from scripts.benchmark_neural_vm_v58 import git_identity
from visuaml.compiler_lab.wasm_control_flow_v51 import MEMORY_BYTES

source = git_identity(Path.cwd())
initial = z3.Array("initial", z3.IntSort(), z3.BitVecSort(8))
src, dst, observed_address = z3.Ints("src dst observed_address")


def transfer(page, reads, word):
    offset = 4 * word
    for lane in range(4):
        page = z3.Store(page, dst + offset + lane, z3.Select(reads, src + offset + lane))
    return page


rows = []
for count in range(1, 5):
    bulk = initial
    for word in range(count):
        bulk = transfer(bulk, initial, word)
    for backward in (False, True):
        actual = initial
        order = reversed(range(count)) if backward else range(count)
        for word in order:
            actual = transfer(actual, actual, word)
        solver = z3.Solver()
        solver.set(timeout=10_000)
        solver.add(src >= 0, dst >= 0, src + 4 * count <= MEMORY_BYTES, dst + 4 * count <= MEMORY_BYTES)
        solver.add(src < dst if backward else src >= dst)
        solver.add(z3.Select(actual, observed_address) != z3.Select(bulk, observed_address))
        result = solver.check()
        assert result == z3.unsat, (count, backward, result, solver.reason_unknown())
        rows.append({"count": count, "direction": "backward" if backward else "forward", "negated_claim": str(result)})
with Path(sys.argv[1]).open("x") as stream:
    json.dump({"source": source, "solver": z3.get_version_string(), "obligations": rows,
               "scope": "arbitrary initial bytes and guarded addresses, model only"},
              stream, indent=2, sort_keys=True)
    stream.write("\n")
