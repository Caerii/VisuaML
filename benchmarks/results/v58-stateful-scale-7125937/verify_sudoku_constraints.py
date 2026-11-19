"""Check all returned grids independently and prove curated puzzle status in Z3."""
import json
import sys
from pathlib import Path

import z3

from scripts.benchmark_neural_vm_v58 import git_identity

out = Path(sys.argv[1])
source = git_identity(Path.cwd())
record = json.loads((out / "runtime-1.json").read_bytes())
assert record["source"] == source
rows = [row for row in record["cases"] if row["family"] == "sudoku"]
units = [
    [9 * row + col for col in range(9)] for row in range(9)
] + [
    [9 * row + col for row in range(9)] for col in range(9)
] + [
    [9 * (row + dy) + col + dx for dy in range(3) for dx in range(3)]
    for row in range(0, 9, 3) for col in range(0, 9, 3)
]
digits = set(range(1, 10))
for row in rows:
    grid = tuple(map(int, row["grid"]))
    givens = tuple(map(int, row["input"]["givens"]))
    assert all(c == grid[i] for i, c in enumerate(givens) if c)
    if row["result"]:
        assert all({grid[i] for i in unit} == digits for unit in units)
checks = []
for row in rows[:5]:
    candidates = [[z3.Bool(f"x{i}_{digit}") for digit in range(9)] for i in range(81)]
    solver = z3.Solver()
    solver.set(timeout=20000)
    solver.add(*(z3.PbEq([(v, 1) for v in choices], 1) for choices in candidates))
    solver.add(*(z3.PbEq([(candidates[i][digit], 1) for i in unit], 1)
                 for unit in units for digit in range(9)))
    solver.add(*(candidates[i][int(digit) - 1] for i, digit in enumerate(row["input"]["givens"]) if digit != "0"))
    if row["result"]:
        solver.add(z3.Or(*(z3.Not(candidates[i][int(digit) - 1]) for i, digit in enumerate(row["grid"]))))
    result = solver.check()
    assert result == z3.unsat, (row["name"], result)
    checks.append({"puzzle": row["name"], "claim": "unique" if row["result"] else "unsatisfiable"})
with (out / "sudoku-checks.json").open("x") as stream:
    json.dump({"source": source, "solver": z3.get_version_string(), "cases": len(rows),
               "valid_solutions": sum(row["result"] for row in rows), "curated_checks": checks},
              stream, indent=2, sort_keys=True)
    stream.write("\n")
