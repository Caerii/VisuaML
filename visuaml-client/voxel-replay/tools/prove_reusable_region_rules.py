"""Offline SMT checks for a versioned pair of reviewed lowering templates."""
import json
from pathlib import Path
from time import perf_counter_ns

import z3

from replay_frozen_multistep import CALLS
from replay_frozen_recurrent import EVIDENCE, sha
from replay_guest_rewrite import EXECUTOR
from visuaml.compiler_lab import frozen_calls_v58, frozen_recurrent_v58, frozen_region_v58

OUT = EVIDENCE.parent / "reusable-region-rules"
VERSION = "ordered-local-pairs-and-guarded-loads-v1"


def binding():
    paths = {
        "single_step": CALLS / "machine.pt", "baseline": EXECUTOR / "machine.pt",
        "lowering": OUT / "machine.pt", "symbolic_model": Path(__file__),
        "ordinary_transition": Path(frozen_recurrent_v58.__file__),
        "call_transition": Path(frozen_calls_v58.__file__),
        "region_transition": Path(frozen_region_v58.__file__),
    }
    return {label: sha(path.read_bytes()) for label, path in paths.items()}


def unequal(name, left, right, assumptions):
    solver = z3.Solver()
    solver.add(*assumptions, z3.Or(*(a != b for a, b in zip(left, right, strict=True))))
    status = solver.check()
    record = {"obligation": name, "result": str(status), "smt2": solver.to_smt2()}
    assert status in (z3.unsat, z3.sat)
    return record, solver.model() if status == z3.sat else None


def prove():
    cap = frozen_recurrent_v58.MAX_STACK
    height, pc, fuel, a, b = z3.Ints("height pc fuel slot_a slot_b")
    locals_ = z3.Array("locals", z3.IntSort(), z3.BitVecSort(32))
    stack = [z3.BitVec(f"stack_{i}", 32) for i in range(cap)]
    zero = z3.BitVecVal(0, 32)
    conditions = [height >= 0, height <= cap - 2, pc >= 0, fuel >= 2, a >= 0, b >= 0,
                  a <= frozen_recurrent_v58.MAX_LOCALS, b <= frozen_recurrent_v58.MAX_LOCALS,
                  *(z3.Implies(i >= height, cell == zero) for i, cell in enumerate(stack))]
    first, second = locals_[a], locals_[b]

    def push(active, at, val):
        return [z3.If(i < at, cell, z3.If(i == at, val, zero)) for i, cell in enumerate(active)]

    source_one = push(stack, height, first)
    pair_one = [z3.If(i == height, first, cell) for i, cell in enumerate(stack)]
    pair_two = [z3.If(i == height, first, z3.If(i == height + 1, second, cell))
                for i, cell in enumerate(stack)]
    obligations = [
        unequal("pair: one-source-step fallback", (*source_one, pc + 1, fuel - 1, height + 1),
                (*pair_one, pc + 1, fuel - 1, height + 1), conditions),
        unequal("pair: ordered scatter", (*push(source_one, height + 1, second), pc + 2, fuel - 2, height + 2),
                (*pair_two, pc + 2, fuel - 2, height + 2), conditions),
    ]

    page = z3.Array("page", z3.BitVecSort(64), z3.BitVecSort(8))
    offset = z3.BitVec("offset_u32", 32)
    capacity = z3.BitVec("page_bytes", 64)
    load_conditions = [*conditions, z3.UGE(capacity, 4), z3.ULE(capacity, 256 * 65536)]
    source_address = z3.ZeroExt(32, first) + z3.ZeroExt(32, offset)
    fused_address = z3.ZeroExt(32, offset) + z3.ZeroExt(32, first)
    source_legal = z3.ULE(source_address + 4, capacity)
    fused_legal = z3.ULE(fused_address, capacity - 4)
    source_word = z3.Concat(*(page[source_address + i] for i in (3, 2, 1, 0)))
    fused_word = sum(z3.ZeroExt(24, page[fused_address + i]) << (8 * i) for i in range(4))
    one_step_fuel = [condition for i, condition in enumerate(load_conditions) if i != 3] + [fuel == 1]
    # When the guard is closed only the get can commit. No event or additional fuel is consumed.
    obligations += [
        unequal("load: bounds, including unsigned offset", (source_legal,), (fused_legal,), load_conditions),
        unequal("load: cutoff before the read", (*source_one, pc + 1, fuel - 1, height + 1, 0),
                (*pair_one, pc + 1, fuel - 1, height + 1, 0), one_step_fuel),
        unequal("load: in-bounds final state and access event",
                (*push(source_one, height, source_word), pc + 2, fuel - 2, height + 1,
                 1, source_address, source_word, source_word),
                (*(z3.If(i == height, fused_word, cell) for i, cell in enumerate(stack)),
                 pc + 2, fuel - 2, height + 1, 1, fused_address, fused_word, fused_word),
                [*load_conditions, source_legal]),
    ]
    rows = [row for row, _ in obligations]
    assert all(row["result"] == "unsat" for row in rows)
    return {"version": VERSION, "sha256_bindings": binding(), "z3": z3.get_version_string(),
            "obligations": rows}


def eager_counterexample():
    x = z3.BitVec("local_argument", 32)
    address = z3.ZeroExt(32, x) + 1
    # The source stops with exhausted fuel and no access. An eager plan
    # instead attempts the load and reports a trap/access for this page.
    bad = z3.UGT(address, 65536 - 4)
    record, model = unequal("eager plan crosses the one-fuel cutoff", (2, 0),
                            (z3.If(bad, 3, 0), z3.If(bad, 3, 1)),
                            [z3.UGE(address, 65536 - 3), z3.ULT(address, 65536)])
    assert model is not None
    return {**record, "argument": model.eval(x).as_long(), "offset": 1, "fuel": 1}


def main():
    started = perf_counter_ns()
    certificate = prove()
    elapsed = perf_counter_ns() - started
    path = OUT / "rules.json"
    data = (json.dumps(certificate, indent=2) + "\n").encode()
    path.write_bytes(data)
    (OUT / "rules.sha256").write_text(sha(data) + "\n")
    print(json.dumps({"offline_proof_ns": elapsed, "certificate_sha256": sha(data)}))


if __name__ == "__main__":
    main()
