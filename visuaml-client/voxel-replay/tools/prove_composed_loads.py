"""Offline SMT obligations for a composition of two guarded load templates."""
import json
from pathlib import Path
from time import perf_counter_ns

import z3

from prove_reusable_region_rules import OUT as PARENT, unequal
from prove_rule_portfolio import OUT as PORTFOLIO
from replay_frozen_recurrent import EVIDENCE, sha
from visuaml.compiler_lab import frozen_composed_v58, frozen_recurrent_v58

OUT = EVIDENCE.parent / "composed-load-block"
VERSION = "two-guarded-loads-source-boundaries-v1"


def bindings():
    return {name: sha(path.read_bytes()) for name, path in {
        "parent_load_proof": PARENT / "rules.json", "portfolio_proof": PORTFOLIO / "rules.json",
        "graph": OUT / "machine.pt", "planner": OUT / "planner.wasm",
        "lowering": Path(frozen_composed_v58.__file__), "checker": Path(__file__),
        "admission": Path(__file__).with_name("replay_composed_loads.py"),
    }.items()}


def prove():
    n = frozen_recurrent_v58.MAX_STACK
    words = [z3.BitVec(f"stack_{i}", 32) for i in range(n)]
    height, fuel, pc = z3.Ints("height fuel pc")
    page = z3.Array("memory", z3.BitVecSort(64), z3.BitVecSort(8))
    p, q, off_p, off_q = z3.BitVecs("local_p local_q offset_p offset_q", 32)
    size = z3.BitVec("size", 64)
    zero = z3.BitVecVal(0, 32)
    addr = [z3.ZeroExt(32, x) + z3.ZeroExt(32, off) for x, off in ((p, off_p), (q, off_q))]
    legal = [z3.ULE(a + 4, size) for a in addr]
    v, w = [z3.Concat(*(page[a + k] for k in (3, 2, 1, 0))) for a in addr]
    vector = [sum(z3.ZeroExt(24, page[a + k]) << (8 * k) for k in range(4)) for a in addr]
    premises = [height >= 0, height <= n - 3, fuel >= 4, pc >= 0,
                z3.UGE(size, 4), z3.ULE(size, 256 * 65536), *legal,
                *(z3.Implies(i >= height, word == zero) for i, word in enumerate(words))]

    def push(stack, h, value):
        return [z3.If(i < h, word, z3.If(i == h, value, zero)) for i, word in enumerate(stack)]

    first = push(words, height, v)
    final = push(first, height + 1, w)
    composed = [z3.If(i == height, vector[0], z3.If(i == height + 1, vector[1], item))
                for i, item in enumerate(words)]
    checks = [
        unequal("first load exit satisfies second source-entry height and fuel",
                (height + 1 <= n - 2, fuel - 2 >= 2), (True, True), premises)[0],
        unequal("both little-endian lanes survive one scatter; final PC, fuel and event",
                (*final, pc + 4, fuel - 4, height + 2, addr[1], w),
                (*composed, pc + 4, fuel - 4, height + 2, addr[1], vector[1]), premises)[0],
    ]
    for prefix, source in enumerate((push(words, height, p), first,
                                     push(first, height + 1, q)), 1):
        # Interior paths dispatch the existing original transitions.
        checked = [tuple(words), push(words, height, p), first, push(first, height + 1, q)][prefix]
        checks.append(unequal(f"ordinary delegated source prefix {prefix}",
                              (*source, pc + prefix, fuel - prefix),
                              (*checked, pc + prefix, fuel - prefix), premises)[0])
    assert all(x["result"] == "unsat" for x in checks)
    wrong = [z3.If(i == height, w, item) for i, item in enumerate(words)]
    rejected, witness = unequal("rejected seam uses the second word at prefix two", first, wrong,
                                [*premises, v == 7, w == 11])
    assert witness is not None
    return {"version": VERSION, "bindings": bindings(), "solver": z3.get_version_string(),
            "proof_kind": "recorded SMT results and replayable queries", "obligations": checks,
            "negative": {**rejected, "counterexample_words": [7, 11],
                         "completion": unequal("negative final words still agree", final, composed, premises)[0]}}


def main():
    start = perf_counter_ns()
    record = prove()
    elapsed = perf_counter_ns() - start
    data = (json.dumps(record, indent=2) + "\n").encode()
    (OUT / "rules.json").write_bytes(data)
    (OUT / "rules.sha256").write_text(sha(data) + "\n")
    print(json.dumps({"proof_ns": elapsed, "certificate_sha256": sha(data)}))


if __name__ == "__main__":
    main()
