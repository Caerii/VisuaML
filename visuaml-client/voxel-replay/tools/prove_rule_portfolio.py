"""Offline proof for the new constant/binary-arithmetic template."""
import json
from pathlib import Path
from time import perf_counter_ns

import z3

from prove_reusable_region_rules import OUT as PRIOR, unequal
from replay_frozen_recurrent import EVIDENCE, sha
from visuaml.compiler_lab import frozen_portfolio_v58, frozen_recurrent_v58

OUT = EVIDENCE.parent / "rule-portfolio"
VERSION = "const-binary-preserve-source-prefix-v1"


def bindings():
    return {key: sha(path.read_bytes()) for key, path in {
        "parent_certificate": PRIOR / "rules.json", "runner": OUT / "machine.pt",
        "chooser": OUT / "chooser.wasm", "checker": Path(__file__),
        "lowering": Path(frozen_portfolio_v58.__file__),
        "arithmetic": Path(frozen_recurrent_v58.__file__),
        "admission": Path(__file__).with_name("replay_rule_portfolio.py"),
    }.items()}


def prove():
    size = frozen_recurrent_v58.MAX_STACK
    stack = [z3.BitVec(f"s{i}", 32) for i in range(size)]
    c = z3.BitVec("constant_word", 32)
    height, pc, fuel = z3.Ints("height pc fuel")
    position = height - 1
    zero = z3.BitVecVal(0, 32)
    constraints = [height >= 1, height <= size - 2, pc >= 0, fuel >= 2,
                   *(z3.Implies(i >= height, cell == zero) for i, cell in enumerate(stack))]
    source_prefix = [z3.If(i < height, cell, z3.If(i == height, c, zero)) for i, cell in enumerate(stack)]
    fallback_prefix = [z3.If(i == height, c, cell) for i, cell in enumerate(stack)]
    checks = [unequal("one-step ordinary const prefix", (*source_prefix, pc + 1, fuel - 1, height + 1),
                      (*fallback_prefix, pc + 1, fuel - 1, height + 1), constraints)[0]]
    left = zero
    for i, cell in enumerate(source_prefix):
        left = z3.If(position == i, cell, left)
    entry = zero
    for i, cell in enumerate(stack):
        entry = z3.If(position == i, cell, entry)
    # The i64 signed product always fits. Taking its low 32 bits gives
    # the same residue as multiplication in the ring Z/(2**32).
    operations = (("add", left + c, entry + c),
                  ("mul", left * c, entry * c),
                  ("lt_s", z3.If(left < c, z3.BitVecVal(1, 32), zero),
                   z3.If(entry < c, z3.BitVecVal(1, 32), zero)))
    for name, source_word, fused_word in operations:
        source = [z3.If(i < height, z3.If(position == i, source_word, cell), zero)
                  for i, cell in enumerate(source_prefix)]
        fused = [z3.If(position == i, fused_word, cell) for i, cell in enumerate(stack)]
        check, _ = unequal(f"{name}: final source boundary", (*source, height, pc + 2, fuel - 2),
                           (*fused, height, pc + 2, fuel - 2), constraints)
        checks.append(check)
    assert all(check["result"] == "unsat" for check in checks)
    return {"version": VERSION, "bindings": bindings(), "obligations": checks,
            "evidence_kind": "recorded solver result with independently rerunnable SMT-LIB queries"}


def main():
    started = perf_counter_ns()
    certificate = prove()
    elapsed = perf_counter_ns() - started
    raw = (json.dumps(certificate, indent=2) + "\n").encode()
    (OUT / "rules.json").write_bytes(raw)
    (OUT / "rules.sha256").write_text(sha(raw) + "\n")
    print(json.dumps({"offline_proof_ns": elapsed, "certificate_sha256": sha(raw)}))


if __name__ == "__main__":
    main()
