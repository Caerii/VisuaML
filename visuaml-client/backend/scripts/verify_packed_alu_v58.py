"""Check the canonical v58 ALU lowering using bitvectors and carry invariants."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import z3

from scripts.benchmark_neural_vm_v58 import git_identity
from visuaml.compiler_lab.bit_neural_vm_v49 import WIDTH, construct_bit_weights


def prove(name, claim):
    solver = z3.Solver()
    solver.set(timeout=30000)
    solver.add(z3.Not(claim))
    result = solver.check()
    if result != z3.unsat:
        raise AssertionError((name, str(result)))
    return name


def relu(value):
    return max(value, 0)


def verify():
    assert WIDTH == 32
    weights = construct_bit_weights()
    and_coefficients = weights.and_kernel.astype(int).tolist()
    xor_coefficients = weights.xor_kernel.astype(int).tolist()
    assert weights.and_kernel.tolist() == and_coefficients == [1, 1, -1]
    assert weights.xor_kernel.tolist() == xor_coefficients == [[1, -1], [-1, 1]]
    add_thresholds = weights.adder_thresholds.tolist()
    mul_thresholds = weights.carry_thresholds.tolist()
    assert add_thresholds == [1, 2]
    assert mul_thresholds == list(range(2, 65, 2))
    for a in (0, 1):
        for b in (0, 1):
            assert relu(a + b - 1) == a & b
            assert relu(a - b) + relu(b - a) == a ^ b
    for n in range(4):
        assert relu(n - add_thresholds[0]) - relu(n - add_thresholds[1]) == n // 2
    for n in range(64):
        assert sum(relu(n - (t - 1)) - relu(n - t) for t in mul_thresholds) == n // 2

    a, b = z3.BitVecs("a b", WIDTH)
    bits_a = [z3.Extract(i, i, a) for i in range(WIDTH)]
    bits_b = [z3.Extract(i, i, b) for i in range(WIDTH)]
    checks = []
    for subtract in (False, True):
        carry = z3.BitVecVal(int(subtract), 1)
        digits = []
        for left, right in zip(bits_a, bits_b, strict=True):
            digit = ~right if subtract else right
            digits.append(left ^ digit ^ carry)
            carry = (left & digit) | (carry & (left ^ digit))
        expected = a - b if subtract else a + b
        checks.append(prove("subtract" if subtract else "add", z3.Concat(*reversed(digits)) == expected))

    def signed_less(left, right):
        same, less = z3.BoolVal(True), z3.BoolVal(False)
        for x, y in zip(reversed(left), reversed(right), strict=True):
            less = z3.Or(less, z3.And(same, x == 0, y == 1))
            same = z3.And(same, x == y)
        return z3.Or(z3.And(left[-1] == 1, right[-1] == 0),
                     z3.And(left[-1] == right[-1], less))

    sign = 1 << (WIDTH - 1)
    checks.append(prove("signed less", signed_less(bits_a, bits_b) == z3.ULT(a ^ sign, b ^ sign)))
    checks.append(prove("signed greater", signed_less(bits_b, bits_a) == z3.UGT(a ^ sign, b ^ sign)))
    checks.append(prove("zero", z3.And(*(bit == 0 for bit in bits_a)) == (a == 0)))

    # The convolution groups exactly the low-degree monomials of the
    # binary word product. Higher degrees vanish modulo 2^32.
    columns = {(i, k - i): k for k in range(WIDTH) for i in range(k + 1)}
    product = {(i, j): i + j for i in range(WIDTH) for j in range(WIDTH) if i + j < WIDTH}
    assert columns == product
    partial, carry = z3.Ints("partial carry")
    total = partial + carry
    next_carry, bit = total / 2, total % 2
    # This identity advances the invariant at any column: replacing the
    # weighted coefficient and old carry with a bit and the next carry
    # leaves the accumulated product unchanged.
    checks.append(prove("multiply column invariant", z3.Implies(
        z3.And(partial >= 0, partial <= 32, carry >= 0, carry <= 31),
        z3.And(total == bit + 2 * next_carry, next_carry >= 0, next_carry <= 31,
               z3.Or(bit == 0, bit == 1)),
    )))
    return {"checks": checks, "finite_carry_tables": {"add": 4, "multiply": 64},
            "multiply_retained_monomials": len(columns), "solver": z3.get_version_string()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    source = git_identity(root)
    with args.output.open("x") as stream:
        json.dump({"source": source, **verify()}, stream, sort_keys=True, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
