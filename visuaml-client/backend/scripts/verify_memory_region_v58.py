"""Check guarded copy lowering and symbolic recovery against the same WASM contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import wasmtime
import z3

from scripts.benchmark_neural_vm_v58 import git_identity
from scripts.verify_memory_recovery_v58 import initial_page, replay_trace, word_values
from scripts.verify_packed_control_v58 import GATES, first_divergence, prepare
from visuaml.compiler_lab.memory_region_contract_v58 import (
    RegionObservation, first_access_difference, first_counterexample, first_difference, observe,
)
from visuaml.compiler_lab.neural_microstep_executor_v58 import MemoryWeights
from visuaml.compiler_lab.wasm_control_flow_v51 import MEMORY_BYTES, decode_control, encode_control
from visuaml.compiler_lab.wasm_copy_region_v58 import PreparedCopyRegion, copy_program

ADDRESSES = (0, 4, 8, 9, 12, 16, 24, MEMORY_BYTES - 4, MEMORY_BYTES - 3, (1 << 32) - 1)
COUNTS = (-1, 0, 1, 2, 3, 4)
PATTERNS = ((1, 2, 3, 4, 5, 6), (-1, 0, (1 << 31) - 1, -(1 << 31), 33, 71))


@dataclass(frozen=True, order=True)
class Case:
    count: int
    src: int
    dst: int
    pattern: int = 0

    def memory(self):
        page = initial_page((0, 0, *PATTERNS[self.pattern]))
        struct.pack_into("<II", page, 0, self.src, self.dst)
        struct.pack_into("<i", page, MEMORY_BYTES - 4, 109)
        return page

    def fits(self):
        return self.count <= 0 or max(self.src, self.dst) + 4 * self.count <= MEMORY_BYTES


def _record(kind, page, address, after=None):
    before = page[address:address + 4].hex()
    return {"kind": kind, "address": address, "before_hex": before,
            "after_hex": before if after is None else after.hex()}


def symbolic(case, candidate):
    """Declarative candidates in the small recovery grammar."""
    page = case.memory()
    accesses = []
    if case.count > 0:
        src, dst = struct.unpack_from("<II", page)
        accesses.extend(_record("i32.load", page, address) for address in (0, 4))

        def signed(value):
            return value if value < 1 << 31 else value - (1 << 32)

        backward = candidate == "backward" or (candidate == "directional" and signed(src) < signed(dst))
        indexes = reversed(range(case.count)) if backward else range(case.count)
        for index in indexes:
            source = (src + 4 * index) & 0xffffffff
            target = (dst + 4 * index) & 0xffffffff
            if source > MEMORY_BYTES - 4 or target > MEMORY_BYTES - 4:
                return RegionObservation("memory_oob", None, bytes(page))
            value = page[source:source + 4]
            accesses.append(_record("i32.load", page, source))
            accesses.append(_record("i32.store", page, target, value))
            page[target:target + 4] = value
    return RegionObservation("return", case.count, bytes(page), accesses=tuple(accesses))


def wrong_forward_lowering(case):
    """An in-bounds compiler candidate that always issues low addresses first."""
    if not case.fits():
        raise ValueError("Forward lowering assumes in-bounds ranges")
    page = case.memory()
    with memoryview(page) as memory:
        for offset in range(0, 4 * case.count, 4):
            src, dst = case.src + offset, case.dst + offset
            memory[dst:dst + 4] = memory[src:src + 4].tobytes()
    return RegionObservation("return", case.count, bytes(page))


def prove_direction_lemmas():
    """Prove byte-interval facts used by the bulk-lowering argument."""
    src, dst, i, j, count = z3.Ints("src dst i j count")
    fits = z3.And(src >= 0, dst >= 0, count >= 1,
                  src + 4 * count <= MEMORY_BYTES, dst + 4 * count <= MEMORY_BYTES)
    indexes = z3.And(i >= 0, j >= 0, i < count, j < count)
    obligations = {
        "forward_writes_preserve_unread_words": z3.Implies(
            z3.And(fits, indexes, dst <= src, i < j), dst + 4 * i + 4 <= src + 4 * j,
        ),
        "backward_writes_preserve_unread_words": z3.Implies(
            z3.And(fits, indexes, src < dst, j < i), src + 4 * j + 4 <= dst + 4 * i,
        ),
        "guarded_word_addresses_do_not_wrap": z3.Implies(
            z3.And(fits, indexes),
            z3.And(src + 4 * i >= 0, dst + 4 * j >= 0,
                   src + 4 * i + 3 < 1 << 32, dst + 4 * j + 3 < 1 << 32),
        ),
    }
    for claim, expression in obligations.items():
        solver = z3.Solver()
        solver.add(z3.Not(expression))
        if solver.check() != z3.unsat:
            raise AssertionError(("Unproved region obligation", claim))
    return list(obligations)


def collect(root):
    identity = git_identity(root)
    raw = encode_control(copy_program())
    source = decode_control(raw)
    packed = prepare(source)
    lowered = PreparedCopyRegion(source)
    store = wasmtime.Store()
    wasm = wasmtime.Instance(store, wasmtime.Module(store.engine, raw), []).exports(store)
    cases = tuple(Case(*case) for case in product(COUNTS, ADDRESSES, ADDRESSES, range(len(PATTERNS))))

    def reference(case, *, capture_trace=False, fuel=400):
        return observe(packed.run_memory, case.count, case.memory(), capture_trace=capture_trace, fuel=fuel)

    def candidate(case, *, capture_trace=False, fuel=400):
        return observe(lowered.run_memory, case.count, case.memory(), capture_trace=capture_trace, fuel=fuel)

    domain = {}
    fuel_cases = trap_cases = fused_cases = 0
    for case in cases:
        initial = case.memory()
        wasm["memory"].write(store, initial, 0)
        try:
            result = wasm["run"](store, case.count)
        except wasmtime.Trap as error:
            assert error.trap_code == wasmtime.TrapCode.MEMORY_OUT_OF_BOUNDS
            origin = RegionObservation("memory_oob", None, bytes(wasm["memory"].read(store)))
            trap_cases += 1
        else:
            origin = RegionObservation("return", result, bytes(wasm["memory"].read(store)))
        expected = reference(case)
        assert first_difference(origin, expected) is None, case
        domain[case] = expected
        actual_page = bytearray(initial)
        if case.fits():
            result = lowered.run_memory(case.count, actual_page, capture_trace=False)
            assert result.fused
            actual = RegionObservation("return", result.result_i32, bytes(actual_page), result.steps)
            assert first_difference(expected, actual, source_steps=True) is None, case
            fused_cases += 1
        else:
            assert first_difference(expected, candidate(case)) is None, case
        if expected.exit == "return":
            prior_fuel = expected.source_steps - 1
            prior = reference(case, fuel=prior_fuel)
            assert prior.exit == "fuel"
            assert first_difference(prior, candidate(case, fuel=prior_fuel)) is None, case
            assert first_difference(expected, candidate(case, fuel=expected.source_steps), source_steps=True) is None
            fuel_cases += 2

    # Disjoint evidence leaves traversal direction ambiguous. Both the
    # first recovery hypothesis and wrong lowering face the same domain.
    seen = [Case(3, 8, 24)]
    in_bounds = tuple(case for case in cases if case.fits())
    rounds = []
    for _ in range(3):
        hypothesis = next(name for name in ("forward", "backward", "directional")
                          if all(first_difference(domain[case], symbolic(case, name)) is None for case in seen))
        bad = first_counterexample(domain.__getitem__, lambda case: symbolic(case, hypothesis), cases)
        if bad is None:
            rounds.append({"hypothesis": hypothesis, "counterexample": None})
            break
        case, difference = bad
        rounds.append({"hypothesis": hypothesis, "counterexample": asdict(case), "difference": difference})
        seen.append(case)
    assert hypothesis == "directional" and bad is None
    wrong = first_counterexample(domain.__getitem__, wrong_forward_lowering, in_bounds)
    assert wrong[0] == Case(**rounds[0]["counterexample"])

    overlap = Case(3, 8, 12)
    original = reference(overlap, capture_trace=True)
    recovered = symbolic(overlap, hypothesis)
    assert first_difference(original, recovered, access_order=True) is None
    assert word_values(original.page[8:], 4) == (1, 1, 2, 3)
    assert word_values(wrong_forward_lowering(overlap).page[8:], 4) == (1, 1, 1, 1)
    traced = lowered.run_memory(overlap.count, overlap.memory())
    assert not traced.fused
    assert first_difference(original, candidate(overlap, capture_trace=True), access_order=True) is None
    assert replay_trace(overlap.memory(), traced.trace)[0] == original.page

    same_final = Case(2, 8, 24)
    order_ref = reference(same_final, capture_trace=True)
    order_candidate = symbolic(same_final, "forward")
    assert first_difference(order_ref, order_candidate) is None
    order_difference = first_access_difference(order_ref.accesses, order_candidate.accesses)
    assert order_difference["index"] == 2

    late_trap = Case(2, MEMORY_BYTES - 4, 8)
    committed = reference(late_trap)
    assert committed.exit == "memory_oob" and word_values(committed.page[8:], 1) == (109,)
    assert first_difference(committed, candidate(late_trap)) is None
    eager = RegionObservation("memory_oob", None, bytes(late_trap.memory()))
    eager_difference = first_difference(committed, eager)
    assert eager_difference["address"] == 8

    altered = PreparedCopyRegion(source, memory_fixed=MemoryWeights(store_gain=0))
    altered_ref = prepare(source, memory_fixed=MemoryWeights(store_gain=0))
    initial = overlap.memory()
    altered_page = bytearray(initial)
    assert not altered.run_memory(overlap.count, altered_page, capture_trace=False).fused
    assert altered_page == initial
    observed = observe(altered.run_memory, overlap.count, initial, capture_trace=True)
    assert first_difference(observe(altered_ref.run_memory, overlap.count, initial, capture_trace=True),
                            observed, access_order=True) is None
    gain_difference = first_difference(original, observed)
    assert gain_difference["address"] == 12
    left = lowered.run_memory(overlap.count, overlap.memory())
    right = altered.run_memory(overlap.count, overlap.memory())
    first_step = first_divergence(left.trace, right.trace, (*GATES, "memory_access"))
    assert first_step["changes"]["memory_access"][0]["kind"] == "i32.store"

    return {
        "source": identity, "wasm_hex": raw.hex(), "wasm_sha256": hashlib.sha256(raw).hexdigest(),
        "contract": {
            "page_bytes": MEMORY_BYTES, "entry_pc": 0,
            "descriptor": {"source": 0, "destination": 4, "encoding": "u32 little-endian"},
            "initial_word_patterns": PATTERNS, "sentinel_byte": 165, "last_word": 109,
            "counts": COUNTS, "addresses": ADDRESSES, "default_fuel": 400,
            "domain_witnesses": len(cases), "fused_witnesses": fused_cases, "memory_traps": trap_cases,
            "source_fuel_boundaries": fuel_cases,
            "optimization_guard": ["exact source", "default gains", "untraced",
                                   "in-bounds ranges", "sufficient source fuel"],
            "proof": {"kind": "SMT interval lemmas", "checks": prove_direction_lemmas()},
            "terminal_equivalence": "exhaustive on the declared domain",
            "diagnostic_equivalence": "packed replay for traced invocations",
            "open_claims": ["general source-region discovery", "access history before a trap",
                            "all-input executable equivalence"],
        },
        "recovery": {"seed": asdict(seen[0]), "rounds": rounds,
                     "recovered": "Copy high to low when signed destination > signed source; otherwise low to high."},
        "wrong_lowering": {"counterexample": asdict(wrong[0]), "difference": wrong[1]},
        "aligned_overlap": {
            "case": asdict(overlap), "reference_words": [1, 1, 2, 3], "wrong_words": [1, 1, 1, 1],
            "first_access_difference": first_access_difference(original.accesses, symbolic(overlap, "forward").accesses),
            "trace": traced.trace,
        },
        "same_final_different_accesses": {"case": asdict(same_final), "difference": order_difference},
        "late_trap": {"case": asdict(late_trap), "exit": committed.exit,
                      "committed_word": 109, "eager_trap_difference": eager_difference},
        "store_gain_intervention": {"first_step": first_step, "terminal_difference": gain_difference},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    if args.output.exists() or args.output.resolve().is_relative_to(root):
        parser.error("Use a fresh output path outside the checkout")
    evidence = collect(root)
    with args.output.open("x") as stream:
        json.dump(evidence, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
