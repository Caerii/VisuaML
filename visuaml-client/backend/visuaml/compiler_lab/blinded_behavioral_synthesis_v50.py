"""v50 bounded blinded behavioral synthesis from neural-VM output observations.

This module sees ONLY a black-box integer oracle and an opaque equivalence
verifier callback. It cannot read the source program, program tokens, model
weights, construction receipt, or instruction trace. It proposes the simplest
function in a declared grammar, uses counterexamples to refine, and records
all queries.

Grammar: signed-i32 modular polynomials degree <=2 with coefficients in
[-16,16], eqz(x), and signed lt_s/gt_s with constant threshold [-16,16].
It is not arbitrary pretrained-weight decompilation, not unique historical
source recovery, not identification of an unknown transformer architecture.

Source/target equivalence is established ONLY by a separate optional Z3
bitvector verifier against a known typed source program; the synthesis
function never receives that source. The full neural bit-gate model itself
is independently checked on finite adversarial inputs, not proved for all
2**32 inputs by this CEGIS procedure.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from collections.abc import Callable

from .wasm_state_vm_v40 import I32Program, VMOperation, encode_program

SCHEMA = "visuaml.blinded-neural-behavior-synthesis.v50"
GRAMMAR = "i32-poly-degree2-smallcoeff-or-signed-predicate.v50"
INT_MIN = -(1 << 31)
INT_MAX = (1 << 31)-1
INITIAL = (-3, -2, -1, 0, 1, 2, 3)
VALID_KINDS = ("poly", "eqz", "lt_s", "gt_s")


class BehavioralSynthesisError(ValueError):
    """Insufficient observations, unsupported behavior, or invalid oracle."""


def _wrap(value: int) -> int:
    u = value & 0xFFFFFFFF
    return u if u < (1 << 31) else u - (1 << 32)


def _i32(value: object) -> bool:
    return type(value) is int and INT_MIN <= value <= INT_MAX


@dataclass(frozen=True)
class Candidate:
    kind: str
    a: int = 0
    b: int = 0
    c: int = 0
    threshold: int = 0

    def expression(self) -> str:
        if self.kind == "poly":
            return f"({self.a})*x*x+({self.b})*x+({self.c}) mod 2^32"
        if self.kind == "eqz":
            return "i32.eqz(x)"
        return f"i32.{self.kind}(x,{self.threshold})"

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()

    def value(self, x: int) -> int:
        if not _i32(x):
            raise BehavioralSynthesisError("Oracle input is not signed i32")
        if self.kind == "poly":
            return _wrap(self.a*x*x + self.b*x + self.c)
        if self.kind == "eqz":
            return int(x == 0)
        if self.kind == "lt_s":
            return int(x < self.threshold)
        if self.kind == "gt_s":
            return int(x > self.threshold)
        raise BehavioralSynthesisError("Unsupported candidate kind")


def _admit_candidate(candidate: Candidate, limit: int) -> None:
    if (not isinstance(candidate, Candidate)
            or candidate.kind not in VALID_KINDS
            or any(type(v) is not int for v in (
                candidate.a, candidate.b, candidate.c, candidate.threshold
            ))
            or any(abs(v) > limit for v in (
                candidate.a, candidate.b, candidate.c, candidate.threshold
            ))
            or (candidate.kind == "poly" and candidate.threshold != 0)
            or (candidate.kind != "poly" and any((
                candidate.a, candidate.b, candidate.c
            )))
            or (candidate.kind == "eqz" and candidate.threshold != 0)):
        raise BehavioralSynthesisError("Unsupported bounded hypothesis")


def _complexity(h: Candidate) -> tuple:
    if h.kind == "poly":
        degree = 2 if h.a else (1 if h.b else 0)
        if h.a == 0 and h.b == 1 and h.c == 0:
            degree = -1  # canonical identity
        used = sum(int(v != 0) for v in (h.a, h.b, h.c))
        return degree + 2, used, sum(abs(v) for v in (h.a, h.b, h.c)), h.kind, h.a, h.b, h.c
    if h.kind == "eqz":
        return 1, 1, 0, h.kind, 0, 0, 0
    return 3, 1, abs(h.threshold), h.kind, 0, 0, h.threshold


def _hypotheses(observed: dict[int, int], limit: int) -> tuple[Candidate, ...]:
    if not all(x in observed for x in (0, 1, -1)):
        raise BehavioralSynthesisError("Required three-point discovery observations")
    if type(limit) is not int or not 1 <= limit <= 64:
        raise BehavioralSynthesisError("Unbounded hypothesis coefficient budget")

    candidates = []
    c = observed[0]
    if -limit <= c <= limit:
        # y1 = a+b+c in i32 modular arithmetic. Enumerate one coefficient,
        # derive the other algebraically instead of testing an O(K^3) grid.
        for a in range(-limit, limit+1):
            b = _wrap(observed[1] - c - a)
            if -limit <= b <= limit:
                candidates.append(Candidate("poly", a, b, c))
    candidates.append(Candidate("eqz"))
    for k in range(-limit, limit+1):
        candidates.extend((Candidate("lt_s", threshold=k),
                           Candidate("gt_s", threshold=k)))
    return tuple(sorted(
        (h for h in candidates if all(h.value(x) == y for x, y in observed.items())),
        key=_complexity,
    ))


def emit_canonical_wasm(candidate: Candidate, *, limit: int = 16) -> tuple[I32Program, bytes]:
    """Compile a proposed behavior to a DIFFERENT, canonical real WASM program."""
    _admit_candidate(candidate, limit)
    op = lambda kind, imm=None: VMOperation(kind, imm)
    if candidate.kind == "poly":
        instructions = (
            op("local.get", 0), op("local.get", 0), op("i32.mul"),
            op("i32.const", candidate.a), op("i32.mul"),
            op("local.get", 0), op("i32.const", candidate.b),
            op("i32.mul"), op("i32.add"),
            op("i32.const", candidate.c), op("i32.add"), op("end"),
        )
    elif candidate.kind == "eqz":
        instructions = (op("local.get", 0), op("i32.eqz"), op("end"))
    else:
        instructions = (
            op("local.get", 0), op("i32.const", candidate.threshold),
            op(f"i32.{candidate.kind}"), op("end"),
        )
    program = I32Program(0, instructions)
    return program, encode_program(program)


def verify_against_typed_source(
    source: I32Program, candidate: Candidate, *, timeout_ms: int = 15000
) -> dict:
    """Separate independent 32-bit bitvector equivalence oracle for CEGIS.

    Only this verifier has access to the source. Its results do not establish
    all-input equivalence of a neural network's float64/binary gate execution.
    """
    if type(timeout_ms) is not int or not 1 <= timeout_ms <= 60000:
        raise BehavioralSynthesisError("Invalid solver budget")
    import z3

    from .neural_symbolic_recovery_v43 import _encode_wasm_in_z3

    _admit_candidate(candidate, 64)
    x = z3.BitVec("input_i32", 32)
    true_behavior = _encode_wasm_in_z3(source, x, z3)
    if candidate.kind == "poly":
        hypothesis = (
            z3.BitVecVal(candidate.a, 32)*x*x
            + z3.BitVecVal(candidate.b, 32)*x
            + z3.BitVecVal(candidate.c, 32)
        )
    elif candidate.kind == "eqz":
        hypothesis = z3.If(
            x == z3.BitVecVal(0, 32), z3.BitVecVal(1, 32), z3.BitVecVal(0, 32)
        )
    else:
        threshold = z3.BitVecVal(candidate.threshold, 32)
        condition = x < threshold if candidate.kind == "lt_s" else x > threshold
        hypothesis = z3.If(condition, z3.BitVecVal(1, 32), z3.BitVecVal(0, 32))
    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    solver.add(true_behavior != hypothesis)
    formula = solver.to_smt2().encode()
    result = solver.check()
    evidence = {
        "schema": "visuaml.blinded-candidate-i32-smt.v50",
        "candidate_sha256": candidate.digest(),
        "source_wasm_sha256": hashlib.sha256(encode_program(source)).hexdigest(),
        "smt_formula_sha256": hashlib.sha256(formula).hexdigest(),
        "z3_version": z3.get_version_string(),
        "solver_outcome": str(result),
        "all_i32_typed_source_equivalent": result == z3.unsat,
        "all_i32_neural_circuit_proved": False,
        "unique_original_source_recovered": False,
        "machine_checked_proof_object": False,
    }
    if result == z3.unknown:
        raise BehavioralSynthesisError("SMT verifier returned unknown: " + solver.reason_unknown())
    if result == z3.sat:
        model = solver.model()
        u = int(model.eval(x, model_completion=True).as_long())
        evidence["counterexample_i32"] = _wrap(u)
        evidence["source_output_u32"] = int(
            model.eval(true_behavior, model_completion=True).as_long()
        )
        evidence["hypothesis_output_u32"] = int(
            model.eval(hypothesis, model_completion=True).as_long()
        )
    return evidence


def synthesize_from_oracle(
    oracle: Callable[[int], int],
    verifier: Callable[[Candidate], dict],
    *, coefficient_limit: int = 16,
    max_refinements: int = 8,
) -> dict:
    """Opaque black-box oracle + counterexample oracle, no source/weight access."""
    if (not callable(oracle) or not callable(verifier)
            or type(coefficient_limit) is not int
            or not 1 <= coefficient_limit <= 64
            or type(max_refinements) is not int
            or not 1 <= max_refinements <= 24):
        raise BehavioralSynthesisError("Invalid bounded blinded synthesis setup")
    observations = {}
    for x in INITIAL:
        result = oracle(x)
        if not _i32(result):
            raise BehavioralSynthesisError("Black-box oracle returned non-i32")
        observations[x] = result

    rounds = []
    for iteration in range(max_refinements + 1):
        candidates = _hypotheses(observations, coefficient_limit)
        if not candidates:
            raise BehavioralSynthesisError("No surviving candidate in admitted grammar")
        hypothesis = candidates[0]
        evidence = verifier(hypothesis)
        if (not isinstance(evidence, dict)
                or evidence.get("candidate_sha256") != hypothesis.digest()):
            raise BehavioralSynthesisError("Unbound verifier claim for candidate")
        if evidence.get("solver_outcome") not in ("sat", "unsat"):
            raise BehavioralSynthesisError("Verifier did not return conclusive result")
        round_report = {
            "iteration": iteration,
            "surviving_candidate_count": len(candidates),
            "selected_hypothesis": asdict(hypothesis),
            "selected_hypothesis_sha256": hypothesis.digest(),
            "verification": evidence,
        }
        rounds.append(round_report)
        if evidence["solver_outcome"] == "unsat":
            compiled, binary = emit_canonical_wasm(
                hypothesis, limit=coefficient_limit
            )
            return {
                "schema": SCHEMA,
                "grammar": GRAMMAR,
                "coefficient_limit": coefficient_limit,
                "source_program_received_by_synthesizer": False,
                "construction_manifest_received_by_synthesizer": False,
                "neural_weights_received_by_synthesizer": False,
                "accepted_candidate": asdict(hypothesis),
                "accepted_expression": hypothesis.expression(),
                "accepted_wasm_sha256": hashlib.sha256(binary).hexdigest(),
                "accepted_wasm_hex": binary.hex(),
                "accepted_typed_program_sha256": compiled.digest(),
                "observed_input_outputs": [
                    {"input_i32": x, "output_i32": y} for x, y in sorted(
                        observations.items()
                    )
                ],
                "rounds": rounds,
                "all_i32_source_vs_candidate_equivalent": True,
                "all_i32_neural_weight_equivalence_proved": False,
                "arbitrary_unknown_weight_decompiled": False,
                "historical_source_uniquely_recovered": False,
            }
        x = evidence.get("counterexample_i32")
        if not _i32(x):
            raise BehavioralSynthesisError("SAT verifier omitted valid i32 counterexample")
        if x in observations:
            raise BehavioralSynthesisError("Verifier contradicted earlier oracle observation")
        y = oracle(x)
        if not _i32(y) or hypothesis.value(x) == y:
            raise BehavioralSynthesisError("Counterexample did not actually falsify hypothesis")
        observations[x] = y
    raise BehavioralSynthesisError("CEGIS refinement fuel exhausted")
