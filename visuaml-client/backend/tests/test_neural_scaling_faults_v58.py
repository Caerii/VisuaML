"""v58 scaling and adversarial campaign source-disjointness tests."""
from __future__ import annotations

import hashlib

import pytest

from scripts.benchmark_faults_v58 import ATTACKS, campaign, run_attack
from scripts.benchmark_neural_vm_v58 import workloads
from visuaml.compiler_lab.wasm_control_flow_v51 import (
    MAX_NESTING, MAX_OPERATIONS, MAX_STACK, decode_control, encode_control,
)
from visuaml.compiler_lab.wasm_scale_corpus_v58 import (
    arithmetic_chain, nested_frames, scaling_corpus, stack_depth,
    unwind_frames,
)


def test_scaling_corpus_is_canonical_bounded_and_source_distinct():
    cases = scaling_corpus()
    assert len(cases) == 26
    assert len({name for name in cases}) == len(cases)
    hashes = set()
    for program, args, dimension in cases.values():
        assert len(program.operations) <= MAX_OPERATIONS
        assert args and dimension
        raw = encode_control(program)
        assert decode_control(raw) == program
        hashes.add(hashlib.sha256(raw).hexdigest())
    assert len(hashes) == 26


@pytest.mark.parametrize("depth", (1, 2, 4, 8, 12))
def test_reviewed_control_frame_boundaries(depth):
    assert depth <= MAX_NESTING
    for program in (nested_frames(depth), nested_frames(depth, kind="loop"),
                    unwind_frames(depth)):
        assert decode_control(encode_control(program)) == program


def test_24_bit_sliced_operands_and_40_alu_steps_admitted():
    assert decode_control(
        encode_control(stack_depth(MAX_STACK))
    ) == stack_depth(MAX_STACK)
    assert decode_control(
        encode_control(arithmetic_chain(40))
    ) == arithmetic_chain(40)
    with pytest.raises(ValueError):
        stack_depth(MAX_STACK + 1)
    with pytest.raises(ValueError):
        arithmetic_chain(41)
    with pytest.raises(ValueError):
        unwind_frames(13)


def test_scaling_inventory_does_not_reuse_original_source_bytes():
    work = workloads(holdout=8, scaling=True)
    assert len(work) == 53
    assert sum(len(item.arguments) for item in work) == 238
    assert sum(item.split == "scaling" for item in work) == 26


def test_scaling_sources_are_accepted_by_independent_wasmtime():
    wasmtime = pytest.importorskip("wasmtime")
    engine = wasmtime.Engine()
    for program, _, _ in scaling_corpus().values():
        wasmtime.Module.validate(engine, encode_control(program))


def test_fault_campaign_explicit_inventory_and_active_targets():
    assert len(ATTACKS) == 10
    assert len({attack.name for attack in ATTACKS}) == len(ATTACKS)
    for attack in ATTACKS:
        assert attack.engine in ("v57", "v58")
        assert attack.source in ("countdown", "generated")
        assert attack.opcode_activated


@pytest.mark.parametrize("index", range(10))
def test_every_corruption_reaches_relevant_source_instruction(index):
    result = run_attack(ATTACKS[index])
    assert result["opcode_activated"]
    assert result["source_program_sha256"]
    assert isinstance(result["detected"], bool)
    assert (result["exception"] is not None
            or result["mutated_trace_sha256"] is not None)


def test_campaign_never_drops_undetected_faults():
    result = campaign()
    assert result["injected_faults"] == len(ATTACKS)
    assert result["detected_faults"] + len(result["undetected_faults"]) == len(ATTACKS)
    assert len(result["cases"]) == len(ATTACKS)
