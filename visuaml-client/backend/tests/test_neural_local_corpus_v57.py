"""Adversarial v57 local-bank and deterministic WASM corpus tests."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from visuaml.compiler_lab.bit_neural_vm_v49 import _bits, _int
from visuaml.compiler_lab.neural_local_bank_v57 import (
    MODES,
    apply_local_microop,
    construct_local_bank_weights,
)
from visuaml.compiler_lab.typed_microop_ir_v57 import (
    LOCALS_WIDTH,
    MicroopAdmissionError,
    compile_microop_tokens,
    validate_microop_tokens,
)
from visuaml.compiler_lab.wasm_control_flow_v51 import (
    decode_control,
    encode_control,
)
from visuaml.compiler_lab.wasm_research_corpus_v57 import corpus


def hot(index: int, width: int) -> np.ndarray:
    result = np.zeros(width, dtype=np.float64)
    result[index] = 1.
    return result


def test_one_hot_local_bank_get_set_tee_are_exact_and_isolated():
    bank = np.zeros((LOCALS_WIDTH, 32), dtype=np.float64)
    bank[0] = _bits(-(1 << 31))
    bank[1] = _bits(7)
    select = hot(1, LOCALS_WIDTH)
    current = bank.copy()
    got, value = apply_local_microop(
        bank, select, hot(MODES.index("get"), len(MODES)),
        _bits(123),
    )
    assert np.array_equal(got, current)
    assert _int(value) == 7
    updated, output = apply_local_microop(
        got, select, hot(MODES.index("set"), len(MODES)),
        _bits((1 << 31) - 1),
    )
    assert not output.any()
    assert _int(updated[1]) == (1 << 31) - 1
    assert _int(updated[0]) == -(1 << 31)
    updated, output = apply_local_microop(
        updated, select, hot(MODES.index("tee"), len(MODES)),
        _bits(-17),
    )
    assert not output.any()
    assert _int(updated[1]) == -17
    assert _int(updated[0]) == -(1 << 31)


def test_local_mode_and_weight_ablation_fail_closed():
    bank = np.zeros((LOCALS_WIDTH, 32), dtype=np.float64)
    select = hot(0, LOCALS_WIDTH)
    invalid = hot(MODES.index("get"), len(MODES))
    invalid[2] = 1.
    with pytest.raises(MicroopAdmissionError):
        apply_local_microop(bank, select, invalid, _bits(0))
    fixed = construct_local_bank_weights()
    broken = fixed.read_gain.copy()
    broken[0] = 0.
    with pytest.raises(MicroopAdmissionError):
        apply_local_microop(
            bank, select, hot(MODES.index("get"), len(MODES)),
            _bits(0), fixed=replace(fixed, read_gain=broken),
        )
    bad_index = select.copy()
    bad_index[2] = 1.
    with pytest.raises(MicroopAdmissionError):
        apply_local_microop(
            bank, bad_index, hot(MODES.index("set"), len(MODES)),
            _bits(2),
        )


def test_corpus_has_19_unique_canonical_source_wasm_modules():
    cases = corpus()
    assert len(cases) == 19
    assert sum(len(samples) for _, samples in cases.values()) == 138
    signatures = set()
    for program, samples in cases.values():
        raw = encode_control(program)
        assert decode_control(raw) == program
        tokens = compile_microop_tokens(program)
        validate_microop_tokens(tokens, program)
        assert samples
        signatures.add(tokens.source_wasm_sha256)
    assert len(signatures) == len(cases)


def test_real_wasmtime_validates_every_new_source_program():
    wasmtime = pytest.importorskip("wasmtime")
    engine = wasmtime.Engine()
    for program, _ in corpus().values():
        wasmtime.Module.validate(engine, encode_control(program))
