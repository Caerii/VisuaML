from dataclasses import fields, replace
import shutil

import numpy as np
import pytest

from visuaml.compiler_lab.bit_neural_vm_v49 import _bits, construct_bit_weights
from visuaml.compiler_lab.compact_routing_v58 import CompactRouter
from visuaml.compiler_lab.neural_control_routing_v52 import (
    TERMINAL, NeuralControlError, _onehot, compile_route_tokens, construct_router_weights,
)
from visuaml.compiler_lab.neural_microstep_executor_v58 import execute_microstep_control
from visuaml.compiler_lab.typed_microop_ir_v57 import (
    EFFECTS, OPCODES, IndexedMicroops, MicroopAdmissionError, compile_microop_tokens,
    construct_microop_weights, read_microop,
)
from visuaml.compiler_lab.wasm_research_corpus_v57 import corpus, countdown, generated

requires_compiler = pytest.mark.skipif(shutil.which("c++") is None, reason="requires C++17")


def test_indexed_decoder_matches_all_reviewed_rows_and_immutable_admission():
    for program, _ in corpus().values():
        tokens = compile_microop_tokens(program)
        fixed = construct_microop_weights()
        prepared = IndexedMicroops(program, tokens, fixed)
        for pc in range(tokens.length):
            row = prepared.read(pc)
            expected = read_microop(_onehot(pc), tokens, fixed)
            for field in fields(row):
                value = getattr(row, field.name)
                np.testing.assert_array_equal(value, getattr(expected, field.name))
                assert not value.flags.writeable
        for invalid in (False, -1, tokens.length, 1.0):
            with pytest.raises(MicroopAdmissionError, match="Inactive microop"):
                prepared.read(invalid)

    corrupted = tokens.opcode_rows.copy()
    corrupted[0] = np.roll(corrupted[0], 1)
    with pytest.raises(MicroopAdmissionError, match="differ from canonical"):
        IndexedMicroops(program, replace(tokens, opcode_rows=corrupted), fixed)


def test_indexed_decoder_reads_altered_fixed_rows_and_rejects_corruption():
    program = generated(2)
    tokens = compile_microop_tokens(program)
    fixed = construct_microop_weights()
    pc = next(i for i, op in enumerate(program.operations) if op.opcode == "i32.mul")
    effects = fixed.effects.copy()
    effects[OPCODES.index("i32.mul")] = 0.
    effects[OPCODES.index("i32.mul"), EFFECTS.index("retain")] = 1.
    changed = replace(fixed, effects=effects)
    decoder = IndexedMicroops(program, tokens, changed)
    expected = read_microop(_onehot(pc), tokens, changed)
    effects[:] = 0.  # Later mutation cannot invalidate an admitted snapshot.
    np.testing.assert_array_equal(decoder.read(pc).effect, expected.effect)
    bad = IndexedMicroops(program, tokens, changed)
    with pytest.raises(MicroopAdmissionError, match="one-hot"):
        bad.read(pc)


@requires_compiler
def test_index_route_uses_live_zero_test_and_path_gains():
    program = countdown()
    tokens = compile_route_tokens(program)
    weights = construct_router_weights()
    bit = construct_bit_weights()
    pc = int(np.flatnonzero(tokens.guard_kind[:, 1])[0])
    router = CompactRouter(program, tokens, weights, bit)
    for index in (*range(tokens.instruction_count), TERMINAL):
        for argument in (0, 1, -(1 << 31)):
            vector, details = router.step(_onehot(index), _bits(argument))
            assert router.step_index(index, _bits(argument)) == (int(vector.argmax()), details)

    modified = CompactRouter(program, tokens, weights, replace(bit, and_kernel=np.zeros(3)))
    assert modified.step_index(pc, _bits(0))[0] != router.step_index(pc, _bits(0))[0]
    ablated = CompactRouter(program, tokens, replace(weights, path_weights=np.array([1., 0.])), bit)
    with pytest.raises(NeuralControlError, match="lost unique"):
        ablated.step_index(pc, _bits(1))
    with pytest.raises(NeuralControlError, match="Invalid active PC"):
        router.step_index(-1, _bits(1))


@requires_compiler
@pytest.mark.parametrize("trace", (True, False))
def test_indexed_executor_retains_state_and_provenance(trace):
    program = generated(2)
    options = dict(capture_trace=trace, use_compact_routing=True,
                   use_fused_state=True, use_compiled_alu=True)
    assert execute_microstep_control(program, -17, use_indexed_pc=True, **options) == (
        execute_microstep_control(program, -17, **options)
    )
