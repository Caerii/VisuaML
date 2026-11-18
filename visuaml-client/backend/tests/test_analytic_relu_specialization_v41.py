"""v41 real WASM vs analytically synthesized neural circuit vs manifest-free inversion."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
from visuaml.compiler_lab.analytic_relu_specialization_v41 import (
    COUNT,
    HIDDEN,
    HIGH,
    LOW,
    AnalyticRelu,
    NeuralSpecializationError,
    compile_bounded_program,
    decompile_function_from_weights,
    evaluate_bounded,
    permute_hidden_neurons,
)
from visuaml.compiler_lab.wasm_state_vm_v40 import (
    I32Program,
    VMOperation,
    encode_program,
    execute_program,
)


def quadratic() -> I32Program:
    # Signed i32 f(x) = x*x - 3*x + 7, which does not overflow on i8 inputs.
    op = lambda name, value=None: VMOperation(name, value)
    return I32Program(0, (
        op("local.get", 0), op("local.get", 0), op("i32.mul"),
        op("local.get", 0), op("i32.const", 3), op("i32.mul"),
        op("i32.sub"), op("i32.const", 7), op("i32.add"), op("end"),
    ))


def test_neural_weights_are_analytically_constructed_and_exact_on_finite_domain():
    source = quadratic()
    net, receipt = compile_bounded_program(source)
    recovered = decompile_function_from_weights(net)
    assert receipt["integer_domain_exactly_verified"]
    assert receipt["manifest_free_function_extraction"]
    assert receipt["source_program_decompiled"] is False
    assert receipt["universal_neural_interpreter"] is False
    assert receipt["all_i32_inputs_proved"] is False
    assert receipt["program_sha256"] == source.digest()
    assert net.w_in.shape == (HIDDEN, 1)
    assert net.w_out.shape == (1, HIDDEN)
    assert recovered.domain == (LOW, HIGH)
    assert not recovered.manifest_used
    for x in range(LOW, HIGH + 1):
        expected = x * x - 3 * x + 7
        assert execute_program(source, x).value == expected
        assert evaluate_bounded(net, x) == expected
        assert recovered.outputs[x - LOW] == expected
    assert len(recovered.outputs) == COUNT
    assert len(recovered.piecewise_slopes) == COUNT - 1
    assert len(recovered.function_sha256) == 64


def test_hidden_neuron_reordering_preserves_function_but_changes_tensor_bytes():
    net, _ = compile_bounded_program(quadratic())
    rng = np.random.default_rng(41)
    order = tuple(int(x) for x in rng.permutation(HIDDEN))
    other = permute_hidden_neurons(net, order)
    assert hashlib.sha256(net.b_in.tobytes()).hexdigest() != hashlib.sha256(
        other.b_in.tobytes()
    ).hexdigest()
    # Deliberately throw away program identity. This is function recovery.
    manifestless = AnalyticRelu(other.w_in, other.b_in, other.w_out, other.b_out)
    got = decompile_function_from_weights(manifestless)
    assert not got.manifest_used
    assert got.outputs == decompile_function_from_weights(net).outputs
    for x in (-128, -51, -1, 0, 1, 17, 126, 127):
        assert evaluate_bounded(other, x) == evaluate_bounded(net, x)
    inverse = np.argsort(np.asarray(order))
    np.testing.assert_array_equal(other.w_in[inverse], net.w_in)
    np.testing.assert_array_equal(other.b_in[inverse], net.b_in)
    np.testing.assert_array_equal(other.w_out[:, inverse], net.w_out)


def test_neural_model_is_not_an_unbounded_virtual_machine():
    net, _ = compile_bounded_program(quadratic())
    with pytest.raises(NeuralSpecializationError, match="finite i8 domain"):
        evaluate_bounded(net, 128)
    # Outside the certified domain, the network extrapolates affinely, not
    # as the quadratic program. This is a concrete falsifying witness.
    x = 128
    extrapolation = float(
        (net.w_out @ np.maximum(0, net.w_in @ np.array([float(x)]) + net.b_in))[0]
        + net.b_out[0]
    )
    assert int(extrapolation) != execute_program(quadratic(), x).value


def test_malformed_or_reweighted_analytic_circuits_fail_or_change_behavior():
    net, _ = compile_bounded_program(quadratic())
    bad_biases = net.b_in.copy()
    bad_biases[0] = bad_biases[1]
    corrupted = AnalyticRelu(net.w_in, bad_biases, net.w_out, net.b_out)
    with pytest.raises(NeuralSpecializationError, match="hinge"):
        decompile_function_from_weights(corrupted)
    invalid_output = net.w_out.copy()
    invalid_output[0, 1] += 1
    alternative = AnalyticRelu(net.w_in, net.b_in, invalid_output, net.b_out)
    original = decompile_function_from_weights(net)
    changed = decompile_function_from_weights(alternative)
    assert changed.function_sha256 != original.function_sha256
    assert changed.outputs != original.outputs
    with pytest.raises(NeuralSpecializationError, match="bijective"):
        permute_hidden_neurons(net, tuple(0 for _ in range(HIDDEN)))
    with pytest.raises(NeuralSpecializationError, match="exact-integer"):
        decompile_function_from_weights(
            AnalyticRelu(net.w_in.astype(np.float32), net.b_in, net.w_out, net.b_out)
        )


def test_real_wasm_in_wasmtime_matches_neural_exactly_for_every_i8_input():
    wasmtime = pytest.importorskip("wasmtime")
    source = quadratic()
    raw = encode_program(source)
    engine = wasmtime.Engine()
    wasmtime.Module.validate(engine, raw)
    store = wasmtime.Store(engine)
    module = wasmtime.Module(engine, raw)
    instance = wasmtime.Instance(store, module, [])
    execute_wasm = instance.exports(store)["run"]
    net, receipt = compile_bounded_program(source)
    recovered = decompile_function_from_weights(AnalyticRelu(
        net.w_in.copy(), net.b_in.copy(), net.w_out.copy(), net.b_out.copy()
    ))
    for x in range(LOW, HIGH + 1):
        wasm = execute_wasm(store, x)
        neural = evaluate_bounded(net, x)
        typed = execute_program(source, x).value
        decompiled = recovered.outputs[x - LOW]
        assert wasm == neural == typed == decompiled
    assert not receipt["wasmtime_executed_by_this_function"]
