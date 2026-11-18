"""v39 adversarial proofs for restricted WASM and invertible neural layouts."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
from visuaml.compiler_lab.reversible_weight_layout_v39 import (
    LayoutRewriteError,
    permute_relu_hidden,
)
from visuaml.compiler_lab.wasm_bitstream_v39 import (
    MAGIC,
    WasmBinaryError,
    _u32,
    parse_section_envelope,
    relocate_custom_sections,
)


def section(kind: int, payload: bytes) -> bytes:
    return bytes([kind]) + _u32(len(payload)) + payload


def custom(name: str, body: bytes) -> bytes:
    label = name.encode()
    return section(0, _u32(len(label)) + label + body)


def wasm_answer_42() -> bytes:
    """Minimal WASM MVP module exporting fn answer() -> i32 { return 42; }."""
    typ = section(1, bytes([1, 0x60, 0, 1, 0x7F]))
    funcs = section(3, bytes([1, 0]))
    export = section(7, bytes([1, 6]) + b"answer" + bytes([0, 0]))
    body = section(10, bytes([1, 4, 0, 0x41, 42, 0x0B]))
    return (MAGIC + custom("a", b"debug") + typ + custom("z", b"meta")
            + funcs + export + body + custom("large", b"L" * 300))


def _core(data):
    return tuple(s.raw for s in parse_section_envelope(data) if s.section_id != 0)


def test_custom_section_binary_relocation_preserves_every_standard_byte():
    source = wasm_answer_42()
    assert len(parse_section_envelope(source)) == 7
    for placement in ("start", "end"):
        transformed, receipt = relocate_custom_sections(source, placement=placement)
        assert source != transformed
        assert _core(source) == _core(transformed)
        assert receipt["core_bytes_exactly_unchanged"] is True
        assert receipt["source_sha256"] == hashlib.sha256(source).hexdigest()
        assert receipt["target_sha256"] == hashlib.sha256(transformed).hexdigest()
        assert receipt["source_bytes"] == receipt["target_bytes"] == len(source)
        assert receipt["wasm_module_typecheck_or_runtime_validated"] is False
        assert receipt["custom_sections"] == 3
        repeat, _ = relocate_custom_sections(transformed, placement=placement)
        assert repeat == transformed


def test_wasm_sections_fail_closed_under_corruption():
    source = wasm_answer_42()
    for malformed in (
        b"not-wasm",
        source[:-1],
        MAGIC + section(1, b"x") + section(1, b"y"),
        MAGIC + section(10, b"x") + section(1, b"y"),
        MAGIC + b"\x00\x80\x00",
        MAGIC + b"\x00\x80\x80\x80\x80\x10",
        MAGIC + section(0, b"\x01\xff"),
        MAGIC + section(14, b"hello"),
        MAGIC + b"\x01\x7f" + b"truncated",
    ):
        with pytest.raises(WasmBinaryError):
            parse_section_envelope(malformed)
    with pytest.raises(WasmBinaryError):
        relocate_custom_sections(source, placement="middle")


def test_wasmtime_independently_executes_original_and_reordered_binary():
    wasmtime = pytest.importorskip("wasmtime")
    engine = wasmtime.Engine()
    original = wasm_answer_42()
    for candidate in (original, *(
        relocate_custom_sections(original, placement=p)[0]
        for p in ("start", "end")
    )):
        store = wasmtime.Store(engine)
        module = wasmtime.Module(engine, candidate)
        instance = wasmtime.Instance(store, module, [])
        assert instance.exports(store)["answer"](store) == 42


def _integer_forward(w1, b, w2, x):
    hidden = [max(0, sum(int(w1[i, j]) * int(x[j])
                          for j in range(len(x))) + int(b[i]))
              for i in range(w1.shape[0])]
    return [sum(int(w2[i, j]) * hidden[j]
                for j in range(w1.shape[0])) for i in range(w2.shape[0])]


@pytest.mark.parametrize("seed", range(20))
def test_reversible_neuron_reordering_is_algebraically_exact_for_integer_weights(seed):
    rng = np.random.default_rng(seed)
    a = rng.integers(-3, 4, (7, 5)).astype(np.float32)
    b = rng.integers(-2, 3, (7,)).astype(np.float32)
    c = rng.integers(-3, 4, (3, 7)).astype(np.float32)
    x = rng.integers(-3, 4, (5,)).astype(np.int64)
    permutation = tuple(int(y) for y in rng.permutation(7))
    rewritten = permute_relu_hidden(a, b, c, permutation)
    assert _integer_forward(a, b, c, x) == _integer_forward(
        rewritten.first, rewritten.bias, rewritten.second, x
    )
    inverse = np.array(rewritten.inverse_permutation)
    np.testing.assert_array_equal(rewritten.first[inverse], a)
    np.testing.assert_array_equal(rewritten.bias[inverse], b)
    np.testing.assert_array_equal(rewritten.second[:, inverse], c)
    assert rewritten.receipt["weights_recovered_byte_exact"] is True
    assert rewritten.receipt["general_graph_equivalence_proved"] is False


@pytest.mark.parametrize("seed", (1, 2, 27))
def test_finite_float32_reorder_is_numerically_bounded(seed):
    rng = np.random.default_rng(seed)
    w1 = rng.normal(size=(11, 8)).astype(np.float32)
    b1 = rng.normal(size=(11,)).astype(np.float32)
    w2 = rng.normal(size=(4, 11)).astype(np.float32)
    perm = tuple(int(i) for i in rng.permutation(11))
    result = permute_relu_hidden(w1, b1, w2, perm)
    for _ in range(15):
        x = rng.normal(size=(8,)).astype(np.float32)
        expected = w2 @ np.maximum(w1 @ x + b1, np.float32(0))
        actual = result.second @ np.maximum(result.first @ x + result.bias, np.float32(0))
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert not result.receipt["floating_point_output_bitwise_equivalence_proved"]


def test_neural_layout_rewrite_rejects_invalid_permutations_and_nan():
    w1 = np.ones((3, 2), dtype=np.float32)
    b = np.ones(3, dtype=np.float32)
    w2 = np.ones((2, 3), dtype=np.float32)
    for p in ((0, 0, 1), (0, 1), (0, 1, True), (0, 1, 3)):
        with pytest.raises(LayoutRewriteError):
            permute_relu_hidden(w1, b, w2, p)
    with pytest.raises(LayoutRewriteError):
        permute_relu_hidden(w1.astype(np.float64), b, w2, (2, 0, 1))
    with pytest.raises(LayoutRewriteError):
        permute_relu_hidden(w1, b, np.full_like(w2, np.nan), (2, 0, 1))
