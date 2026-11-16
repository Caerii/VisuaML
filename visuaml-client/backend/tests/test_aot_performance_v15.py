"""v15 correctness and admission tests for optimized NumPy-AOT execution."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from visuaml.compiler_lab.aot_compile import compile_numpy_aot
from visuaml.compiler_lab.ir_contract import ResourcePolicy
from visuaml.compiler_lab.numpy_runtime import (
    BundleError,
    _execute,
    execute,
    prepare_numpy_aot,
)
from visuaml.compiler_spike import capture


class _SmallProgram(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 5)

    def forward(self, x):
        return torch.relu(self.fc(x))


def _artifact():
    torch.manual_seed(37)
    x = torch.randn(2, 4)
    return compile_numpy_aot(capture(_SmallProgram().eval(), x)), x.numpy()


def test_prepared_matches_legacy_and_is_reusable():
    artifact, x = _artifact()
    prepared = prepare_numpy_aot(artifact.manifest, artifact.constants)
    expected = execute(artifact.manifest, artifact.constants, x)
    for scale in (1.0, -2.0, 0.5, 3.0):
        np.testing.assert_allclose(
            prepared.run(x * np.float32(scale)),
            execute(artifact.manifest, artifact.constants, x * np.float32(scale)),
            atol=0,
            rtol=0,
        )
    np.testing.assert_array_equal(prepared.run(x), expected)


def test_prepared_keeps_immutable_independent_constant_snapshot():
    artifact, x = _artifact()
    prepared = prepare_numpy_aot(artifact.manifest, artifact.constants)
    original = prepared.run(x)
    first_key = next(iter(artifact.constants))
    artifact.constants[first_key][...] = 100
    artifact.manifest['source'] = 'mutated after preparation'
    np.testing.assert_array_equal(prepared.run(x), original)
    with pytest.raises(BundleError, match='digest'):
        prepare_numpy_aot(artifact.manifest, artifact.constants)
    for value in prepared._constants.values():
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.flags.writeable = True


def test_prepared_enforces_input_shape_dtype_and_retained_budget():
    artifact, x = _artifact()
    prepared = prepare_numpy_aot(artifact.manifest, artifact.constants)
    with pytest.raises(BundleError, match='requires'):
        prepared.run(x.astype(np.float64))
    with pytest.raises(BundleError, match='requires'):
        prepared.run(x[:1])
    with pytest.raises(BundleError, match='Expected'):
        prepared.run()
    policy = ResourcePolicy(max_retained_value_bytes=32)
    with pytest.raises(BundleError, match='retained-value'):
        prepare_numpy_aot(artifact.manifest, artifact.constants, policy=policy).run(x)


def test_prepared_bounded_integer_constraints():
    from visuaml.compiler_lab.stacked_decoder_v14 import (
        StackedWeights,
        compile_stacked_chunk,
    )

    w = StackedWeights.seeded(8, layers=1)
    a = compile_stacked_chunk(w, batch=1, query=1, past=0)
    p = prepare_numpy_aot(a.manifest, a.constants)
    _, _, depth, kv = w.validate()
    z = np.zeros((1, 0, kv, depth), np.float32)
    assert len(p.run(np.array([[3]], np.int64), z, z)) == 3
    with pytest.raises(BundleError, match='bounded integer'):
        p.run(np.array([[999]], np.int64), z, z)


@pytest.mark.parametrize('batch,ci,co,h,w,kh,kw,stride,padding,dilation,groups', [
    (1, 3, 8, 16, 12, 3, 3, 1, 1, 1, 1),
    (2, 4, 4, 18, 13, 3, 2, (2, 1), (0, 1), (2, 1), 2),
    (1, 6, 6, 15, 19, 2, 2, 1, 0, 1, 6),
    (2, 2, 4, 21, 17, 1, 3, 2, 1, 1, 1),
    (1, 8, 8, 9, 11, 3, 3, 1, 1, 2, 4),
])
def test_im2col_matches_pytorch_conv(batch, ci, co, h, w, kh, kw, stride, padding, dilation, groups):
    gen = np.random.default_rng(33)
    x = gen.normal(size=(batch, ci, h, w)).astype(np.float32)
    kernel = gen.normal(size=(co, ci // groups, kh, kw)).astype(np.float32)
    bias = gen.normal(size=(co,)).astype(np.float32)
    with torch.no_grad():
        reference = F.conv2d(torch.from_numpy(x), torch.from_numpy(kernel), torch.from_numpy(bias),
                             stride=stride, padding=padding, dilation=dilation, groups=groups).numpy()
    optimized = _execute('conv2d', (x, kernel, bias, stride, padding, dilation, groups), {})
    np.testing.assert_allclose(optimized, reference, atol=1.5e-5, rtol=1.5e-5)


def test_non_float32_dtype_uses_original_reference_kernel():
    gen = np.random.default_rng(1)
    x = gen.normal(size=(1, 2, 9, 11)).astype(np.float64)
    w = gen.normal(size=(4, 2, 3, 3)).astype(np.float64)
    out = _execute('conv2d', (x, w, None, 1, 1, 1, 1), {})
    want = F.conv2d(torch.from_numpy(x), torch.from_numpy(w), padding=1).numpy()
    np.testing.assert_allclose(out, want, atol=1e-12, rtol=1e-12)


def test_prepared_fails_closed_on_unknown_op_even_if_unused():
    artifact, _ = _artifact()
    broken = deepcopy(artifact.manifest)
    node = next(x for x in broken['nodes'] if x['kind'] == 'call')
    node['op'] = 'eval_python'
    with pytest.raises(BundleError):
        prepare_numpy_aot(broken, artifact.constants)
