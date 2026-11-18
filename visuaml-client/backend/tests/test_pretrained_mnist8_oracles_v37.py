"""Independent scalar convolution/pooling and fail-closed geometry tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scripts.probe_pretrained_mnist8_oracles_v37 import (
    _padding,
    _reshape_onnx,
    conv2d_scalar,
    maxpool2d_scalar,
)


def test_scalar_conv_matches_torch_on_two_channel_geometry():
    rng = np.random.default_rng(410)
    x = rng.normal(size=(1, 2, 7, 7)).astype(np.float32) * 0.2
    w = rng.normal(size=(3, 2, 5, 5)).astype(np.float32) * 0.2
    pad = _padding((7, 7), (5, 5), (1, 1), "SAME_UPPER", (0, 0, 0, 0))
    oracle = conv2d_scalar(x, w, kernel=(5, 5), stride=(1, 1), pads=pad)
    torch_result = F.conv2d(torch.from_numpy(x), torch.from_numpy(w), padding=2).numpy()
    np.testing.assert_allclose(oracle, torch_result, rtol=1e-5, atol=1e-5)


def test_maxpool_matches_independent_torch():
    x = np.random.default_rng(99).normal(size=(1, 2, 9, 9)).astype(np.float32)
    result = maxpool2d_scalar(x, kernel=(3, 3), stride=(3, 3), pads=(0, 0, 0, 0))
    expected = F.max_pool2d(torch.from_numpy(x), kernel_size=3, stride=3).numpy()
    np.testing.assert_array_equal(result, expected)


def test_invalid_padding_rejected():
    with pytest.raises(ValueError, match="padding"):
        _padding((7, 7), (3, 3), (1, 1), "SAME_LOWER", (0, 0, 0, 0))
    with pytest.raises(ValueError, match="padding"):
        _padding((7, 7), (3, 3), (1, 1), "NOTSET", (0, 0, 0, 100))


def test_reshape_is_static_and_never_changes_values():
    a = np.arange(256, dtype=np.float32).reshape(1, 16, 4, 4)
    b = _reshape_onnx(a, np.array([1, -1], dtype=np.int64))
    assert b.shape == (1, 256)
    np.testing.assert_array_equal(a.ravel(), b.ravel())
    with pytest.raises(ValueError, match="reshape dimension"):
        _reshape_onnx(a, np.array([-1, -1], dtype=np.int64))
