"""Compile the canonical v49 float64 gate equations into one CPU call."""
from __future__ import annotations

import ctypes
import hashlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .bit_neural_vm_v49 import WIDTH, BitCircuitWeights, BitVMAdmissionError, construct_bit_weights
from .exact_binary import is_exact_binary

_FLAGS = ("-std=c++17", "-O3", "-fno-fast-math", "-ffp-contract=off", "-shared", "-fPIC")
_P = ctypes.c_void_p
_NAMES = ("and_kernel", "xor_kernel", "carry_thresholds", "adder_thresholds")


@dataclass(frozen=True)
class CompiledAlu:
    function: object
    source_sha256: str
    binary_sha256: str
    compiler: str
    flags: tuple[str, ...]


@lru_cache(maxsize=1)
def compiled_alu():
    source = Path(__file__).with_suffix(".cpp")
    compiler = shutil.which("c++")
    if compiler is None:
        raise RuntimeError("Constructed ALU compilation requires a C++17 compiler")
    version = subprocess.check_output([compiler, "--version"], text=True, timeout=10).splitlines()[0]
    with tempfile.TemporaryDirectory(prefix="visuaml-constructed-alu-") as directory:
        binary = Path(directory) / "alu.so"
        subprocess.run([compiler, *_FLAGS, str(source), "-o", str(binary)], check=True, timeout=30)
        binary_id = hashlib.sha256(binary.read_bytes()).hexdigest()
        function = ctypes.CDLL(str(binary)).visuaml_alu_candidates
        function.argtypes = (_P, _P, _P, _P, _P, ctypes.c_size_t, _P, _P)
        function.restype = None
    return CompiledAlu(function, hashlib.sha256(source.read_bytes()).hexdigest(),
                       binary_id, version, _FLAGS)


@lru_cache(maxsize=1)
def _canonical_coefficients():
    fixed = construct_bit_weights()
    return tuple(getattr(fixed, name) for name in _NAMES)


def evaluate_candidates(left, right, fixed: BitCircuitWeights) -> np.ndarray:
    """Evaluate all six candidates, with Python retaining binary admission and selection."""
    if type(fixed) is not BitCircuitWeights:
        raise BitVMAdmissionError("Compiled ALU requires canonical v49 arithmetic coefficients")
    coefficients = []
    for name, expected in zip(_NAMES, _canonical_coefficients(), strict=True):
        actual = getattr(fixed, name)
        if type(actual) is not np.ndarray or actual.dtype != np.float64 or actual.shape != expected.shape:
            raise BitVMAdmissionError("Compiled ALU requires canonical v49 arithmetic coefficients")
        snapshot = actual.copy(order="C")
        if not np.array_equal(snapshot, expected):
            raise BitVMAdmissionError("Compiled ALU requires canonical v49 arithmetic coefficients")
        coefficients.append(snapshot)
    a, b = (np.array(value, copy=True, order="C", subok=False) for value in (left, right))
    if any(value.dtype != np.float64 or value.shape != (WIDTH,) or not is_exact_binary(value)
           for value in (a, b)):
        raise BitVMAdmissionError("Compiled ALU requires exact binary i32 inputs")
    and_weight, xor_weight, carry, adder = coefficients
    candidates = np.zeros((6, WIDTH), dtype=np.float64)
    compiled_alu().function(
        a.ctypes.data, b.ctypes.data, and_weight.ctypes.data, xor_weight.ctypes.data,
        carry.ctypes.data, carry.size, adder.ctypes.data, candidates.ctypes.data,
    )
    return candidates
