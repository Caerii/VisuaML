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
    route: object
    indexed_route: object
    data_microstep: object
    prepared_loop: object
    prepared_batch: object
    prepared_simd_available: object
    structured_batch: object
    frame_event: object
    frame_event_retain: object
    prepared_frame_event_retain: object
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
        library = ctypes.CDLL(str(binary))
        function = library.visuaml_alu_candidates
        function.argtypes = (_P, _P, _P, _P, _P, ctypes.c_size_t, _P, _P)
        function.restype = None
        route = library.visuaml_compact_route
        route.argtypes = (_P, ctypes.c_size_t, _P, _P, _P, _P, _P, _P, _P)
        route.restype = ctypes.c_int
        indexed_route = library.visuaml_indexed_route
        indexed_route.argtypes = (ctypes.c_size_t, *route.argtypes[1:])
        indexed_route.restype = ctypes.c_int
        data_microstep = library.visuaml_data_microstep
        data_microstep.argtypes = (ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, _P, _P, _P, _P, _P)
        data_microstep.restype = ctypes.c_int
        prepared_loop = library.visuaml_prepared_loop
        prepared_loop.argtypes = (ctypes.c_int32, *((ctypes.c_size_t,) * 4), *((_P,) * 10), ctypes.c_bool)
        prepared_loop.restype = ctypes.c_int
        prepared_batch = library.visuaml_prepared_batch
        prepared_batch.argtypes = (_P, _P, *((ctypes.c_size_t,) * 5), *((_P,) * 11), ctypes.c_bool, ctypes.c_bool)
        prepared_batch.restype = None
        prepared_simd_available = library.visuaml_prepared_simd_available
        prepared_simd_available.argtypes = ()
        prepared_simd_available.restype = ctypes.c_bool
        structured_batch = library.visuaml_structured_batch
        structured_batch.argtypes = (_P, _P, *((ctypes.c_size_t,) * 5), *((_P,) * 20))
        structured_batch.restype = None
        frame_event = library.visuaml_frame_event_gate
        frame_event.argtypes = (_P, *((ctypes.c_bool,) * 4), _P)
        frame_event.restype = ctypes.c_int
        frame_event_retain = library.visuaml_frame_event_retain
        frame_event_retain.argtypes = (*frame_event.argtypes, _P, _P, ctypes.c_size_t)
        frame_event_retain.restype = ctypes.c_int
        prepared_frame_event_retain = library.visuaml_prepared_frame_event_retain
        prepared_frame_event_retain.argtypes = (_P, *((ctypes.c_bool,) * 3), _P, _P, _P, ctypes.c_size_t)
        prepared_frame_event_retain.restype = ctypes.c_int
    return CompiledAlu(function, route, indexed_route, data_microstep, prepared_loop, prepared_batch,
                       prepared_simd_available, structured_batch, frame_event, frame_event_retain, prepared_frame_event_retain,
                       hashlib.sha256(source.read_bytes()).hexdigest(),
                       binary_id, version, _FLAGS)


@lru_cache(maxsize=1)
def _canonical_coefficients():
    fixed = construct_bit_weights()
    return tuple(getattr(fixed, name) for name in _NAMES)


def compiled_coefficients(fixed: BitCircuitWeights) -> tuple[np.ndarray, ...]:
    """Snapshot the coefficient arrays admitted by the compiled bit gates."""
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
    return tuple(coefficients)


def evaluate_candidates(left, right, fixed: BitCircuitWeights) -> np.ndarray:
    """Evaluate all six candidates, with Python retaining binary admission and selection."""
    and_weight, xor_weight, carry, adder = compiled_coefficients(fixed)
    a, b = (np.array(value, copy=True, order="C", subok=False) for value in (left, right))
    if any(value.dtype != np.float64 or value.shape != (WIDTH,) or not is_exact_binary(value)
           for value in (a, b)):
        raise BitVMAdmissionError("Compiled ALU requires exact binary i32 inputs")
    candidates = np.zeros((6, WIDTH), dtype=np.float64)
    compiled_alu().function(
        a.ctypes.data, b.ctypes.data, and_weight.ctypes.data, xor_weight.ctypes.data,
        carry.ctypes.data, carry.size, adder.ctypes.data, candidates.ctypes.data,
    )
    return candidates
