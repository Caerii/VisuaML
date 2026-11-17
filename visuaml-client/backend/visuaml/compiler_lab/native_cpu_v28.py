"""v28: independently compiled native C++ CPU target for canonical Llama IR.

A trusted static C++17 source implements every decoder mathematical operation;
PyTorch, TorchScript and NumPy do NOT execute the native model's arithmetic.
Python independently validates the checkpoint and executable, supplies
contiguous tensor buffers, checks outputs and commits transactional KV state.

The backend compiles locally with an installed C++ compiler. It does NOT
download compilers or accept user-controlled C++ source. The resulting library
is a temporary process-owned native binary, NOT a portable signed AOT artifact,
a sandbox for hostile C++ callers, a production high-performance kernel, or a
GPU / photonic implementation.
"""

from __future__ import annotations

import ctypes
import hashlib
import platform
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .family_blocks_v12 import KVState
from .kv_lowering_v17 import (
    KVSnapshot,
    lower_kv_semantic_program,
    make_kv_semantic_program,
)
from .llama_semantic_v20 import (
    LlamaSemanticError,
    LlamaSemanticPlan,
    lower_llama_semantic_program,
)
from .stacked_decoder_v14 import StackState

SCHEMA = "visuaml.native-cpu-llama.v28"
_SOURCE = Path(__file__).with_suffix(".cpp")
_P = ctypes.c_void_p
_PP = ctypes.POINTER(_P)


class NativeCpuError(LlamaSemanticError):
    """Invalid native lowering, incompatible compiled ABI or runtime result."""


class _Config(ctypes.Structure):
    _fields_ = [
        ("batch", ctypes.c_int),
        ("layers", ctypes.c_int),
        ("hidden", ctypes.c_int),
        ("heads", ctypes.c_int),
        ("kv_heads", ctypes.c_int),
        ("head_dim", ctypes.c_int),
        ("inner", ctypes.c_int),
        ("vocabulary", ctypes.c_int),
        ("capacity", ctypes.c_int),
        ("epsilon", ctypes.c_float),
        ("rope_theta", ctypes.c_double),
        ("tied_head", ctypes.c_int),
    ]


@dataclass(frozen=True)
class NativeCompilerEvidence:
    schema: str
    semantic_sha256: str
    checkpoint_sha256: str
    cpp_source_sha256: str
    executable_sha256: str
    compiler_identity: str
    target: str
    compilation_ms: float
    independent_max_abs_error: float


def compiler_available() -> bool:
    """Native compilation is optional and never fetched automatically."""
    return any(shutil.which(command) for command in ("c++", "g++", "clang++"))


def _compiler() -> str:
    for command in ("c++", "g++", "clang++"):
        found = shutil.which(command)
        if found:
            return found
    raise NativeCpuError("A trusted C++17 compiler is required for the native backend")


def _tensor_names(plan: LlamaSemanticPlan) -> tuple[str, ...]:
    p = plan.weights.profile
    names = ["model.embed_tokens.weight", "model.norm.weight"]
    if not p.tied_head:
        names.append("lm_head.weight")
    for index in range(p.layers):
        prefix = f"model.layers.{index}."
        names.extend(
            prefix + name
            for name in (
                "input_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "post_attention_layernorm.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            )
        )
    return tuple(names)


class NativeCpuLlamaCompiler:
    """Source-bound compiler for one validated Llama program and weight layout."""

    def __init__(self, plan: LlamaSemanticPlan, *, reference_kernels: bool = False):
        if type(reference_kernels) is not bool:
            raise NativeCpuError("reference_kernels must be a boolean")
        self.reference_kernels = reference_kernels
        if not isinstance(plan, LlamaSemanticPlan):
            raise NativeCpuError("Expected a canonical Llama semantic plan")
        self.plan = lower_llama_semantic_program(
            plan.program, plan.weights, batch=plan.batch
        )
        p = self.plan.weights.profile
        self._config = _Config(
            self.plan.batch,
            p.layers,
            p.hidden,
            p.heads,
            p.kv_heads,
            p.head_dim,
            p.intermediate,
            p.vocabulary,
            self.plan.capacity,
            p.eps,
            p.rope_theta,
            int(p.tied_head),
        )
        names = _tensor_names(self.plan)
        if set(names) != set(self.plan.weights.arrays):
            raise NativeCpuError("Native ABI weight set disagrees with canonical checkpoint")
        self._weight_arrays = tuple(
            np.ascontiguousarray(self.plan.weights.arrays[name])
            for name in names
        )
        self._weight_ptrs = (_P * len(names))(
            *(array.ctypes.data for array in self._weight_arrays)
        )
        if not _SOURCE.is_file():
            raise NativeCpuError("Trusted native C++ source is missing")
        cpp = _SOURCE.read_bytes()
        self._temporary = tempfile.TemporaryDirectory(
            prefix="visuaml-verified-native-"
        )
        compiler = _compiler()
        root = Path(self._temporary.name)
        target = root / (
            "libvisuaml_native.dylib" if platform.system() == "Darwin"
            else "libvisuaml_native.so"
        )
        started = time.perf_counter_ns()
        args = [
            compiler, "-std=c++17", "-O2", "-fPIC", "-shared",
            "-fno-fast-math",
        ]
        if self.reference_kernels:
            args.append("-DVISUAML_NATIVE_SCALAR_REFERENCE=1")
        args.extend((str(_SOURCE), "-o", str(target)))
        try:
            result = subprocess.run(
                args, capture_output=True, text=True, check=False, timeout=90,
            )
            identity = subprocess.run(
                [compiler, "--version"],
                capture_output=True, text=True, check=True, timeout=10,
            ).stdout.splitlines()[0][:120]
        except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            raise NativeCpuError("Trusted native compiler failed or timed out") from exc
        if result.returncode != 0 or not target.is_file():
            raise NativeCpuError(
                "Native compiler rejected the C++17 Llama dialect: "
                + result.stderr[:1000]
            )
        elapsed = (time.perf_counter_ns() - started) / 1e6
        binary_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        try:
            self._library = ctypes.CDLL(str(target))
        except OSError as exc:
            raise NativeCpuError("Compiled native library failed to load") from exc
        self._decode = self._library.visuaml_native_llama_decode
        self._decode.restype = ctypes.c_int
        self._decode.argtypes = [
            ctypes.POINTER(_Config),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int,
            ctypes.c_int,
            _PP, _PP, _PP, ctypes.POINTER(ctypes.c_float),
        ]
        self._evidence = NativeCompilerEvidence(
            SCHEMA,
            self.plan.semantic_sha256,
            self.plan.weights.checkpoint_sha256,
            hashlib.sha256(cpp).hexdigest(),
            binary_hash,
            identity,
            platform.machine(),
            round(elapsed, 6),
            0.0,
        )
        # The compiler may only be published after independent full-output
        # and every-layer cache numerical witnesses succeed.
        reference = self.plan.new_session(backend="numpy")
        native = self.new_session()
        worst = 0.0
        for length in (3, 1):
            tokens = np.arange(1, length + 1, dtype=np.int64)[None, :]
            tokens = np.repeat(tokens, self.plan.batch, axis=0)
            expected = reference.step(tokens)
            actual = native.step(tokens)
            try:
                np.testing.assert_allclose(
                    actual, expected, atol=1e-4, rtol=1e-4
                )
                for a, b in zip(native.snapshot().layers, reference.snapshot().layers):
                    np.testing.assert_allclose(
                        a.keys, b.keys, atol=1e-4, rtol=1e-4
                    )
                    np.testing.assert_allclose(
                        a.values, b.values, atol=1e-4, rtol=1e-4
                    )
                    worst = max(
                        worst, float(np.max(np.abs(a.keys - b.keys))),
                        float(np.max(np.abs(a.values - b.values))),
                    )
            except AssertionError as exc:
                raise NativeCpuError(
                    "Independent whole-model numerical certification failed"
                ) from exc
            worst = max(worst, float(np.max(np.abs(actual - expected))))
        self._evidence = NativeCompilerEvidence(
            *list(asdict(self._evidence).values())[:-1], worst
        )

    def new_session(self) -> NativeCpuLlamaSession:
        return NativeCpuLlamaSession(self)

    def report(self) -> dict:
        return {
            **asdict(self._evidence),
            "linear_kernel": (
                "scalar_reference" if self.reference_kernels else "four_row"
            ),
        }


class NativeCpuLlamaSession:
    """Private, transactional native KV state; never safe for concurrent step()."""

    def __init__(self, compiler: NativeCpuLlamaCompiler):
        self._compiler = compiler
        plan = compiler.plan
        p = plan.weights.profile
        self.batch = plan.batch
        self.capacity = plan.capacity
        self.vocabulary = p.vocabulary
        self.layers = p.layers
        self.kv_heads = p.kv_heads
        self.head_dim = p.head_dim
        self.cache = lower_kv_semantic_program(
            make_kv_semantic_program(
                batch=self.batch, layers=p.layers,
                kv_heads=p.kv_heads, head_dim=p.head_dim,
                capacity=self.capacity,
            )
        ).new_session()

    @property
    def cursor(self) -> int:
        return self.cache.cursor

    def step(self, tokens: np.ndarray) -> np.ndarray:
        if (
            not isinstance(tokens, np.ndarray) or tokens.dtype != np.int64
            or tokens.ndim != 2 or tokens.shape[0] != self.batch
            or not 1 <= tokens.shape[1] <= 32
            or self.cursor + tokens.shape[1] > self.capacity
            or np.any(tokens < 0) or np.any(tokens >= self.vocabulary)
        ):
            raise NativeCpuError("Native token shape, IDs, batch or context budget invalid")
        tokens = np.ascontiguousarray(tokens)
        query = tokens.shape[1]
        past = self.cursor
        held: list[np.ndarray] = []
        output_arrays: list[np.ndarray] = []
        for layer in range(self.layers):
            for kind in ("k", "v"):
                # Detached copies protect the committed NumPy prefix from
                # any native write, including a failed kernel invocation.
                prior = self.cache._arrays[f"{kind}{layer}"][:, :past].copy(order="C")
                held.append(prior)
                output_arrays.append(
                    np.empty((self.batch, query, self.kv_heads, self.head_dim), np.float32)
                )
        old_ptrs = (_P * len(held))(*(array.ctypes.data for array in held))
        new_ptrs = (_P * len(output_arrays))(
            *(array.ctypes.data for array in output_arrays)
        )
        logits = np.empty((self.batch, query, self.vocabulary), dtype=np.float32)
        result = self._compiler._decode(
            ctypes.byref(self._compiler._config),
            tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            query, past, self._compiler._weight_ptrs,
            old_ptrs, new_ptrs,
            logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if result != 0:
            raise NativeCpuError("Native C++ decoder rejected or failed: code " + str(result))
        if not np.isfinite(logits).all() or any(
            not np.isfinite(array).all() for array in output_arrays
        ):
            raise NativeCpuError("Native compiler produced nonfinite logits or KV")
        with self.cache.begin_append(query) as tx:
            for layer in range(self.layers):
                tx.stage(
                    layer, output_arrays[2 * layer], output_arrays[2 * layer + 1]
                )
            tx.commit()
        return logits

    def reset(self) -> None:
        self.cache.reset()

    def snapshot(self) -> StackState:
        state = dict(self.cache.snapshot().buffers)
        return StackState(tuple(
            KVState(state[f"k{layer}"], state[f"v{layer}"], self.capacity)
            for layer in range(self.layers)
        ))

    def restore(self, state: StackState) -> None:
        if not isinstance(state, StackState) or len(state.layers) != self.layers:
            raise NativeCpuError("Native session snapshot has incompatible layers")
        first = state.layers[0]
        if not isinstance(first, KVState) or not isinstance(first.keys, np.ndarray):
            raise NativeCpuError("Malformed native KV snapshot")
        if first.keys.ndim != 4:
            raise NativeCpuError("Malformed native KV rank")
        length = first.keys.shape[1]
        cells = []
        for layer, item in enumerate(state.layers):
            if not isinstance(item, KVState) or item.max_context != self.capacity:
                raise NativeCpuError("Incompatible cache geometry")
            cells.extend(
                ((f"k{layer}", item.keys), (f"v{layer}", item.values))
            )
        self.cache.restore(
            KVSnapshot(self.cache.plan.semantic_sha256, length, tuple(cells))
        )

    def generate(self, prompt: np.ndarray, *, steps: int) -> tuple[np.ndarray, StackState]:
        if type(steps) is not int or not 1 <= steps <= 24:
            raise NativeCpuError("Invalid generated step count")
        if (
            not isinstance(prompt, np.ndarray) or prompt.ndim != 2
            or prompt.dtype != np.int64
        ):
            raise NativeCpuError("Prompt must be int64 tensor [batch,tokens]")
        if prompt.shape[1] + steps > self.capacity:
            raise NativeCpuError("Generation exceeds cache capacity")
        self.reset()
        logits = self.step(prompt)
        produced = []
        for _ in range(steps):
            ids = np.argmax(logits[:, -1], axis=-1).astype(np.int64)[:, None]
            produced.append(ids)
            logits = self.step(ids)
        return np.concatenate(produced, axis=1), self.snapshot()

    def report(self) -> dict:
        return {
            **self.cache.report(),
            **self._compiler.report(),
            "backend": "native-c++17-cpu",
            "session_state_owned": True,
            "native_math_independent_of_torch": True,
            "limitations": [
                "Scalar reference C++: no SIMD/BLAS or performance-win claim.",
                "Native library compiled locally from trusted source.",
                "No native code sandbox, GPU, optical device or portable artifact.",
            ],
        }
