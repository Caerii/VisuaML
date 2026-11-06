"""Explicit backend capability contracts and graph-level placement diagnostics.

An adapter is responsible for genuine code generation/execution. A placement
plan alone is NOT evidence of an accelerator kernel or physical deployment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

from visuaml.compiler_spike import CapturedProgram, ProgramIR, UnsupportedOperation
from .aot_compile import AOTArtifact, compile_numpy_aot


@dataclass(frozen=True)
class BackendCapabilities:
    name: str
    revision: str
    native_ops: frozenset[str]
    fallback_ops: frozenset[str]
    supported_dtypes: frozenset[str]
    max_tensor_rank: int | None = None
    description: str = ""
    is_physical_hardware: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.revision:
            raise ValueError("Backend name and revision are required")
        if self.native_ops & self.fallback_ops:
            raise ValueError("Native/fallback operator sets must be disjoint")


@dataclass(frozen=True)
class BackendPlacement:
    name: str
    op: str
    assigned: str
    reason: str


@dataclass(frozen=True)
class PartitionReport:
    backend: str
    revision: str
    placements: tuple[BackendPlacement, ...]
    unsupported: tuple[str, ...]
    estimated_boundary_bytes: int
    unknown_boundary_edges: tuple[str, ...]
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def partition(ir: ProgramIR, capabilities: BackendCapabilities) -> PartitionReport:
    placements: list[BackendPlacement] = []
    locations: dict[str, str] = {}
    bytes_by_name: dict[str, int | None] = {
        n.id: (
            sum(t.bytes for t in n.tensors)
            if n.tensors and all(t.bytes is not None for t in n.tensors)
            else None
        )
        for n in ir.nodes
    }
    unknown_edges: list[str] = []
    unsupported: list[str] = []
    transfer_bytes = 0
    for n in ir.nodes:
        if n.op in {"input", "constant", "output"}:
            assigned, reason = "host", "external state / IO"
        else:
            rank_ok = capabilities.max_tensor_rank is None or all(
                len(t.shape) <= capabilities.max_tensor_rank for t in n.tensors
            )
            dtype_ok = all(t.dtype in capabilities.supported_dtypes for t in n.tensors)
            if (
                (n.op in capabilities.native_ops or "*" in capabilities.native_ops)
                and rank_ok
                and dtype_ok
            ):
                assigned, reason = capabilities.name, "declared native operator"
            elif n.op in capabilities.fallback_ops:
                assigned, reason = "host", "declared digital fallback"
            else:
                assigned, reason = (
                    "unsupported",
                    "missing op or incompatible tensor contract",
                )
                unsupported.append(n.id)
        placements.append(BackendPlacement(n.id, n.op, assigned, reason))
        locations[n.id] = assigned
        for source in n.inputs:
            left = locations.get(source)
            if left is None:
                raise ValueError(f"IR is not topologically ordered at {source}->{n.id}")
            if {left, assigned} == {"host", capabilities.name}:
                count = bytes_by_name.get(source)
                if count is None:
                    unknown_edges.append(f"{source}->{n.id}")
                else:
                    transfer_bytes += count
    return PartitionReport(
        capabilities.name,
        capabilities.revision,
        tuple(placements),
        tuple(unsupported),
        transfer_bytes,
        tuple(unknown_edges),
        (
            "Boundary byte total is a static *tensor-size estimate*, not actual PCIe/optical conversion traffic.",
            "Device placement is declarative; only registered code generators produce executable artifacts.",
        ),
    )


class BackendCompiler(Protocol):
    capabilities: BackendCapabilities

    def compile(self, program: CapturedProgram) -> Any: ...


@dataclass(frozen=True)
class CompiledResult:
    artifact: Any
    placement: PartitionReport


class BackendRegistry:
    def __init__(self) -> None:
        self._items: dict[str, BackendCompiler] = {}

    def register(self, compiler: BackendCompiler) -> None:
        name = compiler.capabilities.name
        if name in self._items:
            raise ValueError(
                f"Backend {name!r} already registered; explicit replacement required"
            )
        if not hasattr(compiler, "compile"):
            raise ValueError("Backend must provide compile()")
        self._items[name] = compiler

    def list(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def compile(self, program: CapturedProgram, name: str) -> CompiledResult:
        if name not in self._items:
            raise KeyError(f"Unknown backend {name!r}; installed: {self.list()}")
        backend = self._items[name]
        report = partition(program.ir, backend.capabilities)
        if report.unsupported:
            raise UnsupportedOperation(
                f"Backend {name} cannot lower: {', '.join(report.unsupported)}"
            )
        return CompiledResult(backend.compile(program), report)


class NumpyAOTBackend:
    capabilities = BackendCapabilities(
        name="numpy-aot",
        revision="1",
        native_ops=frozenset(
            {
                "linear",
                "matmul",
                "add",
                "sub",
                "mul",
                "relu",
                "sigmoid",
                "tanh",
                "topk",
                "zeros_like",
                "getitem",
                "scatter",
                "transpose",
                "reshape",
                "view",
                "flatten",
                "contiguous",
                "permute",
                "softmax",
                "exp",
                "sum",
                "amax",
                "rsqrt",
                "div",
                "gelu",
                "layer_norm",
                "argsort",
                "conv2d",
            }
        ),
        fallback_ops=frozenset(),
        supported_dtypes=frozenset(
            {
                "torch.float32",
                "torch.float64",
                "torch.int64",
                "torch.int32",
                "torch.float16",
                "torch.bool",
            }
        ),
        description="Pure NumPy portable AOT, standalone runtime; not an accelerator",
    )

    def compile(self, program: CapturedProgram) -> AOTArtifact:
        return compile_numpy_aot(program)


def default_registry() -> BackendRegistry:
    result = BackendRegistry()
    result.register(NumpyAOTBackend())
    return result
