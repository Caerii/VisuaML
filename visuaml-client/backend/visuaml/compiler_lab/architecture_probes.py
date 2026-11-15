"""Allowlisted microarchitecture probes through real VisuaML FX and NumPy-AOT.

Success certifies ONE frozen tiny fixture, shape, dtype, seed and witness; it
never certifies an entire model family, arbitrary checkpoint, training graph,
KV-cache lifecycle, quantization scheme, or accelerator execution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from visuaml.compiler_spike import capture

from .aot_compile import compile_numpy_aot
from .numpy_runtime import OPS
from .verification import verify_outputs


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 12), nn.GELU(), nn.Linear(12, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, 3, 3, padding=1)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.proj(x))


class TinyAttention(nn.Module):
    """Attention operator motif, no masks, projections, rotary or KV caching."""

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        scores = (q @ k.transpose(-1, -2)) * 0.5
        return torch.softmax(scores, dim=-1) @ v


class TinyGNN(nn.Module):
    def forward(
        self, nodes: torch.Tensor, src: torch.Tensor, dst: torch.Tensor
    ) -> torch.Tensor:
        messages = nodes[src]
        return nodes.index_add(0, dst, messages)


class TinyScan(nn.Module):
    """A fixed-length, pure recurrence, not Mamba or a full SSM."""

    def forward(self, u: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        state = torch.zeros_like(u[:, 0])
        outputs: list[torch.Tensor] = []
        for t in range(3):
            state = a[:, t] * state + u[:, t]
            outputs.append(state)
        return torch.stack(outputs, dim=1)


class TinySpiking(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        membrane = x * 0.8 + 0.3
        return torch.where(membrane >= 1.0, torch.zeros_like(membrane), membrane)


class TinyUNetLike(nn.Module):
    """Minimal skip+upsample motif, not a conditional diffusion pipeline."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        small = F.avg_pool2d(x, 2)
        scaled = F.interpolate(small, scale_factor=2, mode="nearest")
        return self.conv(x + scaled)


class TinySplineLike(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x) + x * x


class TinyMoERouter(nn.Module):
    """Only top-k router selection; NOT full expert dispatch / grouping."""

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.topk(logits, 2, dim=-1).indices


class TinyGRU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.GRUCell(4, 4)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.cell(x, state)


# Only factory-held safe local models may execute here. Do NOT accept arbitrary
# module paths or run downloaded code from a model card/config.
_FIXTURES = {
    "mlp": (TinyMLP, ((3, 8),)),
    "residual-conv": (TinyResNet, ((1, 3, 6, 6),)),
    "attention-core": (TinyAttention, ((2, 3, 4), (2, 3, 4), (2, 3, 4))),
    "graph-scatter": (TinyGNN, ((6, 4), (5,), (5,))),
    "recurrent-scan": (TinyScan, ((2, 3, 4), (2, 3, 4))),
    "lif-gate": (TinySpiking, ((2, 7),)),
    "unet-upsampling": (TinyUNetLike, ((1, 2, 8, 8),)),
    "spline-sine": (TinySplineLike, ((2, 5),)),
    "moe-topk": (TinyMoERouter, ((3, 6),)),
    "gru-cell": (TinyGRU, ((2, 4), (2, 4))),
}


def _inputs(fixture: str, generator: torch.Generator) -> tuple[torch.Tensor, ...]:
    _class, shapes = _FIXTURES[fixture]
    if fixture == "graph-scatter":
        nodes = torch.randn((6, 4), generator=generator)
        return (
            nodes,
            torch.tensor([0, 1, 2, 0, 5], dtype=torch.int64),
            torch.tensor([1, 2, 3, 3, 0], dtype=torch.int64),
        )
    return tuple(torch.randn(shape, generator=generator) for shape in shapes)


@dataclass(frozen=True)
class ProbeResult:
    fixture: str
    status: str
    capture_passed: bool
    aot_passed: bool
    reference_executed: bool
    primitive_operations: tuple[str, ...]
    unsupported_aot_operations: tuple[str, ...]
    max_abs_error: float | None
    error_kind: str | None
    note: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def probe_fixture(
    fixture: str, *, seed: int = 41, atol: float = 1e-4, rtol: float = 1e-4
) -> ProbeResult:
    if fixture not in _FIXTURES:
        raise ValueError("Unknown fixture; only allowlisted tiny models are accepted")
    if type(seed) is not int or not 0 <= seed <= 100_000:
        raise ValueError("Probe seed out of range")
    torch_generator = torch.Generator(device="cpu").manual_seed(seed)
    # Module weight initialization uses the process RNG. Fork to avoid changing
    # the host caller's global random state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        constructor, _shapes = _FIXTURES[fixture]
        model = constructor().eval()
    inputs = _inputs(fixture, torch_generator)
    primitive_ops: tuple[str, ...] = ()
    capture_passed = False
    reference_executed = False
    missing: tuple[str, ...] = ()
    try:
        with torch.no_grad():
            captured = capture(model, *inputs)
            expected = model(*inputs)
        capture_passed = True
        reference_executed = True
        primitive_ops = tuple(
            dict.fromkeys(
                n.op
                for n in captured.ir.nodes
                if n.op not in {"input", "output", "constant"}
            )
        )
        missing = tuple(op for op in primitive_ops if op not in OPS)
        try:
            artifact = compile_numpy_aot(captured)
            with torch.no_grad():
                actual = artifact.run(*(tensor.detach().numpy() for tensor in inputs))
            report = verify_outputs(expected, actual, atol=atol, rtol=rtol)
            max_error = max((item.max_abs_error for item in report.leaves), default=0.0)
            if not report.passed:
                return ProbeResult(
                    fixture,
                    "numeric-mismatch",
                    True,
                    False,
                    True,
                    primitive_ops,
                    missing,
                    max_error,
                    "NumericalMismatch",
                    "Tiny CPU witness failed; no model-family coverage inferred",
                )
            return ProbeResult(
                fixture,
                "tiny-aot-witness-passed",
                True,
                True,
                True,
                primitive_ops,
                missing,
                max_error,
                None,
                "Only this synthetic fixture, captured input shape/dtype and seed were tested",
            )
        except Exception as exc:  # noqa: BLE001 - clearly bounded local probe diagnostics
            return ProbeResult(
                fixture,
                "aot-rejected",
                True,
                False,
                True,
                primitive_ops,
                missing,
                None,
                type(exc).__name__,
                "FX capture succeeded; independent NumPy execution is not validated",
            )
    except Exception as exc:  # noqa: BLE001 - trusted fixtures only
        return ProbeResult(
            fixture,
            "capture-rejected",
            capture_passed,
            False,
            reference_executed,
            primitive_ops,
            missing,
            None,
            type(exc).__name__,
            "FX capture or reference failed; this fixture is not supported",
        )


def probe_all(*, seed: int = 41) -> list[dict[str, Any]]:
    return [probe_fixture(name, seed=seed).as_dict() for name in _FIXTURES]


def fixture_names() -> tuple[str, ...]:
    return tuple(_FIXTURES)
