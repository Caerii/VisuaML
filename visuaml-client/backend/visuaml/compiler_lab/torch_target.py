"""Second genuine AOT target: serialized torch.export ExportedProgram (.pt2).

This artifact requires the PyTorch runtime (unlike the NumPy-AOT target).
It is not CUDA code generation or an accelerator compiler on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from visuaml.compiler_spike import CapturedProgram, UnsupportedOperation

from .backends import BackendCapabilities


@dataclass
class TorchExportArtifact:
    exported: torch.export.ExportedProgram
    input_specs: tuple[tuple[tuple[int, ...], torch.dtype], ...]

    def check_inputs(self, *args: torch.Tensor) -> None:
        if len(args) != len(self.input_specs):
            raise UnsupportedOperation("Torch-export input arity differs from capture")
        for i, (arg, (shape, dtype)) in enumerate(zip(args, self.input_specs)):
            if (
                not isinstance(arg, torch.Tensor)
                or tuple(arg.shape) != shape
                or arg.dtype != dtype
            ):
                raise UnsupportedOperation(f"Input {i} violates fixed export contract")

    def run(self, *args: torch.Tensor) -> Any:
        self.check_inputs(*args)
        return self.exported.module()(*args)

    def write(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "model.pt2"
        torch.export.save(self.exported, target)
        return target

    @classmethod
    def load(
        cls,
        path: str | Path,
        input_specs: tuple[tuple[tuple[int, ...], torch.dtype], ...],
    ):
        return cls(torch.export.load(path), input_specs)


class TorchExportBackend:
    capabilities = BackendCapabilities(
        name="torch-export",
        revision="1",
        native_ops=frozenset({"*"}),
        fallback_ops=frozenset(),
        supported_dtypes=frozenset(
            {
                "torch.float16",
                "torch.float32",
                "torch.float64",
                "torch.bfloat16",
                "torch.int64",
                "torch.int32",
                "torch.bool",
            }
        ),
        description="PyTorch ExportedProgram AOT; runtime still requires PyTorch",
    )

    def __init__(self, example_inputs: tuple[torch.Tensor, ...]):
        if not example_inputs or any(
            not isinstance(v, torch.Tensor) for v in example_inputs
        ):
            raise ValueError("Export backend needs concrete example tensors")
        self.inputs = example_inputs

    def compile(self, program: CapturedProgram) -> TorchExportArtifact:
        try:
            with torch.no_grad():
                exported = torch.export.export(
                    program.graph_module.eval(), self.inputs, strict=True
                )
        except Exception as exc:
            raise UnsupportedOperation(f"Torch AOT export failed: {exc}") from exc
        return TorchExportArtifact(
            exported, tuple((tuple(x.shape), x.dtype) for x in self.inputs)
        )
