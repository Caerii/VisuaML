"""Small working example: PYTHONPATH=visuaml-client/backend python visuaml-client/backend/examples/compiler_spike_demo.py"""

import json

import torch
from torch import nn
from visuaml.compiler_spike import (
    capture,
    decompose_linear,
    fuse_linear_pattern,
    lower,
    compare_outputs,
    to_visuaml_graph,
)


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 16)
        self.output = nn.Linear(16, 4)

    def forward(self, x):
        return self.output(torch.relu(self.linear(x)))


def main():
    torch.manual_seed(7)
    x = torch.randn(3, 8)
    original = capture(SmallModel(), x)
    decomposed = decompose_linear(original, x)
    print("ORIGINAL IR", json.dumps(original.ir.as_dict(), indent=2))
    print("DECOMPOSED SUMMARY", json.dumps(decomposed.ir.summary(), indent=2))
    fused = fuse_linear_pattern(decomposed, x)
    print("REFUSED SUMMARY", json.dumps(fused.ir.summary(), indent=2))
    reference = original.graph_module(x)
    for b in ("torch", "numpy", "optical-surrogate"):
        result = lower(decomposed, b)
        print(
            b,
            json.dumps(
                {
                    "visual_graph": to_visuaml_graph(decomposed.ir, result.plan),
                    "placement": result.plan.as_dict(),
                    "sample_error": compare_outputs(reference, result.run(x)),
                },
                indent=2,
            ),
        )


if __name__ == "__main__":
    main()
