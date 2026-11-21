"""Create two aligned PyTorch exports and a common input."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch


class MaskedResidual(torch.nn.Module):
    def __init__(self, changed):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(2))
        self.register_buffer("mask", torch.tensor([[0., -2.], [-2., 0. if changed else -2.]]))

    def forward(self, x):
        value = x @ self.weight
        scores = value @ value.T
        attention = (scores + self.mask).softmax(-1)
        return x + (attention @ value) @ self.weight


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    sample = torch.tensor([[1., 2.], [2., -1.]])
    np.save(args.directory / "input.npy", sample.numpy())
    for name, changed in (("reference", False), ("candidate", True)):
        module = MaskedResidual(changed).eval()
        torch.export.save(torch.export.export(module, (sample,)), args.directory / f"{name}.pt2")


if __name__ == "__main__":
    main()
