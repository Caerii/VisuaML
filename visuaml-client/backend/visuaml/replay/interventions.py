"""Checked PyTorch activation observations and whole-tensor substitutions."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ActivationSite:
    module: str
    phase: str = "forward"

    def __post_init__(self):
        if not self.module or self.phase != "forward":
            raise ValueError("Expected a named module output in the forward phase")


def run_intervention(model, site, inputs, *, replacement=None):
    """Observe exactly one module output and optionally replace its tensor.

    Return the model output and independent before/after activation snapshots.
    Use an isolated model instance for concurrent requests.
    """
    module = model.get_submodule(site.module)
    observed = []

    def hook(_module, _args, output):
        if observed:
            raise ValueError(f"Activation site {site.module} was called more than once")
        if not isinstance(output, torch.Tensor):
            raise ValueError(f"Activation site {site.module} must yield one tensor")
        if replacement is not None and (not isinstance(replacement, torch.Tensor)
                or replacement.shape != output.shape or replacement.dtype != output.dtype
                or replacement.device != output.device):
            raise ValueError(f"Replacement must match shape, dtype, and device at {site.module}")
        changed = output if replacement is None else replacement.clone()
        observed.append((output.detach().clone(), changed.detach().clone()))
        return changed if replacement is not None else None

    with module.register_forward_hook(hook), torch.inference_mode():
        result = model(*inputs)
    if not observed:
        raise ValueError(f"Activation site {site.module} was not called")
    before, after = observed[0]
    return result, before, after
