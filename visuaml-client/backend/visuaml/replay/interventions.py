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


def fit_delta_basis(deltas, rank):
    """Fit orthonormal columns to uncentered calibration activation deltas."""
    if (deltas.ndim != 2 or deltas.dtype not in (torch.float32, torch.float64) or
            type(rank) is not int or not 1 <= rank <= min(deltas.shape) or
            not torch.isfinite(deltas).all()):
        raise ValueError("Expected a finite floating matrix and a valid basis rank")
    _, values, right = torch.linalg.svd(deltas, full_matrices=False)
    tolerance = torch.finfo(deltas.dtype).eps*max(deltas.shape)*values[0]
    if values[rank-1] <= tolerance:
        raise ValueError("Requested rank exceeds the numerical rank of calibration deltas")
    return right[:rank].T.contiguous()


def split_donor_delta(recipient, donor, basis):
    """Return recipient activations with the projected and complementary edits."""
    if (recipient.shape != donor.shape or recipient.ndim < 1 or basis.ndim != 2
            or basis.shape[0] != recipient.shape[-1] or basis.shape[1] == 0
            or donor.dtype != recipient.dtype or basis.dtype != recipient.dtype
            or donor.device != recipient.device or basis.device != recipient.device):
        raise ValueError("Donor, recipient and basis must have compatible feature dimensions, dtype, and device")
    gram = basis.T @ basis
    identity = torch.eye(basis.shape[1], dtype=basis.dtype, device=basis.device)
    epsilon = torch.finfo(basis.dtype).eps if basis.is_floating_point() else 0
    if not basis.is_floating_point() or not torch.allclose(gram, identity, atol=8*epsilon, rtol=8*epsilon):
        raise ValueError("Basis must have orthonormal columns")
    delta = donor-recipient
    projected = (delta @ basis) @ basis.T
    return recipient+projected, donor-projected
