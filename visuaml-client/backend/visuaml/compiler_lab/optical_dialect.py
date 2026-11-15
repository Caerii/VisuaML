"""Physically constrained *illustrative* coherent optical linear surrogate.

No Ayo schematics, measured device parameters or electro-optical interfaces are
used. This is NOT a deployable optical lowering. It permits real affine maps
as coherent field transforms with gain and electronic bias, plus explicit
loss, phase discretization, amplitude saturation and illustrative noise.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from math import isfinite, pi
from typing import Literal

import numpy as np
import torch
from torch import nn


class OpticalContractError(ValueError):
    pass


@dataclass(frozen=True)
class OpticalProfile:
    attenuation_db: float = 0.0
    phase_bits: int | None = None
    gaussian_field_noise: float = 0.0
    gaussian_readout_noise: float = 0.0
    saturation_amplitude: float | None = None
    provenance: str = "unmeasured-illustrative-surrogate"
    source_kind: Literal["synthetic"] = "synthetic"

    def __post_init__(self):
        values = (self.attenuation_db, self.gaussian_field_noise, self.gaussian_readout_noise)
        if any(type(v) not in (int, float) or not isfinite(v) or v < 0 for v in values):
            raise OpticalContractError("Loss/noise must be finite nonnegative scalars")
        if self.phase_bits is not None and (type(self.phase_bits) is not int or not 1 <= self.phase_bits <= 16):
            raise OpticalContractError("Phase resolution outside 1..16 bits")
        if self.saturation_amplitude is not None and (type(self.saturation_amplitude) not in (int, float)
            or not isfinite(self.saturation_amplitude) or self.saturation_amplitude <= 0):
            raise OpticalContractError("Saturation amplitude must be positive")
        if not self.provenance or self.source_kind != "synthetic":
            raise OpticalContractError("Only explicitly synthetic optical profiles are supported")


@dataclass(frozen=True)
class OpticalPlan:
    """The photonic part computes complex field U @ x, followed by scalar gain.

    Bias remains an electronic postprocessing operation. A passive coherent
    transfer is normalized to spectral norm <=1. The gain requirement is
    reported explicitly, not silently absorbed into a fictitious passive chip.
    """
    complex_transfer: np.ndarray
    electrical_bias: np.ndarray | None
    required_amplitude_gain: float
    input_features: int
    output_features: int
    source_sha256: str

    def __post_init__(self):
        w = np.asarray(self.complex_transfer)
        if w.ndim != 2 or w.dtype != np.complex128 or w.shape != (self.output_features, self.input_features):
            raise OpticalContractError("Invalid coherent field transfer matrix")
        if np.any(~np.isfinite(w)) or np.linalg.norm(w, 2) > 1.00000001:
            raise OpticalContractError("Passive stage violates spectral norm bound")
        if not isfinite(self.required_amplitude_gain) or self.required_amplitude_gain < 0:
            raise OpticalContractError("Invalid required gain")
        if self.electrical_bias is not None and (self.electrical_bias.shape != (self.output_features,)
            or self.electrical_bias.dtype != np.float64 or not np.isfinite(self.electrical_bias).all()):
            raise OpticalContractError("Invalid electrical bias")


@dataclass(frozen=True)
class OpticalRun:
    result: np.ndarray
    ideal: np.ndarray
    max_abs_error: float
    relative_l2_error: float
    field_energy_input: float
    field_energy_after_stage: float
    plan_sha256: str
    profile: OpticalProfile

    def report(self) -> dict:
        return {
            "schema": "visuaml.optical-surrogate.v1",
            "source_kind": "synthetic",
            "plan_sha256": self.plan_sha256,
            "profile": asdict(self.profile),
            "max_abs_error": self.max_abs_error,
            "relative_l2_error": self.relative_l2_error,
            "field_energy_input": self.field_energy_input,
            "field_energy_after_stage": self.field_energy_after_stage,
            "warnings": [
                "Coherent real-valued field readout is a mathematical surrogate, not a built optical device.",
                "No actual laser, ADC/DAC, tuning, latency, power, or calibration data is included.",
                "Gaussian noise and amplitude saturation are illustrative, not validated device laws.",
            ],
        }


def lower_linear_to_optical_surrogate(layer: nn.Linear, *, max_amplitude_gain: float = 16.0) -> OpticalPlan:
    if not isinstance(layer, nn.Linear) or layer.weight.dtype not in (torch.float32, torch.float64):
        raise OpticalContractError("Only real float32/64 nn.Linear modules are admitted")
    if type(max_amplitude_gain) not in (int, float) or not isfinite(max_amplitude_gain) or max_amplitude_gain <= 0:
        raise OpticalContractError("Gain limit must be finite positive")
    weight = layer.weight.detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(weight).all() or max(weight.shape) > 256:
        raise OpticalContractError("Invalid or oversized optical linear map")
    norm = float(np.linalg.norm(weight, 2))
    if norm > max_amplitude_gain:
        raise OpticalContractError("Required coherent gain exceeds declared budget")
    normalized = (weight / norm if norm > 0 else np.zeros_like(weight)).astype(np.complex128)
    bias = (layer.bias.detach().cpu().numpy().astype(np.float64).copy()
            if layer.bias is not None else None)
    digest = hashlib.sha256(weight.tobytes() + (bias.tobytes() if bias is not None else b"")).hexdigest()
    return OpticalPlan(normalized, bias, norm, layer.in_features, layer.out_features, digest)


def _phase_discretize(w: np.ndarray, bits: int | None) -> np.ndarray:
    if bits is None:
        return w.copy()
    phases = np.angle(w)
    levels = 2 ** bits
    quantized = np.round((phases % (2 * pi)) / (2 * pi) * levels) % levels
    return np.abs(w) * np.exp(1j * quantized * (2 * pi / levels))


def run_optical_surrogate(plan: OpticalPlan, inputs: np.ndarray, profile: OpticalProfile, *, seed: int = 0) -> OpticalRun:
    if not isinstance(plan, OpticalPlan) or not isinstance(profile, OpticalProfile):
        raise OpticalContractError("Plan and profile must be typed")
    if type(seed) is not int or not 0 <= seed <= 2 ** 32 - 1:
        raise OpticalContractError("Invalid reproducibility seed")
    x = np.asarray(inputs)
    if x.ndim != 2 or x.shape[1] != plan.input_features or x.dtype not in (np.float32, np.float64):
        raise OpticalContractError("Inputs must be [batch,in_features] real float32/64")
    if x.size > 65536 or not np.isfinite(x).all():
        raise OpticalContractError("Input budget or finiteness violated")
    ideal = np.asarray(x, np.float64) @ (plan.complex_transfer.real * plan.required_amplitude_gain).T
    if plan.electrical_bias is not None:
        ideal = ideal + plan.electrical_bias
    rng = np.random.default_rng(seed)
    trans = _phase_discretize(plan.complex_transfer, profile.phase_bits)
    amplitude = (np.asarray(x, np.complex128) @ trans.T) * (10 ** (-profile.attenuation_db / 20))
    amplitude *= plan.required_amplitude_gain
    if profile.saturation_amplitude is not None:
        size = np.abs(amplitude)
        amplitude *= np.minimum(1.0, profile.saturation_amplitude / np.maximum(size, 1e-30))
    if profile.gaussian_field_noise:
        noise = rng.normal(0, profile.gaussian_field_noise, amplitude.shape) + 1j * rng.normal(0, profile.gaussian_field_noise, amplitude.shape)
        amplitude += noise
    measured = amplitude.real.copy()  # coherent amplitude readout surrogate
    if profile.gaussian_readout_noise:
        measured += rng.normal(0, profile.gaussian_readout_noise, measured.shape)
    if plan.electrical_bias is not None:
        measured += plan.electrical_bias
    error = measured - ideal
    plan_digest = hashlib.sha256(plan.source_sha256.encode() + plan.complex_transfer.tobytes()).hexdigest()
    return OpticalRun(measured, ideal, float(np.max(np.abs(error))) if error.size else 0.0,
                      float(np.linalg.norm(error) / max(np.linalg.norm(ideal), 1e-12)),
                      float(np.sum(x.astype(np.float64) ** 2)), float(np.sum(np.abs(amplitude) ** 2)),
                      plan_digest, profile)
