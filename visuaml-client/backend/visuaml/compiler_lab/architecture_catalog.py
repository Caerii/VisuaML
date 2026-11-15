"""Versioned neural-architecture atlas and conservative operator-gap planner.

Catalog inclusion is NOT executable support. Entries are representative
computational motifs, not a promise to load arbitrary pretrained checkpoints.
The independent probes must verify real model capture and lowering.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable

SCHEMA = "visuaml.architecture-atlas.v1"
DOMAINS = frozenset({
    "language", "vision", "generative", "audio", "multimodal", "graphs-3d",
    "scientific", "biological", "classical", "robotics",
})
FEATURES = frozenset({
    "dynamic_shapes", "causal_mask", "kv_cache", "expert_dispatch",
    "distributed_collectives", "selective_scan", "sequence_state", "cross_attention",
    "graph_scatter", "sparse_tensor", "variable_graph", "geometric_equivariance",
    "neighbor_search", "stochastic_sampling", "time_conditioning", "sampling_loop",
    "multimodal_preprocessing", "streaming", "autograd_training", "quantized_kernels",
    "complex_arithmetic", "fft_semantics", "differentiable_rendering", "ray_marching",
    "event_driven_state", "adaptive_solver", "learned_plasticity", "dynamic_routing",
    "spatial_3d", "multi_output", "ragged_tensor", "implicit_surface",
    "sampling_state", "sparse_accumulation", "checkpoint_state",
})


@dataclass(frozen=True)
class Architecture:
    id: str
    name: str
    domain: str
    priority: int
    operators: tuple[str, ...]
    features: tuple[str, ...]
    exemplars: tuple[str, ...]
    note: str

    def __post_init__(self) -> None:
        if not self.id or not self.name or self.domain not in DOMAINS or self.priority not in (1, 2, 3):
            raise ValueError("Invalid architecture name, domain, or priority")
        if not self.operators or len(set(self.operators)) != len(self.operators):
            raise ValueError(f"Duplicate/missing primitive in {self.id}")
        if any(not item or not isinstance(item, str) for item in self.operators):
            raise ValueError(f"Malformed operator in {self.id}")
        if len(set(self.features)) != len(self.features) or not set(self.features) <= FEATURES:
            raise ValueError(f"Unknown/duplicate feature in {self.id}")
        if not self.note or not self.exemplars:
            raise ValueError(f"Missing architecture context in {self.id}")


def _make(id: str, name: str, domain: str, priority: int, ops: str, features: str,
          examples: str, note: str) -> Architecture:
    return Architecture(
        id, name, domain, priority,
        tuple(ops.split()), tuple(features.split()) if features else (),
        tuple(x.strip() for x in examples.split("|")), note,
    )


# The primitive sets are REQUIRED MOTIFS FOR REPRESENTATIVE WORKLOADS; actual
# checkpoints may require additional operators or different semantics.
# Priority 1: broad adoption or direct impact on hardware-ML design.
# Priority 2: actively valuable across current research and production.
# Priority 3: specialist, frontier or highly domain-specific workloads.
_RAW = (
    ("mlp", "Dense MLP / residual MLP", "classical", 1, "linear relu gelu add", "", "MLP-Mixer MLP block|Tabular networks", "AOT-friendly inference motif, not arbitrary MLP training"),
    ("resnet", "Residual CNN", "vision", 1, "conv2d relu add pooling2d batch_norm", "", "ResNet|DenseNet", "Full network also needs pooling and batch-normalization semantics"),
    ("convnext", "ConvNeXt / modern convolution", "vision", 1, "conv2d layer_norm gelu add", "", "ConvNeXt V2|MobileNet", "Depthwise convolution and data-layout contracts are important"),
    ("vit", "Vision Transformer", "vision", 1, "conv2d linear layer_norm matmul softmax add reshape transpose", "", "ViT|DINOv3", "Patch embedding plus transformer motif; not a pretrained ViT support claim"),
    ("swin", "Windowed / hierarchical ViT", "vision", 2, "conv2d linear matmul softmax reshape roll gather", "dynamic_shapes", "Swin Transformer|Swin V2", "Shifted windows require layout and attention-mask legalization"),
    ("sam", "Segmentation foundation models", "vision", 2, "conv2d linear matmul softmax upsample interpolate", "", "SAM|SAM 3", "Prompt encoding, mask decoding and image/video state vary by version"),
    ("unet", "U-Net encoder-decoder", "vision", 1, "conv2d relu pooling2d upsample concat", "", "U-Net|nnU-Net", "Skip connections and multiscale shape semantics"),
    ("pointnet", "Point-cloud MLP", "graphs-3d", 2, "linear relu max_reduce", "variable_graph", "PointNet|PointNet++", "Neighborhood queries may require geometric indexing"),
    ("voxel_3d", "Sparse 3D convolution / voxels", "graphs-3d", 2, "conv3d sparse_conv3d scatter_add", "sparse_tensor spatial_3d", "SparseConvNet|3D perception", "Sparse coordinate manager and kernels required"),
    ("decoder_transformer", "Decoder-only Transformer", "language", 1, "linear matmul softmax layer_norm reshape transpose add rotary", "causal_mask kv_cache", "Llama|Qwen|Gemma", "Prefill and autoregressive decode are separate execution modes"),
    ("encoder_transformer", "Encoder Transformer", "language", 1, "linear matmul softmax layer_norm reshape transpose add gelu", "", "BERT|RoBERTa", "Bidirectional attention, padding masks and embedding semantics"),
    ("encoder_decoder", "Encoder-decoder Transformer", "language", 1, "linear matmul softmax layer_norm reshape transpose add", "cross_attention kv_cache", "T5|BART", "Cross-attention cache and decoding must be explicit"),
    ("gqa_mqa", "Grouped / multi-query attention", "language", 1, "linear matmul softmax repeat_interleave rotary", "causal_mask kv_cache", "Llama GQA|MQA", "KV head grouping cannot be hidden inside a generic matmul"),
    ("mla", "Multi-head latent attention", "language", 1, "linear matmul softmax reshape rotary", "causal_mask kv_cache", "DeepSeek MLA", "Latent KV compression and rotary split need dedicated contracts"),
    ("flash_attention", "Fused / blockwise attention", "language", 1, "matmul softmax block_attention", "causal_mask", "FlashAttention|FlashAttention-3", "A kernel schedule, not merely a different mathematical attention family"),
    ("local_attention", "Sliding-window / block-sparse attention", "language", 2, "matmul softmax gather block_sparse_matmul", "causal_mask kv_cache", "Mistral sliding window|Long-context models", "Attention mask sparsity and locality matter"),
    ("linear_attention", "Linear / gated attention", "language", 1, "matmul elu gated_scan", "sequence_state", "GLA|Gated DeltaNet", "Recurrent and parallel forms must agree on state updates"),
    ("mamba", "Selective state-space models", "language", 1, "linear conv1d silu selective_scan", "selective_scan sequence_state", "Mamba|Mamba-2|Mamba-3", "Selective scan and checkpoint state, not just plain recurrence"),
    ("rwkv", "RWKV / weighted recurrent KV", "language", 1, "linear sigmoid exp rwkv_scan", "sequence_state", "RWKV-6|RWKV-7", "State layout and associative scan are architecture-specific"),
    ("xlstm", "Extended LSTM / matrix memory", "language", 2, "linear sigmoid tanh matmul xlstm_cell", "sequence_state", "xLSTM", "mLSTM matrix-state updates and normalization have dedicated semantics"),
    ("attention_ssm_hybrid", "Attention–SSM hybrid", "language", 1, "linear matmul softmax selective_scan", "sequence_state kv_cache", "Jamba|Qwen3-Next|Nemotron-H", "Must schedule mixed KV cache and recurrent state independently"),
    ("moe", "Sparse mixture of experts", "language", 1, "linear softmax topk gather scatter_add grouped_matmul", "expert_dispatch distributed_collectives", "Mixtral|DeepSeekMoE", "Dynamic expert routing, token permutation and capacity policies"),
    ("quantized_llm", "Quantized / ternary transformers", "language", 1, "linear matmul quantized_matmul layer_norm", "quantized_kernels", "BitNet b1.58|Int4/FP8 LLMs", "Packing, dequantization, scale granularity and overflow need contracts"),
    ("diffusion_language", "Discrete / diffusion language models", "language", 2, "linear matmul softmax categorical_sample", "sampling_loop stochastic_sampling", "LLaDA|Diffusion LMs", "Iterative denoising is not autoregressive KV-cache decoding"),
    ("rnn", "Vanilla RNN / GRU / LSTM", "classical", 1, "linear sigmoid tanh recurrent_scan", "sequence_state", "GRU|LSTM", "Hidden/cell state transitions and variable-length sequence handling"),
    ("reservoir", "Reservoir / echo-state networks", "biological", 3, "matmul tanh recurrent_scan", "sequence_state", "Echo-state networks", "Reservoir spectral radius and frozen recurrent connectivity"),
    ("hopfield", "Modern Hopfield / associative memory", "biological", 2, "matmul softmax topk", "sequence_state", "Modern Hopfield nets|Associative recall", "Update dynamics and stopping criteria are algorithmic semantics"),
    ("assembly_calculus", "Neural assemblies / local plasticity", "biological", 1, "gather_sum topk_indices hebbian_update", "learned_plasticity sequence_state", "Caerii/assemblies", "V10 effect dialect supports restricted schedules, not all reference-engine policies"),
    ("snn_lif", "Spiking neural networks", "biological", 2, "matmul lif_step where", "event_driven_state", "LIF|SpikingJelly", "Discrete/event-based time, reset semantics and surrogate gradients"),
    ("liquid", "Liquid neural networks / neural ODE", "biological", 2, "matmul tanh ode_integrate", "adaptive_solver sequence_state", "Liquid Time-Constant networks|Neural ODE", "Solver tolerances and event handling affect semantics"),
    ("nca", "Neural cellular automata", "biological", 3, "conv2d add where", "sequence_state stochastic_sampling", "Growing NCA", "Local update schedule and randomized asynchronous updates"),
    ("kan", "Kolmogorov-Arnold / spline networks", "scientific", 2, "spline_eval add", "", "KAN|B-spline networks", "Spline bases and adaptive grids differ from ordinary MLP operators"),
    ("siren", "Implicit sinusoidal neural fields", "scientific", 2, "linear sin", "", "SIREN|Implicit neural fields", "Sine activations and coordinate gradients matter"),
    ("fno", "Fourier neural operators", "scientific", 2, "fft ifft complex_mul linear", "fft_semantics complex_arithmetic", "FNO|PDE surrogates", "Spectral truncation, complex dtype and FFT conventions"),
    ("deeponet", "Deep operator networks", "scientific", 3, "linear matmul add", "", "DeepONet", "Branch/trunk representations and physical-domain shapes"),
    ("equivariant_gnn", "Equivariant graph / geometric networks", "scientific", 2, "scatter_add gather norm spherical_harmonics", "geometric_equivariance variable_graph", "EGNN|NequIP", "Irreducible representations and physical symmetry constraints"),
    ("graph_conv", "Graph convolutions / message passing", "graphs-3d", 1, "matmul gather scatter_add", "graph_scatter variable_graph", "GCN|GraphSAGE", "Edge-index aggregation and duplicate-edge semantics"),
    ("graph_attention", "Graph attention / Graph Transformer", "graphs-3d", 2, "linear scatter_add segment_softmax gather", "graph_scatter variable_graph", "GAT|GraphGPS", "Segment-wise softmax differs from dense attention"),
    ("nerf", "Neural radiance fields", "graphs-3d", 2, "linear sin exp volume_integrate", "ray_marching differentiable_rendering", "NeRF|Instant-NGP", "Volumetric rendering sampling and gradients"),
    ("gaussian_splatting", "3D Gaussian splatting", "graphs-3d", 2, "project_gaussians alpha_composite", "differentiable_rendering spatial_3d", "3D Gaussian Splatting", "Visibility, rasterization order and differentiable compositing"),
    ("vae", "Variational autoencoders", "generative", 1, "linear conv2d exp add mul", "stochastic_sampling", "VAE|VQ-VAE", "Reparameterization and probabilistic latent semantics"),
    ("gan", "Generative adversarial networks", "generative", 2, "conv2d linear relu", "autograd_training", "StyleGAN|Low-light GAN", "Training involves distinct generator/discriminator update cycles"),
    ("diffusion_unet", "Image diffusion U-Net", "generative", 1, "conv2d group_norm upsample softmax timestep_embedding", "time_conditioning sampling_loop", "Stable Diffusion|SDXL", "Denoiser and scheduler are separate executable subsystems"),
    ("dit", "Diffusion Transformers", "generative", 1, "linear matmul softmax ada_layer_norm timestep_embedding", "time_conditioning sampling_loop", "DiT|FLUX", "Adaptive conditioning, attention and numerical sampler"),
    ("flow_matching", "Rectified flow / flow matching", "generative", 1, "linear matmul ode_integrate", "time_conditioning sampling_loop", "Flow matching|Rectified flow", "Numerical solver and learned vector field must remain distinct"),
    ("video_diffusion", "Spatiotemporal video generators", "generative", 2, "conv3d matmul softmax reshape", "time_conditioning spatial_3d sampling_loop", "CogVideoX|Video DiT", "Temporal attention, 5D layouts and memory budgets"),
    ("autoregressive_image", "Autoregressive image generation", "generative", 2, "conv2d matmul softmax categorical_sample", "sampling_loop stochastic_sampling", "PixelCNN|Image token transformers", "Generation schedule and image tokenization are distinct"),
    ("conformer", "Conformer and acoustic encoders", "audio", 1, "conv1d linear matmul softmax layer_norm", "streaming", "Conformer|ASR encoders", "Depthwise time convolution, streaming chunks and length masks"),
    ("audio_codec", "Neural audio codecs / vocoders", "audio", 2, "conv1d conv_transpose1d quantized_matmul", "streaming quantized_kernels", "EnCodec|SoundStream", "Temporal upsampling, residual vector quantization"),
    ("speech_seq2seq", "Speech encoder–decoder", "audio", 1, "conv1d matmul softmax linear", "cross_attention streaming", "Whisper|Speech LMs", "Acoustic frontend, alignment and streaming decoding"),
    ("speech_diffusion", "Audio diffusion / flow", "audio", 2, "conv1d matmul ode_integrate", "sampling_loop time_conditioning", "AudioLDM|Audio DiT", "Waveform/latent timebase and synthesis solver"),
    ("clip", "Contrastive image-text towers", "multimodal", 1, "conv2d linear matmul softmax layer_norm", "multimodal_preprocessing", "CLIP|SigLIP", "Separate encoders and cross-modal contrastive training"),
    ("vlm", "Vision-language models", "multimodal", 1, "conv2d linear matmul softmax layer_norm", "multimodal_preprocessing kv_cache", "LLaVA|Qwen-VL", "Projection bridge, image tokens, variable token counts"),
    ("omni_modal", "Omni-modal audio/video language", "multimodal", 2, "conv1d conv2d matmul softmax", "multimodal_preprocessing streaming", "Audio/video-text models", "Synchronize modalities with heterogeneous clock rates"),
    ("v_jepa", "Joint embedding predictive models", "vision", 2, "conv2d linear matmul layer_norm", "autograd_training", "I-JEPA|V-JEPA 2", "Predictor and teacher/momentum update are training state"),
    ("world_model", "Latent world models", "robotics", 2, "conv2d recurrent_scan matmul", "sequence_state sampling_loop", "Dreamer|Video world models", "Rollout, imagination and policy loops differ from single-pass inference"),
    ("diffusion_policy", "Diffusion / flow robot policies", "robotics", 2, "conv2d matmul softmax ode_integrate", "sampling_loop time_conditioning", "Diffusion Policy|Flow policies", "Action-horizon conditioning and realtime scheduling"),
    ("energy_based", "Energy-based / Boltzmann networks", "classical", 3, "matmul exp categorical_sample", "stochastic_sampling sampling_loop", "RBM|Energy-based models", "Sampler chains and normalization/partition functions"),
    ("capsule", "Capsule networks", "classical", 3, "matmul softmax routing_update", "dynamic_routing", "CapsNet", "Iterative dynamic routing, not ordinary convolution"),
)

ARCHITECTURES = tuple(_make(*row) for row in _RAW)
_BY_ID = {entry.id: entry for entry in ARCHITECTURES}
if len(_BY_ID) != len(ARCHITECTURES):
    raise RuntimeError("Architecture IDs must be unique")


@dataclass(frozen=True)
class Coverage:
    architecture_id: str
    status: str  # a planning status, never a claim that a model executed
    covered_operators: tuple[str, ...]
    missing_operators: tuple[str, ...]
    missing_features: tuple[str, ...]
    coverage_fraction: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def capability_gap(entry: Architecture, *, available_ops: Iterable[str],
                   available_features: Iterable[str] = ()) -> Coverage:
    """Compute operator/feature *candidate* readiness, not model support."""
    ops, features = frozenset(available_ops), frozenset(available_features)
    missing_ops = tuple(o for o in entry.operators if o not in ops)
    covered = tuple(o for o in entry.operators if o in ops)
    missing_features = tuple(f for f in entry.features if f not in features)
    ready = not missing_ops and not missing_features
    status = "candidate-for-validation" if ready else "blocked-by-contract"
    return Coverage(entry.id, status, covered, missing_ops, missing_features,
                    len(covered) / len(entry.operators))


def catalog_report(*, available_ops: Iterable[str],
                   available_features: Iterable[str] = ()) -> dict[str, Any]:
    """Return a reproducible, versioned model-family coverage *planning* snapshot."""
    ops, features = frozenset(available_ops), frozenset(available_features)
    if any(not isinstance(x, str) or not x for x in ops | features):
        raise ValueError("Capability names must be nonempty strings")
    if not features <= FEATURES:
        raise ValueError("Unknown declared semantic feature")
    records = [capability_gap(f, available_ops=ops, available_features=features) for f in ARCHITECTURES]
    payload = {
        "schema": SCHEMA,
        "claim": "architecture planning only; candidate does not mean a pretrained model runs",
        "available_ops": sorted(ops),
        "available_features": sorted(features),
        "families": [{**asdict(a), "coverage": record.as_dict()} for a, record in zip(ARCHITECTURES, records)],
        "priority_gaps": rank_missing_capabilities(available_ops=ops, available_features=features, limit=16),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {**payload, "sha256": digest}


def rank_missing_capabilities(*, available_ops: Iterable[str],
                              available_features: Iterable[str] = (),
                              limit: int = 18) -> list[dict[str, Any]]:
    """Greedy planning signal: weighted family incidence, not performance gains."""
    if not isinstance(limit, int) or not 0 <= limit <= 100:
        raise ValueError("Invalid planning limit")
    ops, features = frozenset(available_ops), frozenset(available_features)
    weights = {1: 5, 2: 3, 3: 1}
    impact: dict[str, int] = {}
    blocked: dict[str, list[str]] = {}
    for arch in ARCHITECTURES:
        for op in arch.operators:
            if op not in ops:
                key = f"op:{op}"
                impact[key] = impact.get(key, 0) + weights[arch.priority]
                blocked.setdefault(key, []).append(arch.id)
        for feature in arch.features:
            if feature not in features:
                key = f"feature:{feature}"
                impact[key] = impact.get(key, 0) + weights[arch.priority]
                blocked.setdefault(key, []).append(arch.id)
    order = sorted(impact, key=lambda k: (-impact[k], k))[:limit]
    return [{"capability": k, "weighted_incidence": impact[k],
             "families": blocked[k], "metric": "declared incidence; not observed coverage gain"}
            for k in order]


def get_architecture(architecture_id: str) -> Architecture:
    return _BY_ID[architecture_id]
