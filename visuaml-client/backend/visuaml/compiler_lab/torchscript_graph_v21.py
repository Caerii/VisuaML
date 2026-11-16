"""v21 whole-decoder TorchScript graph backend for admitted Llama Semantic IR.

The canonical v20 SemanticProgram is lowered into a *single traced TorchScript
graph* per (batch, query_length, past_length) specialization. The graph
contains every layer's RMSNorm, half-RoPE, causal GQA attention, SwiGLU and
final projection, and returns complete next-layer KV tensors functionally.
The session validates results and publishes the new KV suffix atomically via
v17 transactional state.

This is CPU TorchScript (JIT graph), not ahead-of-time machine code, CUDA,
Triton, IREE, a portable independent binary, or shape-polymorphic compilation.
Graph trace operates only on trusted compiler-generated operators and the
strict local Llama weight ABI. No model Python source is imported or executed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from math import sqrt

import numpy as np
import torch
from torch import nn

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

MAX_TRACED_SPECIALIZATIONS = 32


class WholeGraphError(LlamaSemanticError):
    """A compiled Llama chunk, graph result, or state transition is inadmissible."""


@dataclass(frozen=True)
class GraphTraceRecord:
    batch: int
    query: int
    past: int
    trace_ms: float
    node_count: int


class _LlamaTraceModule(nn.Module):
    """Interpret a *validated* effectful program using PyTorch operations.

    Tracing this module produces one flattened TorchScript tensor graph, with
    no Python per-op dispatch during subsequent execution of this shape.
    """

    def __init__(self, plan: LlamaSemanticPlan, *, cursor_polymorphic: bool = False):
        super().__init__()
        self.cursor_polymorphic = cursor_polymorphic
        p = plan.weights.profile
        self.operations = plan.program.regions[0].operations
        self.batch = plan.batch
        self.heads = p.heads
        self.kv_heads = p.kv_heads
        self.depth = p.head_dim
        self.hidden = p.hidden
        self.layers = p.layers
        self._attributes: dict[str, str] = {}
        for index, (name, array) in enumerate(sorted(plan.weights.arrays.items())):
            attribute = f"w_{index}"
            self.register_buffer(attribute, torch.from_numpy(array.copy(order="C")))
            self._attributes[name] = attribute
        inv = 1.0 / (
            p.rope_theta ** (
                np.arange(0, p.head_dim, 2, dtype=np.float64) / p.head_dim
            )
        )
        theta = np.outer(np.arange(plan.capacity, dtype=np.float64), inv)
        trig_cos = np.concatenate((np.cos(theta), np.cos(theta)), axis=-1).astype(np.float32)
        trig_sin = np.concatenate((np.sin(theta), np.sin(theta)), axis=-1).astype(np.float32)
        self.register_buffer("rotary_cos", torch.from_numpy(trig_cos.copy()))
        self.register_buffer("rotary_sin", torch.from_numpy(trig_sin.copy()))

    def forward(self, tokens: torch.Tensor, *past: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # Python operation dispatch and effect ordering are consumed at trace
        # construction. The resulting graph is executed without this loop.
        history = {
            f"{kind}{layer}": past[2 * layer + (0 if kind == "k" else 1)]
            for layer in range(self.layers) for kind in ("k", "v")
        }
        weights = {name: getattr(self, attr) for name, attr in self._attributes.items()}
        env: dict[str, torch.Tensor] = {"tokens": tokens, **weights}
        pending: dict[int, torch.Tensor] = {}
        start = past[-1] if self.cursor_polymorphic else past[0].shape[1]
        qlength = tokens.shape[1]
        total = start + qlength
        positions = (
            torch.arange(qlength, device=tokens.device, dtype=torch.int64) + start
            if self.cursor_polymorphic else None
        )
        for op in self.operations:
            kind = op.op
            if kind == "read_cache":
                result = (
                    torch.narrow(history[op.state_cell], 1, 0, total)
                    if self.cursor_polymorphic else history[op.state_cell]
                )
            else:
                args = tuple(env[name] for name in op.operands)
                if kind == "stage_key":
                    pending[dict(op.parameters)["layer"]] = args[0]
                    continue
                if kind == "stage_value":
                    layer = dict(op.parameters)["layer"]
                    key_update = pending.pop(layer)
                    if self.cursor_polymorphic:
                        history[f"k{layer}"] = history[f"k{layer}"].index_copy(
                            1, positions, key_update
                        )
                        history[f"v{layer}"] = history[f"v{layer}"].index_copy(
                            1, positions, args[0]
                        )
                    else:
                        history[f"k{layer}"] = torch.cat(
                            (history[f"k{layer}"], key_update), dim=1
                        )
                        history[f"v{layer}"] = torch.cat(
                            (history[f"v{layer}"], args[0]), dim=1
                        )
                    continue
                if kind == "embedding":
                    result = torch.nn.functional.embedding(args[0], args[1])
                elif kind == "rmsnorm_eps":
                    x, norm = args
                    eps = dict(op.parameters)["eps"]
                    result = x / torch.sqrt(
                        torch.mean(x * x, dim=-1, keepdim=True) + eps
                    ) * norm
                elif kind == "linear":
                    result = torch.matmul(args[0], args[1].transpose(0, 1))
                elif kind == "split_heads":
                    result = args[0].reshape(
                        self.batch, qlength, dict(op.parameters)["heads"], self.depth
                    )
                elif kind == "rope_half":
                    tensor = args[0]
                    if self.cursor_polymorphic:
                        cosine = self.rotary_cos.index_select(0, positions)[None, :, None, :]
                        sine = self.rotary_sin.index_select(0, positions)[None, :, None, :]
                    else:
                        cosine = self.rotary_cos[start:total][None, :, None, :]
                        sine = self.rotary_sin[start:total][None, :, None, :]
                    half = self.depth // 2
                    rotated = torch.cat(
                        (-tensor[..., half:], tensor[..., :half]), dim=-1
                    )
                    result = tensor * cosine + rotated * sine
                elif kind == "gqa_attention":
                    query, keys, values = args
                    replicas = self.heads // self.kv_heads
                    grouped = query.reshape(
                        self.batch, qlength, self.kv_heads, replicas, self.depth
                    ).permute(0, 2, 3, 1, 4)
                    k = keys.permute(0, 2, 1, 3)[:, :, None, :, :]
                    scores = torch.matmul(grouped, k.transpose(-1, -2)) * (
                        1.0 / sqrt(self.depth)
                    )
                    causal = (
                        torch.arange(total, device=scores.device)[None, :]
                        <= torch.arange(start, total, device=scores.device)[:, None]
                    )
                    scores = scores.masked_fill(
                        ~causal[None, None, None, :, :], -1e9
                    )
                    probs = torch.softmax(scores, dim=-1)
                    v = values.permute(0, 2, 1, 3)[:, :, None, :, :]
                    result = torch.matmul(probs, v).permute(
                        0, 3, 1, 2, 4
                    ).reshape(self.batch, qlength, self.hidden)
                elif kind == "silu":
                    result = args[0] / (1.0 + torch.exp(-args[0]))
                elif kind == "add":
                    result = args[0] + args[1]
                elif kind == "multiply":
                    result = args[0] * args[1]
                else:
                    raise WholeGraphError(f"Illegal whole-graph operator: {kind}")
            if op.result is None:
                raise WholeGraphError("Compiled graph effect returned an SSA result")
            env[op.result] = result
        if pending or "logits" not in env:
            raise WholeGraphError("Compiled graph left an uncommitted state effect")
        return (
            env["logits"],
            *(
                history[f"{kind}{layer}"]
                for layer in range(self.layers) for kind in ("k", "v")
            ),
        )


class TorchScriptWholeGraphSession:
    """Shape-specialized full-graph execution with transactional KV publication.

    A session owns one checkpoint, one KV cursor, and one bounded TorchScript
    specialization cache. It is deliberately not thread-safe.
    """

    def __init__(
        self, plan: LlamaSemanticPlan, *, cursor_polymorphic: bool = False
    ) -> None:
        if type(cursor_polymorphic) is not bool:
            raise WholeGraphError("Invalid cursor-polymorphism selector")
        admitted = lower_llama_semantic_program(
            plan.program, plan.weights, batch=plan.batch
        )
        self.plan = admitted
        p = admitted.weights.profile
        self.batch = admitted.batch
        self.vocabulary = p.vocabulary
        self.capacity = admitted.capacity
        self.layers = p.layers
        self.kv_heads = p.kv_heads
        self.depth = p.head_dim
        self.cursor_polymorphic = cursor_polymorphic
        self._torch = _LlamaTraceModule(
            admitted, cursor_polymorphic=cursor_polymorphic
        ).eval()
        self.cache = lower_kv_semantic_program(
            make_kv_semantic_program(
                batch=self.batch, layers=self.layers,
                kv_heads=self.kv_heads, head_dim=self.depth,
                capacity=self.capacity,
            )
        ).new_session()
        self._compiled: dict[tuple[int, int, int], torch.jit.ScriptModule] = {}
        self._trace_records: list[GraphTraceRecord] = []

    @property
    def cursor(self) -> int:
        return self.cache.cursor

    def _state_inputs(self) -> tuple[torch.Tensor, ...]:
        values = tuple(
            torch.from_numpy(
                (
                    self.cache._arrays[f"{kind}{layer}"]
                    if self.cursor_polymorphic
                    else self.cache._arrays[f"{kind}{layer}"][:, :self.cursor]
                ).copy(order="C")
            )
            for layer in range(self.layers) for kind in ("k", "v")
        )
        return (
            (*values, torch.tensor(self.cursor, dtype=torch.int64))
            if self.cursor_polymorphic else values
        )

    def step(self, tokens: np.ndarray) -> np.ndarray:
        if (
            not isinstance(tokens, np.ndarray)
            or tokens.dtype != np.int64 or tokens.ndim != 2
            or tokens.shape[0] != self.batch
            or not 1 <= tokens.shape[1] <= 32
            or self.cursor + tokens.shape[1] > self.capacity
            or np.any(tokens < 0) or np.any(tokens >= self.vocabulary)
        ):
            raise WholeGraphError("Invalid token IDs, query, batch, or context budget")
        start = self.cursor
        query = tokens.shape[1]
        signature = (
            (self.batch, -1, -1) if self.cursor_polymorphic
            else (self.batch, query, start)
        )
        input_tokens = torch.from_numpy(tokens.copy(order="C"))
        prior = self._state_inputs()
        if signature not in self._compiled:
            if len(self._compiled) >= MAX_TRACED_SPECIALIZATIONS:
                raise WholeGraphError("TorchScript shape-specialization budget exceeded")
            begun = time.perf_counter_ns()
            with torch.no_grad():
                # Only compiler-generated, allowlisted mathematical semantics
                # are traced; the graph is checked by independent oracle tests.
                compiled = torch.jit.trace(
                    self._torch, (input_tokens, *prior),
                    check_trace=False, strict=True,
                )
            elapsed = (time.perf_counter_ns() - begun) / 1e6
            self._compiled[signature] = compiled
            self._trace_records.append(GraphTraceRecord(
                self.batch, signature[1], signature[2], elapsed,
                len(tuple(compiled.graph.nodes())),
            ))
        with torch.no_grad():
            result = self._compiled[signature](input_tokens, *prior)
        if (
            not isinstance(result, tuple)
            or len(result) != 1 + self.layers * 2
        ):
            raise WholeGraphError("Compiled graph returned malformed outputs")
        logits = result[0].detach().cpu().numpy().copy()
        if (
            logits.shape != (self.batch, query, self.vocabulary)
            or logits.dtype != np.float32 or not np.isfinite(logits).all()
        ):
            raise WholeGraphError("Compiled graph returned invalid logits")
        staged: list[tuple[np.ndarray, np.ndarray]] = []
        for layer in range(self.layers):
            k = result[1 + 2 * layer].detach().cpu().numpy()
            v = result[2 + 2 * layer].detach().cpu().numpy()
            expected = (
                self.batch,
                self.capacity if self.cursor_polymorphic else start + query,
                self.kv_heads, self.depth,
            )
            if (
                k.shape != expected or v.shape != expected
                or k.dtype != np.float32 or v.dtype != np.float32
                or not np.isfinite(k).all() or not np.isfinite(v).all()
            ):
                raise WholeGraphError("Compiled graph returned invalid layer KV state")
            prior_keys = self.cache._arrays[f"k{layer}"][:, :start]
            prior_values = self.cache._arrays[f"v{layer}"][:, :start]
            if start and (
                not np.array_equal(k[:, :start], prior_keys)
                or not np.array_equal(v[:, :start], prior_values)
            ):
                raise WholeGraphError("Compiled graph altered previously committed KV state")
            if self.cursor_polymorphic and (
                not np.array_equal(
                    k[:, start + query:], self.cache._arrays[f"k{layer}"][:, start + query:]
                )
                or not np.array_equal(
                    v[:, start + query:], self.cache._arrays[f"v{layer}"][:, start + query:]
                )
            ):
                raise WholeGraphError("Compiled graph modified unrelated KV storage")
            staged.append((
                k[:, start:start + query].copy(),
                v[:, start:start + query].copy(),
            ))
        with self.cache.begin_append(query) as tx:
            for layer, (keys, values) in enumerate(staged):
                tx.stage(layer, keys, values)
            tx.commit()
        return logits

    def snapshot(self) -> StackState:
        values = dict(self.cache.snapshot().buffers)
        return StackState(tuple(
            KVState(values[f"k{layer}"], values[f"v{layer}"], self.capacity)
            for layer in range(self.layers)
        ))

    def restore(self, state: StackState) -> None:
        if not isinstance(state, StackState) or len(state.layers) != self.layers:
            raise WholeGraphError("Invalid decoder KV state layers")
        if not isinstance(state.layers[0], KVState) or not isinstance(
            state.layers[0].keys, np.ndarray
        ) or state.layers[0].keys.ndim != 4:
            raise WholeGraphError("Invalid decoder KV state shape")
        length = state.layers[0].keys.shape[1]
        pairs: list[tuple[str, np.ndarray]] = []
        for layer, cells in enumerate(state.layers):
            if not isinstance(cells, KVState) or cells.max_context != self.capacity:
                raise WholeGraphError("Incompatible restored decoder cache cell")
            pairs.extend(((f"k{layer}", cells.keys), (f"v{layer}", cells.values)))
        self.cache.restore(KVSnapshot(
            self.cache.plan.semantic_sha256, length, tuple(pairs)
        ))

    def reset(self) -> None:
        self.cache.reset()

    def generate(self, prompt: np.ndarray, *, steps: int) -> tuple[np.ndarray, StackState]:
        if type(steps) is not int or not 1 <= steps <= 24:
            raise WholeGraphError("Invalid generated token count")
        if not isinstance(prompt, np.ndarray) or prompt.ndim != 2:
            raise WholeGraphError("Invalid token prompt shape")
        if prompt.shape[1] + steps > self.capacity:
            raise WholeGraphError("Generation exceeds admitted context")
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
            "schema": (
                "visuaml.torchscript-cursor-polymorphic.v22"
                if self.cursor_polymorphic else "visuaml.torchscript-whole-graph.v21"
            ),
            "backend": (
                "torchscript_poly" if self.cursor_polymorphic else "torchscript_graph"
            ),
            "dynamic_query_and_cursor": self.cursor_polymorphic,
            "checkpoint_sha256": self.plan.weights.checkpoint_sha256,
            "semantic_sha256": self.plan.semantic_sha256,
            "specializations": len(self._compiled),
            "traces": [
                {
                    "batch": entry.batch, "query": entry.query, "past": entry.past,
                    "trace_ms": entry.trace_ms, "graph_nodes": entry.node_count,
                }
                for entry in self._trace_records
            ],
            "limitations": [
                "Compiled TorchScript CPU graph, not standalone AOT machine code.",
                (
                    "v22 traces once per batch/model, using fixed-capacity KV inputs; "
                    "v21 traces per query and past length."
                ),
                "Tensor inputs, KV copies and graph tracing add overhead.",
                "No GPU, dynamic-shape compile-once, or physical photonic device.",
            ],
        }
