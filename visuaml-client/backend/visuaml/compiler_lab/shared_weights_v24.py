"""v24: a preverified immutable-weight runtime with truly lightweight sessions.

The v23 template already shares the traced graph, but every new session still
revalidates a checkpoint, clones the entire model twice and reconstructs a
Torch module. After the first full independent numerical certification, v24
reuses the exact verified graph AND weight-bearing Torch module; each new
session allocates only its own transactional resident KV buffers and local
execution bookkeeping.

This is a process-local *trusted factory*. It does not accept unverified
checkpoints, untrusted Python source, arbitrary graphs or serialized compiled
executables. Python code inside the process can still introspect or mutate
Torch objects; it is not a security boundary against hostile Python code.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .compiled_template_v23 import (
    CompiledLlamaTemplate,
    CompiledTemplateError,
)
from .torchscript_graph_v21 import TorchScriptWholeGraphSession


class SharedWeightsLlamaTemplate(CompiledLlamaTemplate):
    """Compile/verify once, and construct KV-isolated sessions without model copies."""

    def __init__(self, plan: Any) -> None:
        super().__init__(plan)
        self._shared_module = None
        self._shared_kv_plan = None
        self._shared_plan = None
        self._fast_sessions = 0
        self._first_session_created = False

    def new_session(self) -> TorchScriptWholeGraphSession:
        with self._lock:
            if self._compiled_graph is None:
                first = super().new_session()
                if first.report()["specializations"] != 1 or first.cursor != 0:
                    raise CompiledTemplateError("Initial template was not published atomically")
                self._shared_module = first._torch
                self._shared_kv_plan = first.cache.plan
                self._shared_plan = first.plan
                self._first_session_created = True
                return first

            if (
                not self._first_session_created
                or self._shared_plan is None
                or self._shared_kv_plan is None
                or self._shared_module is None
                or self._compiled_graph is None
                or self._shared_plan.semantic_sha256 != self._plan.semantic_sha256
                or self._shared_plan.weights.checkpoint_sha256 != self.checkpoint_sha256
            ):
                raise CompiledTemplateError("Shared model template lacks verified ownership")
            # Avoid _plan.new_session(): that would recursively canonicalize,
            # allocate fresh weight snapshots, and build another Torch module.
            # No model values or tensor buffers are copied in this path.
            session = object.__new__(TorchScriptWholeGraphSession)
            session.plan = self._shared_plan
            p = session.plan.weights.profile
            session.batch = session.plan.batch
            session.vocabulary = p.vocabulary
            session.capacity = session.plan.capacity
            session.layers = p.layers
            session.kv_heads = p.kv_heads
            session.depth = p.head_dim
            session.cursor_polymorphic = True
            session._torch = self._shared_module
            session.cache = self._shared_kv_plan.new_session()
            session._compiled = {(session.batch, -1, -1): self._compiled_graph}
            session._trace_records = []
            session._host_kv_input_bytes = 0
            session._graph_kv_output_bytes = 0
            self._fast_sessions += 1
            return session

    def report(self) -> dict:
        with self._lock:
            base = super().report()
            weight_bytes = (
                sum(b.numel() * b.element_size() for b in self._shared_module.buffers())
                if self._shared_module is not None else 0
            )
            return {
                **base,
                "schema": "visuaml.shared-weights-runtime.v24",
                "fast_sessions_without_model_snapshot": self._fast_sessions,
                "verified_torch_module_instances": 1 if self._first_session_created else 0,
                "weight_module_logical_bytes": weight_bytes,
                "repeated_model_weight_construction_in_fast_path": False,
                "limitations": (
                    "Owned in-process shared module, not a security boundary against Python code.",
                    "Per-session KV storage is allocated; no KV page pooling or speculative decoding.",
                    "This is TorchScript CPU, not offline native AOT, GPU or photonic computing.",
                    "Module logical bytes are not peak RSS or physical allocator measurements.",
                ),
            }
