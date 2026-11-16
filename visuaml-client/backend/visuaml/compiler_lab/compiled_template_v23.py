"""v23 checkpoint-bound compiled Llama template shared across KV-isolated sessions.

A template validates a canonical Llama SemanticProgram, traces the fully
polymorphic v22 TorchScript decoder exactly once, and admits new sessions
that reuse the immutable graph object while owning independent transactional
KV storage. This is process-local, CPU TorchScript graph reuse, NOT a
serializable independently loadable AOT artifact or untrusted model loader.
"""

from __future__ import annotations

from threading import RLock

import numpy as np

from .llama_semantic_v20 import (
    LlamaSemanticPlan,
    lower_llama_semantic_program,
)
from .torchscript_graph_v21 import TorchScriptWholeGraphSession


class CompiledTemplateError(ValueError):
    """The shared executable is not a verified, canonical model compilation."""


class CompiledLlamaTemplate:
    """Explicitly owned reusable graph, with one isolated stateful session per request.

    The compile-once step is protected by a lock. Individual sessions are
    single-owner and must not be concurrently mutated. The TorchScript graph
    reads checkpoint weights but accepts distinct detached per-session KV
    inputs, and never mutates the resident NumPy cache.
    """

    def __init__(self, plan: LlamaSemanticPlan):
        if not isinstance(plan, LlamaSemanticPlan):
            raise CompiledTemplateError("Expected a canonical LlamaSemanticPlan")
        self._plan = lower_llama_semantic_program(
            plan.program, plan.weights, batch=plan.batch
        )
        self._lock = RLock()
        self._compiled_graph = None
        self._first_trace = None
        self._trace_count = 0

    @property
    def semantic_sha256(self) -> str:
        return self._plan.semantic_sha256

    @property
    def checkpoint_sha256(self) -> str:
        return self._plan.weights.checkpoint_sha256

    def new_session(self) -> TorchScriptWholeGraphSession:
        """Create independent KV state; re-use the graph after one certified trace."""
        with self._lock:
            # The lowerer independently revalidates and resnapshots source
            # weights at session creation. Sessions cannot alias KV storage.
            session = self._plan.new_session(backend="torchscript_poly")
            signature = (self._plan.batch, -1, -1)
            if self._compiled_graph is None:
                # Trace against an admitted fixed local token fixture; neither
                # source code nor untrusted external data reaches the tracer.
                tokens = np.ones((self._plan.batch, 1), dtype=np.int64)
                compiled_logits = session.step(tokens)
                oracle = self._plan.new_session(backend="numpy")
                expected = oracle.step(tokens)
                try:
                    np.testing.assert_allclose(
                        compiled_logits, expected, atol=5e-5, rtol=5e-5
                    )
                    for got, want in zip(
                        session.snapshot().layers, oracle.snapshot().layers
                    ):
                        np.testing.assert_allclose(
                            got.keys, want.keys, atol=5e-5, rtol=5e-5
                        )
                        np.testing.assert_allclose(
                            got.values, want.values, atol=5e-5, rtol=5e-5
                        )
                except AssertionError as exc:
                    raise CompiledTemplateError(
                        "Whole-graph witness parity failed before template publication"
                    ) from exc
                self._compiled_graph = session._compiled[signature]
                self._first_trace = session.report()["traces"][0]
                self._trace_count += 1
                session.reset()
            else:
                session._compiled[signature] = self._compiled_graph
            return session

    def report(self) -> dict:
        with self._lock:
            return {
                "schema": "visuaml.shared-llama-template.v23",
                "backend": "CPU TorchScript polymorphic whole-graph",
                "semantic_sha256": self.semantic_sha256,
                "checkpoint_sha256": self.checkpoint_sha256,
                "compiled_graphs": self._trace_count,
                "first_trace": self._first_trace,
                "limitations": (
                    "Process-local in-memory template, not a standalone AOT binary.",
                    "The compiled graph is reused but each session has private KV.",
                    "Only a bounded, checkpoint-bound Llama ABI is admitted.",
                    "Individual KV sessions are not thread-safe.",
                ),
            }
