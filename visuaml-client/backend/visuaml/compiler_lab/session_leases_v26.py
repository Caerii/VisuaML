"""v26 resource-accounted leases for isolated Llama inference sessions.

A verified registry manages templates; its logical cached-model budget does
not account for active client KV storage. This lease registry adds an explicit
per-process *logical KV byte* budget and active-session count, reserves before
allocation, and returns context-managed single-owner sessions.

This is NOT an OS-enforced RSS limit: TorchScript executable storage,
allocators, temporary activations and evicted-but-live model modules are
not captured. Individual leases cannot be used concurrently from multiple
threads and close must not race an ongoing step.
"""

from __future__ import annotations

from weakref import finalize

import numpy as np

from .stacked_decoder_v14 import StackState
from .template_registry_v25 import (
    TemplateNotCached,
    TemplateRegistryError,
    VerifiedLlamaRegistry,
)
from .torchscript_graph_v21 import TorchScriptWholeGraphSession


class LeaseBudgetExceeded(TemplateRegistryError):
    """An independent KV lease would exceed the configured logical budget."""


class LeaseClosed(TemplateRegistryError):
    """An explicitly released session cannot be used again."""


def _release(
    registry: LeaseManagedLlamaRegistry,
    session: TorchScriptWholeGraphSession,
    bytes_: int,
) -> None:
    """Idempotence is owned by weakref.finalize; clear sensitive KV before release."""
    with registry._lock:
        try:
            session.reset()
        finally:
            registry._active_kv_bytes -= bytes_
            registry._active_count -= 1
            registry._released += 1


class LeasedLlamaSession:
    """Closable, context-managed session with private cache and lifetime accounting.

    The wrapped session is intentionally not a public attribute. Python
    introspection remains possible and is not a hostile-code trust boundary.
    """

    def __init__(
        self, registry: LeaseManagedLlamaRegistry,
        session: TorchScriptWholeGraphSession,
        private_kv_bytes: int,
    ):
        self._session: TorchScriptWholeGraphSession | None = session
        self._private_kv_bytes = private_kv_bytes
        self._finalizer = finalize(self, _release, registry, session, private_kv_bytes)

    def _owned(self) -> TorchScriptWholeGraphSession:
        if self._session is None:
            raise LeaseClosed("The inference lease has been released")
        return self._session

    @property
    def cursor(self) -> int:
        return self._owned().cursor

    def step(self, tokens: np.ndarray) -> np.ndarray:
        return self._owned().step(tokens)

    def generate(self, tokens: np.ndarray, *, steps: int) -> tuple[np.ndarray, StackState]:
        return self._owned().generate(tokens, steps=steps)

    def snapshot(self) -> StackState:
        return self._owned().snapshot()

    def restore(self, state: StackState) -> None:
        self._owned().restore(state)

    def reset(self) -> None:
        self._owned().reset()

    def report(self) -> dict:
        return {
            **self._owned().report(),
            "lease_schema": "visuaml.logical-kv-session-lease.v26",
            "private_kv_bytes": self._private_kv_bytes,
        }

    def close(self) -> None:
        if self._session is None:
            return
        try:
            self._finalizer()
        finally:
            self._session = None

    def __enter__(self) -> LeasedLlamaSession:
        self._owned()
        return self

    def __exit__(self, _kind: object, _error: object, _traceback: object) -> None:
        self.close()


class LeaseManagedLlamaRegistry(VerifiedLlamaRegistry):
    """Verified model registry that only issues quota-reserved KV session leases.

    The base unleased factory is deliberately overridden; callers using this
    subclass cannot accidentally bypass its active-KV admission accounting.
    """

    def __init__(
        self, *, max_cached_weight_bytes: int = 256 * 1024 * 1024,
        max_models: int = 4,
        max_active_kv_bytes: int = 64 * 1024 * 1024,
        max_active_sessions: int = 16,
    ) -> None:
        super().__init__(
            max_cached_weight_bytes=max_cached_weight_bytes,
            max_models=max_models,
        )
        if (
            type(max_active_kv_bytes) is not int
            or not 4096 <= max_active_kv_bytes <= 2 * 1024 * 1024 * 1024
            or type(max_active_sessions) is not int
            or not 1 <= max_active_sessions <= 1024
        ):
            raise TemplateRegistryError("Invalid active KV budget or session limit")
        self._max_active_kv_bytes = max_active_kv_bytes
        self._max_active_sessions = max_active_sessions
        self._active_kv_bytes = 0
        self._active_count = 0
        self._admitted_leases = 0
        self._rejected_leases = 0
        self._released = 0

    def new_session(self, semantic_sha256: str) -> LeasedLlamaSession:
        key = self._key(semantic_sha256)
        with self._lock:
            owner = self._entries.get(key)
            if owner is None:
                self._misses += 1
                raise TemplateNotCached("Model not admitted to leased registry")
            plan = owner.template._plan
            p = plan.weights.profile
            required = (
                plan.batch * p.layers * plan.capacity
                * p.kv_heads * p.head_dim * 2 * 4
            )
            if (
                self._active_count >= self._max_active_sessions
                or self._active_kv_bytes + required > self._max_active_kv_bytes
            ):
                self._rejected_leases += 1
                raise LeaseBudgetExceeded("Active KV session logical quota exceeded")
            session = super().new_session(key)
            actual = sum(a.nbytes for a in session.cache._arrays.values())
            if actual != required:
                raise TemplateRegistryError("Session allocation breached KV byte contract")
            self._active_count += 1
            self._active_kv_bytes += required
            self._admitted_leases += 1
            return LeasedLlamaSession(self, session, required)

    def report(self) -> dict:
        with self._lock:
            return {
                **super().report(),
                "schema": "visuaml.leased-llama-registry.v26",
                "active_sessions": self._active_count,
                "active_kv_logical_bytes": self._active_kv_bytes,
                "max_active_sessions": self._max_active_sessions,
                "max_active_kv_bytes": self._max_active_kv_bytes,
                "admitted_leases": self._admitted_leases,
                "rejected_leases": self._rejected_leases,
                "released_leases": self._released,
                "quota_scope": (
                    "Logical private KV bytes of sessions acquired through this registry; "
                    "not physical RSS, compiled graph bytes or evicted-but-live model modules."
                ),
            }
