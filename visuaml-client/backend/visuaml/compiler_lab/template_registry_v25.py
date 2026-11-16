"""v25: bounded LRU registry for verified compiled Llama model templates.

Caches only closed-world, checkpoint-bound SharedWeightsLlamaTemplate objects.
A single process-wide lock prevents racing admissions or duplicate traces.
Eviction removes registry ownership; in-flight sessions retain their own
read-only graph/module references and private resident KV state.

The logical weight-module byte cap is a cache-residency planning budget,
NOT a hard cap on physical RSS, JIT-generated memory, active session KV,
or evicted executables retained by live callers.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock

from .llama_semantic_v20 import (
    LlamaSemanticPlan,
    lower_llama_semantic_program,
)
from .shared_weights_v24 import SharedWeightsLlamaTemplate
from .torchscript_graph_v21 import TorchScriptWholeGraphSession


class TemplateRegistryError(ValueError):
    """Model admission, identity, or logical cache ownership is invalid."""


class TemplateNotCached(TemplateRegistryError):
    """Requested verified model has no current registry ownership."""


@dataclass(frozen=True)
class _OwnedTemplate:
    template: SharedWeightsLlamaTemplate
    checkpoint_sha256: str
    weight_module_logical_bytes: int


class VerifiedLlamaRegistry:
    """Synchronous, bounded LRU cache of already verified executable templates."""

    def __init__(
        self, *, max_cached_weight_bytes: int = 256 * 1024 * 1024,
        max_models: int = 4,
    ) -> None:
        if (
            type(max_cached_weight_bytes) is not int
            or not 4096 <= max_cached_weight_bytes <= 2 * 1024 * 1024 * 1024
            or type(max_models) is not int or not 1 <= max_models <= 64
        ):
            raise TemplateRegistryError("Invalid cache byte or model count bounds")
        self._max_cached_weight_bytes = max_cached_weight_bytes
        self._max_models = max_models
        self._entries: OrderedDict[str, _OwnedTemplate] = OrderedDict()
        self._lock = RLock()
        self._owned_bytes = 0
        self._evictions = 0
        self._admissions = 0
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _key(key: str) -> str:
        if not isinstance(key, str) or re.fullmatch(r"[a-f0-9]{64}", key) is None:
            raise TemplateRegistryError("Require canonical 64-character semantic digest")
        return key

    def register(self, plan: LlamaSemanticPlan) -> str:
        """Verify the complete checkpoint/IR, then atomically admit or reuse it.

        Failed validation, oversize plans, and failed initial numerical
        certification leave the existing LRU contents untouched.
        """
        if not isinstance(plan, LlamaSemanticPlan):
            raise TemplateRegistryError("Expected compiler-validated Llama plan")
        with self._lock:
            admitted = lower_llama_semantic_program(
                plan.program, plan.weights, batch=plan.batch
            )
            key = admitted.semantic_sha256
            existing = self._entries.get(key)
            if existing is not None:
                if existing.checkpoint_sha256 != admitted.weights.checkpoint_sha256:
                    raise TemplateRegistryError("Semantic key collision with checkpoint")
                self._entries.move_to_end(key)
                self._hits += 1
                return key

            # Estimate before doing a TorchScript trace; exact module tensor
            # bytes are read from the verified template after publication.
            p = admitted.weights.profile
            candidate_bytes = (
                sum(array.nbytes for array in admitted.weights.arrays.values())
                + 2 * p.head_dim * admitted.capacity * 4
            )
            if candidate_bytes > self._max_cached_weight_bytes:
                raise TemplateRegistryError("Model exceeds logical template cache budget")
            candidate = SharedWeightsLlamaTemplate(admitted)
            candidate.new_session()  # Compiles and independently verifies source.
            cost = candidate.report()["weight_module_logical_bytes"]
            if (
                type(cost) is not int or cost <= 0
                or cost > self._max_cached_weight_bytes
            ):
                raise TemplateRegistryError("Verified module violates cache byte contract")
            while self._entries and (
                len(self._entries) >= self._max_models
                or self._owned_bytes + cost > self._max_cached_weight_bytes
            ):
                _, victim = self._entries.popitem(last=False)
                self._owned_bytes -= victim.weight_module_logical_bytes
                self._evictions += 1
            self._entries[key] = _OwnedTemplate(
                candidate, admitted.weights.checkpoint_sha256, cost
            )
            self._owned_bytes += cost
            self._admissions += 1
            return key

    def new_session(self, semantic_sha256: str) -> TorchScriptWholeGraphSession:
        """Acquire one private KV session using a verified cached executable."""
        key = self._key(semantic_sha256)
        with self._lock:
            owner = self._entries.get(key)
            if owner is None:
                self._misses += 1
                raise TemplateNotCached("No admitted template for semantic digest")
            self._entries.move_to_end(key)
            self._hits += 1
            return owner.template.new_session()

    def evict(self, semantic_sha256: str) -> bool:
        """Explicitly discard registry ownership; active sessions remain valid."""
        key = self._key(semantic_sha256)
        with self._lock:
            owner = self._entries.pop(key, None)
            if owner is None:
                return False
            self._owned_bytes -= owner.weight_module_logical_bytes
            self._evictions += 1
            return True

    def report(self) -> dict:
        with self._lock:
            return {
                "schema": "visuaml.verified-template-registry.v25",
                "models": len(self._entries),
                "max_models": self._max_models,
                "cached_weight_module_logical_bytes": self._owned_bytes,
                "max_cached_weight_bytes": self._max_cached_weight_bytes,
                "entries_lru_to_mru": [
                    {
                        "semantic_sha256": key,
                        "checkpoint_sha256": owner.checkpoint_sha256,
                        "weight_module_logical_bytes": owner.weight_module_logical_bytes,
                        "compiled_graphs": owner.template.report()["compiled_graphs"],
                    }
                    for key, owner in self._entries.items()
                ],
                "admissions": self._admissions,
                "cache_hits": self._hits,
                "cache_misses": self._misses,
                "evictions": self._evictions,
                "limitations": [
                    "Logical weights retained by registry only; not physical process RSS.",
                    "Eviction does not cancel or reclaim active external sessions.",
                    "JIT executable overhead, PyTorch allocator and per-session KV not in byte cap.",
                    "Trusted in-process admission; arbitrary Python code is not sandboxed.",
                ],
            }
