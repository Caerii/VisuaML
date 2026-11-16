"""v27: account for distinct model modules retained by active inference leases.

Evicting a cached model does not evict its live sessions. A registry's cached
weight budget therefore understates the logical weight footprint. Here we
retain one strong reference and one exact module-byte count per distinct
*verified template object* while any associated session remains active.

Identity is by template object, not merely semantic digest: a model evicted
and later re-admitted can have two distinct modules alive concurrently.
Accounting remains logical Torch buffer bytes, not measured process RSS.
"""

from __future__ import annotations

from dataclasses import dataclass

from .session_leases_v26 import (
    LeaseManagedLlamaRegistry,
)
from .shared_weights_v24 import SharedWeightsLlamaTemplate
from .template_registry_v25 import TemplateRegistryError


class ActiveModelBudgetExceeded(TemplateRegistryError):
    """A session's model module would exceed the logical active-weight budget."""


@dataclass
class _ActiveTemplate:
    template: SharedWeightsLlamaTemplate
    weight_bytes: int
    leases: int


class ActiveModelLlamaRegistry(LeaseManagedLlamaRegistry):
    """Quota accounting across cached and evicted-but-live compiled models."""

    def __init__(
        self, *, max_cached_weight_bytes: int = 256 * 1024 * 1024,
        max_models: int = 4, max_active_kv_bytes: int = 64 * 1024 * 1024,
        max_active_sessions: int = 16,
        max_active_model_weight_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        super().__init__(
            max_cached_weight_bytes=max_cached_weight_bytes,
            max_models=max_models,
            max_active_kv_bytes=max_active_kv_bytes,
            max_active_sessions=max_active_sessions,
        )
        if (
            type(max_active_model_weight_bytes) is not int
            or not 4096 <= max_active_model_weight_bytes <= 2 * 1024 * 1024 * 1024
        ):
            raise TemplateRegistryError("Invalid active model weight budget")
        self._max_active_model_weight_bytes = max_active_model_weight_bytes
        self._active_weight_bytes = 0
        self._active_templates: dict[int, _ActiveTemplate] = {}
        self._rejected_weight_leases = 0

    def _check_template_lease(self, template: object) -> None:
        if not isinstance(template, SharedWeightsLlamaTemplate):
            raise TemplateRegistryError("Unverified model template cannot acquire lease")
        identity = id(template)
        existing = self._active_templates.get(identity)
        if existing is not None:
            if existing.template is not template:
                raise TemplateRegistryError("Template identity reused unexpectedly")
            return
        bytes_ = template.report()["weight_module_logical_bytes"]
        if (
            type(bytes_) is not int or bytes_ <= 0
            or self._active_weight_bytes + bytes_ > self._max_active_model_weight_bytes
        ):
            self._rejected_weight_leases += 1
            raise ActiveModelBudgetExceeded("Active model weights quota exceeded")

    def _register_template_lease(self, template: object) -> None:
        # This hook runs only under the same lock after successful private KV
        # allocation. The preflight check cannot race another admission.
        identity = id(template)
        current = self._active_templates.get(identity)
        if current is None:
            bytes_ = template.report()["weight_module_logical_bytes"]
            self._active_templates[identity] = _ActiveTemplate(template, bytes_, 1)
            self._active_weight_bytes += bytes_
        else:
            if current.template is not template:
                raise TemplateRegistryError("Template identity mismatch")
            current.leases += 1

    def _release_template_lease(self, template_id: int | None) -> None:
        if template_id is None:
            return
        current = self._active_templates.get(template_id)
        if current is None or current.leases <= 0:
            raise TemplateRegistryError("Active model lease accounting underflow")
        current.leases -= 1
        if current.leases == 0:
            self._active_weight_bytes -= current.weight_bytes
            del self._active_templates[template_id]

    def report(self) -> dict:
        with self._lock:
            return {
                **super().report(),
                "schema": "visuaml.active-model-lifetime.v27",
                "distinct_active_model_modules": len(self._active_templates),
                "active_model_weight_logical_bytes": self._active_weight_bytes,
                "max_active_model_weight_bytes": self._max_active_model_weight_bytes,
                "active_model_lease_counts": {
                    str(identity): {
                        "semantic_sha256": record.template.semantic_sha256,
                        "leases": record.leases,
                        "weight_module_logical_bytes": record.weight_bytes,
                        "still_cached": any(
                            e.template is record.template for e in self._entries.values()
                        ),
                    }
                    for identity, record in self._active_templates.items()
                },
                "rejected_weight_leases": self._rejected_weight_leases,
                "limitations": [
                    "Includes unique model-module buffers held by managed active leases.",
                    "Does not count evicted idle templates held by other external callers.",
                    "Not total RSS, JIT graph memory, activations or OS-enforced isolation.",
                    "The same module counts once regardless of number of its live leases.",
                ],
            }
