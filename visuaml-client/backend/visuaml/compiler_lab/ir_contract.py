"""Versioned, closed-world portable IR validation and resource admission.

The NumPy manifest is an explicitly ordered SSA-like program: values are
introduced once, and every reference must point to a previously defined value.
This verifier checks the *structure* prior to running any operator. It is not a
Python sandbox or a proof of numerical equivalence, and it does not make an
untrusted archive safe to load in an unrestricted process.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Mapping


class IRContractError(ValueError):
    """Invalid typed-program representation or resource admission."""


@dataclass(frozen=True)
class ResourcePolicy:
    max_nodes: int = 10_000
    max_inputs: int = 64
    max_rank: int = 16
    max_tensor_elements: int = 50_000_000
    max_constant_bytes: int = 256 * 1024 * 1024
    max_input_bytes: int = 256 * 1024 * 1024
    max_manifest_bytes: int = 8 * 1024 * 1024
    max_archive_members: int = 10000
    max_retained_value_bytes: int = 512 * 1024 * 1024


DEFAULT_POLICY = ResourcePolicy()


@dataclass(frozen=True)
class IRValidation:
    schema: str
    inputs: tuple[str, ...]
    constants: tuple[str, ...]
    calls: tuple[str, ...]
    outputs: tuple[str, ...]
    op_histogram: tuple[tuple[str, int], ...]
    declared_input_bytes: int
    declared_constant_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            'schema': self.schema, 'inputs': list(self.inputs),
            'constants': list(self.constants), 'calls': list(self.calls),
            'outputs': list(self.outputs),
            'op_histogram': dict(self.op_histogram),
            'declared_input_bytes': self.declared_input_bytes,
            'declared_constant_bytes': self.declared_constant_bytes,
        }


def _identifier(x: Any) -> str:
    if not isinstance(x, str) or not x or len(x) > 256:
        raise IRContractError('SSA value name must be a nonempty string <=256 characters')
    return x


def _shape(shape: Any, policy: ResourcePolicy) -> tuple[int, ...]:
    if not isinstance(shape, list) or len(shape) > policy.max_rank:
        raise IRContractError('Invalid shape rank')
    if any(type(v) is not int or v < 0 for v in shape):
        raise IRContractError('Only nonnegative concrete dimensions are admitted')
    if prod(shape) > policy.max_tensor_elements:
        raise IRContractError('Tensor element budget exceeded')
    return tuple(shape)


def _expr_refs(expr: Any, defined: set[str], *, depth: int = 0) -> set[str]:
    if depth > 64:
        raise IRContractError('Expression nesting limit exceeded')
    if expr is None or type(expr) in (str, int, float, bool):
        return set()
    if isinstance(expr, list):
        refs: set[str] = set()
        for v in expr:
            refs.update(_expr_refs(v, defined, depth=depth + 1))
        return refs
    if not isinstance(expr, dict) or len(expr) != 1:
        raise IRContractError('Malformed expression tree')
    tag, value = next(iter(expr.items()))
    if tag == 'ref':
        name = _identifier(value)
        if name not in defined:
            raise IRContractError(f'Use before definition or unknown SSA value: {name}')
        return {name}
    if tag == 'tuple':
        if not isinstance(value, list):
            raise IRContractError('Tuple must contain a list')
        return _expr_refs(value, defined, depth=depth + 1)
    if tag == 'slice':
        if not isinstance(value, list) or len(value) != 3:
            raise IRContractError('Slice must have start, stop, step')
        return _expr_refs(value, defined, depth=depth + 1)
    if tag == 'map':
        if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
            raise IRContractError('Map keys must be strings')
        refs = set()
        for v in value.values():
            refs.update(_expr_refs(v, defined, depth=depth + 1))
        return refs
    raise IRContractError(f'Unsupported expression tag: {tag}')


def validate_portable_ir(manifest: Mapping[str, Any], constants: Mapping[str, Any],
                         *, policy: ResourcePolicy = DEFAULT_POLICY) -> IRValidation:
    """Reject cyclic, unbound, malformed, over-budget, or unknown-op manifests.

    Digest verification is a separate prerequisite when loading from disk.
    """
    # Local imports avoid importing PyTorch and preserve standalone runtime.
    import numpy as np
    try:
        from .numpy_runtime import OPS, SCHEMA
    except ImportError:  # standalone runtime directory, without a package context
        from numpy_runtime import OPS, SCHEMA

    if manifest.get('schema') != SCHEMA:
        raise IRContractError('Unsupported portable IR schema')
    allowed = {'schema', 'source', 'numeric_contract', 'inputs', 'nodes', 'output',
               'output_tree', 'constant_hashes', 'manifest_sha256'}
    if set(manifest) - allowed:
        raise IRContractError('Unknown top-level manifest fields')
    ins, nodes = manifest.get('inputs'), manifest.get('nodes')
    if not isinstance(ins, list) or not isinstance(nodes, list):
        raise IRContractError('IR inputs and nodes must be ordered lists')
    if len(ins) > policy.max_inputs or len(nodes) > policy.max_nodes:
        raise IRContractError('IR node or input budget exceeded')
    defined: set[str] = set()
    input_names: list[str] = []
    total_inputs = 0
    for item in ins:
        if not isinstance(item, dict) or set(item) != {'name', 'shape', 'dtype'}:
            raise IRContractError('Malformed input contract')
        name = _identifier(item['name'])
        if name in defined:
            raise IRContractError(f'Duplicate SSA value: {name}')
        try:
            dtype = np.dtype(item['dtype'])
        except (TypeError, ValueError) as exc:
            raise IRContractError('Unrecognized input dtype') from exc
        if dtype.hasobject or dtype.kind not in 'biufc':
            raise IRContractError('Input dtype is not numeric')
        shape = _shape(item['shape'], policy)
        total_inputs += prod(shape) * dtype.itemsize
        if total_inputs > policy.max_input_bytes:
            raise IRContractError('Declared input byte budget exceeded')
        input_names.append(name)
        defined.add(name)

    const_names: list[str] = []
    call_names: list[str] = []
    hist: dict[str, int] = {}
    total_constants = 0
    for item in nodes:
        if not isinstance(item, dict) or not {'name', 'kind', 'op'} <= item.keys():
            raise IRContractError('Malformed instruction')
        name = _identifier(item['name'])
        if name in defined:
            raise IRContractError(f'Duplicate SSA value: {name}')
        kind = item['kind']
        op = item['op']
        if kind == 'constant':
            if set(item) != {'name', 'kind', 'op'} or op != 'constant' or name not in constants:
                raise IRContractError('Malformed or missing constant')
            arr = constants[name]
            if not isinstance(arr, np.ndarray) or arr.dtype.hasobject or arr.dtype.kind not in 'biufc':
                raise IRContractError('Invalid constant dtype')
            _shape(list(arr.shape), policy)
            total_constants += arr.nbytes
            if total_constants > policy.max_constant_bytes:
                raise IRContractError('Constant memory budget exceeded')
            const_names.append(name)
        elif kind == 'call':
            if set(item) != {'name', 'kind', 'op', 'args', 'kwargs'} or op not in OPS:
                raise IRContractError(f'Unsupported or malformed operator: {op}')
            if not isinstance(item['kwargs'], dict) or set(item['kwargs']) != {'map'}:
                raise IRContractError('Instruction keyword arguments must use tagged map')
            _expr_refs(item['args'], defined)
            _expr_refs(item['kwargs'], defined)
            call_names.append(name)
            hist[op] = hist.get(op, 0) + 1
        else:
            raise IRContractError(f'Unsupported instruction kind: {kind}')
        defined.add(name)
    if set(constants) != set(const_names):
        raise IRContractError('Orphan or undeclared constant arrays')
    outputs = sorted(_expr_refs(manifest.get('output'), defined))
    if not outputs:
        raise IRContractError('Program output does not reference a computed value')
    return IRValidation(SCHEMA, tuple(input_names), tuple(const_names), tuple(call_names),
                        tuple(outputs), tuple(sorted(hist.items())), total_inputs, total_constants)
