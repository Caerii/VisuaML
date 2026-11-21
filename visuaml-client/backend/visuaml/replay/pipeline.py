"""Runtime-neutral, ordered observations for pipeline regression harnesses."""
from hashlib import sha256
from itertools import zip_longest
import json
from math import isfinite

from .checkpoint import canonical

SCHEMA = "visuaml-pipeline-comparison-v1"
MAX_OBSERVATION_BYTES = 4096
MAX_LINE_CHARS = 16384
_MISSING = object()


def observation(request, boundary, value, *, runtime):
    """Capture a small JSON value. Store larger tensors in your harness by digest."""
    if not all(isinstance(key, str) and 1 <= len(key) <= 256 for key in (request, boundary, runtime)):
        raise ValueError("Request, boundary and runtime must contain 1–256 characters")
    try:
        data = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    except (TypeError, ValueError) as error:
        raise ValueError("Observation value must be finite JSON") from error
    if len(data) > MAX_OBSERVATION_BYTES:
        raise ValueError("Observation exceeds 4096 bytes; record a digest or a selected slice")
    return dict(request=request, boundary=boundary, runtime=runtime, value=json.loads(data))


def _matches(reference, candidate, rtol, atol):
    if isinstance(reference, bool) or isinstance(candidate, bool):
        return type(reference) is type(candidate) and reference == candidate
    if isinstance(reference, int) and isinstance(candidate, int):
        return reference == candidate
    if isinstance(reference, (float, int)) and isinstance(candidate, (float, int)):
        return abs(reference-candidate) <= atol + rtol*abs(reference)
    if type(reference) is not type(candidate):
        return False
    if isinstance(reference, list):
        return len(reference) == len(candidate) and all(
            _matches(a, b, rtol, atol) for a, b in zip(reference, candidate))
    if isinstance(reference, dict):
        return reference.keys() == candidate.keys() and all(
            _matches(value, candidate[key], rtol, atol) for key, value in reference.items())
    return reference == candidate


def compare_observations(reference, candidate, *, tolerances=None):
    """Compare two ordered streams using constant storage and stop at the first mismatch.

    Streams must already have a common logical ordering. Runtime labels are
    provenance. Tolerances are keyed by boundary and only adjust float comparisons.
    """
    rules = {}
    if tolerances is not None and not isinstance(tolerances, dict):
        raise ValueError("Expected a mapping of boundary names to [rtol, atol]")
    for boundary, pair in (tolerances or {}).items():
        if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                or not all(type(v) in (int, float) and 0 <= v <= 1e308 and isfinite(v) for v in pair)):
            raise ValueError("Tolerances must be finite and nonnegative")
        rules[boundary] = tuple(pair)
    count = 0
    difference = None
    for left, right in zip_longest(reference, candidate, fillvalue=_MISSING):
        if left is _MISSING or right is _MISSING:
            reason = "missing-observation"
        elif (left["request"], left["boundary"]) != (right["request"], right["boundary"]):
            reason = "boundary"
        else:
            rtol, atol = rules.get(left["boundary"], (0, 0))
            if _matches(left["value"], right["value"], rtol, atol):
                count += 1
                continue
            reason = "value"
        difference = dict(index=count, reason=reason,
                          reference=None if left is _MISSING else left,
                          candidate=None if right is _MISSING else right)
        break
    return dict(profile=SCHEMA, matched_observations=count, first_difference=difference, tolerances=rules)


def read_observations(stream):
    """Read one observation per JSON Lines entry with useful parse diagnostics."""
    for line_number, line in enumerate(iter(lambda: stream.readline(MAX_LINE_CHARS+1), ""), 1):
        try:
            if len(line) > MAX_LINE_CHARS:
                raise ValueError("Observation line exceeds 16,384 characters")
            record = json.loads(line)
            if not isinstance(record, dict) or set(record) != {"request", "boundary", "runtime", "value"}:
                raise ValueError("Expected request, boundary, runtime, and value")
            yield observation(record["request"], record["boundary"], record["value"],
                              runtime=record["runtime"])
        except ValueError as error:
            raise ValueError(f"Observation line {line_number}: {error}") from error


def tensor_identity(array):
    """Identify a numerical NumPy tensor without retaining its contents."""
    import numpy as np
    tensor = np.asarray(array)
    if tensor.dtype.kind not in "biufc":
        raise ValueError("Expected a numeric or boolean tensor")
    metadata = dict(shape=list(tensor.shape), dtype=tensor.dtype.str)
    hash_ = sha256(canonical(metadata))
    hash_.update(memoryview(np.ascontiguousarray(tensor).reshape(-1)).cast("B"))
    return dict(metadata, sha256=hash_.hexdigest())
