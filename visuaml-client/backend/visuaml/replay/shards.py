"""Merge ordered, request-owned worker journals without collecting a run in memory."""
from contextlib import ExitStack
from heapq import heapify, heappop, heappush

from .pipeline import compare_observations, read_observations


def _ordered(stream, shard):
    previous = None
    try:
        for line, row in enumerate(read_observations(stream), 1):
            request = row["request"]
            if previous is not None and request < previous:
                raise ValueError(f"Line {line} has request {request!r} after {previous!r}")
            previous = request
            yield dict(row, source=dict(shard=shard, line=line))
    except ValueError as error:
        raise ValueError(f"Shard {shard}: {error}") from error


def merge_shards(streams):
    """Yield rows in lexical request order with one lookahead per input stream.

    Each request must live in one shard, with its observations in execution
    order. Shard indices are zero-based and source line numbers are one-based.
    """
    pending = []
    for shard, stream in enumerate(streams):
        rows = _ordered(stream, shard)
        first = next(rows, None)
        if first is not None:
            pending.append((first["request"], shard, first, rows))
    heapify(pending)
    while pending:
        request, shard, row, rows = heappop(pending)
        if pending and pending[0][0] == request:
            raise ValueError(f"Request {request!r} appears in shards {shard} and {pending[0][1]}")
        while row is not None and row["request"] == request:
            yield row
            row = next(rows, None)
        if row is not None:
            heappush(pending, (row["request"], shard, row, rows))


def compare_shards(reference, candidate, *, tolerances=None):
    """Compare files from differently partitioned, lexically ordered runs."""
    with ExitStack() as stack:
        left = [stack.enter_context(path.open(encoding="utf-8")) for path in reference]
        right = [stack.enter_context(path.open(encoding="utf-8")) for path in candidate]
        result = compare_observations(merge_shards(left), merge_shards(right), tolerances=tolerances)
    return dict(result, reference_shards=len(left), candidate_shards=len(right))
