"""Immutable block indexes for repeated ordered pipeline comparisons."""
from contextlib import ExitStack
from hashlib import sha256
from itertools import zip_longest
import json
import os
from struct import Struct
from tempfile import NamedTemporaryFile

from ._codec import canonical
from .pipeline import compare_observations
from .shards import merge_shards

BLOCK_ROWS = 256
_MAGIC = b"VMLIDX01"
_ENTRY = Struct(">QQ32s32s")
_FOOTER = Struct(">8sQQ32s")
_MAX_BLOCK_BYTES = BLOCK_ROWS * 17_000


def build_index(paths, output):
    """Atomically publish a validated journal snapshot with block digests."""
    paths = tuple(paths)
    if any(path.resolve() == output.resolve() for path in paths):
        raise ValueError("The index output must be separate from its input journals")
    with ExitStack() as stack:
        inputs = [stack.enter_context(path.open(encoding="utf-8")) for path in paths]
        temp = stack.enter_context(NamedTemporaryFile(dir=output.parent, delete=False))
        try:
            entries = []
            start = count = 0
            semantic, stored = sha256(), sha256()

            def finish_block():
                length = temp.tell()-start
                entries.append(_ENTRY.pack(start, length, semantic.digest(), stored.digest()))

            for row in merge_shards(inputs):
                if count and count % BLOCK_ROWS == 0:
                    finish_block()
                    start = temp.tell()
                    semantic, stored = sha256(), sha256()
                encoded = canonical(row)+b"\n"
                stats = [row[key] for key in ("request", "boundary", "value")]
                semantic.update(canonical(stats)+b"\n")
                stored.update(encoded)
                temp.write(encoded)
                count += 1
            if count:
                finish_block()
            offset = temp.tell()
            table = sha256()
            for entry in entries:
                table.update(entry)
                temp.write(entry)
            temp.write(_FOOTER.pack(_MAGIC, offset, count, table.digest()))
            temp.flush()
            temp.close()
            os.replace(temp.name, output)
        finally:
            if os.path.exists(temp.name):
                os.unlink(temp.name)
    return dict(observations=count, blocks=len(entries), block_rows=BLOCK_ROWS, bytes=output.stat().st_size)


class _Index:
    def __init__(self, stream):
        self.stream = stream
        size = stream.seek(0, os.SEEK_END)
        if size < _FOOTER.size:
            raise ValueError("Truncated journal index")
        stream.seek(size-_FOOTER.size)
        magic, self.offset, self.count, checksum = _FOOTER.unpack(stream.read(_FOOTER.size))
        self.blocks = (self.count+BLOCK_ROWS-1)//BLOCK_ROWS
        if magic != _MAGIC or self.offset+self.blocks*_ENTRY.size+_FOOTER.size != size:
            raise ValueError("Invalid journal index footer")
        self.metadata = 0
        self.bytes_read = 0
        table = sha256()
        for data in self._entry_batches():
            table.update(data)
        if table.digest() != checksum:
            raise ValueError("Journal index metadata integrity mismatch")

    def _entry_batches(self):
        # Use bounded batches, and retain an independent position from the payload reads.
        for begin in range(0, self.blocks, 1024):
            count = min(self.blocks-begin, 1024)
            self.stream.seek(self.offset+begin*_ENTRY.size)
            data = self.stream.read(count*_ENTRY.size)
            if len(data) != count*_ENTRY.size:
                raise ValueError("Truncated journal index metadata")
            self.metadata += len(data)
            yield data

    def entries(self):
        for data in self._entry_batches():
            yield from _ENTRY.iter_unpack(data)

    def read_block(self, entry):
        if entry is None:
            return []
        offset, size, _, expected = entry
        if not 0 < size <= _MAX_BLOCK_BYTES or offset+size > self.offset:
            raise ValueError("Invalid journal index block range")
        self.stream.seek(offset)
        data = self.stream.read(size)
        self.bytes_read += len(data)
        if sha256(data).digest() != expected:
            raise ValueError("Journal index block integrity mismatch")
        return [json.loads(row) for row in data.splitlines()]


def compare_indexes(reference, candidate, *, tolerances=None):
    """Skip matching semantic blocks and use the exact comparator on unequal blocks.

    Use indexes produced by the trusted capture pipeline and kept immutable.
    The scan validates payload digests for blocks it opens.
    """
    result = compare_observations([], [], tolerances=tolerances)
    skipped = inspected = matched = 0
    with reference.open("rb") as left, candidate.open("rb") as right:
        lhs, rhs = _Index(left), _Index(right)
        for block, (a, b) in enumerate(zip_longest(lhs.entries(), rhs.entries())):
            left_count = min(BLOCK_ROWS, max(0, lhs.count-block*BLOCK_ROWS))
            right_count = min(BLOCK_ROWS, max(0, rhs.count-block*BLOCK_ROWS))
            if a and b and a[2] == b[2] and left_count == right_count:
                skipped += 1
                matched += left_count
                continue
            inspected += 1
            left_rows, right_rows = lhs.read_block(a), rhs.read_block(b)
            if len(left_rows) != left_count or len(right_rows) != right_count:
                raise ValueError("Journal index block row count mismatch")
            result = compare_observations(left_rows, right_rows, tolerances=tolerances)
            matched += result["matched_observations"]
            if result["first_difference"] is not None:
                result["first_difference"]["index"] = matched
                break
        result.update(matched_observations=matched, skipped_blocks=skipped, inspected_blocks=inspected,
                      metadata_bytes=lhs.metadata+rhs.metadata, payload_bytes=lhs.bytes_read+rhs.bytes_read)
    return result


def verify_index(path):
    """Check every stored block, including ranges a comparison could skip."""
    with path.open("rb") as stream:
        index = _Index(stream)
        total = 0
        for entry in index.entries():
            rows = index.read_block(entry)
            expected_count = min(BLOCK_ROWS, index.count-total)
            semantic = sha256()
            for row in rows:
                semantic.update(canonical([row[key] for key in ("request", "boundary", "value")])+b"\n")
            if len(rows) != expected_count or semantic.digest() != entry[2]:
                raise ValueError("Journal index semantic integrity mismatch")
            total += len(rows)
    return dict(observations=total, verified_blocks=index.blocks, payload_bytes=index.bytes_read)
