"""Direct journal writer and stable request-prefix Merkle comparison."""
from contextlib import ExitStack
from functools import lru_cache
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
from tempfile import NamedTemporaryFile
from urllib.parse import quote
from uuid import uuid4
import zlib

from ._codec import canonical
from .pipeline import MAX_OBSERVATION_BYTES, compare_observations
from .shards import merge_shards

_ROWS_PER_CHUNK = 256
_SCHEMA = 1
_CHILD_QUERY = "SELECT key,count,digest FROM nodes WHERE key IN ("+",".join(["?"]*256)+")"

@lru_cache(maxsize=512)
def _encoded_label(label):
    return canonical(label)


def _encoded_value(value):
    # Keep the exact bytes of the canonical encoder for the frequent integer case.
    if type(value) is int:
        return str(value).encode("ascii")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()



class JournalWriter:
    """Publish a complete indexed journal on successful context exit."""

    def __init__(self, path, *, prefix_bytes=6):
        if type(prefix_bytes) is not int or not 1 <= prefix_bytes <= 16:
            raise ValueError("Request prefix width must be 1–16 bytes")
        self.path = Path(path)
        self.width = prefix_bytes

    def __enter__(self):
        with NamedTemporaryFile(dir=self.path.parent, suffix=".partial", delete=False) as temp:
            self.pending = Path(temp.name)
        self.db = sqlite3.connect(self.pending)
        self.db.executescript("""
            PRAGMA journal_mode=OFF;
            CREATE TABLE metadata(version INTEGER, width INTEGER, artifact_id TEXT);
            CREATE TABLE nodes(key BLOB PRIMARY KEY, count INTEGER, digest BLOB) WITHOUT ROWID;
            CREATE TABLE chunks(key BLOB, line INTEGER, data BLOB, digest BLOB,
                                PRIMARY KEY(key,line)) WITHOUT ROWID;
        """)
        self.previous = None
        self.key = None
        self.count = self.leaf_count = 0
        self.rows = []
        self.semantics = sha256()
        return self

    def write(self, request, boundary, value, *, runtime, source=None):
        if (not isinstance(request, str) or not 1 <= len(request) <= 256
                or not isinstance(boundary, str) or not 1 <= len(boundary) <= 256
                or not isinstance(runtime, str) or not 1 <= len(runtime) <= 256):
            raise ValueError("Request, boundary and runtime must contain 1–256 characters")
        if self.previous is not None and request < self.previous:
            raise ValueError(f"Request {request!r} follows {self.previous!r} out of order")
        encoded = _encoded_value(value)
        if len(encoded) > MAX_OBSERVATION_BYTES:
            raise ValueError("Observation exceeds 4096 bytes")
        key = request.encode("utf-8", "surrogatepass")[:self.width].ljust(self.width, b"\0")
        if key != self.key:
            self._seal_leaf()
            self.key = key
        self.previous = request
        request_json = b'"'+request.encode("ascii")+b'"' if request.isascii() and request.isalnum() else canonical(request)
        semantic = b"["+request_json+b","+_encoded_label(boundary)+b","+encoded+b"]"
        provenance = (b'{"line":'+str(self.count+1).encode("ascii")+b',"shard":0}'
                      if source is None else canonical(source))
        self.rows.append(b"["+semantic+b","+_encoded_label(runtime)+b","+provenance+b"]\n")
        self.semantics.update(semantic+b"\n")
        self.count += 1
        self.leaf_count += 1
        if len(self.rows) == _ROWS_PER_CHUNK:
            self._flush()

    def _flush(self):
        if self.rows:
            packed = zlib.compress(b"".join(self.rows), level=1)
            self.db.execute("INSERT INTO chunks VALUES (?, ?, ?, ?)",
                            (self.key, self.count-len(self.rows)+1, packed, sha256(packed).digest()))
            self.rows.clear()

    def _seal_leaf(self):
        if self.key is not None:
            self._flush()
            self.db.execute("INSERT INTO nodes VALUES (?, ?, ?)",
                            (self.key, self.leaf_count, self.semantics.digest()))
            self.leaf_count = 0
            self.semantics = sha256()

    def _publish(self):
        self._seal_leaf()
        for depth in range(self.width, 0, -1):
            parent = None
            for key, count, digest in self.db.execute(
                    "SELECT key,count,digest FROM nodes WHERE length(key)=? ORDER BY key", (depth,)):
                if key[:-1] != parent:
                    if parent is not None:
                        self.db.execute("INSERT INTO nodes VALUES (?,?,?)", (parent, total, tree.digest()))
                    parent, total, tree = key[:-1], 0, sha256()
                tree.update(key[-1:]+count.to_bytes(8, "big")+digest)
                total += count
            if parent is not None:
                self.db.execute("INSERT INTO nodes VALUES (?,?,?)", (parent, total, tree.digest()))
        if not self.count:
            self.db.execute("INSERT INTO nodes VALUES (?,?,?)", (b"", 0, sha256().digest()))
        self.db.execute("CREATE INDEX child_depth ON nodes(length(key), key)")
        self.db.execute("INSERT INTO metadata VALUES (?,?,?)", (_SCHEMA, self.width, uuid4().hex))
        self.db.commit()

    def __exit__(self, kind, value, traceback):
        try:
            try:
                if kind is None:
                    self._publish()
            finally:
                self.db.close()
            if kind is None:
                os.replace(self.pending, self.path)
        finally:
            self.pending.unlink(missing_ok=True)


def build_tree(shards, output, *, prefix_bytes=6):
    """Convert existing worker JSONL in canonical request order."""
    shards = tuple(Path(path) for path in shards)
    output = Path(output)
    if any(path.resolve() == output.resolve() for path in shards):
        raise ValueError("The tree output must be separate from its input journals")
    with ExitStack() as stack:
        inputs = [stack.enter_context(path.open(encoding="utf-8")) for path in shards]
        with JournalWriter(output, prefix_bytes=prefix_bytes) as writer:
            for row in merge_shards(inputs):
                writer.write(**row)
    return {"observations": writer.count, "bytes": Path(output).stat().st_size}


class JournalTree:
    """Reuse a read-only connection across candidate comparisons."""

    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        uri = "file:"+quote(str(self.path.resolve()))+"?mode=ro&immutable=1"
        self.db = sqlite3.connect(uri, uri=True)
        try:
            meta = self.db.execute("SELECT * FROM metadata").fetchone()
            if meta is None or meta[0] != _SCHEMA or not 1 <= meta[1] <= 16:
                raise ValueError("Incomplete or unsupported request tree")
            self.chunks_read = self.payload_bytes = 0
            self.indexed_children = self.db.execute(
                "SELECT 1 FROM sqlite_schema WHERE name='child_depth' AND type='index'").fetchone() is not None
            self.width = meta[1]
            self.artifact_id = meta[2] if len(meta) == 3 else None
            self.root = self.db.execute("SELECT count,digest FROM nodes WHERE key=x''").fetchone()
            if self.root is None:
                raise ValueError("Missing request tree root")
        except (sqlite3.Error, ValueError) as error:
            self.db.close()
            raise ValueError(f"Invalid request tree {self.path}: {error}") from error
        return self

    def __exit__(self, *args):
        self.db.close()

    def children(self, prefix):
        if self.indexed_children:
            rows = self.db.execute(
                "SELECT key,count,digest FROM nodes WHERE length(key)=? AND key BETWEEN ? AND ?",
                (len(prefix)+1, prefix+b"\0", prefix+b"\xff"))
        else:
            keys = [prefix+bytes([branch]) for branch in range(256)]
            rows = self.db.execute(_CHILD_QUERY, keys)
        return {key: (count, digest) for key, count, digest in rows}

    def rows(self, key, *, later=False):
        if later:
            chunks = self.db.execute("SELECT data,digest FROM chunks WHERE key>? ORDER BY key,line", (key,))
        else:
            chunks = self.db.execute("SELECT data,digest FROM chunks WHERE key=? ORDER BY line", (key,))
        yield from self._decode(chunks)

    def all_rows(self, prefix=b""):
        width = self.width-len(prefix)
        chunks = self.db.execute("SELECT data,digest FROM chunks WHERE key BETWEEN ? AND ? ORDER BY key,line",
                                 (prefix+b"\0"*width, prefix+b"\xff"*width))
        yield from self._decode(chunks)

    def _decode(self, chunks):
        for data, digest in chunks:
            if sha256(data).digest() != digest:
                raise ValueError("Request tree chunk integrity mismatch")
            self.chunks_read += 1
            self.payload_bytes += len(data)
            decoder = zlib.decompressobj()
            decoded = decoder.decompress(data, _ROWS_PER_CHUNK*17_000+1)
            if not decoder.eof or decoder.unused_data or len(decoded) > _ROWS_PER_CHUNK*17_000:
                raise ValueError("Invalid request tree chunk size")
            for line in decoded.splitlines():
                (request, boundary, value), runtime, source = json.loads(line)
                yield dict(request=request, boundary=boundary, value=value, runtime=runtime, source=source)


def compare_trees(reference, candidate, *, tolerances=None):
    """Find the first disagreement using subtrees and the existing row comparator."""
    if reference.width != candidate.width:
        raise ValueError("Request trees must have the same prefix width")
    result = compare_observations([], [], tolerances=tolerances)
    visited = skipped = leaves = matched = 0
    start_reads = reference.chunks_read+candidate.chunks_read
    start_bytes = reference.payload_bytes+candidate.payload_bytes

    def visit(key, left, right):
        nonlocal visited, skipped, leaves, matched, result
        visited += 1
        if left == right:
            skipped += 1
            matched += left[0]
            return False
        if len(key) == reference.width:
            leaves += 1
            result = compare_observations(reference.rows(key), candidate.rows(key), tolerances=tolerances)
            matched += result["matched_observations"]
            first = result["first_difference"]
            if first is not None:
                # An exhausted leaf may have later rows. Look one row ahead
                # so missing leaves match the ordinary streaming diagnosis.
                for side, tree in (("reference", reference), ("candidate", candidate)):
                    if first[side] is None:
                        first[side] = next(tree.rows(key, later=True), None)
                if first["reference"] is not None and first["candidate"] is not None:
                    if first["reason"] == "missing-observation":
                        first["reason"] = "boundary"
                first["index"] = matched
                return True
            return False
        a = reference.children(key) if left is not None else {}
        b = candidate.children(key) if right is not None else {}
        return any(visit(child, a.get(child), b.get(child)) for child in sorted(a.keys() | b.keys()))

    visit(b"", reference.root, candidate.root)
    return dict(result, matched_observations=matched, visited_nodes=visited, skipped_subtrees=skipped,
                inspected_leaves=leaves, chunks_read=reference.chunks_read+candidate.chunks_read-start_reads,
                payload_bytes=reference.payload_bytes+candidate.payload_bytes-start_bytes)


class JournalForest:
    """Link disjoint worker-owned request ranges without copying payloads."""

    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        manifest = json.loads(self.path.read_text())
        if manifest.get("profile") != "visuaml-request-forest-v1":
            raise ValueError("Unsupported worker-tree manifest")
        self.stack = ExitStack()
        try:
            self.trees = [self.stack.enter_context(JournalTree(self.path.parent / worker["path"]))
                          for worker in manifest["workers"]]
            if any(tree.root[1].hex() != worker["root"] or tree.artifact_id != worker["artifact_id"]
                   for tree, worker in zip(self.trees, manifest["workers"], strict=True)):
                raise ValueError("Worker tree root changed after linking")
            self._bind(manifest["partition_bytes"])
        except BaseException:
            self.stack.close()
            raise
        return self

    def _bind(self, depth):
        if not self.trees or type(depth) is not int:
            raise ValueError("Expected worker trees and a partition width")
        self.width = self.trees[0].width
        if not 1 <= depth <= self.width or any(tree.width != self.width for tree in self.trees):
            raise ValueError("Invalid worker partition width or mixed tree widths")
        self.depth, self.owners, self.upper = depth, {}, {}
        for shard, tree in enumerate(self.trees):
            for key, count, digest in tree.db.execute(
                    "SELECT key,count,digest FROM nodes WHERE length(key)=?", (depth,)):
                if key in self.owners:
                    raise ValueError(f"Workers share request prefix {key.hex()}")
                self.owners[key] = shard
                self.upper[key] = (count, digest)
        for level in range(depth, 0, -1):
            groups = {}
            for key in sorted(self.upper):
                if len(key) == level:
                    count, digest = self.upper[key]
                    total, hasher = groups.setdefault(key[:-1], (0, sha256()))
                    hasher.update(key[-1:]+count.to_bytes(8, "big")+digest)
                    groups[key[:-1]] = (total+count, hasher)
            self.upper.update((key, (total, hasher.digest())) for key, (total, hasher) in groups.items())
        self.root = self.upper.get(b"", (0, sha256().digest()))
        self.links = {}
        for key, node in self.upper.items():
            if key:
                self.links.setdefault(key[:-1], {})[key] = node

    def __exit__(self, *args):
        self.stack.close()

    @property
    def chunks_read(self):
        return sum(tree.chunks_read for tree in self.trees)

    @property
    def payload_bytes(self):
        return sum(tree.payload_bytes for tree in self.trees)

    def children(self, prefix):
        if len(prefix) < self.depth:
            return self.links.get(prefix, {})
        return self.trees[self.owners[prefix[:self.depth]]].children(prefix)

    def all_rows(self):
        for prefix, shard in sorted(self.owners.items()):
            for row in self.trees[shard].all_rows(prefix):
                yield dict(row, source=dict(row["source"], shard=shard))

    def rows(self, key, *, later=False):
        owner = self.owners.get(key[:self.depth])
        if later:
            # Only the first successor is needed by the comparator.
            candidates = [(row, shard) for shard, tree in enumerate(self.trees)
                          if (row := next(tree.rows(key, later=True), None)) is not None]
            if candidates:
                row, shard = min(candidates, key=lambda pair: pair[0]["request"])
                yield dict(row, source=dict(row["source"], shard=shard))
        elif owner is not None:
            for row in self.trees[owner].rows(key):
                yield dict(row, source=dict(row["source"], shard=owner))


def open_tree(path):
    path = Path(path)
    return JournalForest(path) if path.suffix == ".json" else JournalTree(path)


def link_trees(paths, output, *, partition_bytes):
    """Publish a manifest over immutable workers partitioned by request prefix."""
    output = Path(output)
    if output.suffix != ".json":
        raise ValueError("Worker manifest must use a .json extension")
    paths = tuple(Path(path) for path in paths)
    if any(path.resolve() == output.resolve() for path in paths):
        raise ValueError("Worker manifest must be separate from its input files")
    with ExitStack() as stack:
        forest = JournalForest(output)
        forest.trees = [stack.enter_context(JournalTree(path)) for path in paths]
        if any(tree.artifact_id is None for tree in forest.trees):
            raise ValueError("Worker tree must carry a file identity")
        forest._bind(partition_bytes)
        manifest = dict(profile="visuaml-request-forest-v1", partition_bytes=partition_bytes, workers=[
            dict(path=os.path.relpath(path, output.parent), root=tree.root[1].hex(), artifact_id=tree.artifact_id)
            for path, tree in zip(paths, forest.trees, strict=True)])
        with NamedTemporaryFile(dir=output.parent, suffix=".partial", delete=False) as temp:
            pending = Path(temp.name)
            try:
                temp.write(canonical(manifest)+b"\n")
                temp.close()
                os.replace(pending, output)
            finally:
                pending.unlink(missing_ok=True)
    return {"observations": forest.root[0], "workers": len(paths), "owned_ranges": len(forest.owners)}
