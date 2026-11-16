"""Strict local SentencePiece/Llama tokenizer bridge for admitted real checkpoints.

Loads only tokenizer.model and tokenizer_config.json from a controlled local
directory. No model Python imports, pickle, remote fetches or arbitrary code.
Hashes record byte identity; publisher identity remains a separate trust gate.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class TokenizerContractError(ValueError):
    """Tokenizer files or requested encoding are outside the admitted contract."""


@dataclass(frozen=True)
class TokenizerIdentity:
    declared_revision: str
    model_sha256: str
    config_sha256: str
    vocabulary: int
    bos_id: int
    eos_id: int
    unk_id: int


class LlamaSentencePieceBridge:
    """Bounded Unicode/SentencePiece text encoder for one immutable local model."""

    def __init__(self, processor: object, identity: TokenizerIdentity):
        self._processor = processor
        self.identity = identity

    def encode(
        self, text: str, *, add_bos: bool = True, add_eos: bool = False
    ) -> np.ndarray:
        if (
            not isinstance(text, str) or not 1 <= len(text) <= 2048
            or type(add_bos) is not bool or type(add_eos) is not bool
        ):
            raise TokenizerContractError("Text or special-token mode is unsupported")
        ids = list(self._processor.encode(text, out_type=int))
        if add_bos:
            ids.insert(0, self.identity.bos_id)
        if add_eos:
            ids.append(self.identity.eos_id)
        if (
            not ids or len(ids) > 256
            or any(type(x) is not int or not 0 <= x < self.identity.vocabulary for x in ids)
        ):
            raise TokenizerContractError("Encoded sequence exceeds admitted token budget")
        return np.asarray(ids, dtype=np.int64)[None, :]

    def decode(self, ids: np.ndarray) -> str:
        if (
            not isinstance(ids, np.ndarray) or ids.ndim != 1
            or ids.dtype != np.int64 or not 1 <= ids.size <= 256
            or np.any(ids < 0) or np.any(ids >= self.identity.vocabulary)
        ):
            raise TokenizerContractError("Unsupported tokenizer decode input")
        return str(self._processor.decode(ids.tolist()))


def _bounded(path: Path, max_bytes: int) -> bytes:
    if (
        path.is_symlink() or not path.is_file()
        or not 0 < path.stat().st_size <= max_bytes
    ):
        raise TokenizerContractError("Missing, linked, or oversized tokenizer file")
    return path.read_bytes()


def _unique(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for name, value in pairs:
        if name in result:
            raise TokenizerContractError("Duplicate tokenizer config JSON key")
        result[name] = value
    return result


def load_sentencepiece_tokenizer(
    directory: str | Path, *, declared_revision: str,
    vocabulary: int, expected_model_sha256: str | None = None,
) -> LlamaSentencePieceBridge:
    if (
        not isinstance(declared_revision, str)
        or not re.fullmatch(r"[a-f0-9]{40}", declared_revision)
        or type(vocabulary) is not int or not 2 <= vocabulary <= 32768
    ):
        raise TokenizerContractError("Invalid tokenizer revision or vocabulary")
    root = Path(directory)
    model = root / "tokenizer.model"
    config = root / "tokenizer_config.json"
    raw_model = _bounded(model, 8 * 1024 * 1024)
    raw_config = _bounded(config, 32768)
    digest = hashlib.sha256(raw_model).hexdigest()
    if expected_model_sha256 is not None and digest != expected_model_sha256:
        raise TokenizerContractError("Tokenizer model digest differs from pinned expectation")
    try:
        metadata = json.loads(raw_config, object_pairs_hook=_unique)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise TokenizerContractError("Malformed tokenizer config") from exc
    if not isinstance(metadata, dict):
        raise TokenizerContractError("Tokenizer metadata must be an object")
    if metadata.get("tokenizer_class") not in (None, "LlamaTokenizer", "LlamaTokenizerFast"):
        raise TokenizerContractError("Unsupported tokenizer class")
    try:
        import sentencepiece as spm

        processor = spm.SentencePieceProcessor(model_proto=raw_model)
    except (ImportError, RuntimeError, ValueError) as exc:
        raise TokenizerContractError("Cannot initialize bounded SentencePiece tokenizer") from exc
    if processor.vocab_size() != vocabulary:
        raise TokenizerContractError("Tokenizer/model vocabulary mismatch")
    bos, eos, unk = processor.bos_id(), processor.eos_id(), processor.unk_id()
    if any(not 0 <= x < vocabulary for x in (bos, eos, unk)):
        raise TokenizerContractError("Expected valid BOS/EOS/UNK token IDs")
    return LlamaSentencePieceBridge(
        processor, TokenizerIdentity(
            declared_revision, digest, hashlib.sha256(raw_config).hexdigest(),
            vocabulary, bos, eos, unk,
        ),
    )
