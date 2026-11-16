"""Bounded tokenizer-file loading and deterministic sentencepiece parity."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import sentencepiece as spm
from visuaml.compiler_lab.sentencepiece_tokenizer_v20 import (
    TokenizerContractError,
    load_sentencepiece_tokenizer,
)


@pytest.fixture
def tokenizer_fixture(tmp_path):
    data = tmp_path / "training.txt"
    data.write_text(
        ("Once upon a time a little cat walked through the garden.\n"
         "The girl told a story about a magical tree and a friendly dragon.\n"
         "A boy found a red balloon and shared it with his friends.\n") * 20
    )
    spm.SentencePieceTrainer.train(
        input=str(data), model_prefix=str(tmp_path / "tokenizer"),
        vocab_size=64, model_type="unigram",
        hard_vocab_limit=False, bos_id=1, eos_id=2, unk_id=0,
    )
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "LlamaTokenizer", "add_bos_token": True,
        "legacy": True,
    }))
    processor = spm.SentencePieceProcessor(model_file=str(tmp_path / "tokenizer.model"))
    return tmp_path, processor


def test_sentencepiece_codec_and_bos_token_match_independent_spm(tokenizer_fixture):
    directory, reference = tokenizer_fixture
    raw = (directory / "tokenizer.model").read_bytes()
    bridge = load_sentencepiece_tokenizer(
        directory, declared_revision="a" * 40,
        vocabulary=reference.vocab_size(),
        expected_model_sha256=hashlib.sha256(raw).hexdigest(),
    )
    for text in ("Once upon a time", "the garden", "a friendly dragon"):
        expected = reference.encode(text, out_type=int)
        actual = bridge.encode(text, add_bos=False).tolist()[0]
        assert actual == expected
        assert bridge.encode(text)[0, 0] == reference.bos_id()
        assert bridge.decode(np.asarray(expected, dtype=np.int64)) == reference.decode(expected)
    assert bridge.identity.model_sha256 == hashlib.sha256(raw).hexdigest()
    assert bridge.identity.vocabulary == reference.vocab_size()


def test_reject_wrong_vocab_revision_hash_and_input(tokenizer_fixture):
    directory, reference = tokenizer_fixture
    with pytest.raises(TokenizerContractError):
        load_sentencepiece_tokenizer(directory, declared_revision="main", vocabulary=64)
    with pytest.raises(TokenizerContractError):
        load_sentencepiece_tokenizer(
            directory, declared_revision="a" * 40, vocabulary=reference.vocab_size() + 1
        )
    with pytest.raises(TokenizerContractError):
        load_sentencepiece_tokenizer(
            directory, declared_revision="a" * 40,
            vocabulary=reference.vocab_size(), expected_model_sha256="0" * 64,
        )
    bridge = load_sentencepiece_tokenizer(
        directory, declared_revision="a" * 40, vocabulary=reference.vocab_size(),
    )
    with pytest.raises(TokenizerContractError):
        bridge.encode("")
    with pytest.raises(TokenizerContractError):
        bridge.encode("x" * 3000)
    with pytest.raises(TokenizerContractError):
        bridge.decode(np.array([-1], np.int64))
    with pytest.raises(TokenizerContractError):
        bridge.decode(np.array([1], np.int32))


def test_reject_symlink_and_invalid_metadata(tokenizer_fixture, tmp_path):
    directory, reference = tokenizer_fixture
    model = directory / "tokenizer.model"
    source = tmp_path / "original.model"
    source.write_bytes(model.read_bytes())
    model.unlink()
    model.symlink_to(source)
    with pytest.raises(TokenizerContractError):
        load_sentencepiece_tokenizer(
            directory, declared_revision="a" * 40, vocabulary=reference.vocab_size()
        )
    model.unlink()
    model.write_bytes(source.read_bytes())
    config = directory / "tokenizer_config.json"
    config.write_text('{"tokenizer_class":"LlamaTokenizer","tokenizer_class":"evil"}')
    with pytest.raises(TokenizerContractError):
        load_sentencepiece_tokenizer(
            directory, declared_revision="a" * 40, vocabulary=reference.vocab_size()
        )



def test_nonlegacy_prefix_rules_match_pinned_hf_llama_tokenizer(tokenizer_fixture):
    from transformers import LlamaTokenizer

    directory, reference = tokenizer_fixture
    config = directory / "tokenizer_config.json"
    config.write_text(json.dumps({
        "tokenizer_class": "LlamaTokenizer",
        "add_bos_token": True, "legacy": False,
        "add_prefix_space": True,
    }))
    hf = LlamaTokenizer(
        vocab_file=str(directory / "tokenizer.model"),
        legacy=False, add_prefix_space=True,
    )
    bridge = load_sentencepiece_tokenizer(
        directory, declared_revision="a" * 40, vocabulary=reference.vocab_size(),
    )
    assert bridge.identity.legacy is False
    for text in ("Once upon a time", "a friendly dragon", "the garden"):
        expected = hf.encode(text, add_special_tokens=True)
        actual = bridge.encode(text).tolist()[0]
        assert actual == expected
