"""v14: bounded multi-layer checkpoint ABI, independent reference and real AOT execution."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from visuaml.compiler_lab.numpy_runtime import BundleError, run_bundle
from visuaml.compiler_lab.stacked_decoder_v14 import (
    CompiledStackedDecoder,
    StackContractError,
    StackedWeights,
    StackState,
    checkpoint_digest,
    compile_stacked_chunk,
    empty_stack,
    load_canonical_checkpoint,
    reference_stack_step,
    verify_stacked_witness,
    write_canonical_checkpoint,
)


@pytest.mark.parametrize("seed,chunks", [(0, (3,)), (19, (2, 1, 2)), (47, (1, 3, 1))])
def test_compiled_two_layer_prefill_and_cache(seed, chunks):
    weights = StackedWeights.seeded(seed)
    tokens = np.random.default_rng(seed + 1).integers(
        0, 19, size=(2, sum(chunks)), dtype=np.int64
    )
    report = verify_stacked_witness(weights, tokens, chunks)
    assert report["passed"] is True
    assert report["arrays_checked"] == len(chunks) * 5
    assert report["max_abs_error"] < 5e-5
    assert report["layers"] == 2
    assert len(report["checkpoint_sha256"]) == 64


def test_compiled_generation_matches_reference_even_with_cache_per_layer():
    weights = StackedWeights.seeded(13)
    tokens = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    predicted, after = CompiledStackedDecoder(weights).generate(tokens, steps=3)
    state = empty_stack(weights, batch=2)
    logits, state = reference_stack_step(weights, tokens, state)
    expected = []
    for _ in range(3):
        next_token = np.argmax(logits[:, -1, :], axis=-1).astype(np.int64)[:, None]
        expected.append(next_token)
        logits, state = reference_stack_step(weights, next_token, state)
    np.testing.assert_array_equal(predicted, np.concatenate(expected, axis=1))
    for observed_layer, expected_layer in zip(after.layers, state.layers):
        np.testing.assert_allclose(
            observed_layer.keys, expected_layer.keys, atol=5e-5, rtol=5e-5
        )
        np.testing.assert_allclose(
            observed_layer.values, expected_layer.values, atol=5e-5, rtol=5e-5
        )


def test_canonical_checkpoint_roundtrip_and_content_integrity(tmp_path):
    w = StackedWeights.seeded(9)
    loc = write_canonical_checkpoint(
        w, tmp_path / "weights", provenance="synthetic-test-only"
    )
    loaded = load_canonical_checkpoint(loc)
    assert checkpoint_digest(w) == checkpoint_digest(loaded)
    assert loaded.heads == 4 and loaded.max_context == 64
    for name, val in w.arrays().items():
        np.testing.assert_array_equal(loaded.arrays()[name], val)
    s = empty_stack(loaded, 1)
    a, _ = CompiledStackedDecoder(loaded).step(np.array([[3]], np.int64), s)
    b, _ = reference_stack_step(w, np.array([[3]], np.int64), s)
    np.testing.assert_allclose(a, b, atol=5e-5, rtol=5e-5)


def test_checkpoint_source_claim_is_declared_not_attested(tmp_path):
    w = StackedWeights.seeded(1)
    loc = write_canonical_checkpoint(
        w, tmp_path / "w", provenance="user-generated-test"
    )
    doc = json.loads((loc / "checkpoint.json").read_text())
    assert doc["source_kind"] == "user-declared-canonical-weights"
    assert doc["provenance"] == "user-generated-test"
    doc["source_kind"] = "verified-physical-checkpoint"
    (loc / "checkpoint.json").write_text(json.dumps(doc))
    with pytest.raises(StackContractError, match="provenance"):
        load_canonical_checkpoint(loc)


def test_checkpoint_tampering_rejected(tmp_path):
    w = StackedWeights.seeded(19)
    loc = write_canonical_checkpoint(w, tmp_path / "w", provenance="synthetic")
    with np.load(loc / "weights.npz", allow_pickle=False) as data:
        arrays = {n: data[n].copy() for n in data.files}
    arrays["layer1.attention.q"][0, 0] += 0.1
    np.savez_compressed(loc / "weights.npz", **arrays)
    with pytest.raises(StackContractError, match="digest"):
        load_canonical_checkpoint(loc)


def test_checkpoint_missing_weights_rejected(tmp_path):
    w = StackedWeights.seeded(2)
    loc = write_canonical_checkpoint(w, tmp_path / "w", provenance="synthetic")
    with np.load(loc / "weights.npz", allow_pickle=False) as data:
        arrays = {n: data[n].copy() for n in data.files if n != "layer1.gate"}
    np.savez_compressed(loc / "weights.npz", **arrays)
    with pytest.raises(StackContractError, match="weight keys"):
        load_canonical_checkpoint(loc)


def test_state_validation_fail_closed_and_preserves_inputs():
    w = StackedWeights.seeded(5)
    driver = CompiledStackedDecoder(w)
    state = empty_stack(w, 1)
    old = state.layers[0].keys.copy()
    _, next_state = driver.step(np.array([[1, 4]], np.int64), state)
    np.testing.assert_array_equal(state.layers[0].keys, old)
    assert not np.shares_memory(state.layers[0].keys, next_state.layers[0].keys)
    invalid = StackState(
        (
            replace(next_state.layers[0], keys=np.ones((1, 1, 2, 2), np.float32)),
            next_state.layers[1],
        )
    )
    with pytest.raises(StackContractError, match="state"):
        driver.step(np.array([[2]], np.int64), invalid)
    for tokens in (
        np.array([[-1]], np.int64),
        np.array([[19]], np.int64),
        np.array([[1.0]], np.float32),
    ):
        with pytest.raises(StackContractError):
            driver.step(tokens, state)


def test_artifact_manifest_token_domain_enforced_without_driver():
    w = StackedWeights.seeded(5)
    art = compile_stacked_chunk(w, batch=1, query=1, past=0)
    state = empty_stack(w, 1)
    args = tuple(x for layer in state.layers for x in (layer.keys, layer.values))
    with pytest.raises(BundleError, match="bounded integer domain"):
        art.run(np.array([[19]], np.int64), *args)
    with pytest.raises(BundleError, match="bounded integer domain"):
        art.run(np.array([[-1]], np.int64), *args)
    with pytest.raises(BundleError, match="requires"):
        art.run(np.array([[1]], np.int32), *args)


def test_bundle_roundtrip_and_pytorch_free_subprocess(tmp_path):
    w = StackedWeights.seeded(5)
    art = compile_stacked_chunk(w, batch=1, query=2, past=0)
    path = art.write(tmp_path / "compiled")
    state = empty_stack(w, 1)
    args = tuple(x for layer in state.layers for x in (layer.keys, layer.values))
    tokens = np.array([[1, 3]], np.int64)
    original = art.run(tokens, *args)
    reread = run_bundle(path, tokens, *args)
    for a, b in zip(original, reread):
        np.testing.assert_array_equal(a, b)
    assert len(art.manifest["nodes"]) > 100
    assert len(art.constants) >= 30
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    import visuaml.compiler_lab.numpy_runtime as library

    for name in ("numpy_runtime.py", "ir_contract.py"):
        (runtime / name).write_bytes(
            (Path(library.__file__).parent / name).read_bytes()
        )
    script = """import builtins,sys,numpy as np
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name == "torch" or name.startswith("torch."): raise RuntimeError("PyTorch forbidden")
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from numpy_runtime import run_bundle
z=np.zeros((1,0,2,2),np.float32)
out=run_bundle(sys.argv[1],np.array([[1,3]],np.int64),z,z,z,z)
assert len(out)==5 and out[0].shape==(1,2,19)
assert all(x.shape==(1,2,2,2) for x in out[1:])
"""
    subprocess.run(
        [sys.executable, "-c", script, str(path)],
        check=True,
        cwd=runtime,
        env={**os.environ, "PYTHONPATH": str(runtime)},
        timeout=30,
    )


def test_digest_changes_for_modified_weights():
    w = StackedWeights.seeded(12)
    modified = replace(w, embedding=w.embedding.copy())
    modified.embedding[0, 0] += 0.01
    assert checkpoint_digest(w) != checkpoint_digest(modified)
    assert (
        compile_stacked_chunk(w, batch=1, query=2, past=0).manifest["manifest_sha256"]
        != compile_stacked_chunk(modified, batch=1, query=2, past=0).manifest[
            "manifest_sha256"
        ]
    )


def test_cannot_claim_arbitrary_layer_count_or_precision():
    w = StackedWeights.seeded(10)
    with pytest.raises(StackContractError, match="1..4"):
        replace(w, blocks=()).validate()
    with pytest.raises(StackContractError, match="1..4"):
        replace(w, blocks=(w.blocks[0],) * 5).validate()
    with pytest.raises(StackContractError, match="float32"):
        replace(w, embedding=w.embedding.astype(np.float64)).validate()
    with pytest.raises(StackContractError):
        replace(w, heads=3).validate()
    with pytest.raises(StackContractError):
        replace(w, max_context=1000).validate()


def test_artifact_manifest_digest_matches_byte_count():
    w = StackedWeights.seeded(1)
    art = compile_stacked_chunk(w, batch=1, query=1, past=0)
    assert len(art.constants) > 20
    assert len(art.manifest["manifest_sha256"]) == 64
    assert hashlib.sha256(art.constants["embedding"].tobytes()).hexdigest()


@pytest.mark.parametrize("layers", [1, 2, 3, 4])
def test_multiple_depths_share_one_compilation_pipeline(layers):
    w = StackedWeights.seeded(seed=23, layers=layers)
    tok = np.array([[1, 3, 5]], np.int64)
    report = verify_stacked_witness(w, tok, (2, 1))
    assert report["layers"] == layers
    assert report["arrays_checked"] == 2 * (1 + 2 * layers)
    assert report["max_abs_error"] < 5e-5


@pytest.mark.parametrize("layers", [1, 4])
def test_checkpoint_roundtrip_for_minimum_and_maximum_depth(tmp_path, layers):
    w = StackedWeights.seeded(seed=42, layers=layers)
    root = write_canonical_checkpoint(
        w, tmp_path / f"layer_{layers}", provenance="synthetic"
    )
    result = load_canonical_checkpoint(root)
    assert len(result.blocks) == layers
    assert checkpoint_digest(w) == checkpoint_digest(result)
    assert len(result.arrays()) == 2 + 9 * layers


def test_flattened_fx_tuple_has_fixed_tensor_placeholders():
    w = StackedWeights.seeded(seed=11, layers=3)
    artifact = compile_stacked_chunk(w, batch=2, query=3, past=2)
    specs = artifact.manifest["inputs"]
    assert len(specs) == 1 + 2 * 3
    assert specs[0]["shape"] == [2, 3]
    assert all(x["shape"] == [2, 2, 2, 2] for x in specs[1:])
    # All six cache tensors remain separate formal inputs, not opaque tuples.
    assert len({x["name"] for x in specs}) == len(specs)
    assert all(
        item["kind"] in {"constant", "call"} for item in artifact.manifest["nodes"]
    )


def test_mutating_weights_after_compilation_fails_closed():
    w = StackedWeights.seeded(seed=9)
    d = CompiledStackedDecoder(w)
    tokens = np.array([[1]], np.int64)
    _ = d.step(tokens, empty_stack(w, 1))
    w.blocks[0].attention.q[0, 0] += np.float32(0.01)
    with pytest.raises(StackContractError, match="mutated after compilation"):
        d.step(tokens, empty_stack(w, 1))


def test_checkpoint_duplicate_archive_entry_rejected(tmp_path):
    from zipfile import ZipFile

    w = StackedWeights.seeded(seed=10)
    loc = write_canonical_checkpoint(w, tmp_path / "w", provenance="synthetic")
    archive = loc / "weights.npz"
    with ZipFile(archive) as z:
        content = [(x.filename, z.read(x.filename)) for x in z.infolist()]
    with ZipFile(archive, "w") as z:
        for name, data in content:
            z.writestr(name, data)
        z.writestr(content[0][0], content[0][1])
    with pytest.raises(StackContractError, match="archive"):
        load_canonical_checkpoint(loc)
