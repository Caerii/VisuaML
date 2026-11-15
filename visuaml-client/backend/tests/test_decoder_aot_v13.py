"""v13 full tiny-decoder portable AOT: prefill, immutable KV, generation, fail-closed admission.

These synthetic fixtures certify only a shape-specialized 1-layer small decoder,
not any production checkpoint or physical accelerator.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from visuaml.compiler_lab.decoder_aot_v13 import (
    CompiledTinyDecoder,
    DecoderCompileError,
    compile_decoder_chunk,
    verify_decoder_artifact,
)
from visuaml.compiler_lab.family_blocks_v12 import empty_cache
from visuaml.compiler_lab.numpy_runtime import BundleError, run_bundle
from visuaml.compiler_lab.tiny_decoder_v12 import TinyDecoderWeights, greedy_generate


@pytest.mark.parametrize('seed,segments', [(0,(4,)), (19,(2,2,1)), (41,(1,3,1)), (77,(1,1,1,1))])
def test_full_chunk_and_cache_differential(seed: int, segments: tuple[int, ...]):
    weights = TinyDecoderWeights.seeded(seed=seed)
    rng = np.random.default_rng(seed + 7)
    tokens = rng.integers(0, 19, size=(2, sum(segments)), dtype=np.int64)
    result = verify_decoder_artifact(weights, tokens, segments)
    assert result['passed']
    assert result['arrays_checked'] == len(segments)*6
    assert result['max_abs_error'] < 4e-5
    assert result['specializations'] == len(segments)


@pytest.mark.parametrize('seed', [3, 19, 41])
def test_generated_token_and_cache_parity(seed: int):
    w = TinyDecoderWeights.seeded(seed=seed)
    tokens = np.array([[1,2,3], [3,4,2]], np.int64)
    got, state = CompiledTinyDecoder(w).generate(tokens, steps=3)
    reference, rstate = greedy_generate(tokens, w, steps=3, backend='numpy')
    other, tstate = greedy_generate(tokens, w, steps=3, backend='torch')
    np.testing.assert_array_equal(got, reference)
    np.testing.assert_array_equal(got, other)
    np.testing.assert_allclose(state.keys, rstate.keys, atol=4e-5, rtol=4e-5)
    np.testing.assert_allclose(state.values, tstate.values, atol=4e-5, rtol=4e-5)
    assert len(got[0]) == 3


def test_aot_artifact_write_read_and_content_digests(tmp_path: Path):
    w = TinyDecoderWeights.seeded(seed=13)
    artifact = compile_decoder_chunk(w, batch=1, query=3, past=0)
    path = artifact.write(tmp_path/'prefill')
    tokens = np.array([[1,3,8]], dtype=np.int64)
    zero = np.zeros((1,0,2,2),np.float32)
    from_memory = artifact.run(tokens,zero,zero)
    from_bundle = run_bundle(path,tokens,zero,zero)
    for a,b in zip(from_memory,from_bundle):
        np.testing.assert_array_equal(a,b)
    assert len(artifact.constants) >= 10
    assert len(artifact.manifest['manifest_sha256']) == 64
    assert 'repeat_interleave' in [x['op'] for x in artifact.manifest['nodes']]
    assert 'cat' in [x['op'] for x in artifact.manifest['nodes']]
    payload = json.loads((path/'manifest.json').read_text())
    payload['source'] = 'tampered'
    (path/'manifest.json').write_text(json.dumps(payload))
    with pytest.raises(BundleError,match='digest'):
        run_bundle(path,tokens,zero,zero)


def test_standalone_runtime_requires_no_pytorch(tmp_path: Path):
    w=TinyDecoderWeights.seeded(seed=5)
    source=compile_decoder_chunk(w,batch=1,query=2,past=0).write(tmp_path/'standalone')
    # One process imports only NumPy + standard-library source copies.
    runtime=tmp_path/'runtime'
    runtime.mkdir()
    import visuaml.compiler_lab.numpy_runtime as library
    for name in ('numpy_runtime.py','ir_contract.py'):
        (runtime/name).write_bytes((Path(library.__file__).parent/name).read_bytes())
    inp = tmp_path/'tokens.npy'
    np.save(inp,np.array([[2,3]],dtype=np.int64))
    script='''import builtins,sys,numpy as np
orig=builtins.__import__
def no_torch(name,*args,**kwargs):
    if name == "torch" or name.startswith("torch."): raise RuntimeError("Torch forbidden")
    return orig(name,*args,**kwargs)
builtins.__import__=no_torch
from numpy_runtime import run_bundle
x=np.load(sys.argv[2]); z=np.zeros((1,0,2,2),np.float32)
logits,k,v=run_bundle(sys.argv[1],x,z,z)
assert logits.shape==(1,2,19) and k.shape==(1,2,2,2)
assert np.isfinite(logits).all()
'''
    subprocess.run([sys.executable,'-c',script,str(source),str(inp)],
                   cwd=runtime,env={**os.environ,'PYTHONPATH':str(runtime)},check=True,timeout=30)


def test_shape_guard_and_cache_immutability():
    w=TinyDecoderWeights.seeded(seed=1)
    d=CompiledTinyDecoder(w,max_context=6)
    z=empty_cache(1,2,2,6)
    tok=np.array([[1,2,3]],np.int64)
    old_k=z.keys.copy()
    old_v=z.values.copy()
    logits,new=d.step(tok,z)
    assert logits.shape==(1,3,19)
    np.testing.assert_array_equal(z.keys,old_k)
    np.testing.assert_array_equal(z.values,old_v)
    assert not np.shares_memory(new.keys,z.keys)
    with pytest.raises(DecoderCompileError,match='capacity'):
        d.step(np.array([[1,2,3,4]],np.int64),new)
    with pytest.raises(BundleError,match='requires'):
        d._compiled[(1,3,0)].run(np.array([[1,2,3]],np.int32),old_k,old_v)


@pytest.mark.parametrize('invalid', [np.array([[-1]],np.int64),np.array([[900]],np.int64),
                                     np.array([[True]]),np.array([[np.nan]])])
def test_out_of_contract_ids_rejected(invalid):
    w=TinyDecoderWeights.seeded()
    d=CompiledTinyDecoder(w)
    z=empty_cache(1,2,2,64)
    with pytest.raises(DecoderCompileError):
        d.step(invalid,z)


def test_cache_shape_and_nan_rejected():
    w=TinyDecoderWeights.seeded()
    d=CompiledTinyDecoder(w)
    bad=empty_cache(1,2,2,64)
    from dataclasses import replace
    with pytest.raises(DecoderCompileError):
        d.step(np.array([[1]],np.int64),replace(bad,values=np.zeros((1,1,2,2),np.float32)))
    with pytest.raises(DecoderCompileError):
        d.step(np.array([[1]],np.int64),replace(bad,keys=np.full((1,1,2,2),np.inf,np.float32),
                                                         values=np.zeros((1,1,2,2),np.float32)))


def test_compilation_output_has_reproducible_digest():
    w=TinyDecoderWeights.seeded(seed=19)
    a=compile_decoder_chunk(w,batch=1,query=3,past=0)
    b=compile_decoder_chunk(w,batch=1,query=3,past=0)
    assert a.manifest['manifest_sha256']==b.manifest['manifest_sha256']
    c=compile_decoder_chunk(w,batch=1,query=1,past=3)
    assert a.manifest['manifest_sha256']!=c.manifest['manifest_sha256']
    assert hashlib.sha256(a.constants['embedding'].tobytes()).digest()


def test_portable_manifest_enforces_token_domain_without_python_driver():
    w = TinyDecoderWeights.seeded(seed=3)
    artifact = compile_decoder_chunk(w, batch=1, query=2, past=0)
    zero = np.zeros((1, 0, 2, 2), np.float32)
    for invalid in (np.array([[-1, 1]], np.int64), np.array([[1, 19]], np.int64)):
        with pytest.raises(BundleError, match="outside declared bounded integer domain"):
            artifact.run(invalid, zero, zero)


def test_malformed_integer_value_contract_rejected():
    from copy import deepcopy
    from visuaml.compiler_lab.ir_contract import IRContractError, validate_portable_ir

    w = TinyDecoderWeights.seeded()
    artifact = compile_decoder_chunk(w, batch=1, query=1, past=0)
    invalid_cases = (
        {"token_ids": {"min_inclusive": True, "max_exclusive": 19}},
        {"token_ids": {"min_inclusive": 19, "max_exclusive": 19}},
        {"not_an_input": {"min_inclusive": 0, "max_exclusive": 19}},
        {"token_ids": {"min_inclusive": 0, "max_exclusive": 19, "eval": "yes"}},
    )
    for limits in invalid_cases:
        changed = deepcopy(artifact.manifest)
        changed["input_constraints"] = limits
        with pytest.raises(IRContractError):
            validate_portable_ir(changed, artifact.constants)
