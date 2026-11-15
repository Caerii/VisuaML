"""Reproducible, claim-limited v12 representative neural-block certificates.

A certificate covers pinned synthetic CPU witnesses and NumPy/Torch parity;
it does not imply that a full framework checkpoint, custom op, general model,
training algorithm, compiler backend, or physical device has been certified.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .family_blocks_v12 import (
    DecoderWeights, ScanState, ScanWeights, decoder_step_numpy,
    decoder_step_torch, empty_cache, selective_conv_scan_numpy,
    selective_conv_scan_torch, moe_numpy, moe_torch, flow_numpy, flow_torch,
    gnn_numpy, gnn_torch, quantize_per_channel_numpy,
    quantize_per_channel_torch, sgd_numpy, sgd_torch,
)

SCHEMA="visuaml.representative-block-certification.v12"
FAMILIES=(
    ("decoder_transformer", "GQA/RoPE decoder with append-only KV-cache"),
    ("mamba", "Causal depthwise Conv1D and diagonal recurrent scan"),
    ("moe", "Top-k sparse expert routing with finite capacity"),
    ("flow_matching", "Time-conditioned deterministic Euler/Heun flow sampler"),
    ("graph_attention", "Irregular COO graph attention and segment aggregation"),
    ("quantized_training", "Per-channel int8 weights and analytic/autograd SGD"),
)


@dataclass(frozen=True)
class BlockWitness:
    family: str
    seed: int
    input_sha256: str
    reference_sha256: str
    max_abs_error: float
    checked_arrays: int
    passed: bool


def _hash_arrays(*args:np.ndarray)->str:
    h=hashlib.sha256(b"visuaml-v12-array-corpus\x00")
    for item in args:
        a=np.ascontiguousarray(item)
        h.update(str(a.dtype).encode());h.update(str(tuple(a.shape)).encode());h.update(a.tobytes())
    return h.hexdigest()


def _compare(a:tuple[np.ndarray,...],b:tuple[np.ndarray,...], *, rtol=5e-5,atol=5e-5)->float:
    if len(a)!=len(b): raise AssertionError("Witness output arity changed")
    worst=0.0
    for x,y in zip(a,b):
        if x.shape!=y.shape or x.dtype!=y.dtype or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise AssertionError("Witness output type or finite contract changed")
        if x.dtype.kind in "iu":
            if not np.array_equal(x,y):raise AssertionError("Integer / routing semantics differ")
        else:
            np.testing.assert_allclose(x,y,rtol=rtol,atol=atol)
            worst=max(worst,float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64)))))
    return worst


def run_witness(family:str,seed:int)->BlockWitness:
    if family not in {key for key,_ in FAMILIES} or type(seed)is not int or not 0<=seed<=100000:
        raise ValueError("Unknown family or invalid seed")
    r=np.random.default_rng(seed)
    if family=="decoder_transformer":
        w=DecoderWeights.seeded(8,4,2,seed=seed)
        x=r.normal(size=(2,5,8)).astype(np.float32)
        s=empty_cache(2,2,2,16)
        left,right=[],[]
        for chunk in (x[:,:3],x[:,3:4],x[:,4:]):
            na,s=decoder_step_numpy(chunk,w,s,heads=4)
            left.append(na)
        ts=empty_cache(2,2,2,16)
        for chunk in (x[:,:3],x[:,3:4],x[:,4:]):
            ta,ts=decoder_step_torch(chunk,w,ts,heads=4)
            right.append(ta)
        actual=(np.concatenate(left,axis=1),s.keys,s.values)
        reference=(np.concatenate(right,axis=1),ts.keys,ts.values)
        evidence=(x,w.q,w.k,w.v,w.out,w.norm)
    elif family=="mamba":
        x=r.normal(size=(2,6,4)).astype(np.float32)
        w=ScanWeights(r.normal(scale=.1,size=(4,3)).astype(np.float32),r.uniform(.1,.9,4).astype(np.float32),
            r.uniform(.05,.7,4).astype(np.float32),r.normal(scale=.3,size=4).astype(np.float32),r.normal(size=4).astype(np.float32))
        state=ScanState(np.zeros((2,2,4),np.float32),np.zeros((2,4),np.float32))
        a,sa=selective_conv_scan_numpy(x,w,state)
        b,sb=selective_conv_scan_torch(x,w,state)
        actual=(a,sa.convolution,sa.recurrent)
        reference=(b,sb.convolution,sb.recurrent)
        evidence=(x,w.depthwise,w.a,w.b,w.c,w.gate)
    elif family=="moe":
        x=r.normal(size=(7,6)).astype(np.float32)
        logits=r.normal(size=(7,4)).astype(np.float32);logits[0,:]=0
        wi=r.normal(scale=.2,size=(4,6,9)).astype(np.float32)
        wo=r.normal(scale=.2,size=(4,9,6)).astype(np.float32)
        actual=moe_numpy(x,logits,wi,wo,top_k=2,capacity=3)
        reference=moe_torch(x,logits,wi,wo,top_k=2,capacity=3)
        evidence=(x,logits,wi,wo)
    elif family=="flow_matching":
        x=r.normal(size=(3,8)).astype(np.float32)
        condition=r.normal(size=(3,4)).astype(np.float32)
        w=r.normal(scale=.1,size=(13,8)).astype(np.float32)
        actual=(flow_numpy(x,condition,w,steps=9,method="euler"),flow_numpy(x,condition,w,steps=9,method="heun"))
        reference=(flow_torch(x,condition,w,steps=9,method="euler"),flow_torch(x,condition,w,steps=9,method="heun"))
        evidence=(x,condition,w)
    elif family=="graph_attention":
        x=r.normal(size=(5,4)).astype(np.float32)
        src=np.array([0,1,1,4,2,2,0,1],np.int64)
        dst=np.array([1,1,1,0,3,3,3,4],np.int64)
        w=r.normal(scale=.1,size=(4,6)).astype(np.float32)
        attn=r.normal(scale=.15,size=(12,)).astype(np.float32)
        actual=(gnn_numpy(x,src,dst,w,attn),)
        reference=(gnn_torch(x,src,dst,w,attn),)
        evidence=(x,src,dst,w,attn)
    else:
        x=r.normal(size=(6,3)).astype(np.float32)
        y=r.normal(size=(6,2)).astype(np.float32)
        w=r.normal(scale=.2,size=(3,2)).astype(np.float32)
        w[0,:]=0
        q,s=quantize_per_channel_numpy(w)
        qt,st=quantize_per_channel_torch(w)
        wt,lt=sgd_numpy(x,y,w,rate=.04,steps=5)
        wtorch,ltorch=sgd_torch(x,y,w,rate=.04,steps=5)
        actual=(q,s,wt,lt)
        reference=(qt,st,wtorch,ltorch)
        evidence=(x,y,w)
    err=_compare(actual,reference)
    return BlockWitness(family,seed,_hash_arrays(*evidence),_hash_arrays(*reference),err,len(actual),True)


def certification_report(*,seeds:tuple[int,...]=(7,19,41))->dict[str,Any]:
    if not isinstance(seeds,tuple) or not 1<=len(seeds)<=8 or len(set(seeds))!=len(seeds):
        raise ValueError("Require distinct bounded witness seeds")
    results=[]
    for key,title in FAMILIES:
        cases=[run_witness(key,seed) for seed in seeds]
        results.append({
            "architecture_id":key,
            "representative_block":title,
            "level":"cross-framework-reference-parity",
            "family_checkpoint_certified":False,
            "full_model_compiler_executable":False,
            "physical_hardware_validated":False,
            "source_kind":"synthetic-cpu-fixtures",
            "seed_count":len(cases),
            "max_abs_error":max(c.max_abs_error for c in cases),
            "witnesses":[asdict(c) for c in cases],
        })
    pkg=Path(__file__).resolve()
    source_sha256=hashlib.sha256(pkg.read_bytes()).hexdigest()
    payload={
        "schema":SCHEMA,"source_kind":"synthetic-cpu-fixtures",
        "claim":"Representative BLOCK parity only; no checkpoint-family compatibility, general AOT lowering, or device performance claim",
        "python_backend":"PyTorch CPU vs NumPy independent reference",
        "torch_version":torch.__version__,"numpy_version":np.__version__,
        "source_sha256":source_sha256,"certifications":results,
    }
    payload["sha256"]=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",", ":"),allow_nan=False).encode()).hexdigest()
    return payload
