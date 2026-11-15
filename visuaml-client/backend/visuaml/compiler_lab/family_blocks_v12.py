"""Constrained, transparent cross-framework architecture BLOCKS for v12.

These are not checkpoint loaders, SOTA model implementations, hardware lowering,
or evidence of whole-family support. Every function admits small, finite inputs;
stateful algorithms copy their input state and return updated state separately.
NumPy and PyTorch paths are intentionally independent implementations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
from torch.nn import functional as F


MAX_ELEMENTS = 1_000_000


class BlockContractError(ValueError):
    """Unsupported family-block input, shape, dtype, state, or resource budget."""


def _np(value: np.ndarray, name: str, rank: int) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.ndim != rank:
        raise BlockContractError(f"{name} must be a float32 rank-{rank} array")
    if not value.size or value.size > MAX_ELEMENTS or not bool(np.isfinite(value).all()):
        raise BlockContractError(f"{name} is empty, nonfinite, or over budget")
    return value


def _idx(value: np.ndarray, name: str, size: int) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.int64 or value.ndim != 1:
        raise BlockContractError(f"{name} must be int64 edge indices")
    if value.size > MAX_ELEMENTS or np.any(value < 0) or np.any(value >= size):
        raise BlockContractError(f"{name} contains out-of-range indices")
    return value


def _torch(v: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(v.copy())


@dataclass(frozen=True)
class DecoderWeights:
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    out: np.ndarray
    norm: np.ndarray

    @classmethod
    def seeded(cls, hidden: int, heads: int, kv_heads: int, *, seed: int) -> "DecoderWeights":
        if not (0 < hidden <= 128 and 0 < kv_heads <= heads <= 16
                and heads % kv_heads == 0 and hidden % heads == 0):
            raise BlockContractError("Invalid small-decoder head dimensions")
        r = np.random.default_rng(seed)
        depth = hidden // heads
        make = lambda shape: (r.standard_normal(shape) * 0.09).astype(np.float32)
        return cls(make((hidden, hidden)), make((hidden, kv_heads * depth)),
                   make((hidden, kv_heads * depth)), make((hidden, hidden)),
                   np.ones(hidden, dtype=np.float32))


@dataclass(frozen=True)
class KVState:
    keys: np.ndarray  # [B, past, KV, dim]
    values: np.ndarray
    max_context: int


def _validate_cache(x: np.ndarray, w: DecoderWeights, state: KVState, heads: int) -> tuple[int,int,int,int,int]:
    _np(x, "decoder input", 3)
    batch, step, hidden = x.shape
    if not (1 <= step <= 64 and 1 <= batch <= 8 and 0 < hidden <= 128 and hidden % heads == 0):
        raise BlockContractError("Decoder dimensions exceed supported block")
    depth = hidden // heads
    if w.q.shape != (hidden, hidden) or w.out.shape != (hidden, hidden) or w.norm.shape != (hidden,):
        raise BlockContractError("Invalid decoder projection shapes")
    if w.k.shape[0] != hidden or w.k.shape != w.v.shape or w.k.shape[1] % depth:
        raise BlockContractError("Invalid key/value projection shapes")
    kv_heads = w.k.shape[1] // depth
    if not (1 <= kv_heads <= heads and heads % kv_heads == 0 and depth % 2 == 0):
        raise BlockContractError("GQA head count or rotary dimension unsupported")
    for name, t in (("key cache", state.keys),("value cache", state.values)):
        if not isinstance(t, np.ndarray) or t.dtype != np.float32 or t.ndim != 4 or t.shape[0] != batch or t.shape[2:] != (kv_heads, depth):
            raise BlockContractError(f"Invalid {name} shape")
        if not np.isfinite(t).all():
            raise BlockContractError(f"Nonfinite {name}")
    past = state.keys.shape[1]
    if state.values.shape[1] != past or type(state.max_context) is not int or not (past + step <= state.max_context <= 1024):
        raise BlockContractError("KV-cache overflow or invalid context limit")
    if any(not np.isfinite(a).all() for a in (w.q,w.k,w.v,w.out,w.norm)):
        raise BlockContractError("Nonfinite decoder weights")
    return batch, step, hidden, depth, kv_heads


def empty_cache(batch: int, kv_heads: int, depth: int, max_context: int = 64) -> KVState:
    if not (1 <= batch <= 8 and 1 <= kv_heads <= 16 and 0 < depth <= 128 and 1 <= max_context <= 1024):
        raise BlockContractError("Invalid KV-cache allocation")
    z = np.zeros((batch, 0, kv_heads, depth), np.float32)
    return KVState(z, z.copy(), max_context)


def _rotary_np(a: np.ndarray, positions: np.ndarray) -> np.ndarray:
    depth = a.shape[-1]
    inv = 1.0 / (10000.0 ** (np.arange(0, depth, 2, dtype=np.float64) / depth))
    theta = np.outer(positions.astype(np.float64), inv)
    cosine = np.cos(theta)[None,:,None,:].astype(np.float32)
    sine = np.sin(theta)[None,:,None,:].astype(np.float32)
    even, odd = a[...,::2], a[...,1::2]
    return np.stack((even*cosine - odd*sine, even*sine + odd*cosine),axis=-1).reshape(a.shape)


def _rotary_torch(a: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    dim = a.size(-1)
    inv = torch.exp(-torch.arange(0, dim, 2, dtype=torch.float64) * (np.log(10000.0) / dim))
    angle = positions.to(torch.float64)[:,None] * inv[None,:]
    c = torch.cos(angle).to(torch.float32)[None,:,None,:]
    s = torch.sin(angle).to(torch.float32)[None,:,None,:]
    paired = a.reshape(*a.shape[:-1],dim//2,2)
    return torch.stack((paired[...,0]*c-paired[...,1]*s,paired[...,0]*s+paired[...,1]*c),dim=-1).reshape(a.shape)


def decoder_step_numpy(x: np.ndarray,w:DecoderWeights,state:KVState,*,heads:int)->tuple[np.ndarray,KVState]:
    batch,step,hidden,dim,kv = _validate_cache(x,w,state,heads)
    normed = x / np.sqrt(np.mean(np.square(x),axis=-1,keepdims=True)+1e-5) * w.norm
    q = (normed @ w.q).reshape(batch,step,heads,dim)
    k = (normed @ w.k).reshape(batch,step,kv,dim)
    v = (normed @ w.v).reshape(batch,step,kv,dim)
    positions = np.arange(state.keys.shape[1],state.keys.shape[1]+step)
    q,k = _rotary_np(q,positions),_rotary_np(k,positions)
    kt = np.concatenate((state.keys,k),axis=1)
    vt = np.concatenate((state.values,v),axis=1)
    tiled_k = np.repeat(kt, heads//kv, axis=2).transpose(0,2,1,3)
    tiled_v = np.repeat(vt, heads//kv, axis=2).transpose(0,2,1,3)
    logits = np.matmul(q.transpose(0,2,1,3),np.swapaxes(tiled_k,-1,-2)) / np.sqrt(dim)
    mask = np.arange(kt.shape[1])[None,:] <= positions[:,None]
    logits = np.where(mask[None,None,:,:],logits,-1e9)
    probabilities = np.exp(logits - logits.max(axis=-1,keepdims=True))
    probabilities /= probabilities.sum(axis=-1,keepdims=True)
    context = (probabilities @ tiled_v).transpose(0,2,1,3).reshape(batch,step,hidden)
    return (context @ w.out).astype(np.float32), KVState(kt.copy(),vt.copy(),state.max_context)


def decoder_step_torch(x:np.ndarray,w:DecoderWeights,state:KVState,*,heads:int)->tuple[np.ndarray,KVState]:
    batch,step,hidden,dim,kv = _validate_cache(x,w,state,heads)
    t=_torch(x)
    normed=F.rms_norm(t,(hidden,),weight=_torch(w.norm),eps=1e-5)
    q=F.linear(normed,_torch(w.q).T).reshape(batch,step,heads,dim)
    k=F.linear(normed,_torch(w.k).T).reshape(batch,step,kv,dim)
    v=F.linear(normed,_torch(w.v).T).reshape(batch,step,kv,dim)
    positions=torch.arange(state.keys.shape[1],state.keys.shape[1]+step)
    q,k = _rotary_torch(q,positions),_rotary_torch(k,positions)
    kt=torch.cat((_torch(state.keys),k),dim=1)
    vt=torch.cat((_torch(state.values),v),dim=1)
    ks=kt.repeat_interleave(heads//kv,dim=2).permute(0,2,1,3)
    vs=vt.repeat_interleave(heads//kv,dim=2).permute(0,2,1,3)
    scores=q.permute(0,2,1,3) @ ks.transpose(-1,-2) * (dim**-0.5)
    mask=torch.arange(kt.size(1))[None,:] <= positions[:,None]
    scores=scores.masked_fill(~mask[None,None],-1e9)
    attn=torch.softmax(scores,dim=-1)
    context=(attn @ vs).permute(0,2,1,3).reshape(batch,step,hidden)
    return F.linear(context,_torch(w.out).T).detach().numpy(),KVState(kt.detach().numpy(),vt.detach().numpy(),state.max_context)


@dataclass(frozen=True)
class ScanWeights:
    depthwise: np.ndarray  # [D,K]
    a: np.ndarray          # [D]
    b: np.ndarray          # [D]
    c: np.ndarray          # [D]
    gate: np.ndarray       # [D]


@dataclass(frozen=True)
class ScanState:
    convolution: np.ndarray  # [B,K-1,D]
    recurrent: np.ndarray    # [B,D]


def _validate_scan(x:np.ndarray,w:ScanWeights,state:ScanState)->tuple[int,int,int,int]:
    _np(x,"scan input",3)
    batch, steps, d = x.shape
    if not (1 <= batch <= 8 and 1 <= steps <= 128 and 1 <= d <= 128):
        raise BlockContractError("Unsupported scan shape")
    k=w.depthwise.shape[1] if isinstance(w.depthwise,np.ndarray) and w.depthwise.ndim==2 else 0
    if not (2 <= k <= 16 and w.depthwise.shape==(d,k) and state.convolution.shape==(batch,k-1,d) and state.recurrent.shape==(batch,d)):
        raise BlockContractError("Malformed convolution or scan state")
    for arr in (w.depthwise,w.a,w.b,w.c,w.gate,state.convolution,state.recurrent):
        if not isinstance(arr,np.ndarray) or arr.dtype!=np.float32 or not np.isfinite(arr).all():
            raise BlockContractError("Scan arrays must be finite float32")
    if any(a.shape!=(d,) for a in (w.a,w.b,w.c,w.gate)):
        raise BlockContractError("Invalid diagonal scan weights")
    if np.any(w.a<0) or np.any(w.a>=1):
        raise BlockContractError("Recurrent stability coefficients outside [0,1)")
    return batch,steps,d,k


def selective_conv_scan_numpy(x:np.ndarray,w:ScanWeights,state:ScanState)->tuple[np.ndarray,ScanState]:
    batch,steps,d,k=_validate_scan(x,w,state)
    tail=state.convolution.copy()
    recurrent=state.recurrent.copy()
    ys=[]
    for t in range(steps):
        window=np.concatenate((tail,x[:,t:t+1,:]),axis=1)
        filtered=np.sum(window*w.depthwise.T[None,:,:],axis=1)
        recurrent = w.a*recurrent + w.b*filtered
        gate=1/(1+np.exp(-w.gate*filtered))
        ys.append((w.c*recurrent*gate).astype(np.float32))
        tail=window[:,1:,:].copy()
    return np.stack(ys,axis=1), ScanState(tail,recurrent)


def selective_conv_scan_torch(x:np.ndarray,w:ScanWeights,state:ScanState)->tuple[np.ndarray,ScanState]:
    batch,steps,d,k=_validate_scan(x,w,state)
    tx=_torch(x)
    initial=_torch(state.convolution)
    # Causal grouped Conv1d is distinct from NumPy's explicit window multiplication.
    padded=torch.cat((initial,tx),dim=1).transpose(1,2)
    conv=F.conv1d(padded,_torch(w.depthwise).unsqueeze(1),groups=d)
    s=_torch(state.recurrent)
    outputs=[]
    for t in range(steps):
        z=conv[:,:,t]
        s=_torch(w.a)*s+_torch(w.b)*z
        outputs.append(_torch(w.c)*s*torch.sigmoid(_torch(w.gate)*z))
    return torch.stack(outputs,dim=1).detach().numpy(),ScanState(padded[:,:,-(k-1):].transpose(1,2).detach().numpy(),s.detach().numpy())


def moe_numpy(x:np.ndarray,logits:np.ndarray,w_in:np.ndarray,w_out:np.ndarray,*,top_k:int=2,capacity:int=16)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    _np(x,"tokens",2);_np(logits,"router",2);_np(w_in,"expert input",3);_np(w_out,"expert output",3)
    n,d=x.shape; experts=logits.shape[1]
    if not (logits.shape==(n,experts) and w_in.shape[0]==experts and w_in.shape[1]==d and w_out.shape==(experts,w_in.shape[2],d) and 1 <= top_k <= experts <= 64 and 1<= capacity <=4096):
        raise BlockContractError("Unsupported MoE dimensions/capacity")
    chosen=np.argsort(-logits,axis=1,kind="stable")[:,:top_k]
    selected=np.take_along_axis(logits,chosen,axis=1)
    weights=np.exp(selected-selected.max(axis=-1,keepdims=True))
    weights/=weights.sum(axis=-1,keepdims=True)
    out=np.zeros_like(x); loads=np.zeros(experts,dtype=np.int64)
    for t in range(n):
        for slot,expert in enumerate(chosen[t]):
            if loads[expert]>=capacity: continue
            loads[expert]+=1
            hidden=np.maximum(x[t] @ w_in[expert],0)
            out[t]+=weights[t,slot] * (hidden @ w_out[expert])
    return out,loads,chosen


def moe_torch(x:np.ndarray,logits:np.ndarray,w_in:np.ndarray,w_out:np.ndarray,*,top_k:int=2,capacity:int=16)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    _np(x,"tokens",2);_np(logits,"router",2);_np(w_in,"expert input",3);_np(w_out,"expert output",3)
    n,d=x.shape; experts=logits.shape[1]
    if not (w_in.shape[0]==experts and w_in.shape[1]==d and w_out.shape==(experts,w_in.shape[2],d) and 1<=top_k<=experts<=64 and 1<=capacity<=4096):
        raise BlockContractError("Unsupported MoE")
    tensor=_torch(x); route=_torch(logits)
    indices=torch.argsort(route,dim=-1,descending=True,stable=True)[:,:top_k]
    scores=torch.gather(route,1,indices)
    gates=torch.softmax(scores,dim=-1)
    result=torch.zeros_like(tensor)
    loads=torch.zeros(experts,dtype=torch.int64)
    incoming=_torch(w_in); outgoing=_torch(w_out)
    for token in range(n):
        for rank in range(top_k):
            expert=int(indices[token,rank]);
            if loads[expert]>=capacity: continue
            loads[expert]+=1
            hidden=torch.relu(F.linear(tensor[token],incoming[expert].T))
            result[token]+=gates[token,rank]*F.linear(hidden,outgoing[expert].T)
    return result.detach().numpy(),loads.numpy(),indices.numpy()


def flow_numpy(x:np.ndarray,condition:np.ndarray,weight:np.ndarray,*,steps:int,method:str="euler")->np.ndarray:
    _np(x,"flow state",2);_np(condition,"flow condition",2);_np(weight,"flow weight",2)
    batch,d=x.shape
    if condition.shape[0]!=batch or weight.shape!=(d+condition.shape[1]+1,d) or type(steps)is not int or not 1<=steps<=128 or method not in ("euler","heun"):
        raise BlockContractError("Invalid flow solver contract")
    state=x.copy();h=np.float32(-1.0/steps)
    def vector(z,t):
        payload=np.concatenate((z,condition,np.full((batch,1),t,np.float32)),axis=-1)
        return np.tanh(payload @ weight)
    for j in range(steps):
        t=np.float32(1.0-j/steps)
        velocity=vector(state,t)
        if method=="heun":
            predictor=state+h*velocity
            velocity=(velocity+vector(predictor,np.float32(t+h)))*np.float32(0.5)
        state=state+h*velocity
    return state


def flow_torch(x:np.ndarray,condition:np.ndarray,weight:np.ndarray,*,steps:int,method:str="euler")->np.ndarray:
    _np(x,"flow state",2);_np(condition,"flow condition",2);_np(weight,"flow weight",2)
    batch,d=x.shape
    if condition.shape[0]!=batch or weight.shape!=(d+condition.shape[1]+1,d) or type(steps)is not int or not 1<=steps<=128 or method not in ("euler","heun"):
        raise BlockContractError("Invalid flow solver contract")
    x1=_torch(x);c=_torch(condition);w=_torch(weight)
    def f(z,t):
        return torch.tanh(F.linear(torch.cat((z,c,torch.full((batch,1),float(t))),dim=-1),w.T))
    h=-1.0/steps
    for j in range(steps):
        t=1.0-j/steps
        v=f(x1,t)
        if method=="heun": v=(v+f(x1+h*v,t+h))/2
        x1=x1+h*v
    return x1.detach().numpy()


def gnn_numpy(nodes:np.ndarray,src:np.ndarray,dst:np.ndarray,weight:np.ndarray,attention:np.ndarray)->np.ndarray:
    _np(nodes,"graph nodes",2);_np(weight,"graph weight",2);_np(attention,"graph attention",1)
    n,d=nodes.shape;out=weight.shape[1]
    if weight.shape[0]!=d or attention.shape!=(2*out,):raise BlockContractError("Invalid graph weights")
    _idx(src,"source edges",n);_idx(dst,"target edges",n)
    if src.shape!=dst.shape:raise BlockContractError("Graph edge arrays differ")
    hidden=nodes @ weight
    messages=np.maximum(hidden[src],0)
    logits=np.sum(np.concatenate((hidden[src],hidden[dst]),axis=1)*attention,axis=1) if len(src) else np.zeros(0,np.float32)
    logits=np.maximum(logits,np.float32(0.2)*logits)
    result=np.zeros((n,out),np.float32)
    for target in range(n):
        edge=np.where(dst==target)[0]
        if not edge.size:continue
        p=np.exp(logits[edge]-max(logits[edge]));p/=sum(p)
        result[target]+=np.sum(p[:,None]*messages[edge],axis=0)
    return result


def gnn_torch(nodes:np.ndarray,src:np.ndarray,dst:np.ndarray,weight:np.ndarray,attention:np.ndarray)->np.ndarray:
    _np(nodes,"graph nodes",2);_np(weight,"graph weight",2);_np(attention,"graph attention",1)
    n,d=nodes.shape;out=weight.shape[1]
    if weight.shape[0]!=d or attention.shape!=(2*out,):raise BlockContractError("Invalid graph weights")
    _idx(src,"source edges",n);_idx(dst,"target edges",n)
    if src.shape!=dst.shape:raise BlockContractError("Graph edges differ")
    t=_torch(nodes);s=torch.from_numpy(src.copy());di=torch.from_numpy(dst.copy());w=_torch(weight);a=_torch(attention)
    features=F.linear(t,w.T)
    msg=F.relu(features.index_select(0,s))
    logits=F.leaky_relu(torch.cat((features.index_select(0,s),features.index_select(0,di)),dim=1) @ a,negative_slope=0.2)
    # Segment-softmax, then index_add handles unsorted and duplicate edges.
    weights=torch.zeros_like(logits)
    for target in range(n):
        ids=torch.nonzero(di==target,as_tuple=False).flatten()
        if ids.numel():weights[ids]=torch.softmax(logits.index_select(0,ids),dim=0)
    result=torch.zeros((n,out),dtype=torch.float32)
    result.index_add_(0,di,msg*weights.unsqueeze(1))
    return result.detach().numpy()


def quantize_per_channel_numpy(weight:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    _np(weight,"quantized weight",2)
    maxv=np.max(np.abs(weight),axis=0)
    scale=np.where(maxv>0,maxv/127.0,np.float32(1.0)).astype(np.float32)
    integer=np.clip(np.rint(weight/scale),-127,127).astype(np.int8)
    return integer,scale


def quantize_per_channel_torch(weight:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    _np(weight,"quantized weight",2)
    tensor=_torch(weight)
    maximum=tensor.abs().amax(dim=0)
    scale=torch.where(maximum>0,maximum/127,torch.ones_like(maximum))
    q=torch.round(tensor/scale).clamp(-127,127).to(torch.int8)
    return q.numpy(),scale.numpy()


def sgd_numpy(x:np.ndarray,y:np.ndarray,w:np.ndarray,*,rate:float,steps:int)->tuple[np.ndarray,np.ndarray]:
    _np(x,"training input",2);_np(y,"training target",2);_np(w,"trainable weights",2)
    if x.shape[1]!=w.shape[0] or y.shape!=(x.shape[0],w.shape[1]) or not 0<rate<=1 or not 1<=steps<=32:
        raise BlockContractError("Invalid training example")
    weights=w.copy();losses=[]
    for _ in range(steps):
        err=(x@weights)-y
        losses.append(np.float32(np.mean(err**2)))
        grad=(2.0/err.size)*(x.T @ err)
        weights=weights-rate*grad
    return weights,np.asarray(losses,dtype=np.float32)


def sgd_torch(x:np.ndarray,y:np.ndarray,w:np.ndarray,*,rate:float,steps:int)->tuple[np.ndarray,np.ndarray]:
    _np(x,"training input",2);_np(y,"training target",2);_np(w,"trainable weights",2)
    if x.shape[1]!=w.shape[0] or y.shape!=(x.shape[0],w.shape[1]) or not 0<rate<=1 or not 1<=steps<=32:
        raise BlockContractError("Invalid training example")
    tx=_torch(x);ty=_torch(y);tw=torch.nn.Parameter(_torch(w))
    optim=torch.optim.SGD([tw],lr=rate)
    losses=[]
    for _ in range(steps):
        optim.zero_grad(set_to_none=True)
        loss=F.mse_loss(tx@tw,ty)
        losses.append(float(loss.detach()))
        loss.backward();optim.step()
    return tw.detach().numpy(),np.asarray(losses,dtype=np.float32)
