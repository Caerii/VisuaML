"""Small, fully executable autoregressive decoder *fixture* for compiler research.

This is one GQA/RoPE attention + SwiGLU residual block with embeddings and a
language-model head. It supports chunked prefill, immutable KV state, and
incremental greedy generation. No claim of Llama checkpoint compatibility,
training support, FlashAttention, CUDA execution, or dynamic-shape AOT lowering.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch.nn import functional as F

from .family_blocks_v12 import (
    BlockContractError, DecoderWeights, KVState,
    decoder_step_numpy, decoder_step_torch, empty_cache,
)


@dataclass(frozen=True)
class TinyDecoderWeights:
    embedding: np.ndarray
    attention: DecoderWeights
    final_norm: np.ndarray
    gate: np.ndarray
    up: np.ndarray
    down: np.ndarray
    lm_head: np.ndarray

    @classmethod
    def seeded(cls, *, hidden: int=8, heads: int=4, kv_heads: int=2,
               vocab: int=19, expansion: int=12, seed: int=19) -> "TinyDecoderWeights":
        if not (2 <= vocab <= 128 and 1 <= expansion <= 256):
            raise BlockContractError("Tiny decoder vocabulary or MLP exceeds budget")
        attention = DecoderWeights.seeded(hidden,heads,kv_heads,seed=seed)
        generator = np.random.default_rng(seed+1000)
        def make(shape):
            return generator.normal(scale=.12,size=shape).astype(np.float32)
        return cls(make((vocab,hidden)),attention,np.ones(hidden,np.float32),
                   make((hidden,expansion)),make((hidden,expansion)),
                   make((expansion,hidden)),make((hidden,vocab)))


def _check(tokens: np.ndarray, weights:TinyDecoderWeights) -> tuple[int,int,int]:
    if not isinstance(tokens,np.ndarray) or tokens.dtype!=np.int64 or tokens.ndim!=2:
        raise BlockContractError("Token IDs must be int64 [batch, tokens]")
    batch,step=tokens.shape
    if not (1<=batch<=8 and 1<=step<=64 and not np.any(tokens<0)
            and not np.any(tokens>=weights.embedding.shape[0])):
        raise BlockContractError("Tokens out of bounds or resource budget")
    vocab,hidden=weights.embedding.shape
    if weights.final_norm.shape!=(hidden,) or weights.gate.shape[0]!=hidden or weights.up.shape!=weights.gate.shape or weights.down.shape!=(weights.gate.shape[1],hidden) or weights.lm_head.shape!=(hidden,vocab):
        raise BlockContractError("Decoder FFN / vocabulary shapes incompatible")
    for value in (weights.embedding,weights.final_norm,weights.gate,weights.up,weights.down,weights.lm_head):
        if value.dtype!=np.float32 or not np.isfinite(value).all():
            raise BlockContractError("Decoder weights must be finite float32")
    return batch,step,hidden


def decoder_tokens_numpy(tokens:np.ndarray,weights:TinyDecoderWeights,
                         state:KVState,*,heads:int=4)->tuple[np.ndarray,KVState]:
    _check(tokens,weights)
    x=weights.embedding[tokens]
    attn,next_state=decoder_step_numpy(x,weights.attention,state,heads=heads)
    residual=x+attn
    normal=residual/np.sqrt(np.mean(residual*residual,axis=-1,keepdims=True)+1e-5)*weights.final_norm
    gate=normal @ weights.gate
    silu=gate/(1+np.exp(-gate))
    mlp=(silu*(normal@weights.up))@weights.down
    result=residual+mlp
    return (result @ weights.lm_head).astype(np.float32),next_state


def decoder_tokens_torch(tokens:np.ndarray,weights:TinyDecoderWeights,
                         state:KVState,*,heads:int=4)->tuple[np.ndarray,KVState]:
    _check(tokens,weights)
    emb=F.embedding(torch.from_numpy(tokens.copy()),torch.from_numpy(weights.embedding.copy()))
    attn,next_state=decoder_step_torch(emb.detach().numpy(),weights.attention,state,heads=heads)
    residual=emb+torch.from_numpy(attn)
    h=F.rms_norm(residual,(residual.shape[-1],),weight=torch.from_numpy(weights.final_norm.copy()),eps=1e-5)
    mlp=F.linear(F.silu(F.linear(h,torch.from_numpy(weights.gate.T.copy())))
                 *F.linear(h,torch.from_numpy(weights.up.T.copy())),
                 torch.from_numpy(weights.down.T.copy()))
    logits=F.linear(residual+mlp,torch.from_numpy(weights.lm_head.T.copy()))
    return logits.detach().numpy(),next_state


def greedy_generate(tokens:np.ndarray,weights:TinyDecoderWeights,*,steps:int,
                    heads:int=4,backend:str="numpy",max_context:int=64)->tuple[np.ndarray,KVState]:
    batch,_,hidden=_check(tokens,weights)
    if type(steps)is not int or not 1<=steps<=32 or tokens.shape[1]+steps>max_context or backend not in ("numpy","torch"):
        raise BlockContractError("Invalid greedy generation budget/backend")
    kv_heads=weights.attention.k.shape[1]//(hidden//heads)
    state=empty_cache(batch,kv_heads,hidden//heads,max_context)
    fn=decoder_tokens_numpy if backend=="numpy" else decoder_tokens_torch
    logits,state=fn(tokens,weights,state,heads=heads)
    produced=[]
    for _ in range(steps):
        ids=np.argmax(logits[:,-1,:],axis=-1).astype(np.int64)[:,None]
        produced.append(ids)
        logits,state=fn(ids,weights,state,heads=heads)
    return np.concatenate(produced,axis=1),state
