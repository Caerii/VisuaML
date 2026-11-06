"""Semantic graph rewrites for attention and stable softmax.

These passes are opt-in and must be run through the verified pass manager.
They accept only a fixed documented subset. Algebraic rewrites may change
floating-point rounding and device execution cost, so numeric witnesses are
required to decide whether to accept the rewrite for a given workload.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp

from visuaml.compiler_spike import CapturedProgram, UnsupportedOperation, inspect_fx


def _fresh(source: CapturedProgram):
    return copy.deepcopy(source.graph_module)


def _complete(gm, example_inputs):
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    with torch.no_grad():
        ShapeProp(gm).propagate(*example_inputs)
    return CapturedProgram(gm, inspect_fx(gm))


def decompose_softmax(
    program: CapturedProgram, *example_inputs: torch.Tensor
) -> CapturedProgram:
    """Stable softmax -> amax/sub/exp/sum/div, preserving chosen reduction dim."""
    gm = _fresh(program)
    graph = gm.graph
    for node in list(graph.nodes):
        dim = None
        if node.op == "call_module" and isinstance(
            gm.get_submodule(str(node.target)), nn.Softmax
        ):
            dim = gm.get_submodule(str(node.target)).dim
        elif node.op == "call_function" and node.target in {
            torch.softmax,
            torch.nn.functional.softmax,
        }:
            dim = node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else None)
            if node.kwargs.get("dtype") is not None or len(node.args) > 2:
                raise UnsupportedOperation(
                    "Softmax dtype conversion is not supported by this pass"
                )
        elif node.op == "call_method" and node.target == "softmax":
            dim = node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else None)
        else:
            continue
        if not isinstance(dim, int):
            raise UnsupportedOperation("Softmax rewrite requires an integer dimension")
        x = node.args[0]
        with graph.inserting_before(node):
            peak = graph.call_function(torch.amax, (x,), {"dim": dim, "keepdim": True})
            shifted = graph.call_function(torch.sub, (x, peak))
            ex = graph.call_function(torch.exp, (shifted,))
            den = graph.call_function(torch.sum, (ex,), {"dim": dim, "keepdim": True})
            result = graph.call_function(torch.div, (ex, den))
        node.replace_all_uses_with(result)
        graph.erase_node(node)
    return _complete(gm, example_inputs)


def decompose_sdpa(
    program: CapturedProgram, *example_inputs: torch.Tensor
) -> CapturedProgram:
    """Restricted scaled-dot-product attention -> transpose/matmul/scale/softmax/matmul.

    Supports only inference with no mask, no dropout, no causal masking or GQA.
    The expansion is NOT an equivalent substitute for optimized fused attention
    across arbitrary shapes, precision modes or numerical tolerances.
    """
    gm = _fresh(program)
    graph = gm.graph
    sdpa = torch.nn.functional.scaled_dot_product_attention
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not sdpa:
            continue
        if len(node.args) < 3:
            raise UnsupportedOperation("SDPA needs q, k, v")
        q, k, v = node.args[:3]
        mask = node.kwargs.get(
            "attn_mask", node.args[3] if len(node.args) > 3 else None
        )
        dropout = node.kwargs.get(
            "dropout_p", node.args[4] if len(node.args) > 4 else 0
        )
        causal = node.kwargs.get(
            "is_causal", node.args[5] if len(node.args) > 5 else False
        )
        gqa = node.kwargs.get("enable_gqa", False)
        scale = node.kwargs.get("scale")
        if mask is not None or dropout != 0 or causal or gqa:
            raise UnsupportedOperation(
                "SDPA decomposition supports only unmasked, no-dropout, noncausal, non-GQA attention"
            )
        if not isinstance(q, torch.fx.Node) or not isinstance(k, torch.fx.Node):
            raise UnsupportedOperation("SDPA inputs must be FX tensors")
        qmeta = q.meta.get("tensor_meta")
        if (
            qmeta is None
            or len(qmeta.shape) < 2
            or not isinstance(qmeta.shape[-1], int)
        ):
            raise UnsupportedOperation("SDPA requires known query head dimension")
        head_dim = int(qmeta.shape[-1])
        if head_dim < 1:
            raise UnsupportedOperation("Invalid SDPA head dimension")
        factor = (1 / math.sqrt(head_dim)) if scale is None else float(scale)
        with graph.inserting_before(node):
            kt = graph.call_method("transpose", (k, -2, -1))
            logits = graph.call_function(torch.matmul, (q, kt))
            scaled = graph.call_function(torch.mul, (logits, factor))
            probs = graph.call_function(torch.softmax, (scaled, -1))
            out = graph.call_function(torch.matmul, (probs, v))
        node.replace_all_uses_with(out)
        graph.erase_node(node)
    return _complete(gm, example_inputs)
