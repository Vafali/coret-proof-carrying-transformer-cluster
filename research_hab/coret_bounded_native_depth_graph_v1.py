#!/usr/bin/env python3
"""Depth-parametric execution of the accepted bounded native graph.

The per-block equations and dispatch are identical to the frozen three-block
graph; only the number of native encoder blocks and expected call counts are
bound from the frozen checkpoint depth.
"""
from __future__ import annotations

import math

from coret_native_semantics_production_graph_v1 import (
    FAMILIES, NativeProductionDispatch, direct_margin_zonotope,
)


def expected_counts(depth: int) -> dict[str, int]:
    if depth not in (6, 12):
        raise RuntimeError(f"unsupported production depth: {depth}")
    return {
        "LayerNorm": 1 + 2 * depth,
        "softmax": depth,
        "QK": depth,
        "A.V": depth,
        "ReLU": depth,
        "tanh": 1,
        "generator_reduction": depth,
    }


# Bound by configure_depth() before a production call.  The name is retained
# because the accepted runner performs its final invocation-count assertion
# through this module attribute.
EXPECTED_THREE_BLOCK_COUNTS = None


def configure_depth(depth: int) -> dict[str, int]:
    global EXPECTED_THREE_BLOCK_COUNTS
    EXPECTED_THREE_BLOCK_COUNTS = expected_counts(depth)
    return dict(EXPECTED_THREE_BLOCK_COUNTS)


def _recenter(z):
    if z.error_term_range_low is not None:
        z = z.recenter_zonotope_and_eliminate_error_term_ranges()
    return z


def build(z, model, args, dispatch: NativeProductionDispatch):
    from Verifiers.Zonotope import make_zonotope_new_weights_same_args
    depth = len(model.bert.encoder.layer)
    expected = expected_counts(depth)
    if EXPECTED_THREE_BLOCK_COUNTS != expected:
        raise RuntimeError("depth graph was not bound to loaded checkpoint")
    if int(args.num_layers) != depth:
        raise RuntimeError("model/DeepT argument depth mismatch")
    if args.error_reduction_method != "box":
        raise RuntimeError("bounded native graph requires native box reduction")
    if int(args.num_fast_dot_product_layers_due_to_switch) != -1:
        raise RuntimeError("bounded native graph requires precise QK in every block")

    z = dispatch.layer_norm(z, model.bert.embeddings.LayerNorm, args.layer_norm)
    for layer in model.bert.encoder.layer:
        z = _recenter(z)
        z = dispatch.reduce(z, int(args.max_num_error_terms))
        residual = z
        attention = layer.attention
        heads = int(attention.self.num_attention_heads)
        head_size = int(attention.self.attention_head_size)

        query = z.dense(attention.self.query).add_attention_heads_dim(heads)
        key = z.dense(attention.self.key).add_attention_heads_dim(heads)
        scores = dispatch.qk(query, key).multiply(1.0 / math.sqrt(head_size))
        del query, key
        probability = dispatch.softmax(
            scores, no_constraints=not bool(args.add_softmax_sum_constraint))
        del scores
        value = z.dense(attention.self.value).add_attention_heads_dim(heads)
        context = dispatch.attention_value(probability, value)
        del probability, value
        context = context.remove_attention_heads_dim()

        attention_output = context.dense(attention.output.dense)
        del context
        residual = residual.expand_error_terms_to_match_zonotope(attention_output)
        post_attention = dispatch.layer_norm(
            attention_output.add(residual), attention.output.LayerNorm,
            args.layer_norm)
        del attention_output, residual

        intermediate = dispatch.relu(post_attention.dense(layer.intermediate.dense))
        residual = post_attention.expand_error_terms_to_match_zonotope(intermediate)
        dense = intermediate.dense(layer.output.dense).add(residual)
        del intermediate, residual, post_attention
        z = dispatch.layer_norm(dense, layer.output.LayerNorm, args.layer_norm)
        del dense

    pooled = make_zonotope_new_weights_same_args(
        new_weights=z.zonotope_w[:, :1, :], source_zonotope=z, clone=False)
    del z
    pooled = pooled.dense(model.bert.pooler.dense)
    if bool(getattr(args, "with_relu_in_pooling", False)):
        raise RuntimeError("bounded depth graph requires native tanh pooler")
    return dispatch.tanh(pooled)


def execute(z, model, args, clean_label: int,
            dispatch: NativeProductionDispatch | None = None):
    if dispatch is None:
        raise RuntimeError("depth graph requires accepted structural dispatch")
    pooled = build(z, model, args, dispatch)
    margin = direct_margin_zonotope(pooled, model.classifier, clean_label)
    del pooled
    dispatch.assert_complete(EXPECTED_THREE_BLOCK_COUNTS)
    return margin, dispatch
