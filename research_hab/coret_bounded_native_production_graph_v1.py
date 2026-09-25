#!/usr/bin/env python3
"""Three-block native production graph with bounded materialization."""
from __future__ import annotations

import math

from coret_bounded_native_execution_v1 import BoundedNativeSemanticOperators
from coret_native_semantics_production_graph_v1 import (
    EXPECTED_THREE_BLOCK_COUNTS,
    FAMILIES,
    NativeProductionDispatch,
    direct_margin_zonotope,
)


def new_dispatch() -> NativeProductionDispatch:
    return NativeProductionDispatch(delegate=BoundedNativeSemanticOperators())


def _recenter(z):
    if z.error_term_range_low is not None:
        z = z.recenter_zonotope_and_eliminate_error_term_ranges()
    return z


def build(z, model, args, dispatch: NativeProductionDispatch):
    from Verifiers.Zonotope import make_zonotope_new_weights_same_args
    if len(model.bert.encoder.layer) != 3:
        raise RuntimeError("bounded native graph requires exactly three blocks")
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

        intermediate = dispatch.relu(
            post_attention.dense(layer.intermediate.dense))
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
        raise RuntimeError("bounded tok10 graph requires native tanh pooler")
    return dispatch.tanh(pooled)


def execute(z, model, args, clean_label: int,
            dispatch: NativeProductionDispatch | None = None):
    dispatch = new_dispatch() if dispatch is None else dispatch
    pooled = build(z, model, args, dispatch)
    margin = direct_margin_zonotope(pooled, model.classifier, clean_label)
    del pooled
    dispatch.assert_complete(EXPECTED_THREE_BLOCK_COUNTS)
    return margin, dispatch
