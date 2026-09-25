#!/usr/bin/env python3
"""Production graph whose seven special families are native-DeepT dispatched.

Only affine maps, residual addition, layout changes, scalar multiplication and
final concretization are performed directly on a native DeepT ``Zonotope``.
LayerNorm, softmax, QK, A.V, ReLU, tanh and generator reduction must pass
through :class:`NativeProductionDispatch`; there is no legacy proof-state
builder in this call graph.
"""
from __future__ import annotations

import copy
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import torch

from coret_native_semantics_checker_v1 import check_common
from coret_native_semantics_proof_v1 import NativeSemanticOperators


FAMILIES = (
    "LayerNorm", "softmax", "QK", "A.V", "ReLU", "tanh",
    "generator_reduction",
)
EXPECTED_THREE_BLOCK_COUNTS = {
    "LayerNorm": 7,
    "softmax": 3,
    "QK": 3,
    "A.V": 3,
    "ReLU": 3,
    "tanh": 1,
    "generator_reduction": 3,
}


@dataclass
class NativeProductionDispatch:
    """Fail-closed instrumentation around the validated native facade."""

    delegate: NativeSemanticOperators = field(default_factory=NativeSemanticOperators)
    counts: Counter = field(default_factory=Counter)
    certificates: list[dict[str, Any]] = field(default_factory=list)
    generic_family_invocations: int = 0

    def _accept(self, family: str, result):
        if family not in FAMILIES:
            raise RuntimeError(f"unknown native production family: {family}")
        check_common(result.certificate, family)
        if result.certificate.get("generic_semantic_remainder_used") is not False:
            raise RuntimeError(f"legacy generic uncertainty channel reached by {family}")
        if result.certificate.get("native_result_unchanged") is not True:
            raise RuntimeError(f"native result was changed by {family}")
        self.counts[family] += 1
        self.certificates.append(result.certificate)
        return result.value

    def layer_norm(self, z, normalizer, mode="standard"):
        return self._accept(
            "LayerNorm", self.delegate.layer_norm(z, normalizer, mode))

    def softmax(self, z, *, no_constraints=False):
        return self._accept(
            "softmax", self.delegate.softmax(z, no_constraints=no_constraints))

    def qk(self, q, k):
        return self._accept("QK", self.delegate.qk(q, k))

    def attention_value(self, probability, value):
        return self._accept(
            "A.V", self.delegate.attention_value(probability, value))

    def relu(self, z):
        return self._accept("ReLU", self.delegate.relu(z))

    def tanh(self, z):
        return self._accept("tanh", self.delegate.tanh(z))

    def reduce(self, z, maximum: int):
        return self._accept(
            "generator_reduction", self.delegate.reduce(z, maximum))

    def assert_complete(self, expected=None):
        expected = EXPECTED_THREE_BLOCK_COUNTS if expected is None else expected
        actual = {family: int(self.counts[family]) for family in FAMILIES}
        if actual != expected:
            raise RuntimeError(
                f"native production family coverage mismatch: {actual} != {expected}")
        if self.generic_family_invocations != 0:
            raise RuntimeError("legacy generic family implementation was invoked")
        return actual


def _recenter_native_ranges(z):
    # Pinned VerifierZonotope._bound_layer calls this exactly once and only
    # while explicit ranged-symbol metadata exists.
    if z.error_term_range_low is not None:
        z = z.recenter_zonotope_and_eliminate_error_term_ranges()
    return z


def _maximum_for_block(args, layer_num: int) -> int:
    switching = int(getattr(args, "num_fast_dot_product_layers_due_to_switch", -1))
    if switching == -1:
        return int(args.max_num_error_terms)
    # This production revision is pinned to precise QK in every block.  A
    # mixed fast/precise configuration is not silently accepted.
    raise RuntimeError(
        f"native production graph does not accept layer switching (block {layer_num})")


def build_native_production_graph(z, model, args, dispatch: NativeProductionDispatch):
    """Execute the exact three-block native graph and return pooled state."""
    if len(model.bert.encoder.layer) != 3:
        raise RuntimeError("native production graph requires exactly three blocks")
    z = dispatch.layer_norm(z, model.bert.embeddings.LayerNorm, args.layer_norm)

    for layer_num, layer in enumerate(model.bert.encoder.layer):
        z = _recenter_native_ranges(z)
        if args.error_reduction_method != "box":
            raise RuntimeError("native production graph requires native box reduction")
        z = dispatch.reduce(z, _maximum_for_block(args, layer_num))
        residual = z

        attention = layer.attention
        heads = int(attention.self.num_attention_heads)
        head_size = int(attention.self.attention_head_size)
        query = z.dense(attention.self.query).add_attention_heads_dim(heads)
        key = z.dense(attention.self.key).add_attention_heads_dim(heads)
        scores = dispatch.qk(query, key).multiply(1.0 / math.sqrt(head_size))
        probability = dispatch.softmax(
            scores, no_constraints=not bool(args.add_softmax_sum_constraint))
        value = z.dense(attention.self.value).add_attention_heads_dim(heads)
        context = dispatch.attention_value(probability, value)
        context = context.remove_attention_heads_dim()

        attention_output = context.dense(attention.output.dense)
        residual = residual.expand_error_terms_to_match_zonotope(attention_output)
        post_attention = dispatch.layer_norm(
            attention_output.add(residual), attention.output.LayerNorm,
            args.layer_norm)

        intermediate = dispatch.relu(
            post_attention.dense(layer.intermediate.dense))
        residual = post_attention.expand_error_terms_to_match_zonotope(intermediate)
        z = dispatch.layer_norm(
            intermediate.dense(layer.output.dense).add(residual),
            layer.output.LayerNorm, args.layer_norm)

    # Pinned DeepT pooling: first sequence position, dense, then tanh.
    from Verifiers.Zonotope import make_zonotope_new_weights_same_args
    pooled = make_zonotope_new_weights_same_args(
        new_weights=z.zonotope_w[:, :1, :], source_zonotope=z, clone=False)
    pooled = pooled.dense(model.bert.pooler.dense)
    if bool(getattr(args, "with_relu_in_pooling", False)):
        raise RuntimeError("tok10 native production graph requires tanh pooler")
    return dispatch.tanh(pooled)


def direct_margin_zonotope(pooled, classifier, clean_label: int):
    """Start from the native class-weight difference, never two logit bounds."""
    if clean_label not in (0, 1):
        raise RuntimeError("binary classifier label must be zero or one")
    other = 1 - clean_label
    direct = copy.deepcopy(classifier)
    direct.weight = torch.nn.Parameter(
        (classifier.weight[clean_label:clean_label + 1]
         - classifier.weight[other:other + 1]).clone().detach())
    direct.bias = torch.nn.Parameter(
        (classifier.bias[clean_label] - classifier.bias[other]).reshape(1)
        .clone().detach())
    return pooled.dense(direct)


def execute_native_production(z, model, args, clean_label: int,
                              dispatch: NativeProductionDispatch | None = None):
    dispatch = NativeProductionDispatch() if dispatch is None else dispatch
    pooled = build_native_production_graph(z, model, args, dispatch)
    margin = direct_margin_zonotope(pooled, model.classifier, clean_label)
    dispatch.assert_complete()
    return margin, dispatch
