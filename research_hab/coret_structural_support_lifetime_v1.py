#!/usr/bin/env python3
"""Validated deterministic allocator fence for structural production B2 QK."""
from __future__ import annotations

import gc
import weakref

import torch

import coret_structural_support_precise_dot_v1 as structural


SCHEMA = "CORET_STRUCTURAL_SUPPORT_B2_QK_LIFETIME_FENCE_V1"


def _contains_tensor(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return False
    seen.add(id(value))
    if torch.is_tensor(value):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item, seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(item, seen) for item in value)
    return False


class LifetimeFencedStructuralOperators(
        structural.StructuralNativeSemanticOperators):
    """Apply exactly one cache fence immediately before the third QK call."""

    def __init__(self, dead_holder, **kwargs):
        super().__init__(**kwargs)
        self._dead_holder = dead_holder
        self._dispatch_ref = None
        self.lifetime_fence_records = []

    def bind_dispatch(self, dispatch):
        self._dispatch_ref = weakref.ref(dispatch)

    def _fence_before_b2_qk(self, q, k):
        if self._dispatch_ref is None or self._dispatch_ref() is None:
            raise RuntimeError("B2 QK lifetime fence dispatch unavailable")
        dispatch = self._dispatch_ref()
        if _contains_tensor(dispatch.certificates):
            raise RuntimeError("compact operator certificates retain Tensor state")
        if self._hidden is None:
            raise RuntimeError("B2 QK support state unavailable")
        # Fail closed before releasing anything unless these are exactly the
        # third-QK operands and current support universe.
        if q.num_error_terms != len(self._hidden.masks):
            raise RuntimeError("B2 Q support/generator mismatch before fence")
        if k.num_error_terms != len(self._hidden.masks):
            raise RuntimeError("B2 K support/generator mismatch before fence")
        torch.cuda.synchronize()
        before = {
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        released_names = sorted(self._dead_holder)
        self._dead_holder.clear()
        unreachable = int(gc.collect())
        torch.cuda.synchronize()
        allocated_before_trim = int(torch.cuda.memory_allocated())
        reserved_before_trim = int(torch.cuda.memory_reserved())
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        after = {
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        if after["allocated_bytes"] > allocated_before_trim:
            raise RuntimeError("B2 QK lifetime fence increased live allocation")
        record = {
            "schema": SCHEMA,
            "boundary": "immediately_before_block2_QK",
            "q_shape": list(q.zonotope_w.shape),
            "k_shape": list(k.zonotope_w.shape),
            "generator_count": int(q.num_error_terms),
            "support_class_count": len(set(self._hidden.masks)),
            "released_names": released_names,
            "gc_unreachable_objects": unreachable,
            "allocated_before_trim": allocated_before_trim,
            "reserved_before_trim": reserved_before_trim,
            "before": before, "after": after,
            "allocated_bytes_released": (
                before["allocated_bytes"] - after["allocated_bytes"]),
            "reserved_bytes_released": (
                before["reserved_bytes"] - after["reserved_bytes"]),
            "compact_certificates_tensor_free": True,
            "semantic_operands_retained": True,
        }
        self.lifetime_fence_records.append(record)
        return record

    def qk(self, q, k):
        is_b2 = self._qk_index == 2
        fence = self._fence_before_b2_qk(q, k) if is_b2 else None
        result = super().qk(q, k)
        if is_b2:
            if len(self.lifetime_fence_records) != 1:
                raise RuntimeError("B2 QK lifetime fence count differs")
            result.certificate["lifetime_fence"] = fence
        return result


def make_dispatch(dead_holder, production_module, **kwargs):
    delegate = LifetimeFencedStructuralOperators(dead_holder, **kwargs)
    dispatch = production_module.NativeProductionDispatch(delegate=delegate)
    delegate.bind_dispatch(dispatch)
    return delegate, dispatch
