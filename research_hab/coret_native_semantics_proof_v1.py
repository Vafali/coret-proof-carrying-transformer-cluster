#!/usr/bin/env python3
"""Native-DeepT semantic operator facade with proof-carrying snapshots.

The facade deliberately delegates the *semantic* transformation to the
immutable DeepT implementation at commit 16ffe407... .  It never translates a
native fresh error term into CoReT's former ``semantic_remainder`` channel.
Proof snapshots are consumed by an independent checker which does not import
DeepT.  Numerically-defined native results are returned unchanged; an eventual
operator-local numerical fallback is only legal if it implements the same
equations and is accepted by that checker.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

PINNED_REVISION = "16ffe4075f1f8a7c87fa2a187d8c46cfd51e07bf"
SCHEMA = "CORET_NATIVE_SEMANTICS_OPERATOR_CERTIFICATE_V1"


def sha256_array(value: torch.Tensor) -> str:
    a = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(a.tobytes(order="C")).hexdigest()


def tensor_record(value: torch.Tensor) -> dict[str, Any]:
    a = value.detach().cpu().contiguous()
    return {
        "shape": list(a.shape), "dtype": str(a.dtype),
        "sha256": sha256_array(a), "finite": bool(torch.isfinite(a).all()),
    }


def ranges_record(z) -> dict[str, Any]:
    lo, hi = z.error_term_range_low, z.error_term_range_high
    if lo is None:
        if hi is not None:
            raise RuntimeError("incomplete native range metadata")
        return {"kind": "implicit_minus1_plus1", "count": int(z.num_error_terms)}
    if hi is None or lo.numel() != hi.numel() or bool((lo > hi).any()):
        raise RuntimeError("invalid native range metadata")
    return {"kind": "explicit", "count": int(lo.numel()),
            "low": tensor_record(lo), "high": tensor_record(hi)}


def state_record(z) -> dict[str, Any]:
    return {
        "weights": tensor_record(z.zonotope_w),
        "generator_count": int(z.num_error_terms),
        "special_prefix_count": int(z.num_input_error_terms_special_norm),
        "ranges": ranges_record(z),
    }


@dataclass
class NativeResult:
    value: Any
    certificate: dict[str, Any]


class NativeSemanticOperators:
    """Production-facing native semantic operators.

    ``revision`` is checked by the caller/manifest.  Each method invokes only
    the corresponding pinned DeepT transformer and records enough immutable
    state for independent equation and symbol-universe checking.
    """
    def __init__(self, revision: str = PINNED_REVISION):
        if revision != PINNED_REVISION:
            raise RuntimeError("native semantic operator revision mismatch")
        self.revision = revision

    def _run(self, family: str, inputs: list, call: Callable[[], Any],
             equation: str, range_logic: str, policy: str, **extra) -> NativeResult:
        total_started = time.perf_counter()
        before = [state_record(z) for z in inputs]
        started = time.perf_counter()
        output = call()
        native_elapsed = time.perf_counter() - started
        if not torch.isfinite(output.zonotope_w).all():
            raise RuntimeError(f"{family}: native semantic result is nonfinite")
        cert = {
            "schema": SCHEMA, "family": family, "revision": self.revision,
            "equation": equation, "range_logic": range_logic,
            "symbol_policy": policy, "inputs": before,
            "output": state_record(output), "native_result_unchanged": True,
            "generic_semantic_remainder_used": False,
            "native_operator_seconds": native_elapsed, **extra,
        }
        total_elapsed = time.perf_counter() - total_started
        cert["facade_total_seconds"] = total_elapsed
        cert["numerical_overhead_seconds"] = max(0.0, total_elapsed-native_elapsed)
        return NativeResult(output, cert)

    def layer_norm(self, z, normalizer, mode="standard"):
        return self._run(
            "LayerNorm", [z], lambda: z.layer_norm(normalizer, mode),
            "z=x-mean(x); v=mean(z*z); y=z/sqrt(v+1e-12); gamma*y+beta",
            "native concretize at fast-dot square, sqrt and reciprocal",
            "native fast-dot/sqrt/reciprocal/product fresh symbols in native order",
            mode=mode, epsilon_hex=float(1e-12).hex())

    def softmax(self, z, *, no_constraints=False):
        return self._run(
            "softmax", [z],
            lambda: z.softmax(use_new_softmax=True, no_constraints=no_constraints,
                              use_new_reciprocal=True),
            "p_i=1/sum_j(exp(s_j-s_i))",
            "native pairwise-difference concretize; exp_minimal_area; reciprocal; optional sum equality",
            "native collapsed exp fresh symbols, reciprocal fresh symbols, then native pivot/range substitution",
            no_constraints=bool(no_constraints))

    def qk(self, q, k):
        return self._run(
            "QK", [q, k], lambda: q.dot_product_precise(k),
            "cq.ck + sum_g(cq.bg+ag.ck)e_g + sum_gh ag.bh e_g e_h",
            "native exact coefficient algebra; precise same-ID/cross-ID quadratic enclosure",
            "retain aligned shared symbols; one fresh symbol per head/query/key in native order")

    def attention_value(self, probability, value):
        return self._run(
            "A.V", [probability, value],
            lambda: probability.dot_product_precise(value.t()),
            "sum_j p_ij V_j with native precise bilinear transformer",
            "native exact coefficient algebra; precise same-ID/cross-ID quadratic enclosure",
            "retain aligned shared symbols; one fresh symbol per head/query/output in native order")

    def relu(self, z):
        return self._run(
            "ReLU", [z], z.relu,
            "lambda=u/(u-l+1e-12); delta=max(-lambda*l,(1-lambda)*u)",
            "native concretize lower/upper; unstable iff l*u<0",
            "scale retained symbols; one fresh symbol per unstable coordinate in native boolean-index order")

    def tanh(self, z):
        return self._run(
            "tanh", [z], z.tanh,
            "lambda=min(1-tanh(l)^2,1-tanh(u)^2); endpoint residual midpoint/radius",
            "native concretize lower/upper; fresh iff l!=u",
            "scale retained symbols; one fresh symbol per non-singleton coordinate in native boolean-index order")

    def reduce(self, z, maximum: int):
        return self._run(
            "generator_reduction", [z],
            lambda: z.reduce_num_error_terms_box(maximum),
            "preserve selected rows; box removed absolute coefficient sum per tensor coordinate",
            "native caller recenters ranged symbols before reduction; reducer requires independent native rows",
            "native topk-smallest removal/order; append one coordinate box symbol",
            maximum=int(maximum))


def certificate_digest(certificate: dict[str, Any]) -> str:
    raw = json.dumps(certificate, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
