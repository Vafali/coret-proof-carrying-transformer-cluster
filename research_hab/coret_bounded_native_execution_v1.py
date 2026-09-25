#!/usr/bin/env python3
"""Bounded-memory execution of the pinned native DeepT operator equations.

This module changes materialization only.  Precise dot products retain the
native center, aligned first-order rows and one fresh row per output coordinate
in exactly the native order.  The off-diagonal quadratic sum is evaluated in
fixed generator blocks.  Native softmax uses DeepT's existing per-head batch
path, avoiding a simultaneous all-head five-dimensional repeat.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any, Callable

import torch

import coret_native_semantics_proof_v1 as native_proof


SCHEMA = "CORET_BOUNDED_NATIVE_OPERATOR_CERTIFICATE_V1"
DEFAULT_PAIR_BLOCK = 2048
AV_MIXED_TEMPORARY_CAP_BYTES = 4 * 1024 * 1024
HASH_FIRST_DIM_BLOCK_BYTES = 8 * 1024 * 1024


def _tensor_chunks(value: torch.Tensor, maximum_bytes: int = HASH_FIRST_DIM_BLOCK_BYTES):
    """Yield bounded contiguous CPU chunks in logical C order."""
    detached = value.detach()
    if detached.ndim == 0:
        yield detached.reshape(1).cpu().contiguous()
        return
    bytes_per_first = max(1, detached[0].numel() * detached.element_size())
    count = max(1, maximum_bytes // bytes_per_first)
    for start in range(0, detached.shape[0], count):
        yield detached[start:start + count].cpu().contiguous()


def sha256_tensor_blockwise(value: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for chunk in _tensor_chunks(value):
        digest.update(memoryview(chunk.numpy()).cast("B"))
    return digest.hexdigest()


def tensor_record_blockwise(value: torch.Tensor) -> dict[str, Any]:
    finite = True
    for chunk in _tensor_chunks(value):
        finite = finite and bool(torch.isfinite(chunk).all())
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": sha256_tensor_blockwise(value),
        "finite": finite,
        "hash_materialization": "bounded_first_dimension_chunks",
    }


def ranges_record_blockwise(z) -> dict[str, Any]:
    lo, hi = z.error_term_range_low, z.error_term_range_high
    if lo is None:
        if hi is not None:
            raise RuntimeError("incomplete native range metadata")
        return {"kind": "implicit_minus1_plus1", "count": int(z.num_error_terms)}
    if hi is None or lo.numel() != hi.numel() or bool((lo > hi).any()):
        raise RuntimeError("invalid native range metadata")
    return {
        "kind": "explicit", "count": int(lo.numel()),
        "low": tensor_record_blockwise(lo),
        "high": tensor_record_blockwise(hi),
    }


def state_record_blockwise(z) -> dict[str, Any]:
    return {
        "weights": tensor_record_blockwise(z.zonotope_w),
        "generator_count": int(z.num_error_terms),
        "special_prefix_count": int(z.num_input_error_terms_special_norm),
        "ranges": ranges_record_blockwise(z),
    }


def _bmm_left_fixed(left: torch.Tensor, rights: torch.Tensor) -> torch.Tensor:
    # left: [NA,D], rights: [B,NB,D] -> [B,NA,NB].  expand is a view;
    # any backend packing is bounded by B.
    return torch.bmm(
        left.unsqueeze(0).expand(rights.shape[0], -1, -1),
        rights.transpose(1, 2),
    )


def _bmm_right_fixed(lefts: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    # lefts: [B,NA,D], right: [NB,D] -> [B,NA,NB].
    return torch.bmm(
        lefts,
        right.unsqueeze(0).expand(lefts.shape[0], -1, -1).transpose(1, 2),
    )


def _mixed_pair_block(pair_block: int, na: int, nb: int,
                      dtype: torch.dtype, temporary_cap_bytes: int | None) -> int:
    """Bound one materialized ``mixed`` tensor by a strict byte cap."""
    if temporary_cap_bytes is None:
        return pair_block
    if temporary_cap_bytes <= 0:
        raise ValueError("temporary_cap_bytes must be positive")
    bytes_per_pair = na * nb * torch.empty((), dtype=dtype).element_size()
    return max(1, min(pair_block, temporary_cap_bytes // bytes_per_pair))


def precise_dot_blockwise(left, right, pair_block: int = DEFAULT_PAIR_BLOCK,
                          mixed_temporary_cap_bytes: int | None = None):
    """Native precise-dot equation with bounded off-diagonal temporaries."""
    from Verifiers.Zonotope import make_zonotope_new_weights_same_args

    if pair_block <= 0:
        raise ValueError("pair_block must be positive")
    if left.zonotope_w.ndim != 4 or right.zonotope_w.ndim != 4:
        raise RuntimeError("bounded precise dot requires native four-dimensional operands")
    if left.zonotope_w.shape[0] != right.zonotope_w.shape[0]:
        raise RuntimeError("attention-head count mismatch")
    if left.zonotope_w.shape[-1] != right.zonotope_w.shape[-1]:
        raise RuntimeError("precise-dot inner dimension mismatch")

    a = left.zonotope_w
    b = right.zonotope_w
    heads = a.shape[0]
    ga, gb = left.num_error_terms, right.num_error_terms
    gmin, gmax = min(ga, gb), max(ga, gb)
    na, nb = left.num_words, right.num_words
    mixed_pair_block = _mixed_pair_block(
        pair_block, na, nb, a.dtype, mixed_temporary_cap_bytes)
    fresh = heads * na * nb
    out = torch.zeros(
        heads, 1 + gmax + fresh, na, nb,
        device=a.device, dtype=a.dtype,
    )
    mask = torch.ones(na, nb, dtype=torch.bool, device=a.device)

    for head in range(heads):
        ah, bh = a[head], b[head]
        out[head, 0] = ah[0] @ bh[0].t()

        # Native same-ID quadratic center/radius.
        diagonal = torch.bmm(
            ah[1:1 + gmin], bh[1:1 + gmin].transpose(1, 2))
        out[head, 0] += 0.5 * diagonal.sum(dim=0)
        half = 0.5 * diagonal.abs().sum(dim=0)
        del diagonal

        # Native retained aligned rows, without center repeats.
        for start in range(0, gmax, pair_block):
            stop = min(gmax, start + pair_block)
            retained = torch.zeros(
                stop - start, na, nb, device=a.device, dtype=a.dtype)
            if start < gb:
                bend = min(stop, gb)
                retained[:bend - start] += _bmm_left_fixed(
                    ah[0], bh[1 + start:1 + bend])
            if start < ga:
                aend = min(stop, ga)
                retained[:aend - start] += _bmm_right_fixed(
                    ah[1 + start:1 + aend], bh[0])
            out[head, 1 + start:1 + stop] = retained

        # Native off-diagonal remainder.  Pair identity/order is unchanged;
        # only the suffix is partitioned into fixed-size blocks.
        big = torch.zeros(na, nb, device=a.device, dtype=a.dtype)
        for ai in range(gmin):
            common_start = ai + 1
            for start in range(common_start, gmin, mixed_pair_block):
                stop = min(gmin, start + mixed_pair_block)
                mixed = _bmm_left_fixed(ah[1 + ai], bh[1 + start:1 + stop])
                other_mixed = _bmm_right_fixed(
                    ah[1 + start:1 + stop], bh[1 + ai])
                mixed.add_(other_mixed)
                del other_mixed
                mixed.abs_()
                reduced = mixed.sum(dim=0)
                big.add_(reduced)
                del reduced, mixed
            if gb > gmin:
                for start in range(gmin, gb, mixed_pair_block):
                    stop = min(gb, start + mixed_pair_block)
                    mixed = _bmm_left_fixed(
                        ah[1 + ai], bh[1 + start:1 + stop])
                    mixed.abs_()
                    reduced = mixed.sum(dim=0)
                    big.add_(reduced)
                    del reduced, mixed
            if ga > gmin:
                for start in range(gmin, ga, mixed_pair_block):
                    stop = min(ga, start + mixed_pair_block)
                    mixed = _bmm_right_fixed(
                        ah[1 + start:1 + stop], bh[1 + ai])
                    mixed.abs_()
                    reduced = mixed.sum(dim=0)
                    big.add_(reduced)
                    del reduced, mixed

        # Nonnegative sum is rounded outward.  The gamma factor covers the
        # changed summation tree; it changes numerical enclosure only, never
        # the native real relaxation or fresh-symbol layout.
        unit = torch.finfo(a.dtype).eps
        gamma = unit * (4 * max(1, gmax) + 16)
        fresh_weight = (half + big) * (1.0 + gamma)
        fresh_weight = torch.nextafter(
            fresh_weight, torch.full_like(fresh_weight, float("inf")))
        first = 1 + gmax + head * na * nb
        indices = torch.arange(first, first + na * nb, device=a.device)
        out[head, indices, mask] = fresh_weight[mask]

    return make_zonotope_new_weights_same_args(
        out, source_zonotope=left, clone=False)


class BoundedNativeSemanticOperators(native_proof.NativeSemanticOperators):
    """Validated native facade with bounded materialization and hash capture."""

    def __init__(self, revision=native_proof.PINNED_REVISION,
                 pair_block: int = DEFAULT_PAIR_BLOCK):
        super().__init__(revision)
        self.pair_block = int(pair_block)

    def _run(self, family: str, inputs: list, call: Callable[[], Any],
             equation: str, range_logic: str, policy: str, **extra):
        total_started = time.perf_counter()
        before = [state_record_blockwise(z) for z in inputs]
        started = time.perf_counter()
        output = call()
        native_elapsed = time.perf_counter() - started
        if not bool(torch.isfinite(output.zonotope_w).all()):
            raise RuntimeError(f"{family}: bounded native semantic result is nonfinite")
        certificate = {
            "schema": SCHEMA,
            "family": family,
            "revision": self.revision,
            "equation": equation,
            "range_logic": range_logic,
            "symbol_policy": policy,
            "inputs": before,
            "output": state_record_blockwise(output),
            "native_result_unchanged": True,
            "generic_semantic_remainder_used": False,
            "bounded_materialization_only": True,
            "native_operator_seconds": native_elapsed,
            **extra,
        }
        total = time.perf_counter() - total_started
        certificate["facade_total_seconds"] = total
        certificate["numerical_overhead_seconds"] = max(0.0, total-native_elapsed)
        return native_proof.NativeResult(output, certificate)

    def qk(self, q, k):
        return self._run(
            "QK", [q, k],
            lambda: precise_dot_blockwise(q, k, self.pair_block),
            "cq.ck + sum_g(cq.bg+ag.ck)e_g + sum_gh ag.bh e_g e_h",
            "native precise same-ID/cross-ID quadratic enclosure; blockwise outward summation",
            "retain aligned shared symbols; one fresh symbol per head/query/key in native order",
            pair_block=self.pair_block)

    def attention_value(self, probability, value):
        transposed = value.t()
        return self._run(
            "A.V", [probability, value],
            lambda: precise_dot_blockwise(
                probability, transposed, self.pair_block,
                mixed_temporary_cap_bytes=AV_MIXED_TEMPORARY_CAP_BYTES),
            "sum_j p_ij V_j with native precise bilinear transformer",
            "native precise same-ID/cross-ID quadratic enclosure; blockwise outward summation",
            "retain aligned shared symbols; one fresh symbol per head/query/output in native order",
            pair_block=self.pair_block,
            mixed_temporary_cap_bytes=AV_MIXED_TEMPORARY_CAP_BYTES)

    def softmax(self, z, *, no_constraints=False):
        previous = bool(z.args.batch_softmax_computation)
        z.args.batch_softmax_computation = True
        try:
            return self._run(
                "softmax", [z],
                lambda: z.softmax(
                    use_new_softmax=True, no_constraints=no_constraints,
                    use_new_reciprocal=True),
                "p_i=1/sum_j(exp(s_j-s_i))",
                "native pairwise-difference/exp/reciprocal equations, materialized one head at a time",
                "native exp and reciprocal fresh symbols in native head/row/value order",
                softmax_chunk_axis="attention_head")
        finally:
            z.args.batch_softmax_computation = previous
