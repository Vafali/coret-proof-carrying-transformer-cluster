#!/usr/bin/env python3
"""Exact token-support execution for the frozen native precise dot transformer.

Support masks are proof metadata.  A zero bit authorizes skipping a coefficient
only after :func:`validate_support` has established that the claimed topology is
consistent with the operand.  No magnitude threshold is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import math
import time

import torch

import coret_bounded_native_execution_v1 as frozen
import coret_native_semantics_proof_v1 as native_proof


SCHEMA = "CORET_STRUCTURAL_SUPPORT_PRECISE_DOT_V1"
DEFAULT_GENERATOR_TILE = 32
AV_GENERATOR_TILE = 112
AV_TEMPORARY_CAP_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class SupportProof:
    masks: tuple[int, ...]
    ids: tuple[str, ...]
    reasons: tuple[str, ...]
    num_tokens: int

    def __post_init__(self):
        if not (len(self.masks) == len(self.ids) == len(self.reasons)):
            raise ValueError("support metadata lengths differ")
        limit = (1 << self.num_tokens) - 1
        if any(mask < 0 or mask > limit for mask in self.masks):
            raise ValueError("support mask is outside the token universe")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("support generator IDs are not unique")


def local_mask(token: int) -> int:
    return 1 << int(token)


def dense_mask(num_tokens: int) -> int:
    return (1 << int(num_tokens)) - 1


def attach_support(z, proof: SupportProof):
    if z.num_error_terms != len(proof.masks):
        raise RuntimeError("support/generator count mismatch")
    z._coret_token_support = proof
    return z


def get_support(z) -> SupportProof:
    proof = getattr(z, "_coret_token_support", None)
    if not isinstance(proof, SupportProof):
        raise RuntimeError("exact token-support provenance is unavailable")
    if z.num_error_terms != len(proof.masks):
        raise RuntimeError("stale token-support provenance")
    return proof


def validate_support(z, proof: SupportProof | None = None,
                     token_axis: int = -2) -> dict[str, Any]:
    """Independently reject coefficients outside a claimed structural mask."""
    proof = get_support(z) if proof is None else proof
    weights = z.zonotope_w
    generators = weights[:, 1:] if weights.ndim == 4 else weights[1:]
    if weights.ndim == 4:
        # [H,G,T,D] or [H,G,D,T]
        axis = token_axis if token_axis >= 0 else generators.ndim + token_axis
        present = generators.ne(0).movedim(axis, -1).flatten(0, -2).any(0)
    elif weights.ndim == 3:
        axis = token_axis if token_axis >= 0 else generators.ndim + token_axis
        present = generators.ne(0).movedim(axis, -1).flatten(0, -2).any(0)
    else:
        raise RuntimeError("unsupported state rank for support validation")
    # moved/flattened representation above is [all-other,G?] only for rank-4;
    # compute directly to avoid depending on layout details.
    if weights.ndim == 4 and token_axis in (-2, 2):
        present = generators.ne(0).any(dim=(0, 3))
    elif weights.ndim == 4 and token_axis in (-1, 3):
        present = generators.ne(0).any(dim=(0, 2))
    elif weights.ndim == 3 and token_axis in (-2, 1):
        present = generators.ne(0).any(dim=2)
    elif weights.ndim == 3 and token_axis in (-1, 2):
        present = generators.ne(0).any(dim=1)
    claimed = torch.tensor(
        [[bool(mask & local_mask(token)) for token in range(proof.num_tokens)]
         for mask in proof.masks], device=present.device, dtype=torch.bool)
    if present.shape != claimed.shape:
        raise RuntimeError(
            f"support validation shape mismatch {tuple(present.shape)} != {tuple(claimed.shape)}")
    violation = present & ~claimed
    if bool(violation.any()):
        index = violation.nonzero()[0].tolist()
        raise RuntimeError(f"coefficient outside proven support at {index}")
    return {
        "schema": SCHEMA,
        "generator_count": len(proof.masks),
        "empty_count": sum(mask == 0 for mask in proof.masks),
        "class_count": len(set(proof.masks)),
        "validated": True,
    }


def _classes(proof: SupportProof, maximum: int | None = None):
    groups: dict[int, list[int]] = {}
    stop = len(proof.masks) if maximum is None else min(maximum, len(proof.masks))
    for index, mask in enumerate(proof.masks[:stop]):
        if mask:
            groups.setdefault(mask, []).append(index)
    return groups


def _ordered_qk_radius(a: torch.Tensor, b: torch.Tensor,
                       left: SupportProof, right: SupportProof,
                       gmin: int):
    """QK radius using only proven active generator/output coordinates."""
    heads, _, na, dimension = a.shape
    nb = b.shape[2]
    radius = torch.zeros(heads, na, nb, device=a.device, dtype=a.dtype)
    macs = 0
    launches = 0
    for query in range(na):
        lbit = local_mask(query)
        for key in range(nb):
            rbit = local_mask(key)
            left_active = [index for index in range(gmin)
                           if left.masks[index] & lbit]
            right_active = [index for index in range(gmin)
                            if right.masks[index] & rbit]
            if not left_active or not right_active:
                continue
            active = [
                index for index in range(gmin)
                if (left.masks[index] & lbit) or (right.masks[index] & rbit)
            ]
            indices = torch.tensor(active, device=a.device, dtype=torch.long)
            av = a[:, 1 + indices, query, :]
            bv = b[:, 1 + indices, key, :]
            first = torch.bmm(av, bv.transpose(1, 2))
            launches += 1
            symmetric = first + first.transpose(1, 2)
            diagonal = torch.diagonal(symmetric, dim1=1, dim2=2)
            radius[:, query, key] = (
                0.5 * symmetric.abs().sum(dim=(1, 2))
                - 0.25 * diagonal.abs().sum(dim=1))
            macs += heads * len(active) * len(active) * dimension
    return radius, macs, launches


def _sync_for(tensor: torch.Tensor):
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def _av_radius(a: torch.Tensor, b: torch.Tensor,
               left: SupportProof, right: SupportProof,
               ga: int, gb: int, tile: int):
    """Grouped class-pair evaluation of the exact A.V quadratic radius."""
    heads, _, queries, keys = a.shape
    features = b.shape[2]
    radius = torch.zeros(heads, queries, features, device=a.device, dtype=a.dtype)
    gmax = max(ga, gb)
    # Class identity includes both probability query support and value-key
    # support.  In particular, probability-only suffix generators have an
    # empty value mask but are not globally empty: they interact with every
    # shared value generator through the symmetric cross term.
    groups: dict[tuple[int, int], list[int]] = {}
    for index in range(gmax):
        lm = left.masks[index] if index < ga else 0
        rm = right.masks[index] if index < gb else 0
        if lm or rm:
            groups.setdefault((lm, rm), []).append(index)
    ag = a[:, 1:1+ga]
    if ga < gmax:
        ag = torch.cat([ag, torch.zeros(
            heads, gmax-ga, queries, keys, device=a.device, dtype=a.dtype)], 1)
    bg = b[:, 1:1+gb]
    if gb < gmax:
        bg = torch.cat([bg, torch.zeros(
            heads, gmax-gb, features, keys, device=b.device, dtype=b.dtype)], 1)
    class_items = sorted(groups.items())
    macs = 0
    launches = 0
    peak_temporary = 0
    for class_pos, ((left_i, right_i), members_i) in enumerate(class_items):
        keys_i = [key for key in range(keys) if right_i & local_mask(key)]
        for (left_j, right_j), members_j in class_items[class_pos:]:
            keys_j = [key for key in range(keys) if right_j & local_mask(key)]
            same_class = (left_i, right_i) == (left_j, right_j)
            for ii in range(0, len(members_i), tile):
                inds_i = members_i[ii:ii + tile]
                start_j = ii if same_class else 0
                for jj in range(start_j, len(members_j), tile):
                    inds_j = members_j[jj:jj + tile]
                    if same_class and jj < ii:
                        continue
                    ti = torch.tensor(inds_i, device=a.device, dtype=torch.long)
                    tj = torch.tensor(inds_j, device=a.device, dtype=torch.long)
                    term = None
                    if keys_j:
                        kj = torch.tensor(keys_j, device=a.device, dtype=torch.long)
                        term = torch.einsum(
                            "hiqk,hjfk->hijqf",
                            ag[:, ti, :, :].index_select(3, kj),
                            bg[:, tj, :, :].index_select(3, kj))
                        macs += (heads * len(inds_i) * len(inds_j)
                                 * queries * features * len(keys_j))
                    if keys_i:
                        ki = torch.tensor(keys_i, device=a.device, dtype=torch.long)
                        other = torch.einsum(
                            "hifk,hjqk->hijqf",
                            bg[:, ti, :, :].index_select(3, ki),
                            ag[:, tj, :, :].index_select(3, ki))
                        macs += (heads * len(inds_i) * len(inds_j)
                                 * queries * features * len(keys_i))
                        term = other if term is None else term.add_(other)
                    if term is None:
                        continue
                    launches += 1
                    peak_temporary = max(
                        peak_temporary, term.numel() * term.element_size())
                    if same_class and ii == jj:
                        diagonal = torch.diagonal(term, dim1=1, dim2=2)
                        radius.add_(0.5 * term.abs().sum(dim=(1, 2)))
                        radius.sub_(0.25 * diagonal.abs().sum(dim=-1))
                    else:
                        radius.add_(term.abs().sum(dim=(1, 2)))
                    del term
    return radius, macs, launches, peak_temporary


def _bounded_av_tile(a: torch.Tensor, b: torch.Tensor, requested: int,
                     temporary_cap_bytes: int) -> int:
    """Mechanically cap one grouped A.V temporary before any timing."""
    if requested <= 0 or temporary_cap_bytes <= 0:
        raise ValueError("A.V tile and temporary cap must be positive")
    heads, queries = a.shape[0], a.shape[2]
    features = b.shape[2]
    bytes_per_pair = (heads * queries * features * a.element_size())
    cap = math.isqrt(max(1, temporary_cap_bytes // bytes_per_pair))
    return max(1, min(int(requested), int(cap)))


def _norm_bound(a: torch.Tensor, b: torch.Tensor, ga: int, gb: int):
    # Exact-real O(gD) upper bound on the ordered quadratic absolute sum.
    sa = a[:, 1:1 + ga].abs().sum(dim=1)
    sb = b[:, 1:1 + gb].abs().sum(dim=1)
    return torch.bmm(sa, sb.transpose(1, 2))


def precise_dot_structural(left, right, left_support: SupportProof,
                           right_support: SupportProof, *, mode: str,
                           generator_tile: int = DEFAULT_GENERATOR_TILE,
                           av_temporary_cap_bytes: int = AV_TEMPORARY_CAP_BYTES,
                           diagnostics: dict[str, Any] | None = None):
    """Frozen precise-dot transformer with only proven-zero work omitted."""
    from Verifiers.Zonotope import make_zonotope_new_weights_same_args

    if mode not in {"QK", "A.V"}:
        raise ValueError("mode must be QK or A.V")
    if generator_tile <= 0:
        raise ValueError("generator_tile must be positive")
    a, b = left.zonotope_w, right.zonotope_w
    timing: dict[str, float] = {}
    _sync_for(a)
    stage = time.perf_counter()
    validate_support(left, left_support, token_axis=-2 if mode == "QK" else -2)
    validate_support(right, right_support, token_axis=-2 if mode == "QK" else -1)
    _sync_for(a)
    timing["support_bookkeeping_seconds"] = time.perf_counter() - stage

    stage = time.perf_counter()
    heads, ga, gb = a.shape[0], left.num_error_terms, right.num_error_terms
    gmin, gmax = min(ga, gb), max(ga, gb)
    na, nb = left.num_words, right.num_words
    fresh = heads * na * nb
    out = torch.zeros(heads, 1 + gmax + fresh, na, nb,
                      device=a.device, dtype=a.dtype)
    mask = torch.ones(na, nb, dtype=torch.bool, device=a.device)

    diagonal_all = []
    for head in range(heads):
        ah, bh = a[head], b[head]
        out[head, 0] = ah[0] @ bh[0].t()
        diagonal = torch.bmm(
            ah[1:1 + gmin], bh[1:1 + gmin].transpose(1, 2))
        out[head, 0] += 0.5 * diagonal.sum(dim=0)
        diagonal_all.append(0.5 * diagonal.abs().sum(dim=0))
        del diagonal
        for start in range(0, gmax, frozen.DEFAULT_PAIR_BLOCK):
            stop = min(gmax, start + frozen.DEFAULT_PAIR_BLOCK)
            retained = torch.zeros(stop - start, na, nb,
                                   device=a.device, dtype=a.dtype)
            if start < gb:
                bend = min(stop, gb)
                retained[:bend-start] += frozen._bmm_left_fixed(
                    ah[0], bh[1+start:1+bend])
            if start < ga:
                aend = min(stop, ga)
                retained[:aend-start] += frozen._bmm_right_fixed(
                    ah[1+start:1+aend], bh[0])
            out[head, 1+start:1+stop] = retained
    half = torch.stack(diagonal_all)
    _sync_for(a)
    timing["center_retained_seconds"] = time.perf_counter() - stage

    stage = time.perf_counter()
    if mode == "QK":
        raw_radius, executed_macs, launches = _ordered_qk_radius(
            a, b, left_support, right_support, gmin)
        peak_temporary = 0
    else:
        effective_tile = _bounded_av_tile(
            a, b, generator_tile, av_temporary_cap_bytes)
        raw_radius, executed_macs, launches, peak_temporary = _av_radius(
            a, b, left_support, right_support, ga, gb, effective_tile)
    _sync_for(a)
    timing["grouped_radius_seconds"] = time.perf_counter() - stage

    # The grouped result includes the diagonal radius already.  Add a cheap,
    # independently reproducible envelope covering both grouped arithmetic and
    # the frozen native float32 accumulation; this can only widen fresh rows.
    stage = time.perf_counter()
    unit = torch.finfo(a.dtype).eps
    # Four conservative passes cover: each dot/symmetrization, grouped positive
    # reduction, the frozen native reduction tree used by the parity oracle,
    # and final accumulation.  This is deliberately independent of measured
    # coefficient magnitudes and is checked from the O(gD) norm bound.
    operation_factor = float(64 * a.shape[-1] + 64 * max(1, gmax) + 512)
    bound = _norm_bound(a, b, ga, gb)
    envelope = bound * (unit * operation_factor)
    certified = raw_radius + envelope
    certified = torch.nextafter(
        certified, torch.full_like(certified, float("inf")))
    _sync_for(a)
    timing["outward_envelope_seconds"] = time.perf_counter() - stage

    stage = time.perf_counter()
    for head in range(heads):
        first = 1 + gmax + head * na * nb
        indices = torch.arange(first, first + na * nb, device=a.device)
        out[head, indices, mask] = certified[head, mask]
    result = make_zonotope_new_weights_same_args(out, source_zonotope=left,
                                                  clone=False)
    fresh_masks = tuple(
        local_mask(query)
        for _head in range(heads)
        for query in range(na)
        for _column in range(nb))
    if mode == "QK":
        # A key-local coefficient occupies one score column in every query
        # row, so query-row provenance becomes dense for every nonempty ID.
        all_queries = dense_mask(left_support.num_tokens)
        inherited = tuple(
            all_queries if ((left_support.masks[i] if i < ga else 0)
                            or (right_support.masks[i] if i < gb else 0))
            else 0 for i in range(gmax))
    else:
        # A value-local coefficient is combined with nominal probabilities at
        # every query; probability-only suffix symbols keep their query mask.
        all_queries = dense_mask(left_support.num_tokens)
        inherited = tuple(
            all_queries if (right_support.masks[i] if i < gb else 0)
            else (left_support.masks[i] if i < ga else 0)
            for i in range(gmax))
    proof = SupportProof(
        inherited + fresh_masks,
        tuple(f"retained_{i:06d}" for i in range(gmax))
        + tuple(f"{mode}_fresh_{i:06d}" for i in range(fresh)),
        tuple("native_same_id_union" for _ in range(gmax))
        + tuple("native_output_coordinate_fresh" for _ in range(fresh)),
        left_support.num_tokens)
    attach_support(result, proof)
    _sync_for(a)
    timing["scatter_and_construct_seconds"] = time.perf_counter() - stage
    nominal_macs = heads * na * nb * ga * gb * a.shape[-1]
    if diagnostics is not None:
        diagnostics.update({
            "schema": SCHEMA,
            "mode": mode,
            "support_class_count_left": len(set(left_support.masks)),
            "support_class_count_right": len(set(right_support.masks)),
            "nominal_quadratic_MACs": int(nominal_macs),
            "executed_quadratic_MACs": int(executed_macs),
            "structurally_skipped_quadratic_MACs": int(
                nominal_macs - executed_macs),
            "arithmetic_reduction": (float(nominal_macs) / executed_macs
                                     if executed_macs else float("inf")),
            "grouped_launches": int(launches),
            "peak_temporary_bytes": int(peak_temporary),
            "requested_generator_tile": int(generator_tile),
            "effective_generator_tile": int(
                effective_tile if mode == "A.V" else 0),
            "temporary_cap_bytes": int(
                av_temporary_cap_bytes if mode == "A.V" else 0),
            "envelope_max": float(envelope.max()),
            "operation_factor": operation_factor,
            "support_validated": True,
            "stage_timing_seconds": timing,
        })
    return result


def proof_from_masks(masks: Iterable[int], num_tokens: int,
                     prefix: str = "generator") -> SupportProof:
    values = tuple(int(mask) for mask in masks)
    return SupportProof(
        values,
        tuple(f"{prefix}_{index:06d}" for index in range(len(values))),
        tuple("explicit_provenance_fixture" for _ in values),
        num_tokens)


def _with_ids(masks, num_tokens, prefix, reason):
    masks = tuple(masks)
    return SupportProof(
        masks, tuple(f"{prefix}_{i:06d}" for i in range(len(masks))),
        tuple(reason for _ in masks), num_tokens)


def _union_aligned(first: SupportProof, second: SupportProof, count: int):
    return tuple(
        (first.masks[i] if i < len(first.masks) else 0)
        | (second.masks[i] if i < len(second.masks) else 0)
        for i in range(count))


def _aligned_union_proof(first: SupportProof, second: SupportProof,
                         count: int, label: str) -> SupportProof:
    ids = []
    for index in range(count):
        left_id = first.ids[index] if index < len(first.ids) else None
        right_id = second.ids[index] if index < len(second.ids) else None
        if left_id is not None and right_id is not None and left_id != right_id:
            raise RuntimeError(
                f"{label}: aligned native symbol IDs differ at {index}")
        ids.append(left_id if left_id is not None else right_id)
    return SupportProof(
        _union_aligned(first, second, count), tuple(ids),
        tuple("exact_same_id_residual_union" for _ in range(count)),
        first.num_tokens)


def _replace_proof_ids(proof: SupportProof, inherited: SupportProof,
                       inherited_count: int, fresh_prefix: str) -> SupportProof:
    if inherited_count > len(proof.masks):
        raise RuntimeError("inherited support exceeds output generator count")
    fresh = len(proof.masks) - inherited_count
    return SupportProof(
        proof.masks,
        inherited.ids[:inherited_count] + tuple(
            f"{fresh_prefix}_{i:06d}" for i in range(fresh)),
        inherited.reasons[:inherited_count] + tuple(
            "native_output_coordinate_fresh" for _ in range(fresh)),
        proof.num_tokens)


class StructuralNativeSemanticOperators(frozen.BoundedNativeSemanticOperators):
    """Fixture/production facade carrying exact token-topology provenance."""
    def __init__(self, revision=native_proof.PINNED_REVISION,
                 pair_block=frozen.DEFAULT_PAIR_BLOCK,
                 generator_tile=AV_GENERATOR_TILE):
        super().__init__(revision=revision, pair_block=pair_block)
        self.generator_tile = int(generator_tile)
        self._hidden: SupportProof | None = None
        self._post_attention: SupportProof | None = None
        self._relu: SupportProof | None = None
        self._probability: SupportProof | None = None
        self._score: SupportProof | None = None
        self._value: SupportProof | None = None
        self._attention_output: SupportProof | None = None
        self._layer_norm_index = 0
        self._qk_index = 0
        self._softmax_index = 0
        self._av_index = 0
        self._relu_index = 0
        self._reduction_index = 0

    def _input_proof(self, z):
        proof = getattr(z, "_coret_token_support", None)
        if proof is not None:
            return proof
        n = z.num_words
        token = int(z.perturbed_word_index)
        return _with_ids(
            [local_mask(token)] * z.num_error_terms, n, "input_source",
            "original_perturbed_embedding_coordinate")

    def _layer_norm_output(self, source: SupportProof, output, label):
        n, d = output.num_words, output.word_embedding_size
        affected = 0
        for mask in source.masks:
            affected |= mask
        fresh_masks = []
        # square-and-sum emits one slot per token, including structural zeros.
        fresh_masks.extend(local_mask(t) if affected & local_mask(t) else 0
                           for t in range(n))
        active_tokens = [t for t in range(n) if affected & local_mask(t)]
        for _stage in ("sqrt", "reciprocal"):
            for token in active_tokens:
                fresh_masks.extend([local_mask(token)] * d)
        # Native final product emits one slot per tensor coordinate.
        for token in range(n):
            fresh_masks.extend([
                local_mask(token) if affected & local_mask(token) else 0] * d)
        needed = output.num_error_terms - len(source.masks)
        if len(fresh_masks) != needed:
            raise RuntimeError(
                f"{label}: LayerNorm support transition mismatch "
                f"{len(fresh_masks)} != {needed}")
        return SupportProof(
            source.masks + tuple(fresh_masks),
            source.ids + tuple(
                f"{label}_fresh_{i:06d}" for i in range(len(fresh_masks))),
            source.reasons + tuple(
                "native_tokenwise_layernorm_topology" for _ in fresh_masks), n)

    def layer_norm(self, z, normalizer, mode="standard"):
        index = self._layer_norm_index
        self._layer_norm_index += 1
        if index == 0:
            source = self._input_proof(z)
        elif index % 2 == 1:
            if self._hidden is None or self._attention_output is None:
                raise RuntimeError("attention residual support state unavailable")
            count = max(len(self._hidden.masks),
                        len(self._attention_output.masks))
            source = _aligned_union_proof(
                self._hidden, self._attention_output, count,
                f"residual_attention_{index}")
        else:
            if self._post_attention is None or self._relu is None:
                raise RuntimeError("FFN residual support state unavailable")
            count = max(len(self._post_attention.masks), len(self._relu.masks))
            source = _aligned_union_proof(
                self._post_attention, self._relu, count,
                f"residual_ffn_{index}")
        result = super().layer_norm(z, normalizer, mode)
        proof = self._layer_norm_output(source, result.value,
                                        f"layernorm_{index}")
        attach_support(result.value, proof)
        result.certificate["support_proof"] = validate_support(result.value, proof)
        if index == 0 or index % 2 == 0:
            self._hidden = proof
        else:
            self._post_attention = proof
        return result

    def reduce(self, z, maximum: int):
        reduction_index = self._reduction_index
        self._reduction_index += 1
        source = self._hidden if self._hidden is not None else get_support(z)
        if z.num_error_terms <= maximum:
            result = super().reduce(z, maximum)
            proof = source
        else:
            special = z.num_input_error_terms_special_norm
            n, d = z.num_words, z.word_embedding_size
            keep_other = maximum - special - n * d
            remove = z.num_error_terms - special - keep_other
            candidates = z.zonotope_w[1 + special:]
            metric = candidates.abs().sum(dim=[1, 2])
            _, removed = torch.topk(metric, k=remove, largest=False,
                                    sorted=False)
            kept = torch.ones(metric.numel(), dtype=torch.bool,
                              device=metric.device)
            kept[removed] = False
            kept_indices = kept.nonzero().flatten().cpu().tolist()
            masks = list(source.masks[:special])
            masks.extend(source.masks[special + i] for i in kept_indices)
            masks.extend(local_mask(token) for token in range(n)
                         for _feature in range(d))
            ids = list(source.ids[:special])
            ids.extend(source.ids[special + i] for i in kept_indices)
            reasons = list(source.reasons[:special])
            reasons.extend(source.reasons[special + i] for i in kept_indices)
            box_count = n * d
            ids.extend(f"reduction_{reduction_index}_box_{i:06d}"
                       for i in range(box_count))
            reasons.extend("native_coordinate_box_generator"
                           for _ in range(box_count))
            result = super().reduce(z, maximum)
            proof = SupportProof(tuple(masks), tuple(ids), tuple(reasons), n)
        attach_support(result.value, proof)
        result.certificate["support_proof"] = validate_support(result.value, proof)
        self._hidden = proof
        return result

    def qk(self, q, k):
        qk_index = self._qk_index
        self._qk_index += 1
        if self._hidden is None:
            raise RuntimeError("QK hidden support state unavailable")
        attach_support(q, self._hidden)
        attach_support(k, self._hidden)
        diagnostics = {}
        output = precise_dot_structural(
            q, k, self._hidden, self._hidden, mode="QK",
            generator_tile=self.generator_tile, diagnostics=diagnostics)
        result = self._run(
            "QK", [q, k], lambda: output,
            "cq.ck + sum_g(cq.bg+ag.ck)e_g + sum_gh ag.bh e_g e_h",
            "native precise equation with provenance-proven zero blocks omitted",
            "native retained and fresh symbol order", pair_block=self.pair_block)
        proof = _replace_proof_ids(
            get_support(output), self._hidden, len(self._hidden.masks),
            f"qk_{qk_index}_fresh")
        attach_support(output, proof)
        result.certificate["support_diagnostics"] = diagnostics
        result.certificate["support_proof"] = validate_support(output, proof)
        self._value = self._hidden
        self._score = proof
        return result

    def softmax(self, z, *, no_constraints=False):
        softmax_index = self._softmax_index
        self._softmax_index += 1
        source = getattr(z, "_coret_token_support", None)
        if source is None:
            source = self._score
        if source is None:
            raise RuntimeError("softmax score support state unavailable")
        attach_support(z, source)
        result = super().softmax(z, no_constraints=no_constraints)
        fresh = result.value.num_error_terms - len(source.masks)
        n = source.num_tokens
        heads = z.zonotope_w.shape[0]
        coordinate_count = heads * n * n
        if fresh != 2 * coordinate_count:
            raise RuntimeError("softmax fresh-symbol topology mismatch")
        fresh_masks = tuple(
            local_mask(query) for _head in range(heads)
            for query in range(n) for _key in range(n))
        # Reciprocal removes the head dimension, so native boolean indexing is
        # query/head/key before the head axis is restored.
        fresh_masks += tuple(
            local_mask(query) for query in range(n)
            for _head in range(heads) for _key in range(n))
        proof = SupportProof(
            source.masks + fresh_masks,
            source.ids + tuple(
                f"softmax_{softmax_index}_fresh_{i:06d}"
                for i in range(len(fresh_masks))),
            source.reasons + tuple(
                "native_row_local_softmax_topology" for _ in fresh_masks), n)
        attach_support(result.value, proof)
        result.certificate["support_proof"] = validate_support(result.value, proof)
        self._probability = proof
        return result

    def attention_value(self, probability, value):
        av_index = self._av_index
        self._av_index += 1
        if self._probability is None or self._value is None:
            raise RuntimeError("A.V support state unavailable")
        transposed = value.t()
        attach_support(probability, self._probability)
        attach_support(transposed, self._value)
        diagnostics = {}
        output = precise_dot_structural(
            probability, transposed, self._probability, self._value,
            mode="A.V", generator_tile=self.generator_tile,
            diagnostics=diagnostics)
        result = self._run(
            "A.V", [probability, value], lambda: output,
            "sum_j p_ij V_j with native precise bilinear transformer",
            "native precise equation with provenance-proven zero key work omitted",
            "native retained and fresh symbol order", pair_block=self.pair_block)
        temporary = get_support(output)
        gmax = max(len(self._probability.masks), len(self._value.masks))
        inherited_ids = []
        for index in range(gmax):
            left_id = (self._probability.ids[index]
                       if index < len(self._probability.ids) else None)
            right_id = (self._value.ids[index]
                        if index < len(self._value.ids) else None)
            if left_id is not None and right_id is not None and left_id != right_id:
                raise RuntimeError(f"A.V aligned symbol IDs differ at {index}")
            inherited_ids.append(left_id if left_id is not None else right_id)
        fresh = len(temporary.masks) - gmax
        proof = SupportProof(
            temporary.masks,
            tuple(inherited_ids) + tuple(
                f"av_{av_index}_fresh_{i:06d}" for i in range(fresh)),
            tuple("native_same_id_union" for _ in range(gmax)) + tuple(
                "native_output_coordinate_fresh" for _ in range(fresh)),
            temporary.num_tokens)
        attach_support(output, proof)
        result.certificate["support_diagnostics"] = diagnostics
        result.certificate["support_proof"] = validate_support(output, proof)
        self._attention_output = proof
        return result

    def relu(self, z):
        relu_index = self._relu_index
        self._relu_index += 1
        if self._post_attention is None:
            raise RuntimeError("ReLU source support unavailable")
        lower, upper = z.concretize()
        unstable = (lower * upper < 0)
        result = super().relu(z)
        fresh_masks = tuple(
            local_mask(index[0]) for index in unstable.nonzero().tolist())
        proof = SupportProof(
            self._post_attention.masks + fresh_masks,
            self._post_attention.ids + tuple(
                f"relu_{relu_index}_fresh_{i:06d}"
                for i in range(len(fresh_masks))),
            self._post_attention.reasons + tuple(
                "native_elementwise_unstable_coordinate" for _ in fresh_masks),
            self._post_attention.num_tokens)
        attach_support(result.value, proof)
        result.certificate["support_proof"] = validate_support(result.value, proof)
        self._relu = proof
        return result

    def tanh(self, z):
        source = self._hidden if self._hidden is not None else get_support(z)
        lower, upper = z.concretize()
        different = (lower != upper)
        result = super().tanh(z)
        # Pooler operates on one selected token; output token universe is one.
        fresh_masks = tuple(1 for _ in different.nonzero().tolist())
        existing = tuple(1 if mask else 0 for mask in source.masks)
        proof = SupportProof(
            existing + fresh_masks,
            source.ids + tuple(
                f"tanh_fresh_{i:06d}" for i in range(len(fresh_masks))),
            source.reasons + tuple(
                "native_elementwise_pooler_coordinate" for _ in fresh_masks), 1)
        attach_support(result.value, proof)
        result.certificate["support_proof"] = validate_support(
            result.value, proof, token_axis=-2)
        return result
