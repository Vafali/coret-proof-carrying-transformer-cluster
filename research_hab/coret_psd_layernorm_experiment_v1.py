#!/usr/bin/env python3
"""Opt-in PSD-domain certificate for one production LayerNorm boundary.

This module does not alter the variance affine form.  It independently proves
an additional semantic lower bound for the exact variance and supplies that
bound only to the native ``sqrt`` relaxation's domain/range selection.  The
ordinary production path never imports or calls this module.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "research_hab")]

import diagnose_psd_layernorm_variance_v1 as oracle
import coret_sound_fp64_block0_feasibility_v1 as sound
import coret_structural_support_precise_dot_v1 as structural


SCHEMA = "CORET_PSD_AWARE_LAYERNORM_EXPERIMENT_V1"
TARGET_LABEL = "block2_post_attention"
TARGET_LAYER_NORM_INDEX = 5
EPSILON = 1e-12


def _tensor_hash(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)},
        sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def _json_hash(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def variance_certificate_hash(certificate: dict) -> str:
    fields = (
        "schema", "target_label", "state_identity",
        "generic_negative_token_indices", "token_certificates",
        "variance_semantics", "optimizer_trusted",
        "independent_checker_accepts", "minimum_psd_lower",
        "semantic_variance_lower_by_token", "checked_token_certificates",
        "coefficient_clamping_used")
    if not all(field in certificate for field in fields):
        raise RuntimeError("variance certificate hash fields differ")
    return _json_hash({field: certificate[field] for field in fields})


def state_identity(state, proof) -> dict:
    low, high = sound._ranges(state)
    identity = {
        "weights_sha256": _tensor_hash(state.zonotope_w),
        "range_low_sha256": _tensor_hash(low),
        "range_high_sha256": _tensor_hash(high),
        "ordered_generator_ids_sha256": _json_hash(list(proof.ids)),
        "support_masks_sha256": _json_hash(list(proof.masks)),
        "provenance_reasons_sha256": _json_hash(list(proof.reasons)),
        "generator_count": int(state.num_error_terms),
        "token_count": int(state.num_words),
        "hidden_dimension": int(state.word_embedding_size),
    }
    identity["canonical_state_identity_sha256"] = _json_hash(identity)
    return identity


def _token_arrays(state, token: int):
    low, high = sound._ranges(state)
    weights = state.zonotope_w.detach().cpu().numpy()
    return (weights[0, token].copy(), weights[1:, token].copy(),
            low.detach().cpu().numpy().copy(),
            high.detach().cpu().numpy().copy())


def _witness_from_hex(values) -> np.ndarray:
    if not isinstance(values, list):
        raise RuntimeError("PSD witness encoding differs")
    try:
        result = np.asarray([float.fromhex(value) for value in values],
                            dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise RuntimeError("PSD witness encoding differs") from error
    return result


def _directed_sqrt_binary64(value: float, upward: bool) -> tuple[float, dict]:
    """Tight directed binary64 sqrt, certified by exact rational squares."""
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError("semantic sqrt range input is nonpositive")
    exact = Fraction.from_float(float(value))
    result = math.sqrt(value)
    direction = math.inf if upward else -math.inf
    predicate = ((lambda square: square >= exact) if upward
                 else (lambda square: square <= exact))
    while not predicate(Fraction.from_float(result) ** 2):
        result = math.nextafter(result, direction)
    opposite = -math.inf if upward else math.inf
    adjacent = math.nextafter(result, opposite)
    # Tightness is useful audit evidence: the adjacent float toward the exact
    # root must violate the selected directed relation unless sqrt is exact.
    result_square = Fraction.from_float(result) ** 2
    adjacent_square = Fraction.from_float(adjacent) ** 2
    exact_result = result_square == exact
    adjacent_excluded = (exact_result or not predicate(adjacent_square))
    if not adjacent_excluded:
        raise RuntimeError("semantic sqrt directed result is not tight")
    return result, {
        "input_binary64_hex": float(value).hex(),
        "output_binary64_hex": result.hex(),
        "direction": "upward" if upward else "downward",
        "exact_rational_square_relation": (
            "output^2>=input" if upward else "output^2<=input"),
        "relation_verified": True,
        "adjacent_toward_exact_excluded": True,
    }


def _semantic_sqrt_range(variance_lower_by_token, generic_sqrt_input_high):
    """Certify sqrt-result ranges without modifying its affine coefficients."""
    if generic_sqrt_input_high.ndim != 2:
        raise RuntimeError("semantic sqrt upper shape differs")
    tokens, width = generic_sqrt_input_high.shape
    if len(variance_lower_by_token) != tokens:
        raise RuntimeError("semantic variance token count differs")
    lower_values, upper_values, evidence = [], [], []
    for token in range(tokens):
        row = generic_sqrt_input_high[token]
        if not torch.equal(row, row[0].expand_as(row)):
            raise RuntimeError("native repeated variance upper differs by feature")
        variance_lower = float(variance_lower_by_token[token])
        # nextafter-down makes the binary64 addition itself a lower bound on
        # the exact sum of the already outward-safe variance lower and epsilon.
        sqrt_input_lower = math.nextafter(
            variance_lower + EPSILON, -math.inf)
        sqrt_input_upper = float(row[0])
        lower, lower_rounding = _directed_sqrt_binary64(
            sqrt_input_lower, False)
        upper, upper_rounding = _directed_sqrt_binary64(
            sqrt_input_upper, True)
        if lower <= EPSILON or lower > upper:
            raise RuntimeError("certified sqrt semantic range is invalid")
        lower_values.append(lower)
        upper_values.append(upper)
        evidence.append({
            "token_index": token,
            "variance_lower_binary64_hex": variance_lower.hex(),
            "sqrt_input_lower_binary64_hex": sqrt_input_lower.hex(),
            "sqrt_input_upper_binary64_hex": sqrt_input_upper.hex(),
            "sqrt_output_lower_binary64_hex": lower.hex(),
            "sqrt_output_upper_binary64_hex": upper.hex(),
            "sqrt_output_lower_rounding_proof": lower_rounding,
            "sqrt_output_upper_rounding_proof": upper_rounding,
        })
    device, dtype = generic_sqrt_input_high.device, generic_sqrt_input_high.dtype
    lower_tensor = torch.tensor(
        lower_values, device=device, dtype=dtype).reshape(tokens, 1).expand(
            tokens, width)
    upper_tensor = torch.tensor(
        upper_values, device=device, dtype=dtype).reshape(tokens, 1).expand(
            tokens, width)
    return lower_tensor, upper_tensor, evidence


def reciprocal_relaxation_coefficients(lower: torch.Tensor,
                                        upper: torch.Tensor):
    """Pinned DeepT original reciprocal formula over an authenticated hull."""
    if (lower.shape != upper.shape or not bool(torch.isfinite(lower).all())
            or not bool(torch.isfinite(upper).all())
            or bool((lower <= EPSILON).any()) or bool((lower > upper).any())):
        raise RuntimeError("reciprocal semantic range is invalid")
    lambdas = -1.0 / (upper * upper)
    bottom = 1.0 / upper - lambdas * upper
    top = 1.0 / lower - lambdas * lower
    constants = 0.5 * (top + bottom)
    fresh = 0.5 * (top - bottom)
    if (not bool(torch.isfinite(lambdas).all()
                 and torch.isfinite(constants).all()
                 and torch.isfinite(fresh).all())
            or bool((fresh < -1e-4).any())
            or bool((lambdas > 0).any())):
        raise RuntimeError("reciprocal semantic relaxation is malformed")
    return lambdas, constants, fresh


def build_certificate(state, proof, generic_variance_low: torch.Tensor) -> dict:
    """Use an untrusted optimizer, then persist only independently checked data."""
    structural.validate_support(state, proof, token_axis=1)
    identity = state_identity(state, proof)
    if (generic_variance_low.ndim != 2
            or tuple(generic_variance_low.shape) !=
            (state.num_words, state.word_embedding_size)):
        raise RuntimeError("generic variance shape differs")
    bad_tokens = [
        token for token in range(state.num_words)
        if float(generic_variance_low[token].min()) <= 0.0]
    if not bad_tokens:
        raise RuntimeError("PSD experiment requires a generic domain miss")
    records = []
    for token in bad_tokens:
        center, generators, low, high = _token_arrays(state, token)
        witness, optimizer = oracle.optimize_witness(
            center, generators, low, high)
        checked = oracle.recheck_dual_witness(
            center, generators, low, high, witness)
        records.append({
            "token_index": token,
            "witness_binary64_hex": [float(value).hex() for value in witness],
            "optimizer_diagnostic": optimizer,
            "outward_safe_lower": checked[
                "psd_dual_candidate_lower_outward_safe"],
            "checker_recomputation": checked,
        })
    candidate = {
        "schema": SCHEMA,
        "target_label": TARGET_LABEL,
        "state_identity": identity,
        "generic_negative_token_indices": bad_tokens,
        "token_certificates": records,
        "variance_semantics": (
            "d^-1*||P(c+sum_i g_i*epsilon_i)||_2^2 over authenticated "
            "cartesian generator ranges"),
        "optimizer_trusted": False,
    }
    verified = verify_certificate(state, proof, generic_variance_low, candidate)
    candidate.update(verified)
    return candidate


def verify_certificate(state, proof, generic_variance_low: torch.Tensor,
                       certificate: dict) -> dict:
    """Independent, fail-closed recheck of state identity and every witness."""
    if (not isinstance(certificate, dict)
            or certificate.get("schema") != SCHEMA
            or certificate.get("target_label") != TARGET_LABEL):
        raise RuntimeError("PSD LayerNorm certificate identity differs")
    expected_identity = state_identity(state, proof)
    if certificate.get("state_identity") != expected_identity:
        raise RuntimeError("PSD LayerNorm authenticated state differs")
    bad_tokens = [
        token for token in range(state.num_words)
        if float(generic_variance_low[token].min()) <= 0.0]
    if certificate.get("generic_negative_token_indices") != bad_tokens:
        raise RuntimeError("PSD LayerNorm generic-domain inventory differs")
    records = certificate.get("token_certificates")
    if (not isinstance(records, list)
            or [row.get("token_index") for row in records] != bad_tokens):
        raise RuntimeError("PSD LayerNorm token certificate ordering differs")
    checked_records = []
    token_lowers = generic_variance_low.detach().cpu().amin(dim=1).tolist()
    for row in records:
        token = int(row["token_index"])
        center, generators, low, high = _token_arrays(state, token)
        witness = _witness_from_hex(row.get("witness_binary64_hex"))
        checked = oracle.recheck_dual_witness(
            center, generators, low, high, witness)
        outward = checked["psd_dual_candidate_lower_outward_safe"]
        if (not math.isfinite(outward) or outward <= 0.0
                or row.get("outward_safe_lower") != outward
                or row.get("checker_recomputation") != checked):
            raise RuntimeError("PSD LayerNorm witness recheck failed")
        token_lowers[token] = max(token_lowers[token], outward)
        checked_records.append({"token_index": token, **checked})
    minimum = min(token_lowers)
    if not math.isfinite(minimum) or minimum <= 0.0:
        raise RuntimeError("PSD LayerNorm semantic lower is nonpositive")
    return {
        "independent_checker_accepts": True,
        "minimum_psd_lower": minimum,
        "semantic_variance_lower_by_token": token_lowers,
        "checked_token_certificates": checked_records,
        "coefficient_clamping_used": False,
    }


def _native_layernorm_with_semantic_lower(state, proof, normalizer, delegate,
                                          semantic_lower_by_token,
                                          variance_certificate: dict,
                                          milestones: dict):
    """Run pinned native formulas, changing only sqrt's witnessed domain hull."""
    if delegate._layer_norm_index != TARGET_LAYER_NORM_INDEX:
        raise RuntimeError("PSD experiment reached a non-target LayerNorm")
    if len(proof.masks) != state.num_error_terms:
        raise RuntimeError("PSD LayerNorm input proof count differs")
    structural.validate_support(state, proof, token_axis=1)
    n, d = state.num_words, state.word_embedding_size
    affected = 0
    for mask in proof.masks:
        affected |= mask
    average = torch.ones((d, d), dtype=state.zonotope_w.dtype,
                         device=state.device) / d
    centered = state.add(state.matmul(average).multiply(-1.0))
    variance = centered.square_and_sum_and_repeat().multiply(1.0 / d)
    square_count = variance.num_error_terms - centered.num_error_terms
    if square_count != n:
        raise RuntimeError("PSD LayerNorm variance allocation differs")
    square_masks = tuple(
        structural.local_mask(token)
        if affected & structural.local_mask(token) else 0
        for token in range(n))
    sqrt_input = variance.add(EPSILON)
    generic_low, generic_high = sqrt_input.concretize()
    witnessed_input_values = [
        math.nextafter(float(value) + EPSILON, -math.inf)
        for value in semantic_lower_by_token]
    witnessed_input = torch.tensor(
        witnessed_input_values, dtype=generic_low.dtype,
        device=generic_low.device).reshape(n, 1).expand_as(generic_low)
    semantic_low = torch.maximum(generic_low, witnessed_input)
    if (not bool(torch.isfinite(semantic_low).all())
            or bool((semantic_low <= EPSILON).any())
            or bool((semantic_low > generic_high).any())):
        raise RuntimeError("PSD LayerNorm witnessed sqrt interval is invalid")
    milestones["psd_variance_lower_consumed"] = True
    sqrt_predicate = semantic_low != generic_high
    sqrt_masks, sqrt_flat = structural._native_boolean_membership(
        sqrt_predicate, n, d, affected, "sqrt")

    original_concretize = sqrt_input.concretize
    sqrt_input.concretize = lambda: (semantic_low, generic_high)
    try:
        sqrt_state = sqrt_input.sqrt()
    finally:
        del sqrt_input.concretize
    # Fail closed if the temporary override affected anything beyond the one
    # intended call or if native allocation disagrees with its exact predicate.
    if sqrt_input.concretize.__func__ is not original_concretize.__func__:
        raise RuntimeError("PSD LayerNorm concretize restoration failed")
    if sqrt_state.num_error_terms - sqrt_input.num_error_terms != len(sqrt_flat):
        raise RuntimeError("PSD LayerNorm sqrt allocation differs")
    reciprocal_low, reciprocal_high, range_evidence = _semantic_sqrt_range(
        semantic_lower_by_token, generic_high)
    milestones["sqrt_semantic_lower_constructed"] = True
    semantic_range_certificate = {
        "schema": "CORET_PSD_LAYERNORM_SEMANTIC_RANGE_V1",
        "state_identity": variance_certificate["state_identity"],
        "target_label": TARGET_LABEL,
        "layernorm_index": TARGET_LAYER_NORM_INDEX,
        "variance_certificate_sha256": variance_certificate_hash(
            variance_certificate),
        "psd_witnesses_sha256": _json_hash([
            row["witness_binary64_hex"]
            for row in variance_certificate["token_certificates"]]),
        "variance_lower_by_token_binary64_hex": [
            float(value).hex() for value in semantic_lower_by_token],
        "directed_range_evidence": range_evidence,
        "sqrt_affine_coefficients_modified": False,
        "reciprocal_affine_input_coefficients_modified": False,
        "reciprocal_formula": "pinned_original_implementation_lambda=-1/u^2",
    }
    milestones["semantic_range_certificate"] = semantic_range_certificate
    # Audit the exact formula before invoking pinned DeepT.  The temporary
    # concretize override below supplies this same authenticated range to every
    # interval-dependent reciprocal operation, not merely its domain assert.
    reciprocal_relaxation_coefficients(reciprocal_low, reciprocal_high)
    reciprocal_predicate = reciprocal_low != reciprocal_high
    reciprocal_masks, reciprocal_flat = structural._native_boolean_membership(
        reciprocal_predicate, n, d, affected, "reciprocal")
    original_sqrt_concretize = sqrt_state.concretize
    sqrt_state.concretize = lambda: (reciprocal_low, reciprocal_high)
    milestones["reciprocal_semantic_range_consumed"] = True
    try:
        reciprocal = sqrt_state.reciprocal(
            original_implementation=True, y_positive_constraint=False)
    finally:
        del sqrt_state.concretize
    if sqrt_state.concretize.__func__ is not original_sqrt_concretize.__func__:
        raise RuntimeError("PSD reciprocal concretize restoration failed")
    if reciprocal.num_error_terms - sqrt_state.num_error_terms != len(
            reciprocal_flat):
        raise RuntimeError("PSD LayerNorm reciprocal allocation differs")
    expanded = centered.expand_error_terms_to_match_zonotope(reciprocal)
    product = expanded.multiply(reciprocal)
    product_count = product.num_error_terms - reciprocal.num_error_terms
    if product_count != n * d:
        raise RuntimeError("PSD LayerNorm final product allocation differs")
    product_masks = tuple(
        structural.local_mask(token)
        if affected & structural.local_mask(token) else 0
        for token in range(n) for _feature in range(d))
    output = product.multiply(normalizer.weight).add(normalizer.bias)
    fresh_masks = square_masks + sqrt_masks + reciprocal_masks + product_masks
    fresh_reasons = (
        tuple("native_layernorm_variance_coordinate" for _ in square_masks)
        + tuple("psd_witnessed_native_layernorm_sqrt_l_ne_u"
                for _ in sqrt_masks)
        + tuple("native_layernorm_reciprocal_l_ne_u"
                for _ in reciprocal_masks)
        + tuple("native_layernorm_final_product_coordinate"
                for _ in product_masks))
    output_proof = delegate._layer_norm_output(
        proof, output, "layernorm_5", fresh_masks, fresh_reasons)
    structural.attach_support(output, output_proof)
    structural.validate_support(output, output_proof, token_axis=1)
    delegate._layer_norm_index += 1
    delegate._post_attention = output_proof
    return output, output_proof, {
        "variance_fresh_count": square_count,
        "sqrt_flat_indices": list(sqrt_flat),
        "reciprocal_flat_indices": list(reciprocal_flat),
        "final_product_fresh_count": product_count,
        "output_generator_count": int(output.num_error_terms),
        "sqrt_interval_lower_min": float(semantic_low.min()),
        "sqrt_interval_upper_max": float(generic_high.max()),
        "reciprocal_semantic_lower_min": float(reciprocal_low.min()),
        "reciprocal_semantic_upper_max": float(reciprocal_high.max()),
        "semantic_range_certificate": semantic_range_certificate,
    }


def _majorant_with_semantic_lower(source, normalizer,
                                  semantic_lower_by_token):
    x = sound._absolute_hull(source).double()
    centered = x + x.mean(dim=-1, keepdim=True)
    variance_upper = centered.square().mean(dim=-1, keepdim=True) + EPSILON
    lower = torch.tensor([
        math.nextafter(float(value) + EPSILON, -math.inf)
        for value in semantic_lower_by_token], dtype=torch.float64,
        device=source.device).reshape(source.num_words, 1)
    reciprocal = lower.rsqrt().expand_as(centered)
    normalized = centered * reciprocal
    output = (normalized * normalizer.weight.abs().double()
              + normalizer.bias.abs().double())
    return torch.maximum(torch.maximum(centered, variance_upper),
                         torch.maximum(reciprocal, output))


def execute_experimental_layernorm(*, residual, proof, normalizer, delegate,
                                   diagnostics, milestones=None) -> dict:
    milestones = {} if milestones is None else milestones
    milestones.update({
        "psd_variance_certificate_authenticated": False,
        "psd_variance_lower_consumed": False,
        "sqrt_semantic_lower_constructed": False,
        "reciprocal_semantic_range_consumed": False,
        "full_psd_layernorm_completed": False,
    })
    if diagnostics.get("label") != TARGET_LABEL:
        raise RuntimeError("PSD experiment may only handle Block-2 post-attention")
    d = residual.word_embedding_size
    average = torch.ones((d, d), dtype=residual.zonotope_w.dtype,
                         device=residual.device) / d
    centered = residual.add(residual.matmul(average).multiply(-1.0))
    variance = centered.square_and_sum_and_repeat().multiply(1.0 / d)
    generic_low, _ = variance.concretize()
    certificate = build_certificate(residual, proof, generic_low)
    checked = verify_certificate(residual, proof, generic_low, certificate)
    milestones["psd_variance_certificate_authenticated"] = True
    # Use only the independently returned bounds, never optimizer output.
    lower_by_token = checked["semantic_variance_lower_by_token"]
    output, output_proof, transition = _native_layernorm_with_semantic_lower(
        residual, proof, normalizer, delegate, lower_by_token, certificate,
        milestones)
    operations = 16 * 128 * (residual.num_error_terms + 1) ** 2 + 4096
    reserve = sound._reserve_from_majorant(
        _majorant_with_semantic_lower(residual, normalizer, lower_by_token),
        operations)
    milestones["full_psd_layernorm_completed"] = True
    certificate.update({
        "minimum_psd_lower": checked["minimum_psd_lower"],
        "semantic_variance_lower_by_token": lower_by_token,
        "native_transition": transition,
        "native_variance_affine_coefficients_unchanged": True,
        "native_result_unchanged": False,
        "experimental_trace_parametric_constraint": True,
        "generic_semantic_remainder_used": False,
        "reserve_max": float(reserve.max()),
        "milestones": dict(milestones),
    })
    return {"output": output, "proof": output_proof,
            "reserve": reserve, "certificate": certificate}
