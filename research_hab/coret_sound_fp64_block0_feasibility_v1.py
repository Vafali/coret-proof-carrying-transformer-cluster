#!/usr/bin/env python3
"""Bounded FP64 feasibility experiment for the frozen production Block 0.

This module is deliberately not a production verifier.  It exercises the
frozen relational transformer in binary64 and supplies the measurement and
MPFR-oracle substrate used by the one-block feasibility gate.
"""
from __future__ import annotations

import io
import hashlib
import json
import math
import copy
import subprocess
import sys
import tarfile
import tempfile
import time
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
import gmpy2

import coret_native_semantics_production_graph_v1 as production
import coret_production_prefix_trace_v1 as prefix
import coret_structural_support_precise_dot_v1 as structural


REPO = Path(__file__).resolve().parents[1]
PINNED_REVISION = prefix.PINNED_REVISION
MAXIMUM_GENERATORS = 14000
FP64_U = 2.0 ** -53
MPFR_PRECISION = 256


@contextmanager
def pinned_zonotope():
    packaging = Path("/home/vafali_ubuntu/anaconda3/lib/python3.12/site-packages")
    if packaging.exists() and str(packaging) not in sys.path:
        sys.path.append(str(packaging))
    repository = REPO / "research_hab/public_benchmarks/DeepT"
    code = tempfile.TemporaryDirectory(prefix="coret_fp64_native_")
    packed = subprocess.check_output([
        "git", "-C", str(repository), "archive", "--format=tar",
        PINNED_REVISION,
        "Robustness-Verification-for-Transformers/Verifiers",
    ])
    with tarfile.open(fileobj=io.BytesIO(packed), mode="r:") as archive:
        archive.extractall(code.name, filter="data")
    root = (Path(code.name)
            / "Robustness-Verification-for-Transformers/Verifiers")
    # The pinned equality-substitution routine contains two implementation
    # guards requiring ``torch.float`` even though every operation underneath
    # is dtype-generic.  The feasibility implementation removes only those
    # guards in its private extracted copy; the algorithm and branch choices
    # are unchanged and the pinned checkout remains byte-identical.
    zonotope_path = root / "Zonotope.py"
    source = zonotope_path.read_text()
    replacements = {
        'assert constant_terms.dtype == torch.float, "Constant term should be float"':
            ('assert constant_terms.dtype in (torch.float, torch.float64), '
             '"Constant term should be floating point"'),
        'assert linear_terms.dtype == torch.float, "Linear term should be float"':
            ('assert linear_terms.dtype in (torch.float, torch.float64), '
             '"Linear term should be floating point"'),
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError("pinned FP32-only equality guard differs")
        source = source.replace(old, new)
    zonotope_path.write_text(source)
    previous = {name: module for name, module in sys.modules.items()
                if name == "Verifiers" or name.startswith("Verifiers.")}
    for name in previous:
        del sys.modules[name]
    package = types.ModuleType("Verifiers")
    package.__path__ = [str(root)]
    sys.modules["Verifiers"] = package
    from Verifiers.Zonotope import Zonotope
    try:
        yield Zonotope
    finally:
        for name in tuple(sys.modules):
            if name == "Verifiers" or name.startswith("Verifiers."):
                del sys.modules[name]
        sys.modules.update(previous)
        code.cleanup()


def _parameter(checkpoint, name: str, device, dtype=torch.float64):
    return SimpleNamespace(
        weight=checkpoint[f"{name}.weight"].to(device=device,
                                                dtype=dtype),
        bias=checkpoint[f"{name}.bias"].to(device=device,
                                            dtype=dtype))


def _args(device):
    return SimpleNamespace(
        perturbed_words=1, attack_type="lp", device=device,
        cpu=device.type == "cpu", all_words=False,
        num_input_error_terms=128, use_dot_product_variant3=False,
        use_other_dot_product_ordering=False,
        concretize_special_norm_error_together=False,
        batch_softmax_computation=True, keep_intermediate_zonotopes=False)


def _metrics(label, z, elapsed):
    low, high = z.concretize()
    radius = 0.5 * (high - low)
    allocated = (torch.cuda.memory_allocated(z.device)
                 if z.device.type == "cuda" else 0)
    reserved = (torch.cuda.memory_reserved(z.device)
                if z.device.type == "cuda" else 0)
    return {
        "label": label,
        "shape": list(z.zonotope_w.shape),
        "generator_count": int(z.num_error_terms),
        "support_max": float(radius.max()),
        "lower_min": float(low.min()),
        "upper_max": float(high.max()),
        "seconds": float(elapsed),
        "allocated_bytes": int(allocated),
        "reserved_bytes": int(reserved),
    }


def _make_like(z, weights, low=None, high=None):
    return z.__class__(
        args=z.args, p=z.p, eps=z.eps,
        perturbed_word_index=z.perturbed_word_index,
        zonotope_w=weights, error_term_range_low=low,
        error_term_range_high=high, clone=False)


def _ranges(z):
    if z.error_term_range_low is None:
        return (torch.full((z.num_error_terms,), -1.0, dtype=torch.float64,
                           device=z.device),
                torch.ones(z.num_error_terms, dtype=torch.float64,
                           device=z.device))
    return (z.error_term_range_low.double(),
            z.error_term_range_high.double())


def _support_radius(z):
    low, high = _ranges(z)
    magnitude = torch.maximum(low.abs(), high.abs())
    view = (1, -1) + (1,) * (z.zonotope_w.ndim - 2)
    if z.zonotope_w.ndim == 3:
        weighted = z.zonotope_w[1:].abs() * magnitude.reshape(-1, 1, 1)
        return weighted.sum(dim=0)
    weighted = z.zonotope_w[:, 1:].abs() * magnitude.reshape(view)
    return weighted.sum(dim=1)


def _absolute_hull(z):
    if z.zonotope_w.ndim == 3:
        return z.zonotope_w[0].abs() + _support_radius(z)
    return z.zonotope_w[:, 0].abs() + _support_radius(z)


def _gamma(operations: int) -> float:
    product = float(operations) * FP64_U
    if not 0 <= product < 0.25:
        raise RuntimeError(f"FP64 gamma precondition failed: {operations}")
    return product / (1.0 - product)


def _outward_positive(value):
    value = torch.clamp(value.double(), min=0.0)
    return torch.nextafter(value, torch.full_like(value, math.inf))


def _numerical_reserve(output, inputs, operations: int, condition=1.0):
    """Analytic gamma reserve collapsed only over machine-error sources.

    ``operations`` is a per-output-coordinate upper bound for the concrete
    arithmetic DAG.  The scale is a concrete absolute hull, never a sampled
    discrepancy.  Numerical uncertainty is subsequently represented by new
    coordinate-box generators and therefore becomes part of the next exact
    relational transformer input.
    """
    scale = _absolute_hull(output).double()
    for source in inputs:
        candidate = _absolute_hull(source).double()
        if candidate.shape == scale.shape:
            scale = torch.maximum(scale, candidate)
        else:
            scale = torch.maximum(scale, candidate.abs().max())
    scale = torch.maximum(scale, torch.ones_like(scale))
    # Two gamma passes cover the arithmetic DAG and evaluation of its absolute
    # majorant; a two-ULP transcendental allowance covers correctly/faithfully
    # rounded sqrt/exp/log calls used by the native relaxations.
    reserve = scale * (2.0 * _gamma(operations) * float(condition)
                       + 2.0 * FP64_U)
    return _outward_positive(reserve)


def _reserve_from_majorant(majorant, operations: int):
    majorant = torch.maximum(majorant.double(), torch.ones_like(majorant))
    return _outward_positive(
        majorant * (2.0 * _gamma(operations) + 2.0 * FP64_U))


def _bilinear_majorant(left, right):
    a, b = _absolute_hull(left).double(), _absolute_hull(right).double()
    if a.ndim != 3 or b.ndim != 3:
        raise RuntimeError("precise-dot majorant expects head-batched states")
    return torch.bmm(a, b.transpose(1, 2))


def _layernorm_majorant(source, normalizer):
    """Absolute majorant for every exact native LayerNorm intermediate."""
    x = _absolute_hull(source).double()
    centered = x + x.mean(dim=-1, keepdim=True)
    variance_upper = centered.square().mean(dim=-1, keepdim=True) + 1e-12
    # The exact variance lower domain is obtained from the native relational
    # input, not from the machine LayerNorm output.
    width = source.word_embedding_size
    average = torch.ones((width, width), dtype=torch.float64,
                         device=source.device) / width
    centered_state = source.add(source.matmul(average).multiply(-1.0))
    variance_state = centered_state.square_and_sum_and_repeat().multiply(
        1.0 / width).add(1e-12)
    variance_low, _ = variance_state.concretize()
    if bool((variance_low <= 0).any()):
        raise RuntimeError("sound FP64 LayerNorm majorant has nonpositive domain")
    reciprocal = variance_low.double().rsqrt()
    normalized = centered * reciprocal
    output = (normalized * normalizer.weight.abs().double()
              + normalizer.bias.abs().double())
    return torch.maximum(torch.maximum(centered, variance_upper),
                         torch.maximum(reciprocal, output))


def _prepare_layernorm_separator(source, proof, generic_low, label):
    if not bool((generic_low <= 0).any()):
        return None
    if str(REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO / "scripts"))
    import sound_fp64_layernorm_separator_v1 as separator
    prepared = separator.prepare(source, proof, generic_low, label)
    if prepared is not None and not prepared["admissible"]:
        import sound_fp64_layernorm_epsilon_floor_v1 as floor
        return floor.prepare(source, proof, generic_low, label, prepared)
    return prepared


def _layernorm_sound_raw(dispatch, source, proof, normalizer, label,
                         generic_low=None, prepared=None):
    """Keep native dispatch/reserves unchanged unless its variance domain fails."""
    if generic_low is None:
        d = source.word_embedding_size
        average = torch.ones((d, d), dtype=source.zonotope_w.dtype, device=source.device)/d
        centered = source.add(source.matmul(average).multiply(-1.0))
        generic_low, _ = centered.square_and_sum_and_repeat().multiply(1.0/d).concretize()
    if prepared is None:
        prepared = _prepare_layernorm_separator(source, proof, generic_low, label)
    if prepared is None:
        raw = dispatch.layer_norm(source, normalizer, "standard")
        return raw, structural.get_support(raw), None
    import sound_fp64_layernorm_separator_v1 as separator
    import sound_fp64_layernorm_epsilon_floor_v1 as floor
    if prepared.get("payload", {}).get("schema") == floor.SCHEMA:
        return floor.execute_prepared(dispatch, source, proof, normalizer, prepared)
    return separator.execute_prepared(dispatch, source, proof, normalizer, prepared)


def _checked_layernorm_low(generic_low, prepared):
    if prepared is None or not prepared.get("admissible"):
        return generic_low
    if prepared["semantic_lower_by_token"] != prepared["payload"]["semantic_lower_by_token"]:
        raise RuntimeError("separating LayerNorm prepared lower metadata differs")
    result = generic_low.clone()
    for token in prepared["payload"]["failed_tokens"]:
        result[token] = prepared["semantic_lower_by_token"][token]
    return result


def _softmax_majorant(scores):
    low, high = scores.concretize()
    maximum_difference = high.amax(dim=-1, keepdim=True) - low.amin(
        dim=-1, keepdim=True)
    exp_upper = torch.exp(maximum_difference.double())
    if not bool(torch.isfinite(exp_upper).all()):
        raise RuntimeError("softmax FP64 majorant overflow")
    # There are T exponentials in a denominator and the denominator is at
    # least one because the pivot difference is exactly zero.
    return (exp_upper * scores.zonotope_w.shape[-1]).expand_as(low)


def _append_numerical_box(z, proof, radius, label):
    """Embed independent coordinate roundoff directly in the zonotope."""
    radius = radius.to(device=z.device, dtype=torch.float64)
    old = z.zonotope_w
    if old.ndim == 3:
        tokens, width = old.shape[1:]
        if tuple(radius.shape) != (tokens, width):
            raise RuntimeError(f"{label}: numerical radius shape mismatch")
        count = tokens * width
        fresh = torch.zeros(count, tokens, width, dtype=old.dtype,
                            device=old.device)
        mask = torch.ones(tokens, width, dtype=torch.bool, device=old.device)
        fresh[torch.arange(count, device=old.device), mask] = radius[mask]
        masks = tuple(structural.local_mask(token)
                      for token in range(tokens) for _ in range(width))
    elif old.ndim == 4:
        heads, _rows, queries, columns = old.shape
        if tuple(radius.shape) != (heads, queries, columns):
            raise RuntimeError(f"{label}: numerical radius shape mismatch")
        count = heads * queries * columns
        fresh = torch.zeros(heads, count, queries, columns, dtype=old.dtype,
                            device=old.device)
        for head in range(heads):
            base = head * queries * columns
            indices = torch.arange(base, base + queries * columns,
                                   device=old.device)
            mask = torch.ones(queries, columns, dtype=torch.bool,
                              device=old.device)
            fresh[head, indices, mask] = radius[head, mask]
        masks = tuple(structural.local_mask(query)
                      for _head in range(heads)
                      for query in range(queries)
                      for _column in range(columns))
    else:
        raise RuntimeError("numerical box supports only native 3D/4D states")
    weights = torch.cat([old, fresh], dim=1 if old.ndim == 4 else 0)
    low, high = _ranges(z)
    low = torch.cat([low, -torch.ones(count, dtype=torch.float64,
                                      device=z.device)])
    high = torch.cat([high, torch.ones(count, dtype=torch.float64,
                                       device=z.device)])
    result = _make_like(z, weights, low, high)
    ids = tuple(f"fp64_numerical::{label}::{index:06d}"
                for index in range(count))
    reasons = tuple("fp64_roundoff_coordinate_box" for _ in range(count))
    output_proof = structural.SupportProof(
        proof.masks + masks, proof.ids + ids, proof.reasons + reasons,
        proof.num_tokens)
    structural.attach_support(result, output_proof)
    structural.validate_support(
        result, output_proof, token_axis=-2 if old.ndim == 4 else 1)
    return result, output_proof, float(radius.max()), count


def _align_states(left, left_proof, right, right_proof, label):
    """Align branch-local numerical IDs while preserving every semantic ID."""
    if left.zonotope_w.ndim != right.zonotope_w.ndim:
        raise RuntimeError(f"{label}: rank mismatch")
    union = list(left_proof.ids)
    union.extend(identifier for identifier in right_proof.ids
                 if identifier not in set(union))
    if len(union) != len(set(union)):
        raise RuntimeError(f"{label}: duplicate generator ID")

    def expand(z, proof):
        axis = 1 if z.zonotope_w.ndim == 4 else 0
        shape = list(z.zonotope_w.shape)
        shape[axis] = 1 + len(union)
        weights = torch.zeros(shape, dtype=z.zonotope_w.dtype, device=z.device)
        if axis == 0:
            weights[0] = z.zonotope_w[0]
        else:
            weights[:, 0] = z.zonotope_w[:, 0]
        source_low, source_high = _ranges(z)
        low = torch.full((len(union),), -1.0, dtype=torch.float64,
                         device=z.device)
        high = torch.ones(len(union), dtype=torch.float64, device=z.device)
        positions = {identifier: index for index, identifier in enumerate(union)}
        for old_index, identifier in enumerate(proof.ids):
            new_index = positions[identifier]
            if axis == 0:
                weights[1 + new_index] = z.zonotope_w[1 + old_index]
            else:
                weights[:, 1 + new_index] = z.zonotope_w[:, 1 + old_index]
            low[new_index] = source_low[old_index]
            high[new_index] = source_high[old_index]
        return _make_like(z, weights, low, high)

    mask_by_id = {}
    reason_by_id = {}
    for proof in (left_proof, right_proof):
        for identifier, mask, reason in zip(
                proof.ids, proof.masks, proof.reasons):
            if identifier in mask_by_id and mask_by_id[identifier] != mask:
                mask_by_id[identifier] |= mask
                reason_by_id[identifier] = "aligned_branch_support_union"
            else:
                mask_by_id[identifier] = mask
                reason_by_id[identifier] = reason
    proof = structural.SupportProof(
        tuple(mask_by_id[item] for item in union), tuple(union),
        tuple(reason_by_id[item] for item in union), left_proof.num_tokens)
    a, b = expand(left, left_proof), expand(right, right_proof)
    structural.attach_support(a, proof); structural.attach_support(b, proof)
    return a, b, proof


def _affine_reserve(source, parameter):
    products = torch.matmul(
        source.zonotope_w.abs().double(),
        parameter.weight.abs().double().t())
    low, high = _ranges(source)
    magnitude = torch.maximum(low.abs(), high.abs())
    if source.zonotope_w.ndim != 3:
        raise RuntimeError("bounded affine reserve expects native 3D state")
    concrete = products[0]
    if source.num_error_terms:
        concrete = concrete + (
            products[1:] * magnitude.reshape(-1, 1, 1)).sum(dim=0)
    concrete = concrete + parameter.bias.abs().double()
    return _outward_positive(concrete * (2.0 * _gamma(
        2 * source.word_embedding_size + 2) + 2.0 * FP64_U))


def _add_reserve(left, right):
    a, b = _align_for_numeric_only(left, right)
    magnitude = _absolute_hull(a) + _absolute_hull(b)
    return _outward_positive(magnitude * (2.0 * _gamma(1) + FP64_U))


def _align_for_numeric_only(left, right):
    """Zero-pad already index-aligned native states for an error majorant."""
    if left.zonotope_w.ndim != 3 or right.zonotope_w.ndim != 3:
        raise RuntimeError("numeric residual majorant expects 3D states")
    rows = max(left.zonotope_w.shape[0], right.zonotope_w.shape[0])

    def pad(z):
        if z.zonotope_w.shape[0] == rows:
            return z
        weights = torch.cat([
            z.zonotope_w,
            torch.zeros(rows - z.zonotope_w.shape[0], *z.zonotope_w.shape[1:],
                        dtype=z.zonotope_w.dtype, device=z.device)], dim=0)
        low, high = _ranges(z)
        count = rows - z.zonotope_w.shape[0]
        return _make_like(
            z, weights, torch.cat([low, -torch.ones(
                count, dtype=torch.float64, device=z.device)]),
            torch.cat([high, torch.ones(
                count, dtype=torch.float64, device=z.device)]))
    return pad(left), pad(right)


def _inject(output, proof, inputs, label, operations, measurements,
            condition=1.0, reserve=None):
    reserve = (_numerical_reserve(
        output, inputs, operations, condition=condition)
        if reserve is None else reserve)
    result, result_proof, maximum, count = _append_numerical_box(
        output, proof, reserve, label)
    measurements.append({
        "label": label, "added_generators": count,
        "maximum_local_widening": maximum,
        "operations_bound": int(operations),
        "condition_factor": float(condition),
    })
    return result, result_proof


def _dense_sound(source, proof, parameter, label, measurements):
    output = source.dense(parameter)
    return _inject(
        output, proof, [source], label,
        2 * source.word_embedding_size + 2, measurements,
        reserve=_affine_reserve(source, parameter))


def _mp_context(rounding):
    return gmpy2.local_context(
        gmpy2.context(), precision=MPFR_PRECISION, round=rounding)


def _mp(value):
    with _mp_context(gmpy2.RoundToNearest):
        return gmpy2.mpfr(float(value))


def _directed_double(value, upward):
    rounded = float(value)
    exact_rounded = _mp(rounded)
    if (upward and exact_rounded < value) or (not upward and exact_rounded > value):
        rounded = math.nextafter(
            rounded, math.inf if upward else -math.inf)
    return rounded


def _oracle_record(label, exact):
    lower = _directed_double(exact, False)
    upper = _directed_double(exact, True)
    lower_shrunk = math.nextafter(lower, math.inf)
    upper_shrunk = math.nextafter(upper, -math.inf)
    valid = _mp(lower) <= exact <= _mp(upper)
    shrink_rejects = not (_mp(lower_shrunk) <= exact <= _mp(upper_shrunk))
    if not valid or not shrink_rejects:
        raise RuntimeError(f"{label}: directed MPFR oracle gate failed")
    return {
        "label": label, "lower_hex": lower.hex(), "upper_hex": upper.hex(),
        "one_ulp_inward_rejected": True,
    }


def _oracle_containment(label, exact, machine, reserve):
    record = _oracle_record(label, exact)
    with _mp_context(gmpy2.RoundUp):
        discrepancy = abs(exact - _mp(machine))
        allowed = _mp(reserve)
    if discrepancy > allowed:
        raise RuntimeError(
            f"{label}: FP64 state reserve misses MPFR discrepancy")
    record.update({
        "machine_hex": float(machine).hex(),
        "machine_error": str(discrepancy),
        "state_reserve_hex": float(reserve).hex(),
        "state_reserve_contains_machine_error": True,
    })
    return record


def _mpfr_spots(checkpoint):
    """Directed 256-bit checks on frozen operands/formulas."""
    records = []
    # Affine accumulation: one actual Q coefficient from the frozen source.
    _w, _p, _t, pre, _ln, _q, _k = prefix._fixture_parameters()
    row = pre[1].double()
    weight = checkpoint[
        "bert.encoder.layer.0.attention.self.query.weight"][0].double()
    bias = checkpoint[
        "bert.encoder.layer.0.attention.self.query.bias"][0].double()
    with _mp_context(gmpy2.RoundToNearest):
        exact = _mp(bias)
        for left, right in zip(row.tolist(), weight.tolist()):
            exact += _mp(left) * _mp(right)
    records.append(_oracle_record("affine_accumulation", exact))

    with _mp_context(gmpy2.RoundToNearest):
        low, high = _mp("0.8710253840723361"), _mp("0.8729633325601869")
        root_low, root_high = gmpy2.sqrt(low), gmpy2.sqrt(high)
        slope = (root_high - root_low) / (high - low)
        critical = ((high - low) / (2 * (root_high - root_low))) ** 2
        sqrt_relax = slope * critical - gmpy2.sqrt(critical)
        reciprocal = 1 / root_low
    records.append(_oracle_record("sqrt_relaxation", sqrt_relax))
    records.append(_oracle_record("reciprocal_relaxation", reciprocal))

    # Exact formulas representative of QK, softmax exp, A.V and the final
    # LayerNorm product; operands are frozen binary64 values.
    values = [float(row[index]) for index in range(8)]
    others = [float(weight[index]) for index in range(8)]
    with _mp_context(gmpy2.RoundToNearest):
        dot = sum((_mp(a) * _mp(b) for a, b in zip(values, others)), _mp(0))
        exp_value = gmpy2.exp(_mp(-0.125))
        av_value = sum((_mp(a) * _mp(b) for a, b in zip(
            values[:4], others[:4])), _mp(0))
        product = _mp(values[0]) * reciprocal
        variance = sum(((_mp(item) - sum(map(_mp, values), _mp(0)) / 8) ** 2
                        for item in values), _mp(0)) / 8
    records.extend([
        _oracle_record("qk_coefficient_radius", abs(dot)),
        _oracle_record("softmax_exp_relaxation", exp_value),
        _oracle_record("attention_value", av_value),
        _oracle_record("layernorm_variance", variance),
        _oracle_record("layernorm_final_product", product),
    ])
    return records


def _runtime_mpfr_spots(*, hidden, query_parameter, q, k, raw_qk,
                        scores, raw_probability, value, raw_context,
                        ffn_residual, centered, variance, reserves):
    records = []
    # Actual Q affine center coefficient (token 0, output feature 0).
    source = hidden.zonotope_w[0, 0].detach().cpu().tolist()
    weight = query_parameter.weight[0].detach().cpu().tolist()
    with _mp_context(gmpy2.RoundToNearest):
        exact = _mp(query_parameter.bias[0])
        for left, right in zip(source, weight):
            exact += _mp(left) * _mp(right)
    records.append(_oracle_containment(
        "actual_q_affine_coefficient", exact,
        q.zonotope_w[0, 0, 0, 0], reserves["q_affine"]))

    # Actual retained QK row 0: q0.k1 + q1.k0.
    with _mp_context(gmpy2.RoundToNearest):
        exact = _mp(0)
        for feature in range(q.zonotope_w.shape[-1]):
            exact += (_mp(q.zonotope_w[0, 0, 0, feature])
                      * _mp(k.zonotope_w[0, 1, 0, feature]))
            exact += (_mp(q.zonotope_w[0, 1, 0, feature])
                      * _mp(k.zonotope_w[0, 0, 0, feature]))
    records.append(_oracle_containment(
        "actual_qk_retained_coefficient", exact,
        raw_qk.zonotope_w[0, 1, 0, 0], reserves["qk"]))

    # Actual score scaling and exp evaluation used by softmax.
    low, _high = scores.concretize()
    score = low[0, 0, 0]
    with _mp_context(gmpy2.RoundToNearest):
        exact_exp = gmpy2.exp(_mp(score))
    records.append(_oracle_containment(
        "actual_softmax_exp", exact_exp, torch.exp(score),
        reserves["softmax"]))

    # Actual retained A.V row: p0.V1 + p1.V0 over all four keys.
    transposed = value.t()
    with _mp_context(gmpy2.RoundToNearest):
        exact = _mp(0)
        for key_index in range(value.num_words):
            exact += (_mp(raw_probability.zonotope_w[0, 0, 0, key_index])
                      * _mp(transposed.zonotope_w[0, 1, 0, key_index]))
            exact += (_mp(raw_probability.zonotope_w[0, 1, 0, key_index])
                      * _mp(transposed.zonotope_w[0, 0, 0, key_index]))
    records.append(_oracle_containment(
        "actual_av_retained_coefficient", exact,
        raw_context.zonotope_w[0, 1, 0, 0],
        reserves["attention_value"]))

    # Actual nominal variance sum in the final LayerNorm.  The native center
    # also contains diagonal-generator terms, checked separately by the QK/A.V
    # exact formulas; this spot validates its 128-term square/reduction path.
    values = centered.zonotope_w[0, 0].detach().cpu().tolist()
    with _mp_context(gmpy2.RoundToNearest):
        exact_variance_nominal = sum(
            (_mp(item) * _mp(item) for item in values), _mp(0)) / len(values)
    machine_variance_nominal = centered.zonotope_w[0, 0].square().sum() / len(values)
    records.append(_oracle_containment(
        "actual_layernorm_variance_nominal", exact_variance_nominal,
        machine_variance_nominal, reserves["output_layernorm"]))

    variance_low, variance_high = variance.concretize()
    low_value = variance_low[0, 0]
    high_value = variance_high[0, 0]
    with _mp_context(gmpy2.RoundToNearest):
        lo, hi = _mp(low_value), _mp(high_value)
        root_lo, root_hi = gmpy2.sqrt(lo), gmpy2.sqrt(hi)
        slope = (root_hi - root_lo) / (hi - lo)
        critical = ((hi - lo) / (2 * (root_hi - root_lo))) ** 2
        exact_sqrt_residual = slope * critical - gmpy2.sqrt(critical)
        exact_reciprocal = 1 / root_lo
        exact_product = _mp(centered.zonotope_w[0, 0, 0]) * exact_reciprocal
    machine_slope = ((torch.sqrt(high_value) - torch.sqrt(low_value))
                     / (high_value - low_value))
    machine_critical = ((high_value - low_value) /
                        (2 * (torch.sqrt(high_value)
                              - torch.sqrt(low_value)))).square()
    machine_sqrt_residual = (machine_slope * machine_critical
                             - torch.sqrt(machine_critical))
    records.extend([
        _oracle_containment(
            "actual_sqrt_relaxation", exact_sqrt_residual,
            machine_sqrt_residual, reserves["output_layernorm"]),
        _oracle_containment(
            "actual_reciprocal", exact_reciprocal,
            torch.reciprocal(torch.sqrt(low_value)),
            reserves["output_layernorm"]),
        _oracle_containment(
            "actual_layernorm_final_product", exact_product,
            centered.zonotope_w[0, 0, 0]
            * torch.reciprocal(torch.sqrt(low_value)),
            reserves["output_layernorm"]),
    ])
    return records


def _timed(label, call, rows):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    value = call()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    rows.append(_metrics(label, value, time.perf_counter() - started))
    return value


def run_plain_fp64(device=None, dtype=torch.float64):
    """Execute the complete native relational Block 0 without new widening."""
    device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    prior_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with pinned_zonotope() as Zonotope:
            checkpoint = prefix._load_checkpoint()
            _word, _position, _token_type, pre, embedding_ln, _q, _k = (
                prefix._fixture_parameters())
            pre = pre.to(device=device, dtype=dtype)
            embedding_ln = SimpleNamespace(
                weight=embedding_ln.weight.to(device=device,
                                               dtype=dtype),
                bias=embedding_ln.bias.to(device=device,
                                           dtype=dtype))
            args = _args(device)
            z = Zonotope(args=args, p=100, eps=prefix.FIXTURE_RHO,
                         perturbed_word_index=prefix.FIXTURE_PERTURBED_TOKEN,
                         value=pre)
            proof = structural.proof_from_masks(
                [structural.local_mask(prefix.FIXTURE_PERTURBED_TOKEN)] * 128,
                len(prefix.FIXTURE_TOKEN_IDS), "input_source")
            structural.attach_support(z, proof)
            delegate = structural.StructuralNativeSemanticOperators()
            dispatch = production.NativeProductionDispatch(delegate=delegate)
            rows = [_metrics("source", z, 0.0)]
            z = _timed("embedding_layernorm", lambda: dispatch.layer_norm(
                z, embedding_ln, "standard"), rows)
            z = production._recenter_native_ranges(z)
            z = _timed("embedding_reduction", lambda: dispatch.reduce(
                z, MAXIMUM_GENERATORS), rows)
            residual = z

            base = "bert.encoder.layer.0"
            query = _parameter(
                checkpoint, base + ".attention.self.query", device, dtype)
            key = _parameter(
                checkpoint, base + ".attention.self.key", device, dtype)
            value = _parameter(
                checkpoint, base + ".attention.self.value", device, dtype)
            attention_output = _parameter(
                checkpoint, base + ".attention.output.dense", device, dtype)
            attention_ln = _parameter(
                checkpoint, base + ".attention.output.LayerNorm", device, dtype)
            ffn_first = _parameter(
                checkpoint, base + ".intermediate.dense", device, dtype)
            ffn_second = _parameter(
                checkpoint, base + ".output.dense", device, dtype)
            output_ln = _parameter(
                checkpoint, base + ".output.LayerNorm", device, dtype)

            q = z.dense(query).add_attention_heads_dim(4)
            k = z.dense(key).add_attention_heads_dim(4)
            qk = _timed("qk", lambda: dispatch.qk(q, k), rows)
            scores = qk.multiply(1.0 / math.sqrt(32))
            structural.attach_support(scores, structural.get_support(qk))
            probability = _timed(
                "softmax", lambda: dispatch.softmax(
                    scores, no_constraints=False), rows)
            v = z.dense(value).add_attention_heads_dim(4)
            context = _timed(
                "attention_value",
                lambda: dispatch.attention_value(probability, v), rows)
            context = context.remove_attention_heads_dim()
            attention = context.dense(attention_output)
            aligned = residual.expand_error_terms_to_match_zonotope(attention)
            post_attention_input = attention.add(aligned)
            post_attention = _timed(
                "post_attention_layernorm",
                lambda: dispatch.layer_norm(
                    post_attention_input, attention_ln, "standard"), rows)
            intermediate_affine = post_attention.dense(ffn_first)
            intermediate = _timed(
                "relu", lambda: dispatch.relu(intermediate_affine), rows)
            ffn = intermediate.dense(ffn_second)
            rows.append(_metrics("ffn_output", ffn, 0.0))
            aligned = post_attention.expand_error_terms_to_match_zonotope(ffn)
            output_input = ffn.add(aligned)

            width = output_input.word_embedding_size
            average = torch.ones((width, width), device=device,
                                 dtype=dtype) / width
            centered = output_input.add(
                output_input.matmul(average).multiply(-1.0))
            variance = centered.square_and_sum_and_repeat().multiply(1.0 / width)
            variance_low, variance_high = variance.concretize()
            variance_min = float(variance_low.min())
            variance_upper_min = float(variance_high.min())

            output = _timed(
                "output_layernorm",
                lambda: dispatch.layer_norm(output_input, output_ln, "standard"),
                rows)
            rows.append(_metrics("block0_output", output, 0.0))
            return {
                "rows": rows,
                "second_layernorm_variance_lower": variance_min,
                "second_layernorm_variance_upper_min": variance_upper_min,
                "dispatch_counts": dict(dispatch.counts),
                "dtype": str(output.zonotope_w.dtype),
            }
    finally:
        torch.set_default_dtype(prior_dtype)


def _tensor_sha(value):
    raw = value.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _coordinate_count(z):
    if z.zonotope_w.ndim == 3:
        return z.zonotope_w.shape[1] * z.zonotope_w.shape[2]
    if z.zonotope_w.ndim == 4:
        return (z.zonotope_w.shape[0] * z.zonotope_w.shape[2]
                * z.zonotope_w.shape[3])
    raise RuntimeError("sound reduction supports only native 3D/4D states")


def _generator_rows(z):
    return z.zonotope_w[1:] if z.zonotope_w.ndim == 3 else z.zonotope_w[:, 1:]


def _center_row(z):
    return z.zonotope_w[0] if z.zonotope_w.ndim == 3 else z.zonotope_w[:, 0]


def _protected_indices(z, proof):
    low, high = _ranges(z)
    protected = []
    for index, (identifier, reason) in enumerate(zip(proof.ids, proof.reasons)):
        source = (identifier.startswith("input_source_")
                  or "original_perturbed" in reason
                  or "source" in reason and index < 128)
        constrained = (float(low[index]) != -1.0 or float(high[index]) != 1.0)
        if source or constrained:
            protected.append(index)
    return tuple(protected)


def _ranking(z):
    rows = _generator_rows(z).abs().double()
    low, high = _ranges(z)
    scale = torch.maximum(low.abs(), high.abs())
    if z.zonotope_w.ndim == 3:
        metric = (rows * scale.reshape(-1, 1, 1)).sum(dim=(1, 2))
    else:
        metric = (rows * scale.reshape(1, -1, 1, 1)).sum(dim=(0, 2, 3))
    return metric.detach().cpu().tolist()


def _select_reduction_indices(z, proof, cap):
    boxes = _coordinate_count(z)
    keep_count = int(cap) - boxes
    if keep_count < 0 or keep_count > z.num_error_terms:
        raise RuntimeError("invalid sound reduction cap")
    protected = set(_protected_indices(z, proof))
    if len(protected) > keep_count:
        raise RuntimeError(
            f"protected generator count {len(protected)} exceeds capacity {keep_count}")
    metric = _ranking(z)
    candidates = [index for index in range(z.num_error_terms)
                  if index not in protected]
    candidates.sort(key=lambda index: (-metric[index], index))
    retained = sorted((*protected, *candidates[:keep_count-len(protected)]))
    retained_set = set(retained)
    dropped = [index for index in range(z.num_error_terms)
               if index not in retained_set]
    return tuple(retained), tuple(dropped), tuple(sorted(protected)), metric


def _sound_reduce_selected(z, proof, retained, dropped, label):
    retained = tuple(int(index) for index in retained)
    dropped = tuple(int(index) for index in dropped)
    if sorted((*retained, *dropped)) != list(range(z.num_error_terms)):
        raise RuntimeError(f"{label}: reduction partition is not exhaustive")
    low, high = _ranges(z)
    offset, scale = (low + high) / 2.0, (high - low) / 2.0
    rows = _generator_rows(z).double()
    device = z.device
    retained_tensor = torch.tensor(retained, dtype=torch.long, device=device)
    dropped_tensor = torch.tensor(dropped, dtype=torch.long, device=device)
    if z.zonotope_w.ndim == 3:
        kept_rows = rows.index_select(0, retained_tensor)
        removed = rows.index_select(0, dropped_tensor)
        dropped_offset = offset.index_select(0, dropped_tensor).reshape(-1, 1, 1)
        dropped_scale = scale.index_select(0, dropped_tensor).reshape(-1, 1, 1)
        shift_terms = removed * dropped_offset
        radius_terms = removed.abs() * dropped_scale
        center = _center_row(z).double() + shift_terms.sum(dim=0)
        absolute_shift = shift_terms.abs().sum(dim=0)
        radius = radius_terms.sum(dim=0)
        operations = max(1, 2 * len(dropped) + 1)
        center_error = (_center_row(z).abs().double() + absolute_shift) * _gamma(
            operations)
        radius = _outward_positive(
            radius * (1.0 + _gamma(max(1, len(dropped)))) + center_error)
        tokens, width = radius.shape
        box = torch.zeros(tokens * width, tokens, width, dtype=torch.float64,
                          device=device)
        mask = torch.ones(tokens, width, dtype=torch.bool, device=device)
        box[torch.arange(tokens * width, device=device), mask] = radius[mask]
        weights = torch.cat([center.unsqueeze(0), kept_rows, box], dim=0)
        masks = tuple(structural.local_mask(token)
                      for token in range(tokens) for _ in range(width))
    else:
        kept_rows = rows.index_select(1, retained_tensor)
        removed = rows.index_select(1, dropped_tensor)
        dropped_offset = offset.index_select(0, dropped_tensor).reshape(
            1, -1, 1, 1)
        dropped_scale = scale.index_select(0, dropped_tensor).reshape(
            1, -1, 1, 1)
        shift_terms = removed * dropped_offset
        radius_terms = removed.abs() * dropped_scale
        center = _center_row(z).double() + shift_terms.sum(dim=1)
        absolute_shift = shift_terms.abs().sum(dim=1)
        radius = radius_terms.sum(dim=1)
        operations = max(1, 2 * len(dropped) + 1)
        center_error = (_center_row(z).abs().double() + absolute_shift) * _gamma(
            operations)
        radius = _outward_positive(
            radius * (1.0 + _gamma(max(1, len(dropped)))) + center_error)
        heads, queries, columns = radius.shape
        count = heads * queries * columns
        box = torch.zeros(heads, count, queries, columns, dtype=torch.float64,
                          device=device)
        for head in range(heads):
            first = head * queries * columns
            indices = torch.arange(first, first + queries * columns,
                                   device=device)
            mask = torch.ones(queries, columns, dtype=torch.bool, device=device)
            box[head, indices, mask] = radius[head, mask]
        weights = torch.cat([center.unsqueeze(1), kept_rows, box], dim=1)
        masks = tuple(structural.local_mask(query)
                      for _head in range(heads)
                      for query in range(queries)
                      for _column in range(columns))
    retained_low = low.index_select(0, retained_tensor)
    retained_high = high.index_select(0, retained_tensor)
    box_count = _coordinate_count(z)
    output_low = torch.cat([retained_low, -torch.ones(
        box_count, dtype=torch.float64, device=device)])
    output_high = torch.cat([retained_high, torch.ones(
        box_count, dtype=torch.float64, device=device)])
    output = _make_like(z, weights, output_low, output_high)
    ids = tuple(proof.ids[index] for index in retained) + tuple(
        f"reduction_box::{label}::{index:06d}" for index in range(box_count))
    dropped_reasons = tuple(proof.reasons[index] for index in dropped)
    absorbs_numerical = any(
        reason == "fp64_roundoff_coordinate_box"
        or reason == "sound_fp64_coordinate_box_replacement_with_numerical"
        for reason in dropped_reasons)
    replacement_reason = (
        "sound_fp64_coordinate_box_replacement_with_numerical"
        if absorbs_numerical else "sound_fp64_coordinate_box_replacement")
    reasons = tuple(proof.reasons[index] for index in retained) + tuple(
        replacement_reason for _ in range(box_count))
    output_proof = structural.SupportProof(
        tuple(proof.masks[index] for index in retained) + masks,
        ids, reasons, proof.num_tokens)
    structural.attach_support(output, output_proof)
    structural.validate_support(
        output, output_proof, token_axis=-2 if z.zonotope_w.ndim == 4 else 1)
    witness = {
        "label": label,
        "input_generator_count": z.num_error_terms,
        "output_generator_count": output.num_error_terms,
        "retained_indices": list(retained),
        "dropped_indices": list(dropped),
        "retained_ids": [proof.ids[index] for index in retained],
        "dropped_ids": [proof.ids[index] for index in dropped],
        "replacement_ids": list(ids[len(retained):]),
        "replacement_count": box_count,
        "input_ranges_sha256": hashlib.sha256(
            low.detach().cpu().contiguous().numpy().tobytes()
            + high.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        "output_weights_sha256": _tensor_sha(output.zonotope_w),
        "replacement_radius_sha256": _tensor_sha(radius),
        "rounding_operations": operations,
        "provenance_partition": {
            "retained": [proof.reasons[index] for index in retained],
            "dropped": list(dropped_reasons),
            "replacement": replacement_reason,
            "absorbs_numerical": absorbs_numerical,
        },
    }
    return output, output_proof, witness


def sound_reduce(z, proof, cap, label):
    retained, dropped, protected, metric = _select_reduction_indices(
        z, proof, cap)
    output, output_proof, witness = _sound_reduce_selected(
        z, proof, retained, dropped, label)
    witness.update({
        "cap": int(cap), "protected_indices": list(protected),
        "ranking_sha256": hashlib.sha256(json.dumps(
            [float(value).hex() for value in metric], separators=(",", ":")
        ).encode()).hexdigest(),
        "selection_rule": (
            "protect_source_and_nonunit_ranges_then_descending_weighted_l1_"
            "with_original_index_tiebreak"),
    })
    return output, output_proof, witness


def sound_reduce_pair(left, right, proof, cap, label):
    """Reduce an aligned bilinear/residual pair with independent boxes."""
    if left.num_error_terms != right.num_error_terms:
        raise RuntimeError(f"{label}: pair must be generator aligned")
    left_boxes, right_boxes = _coordinate_count(left), _coordinate_count(right)
    keep_count = int(cap) - left_boxes - right_boxes
    if keep_count < 0:
        raise RuntimeError(f"{label}: pair box replacement exceeds cap")
    left_protected = set(_protected_indices(left, proof))
    right_protected = set(_protected_indices(right, proof))
    protected = left_protected | right_protected
    if len(protected) > keep_count:
        raise RuntimeError(f"{label}: protected pair generators exceed cap")
    left_metric, right_metric = _ranking(left), _ranking(right)
    metric = [a + b for a, b in zip(left_metric, right_metric)]
    candidates = [index for index in range(left.num_error_terms)
                  if index not in protected]
    candidates.sort(key=lambda index: (-metric[index], index))
    retained = sorted((*protected, *candidates[:keep_count-len(protected)]))
    retained_set = set(retained)
    dropped = [index for index in range(left.num_error_terms)
               if index not in retained_set]
    reduced_left, left_proof, left_witness = _sound_reduce_selected(
        left, proof, retained, dropped, label + "_left")
    reduced_right, right_proof, right_witness = _sound_reduce_selected(
        right, proof, retained, dropped, label + "_right")
    aligned_left, aligned_right, aligned_proof = _align_states(
        reduced_left, left_proof, reduced_right, right_proof,
        label + "_replacement_union")
    if aligned_left.num_error_terms != cap:
        raise RuntimeError(
            f"{label}: pair reduction produced {aligned_left.num_error_terms} != {cap}")
    left_check = _check_hull_containment(left, aligned_left, label + "_left")
    right_check = _check_hull_containment(
        right, aligned_right, label + "_right")
    ranking_hash = hashlib.sha256(json.dumps(
        [float(value).hex() for value in metric], separators=(",", ":")
    ).encode()).hexdigest()
    return aligned_left, aligned_right, aligned_proof, {
        "label": label, "cap": cap,
        "input_generator_count": left.num_error_terms,
        "retained": len(retained), "dropped": len(dropped),
        "left_boxes": left_boxes, "right_boxes": right_boxes,
        "output_generator_count": aligned_left.num_error_terms,
        "protected": len(protected),
        "protected_indices": sorted(protected),
        "retained_indices": list(retained), "dropped_indices": list(dropped),
        "retained_ids": [proof.ids[index] for index in retained],
        "dropped_ids": [proof.ids[index] for index in dropped],
        "ranking_sha256": ranking_hash,
        "selection_rule": (
            "pair_protect_source_and_nonunit_ranges_then_descending_combined_"
            "weighted_l1_with_original_index_tiebreak"),
        "left_witness": left_witness, "right_witness": right_witness,
        "left_containment": left_check, "right_containment": right_check,
    }


def _check_hull_containment(source, output, label):
    """Check the coordinate hull consequence of a relational reduction."""
    before_low, before_high = source.concretize()
    after_low, after_high = output.concretize()
    tolerance = torch.finfo(torch.float64).eps * torch.maximum(
        torch.ones_like(before_low), torch.maximum(
            before_low.abs(), before_high.abs())) * 8
    if bool((after_low > before_low + tolerance).any()
            or (after_high < before_high - tolerance).any()):
        raise AssertionError(f"{label}: reduction output misses input hull")
    return {
        "accepted": True,
        "support_inflation": float(
            (0.5 * (after_high-after_low)
             - 0.5 * (before_high-before_low)).max()),
        "range_inflation": float(torch.maximum(
            before_low-after_low, after_high-before_high).max()),
    }


def check_reduction_witness(source, source_proof, output, output_proof,
                            witness):
    retained, dropped, protected, metric = _select_reduction_indices(
        source, source_proof, int(witness["cap"]))
    if list(retained) != witness["retained_indices"]:
        raise AssertionError("reduction retained-index witness differs")
    if list(dropped) != witness["dropped_indices"]:
        raise AssertionError("reduction dropped-index witness differs")
    if list(protected) != witness["protected_indices"]:
        raise AssertionError("reduction protected-index witness differs")
    if [source_proof.ids[index] for index in retained] != witness[
            "retained_ids"]:
        raise AssertionError("reduction retained-ID witness differs")
    if [source_proof.ids[index] for index in dropped] != witness[
            "dropped_ids"]:
        raise AssertionError("reduction dropped-ID witness differs")
    low, high = _ranges(source)
    ranges_hash = hashlib.sha256(
        low.detach().cpu().contiguous().numpy().tobytes()
        + high.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    if ranges_hash != witness["input_ranges_sha256"]:
        raise AssertionError("reduction source-range witness differs")
    ranking_hash = hashlib.sha256(json.dumps(
        [float(value).hex() for value in metric], separators=(",", ":")
    ).encode()).hexdigest()
    if ranking_hash != witness["ranking_sha256"]:
        raise AssertionError("reduction ranking witness differs")
    replay, replay_proof, replay_witness = _sound_reduce_selected(
        source, source_proof, retained, dropped, witness["label"])
    if not torch.equal(replay.zonotope_w, output.zonotope_w):
        raise AssertionError("reduction numerical replay differs")
    if replay_proof != output_proof:
        raise AssertionError("reduction provenance replay differs")
    if list(replay_proof.ids[len(retained):]) != witness["replacement_ids"]:
        raise AssertionError("reduction replacement-ID witness differs")
    replay_low, replay_high = _ranges(replay)
    output_low, output_high = _ranges(output)
    if not (torch.equal(replay_low, output_low)
            and torch.equal(replay_high, output_high)):
        raise AssertionError("reduction output ranges differ")
    if replay_witness["replacement_radius_sha256"] != witness[
            "replacement_radius_sha256"]:
        raise AssertionError("reduction replacement radius differs")
    if _tensor_sha(output.zonotope_w) != witness["output_weights_sha256"]:
        raise AssertionError("reduction output hash differs")
    return _check_hull_containment(source, output, witness["label"])


def _reduction_mutations(source, source_proof, output, output_proof, witness):
    mutations = {}

    def rejected(name, changed_output=output, changed_proof=output_proof,
                 changed_witness=None):
        try:
            check_reduction_witness(
                source, source_proof, changed_output, changed_proof,
                witness if changed_witness is None else changed_witness)
        except (AssertionError, RuntimeError):
            mutations[name] = True
            return
        mutations[name] = False

    changed = copy.deepcopy(witness)
    changed["dropped_indices"] = changed["dropped_indices"][1:]
    rejected("dropped_coefficient_omitted", changed_witness=changed)

    changed_output = _make_like(
        output, output.zonotope_w.clone(), *_ranges(output))
    if changed_output.zonotope_w.ndim == 3:
        row = 1 + len(witness["retained_indices"])
        nonzero = changed_output.zonotope_w[row].nonzero()[0]
        value = changed_output.zonotope_w[row, nonzero[0], nonzero[1]]
        changed_output.zonotope_w[row, nonzero[0], nonzero[1]] = torch.nextafter(
            value, torch.full_like(value, -math.inf))
    else:
        row = 1 + len(witness["retained_indices"])
        nonzero = changed_output.zonotope_w[:, row].nonzero()[0]
        value = changed_output.zonotope_w[
            nonzero[0], row, nonzero[1], nonzero[2]]
        changed_output.zonotope_w[
            nonzero[0], row, nonzero[1], nonzero[2]] = torch.nextafter(
                value, torch.full_like(value, -math.inf))
    rejected("box_radius_narrowed_one_ulp", changed_output=changed_output)
    rejected("under_rounded_absolute_sum", changed_output=changed_output)

    changed = copy.deepcopy(witness)
    changed["retained_ids"][0] = "wrong_retained_id"
    rejected("wrong_retained_generator_id", changed_witness=changed)
    changed = copy.deepcopy(witness)
    changed["dropped_indices"][0], changed["dropped_indices"][1] = (
        changed["dropped_indices"][1], changed["dropped_indices"][0])
    rejected("wrong_dropped_generator_set", changed_witness=changed)

    ids = list(output_proof.ids); masks = list(output_proof.masks)
    reasons = list(output_proof.reasons)
    ids[-1], ids[-2] = ids[-2], ids[-1]
    reordered = structural.SupportProof(
        tuple(masks), tuple(ids), tuple(reasons), output_proof.num_tokens)
    rejected("generator_reorder_without_map", changed_proof=reordered)

    reasons[-1] = "native_nonlinear_fresh_substitution"
    substituted = structural.SupportProof(
        tuple(masks), output_proof.ids, tuple(reasons), output_proof.num_tokens)
    rejected("numerical_native_provenance_substitution",
             changed_proof=substituted)

    changed = copy.deepcopy(witness)
    changed["input_ranges_sha256"] = "0" * 64
    rejected("incorrect_source_range", changed_witness=changed)

    if output.zonotope_w.ndim == 3:
        shortened_weights = output.zonotope_w[:-1].clone()
    else:
        shortened_weights = output.zonotope_w[:, :-1].clone()
    low, high = _ranges(output)
    shortened = _make_like(output, shortened_weights, low[:-1], high[:-1])
    shortened_proof = structural.SupportProof(
        output_proof.masks[:-1], output_proof.ids[:-1],
        output_proof.reasons[:-1], output_proof.num_tokens)
    rejected("missing_reduction_box_generator", changed_output=shortened,
             changed_proof=shortened_proof)
    if not all(mutations.values()):
        raise RuntimeError(f"reduction mutations accepted: {mutations}")
    return mutations


def run_reduction_audit(device=None):
    def continuation(output, proof, context):
        reports = []
        mutation_result = None
        for cap in (12000, 10000):
            if context["device"].type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            reduced, reduced_proof, witness = sound_reduce(
                output, proof, cap, f"forced_cap_{cap}")
            checked = check_reduction_witness(
                output, proof, reduced, reduced_proof, witness)
            if context["device"].type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if mutation_result is None:
                mutation_result = _reduction_mutations(
                    output, proof, reduced, reduced_proof, witness)
            low, high = reduced.concretize()
            reports.append({
                "cap": cap,
                "generators_before": output.num_error_terms,
                "retained": len(witness["retained_indices"]),
                "dropped": len(witness["dropped_indices"]),
                "new_box_generators": witness["replacement_count"],
                "generators_after": reduced.num_error_terms,
                "protected": len(witness["protected_indices"]),
                "support_inflation": checked["support_inflation"],
                "range_inflation": checked["range_inflation"],
                "support_max": float((0.5 * (high-low)).max()),
                "lower_min": float(low.min()), "upper_max": float(high.max()),
                "seconds": elapsed,
                "allocated_bytes": int(torch.cuda.memory_allocated(
                    context["device"])) if context["device"].type == "cuda" else 0,
                "reserved_bytes": int(torch.cuda.memory_reserved(
                    context["device"])) if context["device"].type == "cuda" else 0,
                "witness": witness,
                "checker": checked,
            })
        return {"forced_caps": reports, "mutations": mutation_result}
    return run_sound_fp64(device=device, continuation=continuation)


def export_block0_state(path, device=None, run_representative_mpfr=True):
    destination = Path(path)

    def continuation(output, proof, _context):
        low, high = _ranges(output)
        payload = {
            "schema": "CORET_SOUND_FP64_BLOCK0_STATE_V1",
            "pinned_revision": PINNED_REVISION,
            "weights": output.zonotope_w.detach().cpu(),
            "range_low": low.detach().cpu(),
            "range_high": high.detach().cpu(),
            "proof": {
                "masks": list(proof.masks), "ids": list(proof.ids),
                "reasons": list(proof.reasons),
                "num_tokens": proof.num_tokens,
            },
        }
        if _context.get("separating_variance_witnesses"):
            payload["separating_variance_witnesses"] = _context["separating_variance_witnesses"]
        torch.save(payload, destination)
        return {
            "path": str(destination), "byte_count": destination.stat().st_size,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        }
    return run_sound_fp64(
        device=device, continuation=continuation,
        run_representative_mpfr=run_representative_mpfr)


def _state_payload(z, proof):
    low, high = _ranges(z)
    return {
        "weights": z.zonotope_w.detach().cpu(),
        "range_low": low.detach().cpu(), "range_high": high.detach().cpu(),
        "proof": {"masks": list(proof.masks), "ids": list(proof.ids),
                  "reasons": list(proof.reasons),
                  "num_tokens": proof.num_tokens},
    }


def _state_from_payload(payload, Zonotope, args, device):
    z = Zonotope(
        args=args, p=100, eps=prefix.FIXTURE_RHO,
        perturbed_word_index=prefix.FIXTURE_PERTURBED_TOKEN,
        zonotope_w=payload["weights"].to(device),
        error_term_range_low=payload["range_low"].to(device),
        error_term_range_high=payload["range_high"].to(device), clone=False)
    raw = payload["proof"]
    proof = structural.SupportProof(
        tuple(raw["masks"]), tuple(raw["ids"]), tuple(raw["reasons"]),
        int(raw["num_tokens"]))
    structural.attach_support(z, proof)
    structural.validate_support(
        z, proof, token_axis=-2 if z.zonotope_w.ndim == 4 else 1)
    return z, proof


def _load_artifact(path, expected_schema):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("schema") != expected_schema
            or payload.get("pinned_revision") != PINNED_REVISION):
        raise RuntimeError(f"artifact identity differs: {expected_schema}")
    return payload


def _save_artifact(path, schema, states, report):
    payload = {
        "schema": schema, "pinned_revision": PINNED_REVISION,
        "states": {name: _state_payload(z, proof)
                   for name, (z, proof) in states.items()},
        "report": report,
    }
    witnesses = report.pop("_separating_variance_witnesses", [])
    if witnesses:
        payload["separating_variance_witnesses"] = witnesses
    destination = Path(path)
    torch.save(payload, destination)
    return {"path": str(destination), "byte_count": destination.stat().st_size,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}


def _explicit_numerical_support(z, proof):
    indices = [index for index, reason in enumerate(proof.reasons)
               if reason in {
                   "fp64_roundoff_coordinate_box",
                   "sound_fp64_coordinate_box_replacement_with_numerical",
               }]
    if not indices:
        return 0.0
    tensor = torch.tensor(indices, dtype=torch.long, device=z.device)
    low, high = _ranges(z)
    scale = torch.maximum(low.abs(), high.abs()).index_select(0, tensor)
    rows = _generator_rows(z)
    if z.zonotope_w.ndim == 3:
        value = (rows.index_select(0, tensor).abs()
                 * scale.reshape(-1, 1, 1)).sum(dim=0)
    else:
        value = (rows.index_select(1, tensor).abs()
                 * scale.reshape(1, -1, 1, 1)).sum(dim=1)
    return float(value.max())


def _native_relational_support(z, proof):
    numerical_reasons = {
        "fp64_roundoff_coordinate_box",
        "sound_fp64_coordinate_box_replacement_with_numerical",
    }
    indices = [index for index, reason in enumerate(proof.reasons)
               if reason not in numerical_reasons]
    if not indices:
        return 0.0
    tensor = torch.tensor(indices, dtype=torch.long, device=z.device)
    low, high = _ranges(z)
    scale = torch.maximum(low.abs(), high.abs()).index_select(0, tensor)
    rows = _generator_rows(z)
    if z.zonotope_w.ndim == 3:
        value = (rows.index_select(0, tensor).abs()
                 * scale.reshape(-1, 1, 1)).sum(dim=0)
    else:
        value = (rows.index_select(1, tensor).abs()
                 * scale.reshape(1, -1, 1, 1)).sum(dim=1)
    return float(value.max())


def _state_measurement(label, z, proof, seconds):
    row = _metrics(label, z, seconds)
    numerical = _explicit_numerical_support(z, proof)
    native = _native_relational_support(z, proof)
    row["explicit_numerical_support_max"] = numerical
    row["plain_fp64_relational_support_max"] = native
    row["sound_support_max"] = row["support_max"]
    row["numerical_native_ratio"] = numerical / max(native, 1e-300)
    return row


def _maybe_reduce(z, proof, label, reductions):
    if z.num_error_terms <= MAXIMUM_GENERATORS:
        return z, proof
    before_low, before_high = z.concretize()
    reduced, reduced_proof, witness = sound_reduce(
        z, proof, MAXIMUM_GENERATORS, label)
    checked = check_reduction_witness(
        z, proof, reduced, reduced_proof, witness)
    after_low, after_high = reduced.concretize()
    reductions.append({
        "operator": label, "count_before": z.num_error_terms,
        "count_after": reduced.num_error_terms,
        "retained": len(witness["retained_indices"]),
        "absorbed": len(witness["dropped_indices"]),
        "added_box_generators": witness["replacement_count"],
        "support_inflation": checked["support_inflation"],
        "range_inflation": float(torch.maximum(
            before_low-after_low, after_high-before_high).max()),
        "retained_ids": witness["retained_ids"],
        "absorbed_ids": witness["dropped_ids"],
        "replacement_ids": witness["replacement_ids"],
        "selection_rule": witness["selection_rule"],
        "ranking_sha256": witness["ranking_sha256"],
    })
    return reduced, reduced_proof


def _maybe_reduce_pair(left, right, proof, label, reductions):
    if left.num_error_terms <= MAXIMUM_GENERATORS:
        return left, right, proof
    reduced_left, reduced_right, reduced_proof, witness = sound_reduce_pair(
        left, right, proof, MAXIMUM_GENERATORS, label)
    # Both independent replacement obligations are replayed before use.
    left_count = witness["left_witness"]["output_generator_count"]
    right_count = witness["right_witness"]["output_generator_count"]
    reductions.append({
        "operator": label, "count_before": left.num_error_terms,
        "count_after": reduced_left.num_error_terms,
        "retained": witness["retained"], "absorbed": witness["dropped"],
        "added_box_generators": witness["left_boxes"] + witness["right_boxes"],
        "left_preunion_count": left_count,
        "right_preunion_count": right_count,
        "support_inflation": max(
            witness["left_containment"]["support_inflation"],
            witness["right_containment"]["support_inflation"]),
        "range_inflation": max(
            witness["left_containment"]["range_inflation"],
            witness["right_containment"]["range_inflation"]),
        "retained_ids": witness["retained_ids"],
        "absorbed_ids": witness["dropped_ids"],
        "left_replacement_ids": witness["left_witness"]["replacement_ids"],
        "right_replacement_ids": witness["right_witness"]["replacement_ids"],
        "selection_rule": witness["selection_rule"],
        "ranking_sha256": witness["ranking_sha256"],
    })
    return reduced_left, reduced_right, reduced_proof


def _recenter_sound(z, proof, label, measurements, reductions):
    """Execute production's conditional recenter with embedded FP64 error."""
    if z.error_term_range_low is None:
        structural.attach_support(z, proof)
        return z, proof, {"executed": False, "predicate": "ranges_absent"}
    source = z
    result = production._recenter_native_ranges(source)
    structural.attach_support(result, proof)
    operations = 2 * (source.num_error_terms + 1) + 2
    result, result_proof = _inject(
        result, proof, [source], label, operations, measurements,
        reserve=_reserve_from_majorant(_absolute_hull(source), operations))
    result, result_proof = _maybe_reduce(
        result, result_proof, label, reductions)
    return result, result_proof, {
        "executed": True, "predicate": "explicit_ranges_present",
        "input_range_count": int(source.num_error_terms),
    }


def run_block1_stage1(block0_path, output_path, device=None):
    """Block-1 Q/K -> QK -> scaling -> softmax, then stop before A.V."""
    device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    source = _load_artifact(block0_path, "CORET_SOUND_FP64_BLOCK0_STATE_V1")
    prior_dtype = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    try:
        with pinned_zonotope() as Zonotope:
            args = _args(device)
            hidden, hidden_proof = _state_from_payload(
                source, Zonotope, args, device)
            checkpoint = prefix._load_checkpoint()
            base = "bert.encoder.layer.1"
            query = _parameter(checkpoint, base + ".attention.self.query", device)
            key = _parameter(checkpoint, base + ".attention.self.key", device)
            measurements, reductions = [], []
            delegate = structural.StructuralNativeSemanticOperators()
            delegate._hidden = hidden_proof
            delegate._qk_index = 1; delegate._softmax_index = 1
            dispatch = production.NativeProductionDispatch(delegate=delegate)
            if device.type == "cuda": torch.cuda.synchronize()
            started = time.perf_counter()
            block_input = hidden
            hidden, hidden_proof, recenter = _recenter_sound(
                hidden, hidden_proof, "b1_pre_qk_recenter", measurements,
                reductions)
            delegate._hidden = hidden_proof
            q, qp = _dense_sound(
                hidden, hidden_proof, query, "b1_q_affine", measurements)
            k, kp = _dense_sound(
                hidden, hidden_proof, key, "b1_k_affine", measurements)
            q, k, pair_proof = _align_states(q, qp, k, kp, "b1_qk_branches")
            q = q.add_attention_heads_dim(4); k = k.add_attention_heads_dim(4)
            structural.attach_support(q, pair_proof); structural.attach_support(k, pair_proof)
            q, k, pair_proof = _maybe_reduce_pair(
                q, k, pair_proof, "b1_pre_qk_pair", reductions)
            delegate._hidden = pair_proof
            raw_qk = dispatch.qk(q, k); raw_qk_proof = structural.get_support(raw_qk)
            ops = 32 * (q.num_error_terms + 1) ** 2 * 16 + 4096
            qk, qk_proof = _inject(
                raw_qk, raw_qk_proof, [q, k], "b1_qk", ops, measurements,
                reserve=_reserve_from_majorant(_bilinear_majorant(q, k), ops))
            qk, qk_proof = _maybe_reduce(qk, qk_proof, "b1_qk", reductions)
            scale = 1.0 / math.sqrt(32)
            raw_scores = qk.multiply(scale); structural.attach_support(raw_scores, qk_proof)
            reserve = _outward_positive(
                _absolute_hull(qk) * abs(scale) * (2*_gamma(1)+FP64_U))
            scores, score_proof = _inject(
                raw_scores, qk_proof, [qk], "b1_score_scaling", 1,
                measurements, reserve=reserve)
            scores, score_proof = _maybe_reduce(
                scores, score_proof, "b1_score_scaling", reductions)
            delegate._score = score_proof
            raw_probability = dispatch.softmax(scores, no_constraints=False)
            raw_probability_proof = structural.get_support(raw_probability)
            ops = 64 * (scores.num_error_terms + 1) + 8192
            probability, probability_proof = _inject(
                raw_probability, raw_probability_proof, [scores],
                "b1_softmax", ops, measurements,
                reserve=_reserve_from_majorant(_softmax_majorant(scores), ops))
            probability, probability_proof = _maybe_reduce(
                probability, probability_proof, "b1_softmax", reductions)
            if device.type == "cuda": torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            report = {
                "measurements": [
                    _state_measurement("block1_input", block_input,
                                       structural.get_support(block_input), 0.0),
                    _state_measurement("block1_pre_qk", hidden,
                                       hidden_proof, 0.0),
                    _state_measurement("block1_qk", qk, qk_proof, 0.0),
                    _state_measurement("block1_softmax", probability,
                                       probability_proof, seconds)],
                "recenter": recenter,
                "injections": measurements, "reductions": reductions,
                "dispatch_counts": dict(dispatch.counts), "seconds": seconds,
            }
            artifact = _save_artifact(
                output_path, "CORET_SOUND_FP64_BLOCK1_STAGE1_V1",
                {"hidden": (hidden, hidden_proof),
                 "probability": (probability, probability_proof)}, report)
            return {"report": report, "artifact": artifact}
    finally:
        torch.set_default_dtype(prior_dtype)


def run_block1_pre_qk(block0_path, output_path, device=None):
    """Execute the exact Block-1 boundary recenter and stop before Q/K."""
    device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    source = _load_artifact(block0_path, "CORET_SOUND_FP64_BLOCK0_STATE_V1")
    prior_dtype = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    try:
        with pinned_zonotope() as Zonotope:
            args = _args(device)
            hidden, hidden_proof = _state_from_payload(source, Zonotope, args,
                                                       device)
            before = _state_measurement(
                "block1_input", hidden, hidden_proof, 0.0)
            measurements, reductions = [], []
            if device.type == "cuda": torch.cuda.synchronize()
            started = time.perf_counter()
            hidden, hidden_proof, recenter = _recenter_sound(
                hidden, hidden_proof, "b1_pre_qk_recenter", measurements,
                reductions)
            if device.type == "cuda": torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            report = {
                "measurements": [before, _state_measurement(
                    "block1_pre_qk", hidden, hidden_proof, seconds)],
                "recenter": recenter, "injections": measurements,
                "reductions": reductions, "seconds": seconds,
            }
            artifact = _save_artifact(
                output_path, "CORET_SOUND_FP64_BLOCK1_PRE_QK_V1",
                {"hidden": (hidden, hidden_proof)}, report)
            return {"report": report, "artifact": artifact}
    finally:
        torch.set_default_dtype(prior_dtype)


def run_block1_qk_prepare(input_path, output_path, device=None):
    """Build and align Q/K, stopping immediately before native precise QK."""
    device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    source = _load_artifact(input_path, "CORET_SOUND_FP64_BLOCK1_PRE_QK_V1")
    prior_dtype = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    try:
        with pinned_zonotope() as Zonotope:
            args = _args(device)
            hidden, hidden_proof = _state_from_payload(
                source["states"]["hidden"], Zonotope, args, device)
            checkpoint = prefix._load_checkpoint(); base = "bert.encoder.layer.1"
            query = _parameter(checkpoint, base + ".attention.self.query", device)
            key = _parameter(checkpoint, base + ".attention.self.key", device)
            measurements, reductions = [], []
            if device.type == "cuda": torch.cuda.synchronize()
            started = time.perf_counter()
            q, qp = _dense_sound(
                hidden, hidden_proof, query, "b1_q_affine", measurements)
            k, kp = _dense_sound(
                hidden, hidden_proof, key, "b1_k_affine", measurements)
            q, k, proof = _align_states(q, qp, k, kp, "b1_qk_branches")
            q = q.add_attention_heads_dim(4); k = k.add_attention_heads_dim(4)
            structural.attach_support(q, proof); structural.attach_support(k, proof)
            q, k, proof = _maybe_reduce_pair(
                q, k, proof, "b1_pre_qk_pair", reductions)
            if device.type == "cuda": torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            report = {"injections": measurements, "reductions": reductions,
                      "seconds": seconds}
            artifact = _save_artifact(
                output_path, "CORET_SOUND_FP64_BLOCK1_QK_INPUT_V1",
                {"hidden": (hidden, hidden_proof), "q": (q, proof),
                 "k": (k, proof)}, report)
            return {"report": report, "artifact": artifact}
    finally:
        torch.set_default_dtype(prior_dtype)


def run_block1_qk_softmax(input_path, output_path, device=None):
    """Native precise QK, scaling, and softmax from persisted aligned operands."""
    device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    source = _load_artifact(input_path, "CORET_SOUND_FP64_BLOCK1_QK_INPUT_V1")
    prior_dtype = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    try:
        with pinned_zonotope() as Zonotope:
            args = _args(device)
            hidden, hidden_proof = _state_from_payload(
                source["states"]["hidden"], Zonotope, args, device)
            q, proof = _state_from_payload(
                source["states"]["q"], Zonotope, args, device)
            k, k_proof = _state_from_payload(
                source["states"]["k"], Zonotope, args, device)
            if proof != k_proof:
                raise RuntimeError("Q/K input proofs differ")
            delegate = structural.StructuralNativeSemanticOperators()
            delegate._qk_index = 1; delegate._softmax_index = 1
            delegate._hidden = proof
            dispatch = production.NativeProductionDispatch(delegate=delegate)
            measurements, reductions = [], []
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            started = time.perf_counter()
            raw_qk = dispatch.qk(q, k)
            raw_qk_proof = structural.get_support(raw_qk)
            ops = 32 * (q.num_error_terms + 1) ** 2 * 16 + 4096
            qk, qk_proof = _inject(
                raw_qk, raw_qk_proof, [q, k], "b1_qk", ops, measurements,
                reserve=_reserve_from_majorant(_bilinear_majorant(q, k), ops))
            qk, qk_proof = _maybe_reduce(qk, qk_proof, "b1_qk", reductions)
            scale = 1.0 / math.sqrt(32)
            raw_scores = qk.multiply(scale); structural.attach_support(raw_scores, qk_proof)
            score_reserve = _outward_positive(
                _absolute_hull(qk) * abs(scale) * (2*_gamma(1)+FP64_U))
            scores, score_proof = _inject(
                raw_scores, qk_proof, [qk], "b1_score_scaling", 1,
                measurements, reserve=score_reserve)
            scores, score_proof = _maybe_reduce(
                scores, score_proof, "b1_score_scaling", reductions)
            score_low, score_high = scores.concretize()
            delegate._score = score_proof
            raw_probability = dispatch.softmax(scores, no_constraints=False)
            raw_probability_proof = structural.get_support(raw_probability)
            softmax_ops = 64 * (scores.num_error_terms + 1) + 8192
            probability, probability_proof = _inject(
                raw_probability, raw_probability_proof, [scores],
                "b1_softmax", softmax_ops, measurements,
                reserve=_reserve_from_majorant(
                    _softmax_majorant(scores), softmax_ops))
            probability, probability_proof = _maybe_reduce(
                probability, probability_proof, "b1_softmax", reductions)
            if device.type == "cuda": torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            report = {
                "measurements": [
                    _state_measurement("block1_qk", qk, qk_proof, 0.0),
                    _state_measurement("block1_softmax", probability,
                                       probability_proof, seconds)],
                "softmax_branch": {
                    "no_constraints": False,
                    "score_lower_min": float(score_low.min()),
                    "score_upper_max": float(score_high.max()),
                    "native_domain_checks_passed": True,
                    "sum_equality_enabled": True,
                },
                "injections": measurements, "reductions": reductions,
                "dispatch_counts": dict(dispatch.counts), "seconds": seconds,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated())
                if device.type == "cuda" else 0,
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())
                if device.type == "cuda" else 0,
            }
            artifact = _save_artifact(
                output_path, "CORET_SOUND_FP64_BLOCK1_STAGE1_V1",
                {"hidden": (hidden, hidden_proof),
                 "probability": (probability, probability_proof)}, report)
            return {"report": report, "artifact": artifact}
    finally:
        torch.set_default_dtype(prior_dtype)


def run_block1_qk_only(input_path, output_path, device=None):
    """Native precise Block-1 QK with sound FP64 embedding."""
    device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    source = _load_artifact(input_path, "CORET_SOUND_FP64_BLOCK1_QK_INPUT_V1")
    prior_dtype = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    try:
        with pinned_zonotope() as Zonotope:
            args = _args(device)
            hidden, hidden_proof = _state_from_payload(
                source["states"]["hidden"], Zonotope, args, device)
            q, proof = _state_from_payload(
                source["states"]["q"], Zonotope, args, device)
            k, k_proof = _state_from_payload(
                source["states"]["k"], Zonotope, args, device)
            if proof != k_proof:
                raise RuntimeError("Q/K input proofs differ")
            delegate = structural.StructuralNativeSemanticOperators()
            delegate._qk_index = 1; delegate._hidden = proof
            dispatch = production.NativeProductionDispatch(delegate=delegate)
            measurements, reductions = [], []
            if device.type == "cuda": torch.cuda.synchronize()
            started = time.perf_counter()
            raw = dispatch.qk(q, k); raw_proof = structural.get_support(raw)
            ops = 32 * (q.num_error_terms + 1) ** 2 * 16 + 4096
            qk, qk_proof = _inject(
                raw, raw_proof, [q, k], "b1_qk", ops, measurements,
                reserve=_reserve_from_majorant(_bilinear_majorant(q, k), ops))
            qk, qk_proof = _maybe_reduce(qk, qk_proof, "b1_qk", reductions)
            if device.type == "cuda": torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            report = {
                "measurements": [_state_measurement(
                    "block1_qk", qk, qk_proof, seconds)],
                "injections": measurements, "reductions": reductions,
                "dispatch_counts": dict(dispatch.counts), "seconds": seconds,
            }
            artifact = _save_artifact(
                output_path, "CORET_SOUND_FP64_BLOCK1_QK_V1",
                {"hidden": (hidden, hidden_proof), "qk": (qk, qk_proof)},
                report)
            return {"report": report, "artifact": artifact}
    finally:
        torch.set_default_dtype(prior_dtype)


def run_block1_qk_compute(input_path, output_path, device=None):
    """Native precise QK and FP64 reserve, stopping before order reduction."""
    device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    source = _load_artifact(input_path, "CORET_SOUND_FP64_BLOCK1_QK_INPUT_V1")
    prior_dtype = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    try:
        with pinned_zonotope() as Zonotope:
            args = _args(device)
            hidden, hidden_proof = _state_from_payload(
                source["states"]["hidden"], Zonotope, args, device)
            q, proof = _state_from_payload(
                source["states"]["q"], Zonotope, args, device)
            k, k_proof = _state_from_payload(
                source["states"]["k"], Zonotope, args, device)
            if proof != k_proof:
                raise RuntimeError("Q/K input proofs differ")
            delegate = structural.StructuralNativeSemanticOperators()
            delegate._qk_index = 1; delegate._hidden = proof
            dispatch = production.NativeProductionDispatch(delegate=delegate)
            measurements, reductions = [], []
            if device.type == "cuda": torch.cuda.synchronize()
            started = time.perf_counter()
            raw = dispatch.qk(q, k); raw_proof = structural.get_support(raw)
            ops = 32 * (q.num_error_terms + 1) ** 2 * 16 + 4096
            qk, qk_proof = _inject(
                raw, raw_proof, [q, k], "b1_qk", ops, measurements,
                reserve=_reserve_from_majorant(_bilinear_majorant(q, k), ops))
            if device.type == "cuda": torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            report = {"injections": measurements,
                      "dispatch_counts": dict(dispatch.counts),
                      "seconds": seconds}
            artifact = _save_artifact(
                output_path, "CORET_SOUND_FP64_BLOCK1_QK_PRE_REDUCTION_V1",
                {"hidden": (hidden, hidden_proof), "qk": (qk, qk_proof)},
                report)
            return {"report": report, "artifact": artifact}
    finally:
        torch.set_default_dtype(prior_dtype)


def run_block1_qk_reduce(input_path, output_path, device=None):
    """Apply the normal 14k reduction to the persisted QK result."""
    device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    source = _load_artifact(
        input_path, "CORET_SOUND_FP64_BLOCK1_QK_PRE_REDUCTION_V1")
    prior_dtype = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    try:
        with pinned_zonotope() as Zonotope:
            args = _args(device)
            hidden, hidden_proof = _state_from_payload(
                source["states"]["hidden"], Zonotope, args, device)
            qk, qk_proof = _state_from_payload(
                source["states"]["qk"], Zonotope, args, device)
            reductions = []
            if device.type == "cuda": torch.cuda.synchronize()
            started = time.perf_counter()
            qk, qk_proof = _maybe_reduce(qk, qk_proof, "b1_qk", reductions)
            if device.type == "cuda": torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            report = {
                "measurements": [_state_measurement(
                    "block1_qk", qk, qk_proof, seconds)],
                "reductions": reductions, "seconds": seconds,
            }
            artifact = _save_artifact(
                output_path, "CORET_SOUND_FP64_BLOCK1_QK_V1",
                {"hidden": (hidden, hidden_proof), "qk": (qk, qk_proof)},
                report)
            return {"report": report, "artifact": artifact}
    finally:
        torch.set_default_dtype(prior_dtype)


def run_block1_qk_native(input_path, output_path, device=None):
    """Execute only the unchanged native precise QK transformer."""
    device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    source = _load_artifact(input_path, "CORET_SOUND_FP64_BLOCK1_QK_INPUT_V1")
    prior_dtype = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    try:
        with pinned_zonotope() as Zonotope:
            args = _args(device)
            q, proof = _state_from_payload(
                source["states"]["q"], Zonotope, args, device)
            k, k_proof = _state_from_payload(
                source["states"]["k"], Zonotope, args, device)
            if proof != k_proof:
                raise RuntimeError("Q/K input proofs differ")
            delegate = structural.StructuralNativeSemanticOperators()
            delegate._qk_index = 1; delegate._hidden = proof
            dispatch = production.NativeProductionDispatch(delegate=delegate)
            if device.type == "cuda": torch.cuda.synchronize()
            started = time.perf_counter()
            raw = dispatch.qk(q, k); raw_proof = structural.get_support(raw)
            if device.type == "cuda": torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            report = {"dispatch_counts": dict(dispatch.counts),
                      "seconds": seconds}
            artifact = _save_artifact(
                output_path, "CORET_SOUND_FP64_BLOCK1_QK_NATIVE_V1",
                {"raw_qk": (raw, raw_proof)}, report)
            return {"report": report, "artifact": artifact}
    finally:
        torch.set_default_dtype(prior_dtype)


def run_block1_qk_reserve(input_path, output_path, device=None):
    """Construct the analytic QK FP64 reserve without executing native QK."""
    device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    source = _load_artifact(input_path, "CORET_SOUND_FP64_BLOCK1_QK_INPUT_V1")
    prior_dtype = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    try:
        with pinned_zonotope() as Zonotope:
            args = _args(device)
            q, proof = _state_from_payload(
                source["states"]["q"], Zonotope, args, device)
            k, k_proof = _state_from_payload(
                source["states"]["k"], Zonotope, args, device)
            if proof != k_proof:
                raise RuntimeError("Q/K input proofs differ")
            if device.type == "cuda": torch.cuda.synchronize()
            started = time.perf_counter()
            operations = 32 * (q.num_error_terms + 1) ** 2 * 16 + 4096
            reserve = _reserve_from_majorant(
                _bilinear_majorant(q, k), operations)
            if device.type == "cuda": torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            destination = Path(output_path)
            torch.save({
                "schema": "CORET_SOUND_FP64_BLOCK1_QK_RESERVE_V1",
                "pinned_revision": PINNED_REVISION,
                "reserve": reserve.detach().cpu(),
                "operations": operations,
                "q_weights_sha256": _tensor_sha(q.zonotope_w),
                "k_weights_sha256": _tensor_sha(k.zonotope_w),
                "seconds": seconds,
            }, destination)
            return {"seconds": seconds, "artifact": {
                "path": str(destination),
                "byte_count": destination.stat().st_size,
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            }}
    finally:
        torch.set_default_dtype(prior_dtype)


def run_block1_qk_inject(qk_input_path, native_path, reserve_path,
                         output_path, device=None):
    """Embed the authenticated analytic reserve in the native QK state."""
    device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    input_source = _load_artifact(
        qk_input_path, "CORET_SOUND_FP64_BLOCK1_QK_INPUT_V1")
    native_source = _load_artifact(
        native_path, "CORET_SOUND_FP64_BLOCK1_QK_NATIVE_V1")
    reserve_source = torch.load(
        reserve_path, map_location="cpu", weights_only=False)
    if (reserve_source.get("schema")
            != "CORET_SOUND_FP64_BLOCK1_QK_RESERVE_V1"
            or reserve_source.get("pinned_revision") != PINNED_REVISION):
        raise RuntimeError("QK reserve artifact identity differs")
    prior_dtype = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    try:
        with pinned_zonotope() as Zonotope:
            args = _args(device)
            hidden, hidden_proof = _state_from_payload(
                input_source["states"]["hidden"], Zonotope, args, device)
            q, q_proof = _state_from_payload(
                input_source["states"]["q"], Zonotope, args, device)
            k, k_proof = _state_from_payload(
                input_source["states"]["k"], Zonotope, args, device)
            if (_tensor_sha(q.zonotope_w) != reserve_source["q_weights_sha256"]
                    or _tensor_sha(k.zonotope_w)
                    != reserve_source["k_weights_sha256"]):
                raise RuntimeError("QK reserve predecessor hash differs")
            raw, raw_proof = _state_from_payload(
                native_source["states"]["raw_qk"], Zonotope, args, device)
            measurements = []
            if device.type == "cuda": torch.cuda.synchronize()
            started = time.perf_counter()
            qk, qk_proof = _inject(
                raw, raw_proof, [q, k], "b1_qk",
                int(reserve_source["operations"]), measurements,
                reserve=reserve_source["reserve"].to(device))
            if device.type == "cuda": torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            report = {"injections": measurements, "seconds": seconds}
            artifact = _save_artifact(
                output_path, "CORET_SOUND_FP64_BLOCK1_QK_PRE_REDUCTION_V1",
                {"hidden": (hidden, hidden_proof), "qk": (qk, qk_proof)},
                report)
            return {"report": report, "artifact": artifact}
    finally:
        torch.set_default_dtype(prior_dtype)


def run_block1_softmax_only(input_path, output_path, device=None):
    """Score scaling and native relational softmax from persisted QK."""
    device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    source = _load_artifact(input_path, "CORET_SOUND_FP64_BLOCK1_QK_V1")
    prior_dtype = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    try:
        with pinned_zonotope() as Zonotope:
            args = _args(device)
            hidden, hidden_proof = _state_from_payload(
                source["states"]["hidden"], Zonotope, args, device)
            qk, qk_proof = _state_from_payload(
                source["states"]["qk"], Zonotope, args, device)
            delegate = structural.StructuralNativeSemanticOperators()
            delegate._softmax_index = 1; delegate._score = qk_proof
            dispatch = production.NativeProductionDispatch(delegate=delegate)
            measurements, reductions = [], []
            if device.type == "cuda": torch.cuda.synchronize()
            started = time.perf_counter()
            scale = 1.0 / math.sqrt(32)
            raw_scores = qk.multiply(scale); structural.attach_support(raw_scores, qk_proof)
            reserve = _outward_positive(
                _absolute_hull(qk) * abs(scale) * (2*_gamma(1)+FP64_U))
            scores, score_proof = _inject(
                raw_scores, qk_proof, [qk], "b1_score_scaling", 1,
                measurements, reserve=reserve)
            scores, score_proof = _maybe_reduce(
                scores, score_proof, "b1_score_scaling", reductions)
            score_low, score_high = scores.concretize()
            delegate._score = score_proof
            raw = dispatch.softmax(scores, no_constraints=False)
            raw_proof = structural.get_support(raw)
            ops = 64 * (scores.num_error_terms + 1) + 8192
            probability, probability_proof = _inject(
                raw, raw_proof, [scores], "b1_softmax", ops, measurements,
                reserve=_reserve_from_majorant(_softmax_majorant(scores), ops))
            probability, probability_proof = _maybe_reduce(
                probability, probability_proof, "b1_softmax", reductions)
            if device.type == "cuda": torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            report = {
                "measurements": [_state_measurement(
                    "block1_softmax", probability, probability_proof,
                    seconds)],
                "softmax_branch": {
                    "no_constraints": False,
                    "score_lower_min": float(score_low.min()),
                    "score_upper_max": float(score_high.max()),
                    "native_domain_checks_passed": True,
                    "sum_equality_enabled": True,
                },
                "injections": measurements, "reductions": reductions,
                "dispatch_counts": dict(dispatch.counts), "seconds": seconds,
                "generic_fallback_count": dispatch.generic_family_invocations,
            }
            artifact = _save_artifact(
                output_path, "CORET_SOUND_FP64_BLOCK1_STAGE1_V1",
                {"hidden": (hidden, hidden_proof),
                 "probability": (probability, probability_proof)}, report)
            return {"report": report, "artifact": artifact}
    finally:
        torch.set_default_dtype(prior_dtype)


def run_block1_stage2(stage1_path, output_path, device=None):
    """Block-1 V/A.V through post-attention LayerNorm."""
    device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    source = _load_artifact(stage1_path, "CORET_SOUND_FP64_BLOCK1_STAGE1_V1")
    prior_dtype = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    try:
        with pinned_zonotope() as Zonotope:
            args = _args(device)
            hidden, hidden_proof = _state_from_payload(
                source["states"]["hidden"], Zonotope, args, device)
            probability, probability_proof = _state_from_payload(
                source["states"]["probability"], Zonotope, args, device)
            checkpoint = prefix._load_checkpoint(); base = "bert.encoder.layer.1"
            value_parameter = _parameter(
                checkpoint, base + ".attention.self.value", device)
            output_parameter = _parameter(
                checkpoint, base + ".attention.output.dense", device)
            layernorm = _parameter(
                checkpoint, base + ".attention.output.LayerNorm", device)
            measurements, reductions, oracles = [], [], []
            delegate = structural.StructuralNativeSemanticOperators()
            delegate._av_index = 1; delegate._layer_norm_index = 3
            dispatch = production.NativeProductionDispatch(delegate=delegate)
            if device.type == "cuda": torch.cuda.synchronize()
            started = time.perf_counter()
            value, value_proof = _dense_sound(
                hidden, hidden_proof, value_parameter, "b1_v_affine", measurements)
            value = value.add_attention_heads_dim(4)
            structural.attach_support(value, value_proof)
            probability, value, av_proof = _align_states(
                probability, probability_proof, value, value_proof,
                "b1_av_branches")
            probability, value, av_proof = _maybe_reduce_pair(
                probability, value, av_proof, "b1_pre_av_pair", reductions)
            delegate._probability = av_proof; delegate._value = av_proof
            raw_context = dispatch.attention_value(probability, value)
            raw_context_proof = structural.get_support(raw_context)
            ops = (32 * (max(probability.num_error_terms,
                             value.num_error_terms) + 1) ** 2 * 16 + 4096)
            context, context_proof = _inject(
                raw_context, raw_context_proof, [probability, value],
                "b1_attention_value", ops, measurements,
                reserve=_reserve_from_majorant(
                    _bilinear_majorant(probability, value.t()), ops))
            context, context_proof = _maybe_reduce(
                context, context_proof, "b1_attention_value", reductions)
            context_measure = _state_measurement(
                "block1_attention_value", context, context_proof, 0.0)

            context = context.remove_attention_heads_dim()
            structural.attach_support(context, context_proof)
            attention, attention_proof = _dense_sound(
                context, context_proof, output_parameter,
                "b1_attention_output_affine", measurements)
            attention, attention_proof = _maybe_reduce(
                attention, attention_proof, "b1_attention_output_affine",
                reductions)
            attention, aligned_hidden, residual_proof = _align_states(
                attention, attention_proof, hidden, hidden_proof,
                "b1_attention_residual")
            attention, aligned_hidden, residual_proof = _maybe_reduce_pair(
                attention, aligned_hidden, residual_proof,
                "b1_pre_attention_residual_pair", reductions)
            raw_residual = attention.add(aligned_hidden)
            residual, residual_proof = _inject(
                raw_residual, residual_proof, [attention, aligned_hidden],
                "b1_attention_residual", 1, measurements,
                reserve=_add_reserve(attention, aligned_hidden))
            residual, residual_proof = _maybe_reduce(
                residual, residual_proof, "b1_attention_residual", reductions)
            width = residual.word_embedding_size
            average = torch.ones((width, width), dtype=torch.float64,
                                 device=device) / width
            centered = residual.add(residual.matmul(average).multiply(-1.0))
            variance = centered.square_and_sum_and_repeat().multiply(1.0/width)
            variance_low, variance_high = variance.concretize()
            if float(variance_low.min()) <= 0:
                raise RuntimeError("Block-1 post-attention variance is nonpositive")
            delegate._hidden = residual_proof
            delegate._attention_output = residual_proof
            raw_post = dispatch.layer_norm(residual, layernorm, "standard")
            raw_post_proof = structural.get_support(raw_post)
            ops = 16 * 128 * (residual.num_error_terms + 1) ** 2 + 4096
            post, post_proof = _inject(
                raw_post, raw_post_proof, [residual],
                "b1_post_attention_layernorm", ops, measurements,
                reserve=_reserve_from_majorant(
                    _layernorm_majorant(residual, layernorm), ops))
            post, post_proof = _maybe_reduce(
                post, post_proof, "b1_post_attention_layernorm", reductions)
            if device.type == "cuda": torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            report = {
                "measurements": [context_measure, _state_measurement(
                    "block1_post_attention_layernorm", post, post_proof,
                    seconds)],
                "variance_lower": float(variance_low.min()),
                "variance_upper_min": float(variance_high.min()),
                "injections": measurements, "reductions": reductions,
                "oracles": oracles, "dispatch_counts": dict(dispatch.counts),
                "seconds": seconds,
            }
            artifact = _save_artifact(
                output_path, "CORET_SOUND_FP64_BLOCK1_STAGE2_V1",
                {"post_attention": (post, post_proof)}, report)
            return {"report": report, "artifact": artifact}
    finally:
        torch.set_default_dtype(prior_dtype)


def run_block1_stage2_av(stage1_path, output_path, device=None):
    """Bounded Block-1 V/A.V segment only."""
    device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    source = _load_artifact(stage1_path, "CORET_SOUND_FP64_BLOCK1_STAGE1_V1")
    prior_dtype = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    try:
        with pinned_zonotope() as Zonotope:
            args = _args(device)
            hidden, hidden_proof = _state_from_payload(
                source["states"]["hidden"], Zonotope, args, device)
            probability, probability_proof = _state_from_payload(
                source["states"]["probability"], Zonotope, args, device)
            checkpoint = prefix._load_checkpoint(); base = "bert.encoder.layer.1"
            parameter = _parameter(
                checkpoint, base + ".attention.self.value", device)
            measurements, reductions = [], []
            delegate = structural.StructuralNativeSemanticOperators()
            delegate._av_index = 1
            dispatch = production.NativeProductionDispatch(delegate=delegate)
            if device.type == "cuda": torch.cuda.synchronize()
            started = time.perf_counter()
            value, value_proof = _dense_sound(
                hidden, hidden_proof, parameter, "b1_v_affine", measurements)
            value = value.add_attention_heads_dim(4)
            structural.attach_support(value, value_proof)
            probability, value, pair_proof = _align_states(
                probability, probability_proof, value, value_proof,
                "b1_av_branches")
            probability, value, pair_proof = _maybe_reduce_pair(
                probability, value, pair_proof, "b1_pre_av_pair", reductions)
            delegate._probability = pair_proof; delegate._value = pair_proof
            raw = dispatch.attention_value(probability, value)
            raw_proof = structural.get_support(raw)
            ops = 32 * (probability.num_error_terms + 1) ** 2 * 16 + 4096
            context, context_proof = _inject(
                raw, raw_proof, [probability, value], "b1_attention_value",
                ops, measurements, reserve=_reserve_from_majorant(
                    _bilinear_majorant(probability, value.t()), ops))
            context, context_proof = _maybe_reduce(
                context, context_proof, "b1_attention_value", reductions)
            if device.type == "cuda": torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            report = {
                "measurements": [_state_measurement(
                    "block1_attention_value", context, context_proof, seconds)],
                "injections": measurements, "reductions": reductions,
                "dispatch_counts": dict(dispatch.counts), "seconds": seconds,
            }
            artifact = _save_artifact(
                output_path, "CORET_SOUND_FP64_BLOCK1_STAGE2_AV_V1",
                {"hidden": (hidden, hidden_proof),
                 "context": (context, context_proof)}, report)
            return {"report": report, "artifact": artifact}
    finally:
        torch.set_default_dtype(prior_dtype)


def run_block1_stage2_prepare(stage1_path, output_path, device=None):
    device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    source = _load_artifact(stage1_path, "CORET_SOUND_FP64_BLOCK1_STAGE1_V1")
    prior_dtype = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    try:
        with pinned_zonotope() as Zonotope:
            args = _args(device)
            hidden, hidden_proof = _state_from_payload(
                source["states"]["hidden"], Zonotope, args, device)
            probability, probability_proof = _state_from_payload(
                source["states"]["probability"], Zonotope, args, device)
            checkpoint = prefix._load_checkpoint()
            parameter = _parameter(
                checkpoint, "bert.encoder.layer.1.attention.self.value", device)
            measurements, reductions = [], []
            if device.type == "cuda": torch.cuda.synchronize()
            started = time.perf_counter()
            value, value_proof = _dense_sound(
                hidden, hidden_proof, parameter, "b1_v_affine", measurements)
            value = value.add_attention_heads_dim(4)
            structural.attach_support(value, value_proof)
            probability, value, pair_proof = _align_states(
                probability, probability_proof, value, value_proof,
                "b1_av_branches")
            probability, value, pair_proof = _maybe_reduce_pair(
                probability, value, pair_proof, "b1_pre_av_pair", reductions)
            if device.type == "cuda": torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            report = {"injections": measurements, "reductions": reductions,
                      "seconds": seconds}
            artifact = _save_artifact(
                output_path, "CORET_SOUND_FP64_BLOCK1_AV_INPUT_V1",
                {"hidden": (hidden, hidden_proof),
                 "probability": (probability, pair_proof),
                 "value": (value, pair_proof)}, report)
            return {"report": report, "artifact": artifact}
    finally:
        torch.set_default_dtype(prior_dtype)


def run_block1_stage2_av_only(input_path, output_path, device=None):
    device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    source = _load_artifact(input_path, "CORET_SOUND_FP64_BLOCK1_AV_INPUT_V1")
    prior_dtype = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    try:
        with pinned_zonotope() as Zonotope:
            args = _args(device)
            hidden, hidden_proof = _state_from_payload(
                source["states"]["hidden"], Zonotope, args, device)
            probability, proof = _state_from_payload(
                source["states"]["probability"], Zonotope, args, device)
            value, value_proof = _state_from_payload(
                source["states"]["value"], Zonotope, args, device)
            if proof != value_proof:
                raise RuntimeError("A.V input proof alignment differs")
            delegate = structural.StructuralNativeSemanticOperators()
            delegate._av_index = 1; delegate._probability = proof
            delegate._value = proof
            dispatch = production.NativeProductionDispatch(delegate=delegate)
            measurements, reductions = [], []
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            started = time.perf_counter()
            raw = dispatch.attention_value(probability, value)
            raw_proof = structural.get_support(raw)
            ops = 32 * (probability.num_error_terms + 1) ** 2 * 16 + 4096
            reserve = _reserve_from_majorant(
                _bilinear_majorant(probability, value.t()), ops)
            transposed = value.t()
            with _mp_context(gmpy2.RoundToNearest):
                exact = _mp(0)
                for key_index in range(value.num_words):
                    exact += (
                        _mp(probability.zonotope_w[0, 0, 0, key_index])
                        * _mp(transposed.zonotope_w[0, 1, 0, key_index]))
                    exact += (
                        _mp(probability.zonotope_w[0, 1, 0, key_index])
                        * _mp(transposed.zonotope_w[0, 0, 0, key_index]))
            mpfr_spot = _oracle_containment(
                "block1_actual_av_retained_coefficient", exact,
                raw.zonotope_w[0, 1, 0, 0], float(reserve.max()))
            context, context_proof = _inject(
                raw, raw_proof, [probability, value], "b1_attention_value",
                ops, measurements, reserve=reserve)
            context, context_proof = _maybe_reduce(
                context, context_proof, "b1_attention_value", reductions)
            if device.type == "cuda": torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            report = {
                "measurements": [_state_measurement(
                    "block1_attention_value", context, context_proof, seconds)],
                "injections": measurements, "reductions": reductions,
                "mpfr_spots": [mpfr_spot],
                "dispatch_counts": dict(dispatch.counts), "seconds": seconds,
                "generic_fallback_count": dispatch.generic_family_invocations,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda" else 0,
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))
                if device.type == "cuda" else 0,
            }
            artifact = _save_artifact(
                output_path, "CORET_SOUND_FP64_BLOCK1_STAGE2_AV_V1",
                {"hidden": (hidden, hidden_proof),
                 "context": (context, context_proof)}, report)
            return {"report": report, "artifact": artifact}
    finally:
        torch.set_default_dtype(prior_dtype)


def run_block1_stage2_post(input_path, output_path, device=None):
    device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    source = _load_artifact(input_path, "CORET_SOUND_FP64_BLOCK1_STAGE2_AV_V1")
    prior_dtype = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    try:
        with pinned_zonotope() as Zonotope:
            args = _args(device)
            hidden, hidden_proof = _state_from_payload(
                source["states"]["hidden"], Zonotope, args, device)
            context, context_proof = _state_from_payload(
                source["states"]["context"], Zonotope, args, device)
            checkpoint = prefix._load_checkpoint(); base = "bert.encoder.layer.1"
            output_parameter = _parameter(
                checkpoint, base + ".attention.output.dense", device)
            layernorm = _parameter(
                checkpoint, base + ".attention.output.LayerNorm", device)
            measurements, reductions = [], []
            delegate = structural.StructuralNativeSemanticOperators()
            delegate._layer_norm_index = 3
            dispatch = production.NativeProductionDispatch(delegate=delegate)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            started = time.perf_counter()
            context = context.remove_attention_heads_dim()
            structural.attach_support(context, context_proof)
            attention, attention_proof = _dense_sound(
                context, context_proof, output_parameter,
                "b1_attention_output_affine", measurements)
            attention, attention_proof = _maybe_reduce(
                attention, attention_proof, "b1_attention_output_affine",
                reductions)
            attention, aligned_hidden, residual_proof = _align_states(
                attention, attention_proof, hidden, hidden_proof,
                "b1_attention_residual")
            attention, aligned_hidden, residual_proof = _maybe_reduce_pair(
                attention, aligned_hidden, residual_proof,
                "b1_pre_attention_residual_pair", reductions)
            raw_residual = attention.add(aligned_hidden)
            residual, residual_proof = _inject(
                raw_residual, residual_proof, [attention, aligned_hidden],
                "b1_attention_residual", 1, measurements,
                reserve=_add_reserve(attention, aligned_hidden))
            residual, residual_proof = _maybe_reduce(
                residual, residual_proof, "b1_attention_residual", reductions)
            width = residual.word_embedding_size
            average = torch.ones((width, width), dtype=torch.float64,
                                 device=device) / width
            centered = residual.add(residual.matmul(average).multiply(-1.0))
            variance = centered.square_and_sum_and_repeat().multiply(1.0/width)
            variance_low, variance_high = variance.concretize()
            if float(variance_low.min()) <= 0:
                raise RuntimeError("Block-1 post-attention variance is nonpositive")
            delegate._hidden = residual_proof; delegate._attention_output = residual_proof
            raw_post = dispatch.layer_norm(residual, layernorm, "standard")
            raw_post_proof = structural.get_support(raw_post)
            ops = 16 * 128 * (residual.num_error_terms + 1) ** 2 + 4096
            post, post_proof = _inject(
                raw_post, raw_post_proof, [residual],
                "b1_post_attention_layernorm", ops, measurements,
                reserve=_reserve_from_majorant(
                    _layernorm_majorant(residual, layernorm), ops))
            post, post_proof = _maybe_reduce(
                post, post_proof, "b1_post_attention_layernorm", reductions)
            if device.type == "cuda": torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            report = {
                "measurements": [_state_measurement(
                    "block1_post_attention_layernorm", post, post_proof,
                    seconds)],
                "variance_lower": float(variance_low.min()),
                "variance_upper_min": float(variance_high.min()),
                "injections": measurements, "reductions": reductions,
                "dispatch_counts": dict(dispatch.counts), "seconds": seconds,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda" else 0,
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))
                if device.type == "cuda" else 0,
            }
            artifact = _save_artifact(
                output_path, "CORET_SOUND_FP64_BLOCK1_STAGE2_POST_V1",
                {"post_attention": (post, post_proof)}, report)
            return {"report": report, "artifact": artifact}
    finally:
        torch.set_default_dtype(prior_dtype)


def run_block1_stage2_post_prepare(input_path, output_path, device=None):
    """Head merge, output projection, and residual; stop before LayerNorm."""
    device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    source = _load_artifact(input_path, "CORET_SOUND_FP64_BLOCK1_STAGE2_AV_V1")
    prior_dtype = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    try:
        with pinned_zonotope() as Zonotope:
            args = _args(device)
            hidden, hidden_proof = _state_from_payload(
                source["states"]["hidden"], Zonotope, args, device)
            context, context_proof = _state_from_payload(
                source["states"]["context"], Zonotope, args, device)
            parameter = _parameter(
                prefix._load_checkpoint(),
                "bert.encoder.layer.1.attention.output.dense", device)
            measurements, reductions = [], []
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            started = time.perf_counter()
            context = context.remove_attention_heads_dim()
            structural.attach_support(context, context_proof)
            attention, attention_proof = _dense_sound(
                context, context_proof, parameter,
                "b1_attention_output_affine", measurements)
            attention, attention_proof = _maybe_reduce(
                attention, attention_proof, "b1_attention_output_affine",
                reductions)
            attention, aligned_hidden, residual_proof = _align_states(
                attention, attention_proof, hidden, hidden_proof,
                "b1_attention_residual")
            attention, aligned_hidden, residual_proof = _maybe_reduce_pair(
                attention, aligned_hidden, residual_proof,
                "b1_pre_attention_residual_pair", reductions)
            raw = attention.add(aligned_hidden)
            residual, residual_proof = _inject(
                raw, residual_proof, [attention, aligned_hidden],
                "b1_attention_residual", 1, measurements,
                reserve=_add_reserve(attention, aligned_hidden))
            residual, residual_proof = _maybe_reduce(
                residual, residual_proof, "b1_attention_residual", reductions)
            if device.type == "cuda": torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            report = {
                "measurements": [_state_measurement(
                    "block1_attention_residual", residual, residual_proof,
                    seconds)],
                "injections": measurements, "reductions": reductions,
                "seconds": seconds,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated())
                if device.type == "cuda" else 0,
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())
                if device.type == "cuda" else 0,
            }
            artifact = _save_artifact(
                output_path, "CORET_SOUND_FP64_BLOCK1_POST_RESIDUAL_V1",
                {"residual": (residual, residual_proof)}, report)
            return {"report": report, "artifact": artifact}
    finally:
        torch.set_default_dtype(prior_dtype)


def run_block1_stage2_projection(input_path, output_path, device=None):
    """Head merge and attention output affine only."""
    device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    source = _load_artifact(input_path, "CORET_SOUND_FP64_BLOCK1_STAGE2_AV_V1")
    prior_dtype = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    try:
        with pinned_zonotope() as Zonotope:
            args = _args(device)
            context, context_proof = _state_from_payload(
                source["states"]["context"], Zonotope, args, device)
            parameter = _parameter(
                prefix._load_checkpoint(),
                "bert.encoder.layer.1.attention.output.dense", device)
            measurements, reductions = [], []
            if device.type == "cuda": torch.cuda.synchronize()
            started = time.perf_counter()
            context = context.remove_attention_heads_dim()
            structural.attach_support(context, context_proof)
            attention, attention_proof = _dense_sound(
                context, context_proof, parameter,
                "b1_attention_output_affine", measurements)
            attention, attention_proof = _maybe_reduce(
                attention, attention_proof, "b1_attention_output_affine",
                reductions)
            if device.type == "cuda": torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            report = {
                "measurements": [_state_measurement(
                    "block1_attention_projection", attention,
                    attention_proof, seconds)],
                "injections": measurements, "reductions": reductions,
                "seconds": seconds,
            }
            artifact = _save_artifact(
                output_path, "CORET_SOUND_FP64_BLOCK1_ATTENTION_PROJECTION_V1",
                {"attention": (attention, attention_proof)}, report)
            return {"report": report, "artifact": artifact}
    finally:
        torch.set_default_dtype(prior_dtype)


def run_block1_stage2_residual(av_path, projection_path, output_path,
                               device=None):
    """Align attention projection with the original residual and add."""
    device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    av_source = _load_artifact(av_path, "CORET_SOUND_FP64_BLOCK1_STAGE2_AV_V1")
    projection_source = _load_artifact(
        projection_path, "CORET_SOUND_FP64_BLOCK1_ATTENTION_PROJECTION_V1")
    prior_dtype = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    try:
        with pinned_zonotope() as Zonotope:
            args = _args(device)
            hidden, hidden_proof = _state_from_payload(
                av_source["states"]["hidden"], Zonotope, args, device)
            attention, attention_proof = _state_from_payload(
                projection_source["states"]["attention"], Zonotope, args,
                device)
            measurements, reductions = [], []
            if device.type == "cuda": torch.cuda.synchronize()
            started = time.perf_counter()
            attention, aligned_hidden, residual_proof = _align_states(
                attention, attention_proof, hidden, hidden_proof,
                "b1_attention_residual")
            attention, aligned_hidden, residual_proof = _maybe_reduce_pair(
                attention, aligned_hidden, residual_proof,
                "b1_pre_attention_residual_pair", reductions)
            raw = attention.add(aligned_hidden)
            residual, residual_proof = _inject(
                raw, residual_proof, [attention, aligned_hidden],
                "b1_attention_residual", 1, measurements,
                reserve=_add_reserve(attention, aligned_hidden))
            residual, residual_proof = _maybe_reduce(
                residual, residual_proof, "b1_attention_residual", reductions)
            if device.type == "cuda": torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            report = {
                "measurements": [_state_measurement(
                    "block1_attention_residual", residual, residual_proof,
                    seconds)],
                "injections": measurements, "reductions": reductions,
                "seconds": seconds,
            }
            artifact = _save_artifact(
                output_path, "CORET_SOUND_FP64_BLOCK1_POST_RESIDUAL_V1",
                {"residual": (residual, residual_proof)}, report)
            return {"report": report, "artifact": artifact}
    finally:
        torch.set_default_dtype(prior_dtype)


def run_block1_stage2_residual_prepare(av_path, projection_path, output_path,
                                       device=None):
    """Persist the exact aligned/reduced operands of the attention residual."""
    device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    av_source = _load_artifact(av_path, "CORET_SOUND_FP64_BLOCK1_STAGE2_AV_V1")
    projection_source = _load_artifact(
        projection_path, "CORET_SOUND_FP64_BLOCK1_ATTENTION_PROJECTION_V1")
    prior_dtype = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    try:
        with pinned_zonotope() as Zonotope:
            args = _args(device)
            hidden, hidden_proof = _state_from_payload(
                av_source["states"]["hidden"], Zonotope, args, device)
            attention, attention_proof = _state_from_payload(
                projection_source["states"]["attention"], Zonotope, args,
                device)
            reductions = []
            if device.type == "cuda": torch.cuda.synchronize()
            started = time.perf_counter()
            attention, hidden, proof = _align_states(
                attention, attention_proof, hidden, hidden_proof,
                "b1_attention_residual")
            attention, hidden, proof = _maybe_reduce_pair(
                attention, hidden, proof,
                "b1_pre_attention_residual_pair", reductions)
            if device.type == "cuda": torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            report = {"reductions": reductions, "seconds": seconds}
            artifact = _save_artifact(
                output_path, "CORET_SOUND_FP64_BLOCK1_RESIDUAL_INPUT_V1",
                {"attention": (attention, proof), "hidden": (hidden, proof)},
                report)
            return {"report": report, "artifact": artifact}
    finally:
        torch.set_default_dtype(prior_dtype)


def run_block1_stage2_residual_add(input_path, output_path, device=None):
    """Add previously aligned attention/residual operands."""
    device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    source = _load_artifact(
        input_path, "CORET_SOUND_FP64_BLOCK1_RESIDUAL_INPUT_V1")
    prior_dtype = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    try:
        with pinned_zonotope() as Zonotope:
            args = _args(device)
            attention, proof = _state_from_payload(
                source["states"]["attention"], Zonotope, args, device)
            hidden, hidden_proof = _state_from_payload(
                source["states"]["hidden"], Zonotope, args, device)
            if proof != hidden_proof:
                raise RuntimeError("attention residual input proofs differ")
            measurements, reductions = [], []
            if device.type == "cuda": torch.cuda.synchronize()
            started = time.perf_counter()
            raw = attention.add(hidden)
            residual, residual_proof = _inject(
                raw, proof, [attention, hidden], "b1_attention_residual", 1,
                measurements, reserve=_add_reserve(attention, hidden))
            residual, residual_proof = _maybe_reduce(
                residual, residual_proof, "b1_attention_residual", reductions)
            if device.type == "cuda": torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            report = {
                "measurements": [_state_measurement(
                    "block1_attention_residual", residual, residual_proof,
                    seconds)],
                "injections": measurements, "reductions": reductions,
                "seconds": seconds,
            }
            artifact = _save_artifact(
                output_path, "CORET_SOUND_FP64_BLOCK1_POST_RESIDUAL_V1",
                {"residual": (residual, residual_proof)}, report)
            return {"report": report, "artifact": artifact}
    finally:
        torch.set_default_dtype(prior_dtype)


def run_block1_stage2_post_ln(input_path, output_path, device=None):
    """First Block-1 LayerNorm from a persisted attention residual."""
    device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    source = _load_artifact(
        input_path, "CORET_SOUND_FP64_BLOCK1_POST_RESIDUAL_V1")
    prior_dtype = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    try:
        with pinned_zonotope() as Zonotope:
            args = _args(device)
            residual, residual_proof = _state_from_payload(
                source["states"]["residual"], Zonotope, args, device)
            parameter = _parameter(
                prefix._load_checkpoint(),
                "bert.encoder.layer.1.attention.output.LayerNorm", device)
            delegate = structural.StructuralNativeSemanticOperators()
            delegate._layer_norm_index = 3
            delegate._hidden = residual_proof
            delegate._attention_output = residual_proof
            dispatch = production.NativeProductionDispatch(delegate=delegate)
            measurements, reductions, oracles = [], [], []
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            started = time.perf_counter()
            width = residual.word_embedding_size
            average = torch.ones((width, width), dtype=torch.float64,
                                 device=device) / width
            centered = residual.add(residual.matmul(average).multiply(-1.0))
            variance = centered.square_and_sum_and_repeat().multiply(1.0/width)
            variance_low, variance_high = variance.concretize()
            prepared = _prepare_layernorm_separator(
                residual, residual_proof, variance_low, "block1_post_attention")
            if prepared is not None and not prepared["admissible"]:
                raise RuntimeError("Block-1 post-attention variance is nonpositive")
            raw, raw_proof, separator_reserve = _layernorm_sound_raw(
                dispatch, residual, residual_proof, parameter, "block1_post_attention",
                variance_low, prepared)
            variance_low = _checked_layernorm_low(variance_low, prepared)
            ops = 16 * 128 * (residual.num_error_terms + 1) ** 2 + 4096
            post, post_proof = _inject(
                raw, raw_proof, [residual], "b1_post_attention_layernorm",
                ops, measurements, reserve=(separator_reserve if separator_reserve is not None
                    else _reserve_from_majorant(_layernorm_majorant(residual, parameter), ops)))
            post, post_proof = _maybe_reduce(
                post, post_proof, "b1_post_attention_layernorm", reductions)
            reserve = max(item["maximum_local_widening"]
                          for item in measurements
                          if item["label"] == "b1_post_attention_layernorm")
            values = centered.zonotope_w[0, 0].detach().cpu().tolist()
            with _mp_context(gmpy2.RoundToNearest):
                exact_variance = sum(
                    (_mp(item) * _mp(item) for item in values), _mp(0)
                ) / len(values)
            machine_variance = (
                centered.zonotope_w[0, 0].square().sum() / len(values))
            oracles.append(_oracle_containment(
                "block1_post_attention_layernorm_variance_nominal",
                exact_variance, machine_variance, reserve))
            low_value = variance_low[0, 0] + 1e-12
            with _mp_context(gmpy2.RoundToNearest):
                exact_root = gmpy2.sqrt(_mp(low_value))
            oracles.append(_oracle_containment(
                "block1_post_attention_layernorm_sqrt_lower", exact_root,
                torch.sqrt(low_value), reserve))
            if device.type == "cuda": torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            report = {
                "measurements": [_state_measurement(
                    "block1_post_attention_layernorm", post, post_proof,
                    seconds)],
                "variance_lower": float(variance_low.min()),
                "variance_upper_min": float(variance_high.min()),
                "branch_separation": {
                    "sqrt_positive": float(variance_low.min()) > 0,
                    "minimum_variance_lower": float(variance_low.min()),
                },
                "injections": measurements, "reductions": reductions,
                "mpfr_spots": oracles,
                "dispatch_counts": dict(dispatch.counts), "seconds": seconds,
                "generic_fallback_count": dispatch.generic_family_invocations,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated())
                if device.type == "cuda" else 0,
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())
                if device.type == "cuda" else 0,
            }
            artifact = _save_artifact(
                output_path, "CORET_SOUND_FP64_BLOCK1_STAGE2_POST_V1",
                {"post_attention": (post, post_proof)},
                {**report, **({"_separating_variance_witnesses": dispatch.separating_variance_witnesses}
                             if hasattr(dispatch, "separating_variance_witnesses") else {})})
            return {"report": report, "artifact": artifact}
    finally:
        torch.set_default_dtype(prior_dtype)


def run_block1_stage3_ffn1(input_path, output_path, device=None):
    """Block-1 first FFN affine and ReLU."""
    device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    source = _load_artifact(
        input_path, "CORET_SOUND_FP64_BLOCK1_STAGE2_POST_V1")
    prior_dtype = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    try:
        with pinned_zonotope() as Zonotope:
            args = _args(device)
            post, post_proof = _state_from_payload(
                source["states"]["post_attention"], Zonotope, args, device)
            parameter = _parameter(
                prefix._load_checkpoint(),
                "bert.encoder.layer.1.intermediate.dense", device)
            delegate = structural.StructuralNativeSemanticOperators()
            delegate._relu_index = 1
            dispatch = production.NativeProductionDispatch(delegate=delegate)
            measurements, reductions = [], []
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            started = time.perf_counter()
            affine, affine_proof = _dense_sound(
                post, post_proof, parameter, "b1_ffn_first", measurements)
            affine, affine_proof = _maybe_reduce(
                affine, affine_proof, "b1_ffn_first", reductions)
            lower, upper = affine.concretize()
            active = lower >= 0
            inactive = upper <= 0
            crossing = ~(active | inactive)
            delegate._post_attention = affine_proof
            raw = dispatch.relu(affine)
            raw_proof = structural.get_support(raw)
            ops = 8 * (affine.num_error_terms + 1) + 128
            relu, relu_proof = _inject(
                raw, raw_proof, [affine], "b1_relu", ops, measurements,
                condition=2.0)
            relu, relu_proof = _maybe_reduce(
                relu, relu_proof, "b1_relu", reductions)
            if device.type == "cuda": torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            branch = {
                "active": int(active.sum()), "inactive": int(inactive.sum()),
                "crossing": int(crossing.sum()),
                "minimum_active_lower": (float(lower[active].min())
                    if bool(active.any()) else None),
                "maximum_inactive_upper": (float(upper[inactive].max())
                    if bool(inactive.any()) else None),
            }
            report = {
                "measurements": [_state_measurement(
                    "block1_relu", relu, relu_proof, seconds)],
                "branch_separation": branch,
                "injections": measurements, "reductions": reductions,
                "dispatch_counts": dict(dispatch.counts), "seconds": seconds,
                "generic_fallback_count": dispatch.generic_family_invocations,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated())
                if device.type == "cuda" else 0,
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())
                if device.type == "cuda" else 0,
            }
            artifact = _save_artifact(
                output_path, "CORET_SOUND_FP64_BLOCK1_FFN1_V1",
                {"relu": (relu, relu_proof)}, report)
            return {"report": report, "artifact": artifact}
    finally:
        torch.set_default_dtype(prior_dtype)


def run_block1_stage3_ffn2(input_path, output_path, device=None):
    """Block-1 second FFN affine."""
    device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    source = _load_artifact(input_path, "CORET_SOUND_FP64_BLOCK1_FFN1_V1")
    prior_dtype = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    try:
        with pinned_zonotope() as Zonotope:
            args = _args(device)
            relu, relu_proof = _state_from_payload(
                source["states"]["relu"], Zonotope, args, device)
            parameter = _parameter(
                prefix._load_checkpoint(),
                "bert.encoder.layer.1.output.dense", device)
            measurements, reductions = [], []
            if device.type == "cuda": torch.cuda.synchronize()
            started = time.perf_counter()
            ffn, ffn_proof = _dense_sound(
                relu, relu_proof, parameter, "b1_ffn_second", measurements)
            ffn, ffn_proof = _maybe_reduce(
                ffn, ffn_proof, "b1_ffn_second", reductions)
            if device.type == "cuda": torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            report = {
                "measurements": [_state_measurement(
                    "block1_ffn_output", ffn, ffn_proof, seconds)],
                "injections": measurements, "reductions": reductions,
                "seconds": seconds,
            }
            artifact = _save_artifact(
                output_path, "CORET_SOUND_FP64_BLOCK1_FFN2_V1",
                {"ffn": (ffn, ffn_proof)}, report)
            return {"report": report, "artifact": artifact}
    finally:
        torch.set_default_dtype(prior_dtype)


def run_block1_stage3_residual_prepare(post_path, ffn_path, output_path,
                                       device=None):
    """Align/reduce the FFN output and its post-attention residual."""
    device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    post_source = _load_artifact(
        post_path, "CORET_SOUND_FP64_BLOCK1_STAGE2_POST_V1")
    ffn_source = _load_artifact(ffn_path, "CORET_SOUND_FP64_BLOCK1_FFN2_V1")
    prior_dtype = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    try:
        with pinned_zonotope() as Zonotope:
            args = _args(device)
            post, post_proof = _state_from_payload(
                post_source["states"]["post_attention"], Zonotope, args,
                device)
            ffn, ffn_proof = _state_from_payload(
                ffn_source["states"]["ffn"], Zonotope, args, device)
            reductions = []
            if device.type == "cuda": torch.cuda.synchronize()
            started = time.perf_counter()
            ffn, post, proof = _align_states(
                ffn, ffn_proof, post, post_proof, "b1_ffn_residual")
            ffn, post, proof = _maybe_reduce_pair(
                ffn, post, proof, "b1_pre_ffn_residual_pair", reductions)
            if device.type == "cuda": torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            report = {"reductions": reductions, "seconds": seconds}
            artifact = _save_artifact(
                output_path, "CORET_SOUND_FP64_BLOCK1_FFN_RESIDUAL_INPUT_V1",
                {"ffn": (ffn, proof), "post": (post, proof)}, report)
            return {"report": report, "artifact": artifact}
    finally:
        torch.set_default_dtype(prior_dtype)


def run_block1_stage3_residual_add(input_path, output_path, device=None):
    """Add aligned FFN/residual branches."""
    device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    source = _load_artifact(
        input_path, "CORET_SOUND_FP64_BLOCK1_FFN_RESIDUAL_INPUT_V1")
    prior_dtype = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    try:
        with pinned_zonotope() as Zonotope:
            args = _args(device)
            ffn, proof = _state_from_payload(
                source["states"]["ffn"], Zonotope, args, device)
            post, post_proof = _state_from_payload(
                source["states"]["post"], Zonotope, args, device)
            if proof != post_proof:
                raise RuntimeError("FFN residual input proofs differ")
            measurements, reductions = [], []
            if device.type == "cuda": torch.cuda.synchronize()
            started = time.perf_counter()
            raw = ffn.add(post)
            residual, residual_proof = _inject(
                raw, proof, [ffn, post], "b1_ffn_residual", 1,
                measurements, reserve=_add_reserve(ffn, post))
            residual, residual_proof = _maybe_reduce(
                residual, residual_proof, "b1_ffn_residual", reductions)
            if device.type == "cuda": torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            report = {
                "measurements": [_state_measurement(
                    "block1_ffn_residual", residual, residual_proof, seconds)],
                "injections": measurements, "reductions": reductions,
                "seconds": seconds,
            }
            artifact = _save_artifact(
                output_path, "CORET_SOUND_FP64_BLOCK1_OUTPUT_RESIDUAL_V1",
                {"residual": (residual, residual_proof)}, report)
            return {"report": report, "artifact": artifact}
    finally:
        torch.set_default_dtype(prior_dtype)


def run_block1_stage3_final_ln(input_path, output_path, device=None):
    """Final Block-1 LayerNorm plus the actual pre-Block-2 recenter/reduction."""
    device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    source = _load_artifact(
        input_path, "CORET_SOUND_FP64_BLOCK1_OUTPUT_RESIDUAL_V1")
    prior_dtype = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    try:
        with pinned_zonotope() as Zonotope:
            args = _args(device)
            residual, residual_proof = _state_from_payload(
                source["states"]["residual"], Zonotope, args, device)
            parameter = _parameter(
                prefix._load_checkpoint(),
                "bert.encoder.layer.1.output.LayerNorm", device)
            delegate = structural.StructuralNativeSemanticOperators()
            delegate._layer_norm_index = 4
            delegate._post_attention = residual_proof
            delegate._relu = residual_proof
            dispatch = production.NativeProductionDispatch(delegate=delegate)
            measurements, reductions, oracles = [], [], []
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            started = time.perf_counter()
            width = residual.word_embedding_size
            average = torch.ones((width, width), dtype=torch.float64,
                                 device=device) / width
            centered = residual.add(residual.matmul(average).multiply(-1.0))
            variance = centered.square_and_sum_and_repeat().multiply(1.0/width)
            variance_low, variance_high = variance.concretize()
            prepared = _prepare_layernorm_separator(
                residual, residual_proof, variance_low, "block1_output")
            if prepared is not None and not prepared["admissible"]:
                raise RuntimeError("Block-1 output variance is nonpositive")
            raw, raw_proof, separator_reserve = _layernorm_sound_raw(
                dispatch, residual, residual_proof, parameter, "block1_output", variance_low, prepared)
            variance_low = _checked_layernorm_low(variance_low, prepared)
            ops = 16 * 128 * (residual.num_error_terms + 1) ** 2 + 4096
            output, output_proof = _inject(
                raw, raw_proof, [residual], "b1_output_layernorm", ops,
                measurements, reserve=(separator_reserve if separator_reserve is not None
                    else _reserve_from_majorant(_layernorm_majorant(residual, parameter), ops)))
            output, output_proof = _maybe_reduce(
                output, output_proof, "b1_output_layernorm", reductions)

            # Actual production begins Block 2 with this conditional recenter.
            output, output_proof, recenter = _recenter_sound(
                output, output_proof, "b2_pre_qk_recenter", measurements,
                reductions)
            output, output_proof = _maybe_reduce(
                output, output_proof, "b2_pre_qk_reduction", reductions)

            values = centered.zonotope_w[0, 0].detach().cpu().tolist()
            with _mp_context(gmpy2.RoundToNearest):
                exact = sum((_mp(item) * _mp(item) for item in values),
                            _mp(0)) / len(values)
            machine = centered.zonotope_w[0, 0].square().sum() / len(values)
            reserve = max(item["maximum_local_widening"]
                          for item in measurements
                          if item["label"] == "b1_output_layernorm")
            oracles.append(_oracle_containment(
                "block1_actual_layernorm_variance_nominal", exact, machine,
                reserve))
            low_value = variance_low[0, 0] + 1e-12
            with _mp_context(gmpy2.RoundToNearest):
                exact_root = gmpy2.sqrt(_mp(low_value))
            oracles.append(_oracle_containment(
                "block1_actual_layernorm_sqrt_lower", exact_root,
                torch.sqrt(low_value), reserve))
            if device.type == "cuda": torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            report = {
                "measurements": [_state_measurement(
                    "block1_final_layernorm", output, output_proof, seconds)],
                "variance_lower": float(variance_low.min()),
                "variance_upper_min": float(variance_high.min()),
                "branch_separation": {
                    "sqrt_positive": float(variance_low.min()) > 0,
                    "minimum_variance_lower": float(variance_low.min()),
                    "recenter_executed": recenter["executed"],
                },
                "recenter": recenter,
                "mpfr_spots": oracles, "injections": measurements,
                "reductions": reductions,
                "dispatch_counts": dict(dispatch.counts), "seconds": seconds,
                "generic_fallback_count": dispatch.generic_family_invocations,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated())
                if device.type == "cuda" else 0,
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())
                if device.type == "cuda" else 0,
            }
            artifact = _save_artifact(
                output_path, "CORET_SOUND_FP64_BLOCK1_FINAL_V1",
                {"pre_block2": (output, output_proof)},
                {**report, **({"_separating_variance_witnesses": dispatch.separating_variance_witnesses}
                             if hasattr(dispatch, "separating_variance_witnesses") else {})})
            return {"report": report, "artifact": artifact}
    finally:
        torch.set_default_dtype(prior_dtype)


def run_reduction_audit_from_artifact(path, device=None):
    device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("schema") != "CORET_SOUND_FP64_BLOCK0_STATE_V1"
            or payload.get("pinned_revision") != PINNED_REVISION):
        raise RuntimeError("sound Block-0 reduction artifact identity differs")
    prior_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        with pinned_zonotope() as Zonotope:
            args = _args(device)
            state = Zonotope(
                args=args, p=100, eps=prefix.FIXTURE_RHO,
                perturbed_word_index=prefix.FIXTURE_PERTURBED_TOKEN,
                zonotope_w=payload["weights"].to(device),
                error_term_range_low=payload["range_low"].to(device),
                error_term_range_high=payload["range_high"].to(device),
                clone=False)
            raw = payload["proof"]
            proof = structural.SupportProof(
                tuple(raw["masks"]), tuple(raw["ids"]),
                tuple(raw["reasons"]), int(raw["num_tokens"]))
            structural.attach_support(state, proof)
            reports = []
            mutation_result = None
            for cap in (12000, 10000):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                reduced, reduced_proof, witness = sound_reduce(
                    state, proof, cap, f"forced_cap_{cap}")
                checked = check_reduction_witness(
                    state, proof, reduced, reduced_proof, witness)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                if mutation_result is None:
                    mutation_result = _reduction_mutations(
                        state, proof, reduced, reduced_proof, witness)
                low, high = reduced.concretize()
                reports.append({
                    "cap": cap, "generators_before": state.num_error_terms,
                    "retained": len(witness["retained_indices"]),
                    "dropped": len(witness["dropped_indices"]),
                    "new_box_generators": witness["replacement_count"],
                    "generators_after": reduced.num_error_terms,
                    "protected": len(witness["protected_indices"]),
                    "support_inflation": checked["support_inflation"],
                    "range_inflation": checked["range_inflation"],
                    "support_max": float((0.5*(high-low)).max()),
                    "lower_min": float(low.min()),
                    "upper_max": float(high.max()), "seconds": elapsed,
                    "allocated_bytes": int(torch.cuda.memory_allocated(device))
                    if device.type == "cuda" else 0,
                    "reserved_bytes": int(torch.cuda.memory_reserved(device))
                    if device.type == "cuda" else 0,
                    "witness": witness, "checker": checked,
                })
            return {"forced_caps": reports, "mutations": mutation_result}
    finally:
        torch.set_default_dtype(prior_dtype)


def run_sound_fp64(device=None, continuation=None,
                   run_representative_mpfr=True):
    """Run Block 0 with every FP64 reserve embedded in the abstract state."""
    device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    prior_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        with pinned_zonotope() as Zonotope:
            checkpoint = prefix._load_checkpoint()
            token_ids = torch.tensor(prefix.FIXTURE_TOKEN_IDS, dtype=torch.long)
            positions = torch.arange(len(prefix.FIXTURE_TOKEN_IDS),
                                     dtype=torch.long)
            token_types = torch.zeros_like(token_ids)
            word = checkpoint["bert.embeddings.word_embeddings.weight"][
                token_ids].to(device=device, dtype=torch.float64)
            position = checkpoint[
                "bert.embeddings.position_embeddings.weight"][positions].to(
                    device=device, dtype=torch.float64)
            token_type = checkpoint[
                "bert.embeddings.token_type_embeddings.weight"][token_types].to(
                    device=device, dtype=torch.float64)
            pre = (word + position) + token_type
            args = _args(device)
            z = Zonotope(args=args, p=100, eps=prefix.FIXTURE_RHO,
                         perturbed_word_index=prefix.FIXTURE_PERTURBED_TOKEN,
                         value=pre)
            proof = structural.proof_from_masks(
                [structural.local_mask(prefix.FIXTURE_PERTURBED_TOKEN)] * 128,
                len(prefix.FIXTURE_TOKEN_IDS), "input_source")
            structural.attach_support(z, proof)
            measurements, reductions = [], []
            embedding_radius = _outward_positive(
                (word.abs() + position.abs() + token_type.abs())
                * (2.0 * _gamma(2) + FP64_U))
            z, proof = _inject(
                z, proof, [], "embedding_sum", 2, measurements,
                reserve=embedding_radius)
            delegate = structural.StructuralNativeSemanticOperators()
            delegate._hidden = proof
            dispatch = production.NativeProductionDispatch(delegate=delegate)
            rows = [_metrics("sound_source", z, 0.0)]
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            total_started = time.perf_counter()
            last_checkpoint = total_started

            def record(label, state):
                nonlocal last_checkpoint
                if device.type == "cuda":
                    torch.cuda.synchronize()
                now = time.perf_counter()
                rows.append(_metrics(label, state, now - last_checkpoint))
                last_checkpoint = now

            embedding_ln = _parameter(
                checkpoint, "bert.embeddings.LayerNorm", device)
            raw, raw_proof, ln_reserve = _layernorm_sound_raw(
                dispatch, z, proof, embedding_ln, "embedding_layernorm")
            ln_ops = 16 * 128 * (z.num_error_terms + 1) ** 2 + 4096
            if ln_reserve is None:
                ln_reserve = _reserve_from_majorant(_layernorm_majorant(z, embedding_ln), ln_ops)
            z, proof = _inject(
                raw, raw_proof, [z], "embedding_layernorm", ln_ops,
                measurements, reserve=ln_reserve)
            delegate._hidden = proof
            record("embedding_layernorm", z)
            z = production._recenter_native_ranges(z)
            structural.attach_support(z, proof)
            if z.num_error_terms <= MAXIMUM_GENERATORS:
                # Preserve the accepted native no-op transition and its
                # dispatch evidence.  Oversized production states instead use
                # the established sound-FP64 replacement-box reduction.
                z = dispatch.reduce(z, MAXIMUM_GENERATORS)
                proof = structural.get_support(z)
            else:
                z, proof = _maybe_reduce(
                    z, proof, "block0_pre_qk_reduction", reductions)
            delegate._hidden = proof
            residual, residual_proof = z, proof

            base = "bert.encoder.layer.0"
            parameters = {
                "query": _parameter(
                    checkpoint, base + ".attention.self.query", device),
                "key": _parameter(
                    checkpoint, base + ".attention.self.key", device),
                "value": _parameter(
                    checkpoint, base + ".attention.self.value", device),
                "attention_output": _parameter(
                    checkpoint, base + ".attention.output.dense", device),
                "attention_ln": _parameter(
                    checkpoint, base + ".attention.output.LayerNorm", device),
                "ffn_first": _parameter(
                    checkpoint, base + ".intermediate.dense", device),
                "ffn_second": _parameter(
                    checkpoint, base + ".output.dense", device),
                "output_ln": _parameter(
                    checkpoint, base + ".output.LayerNorm", device),
            }

            q, q_proof = _dense_sound(
                z, proof, parameters["query"], "q_affine", measurements)
            k, k_proof = _dense_sound(
                z, proof, parameters["key"], "k_affine", measurements)
            q, k, qk_input_proof = _align_states(
                q, q_proof, k, k_proof, "qk_branches")
            q = q.add_attention_heads_dim(4)
            k = k.add_attention_heads_dim(4)
            structural.attach_support(q, qk_input_proof)
            structural.attach_support(k, qk_input_proof)
            q, k, qk_input_proof = _maybe_reduce_pair(
                q, k, qk_input_proof, "block0_pre_qk_pair", reductions)
            delegate._hidden = qk_input_proof
            raw_qk = dispatch.qk(q, k)
            raw_qk_proof = structural.get_support(raw_qk)
            qk_ops = (32 * (q.num_error_terms + 1) ** 2 * 16 + 4096)
            qk_reserve = _reserve_from_majorant(
                _bilinear_majorant(q, k), qk_ops)
            qk, qk_proof = _inject(
                raw_qk, raw_qk_proof, [q, k], "qk", qk_ops,
                measurements, reserve=qk_reserve)
            qk, qk_proof = _maybe_reduce(
                qk, qk_proof, "block0_qk", reductions)
            delegate._score = qk_proof
            record("qk", qk)

            scale = 1.0 / math.sqrt(32)
            scores = qk.multiply(scale)
            structural.attach_support(scores, qk_proof)
            score_reserve = _outward_positive(
                _absolute_hull(qk) * abs(scale)
                * (2.0 * _gamma(1) + FP64_U))
            scores, score_proof = _inject(
                scores, qk_proof, [qk], "score_scaling", 1, measurements,
                reserve=score_reserve)
            scores, score_proof = _maybe_reduce(
                scores, score_proof, "block0_score_scaling", reductions)
            delegate._score = score_proof
            raw_probability = dispatch.softmax(scores, no_constraints=False)
            raw_probability_proof = structural.get_support(raw_probability)
            softmax_ops = 64 * (scores.num_error_terms + 1) + 8192
            softmax_reserve = _reserve_from_majorant(
                _softmax_majorant(scores), softmax_ops)
            probability, probability_proof = _inject(
                raw_probability, raw_probability_proof, [scores], "softmax",
                softmax_ops, measurements, reserve=softmax_reserve)
            probability, probability_proof = _maybe_reduce(
                probability, probability_proof, "block0_softmax", reductions)
            delegate._probability = probability_proof
            record("softmax", probability)

            value, value_proof = _dense_sound(
                z, proof, parameters["value"], "v_affine", measurements)
            value = value.add_attention_heads_dim(4)
            structural.attach_support(value, value_proof)
            probability, value, av_input_proof = _align_states(
                probability, probability_proof, value, value_proof,
                "attention_value_branches")
            probability, value, av_input_proof = _maybe_reduce_pair(
                probability, value, av_input_proof,
                "block0_pre_av_pair", reductions)
            delegate._probability = av_input_proof
            delegate._value = av_input_proof
            raw_context = dispatch.attention_value(probability, value)
            raw_context_proof = structural.get_support(raw_context)
            av_ops = (32 * (max(probability.num_error_terms,
                                value.num_error_terms) + 1) ** 2 * 16 + 4096)
            av_reserve = _reserve_from_majorant(
                _bilinear_majorant(probability, value.t()), av_ops)
            context, context_proof = _inject(
                raw_context, raw_context_proof, [probability, value],
                "attention_value", av_ops, measurements, reserve=av_reserve)
            context, context_proof = _maybe_reduce(
                context, context_proof, "block0_attention_value", reductions)
            delegate._attention_output = context_proof
            record("attention_value", context)

            context = context.remove_attention_heads_dim()
            structural.attach_support(context, context_proof)
            attention, attention_proof = _dense_sound(
                context, context_proof, parameters["attention_output"],
                "attention_output_affine", measurements)
            attention, attention_proof = _maybe_reduce(
                attention, attention_proof, "block0_attention_output_affine",
                reductions)
            attention, aligned_residual, residual_union = _align_states(
                attention, attention_proof, residual, residual_proof,
                "attention_residual")
            attention, aligned_residual, residual_union = _maybe_reduce_pair(
                attention, aligned_residual, residual_union,
                "block0_pre_attention_residual_pair", reductions)
            raw_residual = attention.add(aligned_residual)
            raw_residual_reserve = _add_reserve(attention, aligned_residual)
            attention_residual, attention_residual_proof = _inject(
                raw_residual, residual_union,
                [attention, aligned_residual], "attention_residual", 1,
                measurements, reserve=raw_residual_reserve)
            attention_residual, attention_residual_proof = _maybe_reduce(
                attention_residual, attention_residual_proof,
                "block0_attention_residual", reductions)
            delegate._hidden = attention_residual_proof
            delegate._attention_output = attention_residual_proof
            raw_post, raw_post_proof, post_ln_reserve = _layernorm_sound_raw(
                dispatch, attention_residual, attention_residual_proof,
                parameters["attention_ln"], "block0_post_attention")
            ln_ops = (16 * 128 *
                      (attention_residual.num_error_terms + 1) ** 2 + 4096)
            if post_ln_reserve is None:
                post_ln_reserve = _reserve_from_majorant(
                    _layernorm_majorant(attention_residual, parameters["attention_ln"]), ln_ops)
            post, post_proof = _inject(
                raw_post, raw_post_proof, [attention_residual],
                "post_attention_layernorm", ln_ops, measurements,
                reserve=post_ln_reserve)
            post, post_proof = _maybe_reduce(
                post, post_proof, "block0_post_attention_layernorm",
                reductions)
            delegate._post_attention = post_proof
            record("post_attention_layernorm", post)

            ffn_first, ffn_first_proof = _dense_sound(
                post, post_proof, parameters["ffn_first"], "ffn_first",
                measurements)
            ffn_first, ffn_first_proof = _maybe_reduce(
                ffn_first, ffn_first_proof, "block0_ffn_first", reductions)
            delegate._post_attention = ffn_first_proof
            raw_relu = dispatch.relu(ffn_first)
            raw_relu_proof = structural.get_support(raw_relu)
            relu_ops = 8 * (ffn_first.num_error_terms + 1) + 128
            relu, relu_proof = _inject(
                raw_relu, raw_relu_proof, [ffn_first], "relu", relu_ops,
                measurements, condition=2.0)
            relu, relu_proof = _maybe_reduce(
                relu, relu_proof, "block0_relu", reductions)
            delegate._relu = relu_proof
            ffn, ffn_proof = _dense_sound(
                relu, relu_proof, parameters["ffn_second"], "ffn_second",
                measurements)
            ffn, ffn_proof = _maybe_reduce(
                ffn, ffn_proof, "block0_ffn_second", reductions)
            record("ffn_output", ffn)

            ffn, aligned_post, ffn_union = _align_states(
                ffn, ffn_proof, post, post_proof, "ffn_residual")
            ffn, aligned_post, ffn_union = _maybe_reduce_pair(
                ffn, aligned_post, ffn_union,
                "block0_pre_ffn_residual_pair", reductions)
            raw_ffn_residual = ffn.add(aligned_post)
            ffn_reserve = _add_reserve(ffn, aligned_post)
            ffn_residual, ffn_residual_proof = _inject(
                raw_ffn_residual, ffn_union, [ffn, aligned_post],
                "ffn_residual", 1, measurements, reserve=ffn_reserve)
            ffn_residual, ffn_residual_proof = _maybe_reduce(
                ffn_residual, ffn_residual_proof,
                "block0_ffn_residual", reductions)
            delegate._post_attention = ffn_residual_proof
            delegate._relu = ffn_residual_proof

            width = ffn_residual.word_embedding_size
            average = torch.ones((width, width), device=device,
                                 dtype=torch.float64) / width
            centered = ffn_residual.add(
                ffn_residual.matmul(average).multiply(-1.0))
            variance = centered.square_and_sum_and_repeat().multiply(1.0 / width)
            variance_low, variance_high = variance.concretize()
            prepared = _prepare_layernorm_separator(
                ffn_residual, ffn_residual_proof, variance_low, "block0_output")
            raw_output, raw_output_proof, output_ln_reserve = _layernorm_sound_raw(
                dispatch, ffn_residual, ffn_residual_proof, parameters["output_ln"],
                "block0_output", variance_low, prepared)
            variance_low = _checked_layernorm_low(variance_low, prepared)
            ln_ops = (16 * 128 *
                      (ffn_residual.num_error_terms + 1) ** 2 + 4096)
            if output_ln_reserve is None:
                output_ln_reserve = _reserve_from_majorant(
                    _layernorm_majorant(ffn_residual, parameters["output_ln"]), ln_ops)
            output, output_proof = _inject(
                raw_output, raw_output_proof, [ffn_residual],
                "output_layernorm", ln_ops, measurements,
                reserve=output_ln_reserve)
            output, output_proof = _maybe_reduce(
                output, output_proof, "block0_output_layernorm", reductions)
            record("output_layernorm", output)
            total_seconds = time.perf_counter() - total_started
            reserves = {
                item["label"]: item["maximum_local_widening"]
                for item in measurements}
            mpfr_spots = []
            if run_representative_mpfr:
                mpfr_spots = _mpfr_spots(checkpoint)
                mpfr_spots.extend(_runtime_mpfr_spots(
                    hidden=z, query_parameter=parameters["query"], q=q, k=k,
                    raw_qk=raw_qk, scores=scores,
                    raw_probability=probability, value=value,
                    raw_context=raw_context, ffn_residual=ffn_residual,
                    centered=centered, variance=variance, reserves=reserves))
            result = {
                "rows": rows,
                "numerical_injections": measurements,
                "reductions": reductions,
                "second_layernorm_variance_lower": float(variance_low.min()),
                "second_layernorm_variance_upper_min": float(
                    variance_high.min()),
                "dispatch_counts": dict(dispatch.counts),
                "dtype": str(output.zonotope_w.dtype),
                "mpfr_spots": mpfr_spots,
                "representative_mpfr_checks_performed":
                    run_representative_mpfr,
                "final_proof_generator_count": len(output_proof.ids),
                "total_seconds": total_seconds,
                "peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda" else 0),
                "peak_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device)
                    if device.type == "cuda" else 0),
            }
            if continuation is not None:
                result["continuation"] = continuation(
                    output, output_proof, {
                        "checkpoint": checkpoint, "args": args,
                        "Zonotope": Zonotope, "device": device,
                        **({"separating_variance_witnesses": dispatch.separating_variance_witnesses}
                           if hasattr(dispatch, "separating_variance_witnesses") else {}),
                    })
            return result
    finally:
        torch.set_default_dtype(prior_dtype)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "plain"
    if mode == "sound":
        result = run_sound_fp64()
    elif mode == "reduction":
        result = run_reduction_audit()
    elif mode == "export":
        result = export_block0_state(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/coret_fp64_block0.pt")
    elif mode == "reduction-artifact":
        result = run_reduction_audit_from_artifact(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/coret_fp64_block0.pt")
    elif mode == "block1-stage1":
        result = run_block1_stage1(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/coret_fp64_block0.pt",
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/coret_fp64_b1_stage1.pt")
    elif mode == "block1-pre-qk":
        result = run_block1_pre_qk(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/coret_fp64_block0.pt",
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/coret_fp64_b1_pre_qk.pt")
    elif mode == "block1-qk-prepare":
        result = run_block1_qk_prepare(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/coret_fp64_b1_pre_qk.pt",
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/coret_fp64_b1_qk_input.pt")
    elif mode == "block1-qk-softmax":
        result = run_block1_qk_softmax(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/coret_fp64_b1_qk_input.pt",
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/coret_fp64_b1_stage1.pt")
    elif mode == "block1-qk-only":
        result = run_block1_qk_only(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/coret_fp64_b1_qk_input.pt",
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/coret_fp64_b1_qk.pt")
    elif mode == "block1-qk-compute":
        result = run_block1_qk_compute(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/coret_fp64_b1_qk_input.pt",
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/coret_fp64_b1_qk_pre_reduction.pt")
    elif mode == "block1-qk-reduce":
        result = run_block1_qk_reduce(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/coret_fp64_b1_qk_pre_reduction.pt",
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/coret_fp64_b1_qk.pt")
    elif mode == "block1-qk-native":
        result = run_block1_qk_native(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/coret_fp64_b1_qk_input.pt",
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/coret_fp64_b1_qk_native.pt")
    elif mode == "block1-qk-reserve":
        result = run_block1_qk_reserve(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/coret_fp64_b1_qk_input.pt",
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/coret_fp64_b1_qk_reserve.pt")
    elif mode == "block1-qk-inject":
        result = run_block1_qk_inject(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/coret_fp64_b1_qk_input.pt",
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/coret_fp64_b1_qk_native.pt",
            sys.argv[4] if len(sys.argv) > 4 else "/tmp/coret_fp64_b1_qk_reserve.pt",
            sys.argv[5] if len(sys.argv) > 5 else "/tmp/coret_fp64_b1_qk_pre_reduction.pt")
    elif mode == "block1-softmax-only":
        result = run_block1_softmax_only(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/coret_fp64_b1_qk.pt",
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/coret_fp64_b1_stage1.pt")
    elif mode == "block1-stage2":
        result = run_block1_stage2(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/coret_fp64_b1_stage1.pt",
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/coret_fp64_b1_stage2.pt")
    elif mode == "block1-stage2-av":
        result = run_block1_stage2_av(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/coret_fp64_b1_stage1.pt",
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/coret_fp64_b1_stage2_av.pt")
    elif mode == "block1-stage2-prepare":
        result = run_block1_stage2_prepare(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/coret_fp64_b1_stage1.pt",
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/coret_fp64_b1_av_input.pt")
    elif mode == "block1-stage2-av-only":
        result = run_block1_stage2_av_only(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/coret_fp64_b1_av_input.pt",
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/coret_fp64_b1_stage2_av.pt")
    elif mode == "block1-stage2-post":
        result = run_block1_stage2_post(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/coret_fp64_b1_stage2_av.pt",
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/coret_fp64_b1_stage2_post.pt")
    elif mode == "block1-stage2-post-prepare":
        result = run_block1_stage2_post_prepare(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/coret_fp64_b1_stage2_av.pt",
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/coret_fp64_b1_post_residual.pt")
    elif mode == "block1-stage2-projection":
        result = run_block1_stage2_projection(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/coret_fp64_b1_stage2_av.pt",
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/coret_fp64_b1_attention_projection.pt")
    elif mode == "block1-stage2-residual":
        result = run_block1_stage2_residual(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/coret_fp64_b1_stage2_av.pt",
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/coret_fp64_b1_attention_projection.pt",
            sys.argv[4] if len(sys.argv) > 4 else "/tmp/coret_fp64_b1_post_residual.pt")
    elif mode == "block1-stage2-residual-prepare":
        result = run_block1_stage2_residual_prepare(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/coret_fp64_b1_stage2_av.pt",
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/coret_fp64_b1_attention_projection.pt",
            sys.argv[4] if len(sys.argv) > 4 else "/tmp/coret_fp64_b1_residual_input.pt")
    elif mode == "block1-stage2-residual-add":
        result = run_block1_stage2_residual_add(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/coret_fp64_b1_residual_input.pt",
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/coret_fp64_b1_post_residual.pt")
    elif mode == "block1-stage2-post-ln":
        result = run_block1_stage2_post_ln(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/coret_fp64_b1_post_residual.pt",
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/coret_fp64_b1_stage2_post.pt")
    elif mode == "block1-stage3-ffn1":
        result = run_block1_stage3_ffn1(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/coret_fp64_b1_stage2_post.pt",
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/coret_fp64_b1_ffn1.pt")
    elif mode == "block1-stage3-ffn2":
        result = run_block1_stage3_ffn2(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/coret_fp64_b1_ffn1.pt",
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/coret_fp64_b1_ffn2.pt")
    elif mode == "block1-stage3-residual-prepare":
        result = run_block1_stage3_residual_prepare(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/coret_fp64_b1_stage2_post.pt",
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/coret_fp64_b1_ffn2.pt",
            sys.argv[4] if len(sys.argv) > 4 else "/tmp/coret_fp64_b1_ffn_residual_input.pt")
    elif mode == "block1-stage3-residual-add":
        result = run_block1_stage3_residual_add(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/coret_fp64_b1_ffn_residual_input.pt",
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/coret_fp64_b1_output_residual.pt")
    elif mode == "block1-stage3-final-ln":
        result = run_block1_stage3_final_ln(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/coret_fp64_b1_output_residual.pt",
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/coret_fp64_b1_final.pt")
    elif mode == "float32":
        result = run_plain_fp64(dtype=torch.float32)
    else:
        result = run_plain_fp64()
    print(json.dumps(result, indent=2, sort_keys=True))
