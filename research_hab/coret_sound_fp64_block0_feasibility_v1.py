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
    from Verifiers.Zonotope import make_zonotope_new_weights_same_args
    result = make_zonotope_new_weights_same_args(
        weights, source_zonotope=z, clone=False)
    result.error_term_range_low = low
    result.error_term_range_high = high
    return result


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


def run_sound_fp64(device=None):
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
            measurements = []
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
            raw = dispatch.layer_norm(z, embedding_ln, "standard")
            raw_proof = structural.get_support(raw)
            ln_ops = 16 * 128 * (z.num_error_terms + 1) ** 2 + 4096
            ln_reserve = _reserve_from_majorant(
                _layernorm_majorant(z, embedding_ln), ln_ops)
            z, proof = _inject(
                raw, raw_proof, [z], "embedding_layernorm", ln_ops,
                measurements, reserve=ln_reserve)
            delegate._hidden = proof
            record("embedding_layernorm", z)
            z = production._recenter_native_ranges(z)
            structural.attach_support(z, proof)
            z = dispatch.reduce(z, MAXIMUM_GENERATORS)
            proof = structural.get_support(z)
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
            delegate._hidden = qk_input_proof
            raw_qk = dispatch.qk(q, k)
            raw_qk_proof = structural.get_support(raw_qk)
            qk_ops = (32 * (q.num_error_terms + 1) ** 2 * 16 + 4096)
            qk_reserve = _reserve_from_majorant(
                _bilinear_majorant(q, k), qk_ops)
            qk, qk_proof = _inject(
                raw_qk, raw_qk_proof, [q, k], "qk", qk_ops,
                measurements, reserve=qk_reserve)
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
            delegate._score = score_proof
            raw_probability = dispatch.softmax(scores, no_constraints=False)
            raw_probability_proof = structural.get_support(raw_probability)
            softmax_ops = 64 * (scores.num_error_terms + 1) + 8192
            softmax_reserve = _reserve_from_majorant(
                _softmax_majorant(scores), softmax_ops)
            probability, probability_proof = _inject(
                raw_probability, raw_probability_proof, [scores], "softmax",
                softmax_ops, measurements, reserve=softmax_reserve)
            delegate._probability = probability_proof
            record("softmax", probability)

            value, value_proof = _dense_sound(
                z, proof, parameters["value"], "v_affine", measurements)
            value = value.add_attention_heads_dim(4)
            structural.attach_support(value, value_proof)
            probability, value, av_input_proof = _align_states(
                probability, probability_proof, value, value_proof,
                "attention_value_branches")
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
            delegate._attention_output = context_proof
            record("attention_value", context)

            context = context.remove_attention_heads_dim()
            structural.attach_support(context, context_proof)
            attention, attention_proof = _dense_sound(
                context, context_proof, parameters["attention_output"],
                "attention_output_affine", measurements)
            attention, aligned_residual, residual_union = _align_states(
                attention, attention_proof, residual, residual_proof,
                "attention_residual")
            raw_residual = attention.add(aligned_residual)
            raw_residual_reserve = _add_reserve(attention, aligned_residual)
            attention_residual, attention_residual_proof = _inject(
                raw_residual, residual_union,
                [attention, aligned_residual], "attention_residual", 1,
                measurements, reserve=raw_residual_reserve)
            delegate._hidden = attention_residual_proof
            delegate._attention_output = attention_residual_proof
            raw_post = dispatch.layer_norm(
                attention_residual, parameters["attention_ln"], "standard")
            raw_post_proof = structural.get_support(raw_post)
            ln_ops = (16 * 128 *
                      (attention_residual.num_error_terms + 1) ** 2 + 4096)
            post_ln_reserve = _reserve_from_majorant(
                _layernorm_majorant(
                    attention_residual, parameters["attention_ln"]), ln_ops)
            post, post_proof = _inject(
                raw_post, raw_post_proof, [attention_residual],
                "post_attention_layernorm", ln_ops, measurements,
                reserve=post_ln_reserve)
            delegate._post_attention = post_proof
            record("post_attention_layernorm", post)

            ffn_first, ffn_first_proof = _dense_sound(
                post, post_proof, parameters["ffn_first"], "ffn_first",
                measurements)
            delegate._post_attention = ffn_first_proof
            raw_relu = dispatch.relu(ffn_first)
            raw_relu_proof = structural.get_support(raw_relu)
            relu_ops = 8 * (ffn_first.num_error_terms + 1) + 128
            relu, relu_proof = _inject(
                raw_relu, raw_relu_proof, [ffn_first], "relu", relu_ops,
                measurements, condition=2.0)
            delegate._relu = relu_proof
            ffn, ffn_proof = _dense_sound(
                relu, relu_proof, parameters["ffn_second"], "ffn_second",
                measurements)
            record("ffn_output", ffn)

            ffn, aligned_post, ffn_union = _align_states(
                ffn, ffn_proof, post, post_proof, "ffn_residual")
            raw_ffn_residual = ffn.add(aligned_post)
            ffn_reserve = _add_reserve(ffn, aligned_post)
            ffn_residual, ffn_residual_proof = _inject(
                raw_ffn_residual, ffn_union, [ffn, aligned_post],
                "ffn_residual", 1, measurements, reserve=ffn_reserve)
            delegate._post_attention = ffn_residual_proof
            delegate._relu = ffn_residual_proof

            width = ffn_residual.word_embedding_size
            average = torch.ones((width, width), device=device,
                                 dtype=torch.float64) / width
            centered = ffn_residual.add(
                ffn_residual.matmul(average).multiply(-1.0))
            variance = centered.square_and_sum_and_repeat().multiply(1.0 / width)
            variance_low, variance_high = variance.concretize()
            raw_output = dispatch.layer_norm(
                ffn_residual, parameters["output_ln"], "standard")
            raw_output_proof = structural.get_support(raw_output)
            ln_ops = (16 * 128 *
                      (ffn_residual.num_error_terms + 1) ** 2 + 4096)
            output_ln_reserve = _reserve_from_majorant(
                _layernorm_majorant(
                    ffn_residual, parameters["output_ln"]), ln_ops)
            output, output_proof = _inject(
                raw_output, raw_output_proof, [ffn_residual],
                "output_layernorm", ln_ops, measurements,
                reserve=output_ln_reserve)
            record("output_layernorm", output)
            total_seconds = time.perf_counter() - total_started
            reserves = {
                item["label"]: item["maximum_local_widening"]
                for item in measurements}
            mpfr_spots = _mpfr_spots(checkpoint)
            mpfr_spots.extend(_runtime_mpfr_spots(
                hidden=z, query_parameter=parameters["query"], q=q, k=k,
                raw_qk=raw_qk, scores=scores,
                raw_probability=probability, value=value,
                raw_context=raw_context, ffn_residual=ffn_residual,
                centered=centered, variance=variance, reserves=reserves))
            return {
                "rows": rows,
                "numerical_injections": measurements,
                "second_layernorm_variance_lower": float(variance_low.min()),
                "second_layernorm_variance_upper_min": float(
                    variance_high.min()),
                "dispatch_counts": dict(dispatch.counts),
                "dtype": str(output.zonotope_w.dtype),
                "mpfr_spots": mpfr_spots,
                "final_proof_generator_count": len(output_proof.ids),
                "total_seconds": total_seconds,
                "peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda" else 0),
                "peak_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device)
                    if device.type == "cuda" else 0),
            }
    finally:
        torch.set_default_dtype(prior_dtype)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "plain"
    if mode == "sound":
        result = run_sound_fp64()
    elif mode == "float32":
        result = run_plain_fp64(dtype=torch.float32)
    else:
        result = run_plain_fp64()
    print(json.dumps(result, indent=2, sort_keys=True))
