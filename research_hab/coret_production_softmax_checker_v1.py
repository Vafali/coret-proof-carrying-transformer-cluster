#!/usr/bin/env python3
"""Torch/NumPy-free checker for the bounded native softmax transition."""
from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path

import gmpy2

import coret_production_prefix_checker_v1 as prefix


PRECISION = 256
SCORE_SCALE = 1.0 / math.sqrt(32.0)
FIXTURE_SOFTMAX_STATE_SHA256 = {
    "s0_block0_scaled_scores": "6d4402a0a93f4ca839baed8417eafe1a87b6ccab6f41473608908813a90b1ad3",
    "s1_block0_softmax_denominator": "24eece7ac099e48339a956b51d5cfaa9b8c8f20c616880c25182495e495c8fd2",
    "s2_block0_softmax_preconstraint": "ff61e311897b58aaab5b868d5642e4dfd514dd32ac916b56a8b3c1fe260c4ecf",
    "s3_block0_softmax_output_before_av": "61af8694427515256cdf9505a4440b8f6add91c59a3e43343fc29464303142be",
}
FIXTURE_SOFTMAX_RANGE_SHA256 = {
    "low": "8a938d068c4e41d219cb3f41f2987ec7ed7af7141c053889bbdaede31342924c",
    "high": "087299714411dc16078a7c112891268d67d0e515f16be2d1276095da74c028bb",
}


def _ctx(rounding):
    return gmpy2.local_context(gmpy2.context(), precision=PRECISION,
                               round=rounding)


def _mp(value):
    with _ctx(gmpy2.RoundToNearest):
        return gmpy2.mpfr(value)


def _down(fn):
    with _ctx(gmpy2.RoundDown):
        return +fn()


def _up(fn):
    with _ctx(gmpy2.RoundUp):
        return +fn()


def _fsub(a, b):
    return prefix._f32(prefix._f32(a) - prefix._f32(b))


def _fmul(a, b):
    return prefix._f32(prefix._f32(a) * prefix._f32(b))


def _fdiv(a, b):
    return prefix._f32(prefix._f32(a) / prefix._f32(b))


def _flatten(value):
    if isinstance(value, list):
        for item in value:
            yield from _flatten(item)
    else:
        yield value


def _state(root, record):
    prefix._seal(record, record.get("state_id", "softmax state"))
    weights = prefix._tensor4(
        root, record["producer_tensor_content_ids"]["weights"])
    numerical = prefix._tensor4(
        root, record["producer_tensor_content_ids"]["numerical_radius"])
    shape = (len(weights), len(weights[0]), len(weights[0][0]),
             len(weights[0][0][0]))
    if shape != tuple(record["shape"]):
        raise AssertionError("softmax state shape mismatch")
    if tuple(record["producer_tensor_content_ids"]["numerical_radius"][
            "shape"]) != shape:
        raise AssertionError("softmax sidecar shape mismatch")
    ids = list(record["generator_ids"])
    masks = list(record["generator_support_masks"])
    reasons = list(record["generator_support_reasons"])
    if not (shape[1] == 1 + len(ids) == 1 + len(masks)
            == 1 + len(reasons)):
        raise AssertionError("softmax topology lengths differ")
    mapping = record["ghost_state_linkage"]["ordered_native_to_ghost"]
    if ([entry["native_row"] for entry in mapping]
            != list(range(1, len(ids) + 1))
            or [entry["ghost_id"] for entry in mapping] != ids):
        raise AssertionError("softmax ghost mapping/order mismatch")
    ranges = [(float.fromhex(a), float.fromhex(b))
              for a, b in record["explicit_ranges"]]
    if len(ranges) != len(ids) or any(a > b for a, b in ranges):
        raise AssertionError("softmax explicit ranges invalid")
    native = record["native_range_metadata"]
    if native["kind"] == "absent":
        if native != {"kind": "absent", "low": None, "high": None}:
            raise AssertionError("absent range metadata malformed")
        if ranges != [(-1.0, 1.0)] * len(ids):
            raise AssertionError("implicit ranges differ")
    elif native["kind"] == "explicit":
        lows = prefix._tensor1(root, native["low"])
        highs = prefix._tensor1(root, native["high"])
        if ranges != list(zip(lows, highs)):
            raise AssertionError("explicit/native ranges differ")
    else:
        raise AssertionError("unknown native range kind")
    values = list(_flatten(numerical))
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise AssertionError("invalid numerical sidecar")
    if record["numerical_sidecar_linkage"] != {
            "kind": "coefficient_symmetric_radius",
            "content": "numerical_radius",
            "max_radius_hex": max(values).hex()}:
        raise AssertionError("softmax sidecar metadata mismatch")
    return {"id": record["state_id"], "weights": weights,
            "numerical": numerical, "shape": shape, "ids": ids,
            "masks": masks, "reasons": reasons, "ranges": ranges,
            "range_kind": native["kind"]}


def _blob_values(root, record):
    shape, values = prefix._blob(root, record)
    return shape, values


def _idx4(shape, h, r, q, k):
    return ((h * shape[1] + r) * shape[2] + q) * shape[3] + k


def _get4(flat, shape, h, r, q, k):
    return flat[_idx4(shape, h, r, q, k)]


def _set4(flat, shape, h, r, q, k, value):
    flat[_idx4(shape, h, r, q, k)] = value


def _concretize(weights, numerical, ranges):
    h_count, rows, queries, keys = (len(weights), len(weights[0]),
                                    len(weights[0][0]), len(weights[0][0][0]))
    low = [[[weights[h][0][q][k] for k in range(keys)]
            for q in range(queries)] for h in range(h_count)]
    high = [[[weights[h][0][q][k] for k in range(keys)]
             for q in range(queries)] for h in range(h_count)]
    nrad = [[[numerical[h][0][q][k] for k in range(keys)]
             for q in range(queries)] for h in range(h_count)]
    for h in range(h_count):
        for q in range(queries):
            for k in range(keys):
                for row, (rlo, rhi) in enumerate(ranges, 1):
                    value = weights[h][row][q][k]
                    a, b = value * rlo, value * rhi
                    low[h][q][k] += min(a, b)
                    high[h][q][k] += max(a, b)
                    nrad[h][q][k] += numerical[h][row][q][k]
    return low, high, nrad


def _assert_ge(actual, required, label, slack=2.0 ** -42):
    if actual + slack < float(required):
        raise AssertionError(
            f"{label} numerical sidecar too narrow: "
            f"actual={actual!r} required={str(required)}")


def _close_f32(actual, exact, label, ulps=8):
    """Require a witnessed float32 transcendental/formula result near MPFR."""
    nearest = prefix._f32(float(exact))
    tolerance = ulps * max(math.ulp(nearest), 2.0 ** -149)
    # Python's ulp is binary64; include one float32 ulp explicitly.
    bits = struct.unpack("<I", struct.pack("<f", nearest))[0]
    adjacent_bits = bits + 1 if nearest >= 0 else max(0, bits - 1)
    adjacent = struct.unpack("<f", struct.pack("<I", adjacent_bits))[0]
    tolerance = max(tolerance, ulps * abs(adjacent - nearest))
    if abs(actual - nearest) > tolerance:
        raise AssertionError(
            f"{label} differs from pinned formula: actual={actual!r} "
            f"nearest={nearest!r} delta={abs(actual - nearest)!r} "
            f"tolerance={tolerance!r}")


def _close_formula(actual, exact, operation_scale, label, operations=8):
    """Check a cancellation-prone float32 expression at operation scale.

    ``actual`` is not compared in ulps of the (possibly tiny) result.  The
    pinned formulas form constants/radii by subtracting terms of the size
    supplied in ``operation_scale``.  The standard gamma_n allowance at that
    scale is the appropriate independent rounding obligation.
    """
    with _ctx(gmpy2.RoundToNearest):
        unit = _mp(2) ** -24
        gamma = (operations * unit) / (1 - operations * unit)
        tolerance = gamma * max(_mp(operation_scale), _mp(2) ** -126)
        tolerance += _mp(2) ** -149
        discrepancy = abs(_mp(actual) - _mp(exact))
    if discrepancy > tolerance:
        raise AssertionError(
            f"{label} differs from pinned formula: actual={actual!r} "
            f"exact={str(exact)} tolerance={str(tolerance)}")


def _check_exp_optimal(low, high, optimal):
    """Validate the pinned secant critical point without unstable division."""
    with _ctx(gmpy2.RoundToNearest):
        low, high, optimal = _mp(low), _mp(high), _mp(optimal)
        if not (low <= optimal <= high and optimal <= low + _mp(0.95)):
            raise AssertionError("exp t_opt branch is inadmissible")
        lhs = gmpy2.exp(optimal) * (high - low)
        rhs = gmpy2.exp(high) - gmpy2.exp(low)
        residual = abs(lhs - rhs)
        scale = abs(lhs) + abs(gmpy2.exp(high)) + abs(gmpy2.exp(low))
        # expf/sub/div/log are checked through their defining secant identity.
        # The conditioning factor explicitly accounts for cancellation in
        # exp(high)-exp(low), rather than charging ulps of t_opt itself.
        condition = ((abs(gmpy2.exp(high)) + abs(gmpy2.exp(low)))
                     / max(abs(rhs), _mp(2) ** -126))
        unit = _mp(2) ** -24
        tolerance = (8 * unit * (1 + condition)
                     * max(scale, _mp(2) ** -126))
    if residual > tolerance:
        raise AssertionError("exp t_opt differs from pinned secant branch")


def _check_reciprocal_optimal(low, high, optimal):
    """Validate the pinned reciprocal secant/sqrt critical-point branch."""
    with _ctx(gmpy2.RoundToNearest):
        low, high, optimal = _mp(low), _mp(high), _mp(optimal)
        boundary = high / 2 + _mp(0.01)
        if optimal < boundary:
            raise AssertionError("reciprocal t_opt max branch is inadmissible")
        reciprocal_delta = (1 / high) - (1 / low)
        mean = reciprocal_delta / (high - low)
        residual = abs((-mean) * optimal * optimal - 1)
        scale = abs((-mean) * optimal * optimal) + 1
        condition = ((abs(1 / high) + abs(1 / low))
                     / max(abs(reciprocal_delta), _mp(2) ** -126))
        unit = _mp(2) ** -24
        tolerance = (8 * unit * (1 + condition)
                     * max(scale, _mp(2) ** -126))
    if residual > tolerance:
        raise AssertionError(
            "reciprocal t_opt differs from pinned secant/sqrt branch")


def _residual_bounds(kind, low, high, slope):
    low, high, slope = _mp(low), _mp(high), _mp(slope)
    if kind == "reciprocal" and low <= 0:
        raise AssertionError("reciprocal corrected domain is not positive")
    if kind == "exp":
        candidates = [low, high]
        with _ctx(gmpy2.RoundToNearest):
            critical = gmpy2.log(slope)
        if low <= critical <= high:
            candidates.append(critical)
        values_low = [_down(lambda x=x: gmpy2.exp(x) - slope * x)
                      for x in candidates]
        values_high = [_up(lambda x=x: gmpy2.exp(x) - slope * x)
                       for x in candidates]
    else:
        candidates = [low, high]
        with _ctx(gmpy2.RoundToNearest):
            critical = gmpy2.sqrt(-1 / slope)
        if low <= critical <= high:
            candidates.append(critical)
        values_low = [_down(lambda x=x: 1 / x - slope * x)
                      for x in candidates]
        values_high = [_up(lambda x=x: 1 / x - slope * x)
                       for x in candidates]
    return min(values_low), max(values_high)


def _check_scaling(parent, score, transition):
    prefix._seal(transition, "score scaling")
    tau = transition["tau_k"]
    scale32 = prefix._f32(SCORE_SCALE)
    if (transition.get("operator_family") != "attention_score_scaling"
            or transition.get("input_state_ids") != [parent["id"]]
            or transition.get("output_state_ids") != [score["id"]]
            or tau != {
                "scale_binary64_hex": float(SCORE_SCALE).hex(),
                "scale_float32_hex": scale32.hex(),
                "head_width": 32,
                "generator_transition": "ordered_identity",
                "numerical_policy": "coefficient_local_fixed_scalar",
            }):
        raise AssertionError("attention score scale trace mismatch")
    if (score["ids"] != parent["ids"] or score["masks"] != parent["masks"]
            or score["ranges"] != parent["ranges"]):
        raise AssertionError("score scaling topology mismatch")
    for h in range(4):
        for r in range(parent["shape"][1]):
            for q in range(4):
                for k in range(4):
                    x = parent["weights"][h][r][q][k]
                    expected = _fmul(x, scale32)
                    actual = score["weights"][h][r][q][k]
                    if actual != expected:
                        raise AssertionError("score scaling coefficient mismatch")
                    exact = _mp(x) * _mp(scale32)
                    required = abs(_mp(actual) - exact) + (
                        _mp(parent["numerical"][h][r][q][k])
                        * abs(_mp(scale32)))
                    _assert_ge(score["numerical"][h][r][q][k], required,
                               "score scaling")


def _check_exp_and_denominator(root, score, denominator, witness):
    score_low, score_high, score_n = _concretize(
        score["weights"], score["numerical"], score["ranges"])
    rows = score["shape"][1]
    expected = [[[[0.0 for _ in range(4)] for _ in range(4)]
                 for _ in range(rows + 64)] for _ in range(4)]
    required_n = [[[[0.0 for _ in range(4)] for _ in range(4)]
                    for _ in range(rows + 64)] for _ in range(4)]
    total_active = 0
    for head, record in enumerate(witness["exp_heads"]):
        if record.get("head") != head:
            raise AssertionError("exp head order mutation")
        diff_shape, diff = _blob_values(root, record["input_differences"])
        if diff_shape != (rows, 4, 16):
            raise AssertionError("exp difference shape mismatch")
        lower_shape, lower = _blob_values(root, record["lower"])
        upper_shape, upper = _blob_values(root, record["upper"])
        _, optimal = _blob_values(root, record["t_opt"])
        _, slope = _blob_values(root, record["slope"])
        _, constant = _blob_values(root, record["constant"])
        _, fresh = _blob_values(root, record["fresh"])
        if lower_shape != (4, 16) or upper_shape != (4, 16):
            raise AssertionError("exp domain shape mismatch")
        active = list(record["active_flat_indices"])
        expected_active = []
        compact = {}
        for q in range(4):
            for i in range(4):
                for j in range(4):
                    coordinate = q * 16 + i * 4 + j
                    for r in range(rows):
                        observed = diff[(r * 4 + q) * 16 + i * 4 + j]
                        expected_diff = _fsub(
                            score["weights"][head][r][q][j],
                            score["weights"][head][r][q][i])
                        if observed != expected_diff:
                            raise AssertionError("softmax score-difference mismatch")
                    if any(diff[(r * 4 + q) * 16 + i * 4 + j] != 0
                           for r in range(1, rows)):
                        expected_active.append(coordinate)
        if active != expected_active:
            raise AssertionError("exp fresh allocation/order mismatch")
        compact = {flat: pos for pos, flat in enumerate(active)}
        total_active += len(active)
        for q in range(4):
            for i in range(4):
                old_values = [[] for _ in range(rows)]
                old_eta = [[] for _ in range(rows)]
                fresh_values, fresh_eta = [], []
                for j in range(4):
                    c = q * 16 + i * 4 + j
                    lam, con, rad = slope[c], constant[c], fresh[c]
                    if not math.isfinite(lam) or lam <= 0 or rad < -1e-4:
                        raise AssertionError("exp relaxation parameter invalid")
                    if c in compact:
                        lmp, ump = _mp(lower[c]), _mp(upper[c])
                        with _ctx(gmpy2.RoundToNearest):
                            witnessed_t = _mp(optimal[c])
                            witnessed_lam = _mp(lam)
                            expected_lam = gmpy2.exp(witnessed_t)
                            con_left = (_mp(0.5) * witnessed_lam
                                        * (_mp(1) - witnessed_t - ump))
                            rad_left = (_mp(0.5) * witnessed_lam
                                        * (witnessed_t - ump - _mp(1)))
                            exp_right = _mp(0.5) * gmpy2.exp(ump)
                            expected_con = (con_left
                                            + exp_right)
                            expected_rad = rad_left + exp_right
                        _check_exp_optimal(lower[c], upper[c], optimal[c])
                        _close_f32(lam, expected_lam, "exp slope", 16)
                        _close_formula(
                            con, expected_con, abs(con_left) + abs(exp_right),
                            "exp constant")
                        _close_formula(
                            rad, expected_rad, abs(rad_left) + abs(exp_right),
                            "exp fresh radius")
                    # Independently establish the machine affine domain and
                    # checker-expanded semantic domain.
                    radius = sum(abs(diff[(r * 4 + q) * 16 + i * 4 + j])
                                 for r in range(1, rows))
                    machine_l = diff[(q) * 16 + i * 4 + j] - radius
                    machine_u = diff[(q) * 16 + i * 4 + j] + radius
                    if lower[c] > machine_l + 2e-5 or upper[c] < machine_u - 2e-5:
                        raise AssertionError("exp witnessed domain is inward")
                    eta_rows = []
                    for r in range(rows):
                        if i == j:
                            eta_rows.append(0.0)
                        else:
                            eta_rows.append(
                                score["numerical"][head][r][q][j]
                                + score["numerical"][head][r][q][i])
                    corrected = sum(eta_rows)
                    low = lower[c] - corrected
                    high = upper[c] + corrected
                    if not (low <= optimal[c] <= high):
                        raise AssertionError("exp t_opt outside witnessed domain")
                    residual_lo, residual_hi = _residual_bounds(
                        "exp", low, high, lam)
                    deficit = max(
                        _mp(0), (_mp(con) - _mp(rad)) - residual_lo,
                        residual_hi - (_mp(con) + _mp(rad)))
                    for r in range(rows):
                        x = diff[(r * 4 + q) * 16 + i * 4 + j]
                        if i == j:
                            machine = 1.0 if r == 0 else 0.0
                            req = _mp(0)
                        else:
                            machine = (prefix._fadd(_fmul(lam, x), con)
                                       if r == 0 else _fmul(lam, x))
                            central = _mp(lam) * _mp(x) + (
                                _mp(con) if r == 0 else _mp(0))
                            req = abs(_mp(machine) - central) + (
                                abs(_mp(lam)) * _mp(eta_rows[r]))
                        old_values[r].append(machine)
                        old_eta[r].append(req)
                    is_active = c in compact
                    fresh_values.append(rad if is_active else 0.0)
                    fresh_eta.append(deficit if is_active else _mp(0))
                for r in range(rows):
                    machine_sum = old_values[r][0]
                    for value in old_values[r][1:]:
                        machine_sum = prefix._fadd(machine_sum, value)
                    actual = denominator["weights"][head][r][q][i]
                    if actual != machine_sum:
                        # CUDA reduction order is implementation-defined; its
                        # exact discrepancy is handled below by the numerical
                        # sidecar rather than treated as a semantic mismatch.
                        machine_sum = actual
                    req = sum(old_eta[r], _mp(0))
                    exact_sum = sum((_mp(v) for v in old_values[r]), _mp(0))
                    req += abs(_mp(machine_sum) - exact_sum)
                    _assert_ge(denominator["numerical"][head][r][q][i], req,
                               "exp/denominator")
                    expected[head][r][q][i] = actual
                    required_n[head][r][q][i] = float(req)
                owner = head * 16 + q * 4 + i
                row = rows + owner
                machine_sum = fresh_values[0]
                for value in fresh_values[1:]:
                    machine_sum = prefix._fadd(machine_sum, value)
                actual = denominator["weights"][head][row][q][i]
                if actual != machine_sum:
                    machine_sum = actual
                req = sum(fresh_eta, _mp(0))
                exact_sum = sum((_mp(v) for v in fresh_values), _mp(0))
                req += abs(_mp(machine_sum) - exact_sum)
                _assert_ge(denominator["numerical"][head][row][q][i], req,
                           "collapsed exp fresh")
                expected[head][row][q][i] = actual
                required_n[head][row][q][i] = float(req)
    if total_active != 120:
        raise AssertionError("exp primitive fresh inventory mismatch")
    # Every non-owner collapsed-fresh entry must remain exact zero.
    for h in range(4):
        for r in range(rows, rows + 64):
            for q in range(4):
                for k in range(4):
                    owner = rows + h * 16 + q * 4 + k
                    if r != owner and (denominator["weights"][h][r][q][k] != 0
                                       or denominator["numerical"][h][r][q][k] != 0):
                        raise AssertionError("exp fresh owner placement mismatch")
    return score_low, score_high, score_n


def _check_reciprocal(root, denominator, pre, record):
    _, low = _blob_values(root, record["lower"])
    _, high = _blob_values(root, record["upper"])
    _, optimal = _blob_values(root, record["t_opt"])
    _, slope = _blob_values(root, record["slope"])
    _, constant = _blob_values(root, record["constant"])
    _, fresh = _blob_values(root, record["fresh"])
    active = record["active_flat_indices"]
    if active != list(range(64)):
        raise AssertionError("reciprocal fresh allocation/order mismatch")
    rows = denominator["shape"][1]
    dlo, dhi, dn = _concretize(
        denominator["weights"], denominator["numerical"],
        denominator["ranges"])
    for q in range(4):
        for h in range(4):
          for k in range(4):
            c = q * 16 + h * 4 + k
            if low[c] <= 0 or slope[c] >= 0 or not (low[c] <= optimal[c] <= high[c]):
                raise AssertionError("reciprocal domain/relaxation invalid")
            lmp, ump = _mp(low[c]), _mp(high[c])
            with _ctx(gmpy2.RoundToNearest):
                witnessed_t = _mp(optimal[c])
                witnessed_slope = _mp(slope[c])
                expected_slope = -1 / (witnessed_t * witnessed_t)
                intercept = 1 / lmp - witnessed_slope * lmp
                constant_left = _mp(0.5) * (
                    1 / witnessed_t - witnessed_slope * witnessed_t)
                fresh_left = _mp(0.5) * (
                    witnessed_slope * witnessed_t - 1 / witnessed_t)
                intercept_half = _mp(0.5) * intercept
                expected_constant = constant_left + intercept_half
                expected_fresh = fresh_left + intercept_half
            _check_reciprocal_optimal(low[c], high[c], optimal[c])
            _close_f32(slope[c], expected_slope, "reciprocal slope", 16)
            _close_formula(
                constant[c], expected_constant,
                abs(constant_left) + abs(intercept_half),
                "reciprocal constant")
            _close_formula(
                fresh[c], expected_fresh,
                abs(fresh_left) + abs(intercept_half),
                "reciprocal fresh radius")
            corrected_l, corrected_u = dlo[h][q][k] - dn[h][q][k], dhi[h][q][k] + dn[h][q][k]
            residual_lo, residual_hi = _residual_bounds(
                "reciprocal", corrected_l, corrected_u, slope[c])
            deficit = max(_mp(0), (_mp(constant[c]) - _mp(fresh[c])) - residual_lo,
                          residual_hi - (_mp(constant[c]) + _mp(fresh[c])))
            for r in range(rows):
                x = denominator["weights"][h][r][q][k]
                actual = pre["weights"][h][r][q][k]
                machine = (prefix._fadd(_fmul(slope[c], x), constant[c])
                           if r == 0 else _fmul(slope[c], x))
                if actual != machine:
                    # CUDA pointwise expression lowering may fuse the multiply
                    # and add.  Exact machine discrepancy is independently
                    # charged below; the semantic fixed-slope equation is the
                    # acceptance condition.
                    machine = actual
                required = abs(_mp(actual) - (_mp(slope[c]) * _mp(x)
                           + (_mp(constant[c]) if r == 0 else _mp(0))))
                required += abs(_mp(slope[c])) * _mp(
                    denominator["numerical"][h][r][q][k])
                _assert_ge(pre["numerical"][h][r][q][k], required,
                           "reciprocal")
            owner = rows + c
            actual = pre["weights"][h][owner][q][k]
            if actual != fresh[c]:
                raise AssertionError("reciprocal fresh coefficient/order mismatch")
            _assert_ge(pre["numerical"][h][owner][q][k], deficit,
                       "reciprocal fresh")
    return dlo, dhi, dn


def _check_equality(root, pre, output, record):
    rows, equations, keys = pre["shape"][1], 16, 4
    # Flatten native [head,row,query,key] as [row,head*query,key].
    def value(state, row, equation, key, field="weights"):
        h, q = divmod(equation, 4)
        return state[field][h][row][q][key]
    exact_differences = [[sum(value(pre, row, eq, key) for key in range(keys))
                          - (1.0 if row == 0 else 0.0)
                          for eq in range(equations)] for row in range(rows)]
    cshape, constraint_flat = _blob_values(root, record["constraints"])
    if cshape != (rows, 16):
        raise AssertionError("equality constraint shape mismatch")
    differences = [[constraint_flat[row * 16 + eq]
                    for eq in range(16)] for row in range(rows)]
    gamma4 = (4.0 * 2.0 ** -24) / (1.0 - 4.0 * 2.0 ** -24)
    for row in range(rows):
        for eq in range(16):
            magnitude = sum(abs(value(pre, row, eq, key))
                            for key in range(keys)) + (1.0 if row == 0 else 0.0)
            if abs(differences[row][eq] - exact_differences[row][eq]) > gamma4 * magnitude + 2.0 ** -45:
                raise AssertionError("equality constraint arithmetic mismatch")
    initial = []
    for eq in range(equations):
        candidates = [row for row in range(rows)
                      if differences[row][eq] != 0]
        initial.append(candidates[-1] if candidates else 0)
    if record["initial_pivot_indices"] != initial:
        raise AssertionError("softmax pivot mutation")
    fix = [index for index, pivot in enumerate(initial) if pivot != 0]
    if record["equations_to_fix"] != fix or fix != list(range(16)):
        raise AssertionError("softmax equality-substitution mutation")
    removed = record["removed_generator_indices"]
    denominator_pivots = record["denominator_pivot_indices"]
    if denominator_pivots != initial:
        raise AssertionError("equality denominator pivot mismatch")
    _, optimal = _blob_values(root, record["optimal_values"])
    # Validate pivot admissibility and the witnessed candidate value.
    for eq, pivot in enumerate(removed):
        if pivot <= 0 or pivot >= rows or differences[pivot][eq] == 0:
            raise AssertionError("equality removed pivot inadmissible")
        dp = denominator_pivots[eq]
        denom = differences[dp][eq]
        left_pivot = value(pre, dp, eq, 0)
        constant = prefix._f32(value(pre, pivot, eq, 0)
                               - prefix._f32(left_pivot
                               * prefix._f32(differences[pivot][eq] / denom)))
        linear = prefix._f32(differences[pivot][eq] / denom)
        candidate = _fdiv(-constant, linear)
        if optimal[eq] != candidate:
            raise AssertionError("equality optimal-value/tie mutation")
        # Independently establish the convex width-minimization predicate.
        # For sum_r |a_r+b_r*x|, an optimum has non-positive left and
        # non-negative right subgradient.  This checks the witnessed native
        # candidate without trusting its selected-index list.
        left_derivative = 0.0
        right_derivative = 0.0
        for candidate_row in range(1, rows):
            c = prefix._f32(value(pre, candidate_row, eq, 0)
                            - prefix._f32(left_pivot
                            * prefix._f32(
                                differences[candidate_row][eq] / denom)))
            b = prefix._f32(differences[candidate_row][eq] / denom)
            if b == 0:
                continue
            ratio = _fdiv(-c, b)
            weight = abs(b)
            left_derivative += weight if ratio < optimal[eq] else -weight
            right_derivative += weight if ratio <= optimal[eq] else -weight
        tolerance = 2.0 ** -15 * max(1.0, abs(left_derivative),
                                    abs(right_derivative))
        if left_derivative > tolerance or right_derivative < -tolerance:
            raise AssertionError("equality pivot is not width-admissible")

    # Reconstruct output substitution and coefficient-indexed N propagation.
    for eq, pivot in enumerate(removed):
        dk = differences[pivot][eq]
        for row in range(rows):
            p = -differences[row][eq] / dk
            for key in range(keys):
                x = value(pre, row, eq, key)
                xe = value(pre, row, eq, key, "numerical")
                if key == 0:
                    c = value(pre, pivot, eq, 0) - optimal[eq]
                    ce = value(pre, pivot, eq, 0, "numerical")
                else:
                    c = value(pre, pivot, eq, key)
                    ce = value(pre, pivot, eq, key, "numerical")
                central = x + p * c
                actual = value(output, row, eq, key)
                required = xe + abs(p) * ce + abs(actual - central)
                _assert_ge(value(output, row, eq, key, "numerical"), required,
                           "equality substitution")

    # Independently validate every serialized alpha2 range refinement as an
    # outward projection of the witnessed equality equations.
    current_low = [-1.0] * (rows - 1)
    current_high = [1.0] * (rows - 1)
    for iteration, descriptors in enumerate(record["range_iterations"]):
        _, observed_low = _blob_values(root, descriptors["low"])
        _, observed_high = _blob_values(root, descriptors["high"])
        if len(observed_low) != rows - 1 or len(observed_high) != rows - 1:
            raise AssertionError("equality range iteration shape mismatch")
        required_low = [-1.0] * (rows - 1)
        required_high = [1.0] * (rows - 1)
        for variable in range(rows - 1):
            row = variable + 1
            lo_bound, hi_bound = -1.0, 1.0
            for eq in range(16):
                coeff = differences[row][eq]
                if coeff == 0:
                    continue
                other_lo = differences[0][eq]
                other_hi = differences[0][eq]
                for other in range(rows - 1):
                    if other == variable:
                        continue
                    c = differences[other + 1][eq]
                    a, b = c * current_low[other], c * current_high[other]
                    other_lo += min(a, b); other_hi += max(a, b)
                a, b = -other_hi / coeff, -other_lo / coeff
                lo_bound = max(lo_bound, min(a, b))
                hi_bound = min(hi_bound, max(a, b))
            required_low[variable] = max(-1.0, lo_bound)
            required_high[variable] = min(1.0, hi_bound)
            # Native low is deliberately never narrowed by its pinned clamp.
            if observed_low[variable] > required_low[variable] + 3e-5:
                raise AssertionError("equality lower range narrowed unsafely")
            if observed_high[variable] < required_high[variable] - 3e-5:
                raise AssertionError("equality upper range narrowed unsafely")
            if (observed_low[variable] < current_low[variable] - 1e-7
                    or observed_high[variable] > current_high[variable] + 1e-7):
                raise AssertionError("equality ranges widened")
        current_low, current_high = observed_low, observed_high
    if output["ranges"] != list(zip(current_low, current_high)):
        raise AssertionError("final ranged-symbol metadata mismatch")


def check_production_softmax(root):
    root = Path(root).resolve()
    graph = json.loads((root / "softmax_trace.json").read_text())
    if graph.get("schema") != prefix.SCHEMA:
        raise AssertionError("softmax trace schema mismatch")
    prefix._seal(graph, "softmax trace")
    prefix._seal(graph["run_manifest"], "softmax run manifest")
    manifest = graph["run_manifest"]
    parent_path = root / graph["parent_prefix"]["relative_path"]
    raw = parent_path.read_bytes()
    if (hashlib.sha256(raw).hexdigest() != graph["parent_prefix"]["sha256"]
            or graph["parent_prefix"]["sha256"]
            != manifest["parent_prefix_trace_sha256"]):
        raise AssertionError("parent QK trace identity mismatch")
    parent_result = prefix.check_production_prefix(parent_path.parent)
    if parent_result.get("PRODUCTION_QK_NUMERICAL_CHECK_PASS") is not True:
        raise AssertionError("parent QK checker rejected")
    parent_graph = json.loads(raw)
    parent_qk = prefix._state(parent_path.parent,
                              parent_graph["state_records"][-1])
    if (manifest.get("pinned_deept_revision") != prefix.PINNED_REVISION
            or manifest.get("purpose") != "bounded_real_production_softmax_trace"
            or manifest.get("scientific_query") is not False
            or manifest.get("bound_entrypoint_called") is not False
            or manifest.get("prefix_stop")
            != "block0_native_softmax_output_before_av"):
        raise AssertionError("softmax run identity mismatch")
    records = graph["state_records"]
    states = [_state(root, record) for record in records]
    if graph["graph_nodes"] != [state["id"] for state in states] or len(states) != 4:
        raise AssertionError("softmax state order mismatch")
    score, denominator, pre, output = states
    observed_state_hashes = {
        record["state_id"]: record["producer_tensor_content_ids"][
            "weights"]["sha256"] for record in records}
    if observed_state_hashes != FIXTURE_SOFTMAX_STATE_SHA256:
        raise AssertionError("frozen production softmax state identity mismatch")
    if records[-1]["native_range_metadata"].get("kind") != "explicit":
        raise AssertionError("frozen production softmax ranges missing")
    if ({"low": records[-1]["native_range_metadata"]["low"]["sha256"],
         "high": records[-1]["native_range_metadata"]["high"]["sha256"]}
            != FIXTURE_SOFTMAX_RANGE_SHA256):
        raise AssertionError("frozen production softmax range identity mismatch")
    # Parent _state is rank-4 and has the same representation as this checker.
    _check_scaling(parent_qk, score, graph["transition_records"][0])
    transition = graph["transition_records"][1]
    prefix._seal(transition, "native relational softmax")
    tau = transition["tau_k"]; witness = transition["operator_witness"]
    if (transition.get("operator_family") != "native_relational_softmax"
            or transition.get("input_state_ids") != [score["id"]]
            or transition.get("output_state_ids")
            != [denominator["id"], pre["id"], output["id"]]
            or tau.get("equation") != "1/sum_j(exp(score_j-score_i))"
            or tau.get("batch_softmax_computation") is not True
            or [tau.get("heads"), tau.get("queries"), tau.get("keys")]
            != [4, 4, 4]
            or tau.get("exp_relaxation") != "pinned_minimal_area_v2"
            or tau.get("exp_boolean_order")
            != "head_query_i_j_row_major"
            or tau.get("exp_collapsed_fresh_count") != 64
            or tau.get("reciprocal_relaxation") != "pinned_new_reciprocal"
            or tau.get("reciprocal_boolean_order")
            != "query_head_key_row_major"
            or tau.get("reciprocal_fresh_count") != 64
            or tau.get("numerator_policy")
            != "exact_constant_one_new_softmax_no_multiply"
            or tau.get("equality_branch")
            != "native_sum_constraint_applied"
            or tau.get("equality_initial_pivot_policy")
            != "last_nonzero_generator_row"
            or tau.get("equality_removal_policy")
            != "native_width_minimizing_binary_search"
            or tau.get("equality_tie_breaking")
            != "left_candidate_on_equal_width"
            or tau.get("range_iterations") != 3
            or tau.get("recenter_action")
            != "none_until_downstream_boundary"
            or tau.get("numerical_policy")
            != "coefficient_dependency_aware_fixed_relaxation"):
        raise AssertionError("native softmax trace policy mismatch")
    inherited = len(score["ids"])
    exp_ids = [f"softmax_0_fresh_{i:06d}" for i in range(64)]
    reciprocal_ids = [f"softmax_0_fresh_{i:06d}" for i in range(64, 128)]
    if (tau.get("exp_generator_ids") != exp_ids
            or tau.get("reciprocal_generator_ids") != reciprocal_ids
            or denominator["ids"] != score["ids"] + exp_ids
            or pre["ids"] != score["ids"] + exp_ids + reciprocal_ids
            or output["ids"] != pre["ids"]):
        raise AssertionError("softmax fresh-ID/order mutation")
    expected_exp_masks = [1 << q for _h in range(4)
                          for q in range(4) for _k in range(4)]
    expected_recip_masks = [1 << q for q in range(4)
                            for _h in range(4) for _k in range(4)]
    if (denominator["masks"] != score["masks"] + expected_exp_masks
            or pre["masks"] != denominator["masks"] + expected_recip_masks
            or output["masks"] != pre["masks"]):
        raise AssertionError("softmax support/provenance mutation")
    _check_exp_and_denominator(root, score, denominator, witness)
    _check_reciprocal(root, denominator, pre, witness["reciprocal"])
    _check_equality(root, pre, output, witness["equality"])

    if output["range_kind"] != "explicit" or pre["range_kind"] != "absent":
        raise AssertionError("softmax ranged-symbol transition mismatch")
    transparency = manifest["producer_transparency"]
    out_sha = records[-1]["producer_tensor_content_ids"]["weights"]["sha256"]
    low_sha = records[-1]["native_range_metadata"]["low"]["sha256"]
    high_sha = records[-1]["native_range_metadata"]["high"]["sha256"]
    if transparency != {
            "instrumented_softmax_sha256": out_sha,
            "uninstrumented_softmax_sha256": out_sha,
            "instrumented_range_low_sha256": low_sha,
            "uninstrumented_range_low_sha256": low_sha,
            "instrumented_range_high_sha256": high_sha,
            "uninstrumented_range_high_sha256": high_sha}:
        raise AssertionError("softmax bitwise transparency mismatch")
    if graph.get("final_property_record") is not None:
        raise AssertionError("softmax prefix must not claim a property")

    low, high, numerical = _concretize(
        output["weights"], output["numerical"], output["ranges"])
    native_radius = 0.0; numerical_max = 0.0
    for h in range(4):
        for q in range(4):
            for k in range(4):
                center = output["weights"][h][0][q][k]
                native_radius = max(native_radius, high[h][q][k] - center,
                                    center - low[h][q][k])
                numerical_max = max(numerical_max, numerical[h][q][k])
    sound_low = min(low[h][q][k] - numerical[h][q][k]
                    for h in range(4) for q in range(4) for k in range(4))
    sound_high = max(high[h][q][k] + numerical[h][q][k]
                     for h in range(4) for q in range(4) for k in range(4))
    if not (math.isfinite(sound_low) and math.isfinite(sound_high)
            and sound_low > -0.25 and sound_high < 1.25
            and numerical_max < native_radius):
        raise AssertionError("softmax numerical state is not useful for A.V")
    return {
        "accepted": True,
        "PRODUCTION_SCORE_SCALING_PASS": True,
        "PRODUCTION_SOFTMAX_PIVOT_PASS": True,
        "PRODUCTION_SOFTMAX_EXP_PASS": True,
        "PRODUCTION_SOFTMAX_SUM_RECIPROCAL_PASS": True,
        "PRODUCTION_SOFTMAX_EQUALITY_CONSTRAINTS_PASS": True,
        "PRODUCTION_SOFTMAX_RANGED_SYMBOL_PASS": True,
        "PRODUCTION_SOFTMAX_STATE_CONTINUITY_PASS": True,
        "PRODUCTION_SOFTMAX_BITWISE_EQUIVALENCE_PASS": True,
        "input_shape": list(score["shape"]),
        "output_shape": list(output["shape"]),
        "input_generator_count": len(score["ids"]),
        "output_generator_count": len(output["ids"]),
        "exp_collapsed_fresh_count": 64,
        "reciprocal_fresh_count": 64,
        "exp_primitive_active_count": 120,
        "maximum_native_support": native_radius,
        "maximum_checker_numerical_contribution": numerical_max,
        "numerical_native_ratio": numerical_max / native_radius,
        "softmax_sound_range": [sound_low, sound_high],
    }
