#!/usr/bin/env python3
"""Independent standard-library checker for the bounded production prefix."""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from pathlib import Path


SCHEMA = "CORET_TRACE_WITNESS_V1"
BLOB_SCHEMA = "CORET_TRACE_BLOB_V1"
PINNED_REVISION = "16ffe4075f1f8a7c87fa2a187d8c46cfd51e07bf"
CHECKPOINT_SHA256 = "27ae76c19331bc4d83c2226f9af84650d1ca714c9d0f5f38c1439c620eecda71"
PRODUCTION_MAX_ERROR_TERMS = 14000
FIXTURE_COMPONENT_SHA256 = {
    "word": "03d519c8bb3e9db738e26145cc45b24b7bca074b3ca49d683686845c6b6a3294",
    "position": "99f6c66bb4dc315174155e78ab729b6689a33245f11e11ee7f99f8aae9cc3d23",
    "token_type": "08fa997ee3ac03201ab54b10990815265e9cbf0388b5624586c821a7f716c4f7",
    "gamma": "d6e689cff39650a0fd73e7ba5d730ef7556a5907bcd6aa562138f1cc1a72cb49",
    "beta": "1b3fd7606a55eff0feab88cad22657047a0332022b4dca15d67fb9f27504558e",
    # Q/K hashes are filled from the pinned checkpoint and are independently
    # frozen below; they are not producer-supplied identities.
    "q_weight": "e2d5b4c00b95d23fd6edfd7f5d8ab6908973c423898e17a18fe97a124d713632",
    "q_bias": "8d4581aa1749c7e741193e1836d2777aa58dd93c9d082124a2dd477afcfff47b",
    "k_weight": "7efb433e1bdf839829b7a33144b2763c0c93228636cdf706130bcded9a00b3ce",
    "k_bias": "ba74b3c6c604b3255cf854340a5af3b8326c02d078c4757759f8fe3df0754034",
}
FIXTURE_QK_STATE_SHA256 = {
    "q_projection": "903c20e319ef135a9c9f31e3df198519eaf0c953ef0711bbde652d0ef576beb1",
    "k_projection": "cf4a5fb68395dc34712c48461fdc84d918482eacdc983531028feaad75afffcc",
    "q_heads": "1308685110fbc6e6864886c6594ef6f19491f8f59c6d5b239c5b8ca3070c40d8",
    "k_heads": "f45f6f260723b1c7f052c8c83baea264aaf0d7ce143280feb3f2d264eb9fd508",
    "qk_output": "2dfc476bfb0224ca11f84cb927ec7970e6c748f9fda87b05d00eec76b264f439",
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def _seal(value, label):
    claimed = value.get("canonical_sha256")
    body = dict(value); body.pop("canonical_sha256", None)
    if claimed != hashlib.sha256(_canonical(body)).hexdigest():
        raise AssertionError(f"{label} canonical hash mismatch")


def _dec(value):
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        numerator, denominator = value.as_integer_ratio()
        with localcontext() as ctx:
            ctx.prec = 120
            return Decimal(numerator) / Decimal(denominator)
    return Decimal(value)


class I:
    def __init__(self, lo, hi=None):
        self.lo, self.hi = _dec(lo), _dec(lo if hi is None else hi)
        if self.lo > self.hi:
            raise AssertionError("reversed interval")

    def __add__(self, other):
        other = _i(other)
        with localcontext() as c:
            c.prec = 100; c.rounding = ROUND_FLOOR; lo = self.lo + other.lo
        with localcontext() as c:
            c.prec = 100; c.rounding = ROUND_CEILING; hi = self.hi + other.hi
        return I(lo, hi)

    def __neg__(self):
        return I(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + (-_i(other))

    def __mul__(self, other):
        other = _i(other)
        with localcontext() as c:
            c.prec = 100; c.rounding = ROUND_FLOOR
            lo_values = [a * b for a in (self.lo, self.hi)
                         for b in (other.lo, other.hi)]
        with localcontext() as c:
            c.prec = 100; c.rounding = ROUND_CEILING
            hi_values = [a * b for a in (self.lo, self.hi)
                         for b in (other.lo, other.hi)]
        return I(min(lo_values), max(hi_values))

    def reciprocal(self):
        if self.lo <= 0 <= self.hi:
            raise AssertionError("reciprocal domain")
        with localcontext() as c:
            c.prec = 100; c.rounding = ROUND_FLOOR
            low = [Decimal(1) / self.lo, Decimal(1) / self.hi]
        with localcontext() as c:
            c.prec = 100; c.rounding = ROUND_CEILING
            high = [Decimal(1) / self.lo, Decimal(1) / self.hi]
        return I(min(low), max(high))

    def __truediv__(self, other):
        return self * _i(other).reciprocal()

    def sqrt(self):
        if self.lo < 0:
            raise AssertionError("sqrt domain")
        with localcontext() as c:
            c.prec = 100; c.rounding = ROUND_FLOOR; lo = self.lo.sqrt()
        with localcontext() as c:
            c.prec = 100; c.rounding = ROUND_CEILING; hi = self.hi.sqrt()
        return I(lo, hi)


def _i(value):
    return value if isinstance(value, I) else I(value)


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _fadd(left, right):
    return _f32(_f32(left) + _f32(right))


def _blob(root, record):
    _seal(record, "blob")
    if (record.get("schema") != BLOB_SCHEMA
            or record.get("dtype") != "float32"
            or record.get("byte_order") != "little"):
        raise AssertionError("blob schema mismatch")
    relative = Path(record["relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise AssertionError("blob path escape")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise AssertionError("blob path escape") from error
    raw = path.read_bytes()
    if (len(raw) != record["byte_count"]
            or hashlib.sha256(raw).hexdigest() != record["sha256"]):
        raise AssertionError("blob identity mismatch")
    shape = tuple(int(item) for item in record["shape"])
    values = [item[0] for item in struct.iter_unpack("<f", raw)]
    if math.prod(shape) != len(values):
        raise AssertionError("blob shape mismatch")
    return shape, values


def _tensor3(root, record):
    shape, values = _blob(root, record)
    if len(shape) != 3:
        raise AssertionError("state tensor rank mismatch")
    rows, tokens, width = shape
    return [[[values[(r * tokens + t) * width + j]
              for j in range(width)] for t in range(tokens)]
            for r in range(rows)]


def _tensor4(root, record):
    shape, values = _blob(root, record)
    if len(shape) != 4:
        raise AssertionError("state tensor rank mismatch")
    first, rows, tokens, width = shape
    return [[[[values[((a * rows + r) * tokens + t) * width + j]
               for j in range(width)] for t in range(tokens)]
             for r in range(rows)] for a in range(first)]


def _tensor2(root, record):
    shape, values = _blob(root, record)
    if len(shape) != 2:
        raise AssertionError("embedding component rank mismatch")
    return [values[index * shape[1]:(index + 1) * shape[1]]
            for index in range(shape[0])]


def _tensor1(root, record):
    shape, values = _blob(root, record)
    if len(shape) != 1:
        raise AssertionError("parameter tensor rank mismatch")
    return values


def _state(root, record):
    _seal(record, record.get("state_id", "state"))
    weight_descriptor = record["producer_tensor_content_ids"]["weights"]
    rank = len(weight_descriptor.get("shape", []))
    if rank == 3:
        weights = _tensor3(root, weight_descriptor)
        numerical = _tensor3(
            root, record["producer_tensor_content_ids"]["numerical_radius"])
        generator_rows = len(weights)
        numerical_shape = (len(numerical), len(numerical[0]),
                           len(numerical[0][0]))
        weight_shape = (len(weights), len(weights[0]), len(weights[0][0]))
        numerical_values = (value for row in numerical
                            for token in row for value in token)
    elif rank == 4:
        weights = _tensor4(root, weight_descriptor)
        numerical = _tensor4(
            root, record["producer_tensor_content_ids"]["numerical_radius"])
        generator_rows = len(weights[0])
        numerical_shape = (len(numerical), len(numerical[0]),
                           len(numerical[0][0]), len(numerical[0][0][0]))
        weight_shape = (len(weights), len(weights[0]), len(weights[0][0]),
                        len(weights[0][0][0]))
        numerical_values = (value for head in numerical for row in head
                            for token in row for value in token)
    else:
        raise AssertionError("state tensor rank mismatch")
    ids = list(record["generator_ids"])
    ranges = [(float.fromhex(a), float.fromhex(b))
              for a, b in record["explicit_ranges"]]
    if generator_rows != 1 + len(ids) or len(ranges) != len(ids):
        raise AssertionError("state dimensions mismatch")
    native_range_metadata = record.get("native_range_metadata")
    if native_range_metadata != {"kind": "absent", "low": None,
                                 "high": None}:
        raise AssertionError("unexpected native ranged-symbol metadata")
    mapping = record["ghost_state_linkage"]["ordered_native_to_ghost"]
    if ([item["native_row"] for item in mapping]
            != list(range(1, len(ids) + 1))
            or [item["ghost_id"] for item in mapping] != ids):
        raise AssertionError("generator mapping/order mismatch")
    if numerical_shape != weight_shape:
        raise AssertionError("numerical sidecar shape mismatch")
    sidecar = record["numerical_sidecar_linkage"]
    if (sidecar.get("kind") != "coefficient_symmetric_radius"
            or sidecar.get("content") != "numerical_radius"):
        raise AssertionError("numerical sidecar missing")
    if any(not math.isfinite(value) or value < 0
           for value in numerical_values):
        raise AssertionError("invalid numerical sidecar")
    if (len(record.get("generator_support_masks", [])) != len(ids)
            or len(record.get("generator_support_reasons", [])) != len(ids)
            or len(set(ids)) != len(ids)):
        raise AssertionError("invalid support/provenance state")
    return {"id": record["state_id"], "weights": weights, "ids": ids,
            "ranges": ranges, "numerical": numerical,
            "masks": list(record["generator_support_masks"]),
            "reasons": list(record["generator_support_reasons"]),
            "native_range_kind": "absent", "shape": weight_shape,
            "rank": rank, "weights_sha256": weight_descriptor["sha256"]}


def _exact(weights):
    return [[[I(value) for value in feature] for feature in token]
            for token in weights]


def _concretize(coefficients, ranges):
    rows, tokens, width = len(coefficients), len(coefficients[0]), len(coefficients[0][0])
    low = [[coefficients[0][t][j] for j in range(width)] for t in range(tokens)]
    high = [[coefficients[0][t][j] for j in range(width)] for t in range(tokens)]
    for row in range(1, rows):
        lo_range, hi_range = ranges[row - 1]
        for token in range(tokens):
            for feature in range(width):
                contribution = coefficients[row][token][feature] * I(
                    lo_range, hi_range)
                low[token][feature] = low[token][feature] + I(contribution.lo)
                high[token][feature] = high[token][feature] + I(contribution.hi)
    return low, high


def _relation(state, exact):
    stored = state["weights"]
    if (len(stored) != len(exact)
            or len(stored[0]) != len(exact[0])
            or len(stored[0][0]) != len(exact[0][0])):
        raise AssertionError("exact/native topology mismatch")
    for r in range(len(stored)):
        for t in range(len(stored[0])):
            for j in range(len(stored[0][0])):
                value = _dec(stored[r][t][j])
                expected = exact[r][t][j]
                radius = _dec(state["numerical"][r][t][j])
                represented = I(value) + I(-radius, radius)
                if (represented.lo > expected.lo
                        or represented.hi < expected.hi):
                    raise AssertionError(
                        f"numerical sidecar too narrow at {r},{t},{j}")


def _flat(values):
    for value in values:
        if isinstance(value, list):
            yield from _flat(value)
        else:
            yield value


def _check_affine_projection(root, transition, source, output, label):
    _seal(transition, f"{label} projection transition")
    tau = transition.get("tau_k", {})
    witness = transition.get("operator_witness", {})
    if (transition.get("operator_family") != "affine_projection"
            or transition.get("input_state_ids") != [source["id"]]
            or transition.get("predecessor_state_id") != source["id"]
            or transition.get("output_state_ids") != [output["id"]]
            or tau != {
                "projection": label, "input_width": 128,
                "output_width": 128,
                "native_equation": "matmul(weight_transpose)_then_add_bias",
                "generator_transition": "ordered_identity",
                "support_transition": "token_mask_identity",
            }
            or witness.get("input_weights_sha256")
            != source["weights_sha256"]
            or witness.get("output_weights_sha256")
            != output["weights_sha256"]):
        raise AssertionError(f"{label} projection trace mismatch")
    weight_descriptor = witness.get("weight", {})
    bias_descriptor = witness.get("bias", {})
    expected_weight = FIXTURE_COMPONENT_SHA256[f"{label.lower()}_weight"]
    expected_bias = FIXTURE_COMPONENT_SHA256[f"{label.lower()}_bias"]
    if (weight_descriptor.get("sha256") != expected_weight
            or bias_descriptor.get("sha256") != expected_bias):
        raise AssertionError(f"{label} frozen parameter identity mismatch")
    weight = _tensor2(root, weight_descriptor)
    bias = _tensor1(root, bias_descriptor)
    if (len(weight) != 128 or any(len(row) != 128 for row in weight)
            or len(bias) != 128
            or source["shape"] != (901, 4, 128)
            or output["shape"] != (901, 4, 128)):
        raise AssertionError(f"{label} projection dimension mismatch")
    if (output["ids"] != source["ids"]
            or output["ranges"] != source["ranges"]
            or output["masks"] != source["masks"]
            or output["reasons"] != source["reasons"]):
        raise AssertionError(f"{label} projection topology mismatch")

    # This is deliberately independent of the producer's SGEMM schedule.
    # For every coefficient, |T_real(x)-y_hat| is bounded by the triangle
    # inequality using exact IEEE operands and the predecessor sidecar.
    maximum_input = max(
        abs(_dec(value)) + _dec(radius)
        for value, radius in zip(_flat(source["weights"]),
                                 _flat(source["numerical"])))
    maximum_weight_l1 = max(
        sum((abs(_dec(value)) for value in row), Decimal(0))
        for row in weight)
    maximum_bias = max(abs(_dec(value)) for value in bias)
    maximum_output = max(abs(_dec(value)) for value in _flat(output["weights"]))
    required = (maximum_input * maximum_weight_l1
                + maximum_bias + maximum_output)
    minimum_radius = min(_dec(value) for value in _flat(output["numerical"]))
    if minimum_radius < required:
        raise AssertionError(f"{label} projection numerical sidecar too narrow")
    return required


def _check_head_mapping(transition, source, output, label):
    _seal(transition, f"{label} head mapping transition")
    expected_tau = {
        "projection": label, "heads": 4, "head_width": 32,
        "reshape": [901, 4, 4, 32], "permutation": [2, 0, 1, 3],
        "generator_transition": "ordered_identity",
    }
    witness = transition.get("operator_witness", {})
    if (transition.get("operator_family") != "attention_head_mapping"
            or transition.get("input_state_ids") != [source["id"]]
            or transition.get("predecessor_state_id") != source["id"]
            or transition.get("output_state_ids") != [output["id"]]
            or transition.get("tau_k") != expected_tau
            or witness.get("input_weights_sha256")
            != source["weights_sha256"]
            or witness.get("output_weights_sha256")
            != output["weights_sha256"]
            or source["shape"] != (901, 4, 128)
            or output["shape"] != (4, 901, 4, 32)):
        raise AssertionError(f"{label} head mapping trace mismatch")
    if (output["ids"] != source["ids"]
            or output["ranges"] != source["ranges"]
            or output["masks"] != source["masks"]
            or output["reasons"] != source["reasons"]):
        raise AssertionError(f"{label} head mapping topology mismatch")
    for head in range(4):
        for row in range(901):
            for token in range(4):
                expected_weights = source["weights"][row][token][
                    head * 32:(head + 1) * 32]
                expected_numerical = source["numerical"][row][token][
                    head * 32:(head + 1) * 32]
                if output["weights"][head][row][token] != expected_weights:
                    raise AssertionError(f"{label} head permutation mismatch")
                if output["numerical"][head][row][token] != expected_numerical:
                    raise AssertionError(
                        f"{label} head numerical-sidecar mapping mismatch")


def _check_qk_sidecar(q, k, output):
    maximum_q = max(abs(_dec(value)) + _dec(radius)
                    for value, radius in zip(_flat(q["weights"]),
                                             _flat(q["numerical"])))
    maximum_k = max(abs(_dec(value)) + _dec(radius)
                    for value, radius in zip(_flat(k["weights"]),
                                             _flat(k["numerical"])))
    generators, inner = len(q["ids"]), q["shape"][-1]
    product = maximum_q * maximum_k * Decimal(inner)
    center = product * (Decimal(1) + Decimal(generators) / Decimal(2))
    retained = product * Decimal(2)
    quadratic = product * (
        Decimal(generators * generators) - Decimal(generators) / Decimal(2))
    maximum_exact = max(center, retained, quadratic)
    maximum_output = max(abs(_dec(value)) for value in _flat(output["weights"]))
    required = maximum_exact + maximum_output
    minimum_radius = min(_dec(value) for value in _flat(output["numerical"]))
    if minimum_radius < required:
        raise AssertionError("QK propagated numerical sidecar too narrow")
    return required


def _zero_rows(count, tokens, width):
    return [[[I(0) for _ in range(width)] for _ in range(tokens)]
            for _ in range(count)]


def _abs_upper(value):
    return max(abs(value.lo), abs(value.hi))


def _native_layernorm(coefficients, ids, ranges, tau, gamma, beta):
    rows, tokens, width = len(coefficients), len(coefficients[0]), len(coefficients[0][0])
    if (tau.get("mode") != "standard"
            or tau.get("branch") != "positive_variance_standard_layernorm"
            or tau.get("mean_divisor") != width
            or tau.get("variance_divisor") != width
            or tau.get("variance_fresh_token_order") != list(range(tokens))
            or tau.get("ranged_symbol_action") != "none"
            or tau.get("native_boolean_order") != "row_major"):
        raise AssertionError("LayerNorm trace choice mismatch")
    centered = _zero_rows(rows, tokens, width)
    for r in range(rows):
        for token in range(tokens):
            mean = I(0)
            for value in coefficients[r][token]:
                mean = mean + value
            mean = mean / I(width)
            centered[r][token] = [value - mean
                                  for value in coefficients[r][token]]

    variance = _zero_rows(rows + tokens, tokens, width)
    epsilon = I(float.fromhex(tau["epsilon_hex"]))
    for token in range(tokens):
        center = I(0)
        for value in centered[0][token]:
            center = center + value * value
        center = center / I(width) + epsilon
        variance[0][token] = [center for _ in range(width)]
        for generator in range(1, rows):
            retained = I(0)
            for c, a in zip(centered[0][token], centered[generator][token]):
                retained = retained + c * a
            retained = I(2) * retained / I(width)
            variance[generator][token] = [retained for _ in range(width)]
        support_square = I(0)
        for feature in range(width):
            support = I(0)
            for generator in range(1, rows):
                support = support + I(_abs_upper(
                    centered[generator][token][feature]))
            support_square = support_square + support * support
        variance[rows + token][token] = [support_square / I(width)
                                         for _ in range(width)]
    current_ids = list(ids) + [f"layernorm_0_fresh_{i:06d}"
                               for i in range(tokens)]
    current_ranges = list(ranges) + [(-1.0, 1.0)] * tokens

    def unary(source, active_record, kind):
        lower, upper = _concretize(source, current_ranges)
        active = [token * width + feature for token in range(tokens)
                  for feature in range(width)
                  if lower[token][feature].lo != upper[token][feature].hi]
        if active != active_record:
            raise AssertionError(f"incorrect {kind} allocation")
        output = _zero_rows(len(source) + len(active), tokens, width)
        owner = {flat: offset for offset, flat in enumerate(active)}
        for token in range(tokens):
            for feature in range(width):
                flat = token * width + feature
                low, high = lower[token][feature], upper[token][feature]
                if low.lo <= 0:
                    raise AssertionError(f"{kind} domain")
                if flat not in owner:
                    output[0][token][feature] = (low.sqrt() if kind == "sqrt"
                                                  else low.reciprocal())
                    continue
                if kind == "sqrt":
                    sqrt_l, sqrt_u = low.sqrt(), high.sqrt()
                    slope = (sqrt_u - sqrt_l) / (high - low)
                    critical = ((high - low) / (I(2) * (sqrt_u - sqrt_l)))
                    critical = critical * critical
                    intercept = sqrt_l - slope * low
                    root = critical.sqrt()
                    constant = I(.5) * (root - slope * critical + intercept)
                    fresh = I(.5) * (slope * critical - root + intercept)
                else:
                    slope = -I(1) / (high * high)
                    bottom = I(1) / high - slope * high
                    top = I(1) / low - slope * low
                    constant = I(.5) * (top + bottom)
                    fresh = I(.5) * (top - bottom)
                output[0][token][feature] = (
                    slope * source[0][token][feature] + constant)
                for r in range(1, len(source)):
                    output[r][token][feature] = (
                        slope * source[r][token][feature])
                output[len(source) + owner[flat]][token][feature] = fresh
        start = len(current_ids)
        current_ids.extend(f"layernorm_0_fresh_{start - len(ids) + i:06d}"
                           for i in range(len(active)))
        current_ranges.extend([(-1.0, 1.0)] * len(active))
        return output

    sqrt = unary(variance, tau["sqrt_active_flat_indices"], "sqrt")
    reciprocal = unary(sqrt, tau["reciprocal_active_flat_indices"],
                       "reciprocal")
    centered.extend(_zero_rows(len(reciprocal) - len(centered), tokens, width))
    product = _zero_rows(len(reciprocal), tokens, width)
    for token in range(tokens):
        for feature in range(width):
            product[0][token][feature] = (
                centered[0][token][feature]
                * reciprocal[0][token][feature])
            for r in range(1, len(reciprocal)):
                product[r][token][feature] = (
                    centered[0][token][feature]
                    * reciprocal[r][token][feature]
                    + reciprocal[0][token][feature]
                    * centered[r][token][feature])
    expected_product = list(range(tokens * width))
    if tau["product_active_flat_indices"] != expected_product:
        raise AssertionError("incorrect final product allocation")
    start = len(current_ids)
    for flat in expected_product:
        token, feature = divmod(flat, width)
        left = I(0); right = I(0)
        for r in range(1, len(reciprocal)):
            left = left + I(_abs_upper(centered[r][token][feature]))
            right = right + I(_abs_upper(reciprocal[r][token][feature]))
        row = _zero_rows(1, tokens, width)[0]
        row[token][feature] = left * right
        product.append(row)
    current_ids.extend(f"layernorm_0_fresh_{start - len(ids) + i:06d}"
                       for i in range(tokens * width))
    current_ranges.extend([(-1.0, 1.0)] * (tokens * width))
    if current_ids[len(ids):] != tau["fresh_generator_ids"]:
        raise AssertionError("LayerNorm fresh generator order mismatch")
    affected = 0
    for mask in tau.get("input_support_masks", []):
        affected |= int(mask)
    if not tau.get("input_support_masks"):
        raise AssertionError("LayerNorm inherited support topology missing")
    expected_masks = (
        [(1 << token) if affected & (1 << token) else 0
         for token in range(tokens)]
        + [1 << (flat // width) for flat in tau["sqrt_active_flat_indices"]]
        + [1 << (flat // width)
           for flat in tau["reciprocal_active_flat_indices"]]
        + [(1 << (flat // width))
           if affected & (1 << (flat // width)) else 0
           for flat in expected_product])
    if tau.get("fresh_support_masks") != expected_masks:
        raise AssertionError("LayerNorm fresh support topology mismatch")
    for r in range(len(product)):
        for token in range(tokens):
            for feature in range(width):
                product[r][token][feature] = (
                    product[r][token][feature] * I(gamma[feature])
                    + (I(beta[feature]) if r == 0 else I(0)))
    return product, current_ids, current_ranges


def check_production_prefix(root):
    root = Path(root).resolve()
    graph = json.loads((root / "trace.json").read_text())
    if graph.get("schema") != SCHEMA:
        raise AssertionError("trace schema mismatch")
    _seal(graph, "trace"); _seal(graph["run_manifest"], "run manifest")
    _seal(graph["source_domain"], "source domain")
    manifest = graph["run_manifest"]
    if (manifest.get("pinned_deept_revision") != PINNED_REVISION
            or manifest.get("scientific_query") is not False
            or manifest.get("bound_entrypoint_called") is not False
            or manifest.get("prefix_stop")
            != "block0_native_precise_qk_output_before_softmax"
            or manifest.get("abstract_affine_residual_transition_count") != 0):
        raise AssertionError("production prefix run identity mismatch")
    if graph.get("content_store") != {"schema": BLOB_SCHEMA, "root": "."}:
        raise AssertionError("content store identity mismatch")
    domain = graph["source_domain"]
    if (domain.get("p_cli") != 100
            or domain.get("interpreted_domain") != "Linf"
            or domain.get("checkpoint_sha256") != CHECKPOINT_SHA256
            or domain.get("epsilon_hex") != float(1.0 / 1600.0).hex()
            or domain.get("model_identity") != "deept_table7_stdln3_ckpt5"
            or domain.get("property_identity")
            != "bounded_prefix_tokens_101_2023_2003_102_tok1"
            or domain.get("input_token_ids") != [101, 2023, 2003, 102]
            or domain.get("perturbed_token") != 1
            or domain.get("source_coordinate_order") != list(range(128))
            or domain.get("source_ranges")
            != [[(-1.0).hex(), (1.0).hex()]] * 128
            or domain.get("input_source_mask") != [1] * 128):
        raise AssertionError("production source domain mismatch")
    states = {record["state_id"]: _state(root, record)
              for record in graph["state_records"]}
    if graph["graph_nodes"] != list(states) or len(states) != 9:
        raise AssertionError("prefix state identity/order mismatch")
    source = states[graph["graph_nodes"][0]]
    output = states[graph["graph_nodes"][1]]
    recentered = states[graph["graph_nodes"][2]]
    reduced = states[graph["graph_nodes"][3]]
    q_projection = states[graph["graph_nodes"][4]]
    k_projection = states[graph["graph_nodes"][5]]
    q_heads = states[graph["graph_nodes"][6]]
    k_heads = states[graph["graph_nodes"][7]]
    qk_output = states[graph["graph_nodes"][8]]
    transparency = manifest.get("producer_transparency", {})
    source_commitment = graph["state_records"][0][
        "producer_tensor_content_ids"]["weights"]["sha256"]
    output_commitment = graph["state_records"][1][
        "producer_tensor_content_ids"]["weights"]["sha256"]
    recentered_commitment = graph["state_records"][2][
        "producer_tensor_content_ids"]["weights"]["sha256"]
    reduced_commitment = graph["state_records"][3][
        "producer_tensor_content_ids"]["weights"]["sha256"]
    q_projection_commitment = graph["state_records"][4][
        "producer_tensor_content_ids"]["weights"]["sha256"]
    k_projection_commitment = graph["state_records"][5][
        "producer_tensor_content_ids"]["weights"]["sha256"]
    q_heads_commitment = graph["state_records"][6][
        "producer_tensor_content_ids"]["weights"]["sha256"]
    k_heads_commitment = graph["state_records"][7][
        "producer_tensor_content_ids"]["weights"]["sha256"]
    qk_commitment = graph["state_records"][8][
        "producer_tensor_content_ids"]["weights"]["sha256"]
    if ({"q_projection": q_projection_commitment,
         "k_projection": k_projection_commitment,
         "q_heads": q_heads_commitment, "k_heads": k_heads_commitment,
         "qk_output": qk_commitment} != FIXTURE_QK_STATE_SHA256):
        raise AssertionError("frozen production Q/K/QK state identity mismatch")
    if (transparency.get("uninstrumented_source_sha256") != source_commitment
            or transparency.get("instrumented_source_sha256") != source_commitment
            or transparency.get("uninstrumented_layernorm_sha256")
            != output_commitment
            or transparency.get("instrumented_layernorm_sha256")
            != output_commitment
            or transparency.get("uninstrumented_recenter_sha256")
            != recentered_commitment
            or transparency.get("instrumented_recenter_sha256")
            != recentered_commitment
            or transparency.get("uninstrumented_reduction_sha256")
            != reduced_commitment
            or transparency.get("instrumented_reduction_sha256")
            != reduced_commitment
            or transparency.get("uninstrumented_q_projection_sha256")
            != q_projection_commitment
            or transparency.get("instrumented_q_projection_sha256")
            != q_projection_commitment
            or transparency.get("uninstrumented_k_projection_sha256")
            != k_projection_commitment
            or transparency.get("instrumented_k_projection_sha256")
            != k_projection_commitment
            or transparency.get("uninstrumented_q_heads_sha256")
            != q_heads_commitment
            or transparency.get("instrumented_q_heads_sha256")
            != q_heads_commitment
            or transparency.get("uninstrumented_k_heads_sha256")
            != k_heads_commitment
            or transparency.get("instrumented_k_heads_sha256")
            != k_heads_commitment
            or transparency.get("uninstrumented_qk_sha256") != qk_commitment
            or transparency.get("instrumented_qk_sha256") != qk_commitment
            or transparency.get("membership_replay_bitwise_identical") is not True):
        raise AssertionError("producer prefix bitwise transparency mismatch")
    expected_source_ids = [f"input_source_{index:06d}"
                           for index in range(128)]
    if (source["ids"] != expected_source_ids
            or domain["source_symbol_ids"] != expected_source_ids
            or source["masks"] != [2] * 128
            or source["ranges"] != [(-1.0, 1.0)] * 128
            or len(output["weights"]) != 901):
        raise AssertionError("source symbol identity/order mismatch")
    components = domain["embedding_components"]
    if ({name: descriptor.get("sha256")
         for name, descriptor in components.items()}
            != FIXTURE_COMPONENT_SHA256):
        raise AssertionError("frozen model/input component identity mismatch")
    word = _tensor2(root, components["word"])
    position = _tensor2(root, components["position"])
    token_type = _tensor2(root, components["token_type"])
    if domain.get("embedding_sum_order") != ["word", "position", "token_type"]:
        raise AssertionError("embedding affine order mismatch")
    expected_center = [[_fadd(_fadd(word[t][j], position[t][j]),
                              token_type[t][j])
                        for j in range(128)] for t in range(4)]
    if source["weights"][0] != expected_center:
        raise AssertionError("embedding affine coefficient mismatch")
    rho = _f32(float.fromhex(domain["epsilon_hex"]))
    for row in range(1, 129):
        for token in range(4):
            for feature in range(128):
                expected = rho if token == 1 and feature == row - 1 else 0.0
                if source["weights"][row][token][feature] != expected:
                    raise AssertionError("source coefficient construction mismatch")
    exact = _exact(source["weights"])
    _relation(source, exact)
    if len(graph["transition_records"]) != 8:
        raise AssertionError("prefix transition count mismatch")
    transition = graph["transition_records"][0]
    _seal(transition, "LayerNorm transition")
    if (transition.get("operator_family") != "LayerNorm"
            or transition.get("input_state_ids") != [source["id"]]
            or transition.get("predecessor_state_id") != source["id"]
            or transition.get("output_state_ids") != [output["id"]]):
        raise AssertionError("LayerNorm predecessor/state continuity mismatch")
    gamma = _tensor1(root, transition["operator_witness"]["gamma"])
    beta = _tensor1(root, transition["operator_witness"]["beta"])
    if transition["tau_k"].get("input_support_masks") != source["masks"]:
        raise AssertionError("LayerNorm inherited support topology mismatch")
    exact, ids, ranges = _native_layernorm(
        exact, list(source["ids"]), list(source["ranges"]),
        transition["tau_k"], gamma, beta)
    if output["ids"] != ids or output["ranges"] != ranges:
        raise AssertionError("LayerNorm output generator identity/order mismatch")
    if output["masks"] != ([1 << domain["perturbed_token"]] * 128
                            + transition["tau_k"]["fresh_support_masks"]):
        raise AssertionError("LayerNorm support transition mismatch")
    _relation(output, exact)
    recenter_transition = graph["transition_records"][1]
    _seal(recenter_transition, "ranged-symbol recenter transition")
    recenter_tau = recenter_transition.get("tau_k", {})
    recenter_witness = recenter_transition.get("operator_witness", {})
    expected_indices = list(range(len(output["ids"])))
    if (recenter_transition.get("operator_family")
            != "ranged_symbol_recenter"
            or recenter_transition.get("input_state_ids") != [output["id"]]
            or recenter_transition.get("predecessor_state_id") != output["id"]
            or recenter_transition.get("output_state_ids")
            != [recentered["id"]]
            or recenter_tau != {
                "branch": "skip_no_explicit_ranges",
                "input_range_low_present": False,
                "input_range_high_present": False,
                "native_skip_predicate": "error_term_range_low_is_None",
                "native_return_identity": True,
                "generator_transition": "ordered_identity",
            }
            or recenter_witness.get("input_weights_sha256")
            != output_commitment
            or recenter_witness.get("output_weights_sha256")
            != recentered_commitment
            or recenter_witness.get("recenter_map")
            != "identity_no_explicit_ranges"
            or recenter_witness.get("retained_generator_indices")
            != expected_indices
            or recenter_witness.get("deleted_generator_indices") != []):
        raise AssertionError("ranged-symbol recenter trace mismatch")
    if (output["native_range_kind"] != "absent"
            or recentered["native_range_kind"] != "absent"):
        raise AssertionError("ranged-symbol skip predicate is not established")
    if (recentered["weights"] != output["weights"]
            or recentered["ids"] != output["ids"]
            or recentered["ranges"] != output["ranges"]
            or recentered["masks"] != output["masks"]
            or recentered["numerical"] != output["numerical"]):
        raise AssertionError("ranged-symbol identity transition mismatch")
    # Byte/value identity plus the already established output relation proves
    # the recentered representation relation; do not re-traverse all entries.

    reduction_transition = graph["transition_records"][2]
    _seal(reduction_transition, "generator reduction transition")
    reduction_tau = reduction_transition.get("tau_k", {})
    reduction_witness = reduction_transition.get("operator_witness", {})
    input_count = len(recentered["ids"])
    # Reconstruct native control flow.  No producer-selected index list is
    # trusted: on this branch every ordered input generator must be retained.
    if input_count > PRODUCTION_MAX_ERROR_TERMS:
        raise AssertionError("witnessed native reduction branch is inadmissible")
    if (reduction_transition.get("operator_family") != "generator_reduction"
            or reduction_transition.get("input_state_ids")
            != [recentered["id"]]
            or reduction_transition.get("predecessor_state_id")
            != recentered["id"]
            or reduction_transition.get("output_state_ids") != [reduced["id"]]
            or reduction_tau.get("branch")
            != "no_reduction_input_count_le_maximum"
            or reduction_tau.get("maximum_error_terms")
            != PRODUCTION_MAX_ERROR_TERMS
            or reduction_tau.get("input_generator_count") != input_count
            or reduction_tau.get("input_special_prefix_count") != 0
            or reduction_tau.get("native_return_identity") is not True
            or reduction_tau.get("metric_policy")
            != "not_evaluated_on_identity_branch"
            or reduction_tau.get("ranking_quantities_hex") != []
            or reduction_tau.get("tie_breaking") != "not_applicable"
            or reduction_tau.get("retained_generator_indices")
            != list(range(input_count))
            or reduction_tau.get("removed_generator_indices") != []
            or reduction_tau.get("replacement_coordinate_flat_indices") != []
            or reduction_tau.get("replacement_generator_ids") != []
            or reduction_tau.get("replacement_support_masks") != []
            or reduction_tau.get("output_generator_count") != input_count
            or reduction_witness.get("metric_input_weights_sha256")
            != recentered_commitment
            or reduction_witness.get("output_weights_sha256")
            != reduced_commitment
            or reduction_witness.get("replacement_construction")
            != "none_identity_branch"):
        raise AssertionError("native generator reduction trace mismatch")
    if (reduced["weights"] != recentered["weights"]
            or reduced["ids"] != recentered["ids"]
            or reduced["ranges"] != recentered["ranges"]
            or reduced["masks"] != recentered["masks"]
            or reduced["numerical"] != recentered["numerical"]):
        raise AssertionError("native no-reduction representation mismatch")
    # The native no-reduction output is identical to the accepted predecessor,
    # so its simulation relation follows without duplicating the full scan.

    q_required = _check_affine_projection(
        root, graph["transition_records"][3], reduced, q_projection, "Q")
    k_required = _check_affine_projection(
        root, graph["transition_records"][4], reduced, k_projection, "K")
    _check_head_mapping(
        graph["transition_records"][5], q_projection, q_heads, "Q")
    _check_head_mapping(
        graph["transition_records"][6], k_projection, k_heads, "K")

    qk_transition = graph["transition_records"][7]
    _seal(qk_transition, "QK transition")
    qk_tau = qk_transition.get("tau_k", {})
    qk_witness = qk_transition.get("operator_witness", {})
    generators = len(reduced["ids"])
    fresh = 4 * 4 * 4
    if (qk_transition.get("operator_family") != "QK"
            or qk_transition.get("input_state_ids")
            != [q_heads["id"], k_heads["id"]]
            or qk_transition.get("predecessor_state_id") != reduced["id"]
            or qk_transition.get("output_state_ids") != [qk_output["id"]]
            or qk_tau.get("heads") != 4
            or qk_tau.get("query_tokens") != 4
            or qk_tau.get("key_tokens") != 4
            or qk_tau.get("inner_width") != 32
            or qk_tau.get("retained_generator_count") != generators
            or qk_tau.get("fresh_generator_count") != fresh
            or qk_tau.get("numerical_backend")
            != "checker_only_rigorous_fp64"
            or qk_witness.get("input_q_weights_sha256")
            != q_heads_commitment
            or qk_witness.get("input_k_weights_sha256")
            != k_heads_commitment
            or qk_witness.get("output_weights_sha256") != qk_commitment
            or q_heads["shape"] != (4, 901, 4, 32)
            or k_heads["shape"] != (4, 901, 4, 32)
            or qk_output["shape"] != (4, 965, 4, 4)):
        raise AssertionError("native precise QK trace mismatch")
    if (q_heads["ids"] != reduced["ids"]
            or k_heads["ids"] != reduced["ids"]
            or q_heads["ranges"] != reduced["ranges"]
            or k_heads["ranges"] != reduced["ranges"]
            or q_heads["masks"] != reduced["masks"]
            or k_heads["masks"] != reduced["masks"]):
        raise AssertionError("QK predecessor generator continuity mismatch")
    expected_fresh_ids = [f"qk_0_fresh_{index:06d}"
                          for index in range(fresh)]
    if (qk_tau.get("fresh_generator_ids") != expected_fresh_ids
            or qk_output["ids"] != reduced["ids"] + expected_fresh_ids
            or qk_output["ranges"] != [(-1.0, 1.0)] * (generators + fresh)):
        raise AssertionError("QK fresh generator identity/order mismatch")
    expected_retained_masks = [15 if mask else 0 for mask in reduced["masks"]]
    expected_fresh_masks = [1 << query for _head in range(4)
                            for query in range(4) for _key in range(4)]
    if qk_output["masks"] != expected_retained_masks + expected_fresh_masks:
        raise AssertionError("QK structural support transition mismatch")
    diagnostics = qk_witness.get("structural_diagnostics", {})
    nominal = 4 * 4 * 4 * generators * generators * 32
    executed = diagnostics.get("executed_quadratic_MACs")
    skipped = diagnostics.get("structurally_skipped_quadratic_MACs")
    if (diagnostics.get("schema") != "CORET_STRUCTURAL_SUPPORT_PRECISE_DOT_V1"
            or diagnostics.get("mode") != "QK"
            or diagnostics.get("support_validated") is not True
            or diagnostics.get("nominal_quadratic_MACs") != nominal
            or not isinstance(executed, int) or not isinstance(skipped, int)
            or executed < 0 or skipped < 0 or executed + skipped != nominal
            or diagnostics.get("effective_generator_tile") != 0
            or diagnostics.get("temporary_cap_bytes") != 0):
        raise AssertionError("QK structural execution inventory mismatch")

    precise_witness = qk_witness.get("independent_numerical_witness")
    if not isinstance(precise_witness, dict):
        raise AssertionError("QK independent numerical witness missing")
    if (precise_witness.get("left", {}).get("sha256") != q_heads_commitment
            or precise_witness.get("right", {}).get("sha256")
            != k_heads_commitment
            or precise_witness.get("output", {}).get("sha256")
            != qk_commitment):
        raise AssertionError("QK numerical witness/state continuity mismatch")
    if os.environ.get("CORET_PRECISE_DOT_NUMERICAL_BACKEND") != "rigorous_fp64":
        raise AssertionError("QK checker must use accepted rigorous FP64 backend")
    from coret_native_numerical_checker_v1 import check_precise_dot_witness
    rigorous = check_precise_dot_witness(precise_witness, root)
    if (rigorous.get("accepted") is not True
            or rigorous.get("numerical_backend")
            != "checker_only_rigorous_fp64"):
        raise AssertionError("rigorous QK numerical checker rejected")
    qk_required = _check_qk_sidecar(q_heads, k_heads, qk_output)

    if graph.get("final_property_record") is not None:
        raise AssertionError("prefix trace must not claim a property")
    return {
        "accepted": True,
        "PRODUCTION_SOURCE_DOMAIN_PASS": True,
        "PRODUCTION_AFFINE_RESIDUAL_PREFIX_PASS": True,
        "FIRST_PRODUCTION_LAYERNORM_TRACE_PASS": True,
        "PRODUCTION_PREFIX_STATE_CONTINUITY_PASS": True,
        "PRODUCTION_PREFIX_BITWISE_EQUIVALENCE_PASS": True,
        "PRODUCTION_RANGED_SYMBOL_TRACE_PASS": True,
        "PRODUCTION_GENERATOR_REDUCTION_TRACE_PASS": True,
        "PRODUCTION_REDUCTION_STATE_CONTINUITY_PASS": True,
        "PRODUCTION_REDUCTION_BITWISE_EQUIVALENCE_PASS": True,
        "PRODUCTION_Q_PROJECTION_PASS": True,
        "PRODUCTION_K_PROJECTION_PASS": True,
        "PRODUCTION_HEAD_MAPPING_PASS": True,
        "PRODUCTION_QK_NUMERICAL_CHECK_PASS": True,
        "PRODUCTION_QK_STATE_CONTINUITY_PASS": True,
        "PRODUCTION_QK_BITWISE_EQUIVALENCE_PASS": True,
        "input_generator_count": input_count,
        "output_generator_count": len(reduced["ids"]),
        "retained_count": input_count,
        "removed_count": 0,
        "replacement_count": 0,
        "q_projection_required_sidecar": str(q_required),
        "k_projection_required_sidecar": str(k_required),
        "qk_required_sidecar": str(qk_required),
        "rigorous_qk_checker_seconds": rigorous["backend"]["wall_seconds"],
        "q_shape": list(q_heads["shape"]),
        "k_shape": list(k_heads["shape"]),
        "qk_shape": list(qk_output["shape"]),
        "qk_generator_count": len(qk_output["ids"]),
        "qk_fresh_count": fresh,
        "states": 9, "transitions": 8,
    }
