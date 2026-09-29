#!/usr/bin/env python3
"""Torch-free 256-bit MPFR checker for native numerical witnesses V1.

The checker intentionally imports neither Torch, DeepT, NumPy, nor any CoReT
producer module.  It reads exact IEEE tensor bits, reconstructs current native
precise-dot equations with directed MPFR arithmetic, validates structural-zero
claims directly against the operands, and checks set containment.  Producer
``finite`` flags and FP-envelope summaries are ignored.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import gmpy2


PRECISION = 256
SCHEMA = "CORET_NATIVE_NUMERICAL_WITNESS_V1"
BLOCK_SCHEMA = "CORET_EXACT_TENSOR_BLOCK_V1"


def _context(rounding):
    return gmpy2.local_context(
        gmpy2.context(), precision=PRECISION, round=rounding)


def _mp(value):
    if isinstance(value, gmpy2.mpfr):
        return value
    with _context(gmpy2.RoundToNearest):
        return gmpy2.mpfr(value)


def _exact_float(value: float):
    # Python binary64 represents every decoded binary32 value exactly.
    return _mp(value)


def _down(function):
    with _context(gmpy2.RoundDown):
        return +function()


def _up(function):
    with _context(gmpy2.RoundUp):
        return +function()


@dataclass(frozen=True)
class Interval:
    lo: gmpy2.mpfr
    hi: gmpy2.mpfr

    def __init__(self, lo, hi=None):
        object.__setattr__(self, "lo", _mp(lo))
        object.__setattr__(self, "hi", _mp(lo if hi is None else hi))
        if self.lo > self.hi:
            raise ValueError("reversed interval")

    def __add__(self, other):
        other = _as_interval(other)
        return Interval(_down(lambda: self.lo + other.lo),
                        _up(lambda: self.hi + other.hi))

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + (-_as_interval(other))

    def __mul__(self, other):
        other = _as_interval(other)
        lows = [_down(lambda a=a, b=b: a * b)
                for a in (self.lo, self.hi)
                for b in (other.lo, other.hi)]
        highs = [_up(lambda a=a, b=b: a * b)
                 for a in (self.lo, self.hi)
                 for b in (other.lo, other.hi)]
        return Interval(min(lows), max(highs))

    def __truediv__(self, other):
        return self * _as_interval(other).reciprocal()

    def square(self):
        if self.lo <= 0 <= self.hi:
            return Interval(0, max(_up(lambda: self.lo * self.lo),
                                   _up(lambda: self.hi * self.hi)))
        values = [_down(lambda x=x: x * x) for x in (self.lo, self.hi)]
        upper = [_up(lambda x=x: x * x) for x in (self.lo, self.hi)]
        return Interval(min(values), max(upper))

    def sqrt(self):
        if self.lo < 0:
            raise ValueError("negative sqrt domain")
        return Interval(_down(lambda: gmpy2.sqrt(self.lo)),
                        _up(lambda: gmpy2.sqrt(self.hi)))

    def reciprocal(self):
        if self.lo <= 0 <= self.hi:
            raise ValueError("reciprocal domain crosses zero")
        lows = [_down(lambda x=x: 1 / x) for x in (self.lo, self.hi)]
        highs = [_up(lambda x=x: 1 / x) for x in (self.lo, self.hi)]
        return Interval(min(lows), max(highs))

    def exp(self):
        return Interval(_down(lambda: gmpy2.exp(self.lo)),
                        _up(lambda: gmpy2.exp(self.hi)))

    def tanh(self):
        return Interval(_down(lambda: gmpy2.tanh(self.lo)),
                        _up(lambda: gmpy2.tanh(self.hi)))


def _as_interval(value):
    return value if isinstance(value, Interval) else Interval(value)


def _abs_upper(value: Interval):
    return max(abs(value.lo), abs(value.hi))


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _verify_canonical(value: dict[str, Any], label: str):
    claimed = value.get("canonical_sha256")
    body = dict(value)
    body.pop("canonical_sha256", None)
    actual = hashlib.sha256(_canonical(body)).hexdigest()
    if claimed != actual:
        raise AssertionError(f"{label} canonical digest mismatch")


@dataclass(frozen=True)
class TensorBlock:
    shape: tuple[int, ...]
    values: tuple[float, ...]
    dtype: str
    raw: bytes | None = None

    def at(self, *indices: int) -> float:
        if len(indices) != len(self.shape):
            raise IndexError("tensor index rank differs")
        offset = 0
        for index, size in zip(indices, self.shape):
            if index < 0 or index >= size:
                raise IndexError("tensor index outside shape")
            offset = offset * size + index
        return self.values[offset]


def _load_block(root: Path, record: dict[str, Any]) -> TensorBlock:
    if record.get("schema") != BLOCK_SCHEMA:
        raise AssertionError("tensor block schema mismatch")
    if record.get("byte_order") != "little":
        raise AssertionError("tensor block byte order mismatch")
    relative = Path(record["relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise AssertionError("tensor block path escapes witness root")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise AssertionError("tensor block path escapes witness root") from error
    raw = path.read_bytes()
    if len(raw) != int(record["byte_count"]):
        raise AssertionError("tensor block byte count mismatch")
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise AssertionError("tensor block hash mismatch")
    dtype = record["dtype"]
    if dtype == "float32":
        code, width = "f", 4
    elif dtype == "float64":
        code, width = "d", 8
    else:
        raise AssertionError("unsupported tensor block dtype")
    shape = tuple(int(item) for item in record["shape"])
    count = math.prod(shape)
    if count != int(record["element_count"]) or count * width != len(raw):
        raise AssertionError("tensor block shape/count mismatch")
    values = tuple(item[0] for item in struct.iter_unpack("<" + code, raw))
    if len(values) != count or not all(math.isfinite(item) for item in values):
        raise AssertionError("nonfinite or incomplete tensor block")
    return TensorBlock(shape, values, dtype, raw)


def _support(record: dict[str, Any], expected_count: int, label: str):
    _verify_canonical(record, label)
    masks = tuple(int(item) for item in record["masks"])
    ids = tuple(str(item) for item in record["ids"])
    reasons = tuple(str(item) for item in record["reasons"])
    tokens = int(record["num_tokens"])
    if not (len(masks) == len(ids) == len(reasons) == expected_count):
        raise AssertionError(f"{label} support lengths differ")
    if len(set(ids)) != len(ids):
        raise AssertionError(f"{label} duplicate provenance ID")
    limit = (1 << tokens) - 1
    if any(mask < 0 or mask > limit for mask in masks):
        raise AssertionError(f"{label} support mask outside token universe")
    return masks, ids, reasons, tokens


def _ranges(record: dict[str, Any], expected_count: int, label: str):
    _verify_canonical(record, label)
    if int(record["count"]) != expected_count:
        raise AssertionError(f"{label} range count mismatch")
    if record["kind"] == "implicit_minus1_plus1":
        return tuple((_mp(-1), _mp(1)) for _ in range(expected_count))
    if record["kind"] != "explicit":
        raise AssertionError(f"{label} unknown range kind")
    lows, highs = record["low_hex"], record["high_hex"]
    if len(lows) != expected_count or len(highs) != expected_count:
        raise AssertionError(f"{label} explicit range length mismatch")
    result = []
    for lo, hi in zip(lows, highs):
        lower, upper = _mp(float.fromhex(lo)), _mp(float.fromhex(hi))
        if lower > upper:
            raise AssertionError(f"{label} reversed range")
        result.append((lower, upper))
    return tuple(result)


def _sum_intervals(values: Iterable[Interval]) -> Interval:
    result = Interval(0)
    for value in values:
        result = result + value
    return result


def _dot(left: Iterable[float], right: Iterable[float]) -> Interval:
    return _sum_intervals(
        Interval(_exact_float(a)) * Interval(_exact_float(b))
        for a, b in zip(left, right))


def _vector(tensor: TensorBlock, head: int, generator: int,
            row: int) -> tuple[float, ...]:
    return tuple(tensor.at(head, generator, row, column)
                 for column in range(tensor.shape[3]))


def _check_support_zeros(tensor: TensorBlock, masks: tuple[int, ...],
                         token_axis: int, label: str):
    heads, generators_plus_center, first, second = tensor.shape
    if generators_plus_center != 1 + len(masks):
        raise AssertionError(f"{label} generator dimension mismatch")
    tokens = first if token_axis == 2 else second
    for generator, mask in enumerate(masks):
        for token in range(tokens):
            if mask & (1 << token):
                continue
            for head in range(heads):
                width = second if token_axis == 2 else first
                for other in range(width):
                    value = (tensor.at(head, 1 + generator, token, other)
                             if token_axis == 2 else
                             tensor.at(head, 1 + generator, other, token))
                    if value != 0.0:
                        raise AssertionError(
                            f"{label} coefficient outside claimed support")


def _expected_output_support(mode, left, right, left_ids, right_ids,
                             heads, rows, columns, tokens):
    gmax = max(len(left), len(right))
    all_tokens = (1 << tokens) - 1
    inherited_masks = []
    inherited_ids = []
    for index in range(gmax):
        lm = left[index] if index < len(left) else 0
        rm = right[index] if index < len(right) else 0
        lid = left_ids[index] if index < len(left_ids) else None
        rid = right_ids[index] if index < len(right_ids) else None
        if lid is not None and rid is not None and lid != rid:
            raise AssertionError("aligned precise-dot provenance IDs differ")
        inherited_ids.append(lid if lid is not None else rid)
        inherited_masks.append(
            all_tokens if (lm or rm) and mode == "QK"
            else (all_tokens if rm else lm))
    fresh_masks = tuple(1 << row for _head in range(heads)
                        for row in range(rows) for _column in range(columns))
    return tuple(inherited_masks) + fresh_masks, tuple(inherited_ids)


def _check_precise_dot_fresh_layout(output, heads, gmax, fresh,
                                    rows, columns):
    """Verify native coordinate-local fresh row ownership exactly."""
    for output_head in range(heads):
        for fresh_offset in range(fresh):
            index = 1 + gmax + fresh_offset
            owner_head = fresh_offset // (rows * columns)
            within_head = fresh_offset % (rows * columns)
            owner_row, owner_column = divmod(within_head, columns)
            for row in range(rows):
                for column in range(columns):
                    is_own = (output_head == owner_head
                              and row == owner_row
                              and column == owner_column)
                    if (not is_own
                            and output.at(output_head, index, row, column) != 0.0):
                        raise AssertionError("precise-dot fresh layout mismatch")


def check_precise_dot_witness(witness: dict[str, Any], root: str | Path):
    """Check current QK/A.V set containment from exact operand bits."""
    if witness.get("schema") != SCHEMA or witness.get("kind") != "precise_dot_structural":
        raise AssertionError("precise-dot witness schema mismatch")
    _verify_canonical(witness, "precise-dot witness")
    mode = witness.get("family")
    if mode not in {"QK", "A.V"}:
        raise AssertionError("precise-dot family mismatch")
    root = Path(root)
    left = _load_block(root, witness["left"])
    right = _load_block(root, witness["right"])
    output = _load_block(root, witness["output"])
    if witness.get("frozen_output_sha256") != witness["output"].get("sha256"):
        raise AssertionError("frozen precise-dot output commitment mismatch")
    if len(left.shape) != 4 or len(right.shape) != 4 or len(output.shape) != 4:
        raise AssertionError("precise-dot tensor rank mismatch")
    if left.shape[0] != right.shape[0] or left.shape[0] != output.shape[0]:
        raise AssertionError("precise-dot head count mismatch")
    ga, gb = int(witness["left_generator_count"]), int(witness["right_generator_count"])
    if any(int(witness[name]) != 0 for name in (
            "left_special_prefix_count", "right_special_prefix_count",
            "output_special_prefix_count")):
        raise AssertionError(
            "precise-dot checker V1 supports only the frozen L-infinity path")
    if left.shape[1] != 1 + ga or right.shape[1] != 1 + gb:
        raise AssertionError("precise-dot input generator count mismatch")
    gmax, gmin = max(ga, gb), min(ga, gb)
    heads, rows, inner = left.shape[0], left.shape[2], left.shape[3]
    if mode == "QK":
        columns = right.shape[2]
        if right.shape[3] != inner:
            raise AssertionError("QK inner dimension mismatch")
        right_token_axis = 2
    else:
        columns = right.shape[2]
        if right.shape[3] != inner:
            raise AssertionError("A.V key dimension mismatch")
        right_token_axis = 3
    fresh = heads * rows * columns
    if output.shape != (heads, 1 + gmax + fresh, rows, columns):
        raise AssertionError("precise-dot output shape/fresh count mismatch")
    if int(witness["output_generator_count"]) != gmax + fresh:
        raise AssertionError("precise-dot output generator metadata mismatch")

    lm, li, _lr, tokens = _support(
        witness["left_support"], ga, "left")
    rm, ri, _rr, rtokens = _support(
        witness["right_support"], gb, "right")
    om, oi, _or, otokens = _support(
        witness["output_support"], gmax + fresh, "output")
    if tokens != rtokens or tokens != otokens or rows != tokens:
        raise AssertionError("precise-dot token universe mismatch")
    left_ranges = _ranges(witness["left_ranges"], ga, "left")
    _ranges(witness["right_ranges"], gb, "right")
    output_ranges = _ranges(
        witness["output_ranges"], gmax + fresh, "output")
    if witness["left_ranges"]["kind"] == "implicit_minus1_plus1":
        if witness["output_ranges"]["kind"] != "implicit_minus1_plus1":
            raise AssertionError("precise-dot output range policy mismatch")
    else:
        expected_ranges = left_ranges + tuple(
            (_mp(-1), _mp(1)) for _ in range(gmax + fresh - ga))
        if output_ranges != expected_ranges:
            raise AssertionError("precise-dot output range policy mismatch")
    _check_support_zeros(left, lm, 2, "left")
    _check_support_zeros(right, rm, right_token_axis, "right")
    expected_masks, inherited_ids = _expected_output_support(
        mode, lm, rm, li, ri, heads, rows, columns, tokens)
    if om != expected_masks:
        raise AssertionError("precise-dot output support transition mismatch")
    if oi[:gmax] != inherited_ids:
        raise AssertionError("precise-dot retained provenance order mismatch")
    prefix = "qk_" if mode == "QK" else "av_"
    if any(not identifier.startswith(prefix) for identifier in oi[gmax:]):
        raise AssertionError("precise-dot fresh provenance family mismatch")

    numerical_backend = os.environ.get(
        "CORET_PRECISE_DOT_NUMERICAL_BACKEND", "mpfr")
    if numerical_backend == "rigorous_fp64":
        if not (left.dtype == right.dtype == output.dtype == "float32"):
            raise AssertionError(
                "rigorous FP64 backend supports only frozen float32 precise dot")
        if left.raw is None or right.raw is None or output.raw is None:
            raise AssertionError("rigorous FP64 backend requires immutable raw blocks")
        from coret_rigorous_precise_dot_backend_v1 import check_f32
        accelerated = check_f32(
            left_raw=left.raw, right_raw=right.raw, output_raw=output.raw,
            left_shape=left.shape, right_shape=right.shape,
            output_shape=output.shape,
            device_index=int(os.environ.get("CORET_CHECKER_CUDA_DEVICE", "0")))
        if not accelerated["accepted"]:
            raise AssertionError(
                f"{mode} fresh radius does not enclose rigorous FP64 transition")
        _check_precise_dot_fresh_layout(
            output, heads, gmax, fresh, rows, columns)
        return {
            "accepted": True, "family": mode,
            "precision_bits": PRECISION,
            "coordinates_checked": accelerated["coordinates_checked"],
            "strict_no_empirical_epsilon": True,
            "numerical_backend": "checker_only_rigorous_fp64",
            "backend": accelerated,
        }
    if numerical_backend != "mpfr":
        raise AssertionError("unknown precise-dot numerical backend")

    half = Interval(_mp(1) / 2)
    checked = 0
    maximum_slack = _mp(0)
    for head in range(heads):
        for row in range(rows):
            av0 = _vector(left, head, 0, row)
            for column in range(columns):
                bv0 = _vector(right, head, 0, column)
                exact_center = _dot(av0, bv0)
                for generator in range(gmin):
                    exact_center = exact_center + half * _dot(
                        _vector(left, head, 1 + generator, row),
                        _vector(right, head, 1 + generator, column))
                stored_center = _exact_float(output.at(head, 0, row, column))
                required = _abs_upper(exact_center - Interval(stored_center))

                for generator in range(gmax):
                    expected = Interval(0)
                    if generator < gb:
                        expected = expected + _dot(
                            av0, _vector(right, head, 1 + generator, column))
                    if generator < ga:
                        expected = expected + _dot(
                            _vector(left, head, 1 + generator, row), bv0)
                    stored = _exact_float(output.at(
                        head, 1 + generator, row, column))
                    required = _up(lambda required=required, expected=expected,
                                   stored=stored:
                                   required + _abs_upper(expected-Interval(stored)))

                radius = _mp(0)
                for first in range(gmin):
                    diagonal = _dot(
                        _vector(left, head, 1 + first, row),
                        _vector(right, head, 1 + first, column))
                    radius = _up(lambda radius=radius, diagonal=diagonal:
                                 radius + _abs_upper(diagonal) / 2)
                # Cross terms range over the aligned padded gmax universe.
                # This matters for A.V: probability may have a suffix of
                # symbols absent from V, yet each suffix a_i still interacts
                # with every earlier, present b_j through a_i.b_j.
                for first in range(gmax):
                    for second in range(first + 1, gmax):
                        cross = Interval(0)
                        if first < ga and second < gb:
                            cross = cross + _dot(
                                _vector(left, head, 1 + first, row),
                                _vector(right, head, 1 + second, column))
                        if second < ga and first < gb:
                            cross = cross + _dot(
                                _vector(left, head, 1 + second, row),
                                _vector(right, head, 1 + first, column))
                        radius = _up(lambda radius=radius, cross=cross:
                                     radius + _abs_upper(cross))
                required = _up(lambda: required + radius)
                fresh_index = 1 + gmax + head * rows * columns + row * columns + column
                stored_radius = _exact_float(output.at(
                    head, fresh_index, row, column))
                if stored_radius < required or stored_radius < 0:
                    raise AssertionError(
                        f"{mode} fresh radius does not enclose exact-real transition")
                maximum_slack = max(maximum_slack, stored_radius - required)
                checked += 1
    _check_precise_dot_fresh_layout(
        output, heads, gmax, fresh, rows, columns)
    return {
        "accepted": True, "family": mode,
        "precision_bits": PRECISION,
        "coordinates_checked": checked,
        "strict_no_empirical_epsilon": True,
        "maximum_soundness_slack": str(maximum_slack),
        "numerical_backend": "bounded_mpfr_oracle",
    }


def check_scalar_interval_witness(witness: dict[str, Any]):
    if witness.get("schema") != SCHEMA or witness.get("kind") != "scalar_interval":
        raise AssertionError("scalar witness schema mismatch")
    _verify_canonical(witness, "scalar witness")
    lower = Interval(_mp(float.fromhex(witness["input_lower_hex"])))
    upper = Interval(_mp(float.fromhex(witness["input_upper_hex"])))
    domain = Interval(lower.lo, upper.hi)
    family = witness["family"]
    if family == "sqrt":
        expected = domain.sqrt()
    elif family == "reciprocal":
        expected = domain.reciprocal()
    elif family == "exp":
        expected = domain.exp()
    elif family == "tanh":
        expected = domain.tanh()
    else:
        raise AssertionError("unsupported scalar witness family")
    stored_lower = _mp(float.fromhex(witness["output_lower_hex"]))
    stored_upper = _mp(float.fromhex(witness["output_upper_hex"]))
    if stored_lower > expected.lo or stored_upper < expected.hi:
        raise AssertionError("scalar interval does not contain directed MPFR image")
    return {"accepted": True, "family": family,
            "precision_bits": PRECISION,
            "strict_no_empirical_epsilon": True}


def _hex_point(value: str, label: str) -> Interval:
    try:
        decoded = float.fromhex(value)
    except (TypeError, ValueError) as error:
        raise AssertionError(f"{label} is not an IEEE hexadecimal value") from error
    if not math.isfinite(decoded):
        raise AssertionError(f"{label} is nonfinite")
    return Interval(_exact_float(decoded))


def _shadow_interval(stored: Interval, radius: Interval, label: str) -> Interval:
    if radius.lo != radius.hi or radius.lo < 0:
        raise AssertionError(f"{label} numerical radius is invalid")
    return Interval(
        _down(lambda: stored.lo - radius.lo),
        _up(lambda: stored.hi + radius.hi))


def _require_contains(container: Interval, exact: Interval, label: str):
    if container.lo > exact.lo or container.hi < exact.hi:
        raise AssertionError(f"{label} numerical sidecar is too narrow")


def check_numerical_shadow_scalar_chain_witness(witness: dict[str, Any]):
    """Independently check a composed representation-level error sidecar.

    This deliberately does not allocate a native abstract symbol.  Each
    stored native singleton is paired with a symmetric numerical interval,
    and that represented interval—not the bare stored float—is propagated to
    the next transformer with 256-bit directed MPFR arithmetic.
    """
    if (witness.get("schema") != SCHEMA
            or witness.get("kind") != "numerical_shadow_scalar_chain"):
        raise AssertionError("numerical shadow scalar witness schema mismatch")
    _verify_canonical(witness, "numerical shadow scalar witness")
    if witness.get("equation") != "sqrt(x);reciprocal;scale*y+shift":
        raise AssertionError("numerical shadow scalar equation mismatch")
    if witness.get("native_fresh_counts") != [0, 0, 0]:
        raise AssertionError("numerical sidecar changed native fresh membership")

    source = _hex_point(witness["input_hex"], "shadow input")
    if source.lo <= 0:
        raise AssertionError("shadow sqrt/reciprocal domain is not positive")
    stored_sqrt = _hex_point(witness["stored_sqrt_hex"], "stored sqrt")
    sqrt_radius = _hex_point(
        witness["sqrt_error_radius_hex"], "sqrt radius")
    represented_sqrt = _shadow_interval(
        stored_sqrt, sqrt_radius, "sqrt")
    exact_sqrt = source.sqrt()
    _require_contains(represented_sqrt, exact_sqrt, "sqrt")

    # Composition is checked from the whole represented predecessor interval.
    # This prevents a witness from attaching a valid sqrt error and silently
    # discarding it at reciprocal.
    reciprocal_image = represented_sqrt.reciprocal()
    stored_reciprocal = _hex_point(
        witness["stored_reciprocal_hex"], "stored reciprocal")
    reciprocal_radius = _hex_point(
        witness["reciprocal_error_radius_hex"], "reciprocal radius")
    represented_reciprocal = _shadow_interval(
        stored_reciprocal, reciprocal_radius, "reciprocal")
    _require_contains(
        represented_reciprocal, reciprocal_image, "reciprocal")

    scale = _hex_point(witness["scale_hex"], "LayerNorm scale")
    shift = _hex_point(witness["shift_hex"], "LayerNorm shift")
    affine_image = scale * represented_reciprocal + shift
    stored_output = _hex_point(witness["stored_output_hex"], "stored output")
    output_radius = _hex_point(
        witness["output_error_radius_hex"], "output radius")
    represented_output = _shadow_interval(
        stored_output, output_radius, "LayerNorm affine output")
    _require_contains(
        represented_output, affine_image, "LayerNorm affine output")

    return {
        "accepted": True,
        "family": "numerical_shadow_sqrt_reciprocal_layernorm_affine",
        "precision_bits": PRECISION,
        "native_fresh_counts": [0, 0, 0],
        "sqrt_error_radius": str(sqrt_radius.hi),
        "reciprocal_error_radius": str(reciprocal_radius.hi),
        "output_error_radius": str(output_radius.hi),
        "strict_no_empirical_epsilon": True,
    }


def check_ghost_precise_square_witness(witness: dict[str, Any]):
    """Check the pinned scalar precise-dot equation in a ghost universe.

    This is the fundamental correlation gate: the ordered immutable ghost IDs
    determine which logical generators are diagonal and which pairs are
    off-diagonal.  Coordinate concretization is never used to infer that
    information.
    """
    if (witness.get("schema") != SCHEMA
            or witness.get("kind") != "ghost_precise_square"):
        raise AssertionError("ghost precise-square witness schema mismatch")
    _verify_canonical(witness, "ghost precise-square witness")
    expected_equation = (
        "center=c*c+0.5*sum_i(ai*ai);"
        "retained_i=2*c*ai;"
        "fresh=0.5*sum_i|ai*ai|+sum_i<j|2*ai*aj|")
    if witness.get("equation") != expected_equation:
        raise AssertionError("ghost precise-square equation mismatch")

    center = _hex_point(witness["center_hex"], "ghost square center")
    coefficients = []
    identifiers = []
    for row in witness.get("coefficients", []):
        identifier = str(row["ghost_id"])
        if identifier in identifiers:
            raise AssertionError("duplicate ghost source ID")
        identifiers.append(identifier)
        low = _hex_point(row["range_low_hex"], "ghost range lower")
        high = _hex_point(row["range_high_hex"], "ghost range upper")
        if low.lo != -1 or high.hi != 1:
            raise AssertionError("precise-square ghost range must be [-1,1]")
        coefficients.append(_hex_point(
            row["value_hex"], "ghost square coefficient"))

    retained = witness.get("stored_retained", [])
    if [str(row["ghost_id"]) for row in retained] != identifiers:
        raise AssertionError("ghost retained identity/order mismatch")
    half = Interval(_mp(1) / 2)
    expected_center = center * center
    expected_retained = []
    fresh = Interval(0)
    for coefficient in coefficients:
        diagonal = coefficient * coefficient
        expected_center = expected_center + half * diagonal
        fresh = fresh + half * Interval(
            -_abs_upper(diagonal), _abs_upper(diagonal))
        expected_retained.append(Interval(2) * center * coefficient)
    # ``fresh`` above is symmetric; accumulate its nonnegative radius and the
    # off-diagonal absolute radii separately to avoid losing sign information.
    fresh_radius = sum(
        (_abs_upper(coefficient * coefficient) / 2
         for coefficient in coefficients), _mp(0))
    for first in range(len(coefficients)):
        for second in range(first + 1, len(coefficients)):
            cross = Interval(2) * coefficients[first] * coefficients[second]
            fresh_radius = _up(
                lambda fresh_radius=fresh_radius, cross=cross:
                fresh_radius + _abs_upper(cross))

    stored_center = _hex_point(
        witness["stored_center_hex"], "stored ghost square center")
    if stored_center.lo > expected_center.lo or stored_center.hi < expected_center.hi:
        raise AssertionError("ghost precise-square center is too narrow")
    for row, expected in zip(retained, expected_retained):
        stored = _hex_point(row["value_hex"], "stored retained coefficient")
        if stored.lo > expected.lo or stored.hi < expected.hi:
            raise AssertionError("ghost precise-square retained coefficient is too narrow")
    stored_fresh = _hex_point(
        witness["stored_fresh_hex"], "stored ghost square fresh radius")
    if stored_fresh.lo < 0 or stored_fresh.hi < fresh_radius:
        raise AssertionError("ghost precise-square fresh radius is too narrow")
    lower = _down(lambda: expected_center.lo - fresh_radius)
    upper = _up(lambda: expected_center.hi + fresh_radius)
    return {
        "accepted": True,
        "family": "ghost_precise_square",
        "precision_bits": PRECISION,
        "ghost_ids": identifiers,
        "abstract_lower": str(lower),
        "abstract_upper": str(upper),
        "fresh_radius": str(fresh_radius),
    }


def _flat_spatial(tensor: TensorBlock, generator: int, coordinate: int) -> float:
    spatial = math.prod(tensor.shape[1:])
    if generator < 0 or generator >= tensor.shape[0]:
        raise IndexError("generator outside tensor")
    if coordinate < 0 or coordinate >= spatial:
        raise IndexError("coordinate outside tensor")
    return tensor.values[generator * spatial + coordinate]


def _exact_concretize_coordinate(tensor: TensorBlock, ranges, coordinate: int):
    center = Interval(_exact_float(_flat_spatial(tensor, 0, coordinate)))
    lower, upper = center, center
    for generator, (range_low, range_high) in enumerate(ranges):
        coefficient = Interval(_exact_float(
            _flat_spatial(tensor, 1 + generator, coordinate)))
        first = coefficient * Interval(range_low)
        second = coefficient * Interval(range_high)
        contribution = Interval(
            min(first.lo, second.lo), max(first.hi, second.hi))
        lower = Interval(_down(lambda: lower.lo + contribution.lo),
                         _up(lambda: lower.hi + contribution.lo))
        upper = Interval(_down(lambda: upper.lo + contribution.hi),
                         _up(lambda: upper.hi + contribution.hi))
    return lower, upper


def _sqrt_relaxation(lower: Interval, upper: Interval):
    """Directed reconstruction of pinned DeepT ``Zonotope.sqrt``.

    This is the mathematical transition, not a replay of float32 execution.
    Its returned intervals enclose every 256-bit rounding of the exact formula.
    """
    if lower.lo <= 0:
        raise AssertionError("native affine sqrt domain is not positive")
    if lower.lo == upper.hi:
        value = Interval(lower.lo).sqrt()
        return value, tuple(), None
    l = Interval(lower.lo)
    u = Interval(upper.hi)
    sqrt_l, sqrt_u = l.sqrt(), u.sqrt()
    slope = (sqrt_u - sqrt_l) / (u - l)
    tcrit = ((u - l) / (Interval(2) * (sqrt_u - sqrt_l))).square()
    x_intercept = sqrt_l - slope * l
    sqrt_tcrit = tcrit.sqrt()
    constant = Interval(_mp(1) / 2) * (
        sqrt_tcrit - slope * tcrit + x_intercept)
    fresh = Interval(_mp(1) / 2) * (
        slope * tcrit - sqrt_tcrit + x_intercept)
    return slope, (constant, fresh), tcrit


def check_native_affine_sqrt_witness(witness: dict[str, Any], root: str | Path,
                                     _allow_shadow: bool = False):
    """Check the actual affine/fresh-symbol transition of native sqrt.

    Unlike :func:`check_scalar_interval_witness`, this checks the center,
    every retained generator, the coordinate-local fresh generator layout,
    and whether the stored fresh magnitude absorbs all coefficient-rounding
    error relative to the exact-real native relaxation.
    """
    expected_kind = ("native_affine_sqrt_shadow" if _allow_shadow
                     else "native_affine_sqrt")
    if (witness.get("schema") != SCHEMA
            or witness.get("kind") != expected_kind):
        raise AssertionError("native affine sqrt witness schema mismatch")
    _verify_canonical(witness, "native affine sqrt witness")
    root = Path(root)
    source = _load_block(root, witness["input"])
    output = _load_block(root, witness["output"])
    if witness.get("frozen_output_sha256") != witness["output"].get("sha256"):
        raise AssertionError("frozen native affine sqrt output commitment mismatch")
    if len(source.shape) != 3 or len(output.shape) != 3:
        raise AssertionError("native affine sqrt requires rank-three weights")
    generators = int(witness["input_generator_count"])
    if source.shape[0] != 1 + generators:
        raise AssertionError("native affine sqrt input generator count mismatch")
    if source.shape[1:] != output.shape[1:]:
        raise AssertionError("native affine sqrt spatial shape mismatch")
    if int(witness.get("input_special_prefix_count", -1)) != 0:
        raise AssertionError("native affine sqrt checker supports frozen L-infinity only")
    source_ranges = _ranges(witness["input_ranges"], generators, "sqrt input")
    spatial = math.prod(source.shape[1:])
    shadow_radii = None
    if _allow_shadow:
        encoded = witness.get("coordinate_numerical_radius_hex")
        if not isinstance(encoded, list) or len(encoded) != spatial:
            raise AssertionError("native affine sqrt shadow radius shape mismatch")
        shadow_radii = []
        for item in encoded:
            radius = _hex_point(item, "native affine sqrt shadow radius")
            if radius.lo < 0:
                raise AssertionError("negative native affine sqrt shadow radius")
            shadow_radii.append(radius.hi)

    reconstructed = []
    for coordinate in range(spatial):
        lower, upper = _exact_concretize_coordinate(
            source, source_ranges, coordinate)
        reconstructed.append((lower, upper, lower.lo != upper.hi))
    active = tuple(index for index, item in enumerate(reconstructed) if item[2])
    recorded_active = tuple(int(item) for item in witness["fresh_flat_indices"])
    if recorded_active != active:
        raise AssertionError("native affine sqrt fresh membership/order mismatch")
    fresh_count = len(active)
    if output.shape[0] != 1 + generators + fresh_count:
        raise AssertionError("native affine sqrt output generator count mismatch")
    if int(witness["output_generator_count"]) != generators + fresh_count:
        raise AssertionError("native affine sqrt output metadata mismatch")
    output_ranges = _ranges(
        witness["output_ranges"], generators + fresh_count, "sqrt output")
    expected_ranges = source_ranges + tuple(
        (_mp(-1), _mp(1)) for _ in range(fresh_count))
    if output_ranges != expected_ranges:
        raise AssertionError("native affine sqrt output range policy mismatch")

    owner = {coordinate: offset for offset, coordinate in enumerate(active)}
    maximum_deficit = _mp(0)
    for coordinate, (lower, upper, different) in enumerate(reconstructed):
        center = Interval(_exact_float(_flat_spatial(source, 0, coordinate)))
        if not different:
            expected_center = Interval(lower.lo).sqrt()
            expected_retained = [Interval(0) for _ in range(generators)]
            expected_fresh = Interval(0)
        else:
            slope, (constant, expected_fresh), _tcrit = _sqrt_relaxation(
                lower, upper)
            expected_center = slope * center + constant
            expected_retained = [
                slope * Interval(_exact_float(
                    _flat_spatial(source, 1 + generator, coordinate)))
                for generator in range(generators)]
        stored_center = Interval(_exact_float(
            _flat_spatial(output, 0, coordinate)))
        required = _abs_upper(expected_center - stored_center)
        for generator, expected in enumerate(expected_retained):
            stored = Interval(_exact_float(
                _flat_spatial(output, 1 + generator, coordinate)))
            required = _up(lambda required=required, expected=expected,
                           stored=stored:
                           required + _abs_upper(expected - stored))
        required = _up(lambda required=required, expected_fresh=expected_fresh:
                       required + _abs_upper(expected_fresh))
        stored_radius = _mp(0)
        if different:
            fresh_row = 1 + generators + owner[coordinate]
            stored_radius = abs(_exact_float(
                _flat_spatial(output, fresh_row, coordinate)))
        if stored_radius < required:
            deficit = _up(lambda required=required, stored_radius=stored_radius:
                          required - stored_radius)
            maximum_deficit = max(maximum_deficit, deficit)
            if not _allow_shadow:
                raise AssertionError(
                    "native affine sqrt fresh radius does not enclose "
                    "exact-real transition")
            if shadow_radii[coordinate] < deficit:
                raise AssertionError(
                    "native affine sqrt numerical representation does not "
                    "enclose exact-real transition")

    for offset, coordinate in enumerate(active):
        row = 1 + generators + offset
        for other in range(spatial):
            if other != coordinate and _flat_spatial(output, row, other) != 0.0:
                raise AssertionError("native affine sqrt fresh layout mismatch")
    return {
        "accepted": True, "family": "sqrt_affine",
        "precision_bits": PRECISION, "coordinates_checked": spatial,
        "fresh_count": fresh_count,
        "strict_no_empirical_epsilon": True,
        "maximum_deficit": str(maximum_deficit),
        "representation": ("native_plus_coordinate_numerical_sidecar"
                           if _allow_shadow else "native_only"),
    }


def check_native_affine_sqrt_shadow_witness(
        witness: dict[str, Any], root: str | Path):
    """Check native sqrt topology plus a separate coordinate FP envelope."""
    return check_native_affine_sqrt_witness(
        witness, root, _allow_shadow=True)
