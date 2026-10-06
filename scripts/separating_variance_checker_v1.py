"""Standalone independent separator arithmetic. Standard library only; no producer.

D has rows e_j-e_(d-1).  For fixed rational y, v=D^T y has sum(v)=0.
The exact box support of v.x supplies delta; Cauchy--Schwarz then gives
Var(x) >= delta**2/(d*||v||**2).  No numerical solver conclusion is used.
"""
from fractions import Fraction
import hashlib
import json
import math
import struct

SCHEMA = "CORET_EXACT_SEPARATING_VARIANCE_CERTIFICATE_V1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def rational(value):
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def read_rational(value):
    if (not isinstance(value, dict) or set(value) != {"numerator", "denominator"}
            or not isinstance(value["numerator"], str)
            or not isinstance(value["denominator"], str)):
        raise RuntimeError("malformed certificate rational")
    try:
        result = Fraction(int(value["numerator"]), int(value["denominator"]))
    except (ValueError, ZeroDivisionError) as error:
        raise RuntimeError("malformed certificate rational") from error
    if rational(result) != value:
        raise RuntimeError("noncanonical certificate rational")
    return result


def state_digest(center, generators, low, high, ids):
    d, n = len(center), len(generators)
    if (d < 2 or len(low) != n or len(high) != n or len(ids) != n
            or len(set(ids)) != n or any(not isinstance(x, str) or not x for x in ids)
            or any(len(row) != d for row in generators)):
        raise RuntimeError("separator state topology/ordered IDs differ")
    sha = hashlib.sha256(json.dumps({"dimension": d, "ids": ids},
                                   sort_keys=True, separators=(",", ":")).encode())
    for row in [center, *generators, low, high]:
        for value in row:
            if type(value) is not float or not math.isfinite(value):
                raise RuntimeError("separator state must contain finite binary64 values")
        sha.update(struct.pack("<" + "d" * len(row), *row))
    if any(l > u for l, u in zip(low, high)):
        raise RuntimeError("separator source box is malformed")
    return sha.hexdigest()


def _binary64_dot(integer_v, denominator, values):
    """Exact dyadic support arithmetic, without a dense rational matrix.

    v has a single common denominator. Every input value is binary64. Sum
    integer products with power-of-two alignment, then reduce ONE Fraction.
    No rounded coordinate subtraction or mean is ever formed.
    """
    total, exponent = 0, 0
    for coefficient, value in zip(integer_v, values):
        if not coefficient or value == 0.0:
            continue
        numerator, power_of_two = value.as_integer_ratio()
        shift = power_of_two.bit_length() - 1
        if shift > exponent:
            total <<= shift - exponent
            exponent = shift
        total += (coefficient * numerator) << (exponent - shift)
    return Fraction(total, denominator << exponent)


def exact_claims(center, generators, low, high, ids, y):
    state_sha = state_digest(center, generators, low, high, ids)
    d, n = len(center), len(generators)
    if len(y) != d - 1 or any(type(value) is not Fraction for value in y):
        raise RuntimeError("separator y dimension/type differs")
    v = [*y, -sum(y, Fraction(0))]
    norm_squared = sum((value * value for value in v), Fraction(0))
    if sum(v, Fraction(0)) != 0 or norm_squared <= 0:
        raise RuntimeError("separator direction is zero/invalid")
    denominator = math.lcm(*(value.denominator for value in v))
    integer_v = [value.numerator * (denominator // value.denominator) for value in v]
    center_functional = _binary64_dot(integer_v, denominator, center)
    lower = upper = center_functional
    # Visit ALL n rows, including explicit numerical generators and zero rows.
    for row, lo, hi in zip(generators, low, high):
        a = _binary64_dot(integer_v, denominator, row)
        left, right = a * Fraction.from_float(lo), a * Fraction.from_float(hi)
        lower += min(left, right)
        upper += max(left, right)
    if lower <= 0 <= upper:
        return None
    delta = lower if lower > 0 else -upper
    bound = delta * delta / (d * norm_squared)
    return {"state_sha256": state_sha, "dimension": d, "generator_count": n,
            "included_generator_count": n, "ordered_generator_ids_sha256": digest(ids),
            "difference_reference_coordinate": d - 1,
            "v_rationals": [rational(value) for value in v],
            "center_functional": rational(center_functional),
            "interval_lower": rational(lower), "interval_upper": rational(upper),
            "delta": rational(delta), "norm_v_squared": rational(norm_squared),
            "variance_lower": rational(bound)}


def construct_certificate(center, generators, low, high, ids, y, authenticated_capture):
    claims = exact_claims(center, generators, low, high, ids, y)
    if claims is None:
        return None
    candidate = [rational(value) for value in y]
    result = {"schema": SCHEMA, "authenticated_capture": authenticated_capture,
              "y_rationals": candidate, "candidate_sha256": digest(candidate), **claims}
    result["certificate_sha256"] = digest(result)
    return result


def verify_certificate(center, generators, low, high, ids, authenticated_capture, certificate):
    """Recompute every claim from immutable operands, not stored solver bounds."""
    unsigned = {k: v for k, v in certificate.items() if k != "certificate_sha256"}
    if (certificate.get("schema") != SCHEMA
            or certificate.get("certificate_sha256") != digest(unsigned)
            or certificate.get("authenticated_capture") != authenticated_capture):
        raise RuntimeError("separator certificate hash/capture identity differs")
    candidate = certificate.get("y_rationals", [])
    if certificate.get("candidate_sha256") != digest(candidate):
        raise RuntimeError("separator candidate identity differs")
    y = [read_rational(value) for value in candidate]
    claims = exact_claims(center, generators, low, high, ids, y)
    expected_keys = {"schema", "authenticated_capture", "y_rationals", "candidate_sha256",
                     "certificate_sha256", *(claims or {}).keys()}
    if (claims is None or set(certificate) != expected_keys
            or any(certificate.get(key) != value for key, value in claims.items())):
        raise RuntimeError("separator exact support/norm/variance replay differs")
    return {"verified": True, "all_generators_replayed": len(generators),
            "variance_lower": claims["variance_lower"], "delta": claims["delta"]}


def downward_binary64(value):
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError("separator lower bound is not representable as finite binary64")
    if Fraction.from_float(result) > value:
        result = math.nextafter(result, -math.inf)
    return result
