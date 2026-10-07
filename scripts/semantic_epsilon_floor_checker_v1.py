"""Independent semantic LayerNorm interval-box obligation; stdlib only.

This is a local transition checker, not a complete Transformer checker. The
caller must authenticate the incoming state and frozen gamma/beta identities.
No relaxed affine variance assignments are used as the semantic domain.
"""
from fractions import Fraction
import hashlib
import json
import math
import struct

import separating_variance_checker_v1 as support

SCHEMA = "CORET_SEMANTIC_EPSILON_FLOOR_LAYERNORM_V1"
EPSILON_HEX = "0x1.19799812dea11p-40"
PSD_RULE = "EXACT_CENTERED_SUM_OF_SQUARES_NONNEGATIVE_V1"


def _f(value):
    if type(value) is not float or not math.isfinite(value):
        raise RuntimeError("epsilon-floor nonfinite/non-binary64 operand")
    return Fraction.from_float(value)


def tensor_hash(block):
    shape = block["shape"]
    if (not isinstance(shape, list) or not shape or
            any(type(n) is not int or n < 0 for n in shape) or
            block.get("dtype") != "float64" or block.get("byteorder") != "little" or
            not isinstance(block["data"], bytes) or len(block["data"]) != 8*math.prod(shape)):
        raise RuntimeError("epsilon-floor tensor encoding/topology differs")
    header = json.dumps({"dtype": "float64", "shape": shape},
                        sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(header + block["data"]).hexdigest()


def values(block):
    tensor_hash(block)
    result = [x[0] for x in struct.iter_unpack("<d", block["data"])]
    if not all(math.isfinite(x) for x in result):
        raise RuntimeError("epsilon-floor tensor contains nonfinite values")
    return result


def authenticate_source(source, expected_identity):
    w, lo, hi = (source[k] for k in ("weights", "low", "high"))
    if len(w["shape"]) != 3:
        raise RuntimeError("epsilon-floor source is not a hidden affine state")
    rows, tokens, d = w["shape"]
    n = rows-1
    proof = source["proof"]
    if (d < 2 or tokens < 1 or n < 0 or lo["shape"] != [n] or hi["shape"] != [n]
            or proof["num_tokens"] != tokens or len(set(proof["ids"])) != n
            or any(len(proof[k]) != n for k in ("ids", "masks", "reasons"))
            or any(not isinstance(x, str) or not x for x in proof["ids"])
            or any(type(m) is not int or not 0 <= m < (1 << tokens) for m in proof["masks"])
            or any(not isinstance(x, str) or not x for x in proof["reasons"])):
        raise RuntimeError("epsilon-floor source generator/provenance topology differs")
    identity = {"weights_sha256": tensor_hash(w), "range_low_sha256": tensor_hash(lo),
                "range_high_sha256": tensor_hash(hi),
                "ordered_generator_ids_sha256": support.digest(proof["ids"]),
                "support_masks_sha256": support.digest(proof["masks"]),
                "provenance_reasons_sha256": support.digest(proof["reasons"]),
                "generator_count": n, "token_count": tokens, "hidden_dimension": d}
    identity["canonical_state_identity_sha256"] = support.digest(identity)
    if identity != expected_identity:
        raise RuntimeError("epsilon-floor authenticated predecessor identity differs")
    weights, low, high = values(w), values(lo), values(hi)
    if any(l > u for l, u in zip(low, high)):
        raise RuntimeError("epsilon-floor source box reversed")
    for r, mask in enumerate(proof["masks"], 1):
        for t in range(tokens):
            if not mask & (1 << t) and any(weights[(r*tokens+t)*d:(r*tokens+t+1)*d]):
                raise RuntimeError("epsilon-floor source support omits coefficients")
    return weights, low, high, n, tokens, d


def outward(value, upward):
    """Explicit one-nextafter reserve after correctly directed conversion."""
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError("epsilon-floor bound overflow")
    if (Fraction.from_float(result) < value if upward else Fraction.from_float(result) > value):
        result = math.nextafter(result, math.inf if upward else -math.inf)
    result = math.nextafter(result, math.inf if upward else -math.inf)
    if not math.isfinite(result):
        raise RuntimeError("epsilon-floor outward reserve overflow")
    return result


def directed_sqrt(value, upward):
    # Independent exact-square replay, not the producer's libm error claim.
    target = _f(value)
    if target <= 0:
        raise RuntimeError("epsilon-floor sqrt domain is not positive")
    result = math.sqrt(value)
    while (_f(result)**2 < target if upward else _f(result)**2 > target):
        result = math.nextafter(result, math.inf if upward else -math.inf)
    return result


def token_claim(weights, low, high, n, tokens, d, token, gamma, beta):
    """Exact affine centering/support of ALL generators, then semantic images."""
    centered_low = [Fraction(0)]*d
    centered_high = [Fraction(0)]*d
    for row in range(n+1):
        vector = [_f(x) for x in weights[(row*tokens+token)*d:(row*tokens+token+1)*d]]
        mean = sum(vector, Fraction(0))/d
        centered = [x-mean for x in vector]
        if row == 0:
            centered_low = list(centered)
            centered_high = list(centered)
        else:
            l, u = _f(low[row-1]), _f(high[row-1])
            for j, a in enumerate(centered):
                left, right = a*l, a*u
                centered_low[j] += min(left, right)
                centered_high[j] += max(left, right)
    variance_upper_exact = sum((max(abs(l), abs(u))**2 for l, u in
                                zip(centered_low, centered_high)), Fraction(0))/d
    vu = outward(variance_upper_exact, True)
    # Nonnegativity refers to the exact centered state, NOT a variance zonotope.
    epsilon = _f(float.fromhex(EPSILON_HEX))
    hl = outward(epsilon, False)
    hu = outward(_f(vu)+epsilon, True)
    sl, su = directed_sqrt(hl, False), directed_sqrt(hu, True)
    rl, ru = outward(1/_f(su), False), outward(1/_f(sl), True)
    if not 0 < hl <= hu or not 0 < sl <= su or not 0 < rl <= ru:
        raise RuntimeError("epsilon-floor semantic range malformed")
    boxes = []
    for l, u, g, b in zip(centered_low, centered_high, gamma, beta):
        products = [a*r for a in (l, u) for r in (_f(rl), _f(ru))]
        endpoints = [a*_f(g)+_f(b) for a in (min(products), max(products))]
        ol, ou = min(endpoints), max(endpoints)
        midpoint = (ol+ou)/2
        center = float(midpoint)
        radius = outward(max(_f(center)-ol, ou-_f(center)), True)
        reserve = outward(abs(_f(center)-midpoint), True)
        if _f(center)-_f(radius) > ol or _f(center)+_f(radius) < ou:
            raise RuntimeError("epsilon-floor machine box fails exact containment")
        boxes.append({"center": center.hex(), "radius": radius.hex(),
                      "reserve": reserve.hex(), "exact_lower": support.rational(ol),
                      "exact_upper": support.rational(ou),
                      "midpoint_roundoff": support.rational(abs(_f(center)-midpoint))})
    return {"token_index": token, "included_generator_count": n, "dimension": d,
            "structural_rule": PSD_RULE, "variance_lower_semantic": 0,
            "variance_upper_provenance": "EXACT_INCOMING_CENTERED_AFFINE_BOX_SUPPORT_ALL_GENERATORS",
            "variance_upper_exact": support.rational(variance_upper_exact),
            "variance_upper": vu.hex(), "epsilon": EPSILON_HEX,
            "epsilon_bits": struct.pack(">d", float.fromhex(EPSILON_HEX)).hex(),
            "regularized_range": [hl.hex(), hu.hex()], "sqrt_range": [sl.hex(), su.hex()],
            "reciprocal_range": [rl.hex(), ru.hex()],
            "reciprocal_domain": "AUTHENTICATED_DIRECTED_SQRT_RANGE",
            "numerical_containment": "EXACT_INTERVAL_TO_MACHINE_BOX_WITH_OUTWARD_RADIUS_AND_RESERVE",
            "output_boxes": boxes}


def construct(source, identity, gamma, beta, parameter_hashes, failed_tokens, label):
    weights, low, high, n, tokens, d = authenticate_source(source, identity)
    if (gamma["shape"] != [d] or beta["shape"] != [d] or
            parameter_hashes != {"weight": tensor_hash(gamma), "bias": tensor_hash(beta)} or
            not failed_tokens or failed_tokens != sorted(set(failed_tokens)) or
            any(type(t) is not int or not 0 <= t < tokens for t in failed_tokens)):
        raise RuntimeError("epsilon-floor frozen parameter/token identity differs")
    gv, bv = values(gamma), values(beta)
    return {"schema": SCHEMA, "path": "semantic_epsilon_floor", "label": label,
            "representation_policy": "FAILED_TOKEN_SEMANTIC_IMAGE_INTERVAL_BOX_REPLACES_AFFINE_CORRELATIONS",
            "source_state_identity": identity, "source": source,
            "parameter_hashes": parameter_hashes, "gamma": gamma, "beta": beta,
            "failed_tokens": failed_tokens,
            "tokens": [token_claim(weights, low, high, n, tokens, d, t, gv, bv)
                       for t in failed_tokens]}


def verify(payload, expected_identity, expected_parameter_hashes):
    """Authorization uses authenticated external predecessor/parameter hashes."""
    expected = construct(payload["source"], expected_identity, payload["gamma"], payload["beta"],
                         expected_parameter_hashes, payload["failed_tokens"], payload["label"])
    if any(payload.get(k) != v for k, v in expected.items()):
        raise RuntimeError("epsilon-floor independently replayed semantic witness differs")
    generic = payload.get("generic_lower_by_token")
    if generic is not None:
        if (len(generic) != expected_identity["token_count"] or
                not all(type(x) is float and math.isfinite(x) for x in generic) or
                payload["failed_tokens"] != [t for t,x in enumerate(generic) if x <= 0] or
                payload.get("semantic_lower_by_token") !=
                [0. if t in payload["failed_tokens"] else x for t,x in enumerate(generic)]):
            raise RuntimeError("epsilon-floor generic/semantic domain metadata differs")
    if "allocation" in payload:
        allocation = payload["allocation"]
        start = allocation["start_generator"]
        d = expected_identity["hidden_dimension"]
        fresh = [(t, j) for t in payload["failed_tokens"] for j in range(d)]
        ids = [f"semantic_epsilon_floor::{payload['label']}::{t:06d}::{j:06d}" for t,j in fresh]
        reasons = ["semantic_layernorm_epsilon_floor_coordinate_box"]*len(fresh)
        if (type(start) is not int or start < expected_identity["generator_count"] or
                allocation != {"start_generator": start, "ids": ids,
                               "masks": [1 << t for t,j in fresh], "reasons": reasons,
                               "low": [-1.]*len(fresh), "high": [1.]*len(fresh)}):
            raise RuntimeError("epsilon-floor fresh allocation/order/provenance differs")
    return {"verified": True, "path": "semantic_epsilon_floor",
            "all_generators_replayed": expected_identity["generator_count"]}


def verify_output(payload, output, reserve, expected_identity, expected_parameter_hashes):
    verify(payload, expected_identity, expected_parameter_hashes)
    rows, tokens, d = output["weights"]["shape"]
    allocation = payload["allocation"]
    start, count = allocation["start_generator"], len(allocation["ids"])
    if rows != start+count+1 or (tokens, d) != (expected_identity["token_count"], expected_identity["hidden_dimension"]):
        raise RuntimeError("epsilon-floor output topology differs")
    ow, ol, oh, rv = values(output["weights"]), values(output["low"]), values(output["high"]), values(reserve)
    proof = output["proof"]
    keep_mask = sum(1 << t for t in range(tokens) if t not in payload["failed_tokens"])
    incoming = payload["source"]["proof"]
    if (any(len(proof[k]) != rows-1 for k in ("ids", "masks", "reasons")) or
            proof["num_tokens"] != tokens or reserve["shape"] != [tokens, d] or
            output["low"]["shape"] != [rows-1] or output["high"]["shape"] != [rows-1] or
            any(proof[k][start:] != allocation[k] for k in ("ids", "masks", "reasons")) or
            ol[start:] != allocation["low"] or oh[start:] != allocation["high"] or
            ol[:expected_identity["generator_count"]] != values(payload["source"]["low"]) or
            oh[:expected_identity["generator_count"]] != values(payload["source"]["high"]) or
            proof["ids"][:expected_identity["generator_count"]] != incoming["ids"] or
            proof["reasons"][:expected_identity["generator_count"]] != incoming["reasons"] or
            proof["masks"][:expected_identity["generator_count"]] != [m & keep_mask for m in incoming["masks"]]):
        raise RuntimeError("epsilon-floor output IDs/ranges/provenance differ")
    offset = 0
    for claim in payload["tokens"]:
        t = claim["token_index"]
        for j, box in enumerate(claim["output_boxes"]):
            if ow[t*d+j] != float.fromhex(box["center"]) or rv[t*d+j] != float.fromhex(box["reserve"]):
                raise RuntimeError("epsilon-floor output center/reserve differs")
            for r in range(rows-1):
                expected = float.fromhex(box["radius"]) if r == start+offset else 0.
                if ow[((r+1)*tokens+t)*d+j] != expected:
                    raise RuntimeError("epsilon-floor output fresh membership/coefficient differs")
            offset += 1
    return {"verified": True, "full_floor_output_containment": True}


def replay_persisted(payload, expected_identity, expected_parameter_hashes,
                     expected_domain, expected_output_identity):
    """Bind local proof to externally authenticated graph input/output/domain.

    The native valid-token transitions retain their pre-existing checking
    obligations; this checker discharges only the replaced floor-token image.
    """
    if (payload["source_domain"] != expected_domain or
            payload["output_state_identity"] != expected_output_identity):
        raise RuntimeError("epsilon-floor external domain/output binding differs")
    authenticate_source(payload["output"], expected_output_identity)
    return verify_output(payload, payload["output"], payload["reserve"],
                         expected_identity, expected_parameter_hashes)
