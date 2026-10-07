"""Independent, bit-exact final-token projection obligation (stdlib only).

The region specification is the frozen 3L head: the final encoder output is
indexed at token 0, followed only by feature-affine/tanh/feature-affine maps.
The caller must authenticate the predecessor, execution identity and region in
the graph. This local checker does not certify the preceding Transformer.
"""
import hashlib
import json
import math
import struct

SCHEMA = "CORET_SOUND_FP64_FINAL_TOKEN_PROJECTION_V1"
REGION = {
    "rule": "FINAL_3L_ENCODER_FIRST_TOKEN_POOLER_V1",
    "site": "AFTER_BLOCK2_OUTPUT_LAYERNORM_RESERVE_BEFORE_FINAL_REDUCTION",
    "completed_transformer_blocks": [0, 1, 2],
    "final_layernorm": "block2_output",
    "retained_token_indices": [0],
    "remaining_semantic_operators": [
        "token0_selection", "pooler_feature_affine", "elementwise_tanh",
        "binary_classifier_feature_affine", "margin_concretization"],
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def tensor_hash(block):
    shape = block.get("shape")
    if (not isinstance(shape, list) or not shape
            or any(type(n) is not int or n < 0 for n in shape)
            or block.get("dtype") != "float64"
            or block.get("byteorder") != "little"
            or not isinstance(block.get("data"), bytes)
            or len(block["data"]) != 8 * math.prod(shape)):
        raise RuntimeError("final projection tensor encoding differs")
    h = hashlib.sha256(json.dumps({"dtype": "float64", "shape": shape},
                                 sort_keys=True, separators=(",", ":")).encode())
    h.update(block["data"])
    return h.hexdigest()


def state_identity(source):
    w, lo, hi, p = (source[k] for k in ("weights", "low", "high", "proof"))
    if len(w["shape"]) != 3:
        raise RuntimeError("final projection requires 3D hidden state")
    rows, tokens, width = w["shape"]
    n = rows - 1
    if (n < 0 or tokens < 1 or width < 1 or lo["shape"] != [n]
            or hi["shape"] != [n] or p["num_tokens"] != tokens
            or any(len(p[k]) != n for k in ("ids", "masks", "reasons"))
            or len(set(p["ids"])) != n
            or any(not isinstance(x, str) or not x for x in p["ids"] + p["reasons"])
            or any(type(m) is not int or not 0 <= m < (1 << tokens) for m in p["masks"])):
        raise RuntimeError("final projection generator/provenance topology differs")
    result = {
        "weights_sha256": tensor_hash(w), "range_low_sha256": tensor_hash(lo),
        "range_high_sha256": tensor_hash(hi),
        "ordered_generator_ids_sha256": digest(p["ids"]),
        "support_masks_sha256": digest(p["masks"]),
        "provenance_reasons_sha256": digest(p["reasons"]),
        "generator_count": n, "token_count": tokens, "hidden_dimension": width,
    }
    result["canonical_state_identity_sha256"] = digest(result)
    return result


def _validate_operands(source, expected_identity):
    if state_identity(source) != expected_identity:
        raise RuntimeError("final projection authenticated state identity differs")
    w, lo, hi, proof = (source[k] for k in ("weights", "low", "high", "proof"))
    rows, tokens, width = w["shape"]
    # Stream through bytes rather than allocating a second full float matrix.
    for offset, (value,) in enumerate(struct.iter_unpack("<d", w["data"])):
        if not math.isfinite(value):
            raise RuntimeError("final projection nonfinite coefficient")
        row, coordinate = divmod(offset, tokens * width)
        token = coordinate // width
        if row and value != 0 and not proof["masks"][row-1] & (1 << token):
            raise RuntimeError("final projection support omits nonzero coefficient")
    for (l,), (u,) in zip(struct.iter_unpack("<d", lo["data"]),
                         struct.iter_unpack("<d", hi["data"])):
        if not math.isfinite(l) or not math.isfinite(u) or l > u:
            raise RuntimeError("final projection invalid range")


def verify(payload, expected_input_identity, expected_output_identity,
           expected_execution_identity, expected_region):
    if (expected_region != REGION or payload.get("region_proof") != REGION
            or payload.get("execution_identity") != expected_execution_identity
            or payload.get("schema") != SCHEMA):
        raise RuntimeError("final projection outside authenticated final no-cross-token region")
    source, output = payload["input"], payload["output"]
    _validate_operands(source, expected_input_identity)
    _validate_operands(output, expected_output_identity)
    rows, tokens, width = source["weights"]["shape"]
    retained, discarded, pieces = [], [], []
    data = source["weights"]["data"]
    span = 8 * width
    pieces.append(data[:span])  # Exact center at token 0.
    for i in range(rows-1):
        start = (i+1) * tokens * span
        piece = data[start:start+span]
        # Both signed zeros represent an exactly zero affine contribution.
        if all(v == 0 for (v,) in struct.iter_unpack("<d", piece)):
            discarded.append(i)
        else:
            retained.append(i)
            pieces.append(piece)
    n = len(retained)
    p = source["proof"]
    expected_proof = {
        "ids": [p["ids"][i] for i in retained],
        "masks": [p["masks"][i] & 1 for i in retained],
        "reasons": [p["reasons"][i] for i in retained], "num_tokens": 1,
    }
    def selected_range(key):
        raw = source[key]["data"]
        return b"".join(raw[8*i:8*i+8] for i in retained)
    if (output["weights"]["shape"] != [n+1, 1, width]
            or output["weights"]["data"] != b"".join(pieces)
            or output["low"]["data"] != selected_range("low")
            or output["high"]["data"] != selected_range("high")
            or output["proof"] != expected_proof):
        raise RuntimeError("final projection coefficients/ranges/ordered provenance differ")
    expected_metadata = {
        "original_token_count": tokens, "retained_token_indices": [0],
        "input_state_identity": expected_input_identity,
        "output_state_identity": expected_output_identity,
        "pre_projection_generator_ids": p["ids"],
        "post_projection_generator_ids": expected_proof["ids"],
        "retained_generator_indices": retained,
        "zero_after_projection_discarded_generator_ids": [p["ids"][i] for i in discarded],
        "generator_count_before": rows-1, "generator_count_after": n,
    }
    if any(payload.get(k) != v for k, v in expected_metadata.items()):
        raise RuntimeError("final projection exact transition map differs")
    return {"verified": True, "coefficient_bits_unchanged": True,
            "generator_count_before": rows-1, "generator_count_after": n,
            "discarded_zero_count": len(discarded), "retained_token_indices": [0]}
