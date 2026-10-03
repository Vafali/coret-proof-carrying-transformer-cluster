#!/usr/bin/env python3
"""Read-only semantic comparison for Block-2 LayerNorm linkage states.

The frontier artifact SHA is intentionally *not* the reproduction identity:
``torch.save`` container metadata and paths are not abstract semantics.  The
actual state tensors, ordered source metadata, exact ranges, topology, and
frozen scientific identity are authenticated and compared instead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


FRONTIER_MANIFEST_SCHEMA = "CORET_BLOCK2_FFN_FRONTIER_MANIFEST_V1"
FRONTIER_CAPTURE_SCHEMA = "CORET_BLOCK2_FFN_FRONTIER_CAPTURE_V1"
STATE_KEY = "post_attention_ln_pre_reduction"
TARGET_STAGE = "block2_post_attention"
TARGET_LAYERNORM_INDEX = 5
TARGET_OPERATOR = "bert.encoder.layer.2.attention.output.LayerNorm"
NUMERICAL_REASONS = {
    "fp64_roundoff_coordinate_box",
    "sound_fp64_coordinate_box_replacement_with_numerical",
}
SCHEMA = "CORET_BLOCK2_LAYERNORM_LINKAGE_DIAGNOSTIC_V1"
PROPERTY_ID = "deept_table7_stdln3_s001_line1794_tok11"
MULTIPLIER = "0.75"
TESTED_RADIUS = 0.00060791015625
TESTED_RADIUS_HEX = "0x1.3eb851eb851ecp-11"
FRONTIER_STAGE = "block2_ffn_joint_cancellation_frontier"
PINNED_REVISION = "16ffe4075f1f8a7c87fa2a187d8c46cfd51e07bf"
SCIENTIFIC_MANIFEST_SHA256 = (
    "7e4b2fea94424e554f07272aa8a246da7cda1212be83af57bbcdae7c870ed9dd")
PRODUCTION_MANIFEST_SHA256 = (
    "cd1375408818c8fb93a2f227d31229990d62034dc4997467fbc013cc1eb94ab2")
SOURCE_SET_MODEL = "p100_linf_shared_generator_ids_cartesian_ranges"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _verified_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(value)
    claimed = payload.pop("record_sha256", None)
    if claimed is None or _canonical(payload) != claimed:
        raise RuntimeError(f"canonical hash mismatch: {path}")
    return value


def _tensor_sha(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)},
        sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def _legacy_state_record(state: dict) -> dict:
    weights = state["weights"]
    low, high = state["range_low"], state["range_high"]
    proof = state["proof"]
    range_sha = hashlib.sha256(
        low.contiguous().numpy().tobytes()
        + high.contiguous().numpy().tobytes()).hexdigest()
    ids_sha = _canonical(proof["ids"])
    provenance_sha = _canonical({
        "masks": proof["masks"], "reasons": proof["reasons"],
        "num_tokens": proof["num_tokens"],
    })
    reasons = proof["reasons"]
    return {
        "state_key": STATE_KEY,
        "center_sha256": _tensor_sha(weights[0]),
        "generator_sha256": _tensor_sha(weights[1:]),
        "generator_ids_sha256": ids_sha,
        "ranges_sha256": range_sha,
        "provenance_sha256": provenance_sha,
        "generator_id_range_provenance_sha256": _canonical({
            "generator_ids_sha256": ids_sha,
            "ranges_sha256": range_sha,
            "provenance_sha256": provenance_sha,
        }),
        "generator_count": len(reasons),
        "native_generator_count": sum(
            reason not in NUMERICAL_REASONS for reason in reasons),
        "numerical_generator_count": sum(
            reason in NUMERICAL_REASONS for reason in reasons),
        "token_count": int(weights.shape[1]),
        "feature_dimension": int(weights.shape[2]),
        "dtype": str(weights.dtype).replace("torch.", ""),
    }


def _validate_state(state: dict, label: str) -> None:
    required = {"weights", "range_low", "range_high", "proof"}
    if set(state) != required:
        raise RuntimeError(f"{label}: state inventory differs")
    weights = state["weights"]
    low, high = state["range_low"], state["range_high"]
    proof = state["proof"]
    generators = int(weights.shape[0] - 1)
    if (not isinstance(weights, torch.Tensor) or weights.ndim != 3
            or weights.dtype != torch.float64 or weights.device.type != "cpu"
            or not isinstance(low, torch.Tensor)
            or not isinstance(high, torch.Tensor)
            or low.dtype != torch.float64 or high.dtype != torch.float64
            or tuple(low.shape) != (generators,)
            or tuple(high.shape) != (generators,)
            or not isinstance(proof, dict)
            or set(proof) != {"masks", "ids", "reasons", "num_tokens"}
            or len(proof["ids"]) != generators
            or len(proof["masks"]) != generators
            or len(proof["reasons"]) != generators
            or len(set(proof["ids"])) != generators
            or int(proof["num_tokens"]) != int(weights.shape[1])
            or not bool(torch.isfinite(weights).all())
            or not bool(torch.isfinite(low).all())
            or not bool(torch.isfinite(high).all())
            or bool((low > high).any())):
        raise RuntimeError(f"{label}: state topology/ranges differ")


def _load_frontier(manifest_path: Path, label: str) -> dict:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = _verified_json(manifest_path)
    expected_identity = {
        "schema": FRONTIER_MANIFEST_SCHEMA,
        "property_id": PROPERTY_ID,
        "multiplier": MULTIPLIER,
        "tested_radius": TESTED_RADIUS,
        "tested_radius_hex": TESTED_RADIUS_HEX,
        "stage_label": FRONTIER_STAGE,
        "pinned_deept_revision": PINNED_REVISION,
        "scientific_manifest_sha256": SCIENTIFIC_MANIFEST_SHA256,
        "production_manifest_sha256": PRODUCTION_MANIFEST_SHA256,
        "source_set_model": SOURCE_SET_MODEL,
    }
    if any(manifest.get(key) != value
           for key, value in expected_identity.items()):
        raise RuntimeError(f"{label}: frontier manifest schema differs")
    artifact_identity = {
        key: value for key, value in expected_identity.items()
        if key != "schema"
    }
    if manifest.get("artifact_identity") != artifact_identity:
        raise RuntimeError(f"{label}: frontier artifact identity differs")
    artifact_path = (
        manifest_path.parent / manifest.get("tensor_artifact_path", ""))
    artifact_path = artifact_path.resolve()
    if (not artifact_path.is_file()
            or _sha256(artifact_path) != manifest.get(
                "tensor_artifact_sha256")):
        raise RuntimeError(f"{label}: frontier tensor artifact differs")
    try:
        payload = torch.load(
            artifact_path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:  # Compatibility with torch releases predating mmap.
        payload = torch.load(
            artifact_path, map_location="cpu", weights_only=False)
    if (not isinstance(payload, dict)
            or payload.get("schema") != FRONTIER_CAPTURE_SCHEMA
            or payload.get("pinned_revision") != PINNED_REVISION
            or payload.get("identity") != artifact_identity):
        raise RuntimeError(f"{label}: frontier artifact identity differs")
    states = payload.get("states")
    if not isinstance(states, dict) or STATE_KEY not in states:
        raise RuntimeError(f"{label}: target frontier state is absent")
    variant_keys = [row.get("state_key")
                    for row in manifest.get("variants", [])]
    if (variant_keys != list(states) or len(set(variant_keys)) != len(states)):
        raise RuntimeError(f"{label}: frontier state inventory differs")
    reductions = payload.get("reductions")
    if (not isinstance(reductions, dict)
            or _canonical(reductions)
            != manifest.get("reduction_records_sha256")):
        raise RuntimeError(f"{label}: frontier reduction records differ")
    if payload.get("aliases") != manifest.get("aliases"):
        raise RuntimeError(f"{label}: frontier aliases differ")
    result_path = (manifest_path.parent
                   / manifest.get("result_path", "")).resolve()
    if (not result_path.is_file()
            or _sha256(result_path) != manifest.get("result_sha256")):
        raise RuntimeError(f"{label}: frontier result artifact differs")
    state = states[STATE_KEY]
    _validate_state(state, label)
    rows = [row for row in manifest.get("variants", [])
            if row.get("state_key") == STATE_KEY]
    if len(rows) != 1:
        raise RuntimeError(f"{label}: target manifest row is absent")
    actual = _legacy_state_record(state)
    if rows[0] != actual:
        differing = sorted(set(rows[0]) | set(actual))
        differing = [key for key in differing
                     if rows[0].get(key) != actual.get(key)]
        raise RuntimeError(
            f"{label}: target manifest row differs at {differing[0]}")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "artifact_path": str(artifact_path),
        "artifact_sha256": _sha256(artifact_path),
        "manifest": manifest,
        "manifest_row": rows[0],
        "state": state,
    }


def _first_difference(left: list, right: list):
    stop = min(len(left), len(right))
    for index in range(stop):
        if left[index] != right[index]:
            return {"index": index, "expected": left[index],
                    "reproduced": right[index]}
    if len(left) != len(right):
        return {"index": stop, "expected_length": len(left),
                "reproduced_length": len(right)}
    return None


def _tensor_comparison(left: torch.Tensor, right: torch.Tensor) -> dict:
    same_shape = tuple(left.shape) == tuple(right.shape)
    same_dtype = left.dtype == right.dtype
    exact = bool(same_shape and same_dtype and torch.equal(left, right))
    maximum = None
    if same_shape and left.numel():
        # Chunking avoids materializing another full generator tensor.
        a, b = left.reshape(-1), right.reshape(-1)
        maximum_value = 0.0
        for start in range(0, a.numel(), 1 << 20):
            stop = min(start + (1 << 20), a.numel())
            maximum_value = max(
                maximum_value,
                float((a[start:stop] - b[start:stop]).abs().max()))
        maximum = maximum_value
    elif same_shape:
        maximum = 0.0
    return {
        "expected_shape": list(left.shape),
        "reproduced_shape": list(right.shape),
        "expected_dtype": str(left.dtype).replace("torch.", ""),
        "reproduced_dtype": str(right.dtype).replace("torch.", ""),
        "exact_equal": exact,
        "maximum_absolute_difference": maximum,
    }


def _target_invocation(manifest_path: Path) -> dict:
    trace_path = manifest_path.parent / "psd_layernorm_invocation_trace.json"
    if not trace_path.is_file():
        return {"available": False, "path": str(trace_path)}
    trace = _verified_json(trace_path)
    rows = [row for row in trace.get("invocations", [])
            if row.get("stage") == TARGET_STAGE]
    if len(rows) != 1:
        return {"available": True, "path": str(trace_path),
                "target_count": len(rows), "valid": False}
    row = rows[0]
    return {
        "available": True,
        "path": str(trace_path),
        "sha256": _sha256(trace_path),
        "valid": True,
        "ordinal": row.get("ordinal"),
        "layernorm_index": row.get("layernorm_index"),
        "stage": row.get("stage"),
        "psd_applied": row.get("psd_applied"),
    }


def compare_frontier_manifests(expected_path: Path,
                               reproduced_path: Path) -> dict:
    expected = _load_frontier(expected_path, "expected")
    reproduced = _load_frontier(reproduced_path, "reproduced")
    left, right = expected["state"], reproduced["state"]
    lp, rp = left["proof"], right["proof"]

    classification_left = [
        "numerical" if reason in NUMERICAL_REASONS else "native"
        for reason in lp["reasons"]]
    classification_right = [
        "numerical" if reason in NUMERICAL_REASONS else "native"
        for reason in rp["reasons"]]
    identity_fields = (
        "property_id", "multiplier", "tested_radius", "tested_radius_hex",
        "stage_label", "pinned_deept_revision",
        "scientific_manifest_sha256", "production_manifest_sha256",
        "source_set_model",
    )
    identity_comparison = {
        field: {
            "expected": expected["manifest"].get(field),
            "reproduced": reproduced["manifest"].get(field),
            "exact_equal": (expected["manifest"].get(field)
                            == reproduced["manifest"].get(field)),
        } for field in identity_fields
    }
    sequence_values = {
        "source_ids": (lp["ids"], rp["ids"]),
        "masks": (lp["masks"], rp["masks"]),
        "provenance_reasons": (lp["reasons"], rp["reasons"]),
        "native_numerical_classification": (
            classification_left, classification_right),
    }
    sequence_comparison = {}
    for name, (a, b) in sequence_values.items():
        sequence_comparison[name] = {
            "expected_sha256": _canonical(a),
            "reproduced_sha256": _canonical(b),
            "exact_equal": a == b,
            "first_difference": _first_difference(a, b),
        }
    topology_fields = {
        "weights_shape": (list(left["weights"].shape),
                          list(right["weights"].shape)),
        "range_shape": (list(left["range_low"].shape),
                        list(right["range_low"].shape)),
        "num_tokens": (lp["num_tokens"], rp["num_tokens"]),
        "generator_count": (len(lp["ids"]), len(rp["ids"])),
        "native_generator_count": (
            classification_left.count("native"),
            classification_right.count("native")),
        "numerical_generator_count": (
            classification_left.count("numerical"),
            classification_right.count("numerical")),
    }
    topology = {
        key: {"expected": a, "reproduced": b, "exact_equal": a == b}
        for key, (a, b) in topology_fields.items()
    }
    tensors = {
        "center": _tensor_comparison(
            left["weights"][0], right["weights"][0]),
        "generators": _tensor_comparison(
            left["weights"][1:], right["weights"][1:]),
        "range_low": _tensor_comparison(
            left["range_low"], right["range_low"]),
        "range_high": _tensor_comparison(
            left["range_high"], right["range_high"]),
    }
    semantic_hashes = {
        "center_sha256": (_tensor_sha(left["weights"][0]),
                          _tensor_sha(right["weights"][0])),
        "generators_sha256": (_tensor_sha(left["weights"][1:]),
                              _tensor_sha(right["weights"][1:])),
        "source_ids_sha256": (_canonical(lp["ids"]), _canonical(rp["ids"])),
        "ranges_sha256": (
            expected["manifest_row"]["ranges_sha256"],
            reproduced["manifest_row"]["ranges_sha256"]),
        "masks_sha256": (_canonical(lp["masks"]), _canonical(rp["masks"])),
        "provenance_sha256": (_canonical(lp["reasons"]),
                              _canonical(rp["reasons"])),
        "native_numerical_classification_sha256": (
            _canonical(classification_left), _canonical(classification_right)),
    }
    hash_comparison = {
        key: {"expected": a, "reproduced": b, "exact_equal": a == b}
        for key, (a, b) in semantic_hashes.items()
    }
    checks = []
    checks.extend((f"identity.{key}", row["exact_equal"])
                  for key, row in identity_comparison.items())
    checks.extend((f"topology.{key}", row["exact_equal"])
                  for key, row in topology.items())
    checks.extend((f"hashes.{key}", row["exact_equal"])
                  for key, row in hash_comparison.items())
    checks.extend((f"tensors.{key}", row["exact_equal"])
                  for key, row in tensors.items())
    checks.extend((f"ordered_metadata.{key}", row["exact_equal"])
                  for key, row in sequence_comparison.items())
    first = next((name for name, passed in checks if not passed), None)
    semantic_equal = first is None
    payload = {
        "schema": SCHEMA,
        "classification": (
            "SEMANTIC_STATE_MATCH" if semantic_equal
            else "TRUE_SCIENTIFIC_REPRODUCTION_MISMATCH"),
        "semantic_equal": semantic_equal,
        "first_semantic_difference": first,
        "comparison_target": {
            "state_key": STATE_KEY,
            "stage": TARGET_STAGE,
            "layernorm_index": TARGET_LAYERNORM_INDEX,
            "operator": TARGET_OPERATOR,
        },
        "expected_identity": {
            "manifest_path": expected["manifest_path"],
            "manifest_sha256": expected["manifest_sha256"],
            "artifact_path": expected["artifact_path"],
            "artifact_sha256": expected["artifact_sha256"],
            "manifest_row": expected["manifest_row"],
            "target_invocation": _target_invocation(Path(
                expected["manifest_path"])),
        },
        "reproduced_identity": {
            "manifest_path": reproduced["manifest_path"],
            "manifest_sha256": reproduced["manifest_sha256"],
            "artifact_path": reproduced["artifact_path"],
            "artifact_sha256": reproduced["artifact_sha256"],
            "manifest_row": reproduced["manifest_row"],
            "target_invocation": _target_invocation(Path(
                reproduced["manifest_path"])),
        },
        "identity_comparison": identity_comparison,
        "topology_comparison": topology,
        "semantic_hash_comparison": hash_comparison,
        "tensor_comparison": tensors,
        "ordered_metadata_comparison": sequence_comparison,
        "excluded_from_semantic_identity": [
            "output-root path", "artifact path", "artifact container SHA256",
            "capture timestamp (not serialized)",
        ],
    }
    payload["diagnostic_sha256"] = _canonical(payload)
    return payload


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-manifest", required=True, type=Path)
    parser.add_argument("--reproduced-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = compare_frontier_manifests(
        args.expected_manifest, args.reproduced_manifest)
    _write_json_atomic(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
