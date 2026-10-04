#!/usr/bin/env python3
"""Diagnostic-first CPU authentication of job2997's persisted LN input."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "research_hab")]

import cluster_common
import run_sound_fp64_3l_psd_layernorm_experiment_v1 as runner


SCHEMA = "CORET_JOB2997_PERSISTED_IDENTITY_DIAGNOSTIC_V1"
COMPONENTS = (
    "generator_count", "token_count", "hidden_dimension",
    "weights_sha256", "ordered_generator_ids_sha256",
    "range_low_sha256", "range_high_sha256", "support_masks_sha256",
    "provenance_reasons_sha256", "canonical_state_identity_sha256",
)
CLASSIFICATIONS = {
    "weights_sha256": "PERSISTED_WEIGHTS_DIFFER",
    "ordered_generator_ids_sha256": "PERSISTED_SOURCE_IDS_DIFFER",
    "range_low_sha256": "PERSISTED_RANGE_LOW_DIFFER",
    "range_high_sha256": "PERSISTED_RANGE_HIGH_DIFFER",
    "support_masks_sha256": "PERSISTED_MASKS_DIFFER",
    "provenance_reasons_sha256": "PERSISTED_PROVENANCE_DIFFER",
    "generator_count": "PERSISTED_TOPOLOGY_DIFFER",
    "token_count": "PERSISTED_TOPOLOGY_DIFFER",
    "hidden_dimension": "PERSISTED_TOPOLOGY_DIFFER",
    "canonical_state_identity_sha256":
        "PERSISTED_CANONICAL_DIGEST_CONSTRUCTION_BUG",
}


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _target_identity(trace_path: Path) -> dict:
    trace = cluster_common.verified_json(trace_path)
    matches = [row.get("state_identity") for row in trace.get("invocations", [])
               if row.get("stage") == runner.experiment.TARGET_LABEL
               and int(row.get("layernorm_index", -1)) ==
               runner.INPUT_LAYERNORM_INDEX
               and row.get("target_state_match") is True]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise RuntimeError("authenticated target invocation identity differs")
    return matches[0]


def _tensor_record(tensor, persisted_hash: str,
                   expected_hash: str | None) -> dict:
    exact_equal = expected_hash is not None and persisted_hash == expected_hash
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "persisted_hash": persisted_hash,
        "expected_hash": expected_hash,
        "exact_equal": exact_equal,
        "maximum_absolute_difference": 0.0 if exact_equal else None,
        "difference_values_available": False,
    }


def _first_sequence_difference(expected, persisted) -> dict:
    if expected is None:
        return {"index": None, "expected": None, "persisted": None,
                "reason": "manifest stores only the authenticated hash"}
    limit = min(len(expected), len(persisted))
    for index in range(limit):
        if expected[index] != persisted[index]:
            return {"index": index, "expected": expected[index],
                    "persisted": persisted[index]}
    if len(expected) != len(persisted):
        return {"index": limit,
                "expected": expected[limit] if limit < len(expected) else None,
                "persisted": (
                    persisted[limit] if limit < len(persisted) else None)}
    return {"index": None, "expected": None, "persisted": None}


def _raw_artifact_manifest_comparison(snapshot: dict,
                                      state_record: dict) -> dict:
    hashes = runner._frontier_state_hashes(snapshot)
    hashes.update({
        "source_ids_sha256": hashes["generator_ids_sha256"],
        "hidden_dimension": hashes["feature_dimension"],
    })
    checks = {
        field: {"persisted": value, "manifest": state_record.get(field),
                "equal": value == state_record.get(field)}
        for field, value in hashes.items()
    }
    weights = snapshot["weights"]
    proof = snapshot["proof"]
    tensor_checks = {
        "center": _tensor_record(
            weights[0], hashes["center_sha256"],
            state_record.get("center_sha256")),
        "generators": _tensor_record(
            weights[1:], hashes["generator_sha256"],
            state_record.get("generator_sha256")),
        "range_low": _tensor_record(
            snapshot["range_low"],
            runner.experiment._tensor_hash(snapshot["range_low"]), None),
        "range_high": _tensor_record(
            snapshot["range_high"],
            runner.experiment._tensor_hash(snapshot["range_high"]), None),
    }
    metadata = {
        "ordered_source_ids": {
            "persisted_hash": hashes["generator_ids_sha256"],
            "manifest_hash": state_record.get("generator_ids_sha256"),
            "equal": hashes["generator_ids_sha256"] ==
                     state_record.get("generator_ids_sha256"),
            "first_difference": _first_sequence_difference(None, proof["ids"]),
        },
        "support_masks": {
            "persisted_hash": runner.capture._json_sha(proof["masks"]),
            "manifest_combined_provenance_hash":
                state_record.get("provenance_sha256"),
            "first_difference": _first_sequence_difference(
                None, proof["masks"]),
        },
        "provenance_reasons": {
            "persisted_hash": runner.capture._json_sha(proof["reasons"]),
            "manifest_combined_provenance_hash":
                state_record.get("provenance_sha256"),
            "first_difference": _first_sequence_difference(
                None, proof["reasons"]),
        },
    }
    return {
        "hash_field_checks": checks,
        "tensor_checks": tensor_checks,
        "ordered_metadata_checks": metadata,
        "all_manifest_hash_fields_equal": all(
            row["equal"] for row in checks.values()),
    }


def compare_snapshot(snapshot: dict, expected_live: dict,
                     job2997_live: dict, state_record: dict) -> dict:
    recomputed = runner._snapshot_state_identity(snapshot)
    component_comparison = {
        field: {
            "job2995_expected": expected_live.get(field),
            "job2997_live": job2997_live.get(field),
            "persisted_recomputed": recomputed.get(field),
            "job2995_job2997_equal": (
                expected_live.get(field) == job2997_live.get(field)),
            "persisted_equal_to_both": (
                expected_live.get(field) == job2997_live.get(field)
                == recomputed.get(field)),
        }
        for field in COMPONENTS
    }
    raw = _raw_artifact_manifest_comparison(snapshot, state_record)
    first = next((field for field in COMPONENTS
                  if not component_comparison[field][
                      "persisted_equal_to_both"]), None)
    if not raw["all_manifest_hash_fields_equal"]:
        classification = "PERSISTED_ARTIFACT_MANIFEST_MISMATCH"
        if first is None:
            first = next(field for field, row in
                         raw["hash_field_checks"].items()
                         if not row["equal"])
    elif first is None:
        classification = "PERSISTED_STATE_CANONICALLY_IDENTICAL"
    else:
        classification = CLASSIFICATIONS[first]
    return {
        "expected_live_canonical_identity": expected_live,
        "job2997_live_canonical_identity": job2997_live,
        "persisted_recomputed_canonical_identity": recomputed,
        "component_comparison": component_comparison,
        "raw_artifact_manifest_comparison": raw,
        "first_differing_component": first,
        "classification": classification,
    }


def diagnose(capture_root: Path, job2995_trace: Path) -> dict:
    manifest_path = capture_root / "pre_layernorm_input_manifest.json"
    trace_path = capture_root / "psd_layernorm_invocation_trace.json"
    manifest = cluster_common.verified_json(manifest_path)
    artifact_path = (manifest_path.parent
                     / manifest["tensor_artifact_path"]).resolve()
    payload = runner.capture.sound.torch.load(
        artifact_path, map_location="cpu", weights_only=False)
    snapshot = (payload.get("states") or {}).get("pre_layernorm_input")
    if not isinstance(snapshot, dict):
        raise RuntimeError("persisted pre-LayerNorm state is absent")
    comparison = compare_snapshot(
        snapshot, _target_identity(job2995_trace),
        _target_identity(trace_path), manifest.get("state") or {})
    comparison.update({
        "schema": SCHEMA,
        "persisted_artifact_path": str(artifact_path),
        "persisted_artifact_sha256": cluster_common.sha256(artifact_path),
        "persisted_manifest_path": str(manifest_path),
        "persisted_manifest_sha256": cluster_common.sha256(manifest_path),
        "semantic_verifier_accepts": False,
        "semantic_verifier_error": None,
        "scientific_queries": 0,
        "cuda_executions": 0,
    })
    try:
        verified = runner._verify_pre_layernorm_input_capture(
            manifest_path, allow_legacy_missing_canonical=True,
            expected_canonical_identity=comparison[
                "job2997_live_canonical_identity"])
        comparison["semantic_verifier_accepts"] = True
        comparison["semantic_verifier_result"] = verified
    except Exception as error:
        comparison["semantic_verifier_error"] = (
            f"{type(error).__name__}: {error}")
    identical = comparison["classification"] == \
        "PERSISTED_STATE_CANONICALLY_IDENTICAL"
    comparison["final_status"] = (
        "EXISTING_JOB2997_PRE_LAYERNORM_CAPTURE_AUTHENTICATED"
        if identical and comparison["semantic_verifier_accepts"]
        else "EXISTING_JOB2997_PRE_LAYERNORM_CAPTURE_REJECTED")
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--job2995-trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    try:
        report = diagnose(
            args.capture_root.expanduser().resolve(),
            args.job2995_trace.expanduser().resolve())
        exit_code = 0 if report["final_status"] == \
            "EXISTING_JOB2997_PRE_LAYERNORM_CAPTURE_AUTHENTICATED" else 1
    except Exception as error:
        report = {
            "schema": SCHEMA,
            "expected_live_canonical_identity": None,
            "job2997_live_canonical_identity": None,
            "persisted_recomputed_canonical_identity": None,
            "component_comparison": {},
            "raw_artifact_manifest_comparison": {},
            "first_differing_component": None,
            "classification": "PERSISTED_ARTIFACT_MANIFEST_MISMATCH",
            "diagnostic_error": f"{type(error).__name__}: {error}",
            "traceback_tail": traceback.format_exc().splitlines()[-30:],
            "final_status":
                "EXISTING_JOB2997_PRE_LAYERNORM_CAPTURE_REJECTED",
            "scientific_queries": 0,
            "cuda_executions": 0,
        }
        exit_code = 1
    _write_report(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
