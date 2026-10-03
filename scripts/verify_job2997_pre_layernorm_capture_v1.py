#!/usr/bin/env python3
"""CPU-only authentication of the persisted job2997 pre-LayerNorm state."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "research_hab")]

import cluster_common
import run_sound_fp64_3l_psd_layernorm_experiment_v1 as runner


SCHEMA = "CORET_JOB2997_PRE_LAYERNORM_CAPTURE_VERIFICATION_V1"
COMPONENTS = (
    "weights_sha256", "ordered_generator_ids_sha256",
    "range_low_sha256", "range_high_sha256", "support_masks_sha256",
    "provenance_reasons_sha256", "generator_count", "token_count",
    "hidden_dimension", "canonical_state_identity_sha256",
)


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


def verify(capture_root: Path, job2995_trace: Path) -> dict:
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

    recomputed = runner._snapshot_state_identity(snapshot)
    live_job2997 = _target_identity(trace_path)
    live_job2995 = _target_identity(job2995_trace)
    component_checks = {
        field: {
            "job2997_live": live_job2997.get(field),
            "job2995_live": live_job2995.get(field),
            "persisted_recomputed": recomputed.get(field),
            "equal": (live_job2997.get(field) == live_job2995.get(field)
                      == recomputed.get(field)),
        }
        for field in COMPONENTS
    }
    if not all(row["equal"] for row in component_checks.values()):
        first = next(name for name, row in component_checks.items()
                     if not row["equal"])
        raise RuntimeError(
            f"persisted canonical state differs first at {first}")
    if recomputed["canonical_state_identity_sha256"] != \
            runner.EXPECTED_INPUT_CANONICAL_STATE_IDENTITY_SHA256:
        raise RuntimeError("persisted canonical state identity is unexpected")

    verified = runner._verify_pre_layernorm_input_capture(
        manifest_path, allow_legacy_missing_canonical=True,
        expected_canonical_identity=live_job2997)
    state_hashes = runner._frontier_state_hashes(snapshot)
    state_hashes.update({
        "source_ids_sha256": state_hashes["generator_ids_sha256"],
        "hidden_dimension": state_hashes["feature_dimension"],
    })
    row_hash = runner.capture._json_sha(state_hashes)
    linkage_hash = (manifest.get("invocation_linkage") or {}).get(
        "input_state_identity_sha256")
    if row_hash != linkage_hash:
        raise RuntimeError("manifest state-row linkage hash differs")
    return {
        "schema": SCHEMA,
        "live_trace_canonical_identity": live_job2997,
        "job2995_live_trace_canonical_identity": live_job2995,
        "persisted_recomputed_canonical_identity": recomputed,
        "canonical_identity_equal": True,
        "all_component_checks": component_checks,
        "manifest_state_row_identity_sha256": row_hash,
        "manifest_state_row_identity_schema":
            "CORET_FRONTIER_MANIFEST_STATE_ROW_V1",
        "manifest_state_row_hash_is_canonical_identity": False,
        "persisted_artifact_path": str(artifact_path),
        "persisted_artifact_sha256": cluster_common.sha256(artifact_path),
        "legacy_manifest_without_canonical": verified[
            "legacy_manifest_without_canonical"],
        "final_status":
            "EXISTING_JOB2997_PRE_LAYERNORM_CAPTURE_AUTHENTICATED",
        "scientific_queries": 0,
        "cuda_executions": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--job2995-trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    report = verify(
        args.capture_root.expanduser().resolve(),
        args.job2995_trace.expanduser().resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
