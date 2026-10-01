#!/usr/bin/env python3
"""One-property opt-in PSD-aware Block-2 LayerNorm experiment."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "research_hab")]

import cluster_common
import coret_psd_layernorm_experiment_v1 as experiment
import run_sound_fp64_3l_campaign as campaign
import run_sound_fp64_3l_psd_state_capture_v1 as capture
import run_sound_fp64_finish_3l_v1 as finish3l


SCHEMA = "CORET_PSD_LAYERNORM_EXPERIMENT_JOB_V1"
PROPERTY_ID = "deept_table7_stdln3_s001_line1794_tok11"
MULTIPLIER = "0.75"
TESTED_RADIUS = 0.00060791015625
TESTED_RADIUS_HEX = "0x1.3eb851eb851ecp-11"


def _atomic_json(path: Path, value: dict) -> dict:
    payload = dict(value)
    payload["record_sha256"] = cluster_common.canonical(payload)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return cluster_common.verified_json(path)


def _authenticate_oracle(path: Path) -> dict:
    report = cluster_common.verified_json(path)
    if (report.get("schema") != "CORET_PSD_LAYERNORM_VARIANCE_ORACLE_V1"
            or report.get("property_id") != PROPERTY_ID):
        raise RuntimeError("PSD oracle report identity differs")
    rows = report.get("results")
    matches = [row for row in rows if row.get("multiplier") == MULTIPLIER
               and row.get("variant") == "complete"] if isinstance(rows, list) else []
    if len(matches) != 1:
        raise RuntimeError("PSD oracle complete 0.75x record differs")
    row = matches[0]
    if (float(row.get("tested_radius")) != TESTED_RADIUS
            or row.get("psd_positive") is not True
            or float(row.get("psd_dual_candidate_lower_outward_safe")) <= 0):
        raise RuntimeError("PSD oracle positive certificate differs")
    return {"path": str(path), "sha256": cluster_common.sha256(path),
            "record": row}


@contextlib.contextmanager
def _installed_finish_hook(records: list[dict]):
    original = finish3l.execute

    def wrapped(*args, **kwargs):
        kwargs["experimental_post_attention_layernorm"] = (
            experiment.execute_experimental_layernorm)
        report = original(*args, **kwargs)
        certificate = report.get("experimental_psd_layernorm")
        if not isinstance(certificate, dict):
            raise RuntimeError("experimental PSD certificate was not emitted")
        records.append(certificate)
        return report

    finish3l.execute = wrapped
    try:
        yield
    finally:
        finish3l.execute = original


def execute(campaign_root: Path, artifact_root: Path, capture_manifest: Path,
            oracle_report: Path, output_root: Path, device_index: int) -> dict:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("refusing to overwrite PSD experiment root")
    output_root.mkdir(parents=True, exist_ok=True)
    capture_identity = capture.verify_capture(capture_manifest)
    oracle_identity = _authenticate_oracle(oracle_report)
    row, _source, _identity = capture._prepare_row(campaign_root, artifact_root)
    if (row.get("property_id") != PROPERTY_ID
            or campaign._candidate_radius(row) != TESTED_RADIUS):
        raise RuntimeError("PSD experiment property/radius identity differs")
    records = []
    execution_root = output_root / "scientific_execution"
    device = f"cuda:{device_index}"
    campaign._property_boundary_cleanup(device)
    try:
        with _installed_finish_hook(records):
            result = campaign.execute_property(row, execution_root, device)
    finally:
        campaign._property_boundary_cleanup(device)
    if len(records) != 1:
        raise RuntimeError("PSD experiment LayerNorm invocation count differs")
    certificate_path = output_root / "psd_layernorm_certificate.json"
    certificate = _atomic_json(certificate_path, records[0])
    result_path = execution_root / "properties" / PROPERTY_ID / "result.json"
    if not result_path.is_file():
        raise RuntimeError("PSD experiment property result is absent")
    result = cluster_common.verified_json(result_path)
    return _atomic_json(output_root / "experiment_report.json", {
        "schema": SCHEMA,
        "verdict": "CORET_PSD_LAYERNORM_EXPERIMENT_COMPLETE",
        "property_id": PROPERTY_ID,
        "multiplier": MULTIPLIER,
        "tested_radius": TESTED_RADIUS,
        "tested_radius_hex": TESTED_RADIUS_HEX,
        "capture_manifest": {
            "path": str(capture_manifest),
            "sha256": cluster_common.sha256(capture_manifest),
            "identity": capture_identity,
        },
        "oracle_report": oracle_identity,
        "certificate_path": str(certificate_path),
        "certificate_sha256": cluster_common.sha256(certificate_path),
        "certificate_record_sha256": certificate["record_sha256"],
        "property_result_path": str(result_path),
        "property_result_sha256": cluster_common.sha256(result_path),
        "terminal_status": result.get("terminal_status"),
        "certified": result.get("certified_at_historical_radius"),
        "final_sound_lower_margin": result.get("final_sound_lower_margin"),
        "next_failure_stage": result.get("failure_stage"),
        "next_failure_reason": result.get("failure_reason"),
        "generic_fallback_count": result.get("generic_fallback_count"),
        "scientific_queries": 1,
        "bound_calls": 1,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--capture-manifest", required=True, type=Path)
    parser.add_argument("--oracle-report", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()
    report = execute(
        args.campaign_root.resolve(), args.artifact_root.resolve(),
        args.capture_manifest.resolve(), args.oracle_report.resolve(),
        args.output_root.resolve(), args.device_index)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
