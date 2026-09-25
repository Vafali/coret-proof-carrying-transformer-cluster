#!/usr/bin/env python3
"""Compare cluster calibration output to the immutable A4000 payload."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cluster_common import BASELINE_REL, artifact_root, canonical, verified_json
from run_calibration import CASES


DROP_KEYS = {"record_sha256", "total_wall_time_seconds", "setup_seconds",
             "proof_generation_time_seconds", "independent_checker_time_seconds",
             "certificate_serialization_seconds", "bound_runtime_seconds",
             "concretize_seconds", "peak_CPU_RSS_bytes", "peak_GPU_allocated_bytes",
             "peak_GPU_reserved_bytes", "facade_total_seconds", "native_operator_seconds",
             "numerical_overhead_seconds", "per_block_QK_AV_timing", "family_timing_seconds",
             "operator_calls", "B2_QK_seconds"}


def strip_timing(value):
    if isinstance(value, dict):
        return {key: strip_timing(item) for key, item in value.items()
                if key not in DROP_KEYS and not key.endswith("_seconds")}
    if isinstance(value, list): return [strip_timing(item) for item in value]
    return value


def locate(root: Path, case: dict, suffix: str) -> Path:
    query = (root / BASELINE_REL / "properties" / case["property_id"] / "queries")
    matches = list(query.glob(f"query_{case['query_ordinal']:02d}_*_{suffix}_v1.json"))
    if len(matches) != 1: raise RuntimeError(f"calibration artifact unavailable: {query}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--calibration-output", required=True)
    parser.add_argument("--artifact-root")
    args = parser.parse_args()
    artifact = artifact_root(args.artifact_root); case = CASES[args.case]
    candidate_root = Path(args.calibration_output).resolve()
    reference_root = artifact / "payload"
    result_ref = verified_json(locate(reference_root, case, "result"))
    result_new = verified_json(locate(candidate_root, case, "result"))
    cert_ref = verified_json(locate(reference_root, case, "certificates"))
    cert_new = verified_json(locate(candidate_root, case, "certificates"))
    scientific_ref = {"result": strip_timing(result_ref), "certificate": strip_timing(cert_ref)}
    scientific_new = {"result": strip_timing(result_new), "certificate": strip_timing(cert_new)}
    exact = canonical(scientific_ref) == canonical(scientific_new)
    invariant = (result_new.get("independent_checker_accepted") is True
                 and result_new.get("all_support_claims_validated") is True
                 and result_new.get("generic_fallback_count") == 0
                 and result_new.get("certified") == result_ref.get("certified")
                 and result_new.get("rho_binary64_hex") == result_ref.get("rho_binary64_hex")
                 and cert_new.get("invocation_counts") == cert_ref.get("invocation_counts"))
    outward = False
    if result_new.get("direct_margin_interval") and result_ref.get("direct_margin_interval"):
        outward = (result_new["direct_margin_interval"][0] <= result_ref["direct_margin_interval"][0]
                   and result_new["direct_margin_interval"][1] >= result_ref["direct_margin_interval"][1])
    classification = ("BITWISE_SCIENTIFIC_PARITY" if exact else
        "SOUND_OUTWARD_HARDWARE_VARIATION" if invariant and outward else
        "CROSS_GPU_SCIENTIFIC_MISMATCH")
    print(json.dumps({"classification": classification, "case": args.case,
        "scientific_payload_sha256_reference": canonical(scientific_ref),
        "scientific_payload_sha256_candidate": canonical(scientific_new),
        "certified_reference": result_ref["certified"],
        "certified_candidate": result_new["certified"],
        "interval_reference": result_ref["direct_margin_interval"],
        "interval_candidate": result_new["direct_margin_interval"],
        "checker_invariants_pass": invariant, "outward_final_interval": outward,
        "total_wall_time_seconds": result_new["total_wall_time_seconds"],
        "peak_GPU_allocated_bytes": result_new["peak_GPU_allocated_bytes"],
        "peak_GPU_reserved_bytes": result_new["peak_GPU_reserved_bytes"]}, sort_keys=True))


if __name__ == "__main__": main()
