#!/usr/bin/env python3
"""One preregistered property, metadata inventory, or synthetic-safe aggregation.

No campaign loop, radius search, replacement, new checker, or verifier formulas.
The existing CUDA-only complete FP64 path is imported ONLY for manual execution.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import statistics
import time

import cluster_common as common
import preregister_transformer_benchmark24_v1 as protocol


DEFAULT_MANIFEST = protocol.REPO / "frozen/benchmark24/transformer_benchmark24_manifest_v1.json"
FROZEN_MANIFEST_SHA = "46f5380d7bfacd0cc61da2938465d951fa3e72246b01a98223352510b331a338"
RESULT_SCHEMA = "CORET_BENCHMARK24_ONE_PROPERTY_RESULT_V1"
SUMMARY_SCHEMA = "CORET_BENCHMARK24_AGGREGATE_V1"


def atomic_record(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError("refusing to overwrite a benchmark result")
    payload = {**record, "record_sha256": common.canonical(record)}
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)
    return payload


def read_protocol(path=DEFAULT_MANIFEST, *, expected_sha=FROZEN_MANIFEST_SHA):
    manifest = protocol.load_manifest(path)
    if manifest["manifest_sha256"] != expected_sha:
        raise RuntimeError("frozen benchmark manifest SHA differs")
    return manifest


def execution_source_errors(manifest):
    errors = []
    for name, expected in manifest["frozen_execution_source_hashes"].items():
        path = protocol.REPO / name
        if not path.is_file() or common.sha256(path) != expected:
            errors.append({"kind": "EXECUTION_SOURCE_MISSING_OR_CORRUPT", "path": name})
    return errors


def artifact_errors(manifest, artifact_root):
    errors = []
    for key in ("production_manifest", "scientific_manifest", "DeepT_positions"):
        identity = manifest["source_artifacts"][key]
        path = artifact_root / identity["relative_path"]
        if not path.is_file():
            errors.append({"kind": "FROZEN_ARTIFACT_MISSING", "artifact": key, "path": str(path)})
        elif common.sha256(path) != identity["sha256"]:
            errors.append({"kind": "FROZEN_ARTIFACT_CORRUPT", "artifact": key, "path": str(path)})
    return errors


def inventory(manifest, artifact_root):
    errors = execution_source_errors(manifest) + artifact_errors(manifest, artifact_root)
    return {"schema": "CORET_BENCHMARK24_MEASUREMENT_INVENTORY_V1",
            "manifest_sha256": manifest["manifest_sha256"], "property_count": 24,
            "scientific_queries": 0, "bound_calls": 0, "cuda_calls": 0,
            "properties": [{"property_id": row["property_id"], "tested_radius": row["tested_radius"],
                "input_readiness": "MISSING_OR_CORRUPT_ARTIFACTS" if errors else "METADATA_INPUTS_AVAILABLE",
                "input_errors": errors, "measurement_inventory": row["measurement_inventory"],
                "benchmark_outcome": "NOT_RUN"} for row in manifest["properties"]]}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def normalize(manifest, row, raw, certificate_report=None):
    if (raw.get("property_id") != row["property_id"] or raw.get("historical_candidate_radius_hex") != row["tested_radius_hex"] or
            raw.get("candidate_source") != protocol.RADIUS_SOURCE or
            raw.get("clean_label") != row["clean_label"] or raw.get("binary_search_performed") is not False):
        raise RuntimeError("producer result property/radius/source identity differs")
    report = certificate_report or {}
    complete = raw.get("terminal_status") == "COMPLETE"
    domain = raw.get("terminal_status") == "UNCERTIFIED_DOMAIN_FAILURE"
    if complete and (not _finite(raw.get("final_sound_lower_margin")) or
                     raw.get("certified_at_historical_radius") is not (raw["final_sound_lower_margin"] > 0) or
                     raw.get("scientific_evaluation_complete") is not True or raw.get("generic_fallback_count") != 0):
        raise RuntimeError("completed producer result status differs")
    if domain and (raw.get("scientific_evaluation_complete") is not True or
                   raw.get("certified_at_historical_radius") is not False or raw.get("generic_fallback_count") != 0):
        raise RuntimeError("domain failure result status differs")
    if not _finite(raw.get("runtime_seconds")) or raw["runtime_seconds"] < 0:
        raise RuntimeError("result runtime differs")
    # No existing independent COMPLETE FP64 graph checker is integrated. Do
    # not manufacture PASS from artifact hashes, representative spots or flags.
    checker = "NOT_AVAILABLE"
    producer_positive = complete and raw["certified_at_historical_radius"] is True
    outcome = "INCONCLUSIVE" if complete or domain else "REJECTED_ERROR"
    reductions = report.get("reductions")
    diagnostic = raw.get("domain_failure_diagnostic") or {}
    return {
        "schema": RESULT_SCHEMA, "manifest_sha256": manifest["manifest_sha256"],
        "source_population_sha256": manifest["source_population_sha256"],
        "property_id": row["property_id"], "tested_radius": row["tested_radius"],
        "tested_radius_hex": row["tested_radius_hex"], "radius_source": protocol.RADIUS_SOURCE,
        "radius_stratum": row["radius_stratum"], "sequence_length_stratum": row["sequence_length_stratum"],
        "final_proof_status": outcome,
        "producer_proof_status": "CERTIFIED_MARGIN" if producer_positive else "UNCERTIFIED" if complete or domain else "ERROR",
        "independently_checked_certificate_status": checker,
        "inconclusive_reason": "COMPLETE_INDEPENDENT_CHECKER_NOT_AVAILABLE" if producer_positive else
                               "SOUND_DOMAIN_FAILURE" if domain else "NONPOSITIVE_SOUND_MARGIN" if complete else None,
        "certificate_integrity_status": "PASS" if complete else "NO_COMPLETE_CERTIFICATE",
        "final_sound_lower_margin": raw.get("final_sound_lower_margin"),
        "runtime_seconds": raw["runtime_seconds"],
        "peak_cpu_rss_bytes": raw.get("peak_cpu_rss_bytes"),
        "peak_gpu_allocated_bytes": raw.get("peak_gpu_allocated_bytes"),
        "peak_gpu_reserved_bytes": raw.get("peak_gpu_reserved_bytes"),
        "memory_scope": row["measurement_inventory"]["peak_memory"]["scope"],
        "failure_stage": diagnostic.get("stage") or raw.get("failure_stage"),
        "failure_operator": diagnostic.get("operator"), "failure_reason": raw.get("failure_reason"),
        "final_generator_count": raw.get("final_generator_count"),
        "reduction_count": len(reductions) if isinstance(reductions, list) else None,
        "reduction_scope": "BLOCK2_TO_HEAD_ONLY", "reduction_inflation_max": report.get("reduction_inflation_max"),
        "numerical_widening": raw.get("numerical_widening"),
        "max_numerical_native_ratio": raw.get("max_numerical_native_ratio"),
        "representative_mpfr_checks_performed": report.get("representative_mpfr_checks_performed"),
        "generic_fallback_count": raw.get("generic_fallback_count"), "binary_search_performed": False,
    }


def _existing_backend(row, workspace, artifact_root, device):
    if not device.startswith("cuda:"):
        raise RuntimeError("CPU_EXECUTION_UNSUPPORTED_BY_EXISTING_COMPLETE_FP64_BACKEND")
    import run_sound_fp64_3l_campaign as campaign
    manifest = common.load_production_manifest(artifact_root)
    resolved, _ = campaign._resolved_campaign_inputs(artifact_root, manifest)
    actual = resolved[row["property_id"]]
    if (any(actual[key] != row[key] for key in protocol.FIELDS) or actual["token_ids"] != row["token_ids"] or
            campaign._candidate_radius(actual).hex() != row["tested_radius_hex"]):
        raise RuntimeError("resolved source differs from preregistration")
    raw = campaign.execute_property(actual, workspace, device)
    # This validates artifact/report hashes, not complete numerical soundness.
    directory = workspace / "properties" / row["property_id"]
    campaign._verified_result(directory / "result.json")
    report_path = directory / "certificate_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    return raw, report


def run_one(manifest, property_id, artifact_root, result_root, device, *, backend=None):
    matches = [row for row in manifest["properties"] if row["property_id"] == property_id]
    if len(matches) != 1:
        raise RuntimeError("property ID is not a unique frozen member; no replacement permitted")
    row = matches[0]
    result_root, artifact_root = result_root.resolve(), artifact_root.resolve()
    if (result_root == protocol.REPO or result_root.is_relative_to(artifact_root) or
            result_root.is_relative_to(protocol.REPO / "frozen")):
        raise RuntimeError("result root would modify frozen inputs/repository root")
    result_path = result_root / "records" / f"{property_id}.json"
    if result_path.exists():
        previous = common.verified_json(result_path)
        validate_result(manifest, previous)
        return previous
    started = time.perf_counter()
    try:
        errors = execution_source_errors(manifest) + artifact_errors(manifest, artifact_root)
        if errors:
            raise RuntimeError(json.dumps(errors, sort_keys=True))
        raw, report = (backend or _existing_backend)(row, result_root / "producer", artifact_root, device)
        result = normalize(manifest, row, raw, report)
    except Exception as error:
        result = {"schema": RESULT_SCHEMA, "manifest_sha256": manifest["manifest_sha256"],
                  "source_population_sha256": manifest["source_population_sha256"],
                  "property_id": property_id, "tested_radius": row["tested_radius"],
                  "tested_radius_hex": row["tested_radius_hex"], "radius_source": protocol.RADIUS_SOURCE,
                  "radius_stratum": row["radius_stratum"], "sequence_length_stratum": row["sequence_length_stratum"],
                  "final_proof_status": "REJECTED_ERROR", "producer_proof_status": "NOT_COMPLETED",
                  "independently_checked_certificate_status": "NOT_RUN",
                  "certificate_integrity_status": "NOT_RUN", "failure_stage": "input_or_backend_adapter",
                  "failure_operator": None, "failure_reason": f"{type(error).__name__}: {error}",
                  "runtime_seconds": time.perf_counter() - started,
                  "final_sound_lower_margin": None, "binary_search_performed": False}
    result["device_requested"] = device
    return atomic_record(result_path, result)


def validate_result(manifest, result):
    row = next((item for item in manifest["properties"] if item["property_id"] == result.get("property_id")), None)
    if (row is None or result.get("schema") != RESULT_SCHEMA or
            result.get("manifest_sha256") != manifest["manifest_sha256"] or
            result.get("source_population_sha256") != manifest["source_population_sha256"] or
            result.get("tested_radius_hex") != row["tested_radius_hex"] or result.get("tested_radius") != row["tested_radius"] or
            result.get("radius_source") != protocol.RADIUS_SOURCE or result.get("radius_stratum") != row["radius_stratum"] or
            result.get("sequence_length_stratum") != row["sequence_length_stratum"] or result.get("binary_search_performed") is not False or
            result.get("final_proof_status") not in ("CERTIFIED", "INCONCLUSIVE", "REJECTED_ERROR") or
            not _finite(result.get("runtime_seconds")) or result["runtime_seconds"] < 0):
        raise RuntimeError("benchmark result identity/status differs")
    if result["final_proof_status"] == "CERTIFIED" and (
            result.get("independently_checked_certificate_status") != "PASS" or
            not _finite(result.get("final_sound_lower_margin")) or result["final_sound_lower_margin"] <= 0):
        raise RuntimeError("certification lacks independent check/positive margin")


def aggregate(manifest, rows):
    for row in rows:
        validate_result(manifest, row)
    if len({row["property_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate result; do not silently choose one")
    def summarize(population, results):
        counts = Counter(row["final_proof_status"] for row in results)
        runtimes = [row["runtime_seconds"] for row in results]
        return {"denominator": len(population), "evaluated": len(results), "not_run": len(population) - len(results),
                "certified": counts["CERTIFIED"], "inconclusive": counts["INCONCLUSIVE"], "rejected_error": counts["REJECTED_ERROR"],
                "runtime_median_seconds": statistics.median(runtimes) if runtimes else None,
                "runtime_max_seconds": max(runtimes) if runtimes else None,
                "producer_positive_margin_count": sum(row.get("producer_proof_status") == "CERTIFIED_MARGIN" for row in results),
                "independent_certificate_check_success": sum(row.get("independently_checked_certificate_status") == "PASS" for row in results),
                "independent_checker_status_histogram": dict(Counter(row.get("independently_checked_certificate_status") for row in results)),
                "failure_stage_histogram": dict(Counter(row["failure_stage"] for row in results if row.get("failure_stage")))}
    summary = {"schema": SUMMARY_SCHEMA, "manifest_sha256": manifest["manifest_sha256"],
               "source_population_sha256": manifest["source_population_sha256"], **summarize(manifest["properties"], rows)}
    summary["by_radius_stratum"] = {stratum: summarize([r for r in manifest["properties"] if r["radius_stratum"] == stratum],
        [r for r in rows if r["radius_stratum"] == stratum]) for stratum in protocol.STRATA}
    summary["by_sequence_length_proxy"] = {stratum: summarize([r for r in manifest["properties"] if r["sequence_length_stratum"] == stratum],
        [r for r in rows if r["sequence_length_stratum"] == stratum]) for stratum in ("short_le12", "medium_13to20", "long_gt20")}
    summary["difficulty_definition"] = "historical radius rank and sequence length only; not current verifier success"
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inventory", "one", "aggregate"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--artifact-root", type=Path, default=protocol.REPO / "runtime_inputs")
    parser.add_argument("--property-id")
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = read_protocol(args.manifest)
    if args.mode == "inventory":
        report = inventory(manifest, args.artifact_root)
    elif args.mode == "one":
        if not args.property_id or args.result_root is None:
            parser.error("one requires --property-id and --result-root")
        report = run_one(manifest, args.property_id, args.artifact_root, args.result_root, args.device)
    else:
        if args.result_root is None:
            parser.error("aggregate requires --result-root")
        rows = [common.verified_json(path) for path in sorted((args.result_root / "records").glob("*.json"))]
        report = aggregate(manifest, rows)
    if args.output:
        atomic_record(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
