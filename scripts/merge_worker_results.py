#!/usr/bin/env python3
"""Fail-closed, non-overwriting merge of isolated worker result trees."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from cluster_common import (BASELINE_REL, artifact_root, baseline_root,
    canonical, load_production_manifest, load_shard, sha256, verified_json,
    verify_baseline_immutable)


def copy_no_conflict(source: Path, destination: Path) -> None:
    if destination.exists():
        if not destination.is_file() or sha256(source) != sha256(destination):
            raise RuntimeError(f"conflicting merge record: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def validate_property(root: Path, property_id: str) -> dict:
    directory = root / "properties" / property_id
    result = verified_json(directory / "result_v1.json")
    if (result.get("terminal_status") != "COMPLETE"
            or result.get("independent_checker_accepted") is not True
            or result.get("generic_fallback_count") != 0
            or result.get("all_support_claims_validated") is not True
            or result.get("all_provenance_consistent") is not True
            or result.get("outward_fresh_radius_envelopes") is not True
            or result.get("deterministic_support_reduction_lineage") is not True
            or result.get("checker_failure_count") != 0
            or result.get("soundness_or_proof_failure_count") != 0):
        raise RuntimeError(f"property acceptance failure: {property_id}")
    queries = sorted((directory / "queries").glob("query_*_result_v1.json"))
    rows = [verified_json(path) for path in queries]
    if [row["query_ordinal"] for row in rows] != list(range(len(rows))):
        raise RuntimeError(f"query ordinal chain differs: {property_id}")
    for path, row in zip(queries, rows):
        predicates = row.get("acceptance_predicates", {})
        if (row.get("independent_checker_accepted") is not True
                or row.get("generic_fallback_count") != 0
                or row.get("all_support_claims_validated") is not True
                or row.get("provenance_ID_consistent") is not True
                or row.get("outward_fresh_radius_envelopes") is not True
                or row.get("deterministic_support_reduction_lineage") is not True
                or row.get("acceptance_first_failure") is not None
                or not predicates or not all(predicates.values())):
            raise RuntimeError(f"query acceptance failure: {path}")
        certificate = path.with_name(path.name.replace("_result_", "_certificates_"))
        if row.get("complete_certificate") and not certificate.is_file():
            raise RuntimeError(f"missing query certificate: {certificate}")
        if certificate.is_file():
            cert = verified_json(certificate)
            cert_predicates = cert.get("acceptance_predicates", {})
            if (cert.get("independent_checker_accepted") is not True
                    or cert.get("generic_family_invocations") != 0
                    or cert.get("all_support_claims_validated") is not True
                    or cert.get("provenance_ID_consistent") is not True
                    or cert.get("outward_fresh_radius_envelopes") is not True
                    or cert.get("deterministic_support_reduction_lineage") is not True
                    or not cert_predicates
                    or not all(cert_predicates.values())):
                raise RuntimeError(f"certificate acceptance failure: {certificate}")
    journal = directory / "query_events_v1.jsonl"
    if journal.is_file():
        events = [json.loads(line) for line in journal.read_text().splitlines()
                  if line]
    elif queries:
        raise RuntimeError(f"missing query journal: {property_id}")
    else:
        # A frozen smoke-reuse property has no local verifier evaluations and
        # therefore legitimately has neither query records nor a journal.
        events = []
    for event in events:
        payload = dict(event); claimed = payload.pop("event_sha256")
        if canonical(payload) != claimed:
            raise RuntimeError(f"journal event hash differs: {journal}")
    if {row["record_sha256"] for row in rows} != {
            event["query_result_record_sha256"] for event in events}:
        raise RuntimeError(f"journal/query set differs: {property_id}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root")
    parser.add_argument("--shard-manifest", required=True)
    parser.add_argument("--worker-dir", action="append", required=True,
                        help="WORKER_ID=PATH")
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    artifact = artifact_root(args.artifact_root)
    manifest = load_production_manifest(artifact)
    shard = load_shard(Path(args.shard_manifest))
    workers = {}
    for item in args.worker_dir:
        key, value = item.split("=", 1); workers[int(key)] = Path(value).resolve()
    expected_ids = {row["worker_id"] for row in shard["workers"]}
    if set(workers) != expected_ids: raise RuntimeError("worker directory set differs")
    output = Path(args.output).resolve() / BASELINE_REL
    if output.exists(): raise FileExistsError("merge output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(baseline_root(artifact), output)
    verify_baseline_immutable(baseline_root(artifact), output)
    owner = {}
    for worker in shard["workers"]:
        source_root = workers[worker["worker_id"]] / BASELINE_REL
        verify_baseline_immutable(baseline_root(artifact), source_root)
        for assigned in worker["properties"]:
            property_id = assigned["property_id"]
            if property_id in owner: raise RuntimeError("duplicate property owner")
            owner[property_id] = worker["worker_id"]
            source = source_root / "properties" / property_id
            result = source / "result_v1.json"
            if not result.exists():
                if args.require_complete: raise RuntimeError(f"incomplete worker property: {property_id}")
                continue
            validate_property(source_root, property_id)
            for path in source.rglob("*"):
                if path.is_file(): copy_no_conflict(path, output / path.relative_to(source_root))
    partial_property = shard.get("partial_property")
    expected_partial_count = 1 if partial_property is not None else 0
    if sum(pid == partial_property for pid in owner) != expected_partial_count:
        raise RuntimeError("partial property owner differs")
    complete = []
    for prop in manifest["properties"]:
        path = output / "properties" / prop["property_id"] / "result_v1.json"
        if path.exists(): complete.append(validate_property(output, prop["property_id"]))
    if args.require_complete and len(complete) != 127:
        raise RuntimeError(f"final property count differs: {len(complete)}")
    duplicates = manifest["historical_duplicate_entries"]
    if len(duplicates) != 2:
        raise RuntimeError("historical duplicate preservation failed")
    if args.require_complete and any(
            not (output / "properties" / row["property_id"]).exists()
            for row in duplicates):
        raise RuntimeError("completed historical duplicate is missing")
    report = {"schema": "CORET_CLUSTER_MERGE_REPORT_V1",
              "scientific_manifest_sha256": manifest["historical_scientific_manifest"]["canonical_sha256"],
              "production_manifest_sha256": manifest["canonical_manifest_sha256"],
              "completed_properties": len(complete), "expected_properties": 127,
              "generic_fallbacks": sum(row["generic_fallback_count"] for row in complete),
              "conflicts": 0, "output": str(output)}
    report["record_sha256"] = canonical(report)
    report_path = output.parent / "cluster_merge_report_v1.json"
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__": main()
