#!/usr/bin/env python3
"""Fail-closed merge of two fresh homogeneous-A40 worker trees."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from a40_fresh_common import (TOP_LEVEL_SEED_FILES, assigned_properties,
    load_chunk, load_plan, result_root, seed_record)
from cluster_common import (BASELINE_REL, DEEPT_COMMIT, REPO, artifact_root,
    baseline_root, canonical, load_production_manifest, sha256, verified_json,
    verify_artifact_manifest)
from merge_worker_results import validate_property
from portable_runner import configure


RECIPROCAL_NAN = (
    "Reciprocal: there are NaNs in the new COEFFS, pre-condition not met")
DOMAIN_PREFIXES = (
    "sqrt: Bounds must be positive",
    "reciprocal: Bounds must be positive",
)


def verify_source_and_model_identities(artifact: Path, manifest: dict) -> None:
    source_manifest = json.loads(
        (REPO / "frozen/source_tree_manifest.json").read_text())
    for row in source_manifest["files"]:
        path = REPO / row["path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise RuntimeError(f"frozen source identity differs: {path}")
    deept = REPO / "research_hab/public_benchmarks/DeepT"
    revision = subprocess.check_output(
        ["git", "-C", str(deept), "rev-parse", DEEPT_COMMIT],
        text=True).strip()
    if revision != DEEPT_COMMIT:
        raise RuntimeError("DeepT revision differs")
    for name in ("checkpoint", "config", "vocab"):
        row = manifest["model"][name]
        blob = subprocess.check_output(
            ["git", "-C", str(deept), "show",
             f"{DEEPT_COMMIT}:{row['git_path']}"])
        if hashlib.sha256(blob).hexdigest() != row["sha256"]:
            raise RuntimeError(f"DeepT {name} identity differs")


def validate_domain_failures(property_root: Path) -> None:
    for path in property_root.glob("queries/query_*_result_v1.json"):
        row = verified_json(path)
        if row["terminal_status"] != "UNCERTIFIED_DOMAIN_FAILURE":
            continue
        message = row.get("exception_message", "")
        if row.get("exception_type") != "AssertionError" or not (
                message == RECIPROCAL_NAN
                or any(message.startswith(prefix) for prefix in DOMAIN_PREFIXES)):
            raise RuntimeError(f"unrecognized typed domain failure: {path}")
        if row.get("certified") is not False:
            raise RuntimeError(f"typed domain failure claimed certification: {path}")


def copy_property(source: Path, destination: Path) -> None:
    if destination.exists():
        raise RuntimeError(f"duplicate/conflicting property: {destination.name}")
    shutil.copytree(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-root", required=True)
    parser.add_argument("--artifact-root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    artifact = artifact_root(args.artifact_root)
    verify_artifact_manifest(artifact)
    manifest = load_production_manifest(artifact)
    verify_source_and_model_identities(artifact, manifest)
    plan = load_plan()
    fresh_root = Path(args.fresh_root).resolve()
    output_container = Path(args.output).resolve()
    if output_container.exists():
        raise FileExistsError(output_container)
    output = output_container / BASELINE_REL
    output.mkdir(parents=True)
    (output / "properties").mkdir()
    for name in TOP_LEVEL_SEED_FILES:
        shutil.copy2(baseline_root(artifact) / name, output / name)
    merged = []
    for worker_id in range(2):
        worker_dir = fresh_root / f"worker_{worker_id}"
        marker = verified_json(worker_dir / "fresh_worker_seed_v1.json")
        if marker != seed_record(artifact, worker_id):
            raise RuntimeError("worker fresh-seed identity differs")
        expected = assigned_properties(plan, worker_id)
        actual = {path.parent.name for path in
                  (result_root(worker_dir) / "properties").glob("*/result_v1.json")}
        if actual != expected:
            raise RuntimeError(f"worker {worker_id} completion set differs")
        for chunk_index in range(4):
            chunk = load_chunk(worker_id, chunk_index)
            completion = verified_json(
                worker_dir / "chunk_records"
                / f"chunk_{chunk_index}_complete_v1.json")
            if completion["chunk_manifest_sha256"] != chunk[
                    "canonical_manifest_sha256"]:
                raise RuntimeError("chunk completion identity differs")
        for property_id in sorted(expected):
            source = result_root(worker_dir) / "properties" / property_id
            result = validate_property(result_root(worker_dir), property_id)
            if (result.get("reused_from_optimized_smoke") is not False
                    or int(result.get("fresh_verifier_evaluations_this_run", 0)) <= 0):
                raise RuntimeError(
                    f"property was not freshly evaluated on A40: {property_id}")
            validate_domain_failures(source)
            copy_property(source, output / "properties" / property_id)
            merged.append(property_id)
    ordered = [row["property_id"] for row in manifest["properties"]]
    if len(merged) != 127 or set(merged) != set(ordered):
        raise RuntimeError("fresh merged property union differs")
    for row in manifest["historical_duplicate_entries"]:
        if not (output / "properties" / row["property_id"]
                / "result_v1.json").is_file():
            raise RuntimeError("historical duplicate entry is absent")
    revision = DEEPT_COMMIT
    runner, _, configured = configure(output_container, artifact)
    if configured != output:
        raise RuntimeError("merged output mapping differs")
    runner.summarize()
    summary = verified_json(
        output / "coret_optimized_historical_127_summary_v2.json")
    report = {
        "schema": "CORET_A40_FRESH_127_MERGE_REPORT_V1",
        "status": "A40_FRESH_127_COMPLETE",
        "scientific_manifest_sha256": plan["scientific_manifest_sha256"],
        "production_manifest_sha256": plan["production_manifest_sha256"],
        "plan_sha256": plan["canonical_manifest_sha256"],
        "completed_properties": 127,
        "imported_A4000_completed_properties": 0,
        "generic_fallbacks": summary["generic_fallbacks"],
        "checker_failures": summary["checker_failures"],
        "soundness_or_proof_failures": summary[
            "soundness_or_proof_failures"],
        "summary_record_sha256": summary["record_sha256"],
        "DeepT_revision": revision,
    }
    if any(report[key] != 0 for key in (
            "generic_fallbacks", "checker_failures",
            "soundness_or_proof_failures")):
        raise RuntimeError("merged scientific acceptance failure")
    report["record_sha256"] = canonical(report)
    (output_container / "a40_fresh_merge_report_v1.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__": main()
