#!/usr/bin/env python3
"""ONE manual sound-FP64 property with the semantic epsilon-floor LayerNorm revision.

Uses frozen benchmark metadata unchanged, explicitly records the new producer
revision instead of pretending to satisfy the old producer source hashes.
"""
import argparse
import json
from pathlib import Path

import cluster_common as C
import run_transformer_benchmark24_v1 as B

REVISION = "SOUND_FP64_SEMANTIC_EPSILON_FLOOR_V1"
CHANGED_EXECUTION_FILES = {
    "research_hab/coret_sound_fp64_block0_feasibility_v1.py",
    "research_hab/coret_psd_layernorm_experiment_v1.py",
    "scripts/run_sound_fp64_finish_3l_v1.py",
    "scripts/run_sound_fp64_3l_campaign.py",
}
ADDED_EXECUTION_FILES = {
    "scripts/sound_fp64_layernorm_separator_v1.py",
    "scripts/separating_variance_checker_v1.py",
    "scripts/certify_block2_output_separating_variance_v1.py",
    "scripts/run_sound_fp64_separator_property_v1.py",
    "scripts/semantic_epsilon_floor_checker_v1.py",
    "scripts/sound_fp64_layernorm_epsilon_floor_v1.py",
    "scripts/run_sound_fp64_epsilon_floor_property_v1.py",
}


def source_audit(manifest):
    revision = C.verified_json(C.REPO/"frozen/layernorm_epsilon_floor_source_revision_v1.json")
    if revision["producer_revision"] != REVISION or revision["benchmark_manifest_sha256"] != manifest["manifest_sha256"]:
        raise RuntimeError("separating LayerNorm source revision identity differs")
    expected_files = CHANGED_EXECUTION_FILES | ADDED_EXECUTION_FILES
    if set(revision["source_hashes"]) != expected_files:
        raise RuntimeError("separating LayerNorm source revision inventory differs")
    for name, expected in revision["source_hashes"].items():
        if C.sha256(C.REPO/name) != expected:
            raise RuntimeError(f"separating LayerNorm source revision SHA differs: {name}")
    changed = {}
    for name, original in manifest["frozen_execution_source_hashes"].items():
        current = C.sha256(C.REPO/name)
        if current != original:
            if name not in CHANGED_EXECUTION_FILES:
                raise RuntimeError(f"unapproved source change outside LayerNorm repair: {name}")
            changed[name] = {"original_frozen_sha256": original, "current_sha256": current}
    return {"producer_revision": REVISION, "changed_sources": changed,
            "repair_source_hashes": {name: C.sha256(C.REPO/name) for name in sorted(
                CHANGED_EXECUTION_FILES | ADDED_EXECUTION_FILES)},
            "source_revision_file_sha256": C.sha256(C.REPO/"frozen/layernorm_epsilon_floor_source_revision_v1.json"),
            "benchmark_manifest_unchanged": True}


def execute(property_id, artifact_root, result_root, device):
    manifest = B.read_protocol()
    audit = source_audit(manifest)
    rows = [r for r in manifest["properties"] if r["property_id"] == property_id]
    if len(rows) != 1:
        raise RuntimeError("one exact frozen benchmark member required")
    if (result_root.exists() or result_root.resolve() == C.REPO
            or result_root.resolve().is_relative_to(C.REPO/"frozen")
            or result_root.resolve().is_relative_to(artifact_root.resolve())):
        raise RuntimeError("single validation requires a NEW isolated result root")
    errors = B.artifact_errors(manifest, artifact_root)
    if errors:
        raise RuntimeError(f"frozen artifact errors: {errors}")
    # Reuse the established complete property path; no radius search or new graph.
    raw, producer_report = B._existing_backend(rows[0], result_root/"producer", artifact_root, device)
    normalized = B.normalize(manifest, rows[0], raw, producer_report)
    normalized["producer_revision"] = REVISION
    normalized["source_audit"] = audit
    normalized["layernorm_separator_witnesses_path"] = raw.get("layernorm_separator_witnesses_path")
    normalized["layernorm_separator_witnesses_sha256"] = raw.get("layernorm_separator_witnesses_sha256")
    normalized["layernorm_separator_repaired_tokens"] = raw.get("layernorm_separator_repaired_tokens", [])
    normalized["layernorm_domain_witnesses_path"] = raw.get("layernorm_separator_witnesses_path")
    normalized["layernorm_domain_witnesses_sha256"] = raw.get("layernorm_separator_witnesses_sha256")
    normalized["layernorm_domain_repaired_tokens"] = raw.get("layernorm_separator_repaired_tokens", [])
    # Separator/epsilon-floor obligations are independently checked; a complete FP64 graph
    # checker is still NOT_AVAILABLE. Never upgrade that status here.
    B.validate_result(manifest, normalized)
    B.atomic_record(result_root/"records"/(property_id+".json"), normalized)
    return normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--property-id", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0", choices=["cuda:0"])
    args = parser.parse_args()
    print(json.dumps(execute(args.property_id, args.artifact_root.resolve(), args.result_root.resolve(), args.device),
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
