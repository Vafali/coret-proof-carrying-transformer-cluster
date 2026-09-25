#!/usr/bin/env python3
"""Fresh homogeneous-A40 orchestration around the frozen verifier."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from cluster_common import (BASELINE_REL, PRODUCTION_MANIFEST_NAME, REPO,
    baseline_root, canonical, load_production_manifest, sha256, verified_json,
    verify_artifact_manifest)


PLAN = REPO / "frozen/a40_fresh_127_plan.json"
CHUNK_ROOT = REPO / "frozen/a40_chunks"
TOP_LEVEL_SEED_FILES = (
    "coret_optimized_historical_127_manifest_v1.json",
    "coret_optimized_historical_127_manifest_v2.json",
)


def load_plan() -> dict:
    return verified_json(PLAN, "canonical_manifest_sha256")


def chunk_path(worker_id: int, chunk_index: int) -> Path:
    return CHUNK_ROOT / f"worker_{worker_id}_chunk_{chunk_index}.json"


def load_chunk(worker_id: int, chunk_index: int) -> dict:
    plan = load_plan()
    chunk = verified_json(chunk_path(worker_id, chunk_index),
                          "canonical_manifest_sha256")
    if chunk["parent_plan_sha256"] != plan["canonical_manifest_sha256"]:
        raise RuntimeError("chunk parent plan differs")
    expected = plan["workers"][worker_id]["chunks"][chunk_index]
    if chunk["worker_id"] != worker_id or chunk["chunk_index"] != chunk_index:
        raise RuntimeError("chunk identity differs")
    if chunk["properties"] != expected["properties"]:
        raise RuntimeError("chunk property inventory differs")
    return chunk


def result_root(worker_dir: Path) -> Path:
    return worker_dir / BASELINE_REL


def seed_record(artifact: Path, worker_id: int) -> dict:
    artifact_manifest = verify_artifact_manifest(artifact)
    production = load_production_manifest(artifact)
    plan = load_plan()
    value = {
        "schema": "CORET_A40_FRESH_WORKER_SEED_V1",
        "worker_id": worker_id,
        "scientific_manifest_sha256": plan["scientific_manifest_sha256"],
        "production_manifest_sha256": production["canonical_manifest_sha256"],
        "artifact_manifest_sha256": artifact_manifest[
            "canonical_manifest_sha256"],
        "plan_sha256": plan["canonical_manifest_sha256"],
        "imported_A4000_completed_properties": 0,
        "imported_A4000_query_records": 0,
        "fresh_homogeneous_hardware": "NVIDIA_A40",
    }
    value["record_sha256"] = canonical(value)
    return value


def initialize_fresh_worker(artifact: Path, worker_dir: Path,
                            worker_id: int) -> Path:
    """Create only immutable metadata; never import A4000 property state."""
    source = baseline_root(artifact)
    destination = result_root(worker_dir)
    marker = worker_dir / "fresh_worker_seed_v1.json"
    expected = seed_record(artifact, worker_id)
    worker_dir.mkdir(parents=True, exist_ok=True)
    for name in ("runtime", "tmpdir", "cache", "cuda_cache",
                 "torch_extensions", "pycache", "chunk_records"):
        (worker_dir / name).mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "properties").mkdir(exist_ok=True)
    for name in TOP_LEVEL_SEED_FILES:
        original = source / name
        target = destination / name
        if target.exists():
            if sha256(target) != sha256(original):
                raise RuntimeError(f"fresh seed metadata differs: {name}")
        else:
            shutil.copy2(original, target)
    if marker.exists():
        if verified_json(marker) != expected:
            raise RuntimeError("fresh worker seed identity differs")
    else:
        marker.write_text(json.dumps(expected, sort_keys=True, indent=2) + "\n")
    return destination


def verify_no_imported_results(worker_dir: Path) -> None:
    root = result_root(worker_dir) / "properties"
    if list(root.glob("*/result_v1.json")):
        raise RuntimeError("fresh worker unexpectedly contains completed results")
    if list(root.glob("*/queries/query_*_result_v1.json")):
        raise RuntimeError("fresh worker unexpectedly contains query records")


def assigned_properties(plan: dict, worker_id: int) -> set[str]:
    return {row["property_id"]
            for chunk in plan["workers"][worker_id]["chunks"]
            for row in chunk["properties"]}
