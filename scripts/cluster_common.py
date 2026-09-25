#!/usr/bin/env python3
"""Portable, fail-closed orchestration around the immutable verifier."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESEARCH = REPO / "research_hab"
DEEPT_COMMIT = "16ffe4075f1f8a7c87fa2a187d8c46cfd51e07bf"
SCIENTIFIC_MANIFEST_SHA = "7e4b2fea94424e554f07272aa8a246da7cda1212be83af57bbcdae7c870ed9dd"
PRODUCTION_MANIFEST_SHA = "cd1375408818c8fb93a2f227d31229990d62034dc4997467fbc013cc1eb94ab2"
DEEPT_CACHE_SHA = "67e80d74cf83e5f810726405a6f6f0cf9928f4dda84d7f87decf59f354c8181d"
BASELINE_REL = Path("research_hab/results/coret_optimized_historical_127_v1_20260924")
HISTORICAL_REL = Path("research_hab/results/coret_deept_paper_benchmark_v5_20260917")
SMOKE_REL = Path("research_hab/results/coret_optimized_three_property_smoke_v1_20260923")
PRODUCTION_MANIFEST_NAME = "coret_optimized_historical_127_manifest_v2.json"
BASELINE_STATE = REPO / "frozen/cluster_baseline_state.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def verified_json(path: Path, key="record_sha256") -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(value)
    claimed = payload.pop(key)
    if canonical(payload) != claimed:
        raise RuntimeError(f"canonical hash mismatch: {path}")
    return value


def artifact_root(explicit: str | None = None) -> Path:
    value = explicit or os.environ.get("CORET_ARTIFACT_ROOT")
    if not value:
        value = str(REPO / "runtime_inputs")
    root = Path(value).resolve()
    manifest = root / "artifact_manifest.json"
    if not manifest.is_file():
        raise RuntimeError(f"artifact manifest unavailable: {manifest}")
    return root


def verify_artifact_manifest(root: Path) -> dict:
    manifest = verified_json(root / "artifact_manifest.json",
                             "canonical_manifest_sha256")
    for row in manifest["artifacts"]:
        path = root / row["relative_path"]
        if not path.is_file() or path.stat().st_size != row["byte_size"]:
            raise RuntimeError(f"artifact missing/size mismatch: {path}")
        if sha256(path) != row["sha256"]:
            raise RuntimeError(f"artifact SHA mismatch: {path}")
    return manifest


def baseline_root(root: Path) -> Path:
    return root / "payload" / BASELINE_REL


def load_production_manifest(root: Path) -> dict:
    path = baseline_root(root) / PRODUCTION_MANIFEST_NAME
    manifest = verified_json(path, "canonical_manifest_sha256")
    if manifest["canonical_manifest_sha256"] != PRODUCTION_MANIFEST_SHA:
        raise RuntimeError("production manifest identity differs")
    if manifest["historical_scientific_manifest"]["canonical_sha256"] != SCIENTIFIC_MANIFEST_SHA:
        raise RuntimeError("scientific manifest identity differs")
    if manifest["cached_native_DeepT"]["positions_file_sha256"] != DEEPT_CACHE_SHA:
        raise RuntimeError("DeepT cache identity differs")
    return portable_paths(manifest, root)


def portable_paths(value, root: Path):
    """Rewrite only frozen absolute artifact locations in an in-memory copy."""
    if isinstance(value, dict):
        return {key: portable_paths(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_paths(item, root) for item in value]
    if isinstance(value, str) and "/research_hab/results/" in value:
        suffix = value.split("/research_hab/results/", 1)[1]
        return str(root / "payload/research_hab/results" / suffix)
    return value


def historical_manifest_path(root: Path) -> Path:
    return root / "payload" / HISTORICAL_REL / "coret_deept_paper_benchmark_execution_manifest_v5.json"


def initialize_worker(root: Path, worker_dir: Path) -> Path:
    """Copy the immutable baseline once; never merge worker writes in place."""
    source = baseline_root(root)
    destination = worker_dir / BASELINE_REL
    if destination.exists():
        verify_baseline_immutable(source, destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    verify_baseline_immutable(source, destination)
    return destination


def verify_baseline_immutable(source: Path, destination: Path) -> None:
    """Every pre-existing baseline file must remain present and byte-identical."""
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        other = destination / relative
        if not other.is_file() or sha256(path) != sha256(other):
            raise RuntimeError(f"baseline mutation/missing file: {relative}")


def patch_runner(runner, root: Path, result_root: Path, manifest: dict) -> None:
    """Redirect artifacts outside frozen code without changing scientific logic."""
    runner.OUT = result_root
    runner.SUMMARY = result_root / "coret_optimized_historical_127_summary_v2.json"
    runner.HISTORICAL_MANIFEST = historical_manifest_path(root)
    runner.DEEPT_POSITIONS = root / "payload" / HISTORICAL_REL / "deept_positions_v5.jsonl"
    runner.DEEPT_RESULT = root / "payload" / HISTORICAL_REL / "deept_result_v5.json"
    runner.validate_manifest = lambda: manifest


def load_shard(path: Path) -> dict:
    value = verified_json(path, "canonical_manifest_sha256")
    if value["scientific_manifest_sha256"] != SCIENTIFIC_MANIFEST_SHA:
        raise RuntimeError("shard scientific identity differs")
    if value["production_manifest_sha256"] != PRODUCTION_MANIFEST_SHA:
        raise RuntimeError("shard production identity differs")
    return value


def property_ids(manifest: dict) -> list[str]:
    return [row["property_id"] for row in manifest["properties"]]


def completed_property_ids(root: Path) -> set[str]:
    base = baseline_root(root) / "properties"
    return {path.parent.name for path in base.glob("*/result_v1.json")}


def partial_property_ids(root: Path) -> set[str]:
    base = baseline_root(root) / "properties"
    return {path.parent.name for path in base.glob("*/queries")
            if any(path.glob("query_*_result_v1.json"))
            if not (path.parent / "result_v1.json").exists()}


def load_baseline_state() -> dict:
    return verified_json(BASELINE_STATE, "canonical_manifest_sha256")
