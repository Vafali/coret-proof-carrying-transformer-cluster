#!/usr/bin/env python3
"""Build deterministic source/artifact manifests without scientific calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cluster_common import REPO, canonical, sha256


def role(path: Path) -> str:
    value = str(path)
    if "optimized_historical_127" in value: return "resumable_benchmark_state"
    if "optimized_three_property_smoke" in value: return "frozen_reuse_state"
    if "deept_paper_benchmark" in value: return "cached_deept_reference"
    if "deept_model_blobs" in value: return "pinned_model_artifact"
    if path.suffix == ".jsonl": return "append_only_journal"
    return "immutable_runtime_input"


def artifact_manifest(root: Path) -> dict:
    rows = [{"relative_path": str(path.relative_to(root)),
             "byte_size": path.stat().st_size, "sha256": sha256(path),
             "role": role(path.relative_to(root))}
            for path in sorted((root / "payload").rglob("*")) if path.is_file()]
    value = {"schema": "CORET_CLUSTER_ARTIFACT_MANIFEST_V1",
             "artifact_count": len(rows), "artifacts": rows,
             "scientific_queries": 0, "bound_entrypoint_calls": 0}
    value["canonical_manifest_sha256"] = canonical(value)
    return value


def source_manifest() -> dict:
    excluded = {Path("frozen/source_tree_manifest.json")}
    excluded_parts = {".git", "__pycache__", ".pytest_cache", ".mypy_cache",
                      "runtime_inputs", "worker_runs", "calibration_runs",
                      "merged_results"}
    rows = []
    for path in sorted(REPO.rglob("*")):
        if not path.is_file() or any(part in excluded_parts for part in path.parts):
            continue
        relative = path.relative_to(REPO)
        if relative in excluded: continue
        rows.append({"path": str(relative), "byte_size": path.stat().st_size,
                     "sha256": sha256(path)})
    value = {"schema": "CORET_CLUSTER_SOURCE_TREE_MANIFEST_V1",
             "files": rows, "scientific_queries": 0,
             "bound_entrypoint_calls": 0}
    value["canonical_manifest_sha256"] = canonical(value)
    return value


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("source", "artifact"))
    parser.add_argument("--artifact-root")
    args = parser.parse_args()
    if args.mode == "source":
        path = REPO / "frozen/source_tree_manifest.json"
        write(path, source_manifest())
    else:
        root = Path(args.artifact_root).resolve()
        path = root / "artifact_manifest.json"
        write(path, artifact_manifest(root))
    print(path)


if __name__ == "__main__": main()
