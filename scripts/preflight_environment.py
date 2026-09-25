#!/usr/bin/env python3
"""Environment/source/artifact preflight. It never imports a verifier entrypoint."""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

import torch

from cluster_common import (DEEPT_CACHE_SHA, DEEPT_COMMIT, REPO,
    SCIENTIFIC_MANIFEST_SHA, artifact_root, load_production_manifest, sha256,
    verify_artifact_manifest)


def git_blob_sha(repo: Path, revision: str, path: str) -> str:
    value = subprocess.check_output(["git", "-C", str(repo), "show",
                                     f"{revision}:{path}"])
    import hashlib
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root")
    parser.add_argument("--allow-no-cuda", action="store_true")
    args = parser.parse_args()
    root = artifact_root(args.artifact_root)
    artifact = verify_artifact_manifest(root)
    production = load_production_manifest(root)
    source = json.loads((REPO / "frozen/source_tree_manifest.json").read_text())
    for row in source["files"]:
        path = REPO / row["path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise RuntimeError(f"frozen source differs: {path}")
    if any("fused_av" in (REPO / row["path"]).read_text(errors="ignore")
           for row in source["files"] if row["path"].startswith("research_hab/")):
        raise RuntimeError("fused A.V backend is reachable from scientific source")
    deept = REPO / "research_hab/public_benchmarks/DeepT"
    revision = subprocess.check_output(["git", "-C", str(deept), "rev-parse",
                                        DEEPT_COMMIT], text=True).strip()
    if revision != DEEPT_COMMIT: raise RuntimeError("DeepT revision differs")
    blobs = production["model"]
    for name in ("checkpoint", "config", "vocab"):
        row = blobs[name]
        if git_blob_sha(deept, DEEPT_COMMIT, row["git_path"]) != row["sha256"]:
            raise RuntimeError(f"DeepT {name} blob differs")
    for package in ("numpy", "scipy", "pytorch_pretrained_bert", "opt_einsum", "termcolor"):
        importlib.import_module(package)
    if not torch.cuda.is_available() and not args.allow_no_cuda:
        raise RuntimeError("CUDA is unavailable")
    gpu = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(index)
            gpu.append({"index": index, "name": prop.name,
                        "compute_capability": [prop.major, prop.minor],
                        "total_memory_bytes": prop.total_memory})
    driver = None
    try:
        driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version",
                                          "--format=csv,noheader"], text=True).splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    print(json.dumps({"status": "CLUSTER_ENVIRONMENT_PREFLIGHT_PASS",
        "python": sys.version, "pytorch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "gpu": gpu, "driver": driver,
        "scientific_manifest_sha256": SCIENTIFIC_MANIFEST_SHA,
        "DeepT_cache_sha256": DEEPT_CACHE_SHA,
        "production_manifest_sha256": production["canonical_manifest_sha256"],
        "artifact_manifest_sha256": artifact["canonical_manifest_sha256"],
        "fused_backend_reachable": False,
        "scientific_queries": 0, "bound_entrypoint_calls": 0}, sort_keys=True))


if __name__ == "__main__": main()
