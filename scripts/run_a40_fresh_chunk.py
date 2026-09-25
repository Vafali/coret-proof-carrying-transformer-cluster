#!/usr/bin/env python3
"""Run one immutable A40 chunk in an isolated, resumable worker tree."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from a40_fresh_common import (assigned_properties, initialize_fresh_worker,
    load_chunk, load_plan, result_root)
from cluster_common import artifact_root, canonical
from merge_worker_results import validate_property
from portable_runner import configure


ISOLATION_VARIABLES = (
    "TMPDIR", "XDG_CACHE_HOME", "CUDA_CACHE_PATH", "TORCH_EXTENSIONS_DIR",
    "PYTHONPYCACHEPREFIX",
)


def verify_isolation(worker_dir: Path) -> None:
    if Path.cwd().resolve() != (worker_dir / "runtime").resolve():
        raise RuntimeError("worker must execute from its isolated runtime directory")
    for name in ISOLATION_VARIABLES:
        value = os.environ.get(name)
        if not value or not Path(value).resolve().is_relative_to(worker_dir.resolve()):
            raise RuntimeError(f"worker isolation variable differs: {name}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("exactly one Slurm-assigned GPU must be visible")


def run_property(worker_dir: Path, artifact: Path, property_id: str) -> None:
    runner, _, _ = configure(worker_dir, artifact)

    # The homogeneous A40 benchmark must execute all 127 properties freshly.
    # Disabling the old smoke-record reuse is orchestration-only: the verifier,
    # search, certificates, and all operator semantics remain untouched.
    runner._reuse_record = lambda manifest, pid: None

    def invoke(pid: str, rho: float, ordinal: int) -> None:
        command = [sys.executable,
                   str(Path(__file__).with_name("portable_runner.py")),
                   "query", "--worker-dir", str(worker_dir),
                   "--artifact-root", str(artifact),
                   "--property-id", pid, "--rho-hex", float(rho).hex(),
                   "--query-ordinal", str(ordinal)]
        subprocess.run(command, check=True, cwd=worker_dir / "runtime",
                       env={**os.environ, "PYTHONPATH": (
                           f"{Path(__file__).resolve().parents[1] / 'scripts'}:"
                           f"{runner.WORKTREE / 'research_hab'}")})

    runner._invoke_query = invoke
    runner.run_property(property_id, True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, choices=(0, 1), required=True)
    parser.add_argument("--chunk-index", type=int, choices=range(4), required=True)
    parser.add_argument("--worker-dir", required=True)
    parser.add_argument("--artifact-root")
    args = parser.parse_args()
    artifact = artifact_root(args.artifact_root)
    worker_dir = Path(args.worker_dir).resolve()
    initialize_fresh_worker(artifact, worker_dir, args.worker_id)
    verify_isolation(worker_dir)
    plan = load_plan()
    chunk = load_chunk(args.worker_id, args.chunk_index)
    allowed = assigned_properties(plan, args.worker_id)
    existing = {path.parent.name for path in
                (result_root(worker_dir) / "properties").glob("*/result_v1.json")}
    if not existing.issubset(allowed):
        raise RuntimeError("worker contains a result owned by another worker")
    completed = []
    for prop in chunk["properties"]:
        property_id = prop["property_id"]
        result = (result_root(worker_dir) / "properties" / property_id
                  / "result_v1.json")
        if result.exists():
            validate_property(result_root(worker_dir), property_id)
        else:
            run_property(worker_dir, artifact, property_id)
            validate_property(result_root(worker_dir), property_id)
        completed.append(property_id)
    record = {
        "schema": "CORET_A40_FRESH_CHUNK_COMPLETION_V1",
        "plan_sha256": plan["canonical_manifest_sha256"],
        "chunk_manifest_sha256": chunk["canonical_manifest_sha256"],
        "worker_id": args.worker_id,
        "chunk_index": args.chunk_index,
        "completed_properties": completed,
        "completed_property_count": len(completed),
    }
    record["record_sha256"] = canonical(record)
    target = (worker_dir / "chunk_records"
              / f"chunk_{args.chunk_index}_complete_v1.json")
    if target.exists():
        from cluster_common import verified_json
        if verified_json(target) != record:
            raise RuntimeError("chunk completion record differs")
    else:
        target.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": "A40_FRESH_CHUNK_COMPLETE",
                      "worker_id": args.worker_id,
                      "chunk_index": args.chunk_index,
                      "property_count": len(completed)}, sort_keys=True))


if __name__ == "__main__": main()
