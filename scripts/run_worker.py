#!/usr/bin/env python3
"""Run one immutable shard sequentially in one GPU-bound process lineage."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from cluster_common import artifact_root, initialize_worker, load_shard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-manifest", required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--worker-dir", required=True)
    parser.add_argument("--artifact-root")
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("exactly one physical GPU must be visible to a worker")
    shard = load_shard(Path(args.shard_manifest))
    rows = [row for row in shard["workers"] if row["worker_id"] == args.worker_id]
    if len(rows) != 1: raise RuntimeError("worker ID absent or duplicated")
    root = artifact_root(args.artifact_root)
    worker_dir = Path(args.worker_dir).resolve()
    initialize_worker(root, worker_dir)
    for prop in rows[0]["properties"]:
        command = [sys.executable, str(Path(__file__).with_name("portable_runner.py")),
                   "property", "--worker-dir", str(worker_dir),
                   "--artifact-root", str(root),
                   "--property-id", prop["property_id"]]
        subprocess.run(command, check=True, env={**os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "research_hab")})
    print(json.dumps({"status": "WORKER_SHARD_COMPLETE",
                      "worker_id": args.worker_id,
                      "property_count": len(rows[0]["properties"])}, sort_keys=True))


if __name__ == "__main__": main()
