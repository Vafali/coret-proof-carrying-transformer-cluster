#!/usr/bin/env python3
"""Create a zero-result two-worker A40 production root."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from a40_fresh_common import (initialize_fresh_worker, load_plan,
    verify_no_imported_results)
from cluster_common import artifact_root, canonical


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-root", required=True)
    parser.add_argument("--artifact-root")
    args = parser.parse_args()
    root = Path(args.fresh_root).resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    artifact = artifact_root(args.artifact_root)
    plan = load_plan()
    workers = []
    for worker_id in range(2):
        worker = root / f"worker_{worker_id}"
        initialize_fresh_worker(artifact, worker, worker_id)
        verify_no_imported_results(worker)
        workers.append({"worker_id": worker_id, "path": str(worker),
                        "completed_properties": 0,
                        "persisted_queries": 0})
    value = {
        "schema": "CORET_A40_FRESH_ROOT_V1",
        "status": "A40_FRESH_ROOT_READY",
        "plan_sha256": plan["canonical_manifest_sha256"],
        "root": str(root),
        "completed_properties": 0,
        "persisted_queries": 0,
        "imported_A4000_completed_properties": 0,
        "imported_A4000_query_records": 0,
        "workers": workers,
    }
    value["record_sha256"] = canonical(value)
    (root / "fresh_root_manifest_v1.json").write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n")
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__": main()
