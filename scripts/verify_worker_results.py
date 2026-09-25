#!/usr/bin/env python3
"""Fail-closed validation of a private worker result tree."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cluster_common import (BASELINE_REL, artifact_root, baseline_root,
    load_shard, sha256, verified_json, verify_baseline_immutable)


def verify_worker(worker_dir: Path, root: Path, shard: dict, worker_id: int,
                  require_complete=True) -> dict:
    target = worker_dir / BASELINE_REL
    verify_baseline_immutable(baseline_root(root), target)
    assignment = next(row for row in shard["workers"] if row["worker_id"] == worker_id)
    accepted = 0
    for prop in assignment["properties"]:
        path = target / "properties" / prop["property_id"] / "result_v1.json"
        if not path.exists():
            if require_complete: raise RuntimeError(f"missing worker result: {path}")
            continue
        result = verified_json(path)
        if (result.get("terminal_status") != "COMPLETE"
                or result.get("independent_checker_accepted") is not True
                or result.get("generic_fallback_count") != 0
                or result.get("all_support_claims_validated") is not True
                or result.get("all_provenance_consistent") is not True):
            raise RuntimeError(f"unaccepted worker result: {path}")
        query_dir = path.parent / "queries"
        queries = sorted(query_dir.glob("query_*_result_v1.json"))
        events_path = path.parent / "query_events_v1.jsonl"
        events = [json.loads(line) for line in events_path.read_text().splitlines() if line]
        for query in queries: verified_json(query)
        if {verified_json(q)["record_sha256"] for q in queries} != {
                event["query_result_record_sha256"] for event in events}:
            raise RuntimeError(f"query journal mismatch: {path.parent}")
        accepted += 1
    return {"worker_id": worker_id, "assigned": len(assignment["properties"]),
            "accepted_complete": accepted, "worker_tree_sha256": sha256(path)
            if accepted else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-manifest", required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--worker-dir", required=True)
    parser.add_argument("--artifact-root")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    report = verify_worker(Path(args.worker_dir), artifact_root(args.artifact_root),
                           load_shard(Path(args.shard_manifest)), args.worker_id,
                           not args.allow_incomplete)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__": main()
