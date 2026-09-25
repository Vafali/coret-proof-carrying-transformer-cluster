#!/usr/bin/env python3
"""Deterministic weighted sharding using only frozen sequence length."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cluster_common import (PRODUCTION_MANIFEST_SHA, SCIENTIFIC_MANIFEST_SHA,
    artifact_root, canonical, completed_property_ids, load_production_manifest,
    partial_property_ids)


def make_shards(manifest: dict, complete: set[str], partial: set[str],
                weights: list[float]) -> dict:
    if not weights or any(value <= 0 for value in weights):
        raise ValueError("worker weights must be positive")
    remaining = [row for row in manifest["properties"]
                 if row["property_id"] not in complete]
    assignments = [[] for _ in weights]
    load = [0.0 for _ in weights]
    raw = [0 for _ in weights]
    # If a partial trajectory exists, one forced owner continues it. A clean
    # property boundary legitimately has no partial trajectory.
    partial_rows = [row for row in remaining if row["property_id"] in partial]
    if len(partial_rows) > 1 or len(partial_rows) != len(partial):
        raise RuntimeError("partial-property set is not representable")
    partial_property = (partial_rows[0]["property_id"]
                        if partial_rows else None)
    tasks = sorted(remaining, key=lambda row: (
        -(int(row["sequence_length"]) ** 2), int(row["benchmark_ordinal"])))
    for row in tasks:
        cost = int(row["sequence_length"]) ** 2
        if row["property_id"] == partial_property:
            worker = 0
        else:
            worker = min(range(len(weights)),
                         key=lambda i: (load[i] / weights[i], i))
        item = {"benchmark_ordinal": row["benchmark_ordinal"],
                "property_id": row["property_id"],
                "sentence_ordinal": row["sentence_ordinal"],
                "token_position": row["token_position"],
                "sequence_length": row["sequence_length"],
                "objective_cost": cost,
                "continues_partial_baseline": row["property_id"] == partial_property}
        assignments[worker].append(item)
        raw[worker] += cost
        load[worker] += cost
    assigned = [row["property_id"] for rows in assignments for row in rows]
    expected = [row["property_id"] for row in remaining]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(expected):
        raise RuntimeError("shards are not complete and disjoint")
    expected_partial_count = 1 if partial_property is not None else 0
    if sum(pid == partial_property for pid in assigned) != expected_partial_count:
        raise RuntimeError("partial property assignment count differs")
    workers = []
    for worker, rows in enumerate(assignments):
        rows.sort(key=lambda row: int(row["benchmark_ordinal"]))
        workers.append({"worker_id": worker, "weight": weights[worker],
                        "objective_cost": raw[worker],
                        "properties": rows})
    return {"schema": "CORET_CLUSTER_PROPERTY_SHARDS_V1",
            "scientific_manifest_sha256": SCIENTIFIC_MANIFEST_SHA,
            "production_manifest_sha256": PRODUCTION_MANIFEST_SHA,
            "cost_model": "sequence_length_squared_only",
            "completed_baseline_count": len(complete),
            "remaining_property_count": len(remaining),
            "partial_property": partial_property,
            "worker_count": len(weights), "workers": workers}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--worker-weight", type=float, action="append")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = artifact_root(args.artifact_root)
    manifest = load_production_manifest(root)
    weights = args.worker_weight or ([1.0] * args.workers if args.workers else None)
    if weights is None or (args.workers is not None and len(weights) != args.workers):
        parser.error("supply --workers N or exactly N --worker-weight values")
    value = make_shards(manifest, completed_property_ids(root),
                        partial_property_ids(root), weights)
    value["canonical_manifest_sha256"] = canonical(value)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "SHARDS_FROZEN", "path": str(path),
                      "sha256": value["canonical_manifest_sha256"]}, sort_keys=True))


if __name__ == "__main__": main()
