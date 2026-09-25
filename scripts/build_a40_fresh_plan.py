#!/usr/bin/env python3
"""Freeze deterministic A40-only ownership and sub-12-hour chunks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cluster_common import (PRODUCTION_MANIFEST_SHA, REPO,
    SCIENTIFIC_MANIFEST_SHA, artifact_root, canonical,
    load_production_manifest)


SHORT_SECONDS = 84.7189425136894
LONG_SECONDS = 192.84101035259664
QUERY_COUNT_UPPER_MODEL = 13
CHUNKS_PER_WORKER = 4


def build(manifest: dict) -> dict:
    seconds_per_n2 = max(SHORT_SECONDS / (20 ** 2),
                         LONG_SECONDS / (27 ** 2))
    rows = []
    for row in manifest["properties"]:
        item = {key: row[key] for key in (
            "benchmark_ordinal", "property_id", "sentence_ordinal",
            "token_position", "sequence_length")}
        item["objective_cost"] = int(row["sequence_length"]) ** 2
        item["predicted_seconds"] = (
            item["objective_cost"] * seconds_per_n2
            * QUERY_COUNT_UPPER_MODEL)
        rows.append(item)
    workers = [[], []]
    worker_costs = [0, 0]
    for row in sorted(rows, key=lambda item: (
            -item["objective_cost"], item["benchmark_ordinal"])):
        worker = min(range(2), key=lambda i: (worker_costs[i], i))
        workers[worker].append(row)
        worker_costs[worker] += row["objective_cost"]
    output_workers = []
    for worker_id, owned in enumerate(workers):
        chunks = [[] for _ in range(CHUNKS_PER_WORKER)]
        chunk_costs = [0] * CHUNKS_PER_WORKER
        for row in sorted(owned, key=lambda item: (
                -item["objective_cost"], item["benchmark_ordinal"])):
            index = min(range(CHUNKS_PER_WORKER),
                        key=lambda i: (chunk_costs[i], i))
            chunks[index].append(row)
            chunk_costs[index] += row["objective_cost"]
        chunk_rows = []
        for index, values in enumerate(chunks):
            values.sort(key=lambda item: item["benchmark_ordinal"])
            chunk_rows.append({
                "chunk_index": index,
                "objective_cost": chunk_costs[index],
                "predicted_seconds": (chunk_costs[index] * seconds_per_n2
                                      * QUERY_COUNT_UPPER_MODEL),
                "properties": values,
            })
        output_workers.append({
            "worker_id": worker_id,
            "physical_gpu": f"gpu{worker_id}",
            "property_count": len(owned),
            "objective_cost": worker_costs[worker_id],
            "predicted_seconds": (worker_costs[worker_id] * seconds_per_n2
                                  * QUERY_COUNT_UPPER_MODEL),
            "chunks": chunk_rows,
        })
    value = {
        "schema": "CORET_A40_FRESH_127_PLAN_V1",
        "status": "A40_FRESH_127_PRODUCTION_READY",
        "scientific_manifest_sha256": SCIENTIFIC_MANIFEST_SHA,
        "production_manifest_sha256": PRODUCTION_MANIFEST_SHA,
        "property_count": 127,
        "imported_A4000_completed_properties": 0,
        "imported_A4000_query_records": 0,
        "hardware": "2x_NVIDIA_A40_48GB",
        "partition": "gpu-a40",
        "calibration": {
            "short_sequence_length": 20,
            "short_wall_seconds": SHORT_SECONDS,
            "short_classification": "SOUND_OUTWARD_HARDWARE_VARIATION",
            "long_sequence_length": 27,
            "long_wall_seconds": LONG_SECONDS,
            "long_classification": "CROSS_GPU_SCIENTIFIC_MISMATCH",
            "checker_invariants": True,
            "fresh_homogeneous_A40_execution_authorized": True,
            "A4000_A40_mixing_authorized": False,
        },
        "cost_model": {
            "only_property_input": "sequence_length_squared",
            "seconds_per_sequence_length_squared_per_query": seconds_per_n2,
            "predicted_queries_per_property": QUERY_COUNT_UPPER_MODEL,
            "chunk_target": "approximately_6_to_8_hours",
            "slurm_hard_limit_hours": 12,
        },
        "worker_count": 2,
        "workers": output_workers,
    }
    all_ids = [row["property_id"] for worker in output_workers
               for chunk in worker["chunks"] for row in chunk["properties"]]
    if len(all_ids) != len(set(all_ids)) or len(all_ids) != 127:
        raise RuntimeError("fresh A40 ownership is not complete and disjoint")
    value["canonical_manifest_sha256"] = canonical(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root")
    args = parser.parse_args()
    manifest = load_production_manifest(artifact_root(args.artifact_root))
    plan = build(manifest)
    plan_path = REPO / "frozen/a40_fresh_127_plan.json"
    chunk_root = REPO / "frozen/a40_chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, sort_keys=True, indent=2) + "\n")
    for worker in plan["workers"]:
        for chunk in worker["chunks"]:
            value = {
                "schema": "CORET_A40_FRESH_CHUNK_V1",
                "parent_plan_sha256": plan["canonical_manifest_sha256"],
                "worker_id": worker["worker_id"],
                "physical_gpu": worker["physical_gpu"],
                "chunk_index": chunk["chunk_index"],
                "objective_cost": chunk["objective_cost"],
                "predicted_seconds": chunk["predicted_seconds"],
                "properties": chunk["properties"],
            }
            value["canonical_manifest_sha256"] = canonical(value)
            path = chunk_root / (
                f"worker_{worker['worker_id']}_chunk_{chunk['chunk_index']}.json")
            path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": plan["status"],
                      "plan_sha256": plan["canonical_manifest_sha256"]},
                     sort_keys=True))


if __name__ == "__main__": main()
