#!/usr/bin/env python3
"""Build immutable full depth-6 production and deterministic A40 chunks."""
from __future__ import annotations

import json

import depth6_full_common as common


def _freeze(value: dict) -> dict:
    result = dict(value)
    result["canonical_manifest_sha256"] = common.canonical(result)
    return result


def build_plan(production: dict) -> dict:
    rows = []
    for row in production["properties"]:
        item = {key: row[key] for key in (
            "benchmark_ordinal", "property_id", "sentence_ordinal",
            "token_position", "sequence_length")}
        item["objective_cost"] = int(item["sequence_length"]) ** 2
        rows.append(item)

    workers = [[], []]
    worker_costs = [0, 0]
    for row in sorted(rows, key=lambda item: (
            -item["objective_cost"], item["benchmark_ordinal"])):
        worker = min(range(2), key=lambda index: (worker_costs[index], index))
        workers[worker].append(row)
        worker_costs[worker] += row["objective_cost"]

    central_seconds = common.ESTIMATED_TWO_A40_WALL_CENTRAL_HOURS * 3600.0
    seconds_per_cost = central_seconds / max(worker_costs)
    output_workers = []
    for worker_id, owned in enumerate(workers):
        chunks = [[] for _ in range(common.CHUNKS_PER_WORKER)]
        costs = [0] * common.CHUNKS_PER_WORKER
        for row in sorted(owned, key=lambda item: (
                -item["objective_cost"], item["benchmark_ordinal"])):
            index = min(range(common.CHUNKS_PER_WORKER),
                        key=lambda candidate: (costs[candidate], candidate))
            chunks[index].append(row)
            costs[index] += row["objective_cost"]
        chunk_rows = []
        for index, values in enumerate(chunks):
            values.sort(key=lambda item: item["benchmark_ordinal"])
            chunk_rows.append({
                "chunk_index": index,
                "property_count": len(values),
                "objective_cost": costs[index],
                "predicted_seconds_central": costs[index] * seconds_per_cost,
                "predicted_hours_lower": (
                    costs[index] / worker_costs[worker_id]
                    * common.ESTIMATED_TWO_A40_WALL_LOWER_HOURS),
                "predicted_hours_upper": (
                    costs[index] / worker_costs[worker_id]
                    * common.ESTIMATED_TWO_A40_WALL_UPPER_HOURS),
                "properties": values,
            })
        output_workers.append({
            "worker_id": worker_id,
            "physical_gpu": f"gpu{worker_id}",
            "property_count": len(owned),
            "objective_cost": worker_costs[worker_id],
            "predicted_seconds_central": worker_costs[worker_id] * seconds_per_cost,
            "chunks": chunk_rows,
        })

    plan = {
        "schema": "CORET_DEPTH6_FULL_A40_PLAN_V1",
        "status": "DEPTH6_FULL_A40_PRODUCTION_PREPARED",
        "scientific_manifest_sha256": common.EXPECTED_FULL_MANIFEST_SHA,
        "production_manifest_sha256": production["canonical_manifest_sha256"],
        "scientific_source_commit": common.EXPECTED_SOURCE_COMMIT,
        "property_count": common.EXPECTED_PROPERTY_COUNT,
        "hardware": "2x_NVIDIA_A40_48GB",
        "partition": "gpu-a40",
        "node": "afrodita",
        "worker_count": 2,
        "maximum_concurrent_jobs": 2,
        "chunks_per_worker": common.CHUNKS_PER_WORKER,
        "fresh_root_basename": common.FRESH_ROOT_BASENAME,
        "cost_model": {
            "assignment": "deterministic_LPT_by_sequence_length_squared",
            "tie_break": "benchmark_ordinal_then_lower_worker_or_chunk_index",
            "pre_packed_two_A40_wall_hours":
                common.PRE_PACKED_TWO_A40_WALL_HOURS,
            "packed_speedup_interval": [
                common.PACKED_SPEEDUP_LOWER, common.PACKED_SPEEDUP_UPPER],
            "estimated_two_A40_wall_hours_lower":
                common.ESTIMATED_TWO_A40_WALL_LOWER_HOURS,
            "estimated_two_A40_wall_hours_central":
                common.ESTIMATED_TWO_A40_WALL_CENTRAL_HOURS,
            "estimated_two_A40_wall_hours_upper":
                common.ESTIMATED_TWO_A40_WALL_UPPER_HOURS,
            "estimated_total_GPU_hours_central":
                2.0 * common.ESTIMATED_TWO_A40_WALL_CENTRAL_HOURS,
            "measurement_provenance": (
                "accepted depth6 sanity projection and bounded packed-metadata "
                "audit; no full-production timing performed"),
        },
        "workers": output_workers,
        "preparation_scientific_queries": 0,
        "preparation_bound_entrypoint_calls": 0,
    }
    ids = [row["property_id"] for worker in output_workers
           for item in worker["chunks"] for row in item["properties"]]
    expected = [row["property_id"] for row in production["properties"]]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise RuntimeError("full depth-6 plan is not complete and disjoint")
    if any(item["predicted_hours_upper"] >= 12.0 for worker in output_workers
           for item in worker["chunks"]):
        raise RuntimeError("full depth-6 chunk exceeds 12-hour estimate")
    return _freeze(plan)


def main() -> None:
    common.FROZEN_ROOT.mkdir(parents=True, exist_ok=True)
    common.CHUNK_ROOT.mkdir(parents=True, exist_ok=True)
    production = _freeze(common.expected_production_manifest())
    common.PRODUCTION_MANIFEST_PATH.write_text(
        json.dumps(production, sort_keys=True, indent=2) + "\n")
    plan = build_plan(production)
    common.PLAN_PATH.write_text(json.dumps(plan, sort_keys=True, indent=2) + "\n")
    for worker in plan["workers"]:
        for item in worker["chunks"]:
            chunk = _freeze({
                "schema": "CORET_DEPTH6_FULL_A40_CHUNK_V1",
                "parent_plan_sha256": plan["canonical_manifest_sha256"],
                "production_manifest_sha256": production[
                    "canonical_manifest_sha256"],
                "worker_id": worker["worker_id"],
                "physical_gpu": worker["physical_gpu"],
                **item,
            })
            common.chunk_path(worker["worker_id"], item["chunk_index"]).write_text(
                json.dumps(chunk, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "status": "DEPTH6_FULL_PLAN_FROZEN",
        "production_manifest_sha256": production["canonical_manifest_sha256"],
        "plan_sha256": plan["canonical_manifest_sha256"],
        "property_count": plan["property_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
