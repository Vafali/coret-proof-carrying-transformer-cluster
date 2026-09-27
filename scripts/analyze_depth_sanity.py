#!/usr/bin/env python3
"""Read-only aggregation of completed 6/12-layer A40 sanity results."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import depth_sanity_common as common


def _verified(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.get("record_sha256")
    body = dict(payload)
    body.pop("record_sha256", None)
    if expected != common.canonical(body):
        raise RuntimeError(f"record hash differs: {path}")
    return payload


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def analyze(depth: int, fresh_root: Path, expected_completed: int) -> dict:
    manifest = common.production_manifest(depth)
    expected = {row["property_id"]: row for row in manifest["properties"]}
    results = []
    query_rows = []
    seen = set()
    for worker_id in (0, 1):
        directory = common.worker_dir(fresh_root, depth, worker_id)
        root = common.worker_result_root(directory)
        for path in sorted((root / "properties").glob("*/result_v1.json")):
            row = _verified(path)
            property_id = path.parent.name
            if property_id in seen or property_id not in expected:
                raise RuntimeError("duplicate or foreign depth property")
            if int(expected[property_id]["worker_id"]) != worker_id:
                raise RuntimeError("depth property stored by wrong worker")
            if (row.get("property_id") != property_id
                    or row.get("canonical_manifest_sha256") != manifest[
                        "canonical_manifest_sha256"]):
                raise RuntimeError("property frozen identity differs")
            seen.add(property_id)
            results.append(row)
            for query in sorted((path.parent / "queries").glob(
                    "query_*_result_v1.json")):
                query_rows.append(_verified(query))
    if len(results) != expected_completed:
        raise RuntimeError(
            f"completed count differs: {len(results)} != {expected_completed}")
    integrity = {
        "checker_failures": sum(int(row.get("checker_failure_count", 0))
                                for row in results),
        "provenance_failures": sum(
            row.get("all_provenance_consistent") is not True for row in results),
        "support_failures": sum(
            row.get("all_support_claims_validated") is not True for row in results),
        "generic_fallbacks": sum(int(row.get("generic_fallback_count", 0))
                                 for row in results),
        "unexpected_failures": sum(row.get("terminal_status") != "COMPLETE"
                                   for row in results),
    }
    typed = sum(row.get("terminal_status") == "UNCERTIFIED_DOMAIN_FAILURE"
                for row in query_rows)
    if any(integrity.values()):
        classification = "DEPTH_SANITY_FAIL"
    elif len(results) == 10:
        classification = "DEPTH_SANITY_PASS"
    else:
        classification = "DEPTH_SANITY_INCOMPLETE"
    ratios = [float(row["radius_ratio_to_cached_DeepT"]) for row in results]
    runtimes = [float(row["total_wall_time_seconds"]) for row in results]
    per_draw = {int(expected[row["property_id"]]["accepted_draw_ordinal"]):
                float(row["total_wall_time_seconds"]) for row in results}
    full, _ = common.manifests(depth)
    projected = [per_draw.get(int(row["accepted_draw_ordinal"]))
                 for row in full["properties"]]
    projected_makespan = None
    if all(value is not None for value in projected):
        loads = [0.0, 0.0]
        for value in sorted(projected, reverse=True):
            index = 0 if loads[0] <= loads[1] else 1
            loads[index] += float(value)
        projected_makespan = max(loads)
    return {
        "schema": "CORET_DEPTH_SANITY_A40_ANALYSIS_V1",
        "classification": classification,
        "depth": depth,
        "completed_count": len(results),
        "expected_completed": expected_completed,
        "manifest_sha256": manifest["canonical_manifest_sha256"],
        **integrity,
        "typed_domain_failure_count": typed,
        "ratio_summary": None if not ratios else {
            "min": min(ratios), "median": statistics.median(ratios),
            "mean": statistics.fmean(ratios),
            "geometric_mean": math.exp(statistics.fmean(
                math.log(value) for value in ratios)), "max": max(ratios),
        },
        "runtime_summary": None if not runtimes else {
            "mean_seconds": statistics.fmean(runtimes),
            "median_seconds": statistics.median(runtimes),
            "min_seconds": min(runtimes), "max_seconds": max(runtimes),
            "p95_seconds": _percentile(runtimes, 0.95),
        },
        "estimated_full_property_count": int(full["property_count"]),
        "estimated_full_makespan_2x_A40_seconds": projected_makespan,
        "estimate_method": (
            "sum each full property using its accepted-draw sanity runtime; "
            "deterministic longest-processing-time assignment to two A40s"),
        "properties": [{
            "property_id": row["property_id"],
            "proof_radius": row["certified_radius"],
            "cached_DeepT_radius": row["cached_DeepT_certified_radius"],
            "ratio": row["radius_ratio_to_cached_DeepT"],
            "runtime_seconds": row["total_wall_time_seconds"],
        } for row in sorted(results, key=lambda item: item["benchmark_ordinal"])],
        "scientific_queries_executed_by_analysis": 0,
        "bound_calls_executed_by_analysis": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, choices=(6, 12), required=True)
    parser.add_argument("--fresh-root", required=True)
    parser.add_argument("--expected-completed", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = analyze(args.depth, Path(args.fresh_root).resolve(),
                      args.expected_completed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
