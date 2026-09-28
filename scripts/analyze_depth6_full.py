#!/usr/bin/env python3
"""Read-only final analysis of full historical depth-6 A40 production."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import depth6_full_common as common


def _verified(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.get("record_sha256")
    body = dict(value)
    body.pop("record_sha256", None)
    if common.canonical(body) != claimed:
        raise RuntimeError(f"record identity mismatch: {path}")
    return value


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def analyze(fresh_root: Path) -> dict:
    manifest = common.production_manifest()
    plan = common.plan()
    expected = {row["property_id"]: row for row in manifest["properties"]}
    owner = {row["property_id"]: int(worker["worker_id"])
             for worker in plan["workers"] for item in worker["chunks"]
             for row in item["properties"]}
    results = []
    queries = []
    seen = set()
    duplicates = []
    for worker_id in (0, 1):
        root = common.worker_result_root(common.worker_dir(fresh_root, worker_id))
        for path in sorted((root / "properties").glob("*/result_v1.json")):
            property_id = path.parent.name
            if property_id in seen:
                duplicates.append(property_id)
                continue
            if property_id not in expected or owner[property_id] != worker_id:
                raise RuntimeError(f"foreign or wrongly owned property: {property_id}")
            row = _verified(path)
            if (row.get("property_id") != property_id
                    or row.get("canonical_manifest_sha256")
                    != manifest["canonical_manifest_sha256"]):
                raise RuntimeError("property scientific identity differs")
            seen.add(property_id)
            results.append(row)
            property_queries = []
            for query_path in sorted(path.parent.joinpath("queries").glob(
                    "query_*_result_v1.json")):
                query = _verified(query_path)
                if (query.get("property_id") != property_id
                        or query.get("canonical_manifest_sha256")
                        != manifest["canonical_manifest_sha256"]):
                    raise RuntimeError("query scientific identity differs")
                property_queries.append(query)
                queries.append(query)
            if [row["query_ordinal"] for row in property_queries] != list(
                    range(len(property_queries))):
                raise RuntimeError("query ordinal chain differs")
    missing = sorted(set(expected) - seen,
                     key=lambda pid: expected[pid]["benchmark_ordinal"])
    integrity = {
        "checker_failures": (
            sum(int(row.get("checker_failure_count", 0)) for row in results)
            + sum(row.get("independent_checker_accepted") is not True
                  for row in queries)),
        "provenance_failures": sum(
            row.get("all_provenance_consistent") is not True for row in results)
            + sum(row.get("provenance_ID_consistent") is not True
                  for row in queries),
        "support_failures": sum(
            row.get("all_support_claims_validated") is not True for row in results)
            + sum(row.get("all_support_claims_validated") is not True
                  for row in queries),
        "generic_fallbacks": (
            sum(int(row.get("generic_fallback_count", 0)) for row in results)
            + sum(int(row.get("generic_fallback_count", 0)) for row in queries)),
        "unexpected_failures": (
            sum(row.get("terminal_status") != "COMPLETE" for row in results)
            + sum(row.get("terminal_status") not in (
                "COMPLETE", "UNCERTIFIED_DOMAIN_FAILURE") for row in queries)),
        "typed_domain_failures": sum(
            row.get("terminal_status") == "UNCERTIFIED_DOMAIN_FAILURE"
            for row in queries),
    }
    ratios = [float(row["radius_ratio_to_cached_DeepT"]) for row in results]
    runtimes = [float(row["total_wall_time_seconds"]) for row in results]
    per_draw = defaultdict(list)
    for row in results:
        per_draw[int(row["sentence_ordinal"])].append(row)
    ratio_summary = None if not ratios else {
        "min": min(ratios), "q1": _quantile(ratios, 0.25),
        "median": statistics.median(ratios), "mean": statistics.fmean(ratios),
        "geometric_mean": math.exp(statistics.fmean(math.log(x) for x in ratios)),
        "q3": _quantile(ratios, 0.75), "max": max(ratios),
        "count_gt_1": sum(value > 1.0 for value in ratios),
        "count_lt_1": sum(value < 1.0 for value in ratios),
    }
    runtime_summary = None if not runtimes else {
        "total_GPU_seconds": sum(runtimes),
        "mean_property_seconds": statistics.fmean(runtimes),
        "median_property_seconds": statistics.median(runtimes),
        "min_property_seconds": min(runtimes),
        "max_property_seconds": max(runtimes),
        "total_query_seconds": sum(float(row.get("total_wall_time_seconds", 0.0))
                                   for row in queries),
        "peak_CPU_RSS_bytes": max(int(row.get("peak_CPU_RSS_bytes", 0))
                                  for row in results),
        "peak_GPU_allocated_bytes": max(
            int(row.get("peak_GPU_allocated_bytes", 0)) for row in results),
        "peak_GPU_reserved_bytes": max(
            int(row.get("peak_GPU_reserved_bytes", 0)) for row in results),
    }
    complete_and_clean = (len(results) == 137 and not missing and not duplicates
                          and not any(value for key, value in integrity.items()
                                      if key != "typed_domain_failures"))
    return {
        "schema": "CORET_DEPTH6_FULL_A40_ANALYSIS_V1",
        "classification": (
            "DEPTH6_FULL_137_COMPLETE" if complete_and_clean
            else "DEPTH6_FULL_INCOMPLETE_OR_FAILED"),
        "completed_properties": len(results),
        "expected_properties": 137,
        "missing_property_ids": missing,
        "duplicate_property_ids": sorted(duplicates),
        "scientific_manifest_sha256": common.EXPECTED_FULL_MANIFEST_SHA,
        "production_manifest_sha256": manifest["canonical_manifest_sha256"],
        "plan_sha256": plan["canonical_manifest_sha256"],
        "checkpoint_sha256": common.EXPECTED_CHECKPOINT_SHA,
        "scientific_source_commit": common.EXPECTED_SOURCE_COMMIT,
        **integrity,
        "ratio_summary": ratio_summary,
        "runtime_summary": runtime_summary,
        "per_sentence_draw": [{
            "sentence_ordinal": draw,
            "property_count": len(rows),
            "ratio_min": min(float(row["radius_ratio_to_cached_DeepT"])
                             for row in rows),
            "ratio_median": statistics.median(
                float(row["radius_ratio_to_cached_DeepT"]) for row in rows),
            "ratio_mean": statistics.fmean(
                float(row["radius_ratio_to_cached_DeepT"]) for row in rows),
            "total_runtime_seconds": sum(float(row["total_wall_time_seconds"])
                                         for row in rows),
        } for draw, rows in sorted(per_draw.items())],
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
    parser.add_argument("--fresh-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = analyze(Path(args.fresh_root).resolve())
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
