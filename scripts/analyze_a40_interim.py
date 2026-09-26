#!/usr/bin/env python3
"""Read-only interim analysis of isolated fresh-A40 worker results."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from a40_fresh_common import (assigned_properties, load_plan, result_root,
    seed_record)
from cluster_common import (DEEPT_CACHE_SHA, PRODUCTION_MANIFEST_SHA,
    SCIENTIFIC_MANIFEST_SHA, artifact_root, canonical,
    historical_manifest_path, load_production_manifest, sha256, verified_json,
    verify_artifact_manifest)
from merge_worker_results import validate_property


RECIPROCAL_NAN = (
    "Reciprocal: there are NaNs in the new COEFFS, pre-condition not met")
DOMAIN_FAILURES = (
    ("sqrt_domain_not_positive", "sqrt: Bounds must be positive"),
    ("reciprocal_domain_not_positive",
     "reciprocal: Bounds must be positive"),
)
A4000_REFERENCE = {
    "count": 49,
    "min": 1.0224657534,
    "median": 1.0424319189,
    "mean": 1.0450246819,
    "geometric_mean": 1.0449252893,
    "max": 1.0809128631,
    "count_gt_1": 49,
    "count_lt_1": 0,
}


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("statistics require at least one value")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values: list[float]) -> dict:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("distribution values must be finite and nonempty")
    return {
        "n": len(values),
        "min": min(values),
        "q1": percentile(values, 0.25),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "geometric_mean": math.exp(statistics.fmean(
            math.log(value) for value in values))
            if all(value > 0 for value in values) else None,
        "q3": percentile(values, 0.75),
        "max": max(values),
    }


def timing_distribution(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise RuntimeError("persisted runtime is invalid")
    return {
        "n": len(values),
        "sum_seconds": sum(values),
        "mean_seconds": statistics.fmean(values),
        "median_seconds": statistics.median(values),
        "p90_seconds": percentile(values, 0.90),
        "p95_seconds": percentile(values, 0.95),
        "max_seconds": max(values),
    }


def classify_domain_failure(row: dict, path: Path) -> str:
    if (row.get("terminal_status") != "UNCERTIFIED_DOMAIN_FAILURE"
            or row.get("reason_code") != "UNCERTIFIED_DOMAIN_FAILURE"
            or row.get("exception_type") != "AssertionError"
            or row.get("certified") is not False
            or row.get("complete_certificate") is not False):
        raise RuntimeError(f"invalid typed domain failure: {path}")
    message = row.get("exception_message", "")
    if message == RECIPROCAL_NAN:
        return "reciprocal_nan_native_fail_closed"
    for name, prefix in DOMAIN_FAILURES:
        if message.startswith(prefix):
            return name
    raise RuntimeError(f"unrecognized typed domain failure: {path}")


def load_deept_cache(artifact: Path, manifest: dict) -> dict[str, dict]:
    path = Path(manifest["cached_native_DeepT"]["positions_path"])
    if not path.is_file() or sha256(path) != DEEPT_CACHE_SHA:
        raise RuntimeError("cached DeepT positions identity differs")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != 127:
        raise RuntimeError("cached DeepT property count differs")
    result = {}
    for row in rows:
        property_id = row["property_id"]
        if (property_id in result
                or row.get("manifest_sha256") != SCIENTIFIC_MANIFEST_SHA):
            raise RuntimeError("cached DeepT property identity differs")
        result[property_id] = row
    historical = verified_json(historical_manifest_path(artifact),
                               "canonical_manifest_sha256")
    if historical["canonical_manifest_sha256"] != SCIENTIFIC_MANIFEST_SHA:
        raise RuntimeError("historical scientific manifest differs")
    return result


def validate_final_certificate(property_dir: Path, result: dict) -> None:
    claimed = result.get("final_certificate_record_sha256")
    if not claimed or result.get("complete_certificate") is not True:
        raise RuntimeError(
            f"final certificate identity unavailable: {property_dir.name}")
    observed = set()
    for path in (property_dir / "queries").glob(
            "query_*_certificates_v1.json"):
        certificate = verified_json(path)
        if (certificate.get("canonical_manifest_sha256")
                != PRODUCTION_MANIFEST_SHA
                or certificate.get("property_id") != property_dir.name):
            raise RuntimeError(f"certificate identity differs: {path}")
        observed.add(certificate.get("record_sha256"))
    if claimed not in observed:
        raise RuntimeError(
            f"final certificate hash is not present: {property_dir.name}")


def inspect_queries(property_dir: Path) -> tuple[list[dict], Counter]:
    paths = sorted((property_dir / "queries").glob(
        "query_*_result_v1.json"))
    rows = [verified_json(path) for path in paths]
    categories = Counter()
    for path, row in zip(paths, rows):
        if row.get("canonical_manifest_sha256") != PRODUCTION_MANIFEST_SHA:
            raise RuntimeError(f"query production identity differs: {path}")
        status = row.get("terminal_status")
        if status == "COMPLETE":
            categories["complete"] += 1
        elif status == "UNCERTIFIED_DOMAIN_FAILURE":
            categories[classify_domain_failure(row, path)] += 1
        else:
            raise RuntimeError(f"unexpected query terminal status: {path}")
    return rows, categories


def classification(ratios: list[float], integrity: dict) -> tuple[str, str]:
    if any(integrity.values()):
        return "A40_INTERIM_WEAK", "NO"
    stats = distribution(ratios)
    losses = sum(value < 1.0 for value in ratios)
    if (stats["median"] > 1.0 and stats["geometric_mean"] > 1.0
            and losses == 0):
        return "A40_INTERIM_STRONG", "YES"
    if stats["median"] > 1.0 and stats["geometric_mean"] > 1.0:
        return "A40_INTERIM_MIXED", "NO"
    return "A40_INTERIM_WEAK", "NO"


def analyze(fresh_root: Path, artifact: Path,
            expected_completed: int) -> dict:
    fresh_root = fresh_root.resolve()
    artifact = artifact.resolve()
    if expected_completed <= 0 or expected_completed > 127:
        raise ValueError("expected completed count must be in [1,127]")
    verify_artifact_manifest(artifact)
    manifest = load_production_manifest(artifact)
    if (manifest["canonical_manifest_sha256"] != PRODUCTION_MANIFEST_SHA
            or manifest["historical_scientific_manifest"][
                "canonical_sha256"] != SCIENTIFIC_MANIFEST_SHA):
        raise RuntimeError("frozen manifest identity differs")
    deep_cache = load_deept_cache(artifact, manifest)
    manifest_rows = {row["property_id"]: row for row in manifest["properties"]}
    if len(manifest_rows) != 127:
        raise RuntimeError("production manifest property count differs")
    if set(deep_cache) != set(manifest_rows):
        raise RuntimeError("DeepT/production property inventory differs")
    plan = load_plan()
    if (plan["scientific_manifest_sha256"] != SCIENTIFIC_MANIFEST_SHA
            or plan["production_manifest_sha256"] != PRODUCTION_MANIFEST_SHA):
        raise RuntimeError("A40 plan identity differs")

    discovered: list[tuple[int, str, Path]] = []
    seen = set()
    for worker_id in (0, 1):
        worker_dir = fresh_root / f"worker_{worker_id}"
        marker = verified_json(worker_dir / "fresh_worker_seed_v1.json")
        if marker != seed_record(artifact, worker_id):
            raise RuntimeError(f"worker {worker_id} seed identity differs")
        allowed = assigned_properties(plan, worker_id)
        root = result_root(worker_dir)
        for result_path in sorted(
                (root / "properties").glob("*/result_v1.json")):
            property_id = result_path.parent.name
            if property_id in seen:
                raise RuntimeError(f"duplicate completed property: {property_id}")
            if property_id not in allowed:
                raise RuntimeError(
                    f"worker {worker_id} contains unowned property: {property_id}")
            seen.add(property_id)
            discovered.append((worker_id, property_id, root))
    if len(discovered) != expected_completed:
        raise RuntimeError(
            f"completed property count differs: {len(discovered)} != "
            f"{expected_completed}")

    properties = []
    all_queries = []
    failure_counts = Counter()
    for worker_id, property_id, root in sorted(
            discovered, key=lambda item: manifest_rows[item[1]][
                "benchmark_ordinal"]):
        result = validate_property(root, property_id)
        property_dir = root / "properties" / property_id
        expected = manifest_rows[property_id]
        if (result.get("canonical_manifest_sha256")
                != PRODUCTION_MANIFEST_SHA
                or result.get("property_id") != property_id
                or result.get("benchmark_ordinal")
                != expected["benchmark_ordinal"]
                or result.get("sentence_ordinal")
                != expected["sentence_ordinal"]
                or result.get("token_position") != expected["token_position"]
                or result.get("reused_from_optimized_smoke") is not False
                or int(result.get("fresh_verifier_evaluations_this_run", 0))
                <= 0):
            raise RuntimeError(f"property identity differs: {property_id}")
        validate_final_certificate(property_dir, result)
        query_rows, categories = inspect_queries(property_dir)
        if (result.get("verifier_evaluation_count") != len(query_rows)
                or result.get("domain_failure_count") != sum(
                    categories[key] for key in categories
                    if key != "complete")
                or result.get("certified_query_count") != sum(
                    row.get("certified") is True for row in query_rows)):
            raise RuntimeError(f"property/query counts differ: {property_id}")
        all_queries.extend(query_rows)
        failure_counts.update(categories)
        deep = float(deep_cache[property_id][
            "certified_lower_endpoint_binary64"])
        if deep <= 0 or result.get("cached_DeepT_certified_radius") != deep:
            raise RuntimeError(f"cached DeepT radius differs: {property_id}")
        proof = float(result["certified_radius"])
        ratio = proof / deep
        stored_ratio = result.get("radius_ratio_to_cached_DeepT")
        if (stored_ratio is None
                or not math.isclose(stored_ratio, ratio, rel_tol=0, abs_tol=1e-15)):
            raise RuntimeError(f"stored ratio differs: {property_id}")
        properties.append({
            "benchmark_ordinal": expected["benchmark_ordinal"],
            "worker_id": worker_id,
            "property_id": property_id,
            "sentence_id": expected["sentence_id"],
            "sequence_length": expected["sequence_length"],
            "token_position": expected["token_position"],
            "proof_radius": proof,
            "cached_DeepT_radius": deep,
            "ratio": ratio,
            "property_wall_time_seconds": result["total_wall_time_seconds"],
            "verifier_evaluation_count": result["verifier_evaluation_count"],
            "certified_query_count": result["certified_query_count"],
            "domain_failure_count": result["domain_failure_count"],
            "record_sha256": result["record_sha256"],
        })

    ratios = [row["ratio"] for row in properties]
    ratio_stats = distribution(ratios)
    integrity = {
        "checker_failures": sum(int(verified_json(
            root / "properties" / property_id / "result_v1.json").get(
                "checker_failure_count", 0))
            for _, property_id, root in discovered),
        "provenance_failures": 0,
        "support_failures": 0,
        "generic_fallbacks": sum(int(row.get("generic_fallback_count", 0))
                                  for row in all_queries),
        "unexpected_failures": 0,
    }
    label, continuation = classification(ratios, integrity)
    property_times = [float(row["property_wall_time_seconds"])
                      for row in properties]
    query_times = [float(row["total_wall_time_seconds"])
                   for row in all_queries]
    complete_times = [float(row["total_wall_time_seconds"])
                      for row in all_queries
                      if row["terminal_status"] == "COMPLETE"]
    domain_times = [float(row["total_wall_time_seconds"])
                    for row in all_queries
                    if row["terminal_status"]
                    == "UNCERTIFIED_DOMAIN_FAILURE"]
    output = {
        "schema": "CORET_A40_INTERIM_ANALYSIS_V1",
        "status": label,
        "continue_next_chunk": continuation,
        "scope": "fresh homogeneous A40 interim evidence only",
        "snapshot": {
            "fresh_root": str(fresh_root),
            "artifact_root": str(artifact),
            "expected_completed": expected_completed,
            "validated_completed": len(properties),
            "worker_completed_counts": {
                str(worker): sum(row["worker_id"] == worker
                                 for row in properties)
                for worker in (0, 1)},
            "scientific_manifest_sha256": SCIENTIFIC_MANIFEST_SHA,
            "production_manifest_sha256": PRODUCTION_MANIFEST_SHA,
            "DeepT_cache_sha256": DEEPT_CACHE_SHA,
        },
        "ratio_statistics": ratio_stats,
        "threshold_counts": {
            "ratio_gt_1": sum(value > 1.0 for value in ratios),
            "ratio_lt_1": sum(value < 1.0 for value in ratios),
            "ratio_ge_1.01": sum(value >= 1.01 for value in ratios),
            "ratio_ge_1.03": sum(value >= 1.03 for value in ratios),
            "ratio_ge_1.05": sum(value >= 1.05 for value in ratios),
        },
        "worst_5": sorted(properties, key=lambda row: (
            row["ratio"], row["benchmark_ordinal"]))[:5],
        "best_5": sorted(properties, key=lambda row: (
            -row["ratio"], row["benchmark_ordinal"]))[:5],
        "integrity_and_failures": {
            **integrity,
            "typed_domain_failure_counts": {
                key: failure_counts[key] for key in (
                    "sqrt_domain_not_positive",
                    "reciprocal_domain_not_positive",
                    "reciprocal_nan_native_fail_closed")},
            "typed_domain_failure_total": sum(
                failure_counts[key] for key in failure_counts
                if key != "complete"),
            "successful_query_count": failure_counts["complete"],
        },
        "runtime": {
            "properties": timing_distribution(property_times),
            "all_persisted_queries": timing_distribution(query_times),
            "successful_queries": timing_distribution(complete_times),
            "typed_domain_failure_queries": timing_distribution(domain_times),
            "persisted_component_totals_seconds": {
                key: sum(float(row.get(key, 0.0)) for row in all_queries)
                for key in ("setup_seconds", "bound_runtime_seconds",
                            "proof_generation_time_seconds",
                            "independent_checker_time_seconds",
                            "certificate_serialization_seconds")},
            "peak_CPU_RSS_bytes": max(int(row.get("peak_CPU_RSS_bytes", 0))
                                      for row in all_queries),
            "peak_GPU_allocated_bytes": max(int(row.get(
                "peak_GPU_allocated_bytes", 0)) for row in all_queries),
            "peak_GPU_reserved_bytes": max(int(row.get(
                "peak_GPU_reserved_bytes", 0)) for row in all_queries),
        },
        "A4000_interim_49_descriptive_comparison": {
            "reference": A4000_REFERENCE,
            "datasets_numerically_merged": False,
            "A40_minus_A4000": {
                key: ratio_stats[key] - A4000_REFERENCE[key]
                for key in ("min", "median", "mean", "geometric_mean",
                            "max")},
        },
        "properties": properties,
        "analysis_scientific_queries_executed": 0,
        "analysis_bound_entrypoint_calls": 0,
    }
    output["record_sha256"] = canonical(output)
    return output


def write_output(path: Path, value: dict, fresh_root: Path,
                 artifact: Path) -> None:
    output = path.expanduser().resolve()
    for protected in (fresh_root.resolve(), artifact.resolve()):
        if output == protected or output.is_relative_to(protected):
            raise RuntimeError("analysis output must be outside input trees")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-root", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--expected-completed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    fresh_root = Path(args.fresh_root)
    artifact = artifact_root(args.artifact_root)
    value = analyze(fresh_root, artifact, args.expected_completed)
    write_output(Path(args.output), value, fresh_root, artifact)
    print(json.dumps({
        "status": value["status"],
        "continue_next_chunk": value["continue_next_chunk"],
        "completed": value["snapshot"]["validated_completed"],
        "output": str(Path(args.output).expanduser().resolve()),
        "record_sha256": value["record_sha256"],
        "scientific_queries_executed": 0,
        "bound_entrypoint_calls": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
