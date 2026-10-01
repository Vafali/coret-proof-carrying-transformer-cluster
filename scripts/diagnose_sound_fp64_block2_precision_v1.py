#!/usr/bin/env python3
"""Read-only diagnosis of stopped sound-FP64 Block-2 variance failures."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "research_hab")]

import cluster_common
import run_sound_fp64_3l_campaign as campaign
import run_sound_fp64_3l_radius_recovery_pilot as pilot


SCHEMA = "CORET_SOUND_FP64_BLOCK2_PRECISION_DIAGNOSIS_V1"
DEFAULT_PROPERTY = "deept_table7_stdln3_s001_line1794_tok11"
NUMERICAL_REASONS = {
    "fp64_roundoff_coordinate_box",
    "sound_fp64_coordinate_box_replacement_with_numerical",
}


def _atomic_json(path: Path, value: dict) -> None:
    payload = dict(value)
    payload["record_sha256"] = cluster_common.canonical(payload)
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _source_paths(campaign_root: Path) -> dict[str, Path]:
    found = {}
    for path in sorted(campaign_root.glob("**/properties/*/result.json")):
        row = campaign._verified_result(path)
        property_id = row["property_id"]
        if property_id in found:
            raise RuntimeError(f"duplicate campaign property: {property_id}")
        found[property_id] = path
    if len(found) != pilot.EXPECTED_SOURCE_RESULTS:
        raise RuntimeError("frozen 40-result campaign inventory differs")
    return found


def _pilot_rows(pilot_root: Path, property_id: str) -> list[dict]:
    paths = sorted((pilot_root / "properties" / property_id
                    / "evaluations").glob("*.json"))
    rows = [pilot._verified_pilot_result(path) for path in paths]
    rows.sort(key=lambda row: next(
        index for index, (numerator, denominator) in enumerate(
            pilot.MULTIPLIERS)
        if (row["multiplier"]["numerator"],
            row["multiplier"]["denominator"]) ==
        (numerator, denominator)))
    if not rows:
        raise RuntimeError(f"no stopped-pilot records found: {property_id}")
    if any(row["property_id"] != property_id for row in rows):
        raise RuntimeError("pilot property identity differs")
    return rows


def _domain_row(row: dict) -> dict:
    diagnostic = row.get("layernorm_domain_diagnostic")
    if (row.get("classification") != "FAILED_AT_TESTED_RADIUS"
            or row.get("failure_category") !=
            "SOUND_LAYERNORM_DOMAIN_FAILURE"
            or not isinstance(diagnostic, dict)):
        raise RuntimeError("pilot row is not a LayerNorm-domain failure")
    required = {
        "input_shape", "token_count", "hidden_dimension", "generator_count",
        "minimum_token_index", "minimum_coordinate_index",
        "nominal_centered_second_moment", "variance_affine_center",
        "variance_lower_support", "variance_upper_support",
        "plain_relational_variance_lower_bound",
        "numerical_variance_widening_upper_bound", "sound_variance_lower",
        "sound_variance_upper_at_minimum", "layernorm_epsilon",
        "sqrt_input_lower", "sqrt_safety_margin", "native_generator_count",
        "numerical_generator_count",
    }
    if not required <= set(diagnostic):
        missing = sorted(required - set(diagnostic))
        raise RuntimeError(f"domain diagnostic is incomplete: {missing}")
    result = {
        "multiplier": row["multiplier"]["display"],
        "tested_radius": row["tested_radius"],
        "tested_radius_hex": row["tested_radius_hex"],
        "runtime_seconds": row["runtime_seconds"],
    }
    result.update({key: diagnostic[key] for key in sorted(required)})
    deficit = max(0.0, -float(diagnostic["sound_variance_lower"]))
    numerical = float(diagnostic["numerical_variance_widening_upper_bound"])
    nominal = float(diagnostic["nominal_centered_second_moment"])
    result.update({
        "positivity_deficit": deficit,
        "numerical_widening_to_deficit_ratio": (
            numerical / deficit if deficit else None),
        "numerical_widening_to_nominal_variance_ratio": (
            numerical / abs(nominal) if nominal else math.inf),
        "plain_relational_already_nonpositive": (
            float(diagnostic["plain_relational_variance_lower_bound"]) <= 0),
        "at_generator_cap": (
            int(diagnostic["generator_count"])
            == campaign.sound.MAXIMUM_GENERATORS),
    })
    return result


def _certified_comparator(source_paths: dict[str, Path]) -> dict:
    candidates = []
    for property_id, path in source_paths.items():
        row = campaign._verified_result(path)
        if row.get("classification") == "CERTIFIED_AT_HISTORICAL_RADIUS":
            candidates.append((
                0 if int(row["sentence_ordinal"]) == 2 else 1,
                property_id, path, row))
    if not candidates:
        raise RuntimeError("frozen campaign has no certified comparator")
    _priority, property_id, path, row = sorted(candidates)[0]
    report_path = path.parent / "certificate_report.json"
    if (not report_path.is_file()
            or cluster_common.sha256(report_path)
            != row["certificate_report_sha256"]):
        raise RuntimeError("certified comparator report identity differs")
    report = json.loads(report_path.read_text())
    layernorms = report.get("block2_layernorms")
    if not isinstance(layernorms, dict):
        raise RuntimeError("certified comparator LayerNorm telemetry missing")
    return {
        "selection_rule": (
            "lexicographically_first_certified_property_preferring_sentence_2"),
        "property_id": property_id,
        "sentence_ordinal": row["sentence_ordinal"],
        "sequence_length": report.get("fixture_token_ids") and len(
            report["fixture_token_ids"]),
        "historical_candidate_radius": row["historical_candidate_radius"],
        "final_sound_lower_margin": row["final_sound_lower_margin"],
        "final_generator_count": row["final_generator_count"],
        "max_numerical_native_ratio": row["max_numerical_native_ratio"],
        "block2_layernorms": layernorms,
        "block2_stages": report.get("stages", []),
        "block2_reductions": report.get("reductions", []),
        "certificate_report_sha256": row["certificate_report_sha256"],
    }


def _trend(rows: list[dict]) -> dict:
    first, last = rows[0], rows[-1]

    def ratio(field):
        initial, final = float(first[field]), float(last[field])
        return final / initial if initial else None

    return {
        "first_multiplier": first["multiplier"],
        "last_multiplier": last["multiplier"],
        "tested_radius_ratio": ratio("tested_radius"),
        "nominal_variance_ratio": ratio("nominal_centered_second_moment"),
        "variance_affine_center_ratio": ratio("variance_affine_center"),
        "lower_support_ratio": ratio("variance_lower_support"),
        "plain_relational_lower_ratio": ratio(
            "plain_relational_variance_lower_bound"),
        "numerical_widening_ratio": ratio(
            "numerical_variance_widening_upper_bound"),
        "sound_variance_lower_first": first["sound_variance_lower"],
        "sound_variance_lower_last": last["sound_variance_lower"],
        "generator_count_constant": len({row["generator_count"]
                                          for row in rows}) == 1,
        "numerical_generator_count_constant": len({
            row["numerical_generator_count"] for row in rows}) == 1,
        "failing_token_constant": len({row["minimum_token_index"]
                                        for row in rows}) == 1,
    }


def diagnose(campaign_root: Path, pilot_root: Path, artifact_root: Path,
             property_id: str) -> dict:
    # Reuse the full frozen input/population authentication. No evaluator is
    # reachable from this function.
    preflight = pilot.preflight(campaign_root, artifact_root)
    if property_id not in {row["property_id"]
                           for row in preflight["selected_properties"]}:
        raise RuntimeError("diagnostic property is not in frozen pilot selection")
    source_paths = _source_paths(campaign_root)
    source = campaign._verified_result(source_paths[property_id])
    if (source.get("failure_category") != "SOUND_LAYERNORM_DOMAIN_FAILURE"
            or source.get("scientific_evaluation_complete") is not True):
        raise RuntimeError("source property is not the authenticated domain failure")
    raw_rows = _pilot_rows(pilot_root, property_id)
    rows = [_domain_row(row) for row in raw_rows]
    return {
        "schema": SCHEMA,
        "property_id": property_id,
        "historical_candidate_radius": source["historical_candidate_radius"],
        "historical_candidate_radius_hex": source[
            "historical_candidate_radius_hex"],
        "sequence_length": rows[0]["token_count"],
        "evaluations": rows,
        "radius_dependence": _trend(rows),
        "certified_comparator": _certified_comparator(source_paths),
        "static_mechanisms": {
            "source_support": "O(radius)",
            "bilinear_source_terms": "mixed O(radius) and O(radius^2)",
            "fp64_reserve_floor": (
                "radius-independent because reserve majorants are clamped to 1"),
            "numerical_generator_allocation": (
                "one independent coordinate box per operator output coordinate"),
            "reduction_trigger": (
                "generator-count based; does not disappear as magnitudes shrink"),
            "reduction_effect": (
                "discarded correlated symbols become independent coordinate boxes"),
            "layernorm_variance": (
                "native precise quadratic relaxation of centered affine forms; "
                "its affine lower may be negative although exact variance is nonnegative"),
        },
        "persisted_telemetry_limit": (
            "failed records contain the decisive LayerNorm decomposition but "
            "not every earlier Block0/Block1 reduction witness"),
        "scientific_queries": 0,
        "bound_calls": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--pilot-root", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--property-id", default=DEFAULT_PROPERTY)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = diagnose(
        args.campaign_root.resolve(), args.pilot_root.resolve(),
        args.artifact_root.resolve(), args.property_id)
    _atomic_json(args.output.resolve(), report)
    print(json.dumps(cluster_common.verified_json(args.output.resolve()),
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
