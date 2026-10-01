#!/usr/bin/env python3
"""Frozen six-property sound-FP64 radius-recovery pilot.

This is orchestration only.  Every tested radius is evaluated by the existing
``run_sound_fp64_3l_campaign.execute_property`` path.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import resource
import sys
import time
from fractions import Fraction
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "research_hab")]

import cluster_common
import run_sound_fp64_3l_campaign as campaign


SCHEMA = "CORET_SOUND_FP64_3L_RADIUS_RECOVERY_PILOT_V1"
RESULT_SCHEMA = "CORET_SOUND_FP64_3L_RADIUS_RECOVERY_RESULT_V1"
SUMMARY_SCHEMA = "CORET_SOUND_FP64_3L_RADIUS_RECOVERY_SUMMARY_V1"
EXPECTED_ELIGIBLE = 37
EXPECTED_SOURCE_RESULTS = 40
SELECTION_INDICES = (0, 7, 14, 22, 29, 36)
MULTIPLIERS = ((95, 100), (90, 100), (75, 100), (50, 100), (25, 100))
EXPECTED_SELECTION = (
    ("deept_table7_stdln3_s001_line1794_tok11", 0.0008105468750000001),
    ("deept_table7_stdln3_s001_line1794_tok19", 0.0010491943359375003),
    ("deept_table7_stdln3_s006_line882_tok12", 0.0011120605468750001),
    ("deept_table7_stdln3_s004_line216_tok04", 0.0012731933593749997),
    ("deept_table7_stdln3_s001_line1794_tok03", 0.001326904296875),
    ("deept_table7_stdln3_s003_line2031_tok15", 0.0014318847656249998),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    payload = dict(value)
    payload["record_sha256"] = cluster_common.canonical(payload)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _source_results(campaign_root: Path) -> list[dict]:
    paths = sorted(campaign_root.glob("**/properties/*/result.json"))
    rows, seen = [], set()
    for path in paths:
        row = campaign._verified_result(path)
        property_id = row.get("property_id")
        if not isinstance(property_id, str) or property_id in seen:
            raise RuntimeError("duplicate or malformed exact-radius result")
        seen.add(property_id)
        rows.append(row)
    if len(rows) != EXPECTED_SOURCE_RESULTS:
        raise RuntimeError(
            f"frozen exact-radius result count differs: {len(rows)} != "
            f"{EXPECTED_SOURCE_RESULTS}")
    return rows


def _eligible_population(rows: list[dict]) -> list[dict]:
    eligible = [row for row in rows
                if row.get("scientific_evaluation_complete") is True
                and row.get("classification") ==
                "FAILED_AT_HISTORICAL_RADIUS"
                and row.get("failure_category") ==
                "SOUND_LAYERNORM_DOMAIN_FAILURE"]
    if len(eligible) != EXPECTED_ELIGIBLE:
        raise RuntimeError(
            f"eligible failure count differs: {len(eligible)} != "
            f"{EXPECTED_ELIGIBLE}")
    eligible.sort(key=lambda row: (
        float(row["historical_candidate_radius"]), row["property_id"]))
    return eligible


def _selected_population(eligible: list[dict]) -> list[dict]:
    if len(eligible) != EXPECTED_ELIGIBLE:
        raise RuntimeError("selection requires the frozen 37-property population")
    selected = [eligible[index] for index in SELECTION_INDICES]
    actual = tuple((row["property_id"],
                    float(row["historical_candidate_radius"]))
                   for row in selected)
    expected = tuple((property_id, float(radius))
                     for property_id, radius in EXPECTED_SELECTION)
    if actual != expected:
        raise RuntimeError(
            f"deterministic pilot selection differs: {actual!r} != {expected!r}")
    return selected


def _tested_radius(historical: float, numerator: int,
                   denominator: int) -> float:
    if (not math.isfinite(historical) or historical < 0
            or numerator <= 0 or denominator <= 0
            or numerator >= denominator):
        raise RuntimeError("invalid radius-ladder operand")
    # Fraction.from_float binds the exact input binary64.  float(Fraction)
    # performs the one and only correctly rounded conversion back to binary64.
    return float(Fraction.from_float(historical)
                 * Fraction(numerator, denominator))


def _prepare(campaign_root: Path, artifact_root: Path):
    campaign_identity = campaign.preflight(artifact_root)
    manifest = cluster_common.load_production_manifest(artifact_root)
    resolved, token_source = campaign._resolved_campaign_inputs(
        artifact_root, manifest)
    source_rows = _source_results(campaign_root)
    if (sum(row.get("classification") == "CERTIFIED_AT_HISTORICAL_RADIUS"
            for row in source_rows) != 3
            or any(row.get("scientific_evaluation_complete") is not True
                   for row in source_rows)):
        raise RuntimeError("frozen exact-radius 3/37 scientific split differs")
    eligible = _eligible_population(source_rows)
    selected = _selected_population(eligible)
    prepared = []
    for source in selected:
        property_id = source["property_id"]
        if property_id not in resolved:
            raise RuntimeError(f"selected property missing from manifest: {property_id}")
        row = resolved[property_id]
        historical = campaign._candidate_radius(row)
        if (historical != float(source["historical_candidate_radius"])
                or historical != dict(EXPECTED_SELECTION)[property_id]):
            raise RuntimeError(f"selected historical radius differs: {property_id}")
        if (len(row["token_ids"]) != row["sequence_length"]
                or row["clean_label"] != row["nominal_prediction"]):
            raise RuntimeError(f"selected frozen input differs: {property_id}")
        prepared.append((row, source))
    population_binding = [{
        "property_id": row["property_id"],
        "historical_radius_hex": float(
            row["historical_candidate_radius"]).hex(),
        "source_record_sha256": row["record_sha256"],
    } for row in eligible]
    return prepared, {
        "campaign_identity": campaign_identity,
        "token_source": token_source,
        "eligible_population_sha256": cluster_common.canonical(
            population_binding),
    }


def preflight(campaign_root: Path, artifact_root: Path) -> dict:
    prepared, identity = _prepare(campaign_root, artifact_root)
    return {
        "schema": SCHEMA,
        "eligible_failure_count": EXPECTED_ELIGIBLE,
        "source_result_count": EXPECTED_SOURCE_RESULTS,
        "selection_indices": list(SELECTION_INDICES),
        "selected_properties": [{
            "property_id": row["property_id"],
            "historical_candidate_radius": historical,
            "historical_candidate_radius_hex": historical.hex(),
        } for row, _source in prepared
          for historical in [campaign._candidate_radius(row)]],
        "planned_multipliers": [f"{n}/{d}" for n, d in MULTIPLIERS],
        "planned_multiplier_binary64": [float(n / d) for n, d in MULTIPLIERS],
        "maximum_new_evaluations": len(prepared) * len(MULTIPLIERS),
        "authoritative_token_source": identity["token_source"],
        "scientific_manifest_sha256": identity["campaign_identity"][
            "scientific_manifest_sha256"],
        "production_manifest_sha256": identity["campaign_identity"][
            "production_manifest_sha256"],
        "eligible_population_sha256": identity[
            "eligible_population_sha256"],
        "scientific_queries": 0,
        "bound_calls": 0,
    }


def _hardware(device: str) -> dict:
    torch = campaign.sound.torch
    target = torch.device(device)
    properties = torch.cuda.get_device_properties(target)
    return {
        "device": str(target),
        "gpu_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": int(properties.total_memory),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "host": platform.node(),
    }


def _result_path(output_root: Path, property_id: str, ordinal: int,
                 numerator: int, denominator: int) -> Path:
    return (output_root / "properties" / property_id / "evaluations"
            / f"{ordinal:02d}_{numerator}_of_{denominator}.json")


def _verified_pilot_result(path: Path) -> dict:
    row = cluster_common.verified_json(path)
    required = {
        "schema", "property_id", "historical_candidate_radius",
        "historical_candidate_radius_hex", "multiplier",
        "tested_radius", "tested_radius_hex", "scientific_evaluation_complete",
        "certified_at_tested_radius", "classification", "runtime_seconds",
        "hardware", "source_hashes", "record_sha256",
    }
    if row.get("schema") != RESULT_SCHEMA or not required <= set(row):
        raise RuntimeError(f"pilot result inventory differs: {path}")
    multiplier = row.get("multiplier")
    if (not isinstance(multiplier, dict)
            or set(multiplier) != {"numerator", "denominator", "display",
                                   "binary64", "binary64_hex"}):
        raise RuntimeError(f"pilot multiplier identity differs: {path}")
    expected = _tested_radius(
        float(row["historical_candidate_radius"]),
        int(multiplier["numerator"]), int(multiplier["denominator"]))
    multiplier_binary64 = float(
        int(multiplier["numerator"]) / int(multiplier["denominator"]))
    if (row["historical_candidate_radius_hex"] !=
            float(row["historical_candidate_radius"]).hex()
            or float(row["tested_radius"]) != expected
            or row["tested_radius_hex"] != expected.hex()
            or float(multiplier["binary64"]) != multiplier_binary64
            or multiplier["binary64_hex"] != multiplier_binary64.hex()
            or multiplier["display"] != f"{multiplier_binary64:.2f}"):
        raise RuntimeError(f"pilot radius identity differs: {path}")
    classification = row["classification"]
    if classification == "INFRASTRUCTURE_FAILURE":
        if (row["scientific_evaluation_complete"] is not False
                or row["certified_at_tested_radius"] is not None):
            raise RuntimeError(f"pilot infrastructure status differs: {path}")
    elif classification in {
            "CERTIFIED_AT_TESTED_RADIUS", "FAILED_AT_TESTED_RADIUS"}:
        expected_certified = classification == "CERTIFIED_AT_TESTED_RADIUS"
        if (row["scientific_evaluation_complete"] is not True
                or row["certified_at_tested_radius"] is not expected_certified):
            raise RuntimeError(f"pilot scientific status differs: {path}")
    else:
        raise RuntimeError(f"pilot classification differs: {path}")
    return row


def _evaluation_record(row: dict, source: dict, internal: dict,
                       historical: float, numerator: int, denominator: int,
                       tested: float, elapsed: float, hardware: dict) -> dict:
    scientific = internal["scientific_evaluation_complete"] is True
    certified = (internal.get("certified_at_historical_radius") is True
                 if scientific else None)
    classification = (
        "CERTIFIED_AT_TESTED_RADIUS" if certified is True else
        "FAILED_AT_TESTED_RADIUS" if scientific else
        "INFRASTRUCTURE_FAILURE")
    failure_category = internal.get("failure_category")
    if scientific and not certified and failure_category is None:
        failure_category = "NONPOSITIVE_FINAL_SOUND_MARGIN"
    return {
        "schema": RESULT_SCHEMA,
        "property_id": row["property_id"],
        "sentence_ordinal": int(row["sentence_ordinal"]),
        "token_position": int(row["token_position"]),
        "authoritative_token_source": row["token_input_source"],
        "source_exact_radius_record_sha256": source["record_sha256"],
        "adapter_evaluation_record_sha256": internal.get("record_sha256"),
        "adapter_certificate_sha256": internal.get("certificate_sha256"),
        "adapter_certificate_report_sha256": internal.get(
            "certificate_report_sha256"),
        "historical_candidate_radius": historical,
        "historical_candidate_radius_hex": historical.hex(),
        "multiplier": {
            "numerator": numerator, "denominator": denominator,
            "display": f"{numerator / denominator:.2f}",
            "binary64": float(numerator / denominator),
            "binary64_hex": float(numerator / denominator).hex(),
        },
        "tested_radius": tested,
        "tested_radius_hex": tested.hex(),
        "scientific_evaluation_complete": scientific,
        "certified_at_tested_radius": certified,
        "classification": classification,
        "failure_category": failure_category,
        "failure_stage": internal.get("failure_stage"),
        "final_sound_lower_margin": internal.get("final_sound_lower_margin"),
        "final_sound_upper_margin": internal.get("final_sound_upper_margin"),
        "final_generator_count": internal.get("final_generator_count"),
        "max_numerical_native_ratio": internal.get(
            "max_numerical_native_ratio"),
        "layernorm_domain_diagnostic": internal.get(
            "domain_failure_diagnostic"),
        "runtime_seconds": elapsed,
        "verifier_runtime_seconds": internal.get("runtime_seconds"),
        "peak_gpu_allocated_bytes": internal.get(
            "peak_gpu_allocated_bytes"),
        "peak_gpu_reserved_bytes": internal.get("peak_gpu_reserved_bytes"),
        "peak_cpu_rss_bytes": int(resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "hardware": hardware,
        "source_hashes": {
            "pilot_runner_sha256": _sha256(Path(__file__)),
            "campaign_runner_sha256": _sha256(Path(campaign.__file__)),
            "finish_runner_sha256": _sha256(Path(campaign.finish3l.__file__)),
            "sound_fp64_sha256": _sha256(Path(campaign.sound.__file__)),
        },
        "binary_search_performed": False,
        "historical_radius_reexecuted": False,
    }


def run_worker(campaign_root: Path, artifact_root: Path, output_root: Path,
               device_index: int) -> dict:
    prepared, identity = _prepare(campaign_root, artifact_root)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("pilot requires exactly one visible GPU")
    device = f"cuda:{device_index}"
    hardware = _hardware(device)
    completed = []
    new_evaluations = 0
    started = time.perf_counter()
    for row, source in prepared:
        historical = campaign._candidate_radius(row)
        for ordinal, (numerator, denominator) in enumerate(MULTIPLIERS):
            path = _result_path(
                output_root, row["property_id"], ordinal,
                numerator, denominator)
            if path.exists():
                result = _verified_pilot_result(path)
            else:
                tested = _tested_radius(historical, numerator, denominator)
                evaluation_row = copy.deepcopy(row)
                reference = evaluation_row["cached_DeepT_reference"]
                reference[campaign.CANDIDATE_RADIUS_FIELD] = tested
                reference[campaign.CANDIDATE_RADIUS_HEX_FIELD] = tested.hex()
                internal_root = (output_root / "internal_campaign_adapter"
                                 / row["property_id"]
                                 / f"{ordinal:02d}_{numerator}_of_{denominator}")
                campaign._property_boundary_cleanup(device)
                evaluation_started = time.perf_counter()
                try:
                    internal = campaign.execute_property(
                        evaluation_row, internal_root, device)
                finally:
                    campaign._property_boundary_cleanup(device)
                result = _evaluation_record(
                    row, source, internal, historical, numerator, denominator,
                    tested, time.perf_counter() - evaluation_started,
                    hardware)
                _atomic_json(path, result)
                result = _verified_pilot_result(path)
                new_evaluations += 1
            if (result["property_id"] != row["property_id"]
                    or float(result["historical_candidate_radius"])
                    != historical
                    or result["multiplier"]["numerator"] != numerator
                    or result["multiplier"]["denominator"] != denominator):
                raise RuntimeError(f"pilot resume identity differs: {path}")
            completed.append(result)
            if (result["classification"] == "CERTIFIED_AT_TESTED_RADIUS"
                    or result["classification"] == "INFRASTRUCTURE_FAILURE"):
                break
    report = {
        "schema": SCHEMA,
        "eligible_population_sha256": identity[
            "eligible_population_sha256"],
        "selected_property_count": len(prepared),
        "persisted_evaluation_count": len(completed),
        "new_scientific_evaluations": new_evaluations,
        "wall_seconds": time.perf_counter() - started,
        "hardware": hardware,
        "scientific_queries": new_evaluations,
        "binary_searches": 0,
    }
    _atomic_json(output_root / "worker_summary.json", report)
    return cluster_common.verified_json(output_root / "worker_summary.json")


def summarize(campaign_root: Path, artifact_root: Path, output_root: Path,
              destination: Path) -> dict:
    prepared, identity = _prepare(campaign_root, artifact_root)
    properties, buckets = [], {key: 0 for key in (
        "0.95", "0.90", "0.75", "0.50", "0.25", "none", "infra")}
    for row, _source in prepared:
        historical = campaign._candidate_radius(row)
        evaluations = []
        terminal = False
        for ordinal, (numerator, denominator) in enumerate(MULTIPLIERS):
            path = _result_path(
                output_root, row["property_id"], ordinal,
                numerator, denominator)
            if not path.exists():
                if terminal:
                    break
                raise RuntimeError(f"pilot result is incomplete: {path}")
            result = _verified_pilot_result(path)
            evaluations.append(result)
            if result["classification"] in {
                    "CERTIFIED_AT_TESTED_RADIUS", "INFRASTRUCTURE_FAILURE"}:
                terminal = True
                break
        if not terminal and len(evaluations) != len(MULTIPLIERS):
            raise RuntimeError(f"pilot property is incomplete: {row['property_id']}")
        certified = next((item for item in evaluations
                          if item["certified_at_tested_radius"] is True), None)
        infrastructure = next((item for item in evaluations
                               if item["classification"] ==
                               "INFRASTRUCTURE_FAILURE"), None)
        first = certified["multiplier"]["display"] if certified else None
        if infrastructure:
            buckets["infra"] += 1
        elif first:
            buckets[first] += 1
        else:
            buckets["none"] += 1
        prior_failure = None
        if certified:
            if len(evaluations) == 1:
                # The authenticated source population supplies the already
                # observed 1.00x domain failure; it is never reexecuted.
                prior_failure = "1.00"
            else:
                prior = evaluations[-2]
                if prior["classification"] == "FAILED_AT_TESTED_RADIUS":
                    prior_failure = prior["multiplier"]["display"]
        properties.append({
            "property_id": row["property_id"],
            "historical_candidate_radius": historical,
            "historical_candidate_radius_hex": historical.hex(),
            "historical_1.00x_outcome": "FAIL",
            "historical_1.00x_reexecuted": False,
            "evaluations": [{
                "multiplier": item["multiplier"]["display"],
                "outcome": ("PASS" if item["certified_at_tested_radius"]
                            is True else "INFRA" if item["classification"] ==
                            "INFRASTRUCTURE_FAILURE" else "FAIL"),
            } for item in evaluations],
            "first_observed_certified_multiplier": first,
            "immediately_preceding_tested_failed_multiplier": prior_failure,
            "total_verifier_evaluations": len(evaluations),
            "total_runtime_seconds": sum(
                float(item["runtime_seconds"]) for item in evaluations),
            "hardware": [json.loads(value) for value in sorted({
                json.dumps(item["hardware"], sort_keys=True)
                for item in evaluations})],
        })
    summary = {
        "schema": SUMMARY_SCHEMA,
        "eligible_population_sha256": identity[
            "eligible_population_sha256"],
        "selected_property_count": 6,
        "properties_recovering_at_0.95": buckets["0.95"],
        "first_recovery_at_0.90": buckets["0.90"],
        "first_recovery_at_0.75": buckets["0.75"],
        "first_recovery_at_0.50": buckets["0.50"],
        "first_recovery_at_0.25": buckets["0.25"],
        "no_recovery_observed_at_or_above_0.25": buckets["none"],
        "infrastructure_failures": buckets["infra"],
        "not_a_population_certification_rate": True,
        "no_maximum_radius_claim": True,
        "properties": properties,
    }
    _atomic_json(destination, summary)
    return cluster_common.verified_json(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "worker", "summarize"):
        child = sub.add_parser(name)
        child.add_argument("--campaign-root", required=True, type=Path)
        child.add_argument("--artifact-root", required=True, type=Path)
        if name in {"worker", "summarize"}:
            child.add_argument("--output-root", required=True, type=Path)
        if name == "worker":
            child.add_argument("--device-index", type=int, default=0)
        if name == "summarize":
            child.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(args.campaign_root.resolve(),
                           args.artifact_root.resolve())
    elif args.command == "worker":
        result = run_worker(
            args.campaign_root.resolve(), args.artifact_root.resolve(),
            args.output_root.resolve(), args.device_index)
    else:
        result = summarize(
            args.campaign_root.resolve(), args.artifact_root.resolve(),
            args.output_root.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
