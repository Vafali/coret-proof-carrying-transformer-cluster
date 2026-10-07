#!/usr/bin/env python3
"""Two-worker, one-candidate sound-FP64 campaign over frozen 3L properties."""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import resource
import shutil
import statistics
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "research_hab")]

import a40_fresh_common
import cluster_common
import coret_production_prefix_trace_v1 as prefix
import coret_sound_fp64_block0_feasibility_v1 as sound
import run_sound_fp64_finish_3l_v1 as finish3l


SCHEMA = "CORET_SOUND_FP64_3L_FIRST_CAMPAIGN_V1"
RESULT_SCHEMA = "CORET_SOUND_FP64_3L_PROPERTY_RESULT_V1"
AGGREGATE_SCHEMA = "CORET_SOUND_FP64_3L_CAMPAIGN_SUMMARY_V1"
CANDIDATE_RADIUS_FIELD = "certified_lower_endpoint_binary64"
CANDIDATE_RADIUS_HEX_FIELD = "certified_lower_endpoint_binary64_hex"
CANDIDATE_RADIUS_SOURCE = (
    "cached_DeepT_reference.certified_lower_endpoint_binary64"
)
TOKEN_IDENTITY_FIELDS = (
    "property_id", "canonical_binary_test_index", "clean_label",
    "nominal_prediction", "raw_sentence_sha256", "sentence_id",
    "sentence_ordinal", "sequence_length", "source_dimension",
    "source_test_line", "token", "token_id", "token_position",
)


def _atomic_json(path: Path, value: dict) -> None:
    payload = dict(value)
    payload["record_sha256"] = cluster_common.canonical(payload)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _verified_result(path: Path) -> dict:
    result = cluster_common.verified_json(path)
    if result.get("schema") != RESULT_SCHEMA:
        raise RuntimeError(f"campaign result schema differs: {path}")
    if result.get("layernorm_separator_witnesses_sha256"):
        witness_path = path.parent / "layernorm_separator_witnesses.pt"
        if not witness_path.is_file() or cluster_common.sha256(witness_path) != result["layernorm_separator_witnesses_sha256"]:
            raise RuntimeError(f"LayerNorm separator archive identity differs: {path}")
    artifact = path.parent / "certificate.pt"
    report = path.parent / "certificate_report.json"
    if result.get("terminal_status") == "COMPLETE":
        if (not artifact.is_file() or not report.is_file()
                or cluster_common.sha256(artifact)
                != result.get("certificate_sha256")
                or cluster_common.sha256(report)
                != result.get("certificate_report_sha256")):
            raise RuntimeError(f"campaign certificate identity differs: {path}")
        expected = ("CERTIFIED_AT_HISTORICAL_RADIUS"
                    if result.get("certified_at_historical_radius") is True
                    else "FAILED_AT_HISTORICAL_RADIUS")
        if (result.get("scientific_evaluation_complete") is not True
                or result.get("classification") != expected):
            raise RuntimeError(f"completed campaign result status differs: {path}")
    elif result.get("terminal_status") == "UNCERTIFIED_DOMAIN_FAILURE":
        diagnostic = result.get("domain_failure_diagnostic")
        if (result.get("scientific_evaluation_complete") is not True
                or result.get("certified_at_historical_radius") is not False
                or result.get("classification") !=
                "FAILED_AT_HISTORICAL_RADIUS"
                or result.get("failure_category") !=
                "SOUND_LAYERNORM_DOMAIN_FAILURE"
                or result.get("generic_fallback_count") != 0
                or not isinstance(diagnostic, dict)
                or diagnostic.get("reason_code") !=
                finish3l.LAYERNORM_DOMAIN_REASON):
            raise RuntimeError(f"domain-failed campaign result status differs: {path}")
    elif result.get("terminal_status") == "FAIL_CLOSED":
        if (result.get("scientific_evaluation_complete") is not False
                or result.get("certified_at_historical_radius") is not None
                or result.get("classification") != "INFRASTRUCTURE_FAILURE"):
            raise RuntimeError(f"failed campaign result status differs: {path}")
    else:
        raise RuntimeError(f"campaign terminal status differs: {path}")
    return result


def _status_counts(rows: list[dict]) -> dict:
    completed = [row for row in rows
                 if row.get("scientific_evaluation_complete") is True]
    infrastructure = [row for row in rows
                      if row.get("scientific_evaluation_complete") is False]
    certified = sum(
        row.get("certified_at_historical_radius") is True for row in completed)
    not_certified = sum(
        row.get("certified_at_historical_radius") is False for row in completed)
    return {
        "completed_sound_evaluations": len(completed),
        "certified_at_historical_radius": certified,
        "failed_at_historical_radius": not_certified,
        "infrastructure_failures": len(infrastructure),
    }


def _failure_category(error: Exception) -> str:
    name, message = type(error).__name__, str(error).lower()
    if name == "OutOfMemoryError" or "out of memory" in message:
        return "CUDA_OUT_OF_MEMORY"
    if "pre-block-2 hidden-state shape differs" in message:
        return "ADAPTER_STATE_TRANSFER"
    return "RUNTIME_EXCEPTION"


def _cuda_memory_snapshot(device: str) -> dict:
    torch = sound.torch
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return {"allocated": 0, "reserved": 0,
                "max_allocated": 0, "max_reserved": 0}
    target = torch.device(device)
    return {
        "allocated": int(torch.cuda.memory_allocated(target)),
        "reserved": int(torch.cuda.memory_reserved(target)),
        "max_allocated": int(torch.cuda.max_memory_allocated(target)),
        "max_reserved": int(torch.cuda.max_memory_reserved(target)),
    }


def _property_boundary_cleanup(device: str) -> dict:
    """Release only allocator state after an independent property is dead."""
    collected = gc.collect()
    torch = sound.torch
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return {"gc_collected": collected, "cuda_cache_trimmed": False}
    target = torch.device(device)
    with torch.cuda.device(target):
        torch.cuda.synchronize(target)
        before = int(torch.cuda.memory_reserved(target))
        torch.cuda.empty_cache()
        after = int(torch.cuda.memory_reserved(target))
    return {
        "gc_collected": collected, "cuda_cache_trimmed": True,
        "reserved_before": before, "reserved_after": after,
        "reserved_released": max(0, before - after),
    }


def _ordered_assignments(plan: dict, worker_id: int) -> list[dict]:
    rows = [row for chunk in plan["workers"][worker_id]["chunks"]
            for row in chunk["properties"]]
    if len({row["property_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate property in campaign worker assignment")
    return rows


def _candidate_radius(row: dict) -> float:
    reference = row.get("cached_DeepT_reference")
    if not isinstance(reference, dict):
        raise RuntimeError("cached DeepT reference is missing or malformed")
    if CANDIDATE_RADIUS_FIELD not in reference:
        raise RuntimeError(
            "cached DeepT reference is missing authoritative candidate radius "
            f"{CANDIDATE_RADIUS_FIELD}"
        )

    try:
        candidate = float(reference[CANDIDATE_RADIUS_FIELD])
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("cached DeepT candidate radius is not binary64") from exc
    if not math.isfinite(candidate) or candidate < 0.0:
        raise RuntimeError(
            "cached DeepT candidate radius must be finite and nonnegative"
        )

    if CANDIDATE_RADIUS_HEX_FIELD in reference:
        try:
            hexadecimal = float.fromhex(reference[CANDIDATE_RADIUS_HEX_FIELD])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "cached DeepT optional hexadecimal radius is malformed"
            ) from exc
        if hexadecimal != candidate:
            raise RuntimeError(
                "cached DeepT decimal and optional hexadecimal radii differ"
            )

    return candidate


def _load_authoritative_token_source(artifact_root: Path) -> tuple[dict, dict]:
    path = cluster_common.historical_manifest_path(artifact_root)
    manifest = cluster_common.verified_json(
        path, "canonical_manifest_sha256")
    if manifest.get("canonical_manifest_sha256") != \
            cluster_common.SCIENTIFIC_MANIFEST_SHA:
        raise RuntimeError("authoritative token-source identity differs")
    return manifest, {
        "path": str(path),
        "file_sha256": cluster_common.sha256(path),
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        "source": "frozen historical scientific manifest examples[].token_ids",
    }


def _resolve_campaign_input(row: dict, historical: dict,
                            source_identity: dict) -> dict:
    property_id = row.get("property_id")
    properties = [item for item in historical.get("properties", ())
                  if item.get("property_id") == property_id]
    if len(properties) != 1:
        raise RuntimeError(
            f"authoritative property source is absent or ambiguous: {property_id}"
        )
    authoritative_property = properties[0]
    for field in TOKEN_IDENTITY_FIELDS:
        if field not in row or field not in authoritative_property:
            raise RuntimeError(
                f"property identity field is absent: {property_id}: {field}"
            )
        if row[field] != authoritative_property[field]:
            raise RuntimeError(
                f"property identity differs: {property_id}: {field}"
            )

    examples = [item for item in historical.get("examples", ())
                if item.get("sentence_ordinal") == row["sentence_ordinal"]]
    if len(examples) != 1:
        raise RuntimeError(
            f"authoritative token source is absent or ambiguous: {property_id}"
        )
    example = examples[0]
    example_checks = {
        "canonical_binary_test_index": row["canonical_binary_test_index"],
        "clean_label": row["clean_label"],
        "nominal_prediction": row["nominal_prediction"],
        "raw_sentence_sha256": row["raw_sentence_sha256"],
        "sentence_id": row["sentence_id"],
        "sentence_ordinal": row["sentence_ordinal"],
        "sequence_length_including_special_tokens": row["sequence_length"],
        "source_test_line": row["source_test_line"],
    }
    for field, expected in example_checks.items():
        if example.get(field) != expected:
            raise RuntimeError(
                f"token-source property identity differs: {property_id}: {field}"
            )
    if row["clean_label"] != row["nominal_prediction"]:
        raise RuntimeError(f"clean/nominal label differs: {property_id}")

    raw_token_ids = example.get("token_ids")
    if not isinstance(raw_token_ids, list) or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in raw_token_ids):
        raise RuntimeError(f"authoritative token IDs are malformed: {property_id}")
    token_ids = list(raw_token_ids)
    if len(token_ids) != row["sequence_length"]:
        raise RuntimeError(f"authoritative token count differs: {property_id}")
    position = int(row["token_position"])
    tokens = example.get("tokens")
    if (position < 0 or position >= len(token_ids)
            or token_ids[position] != row["token_id"]
            or not isinstance(tokens, list) or len(tokens) != len(token_ids)
            or tokens[position] != row["token"]
            or position not in example.get("eligible_token_positions", ())):
        raise RuntimeError(f"token/property position identity differs: {property_id}")

    embedded = row.get("token_ids")
    if embedded is not None and list(embedded) != token_ids:
        raise RuntimeError(f"embedded and authoritative token IDs differ: {property_id}")

    return {
        **row,
        "token_ids": token_ids,
        "token_input_source": dict(source_identity),
    }


def _resolved_campaign_inputs(artifact_root: Path,
                              manifest: dict) -> tuple[dict[str, dict], dict]:
    historical, source_identity = _load_authoritative_token_source(
        artifact_root)
    resolved = {}
    for row in manifest["properties"]:
        item = _resolve_campaign_input(row, historical, source_identity)
        property_id = item["property_id"]
        if property_id in resolved:
            raise RuntimeError(f"duplicate resolved campaign property: {property_id}")
        resolved[property_id] = item
    return resolved, source_identity


def preflight(artifact_root: Path) -> dict:
    cluster_common.verify_artifact_manifest(artifact_root)
    manifest = cluster_common.load_production_manifest(artifact_root)
    plan = a40_fresh_common.load_plan()
    workers = [_ordered_assignments(plan, worker) for worker in (0, 1)]
    ids = [row["property_id"] for rows in workers for row in rows]
    manifest_ids = [row["property_id"] for row in manifest["properties"]]
    if (len(ids) != 127 or len(set(ids)) != 127
            or set(ids) != set(manifest_ids)):
        raise RuntimeError("campaign split does not cover frozen 127 properties")
    indexed, token_source = _resolved_campaign_inputs(artifact_root, manifest)
    for property_id in ids:
        row = indexed[property_id]
        _candidate_radius(row)
        if (len(row.get("token_ids", ())) != row["sequence_length"]
                or row["clean_label"] != row["nominal_prediction"]):
            raise RuntimeError(f"invalid frozen campaign property: {property_id}")
    return {
        "schema": SCHEMA,
        "property_count": 127,
        "worker_property_counts": [len(rows) for rows in workers],
        "worker_property_ids": [[row["property_id"] for row in rows]
                                for rows in workers],
        "scientific_manifest_sha256":
            cluster_common.SCIENTIFIC_MANIFEST_SHA,
        "production_manifest_sha256":
            cluster_common.PRODUCTION_MANIFEST_SHA,
        "source_plan_sha256": plan["canonical_manifest_sha256"],
        "candidate_radius_source": CANDIDATE_RADIUS_SOURCE,
        "resolved_token_property_count": len(indexed),
        "authoritative_token_source": token_source,
        "scientific_queries": 0,
    }


@contextlib.contextmanager
def _property_source(row: dict, rho: float):
    original = (prefix.FIXTURE_TOKEN_IDS, prefix.FIXTURE_PERTURBED_TOKEN,
                prefix.FIXTURE_RHO)
    prefix.FIXTURE_TOKEN_IDS = tuple(int(item) for item in row["token_ids"])
    prefix.FIXTURE_PERTURBED_TOKEN = int(row["token_position"])
    prefix.FIXTURE_RHO = float(rho)
    try:
        yield
    finally:
        (prefix.FIXTURE_TOKEN_IDS, prefix.FIXTURE_PERTURBED_TOKEN,
         prefix.FIXTURE_RHO) = original


def _block1_report(final_path: Path, stages: list[dict]) -> Path:
    payload = sound._load_artifact(
        final_path, finish3l.INPUT_SCHEMA)
    state = payload["states"]["pre_block2"]
    generators = int(state["weights"].shape[0]) - 1
    final_sha = cluster_common.sha256(final_path)
    report = {
        "schema": finish3l.INPUT_REPORT_SCHEMA,
        "verdict": "CORET_SOUND_FP64_BLOCK1_READY",
        "final_artifact_schema": finish3l.INPUT_SCHEMA,
        "final_artifact_sha256": final_sha,
        "final_generator_count": generators,
        "block2_feasible": True, "qk_recomputed": False,
        "generic_fallback_count": 0,
        "scientific_properties": 0, "bound_calls": 0,
        "stages": stages,
    }
    destination = final_path.parent / "block1_report.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return destination


def _run_stage(name: str, function, inputs: tuple, output: Path,
               schema: str, device: str) -> dict:
    result = function(*inputs, output, device=device)
    if not output.is_file():
        raise RuntimeError(f"{name}: stage artifact missing")
    payload = sound._load_artifact(output, schema)
    report = result.get("report", result)
    if int(report.get("generic_fallback_count", 0)) != 0:
        raise RuntimeError(f"{name}: generic fallback reached")
    result = {
        "name": name, "output_schema": schema,
        "output_sha256": cluster_common.sha256(output),
        "generic_fallback_count": 0,
    }
    if payload.get("separating_variance_witnesses"):
        result["_separating_variance_witnesses"] = payload["separating_variance_witnesses"]
    return result


def execute_property(row: dict, result_root: Path, device: str) -> dict:
    property_id = row["property_id"]
    directory = result_root / "properties" / property_id
    result_path = directory / "result.json"
    if result_path.exists():
        return _verified_result(result_path)
    # A missing atomic result record means any certificate files are from an
    # interrupted, incomplete attempt and must never be treated as complete.
    for name in ("certificate.pt", "certificate_report.json"):
        incomplete = directory / name
        if incomplete.exists():
            incomplete.unlink()
    candidate = _candidate_radius(row)
    workspace = result_root / "work" / property_id
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    stage = "initialization"
    separator_witnesses = []
    started = time.perf_counter()
    try:
        with _property_source(row, candidate):
            paths = [workspace / f"{index:02d}_{name}.pt" for index, name in
                     enumerate(("block0", "pre_qk", "qk_input", "qk",
                                "softmax", "av_input", "av", "post_residual",
                                "post_ln", "ffn1", "ffn2", "residual_input",
                                "residual", "block1_final"))]
            stages = []
            stage = "block0"
            result = sound.export_block0_state(
                paths[0], device=device, run_representative_mpfr=False)
            block0_payload = sound._load_artifact(paths[0], "CORET_SOUND_FP64_BLOCK0_STATE_V1")
            separator_witnesses.extend(block0_payload.get("separating_variance_witnesses", []))
            del block0_payload
            stages.append({
                "name": stage,
                "output_schema": "CORET_SOUND_FP64_BLOCK0_STATE_V1",
                "output_sha256": cluster_common.sha256(paths[0]),
                "generic_fallback_count": 0,
            })
            specs = [
                ("block1_pre_qk", sound.run_block1_pre_qk, (paths[0],),
                 paths[1], "CORET_SOUND_FP64_BLOCK1_PRE_QK_V1"),
                ("block1_qk_prepare", sound.run_block1_qk_prepare,
                 (paths[1],), paths[2],
                 "CORET_SOUND_FP64_BLOCK1_QK_INPUT_V1"),
                ("block1_qk", sound.run_block1_qk_only, (paths[2],),
                 paths[3], "CORET_SOUND_FP64_BLOCK1_QK_V1"),
                ("block1_softmax", sound.run_block1_softmax_only,
                 (paths[3],), paths[4], "CORET_SOUND_FP64_BLOCK1_STAGE1_V1"),
                ("block1_av_prepare", sound.run_block1_stage2_prepare,
                 (paths[4],), paths[5],
                 "CORET_SOUND_FP64_BLOCK1_AV_INPUT_V1"),
                ("block1_av", sound.run_block1_stage2_av_only, (paths[5],),
                 paths[6], "CORET_SOUND_FP64_BLOCK1_STAGE2_AV_V1"),
                ("block1_attention_residual",
                 sound.run_block1_stage2_post_prepare, (paths[6],), paths[7],
                 "CORET_SOUND_FP64_BLOCK1_POST_RESIDUAL_V1"),
                ("block1_post_attention_ln", sound.run_block1_stage2_post_ln,
                 (paths[7],), paths[8],
                 "CORET_SOUND_FP64_BLOCK1_STAGE2_POST_V1"),
                ("block1_ffn_relu", sound.run_block1_stage3_ffn1,
                 (paths[8],), paths[9], "CORET_SOUND_FP64_BLOCK1_FFN1_V1"),
                ("block1_ffn_output", sound.run_block1_stage3_ffn2,
                 (paths[9],), paths[10], "CORET_SOUND_FP64_BLOCK1_FFN2_V1"),
                ("block1_residual_prepare",
                 sound.run_block1_stage3_residual_prepare,
                 (paths[8], paths[10]), paths[11],
                 "CORET_SOUND_FP64_BLOCK1_FFN_RESIDUAL_INPUT_V1"),
                ("block1_residual", sound.run_block1_stage3_residual_add,
                 (paths[11],), paths[12],
                 "CORET_SOUND_FP64_BLOCK1_OUTPUT_RESIDUAL_V1"),
                ("block1_final_ln", sound.run_block1_stage3_final_ln,
                 (paths[12],), paths[13], finish3l.INPUT_SCHEMA),
            ]
            for stage, function, inputs, output, schema in specs:
                stage_result = _run_stage(stage, function, inputs, output, schema, device)
                separator_witnesses.extend(stage_result.pop("_separating_variance_witnesses", []))
                stages.append(stage_result)
            report_path = _block1_report(paths[13], stages)
            stage = "block2_to_margin"
            directory.mkdir(parents=True, exist_ok=True)
            certificate = directory / "certificate.pt"
            certificate_report = directory / "certificate_report.json"
            report = finish3l.execute(
                paths[13], report_path, certificate, certificate_report,
                int(device.split(":")[-1]),
                cluster_common.sha256(paths[13]),
                cluster_common.sha256(report_path),
                run_representative_mpfr=False,
                expected_num_tokens=len(row["token_ids"]))
            separator_witnesses.extend(report.pop("_separating_variance_witnesses", []))
            if certificate.is_file():
                final_payload = sound._load_artifact(certificate, finish3l.OUTPUT_SCHEMA)
                separator_witnesses.extend(final_payload.get("separating_variance_witnesses", []))
                del final_payload
        if (report["clean_label"] != int(row["clean_label"])
                or report["fixture_token_ids"] != row["token_ids"]
                or report["fixture_rho_hex"] != candidate.hex()):
            raise RuntimeError("property/model/source identity differs at output")
        if (report.get("verdict") ==
                "CORET_SOUND_FP64_3L_UNCERTIFIED_DOMAIN_FAILURE"):
            diagnostic = report.get("domain_failure")
            if (not isinstance(diagnostic, dict)
                    or diagnostic.get("reason_code") !=
                    finish3l.LAYERNORM_DOMAIN_REASON
                    or diagnostic.get("domain_admissible") is not False
                    or report.get("generic_fallback_count") != 0):
                raise RuntimeError("LayerNorm domain-failure evidence differs")
            record = {
                "schema": RESULT_SCHEMA,
                "terminal_status": "UNCERTIFIED_DOMAIN_FAILURE",
                "property_id": property_id,
                "benchmark_ordinal": int(row["benchmark_ordinal"]),
                "sentence_ordinal": int(row["sentence_ordinal"]),
                "token_position": int(row["token_position"]),
                "historical_candidate_radius": candidate,
                "historical_candidate_radius_hex": candidate.hex(),
                "candidate_source": CANDIDATE_RADIUS_SOURCE,
                "clean_label": int(row["clean_label"]),
                "target_comparison": [int(row["clean_label"]),
                                      1 - int(row["clean_label"])],
                "final_sound_lower_margin": None,
                "certified_at_historical_radius": False,
                "scientific_evaluation_complete": True,
                "classification": "FAILED_AT_HISTORICAL_RADIUS",
                "failure_category": "SOUND_LAYERNORM_DOMAIN_FAILURE",
                "domain_failure_diagnostic": diagnostic,
                "numerical_widening": None,
                "max_numerical_native_ratio": None,
                "final_generator_count": int(
                    report["final_generator_count"]),
                "runtime_seconds": time.perf_counter() - started,
                "peak_gpu_allocated_bytes": int(
                    report["peak_allocated_bytes"]),
                "peak_gpu_reserved_bytes": int(
                    report["peak_reserved_bytes"]),
                "peak_cpu_rss_bytes": int(resource.getrusage(
                    resource.RUSAGE_SELF).ru_maxrss) * 1024,
                "failure_stage": stage,
                "failure_reason": finish3l.LAYERNORM_DOMAIN_REASON,
                "binary_search_performed": False,
                "verifier_evaluations": 1,
                "generic_fallback_count": 0,
            }
        else:
            lower = float(report["final_sound_margin"])
            record = {
                "schema": RESULT_SCHEMA, "terminal_status": "COMPLETE",
                "property_id": property_id,
                "benchmark_ordinal": int(row["benchmark_ordinal"]),
                "sentence_ordinal": int(row["sentence_ordinal"]),
                "token_position": int(row["token_position"]),
                "historical_candidate_radius": candidate,
                "historical_candidate_radius_hex": candidate.hex(),
                "candidate_source": CANDIDATE_RADIUS_SOURCE,
                "clean_label": int(row["clean_label"]),
                "target_comparison": [int(row["clean_label"]),
                                      1 - int(row["clean_label"])],
                "final_sound_lower_margin": lower,
                "final_sound_upper_margin": float(
                    report["final_sound_margin_upper"]),
                "certified_at_historical_radius": lower > 0,
                "scientific_evaluation_complete": True,
                "classification": ("CERTIFIED_AT_HISTORICAL_RADIUS"
                                   if lower > 0 else
                                   "FAILED_AT_HISTORICAL_RADIUS"),
                "numerical_widening": float(report["numerical_widening"]),
                "max_numerical_native_ratio": float(
                    report["max_numerical_native_ratio"]),
                "final_generator_count": int(report["final_generator_count"]),
                "runtime_seconds": time.perf_counter() - started,
                "peak_gpu_allocated_bytes": int(report["peak_allocated_bytes"]),
                "peak_gpu_reserved_bytes": int(report["peak_reserved_bytes"]),
                "peak_cpu_rss_bytes": int(resource.getrusage(
                    resource.RUSAGE_SELF).ru_maxrss) * 1024,
                "failure_stage": None, "failure_reason": None,
                "certificate_sha256": cluster_common.sha256(certificate),
                "certificate_report_sha256": cluster_common.sha256(
                    certificate_report),
                "binary_search_performed": False,
                "verifier_evaluations": 1,
                "generic_fallback_count": 0,
            }
    except Exception as error:
        memory = _cuda_memory_snapshot(device)
        for name in ("certificate.pt", "certificate_report.json"):
            candidate_path = directory / name
            if candidate_path.exists():
                candidate_path.unlink()
        record = {
            "schema": RESULT_SCHEMA, "terminal_status": "FAIL_CLOSED",
            "property_id": property_id,
            "benchmark_ordinal": int(row["benchmark_ordinal"]),
            "sentence_ordinal": int(row["sentence_ordinal"]),
            "token_position": int(row["token_position"]),
            "historical_candidate_radius": candidate,
            "historical_candidate_radius_hex": candidate.hex(),
            "candidate_source": CANDIDATE_RADIUS_SOURCE,
            "clean_label": int(row["clean_label"]),
            "target_comparison": [int(row["clean_label"]),
                                  1 - int(row["clean_label"])],
            "final_sound_lower_margin": None,
            "certified_at_historical_radius": None,
            "scientific_evaluation_complete": False,
            "classification": "INFRASTRUCTURE_FAILURE",
            "failure_category": _failure_category(error),
            "domain_failure_diagnostic": None,
            "numerical_widening": None,
            "max_numerical_native_ratio": None,
            "final_generator_count": None,
            "runtime_seconds": time.perf_counter() - started,
            "peak_gpu_allocated_bytes": memory["max_allocated"],
            "peak_gpu_reserved_bytes": memory["max_reserved"],
            "failure_gpu_allocated_bytes": memory["allocated"],
            "failure_gpu_reserved_bytes": memory["reserved"],
            "peak_cpu_rss_bytes": int(resource.getrusage(
                resource.RUSAGE_SELF).ru_maxrss) * 1024,
            "failure_stage": stage,
            "failure_reason": f"{type(error).__name__}: {error}",
            "binary_search_performed": False,
            "verifier_evaluations": 1,
            "generic_fallback_count": None,
        }
    if separator_witnesses:
        witness_path = directory / "layernorm_separator_witnesses.pt"
        directory.mkdir(parents=True, exist_ok=True)
        import sound_fp64_layernorm_separator_v1 as separator
        for payload in separator_witnesses:
            if payload.get("path") == "semantic_epsilon_floor":
                import semantic_epsilon_floor_checker_v1 as floor_checker
                floor_checker.replay_persisted(
                    payload, payload["source_state_identity"], payload["parameter_hashes"],
                    payload["source_domain"], payload["output_state_identity"])
            else:
                separator.replay_payload(payload)
        temporary = witness_path.with_suffix(".tmp.pt")
        domain_schema = ("CORET_SOUND_FP64_LAYERNORM_DOMAIN_ARCHIVE_V2"
                         if any(w.get("path") == "semantic_epsilon_floor" for w in separator_witnesses)
                         else "CORET_SOUND_FP64_LAYERNORM_SEPARATOR_ARCHIVE_V1")
        sound.torch.save({"schema": domain_schema,
                          "property_id": property_id, "candidate_radius_hex": candidate.hex(),
                          "witnesses": separator_witnesses}, temporary)
        os.replace(temporary, witness_path)
        record["layernorm_separator_witnesses_path"] = str(witness_path)
        record["layernorm_separator_witnesses_sha256"] = cluster_common.sha256(witness_path)
        record["layernorm_separator_repaired_tokens"] = [
            {"label": w["label"], "tokens": w["failed_tokens"],
             "lower_by_token": w["semantic_lower_by_token"],
             **({"path": "semantic_epsilon_floor"} if w.get("path") == "semantic_epsilon_floor" else {})}
            for w in separator_witnesses]
    _atomic_json(result_path, record)
    shutil.rmtree(workspace, ignore_errors=True)
    return _verified_result(result_path)


def run_worker(worker_id: int, artifact_root: Path, result_root: Path,
               device_index: int) -> dict:
    identity = preflight(artifact_root)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("campaign requires exactly one visible A40 GPU")
    manifest = cluster_common.load_production_manifest(artifact_root)
    properties, _ = _resolved_campaign_inputs(artifact_root, manifest)
    plan = a40_fresh_common.load_plan()
    assignments = _ordered_assignments(plan, worker_id)
    started = time.perf_counter()
    completed = []
    for assigned in assignments:
        row = properties[assigned["property_id"]]
        device = f"cuda:{device_index}"
        _property_boundary_cleanup(device)
        try:
            completed.append(execute_property(row, result_root, device))
        finally:
            _property_boundary_cleanup(device)
    counts = _status_counts(completed)
    summary = {
        "schema": SCHEMA, "worker_id": worker_id,
        "source_plan_sha256": identity["source_plan_sha256"],
        "property_count": len(assignments),
        "persisted_result_count": len(completed),
        **counts,
        "wall_seconds": time.perf_counter() - started,
    }
    _atomic_json(result_root / f"worker_{worker_id}_summary.json", summary)
    return summary


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def aggregate(artifact_root: Path, worker_roots: list[Path],
              output: Path) -> dict:
    identity = preflight(artifact_root)
    results = {}
    wall = []
    for worker, root in enumerate(worker_roots):
        summary = cluster_common.verified_json(
            root / f"worker_{worker}_summary.json")
        wall.append(float(summary["wall_seconds"]))
        for path in (root / "properties").glob("*/result.json"):
            row = _verified_result(path)
            if row["property_id"] in results:
                raise RuntimeError("duplicate campaign property result")
            results[row["property_id"]] = row
    expected = set(sum(identity["worker_property_ids"], []))
    if set(results) != expected or len(results) != 127:
        raise RuntimeError("campaign result set is incomplete")
    rows = [results[property_id]
            for property_id in sum(identity["worker_property_ids"], [])]
    runtimes = [float(row["runtime_seconds"]) for row in rows]
    ratios = [float(row["max_numerical_native_ratio"]) for row in rows
              if row["max_numerical_native_ratio"] is not None]
    margins = [float(row["final_sound_lower_margin"]) for row in rows
               if row["final_sound_lower_margin"] is not None]
    counts = _status_counts(rows)
    completed_count = counts["completed_sound_evaluations"]
    summary = {
        "schema": AGGREGATE_SCHEMA,
        "total_properties": 127,
        **counts,
        "campaign_status": ("COMPLETE" if not counts["infrastructure_failures"]
                            else "INCOMPLETE_INFRASTRUCTURE_FAILURES"),
        "certification_rate_among_completed": (
            counts["certified_at_historical_radius"] / completed_count
            if completed_count else None),
        "runtime_seconds": {
            "mean": statistics.fmean(runtimes),
            "median": statistics.median(runtimes),
            "p95": _percentile(runtimes, 0.95),
        },
        "numerical_native_ratio": ({
            "mean": statistics.fmean(ratios),
            "median": statistics.median(ratios), "max": max(ratios),
        } if ratios else None),
        "final_sound_margin": ({
            "min": min(margins), "median": statistics.median(margins),
            "mean": statistics.fmean(margins), "max": max(margins),
        } if margins else None),
        "total_gpu_worker_seconds": sum(wall),
        "total_wall_seconds": max(wall),
        "historical_radii_are_candidates_not_coret_maxima": True,
        "results": rows,
    }
    _atomic_json(output, summary)
    return cluster_common.verified_json(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--artifact-root", required=True, type=Path)
    worker = sub.add_parser("worker")
    worker.add_argument("--worker-id", required=True, type=int, choices=(0, 1))
    worker.add_argument("--artifact-root", required=True, type=Path)
    worker.add_argument("--result-root", required=True, type=Path)
    worker.add_argument("--device-index", type=int, default=0)
    combine = sub.add_parser("aggregate")
    combine.add_argument("--artifact-root", required=True, type=Path)
    combine.add_argument("--worker-root", required=True, action="append",
                         type=Path)
    combine.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(args.artifact_root.resolve())
    elif args.command == "worker":
        result = run_worker(
            args.worker_id, args.artifact_root.resolve(),
            args.result_root.resolve(), args.device_index)
    else:
        if len(args.worker_root) != 2:
            parser.error("aggregate requires exactly two --worker-root values")
        result = aggregate(
            args.artifact_root.resolve(),
            [path.resolve() for path in args.worker_root],
            args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
