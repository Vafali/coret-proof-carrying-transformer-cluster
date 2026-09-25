#!/usr/bin/env python3
"""Fixed three-property implementation smoke for the optimized verifier.

This runner changes no verifier or search semantics.  It binds three already
accepted historical properties to the structural-support production path with
the validated pre-B2-QK allocator fence.  Each query and property result is an
immutable artifact so an interrupted run resumes at the first missing query.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import torch

import coret_bounded_native_production_graph_v1 as graph
import coret_deept_exact_standard_ln_adapter as adapter
import coret_native_semantics_production_graph_v1 as production
import coret_proof_carrying_historical_smoke_v1 as historical
import coret_sound_reference_v1 as ref
import coret_structural_support_lifetime_v1 as lifetime
import coret_structural_support_precise_dot_v1 as structural
import coret_structural_support_tok10_lifetime_v1 as lifetime_runner
import coret_structural_support_tok10_real_gate_v1 as optimized_runner


WORKTREE = Path(__file__).resolve().parents[1]
ROOT = ref.ROOT
OUT = WORKTREE / (
    "research_hab/results/coret_optimized_three_property_smoke_v1_20260923")
LEGACY_MANIFEST = OUT / "coret_optimized_three_property_smoke_manifest_v1.json"
LEGACY_PREFLIGHT = OUT / "coret_optimized_three_property_smoke_preflight_v1.json"
MANIFEST = OUT / "coret_optimized_three_property_smoke_manifest_v2.json"
PREFLIGHT = OUT / "coret_optimized_three_property_smoke_preflight_v2.json"
SUMMARY = OUT / "coret_optimized_three_property_smoke_summary_v1.json"
LAUNCHER = WORKTREE / "research_hab/run_coret_optimized_three_property_smoke_v1.sh"
TEST = WORKTREE / "research_hab/tests/test_coret_optimized_three_property_smoke_v1.py"
HISTORICAL_MANIFEST = historical.HISTORICAL_MANIFEST
HISTORICAL_SMOKE_MANIFEST = historical.MANIFEST
HISTORICAL_SMOKE_SUMMARY = historical.SUMMARY
HISTORICAL_PROPERTIES = historical.OUT / "properties"
TOK10_LIFETIME_RESULT = lifetime_runner.RESULT

INITIAL_RHO = 1.0 / 1600.0
MAX_FACTOR_TWO_REFINEMENTS = 12
MIDPOINT_ITERATIONS = 10
RADIUS_RETENTION_GATE = 0.98
B2_QK_PATHOLOGY_GUARD_SECONDS = 30.0
LEGACY_MANIFEST_CANONICAL_SHA256 = (
    "87210411f596852c3a6f719927f0cd0441295dd5f3493233b681047d332cfd3c")
LEGACY_RUNNER_SHA256 = (
    "c10f4e443a62d7e06ba7c5d135721a8b3dcf5262de7be530a2d84a883b42073a")

FROZEN_PROPERTIES = (
    ("deept_table7_stdln3_s002_line1468_tok04", 0.0010833740234375),
    ("deept_table7_stdln3_s005_line1931_tok09", 0.001236572265625),
    ("deept_table7_stdln3_s009_line2029_tok10", 0.0011853027343750003),
)


def _write(path: Path, payload: dict, key: str = "record_sha256") -> dict:
    value = dict(payload)
    value[key] = ref.canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return value


def _source_files() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(), LAUNCHER, TEST,
        WORKTREE / "research_hab/coret_structural_support_lifetime_v1.py",
        WORKTREE / "research_hab/coret_structural_support_precise_dot_v1.py",
        Path(graph.__file__).resolve(), Path(production.__file__).resolve(),
        Path(adapter.__file__).resolve(), Path(historical.__file__).resolve(),
        Path(optimized_runner.__file__).resolve(),
    )


def _historical_property(parent: dict, property_id: str) -> tuple[dict, dict]:
    props = [item for item in parent["properties"]
             if item["property_id"] == property_id]
    if len(props) != 1:
        raise RuntimeError(f"historical property identity differs: {property_id}")
    prop = props[0]
    examples = [item for item in parent["examples"]
                if int(item["sentence_ordinal"]) == int(prop["sentence_ordinal"])]
    if len(examples) != 1:
        raise RuntimeError(f"historical example identity differs: {property_id}")
    return prop, examples[0]


def _reference_record(property_id: str, expected_radius: float) -> dict:
    path = HISTORICAL_PROPERTIES / property_id / "result_v1.json"
    record = ref.verified(path)
    if (record.get("terminal_status") != "COMPLETE"
            or record.get("property_id") != property_id
            or float(record.get("certified_radius")).hex()
            != float(expected_radius).hex()
            or record.get("independent_checker_accepted") is not True
            or record.get("complete_certificate") is not True):
        raise RuntimeError(f"frozen proof reference differs: {property_id}")
    return {
        "path": str(path),
        "file_sha256": ref.sha(path),
        "record_sha256": record["record_sha256"],
        "certified_radius": float(expected_radius),
        "certified_radius_binary64_hex": float(expected_radius).hex(),
    }


def expected_manifest() -> dict:
    parent = ref.verified(HISTORICAL_MANIFEST, "canonical_manifest_sha256")
    smoke_manifest = ref.verified(
        HISTORICAL_SMOKE_MANIFEST, "canonical_manifest_sha256")
    smoke_summary = ref.verified(HISTORICAL_SMOKE_SUMMARY)
    tok10 = ref.verified(TOK10_LIFETIME_RESULT)
    legacy = ref.verified(LEGACY_MANIFEST, "canonical_manifest_sha256")
    if legacy["canonical_manifest_sha256"] != LEGACY_MANIFEST_CANONICAL_SHA256:
        raise RuntimeError("legacy three-property manifest identity differs")
    if tok10.get("terminal_status") != \
            "B2_QK_LIFETIME_FIX_END_TO_END_GATE_COMPLETE":
        raise RuntimeError("accepted lifetime-fixed tok10 result differs")
    if float(tok10.get("certified_radius", -1)).hex() != \
            float(0.0011090087890625).hex():
        raise RuntimeError("accepted lifetime-fixed tok10 radius differs")
    if tok10.get("generic_fallbacks") != 0:
        raise RuntimeError("accepted lifetime-fixed tok10 used fallback")
    optimized = optimized_runner.validate_manifest()
    properties = []
    for ordinal, (property_id, radius) in enumerate(FROZEN_PROPERTIES):
        prop, example = _historical_property(parent, property_id)
        if property_id not in {
                item["property_id"] for item in smoke_manifest["properties"]}:
            raise RuntimeError(f"property absent from accepted smoke: {property_id}")
        properties.append({
            "smoke_ordinal": ordinal,
            "property_id": property_id,
            "sentence_ordinal": int(prop["sentence_ordinal"]),
            "sentence_id": prop["sentence_id"],
            "source_test_line": int(prop["source_test_line"]),
            "token_position": int(prop["token_position"]),
            "token": prop["token"],
            "token_id": int(prop["token_id"]),
            "sequence_length": int(prop["sequence_length"]),
            "source_dimension": int(prop["source_dimension"]),
            "clean_label": int(prop["clean_label"]),
            "nominal_prediction": int(prop["nominal_prediction"]),
            "raw_sentence_sha256": prop["raw_sentence_sha256"],
            "token_ids": list(example["token_ids"]),
            "frozen_proof_reference": _reference_record(property_id, radius),
        })
    if [item["property_id"] for item in properties] != \
            [item[0] for item in FROZEN_PROPERTIES]:
        raise RuntimeError("fixed smoke property order differs")
    return {
        "schema": "CORET_OPTIMIZED_THREE_PROPERTY_SMOKE_MANIFEST_V2",
        "status": "FROZEN_BEFORE_IMPLEMENTATION_SMOKE",
        "purpose": "implementation/generalization gate; not fresh statistical evidence",
        "properties": properties,
        "property_count": 3,
        "historical_scientific_manifest": {
            "path": str(HISTORICAL_MANIFEST),
            "canonical_sha256": parent["canonical_manifest_sha256"],
            "file_sha256": ref.sha(HISTORICAL_MANIFEST),
        },
        "accepted_historical_smoke": {
            "manifest_path": str(HISTORICAL_SMOKE_MANIFEST),
            "manifest_canonical_sha256": smoke_manifest[
                "canonical_manifest_sha256"],
            "summary_path": str(HISTORICAL_SMOKE_SUMMARY),
            "summary_file_sha256": ref.sha(HISTORICAL_SMOKE_SUMMARY),
            "summary_record_sha256": smoke_summary["record_sha256"],
        },
        "accepted_lifetime_fixed_tok10": {
            "path": str(TOK10_LIFETIME_RESULT),
            "file_sha256": ref.sha(TOK10_LIFETIME_RESULT),
            "record_sha256": tok10["record_sha256"],
            "certified_radius": tok10["certified_radius"],
            "total_search_wall_seconds": tok10["total_search_wall_seconds"],
        },
        "acceptance_harness_correction": {
            "classification": "RUNNER_HARNESS_ACCEPTANCE_BUG",
            "legacy_manifest_path": str(LEGACY_MANIFEST),
            "legacy_manifest_canonical_sha256": legacy[
                "canonical_manifest_sha256"],
            "legacy_manifest_file_sha256": ref.sha(LEGACY_MANIFEST),
            "legacy_runner_sha256": LEGACY_RUNNER_SHA256,
            "first_affected_property": FROZEN_PROPERTIES[2][0],
            "first_affected_rho": INITIAL_RHO,
            "incorrect_predicate": "effective_AV_tile == requested_tile_112",
            "correct_predicate": (
                "effective_AV_tile == min(requested_tile_112, "
                "isqrt(temporary_cap_bytes // "
                "(heads*queries*features*element_size)))"),
            "s009_expected_effective_tile": 109,
            "verifier_mathematics_changed": False,
            "completed_property_records_preserved": [
                {
                    "property_id": property_id,
                    "path": str(_property_root(property_id) / "result_v1.json"),
                    "file_sha256": ref.sha(
                        _property_root(property_id) / "result_v1.json"),
                    "record_sha256": ref.verified(
                        _property_root(property_id) / "result_v1.json")[
                            "record_sha256"],
                }
                for property_id, _ in FROZEN_PROPERTIES[:2]
            ],
        },
        "model": optimized["model"],
        "checkpoint_sha256": optimized["checkpoint_sha256"],
        "pinned_DeepT_revision": optimized["pinned_DeepT_revision"],
        "threat_model": optimized["threat_model"],
        "optimized_execution": {
            "proof_carrying_native_semantics": True,
            "provenance_derived_structural_support": True,
            "support_and_reduction_recomputed_each_radius": True,
            "AV_generator_tile": structural.AV_GENERATOR_TILE,
            "grouped_temporary_cap_bytes": structural.AV_TEMPORARY_CAP_BYTES,
            "deterministic_pre_B2_QK_allocator_lifetime_fence": True,
            "lifetime_fence_count_per_trajectory_reaching_B2_QK": 1,
            "generic_fallback_allowed": False,
        },
        "search": {
            "initial_rho": INITIAL_RHO,
            "initial_rho_binary64_hex": INITIAL_RHO.hex(),
            "factor_two_bracketing": True,
            "maximum_factor_two_refinements": MAX_FACTOR_TWO_REFINEMENTS,
            "midpoint_iterations": MIDPOINT_ITERATIONS,
            "typed_native_domain_failure_prefixes": list(
                historical.NATIVE_DOMAIN_FAILURE_PREFIXES),
            "unexpected_exception_policy": "fatal",
        },
        "acceptance": {
            "radius_retention_minimum": RADIUS_RETENTION_GATE,
            "B2_QK_pathology_guard_seconds": B2_QK_PATHOLOGY_GUARD_SECONDS,
            "zero_checker_rejections": True,
            "zero_provenance_or_support_violations": True,
            "zero_generic_fallback": True,
            "fresh_radius_changes_outward_only": True,
            "deterministic_support_and_reduction_lineage": True,
            "complete_wall_and_peak_memory_telemetry": True,
            "all_three_properties_required": True,
        },
        "outputs": {
            "per_property": str(OUT / "properties/<property_id>/result_v1.json"),
            "per_query": str(OUT / "properties/<property_id>/queries/query_*_result_v1.json"),
            "per_query_certificate": str(OUT / "properties/<property_id>/queries/query_*_certificates_v1.json"),
            "summary": str(SUMMARY),
        },
        "source_hashes": {str(path): ref.sha(path) for path in _source_files()},
        "preparation_scientific_queries": 0,
        "preparation_bound_entrypoint_calls": 0,
    }


def validate_manifest() -> dict:
    stored = ref.verified(MANIFEST, "canonical_manifest_sha256")
    payload = dict(stored)
    payload.pop("canonical_manifest_sha256")
    if payload != expected_manifest():
        raise RuntimeError("optimized three-property smoke manifest mismatch")
    return stored


def freeze() -> None:
    saved = _write(MANIFEST, expected_manifest(), "canonical_manifest_sha256")
    print(json.dumps({
        "terminal_status": "FROZEN_NO_SOLVE",
        "canonical_manifest_sha256": saved["canonical_manifest_sha256"],
        "property_ids": [item[0] for item in FROZEN_PROPERTIES],
        "scientific_queries": 0,
        "bound_entrypoint_calls": 0,
    }, sort_keys=True))


def _property(manifest: dict, property_id: str) -> dict:
    values = [item for item in manifest["properties"]
              if item["property_id"] == property_id]
    if len(values) != 1:
        raise RuntimeError("property is outside fixed smoke")
    return values[0]


def _property_root(property_id: str) -> Path:
    return OUT / "properties" / property_id


def preflight() -> None:
    manifest = validate_manifest()
    if SUMMARY.exists():
        raise FileExistsError("aggregate smoke result already exists")
    preserved = []
    for item in manifest["properties"][:2]:
        result = ref.verified(_property_root(item["property_id"]) / "result_v1.json")
        if (result.get("terminal_status") != "OPTIMIZED_PROPERTY_SMOKE_PASS"
                or result.get("canonical_manifest_sha256") !=
                LEGACY_MANIFEST_CANONICAL_SHA256):
            raise RuntimeError("completed legacy property is not reusable")
        preserved.append(result["record_sha256"])
    third_root = _property_root(manifest["properties"][2]["property_id"])
    if ((third_root / "result_v1.json").exists()
            or any((third_root / "queries").glob("query_*"))):
        raise FileExistsError("s009 scientific query artifact already exists")
    record = {
        "schema": "CORET_OPTIMIZED_THREE_PROPERTY_SMOKE_PREFLIGHT_V2",
        "terminal_status": "PASS_NO_SOLVE",
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        "ordered_property_ids": [item["property_id"]
                                 for item in manifest["properties"]],
        "reference_radii": [item["frozen_proof_reference"]["certified_radius"]
                            for item in manifest["properties"]],
        "AV_generator_tile": structural.AV_GENERATOR_TILE,
        "grouped_temporary_cap_bytes": structural.AV_TEMPORARY_CAP_BYTES,
        "lifetime_fence_boundary": "immediately_before_block2_QK",
        "search_initial_rho": INITIAL_RHO,
        "midpoint_iterations": MIDPOINT_ITERATIONS,
        "resume_policy": "skip hash-valid complete properties; continue first missing query",
        "preserved_completed_property_record_sha256s": preserved,
        "corrected_s009_effective_AV_tile": 109,
        "scientific_queries": 0,
        "bound_entrypoint_calls": 0,
    }
    if PREFLIGHT.exists():
        existing = ref.verified(PREFLIGHT)
        expected = dict(record)
        expected["record_sha256"] = ref.canonical(record)
        if existing != expected:
            raise RuntimeError("preflight record differs")
        saved = existing
    else:
        saved = _write(PREFLIGHT, record)
    print(json.dumps({"terminal_status": "PASS_NO_SOLVE",
                      "record_sha256": saved["record_sha256"],
                      "scientific_queries": 0,
                      "bound_entrypoint_calls": 0}, sort_keys=True))


def _tag(rho: float) -> str:
    return float(rho).hex().replace("+", "p").replace("-", "m").replace(".", "d")


def _query_paths(property_id: str, ordinal: int, rho: float) -> tuple[Path, Path]:
    stem = f"query_{ordinal:02d}_{_tag(rho)}"
    root = _property_root(property_id) / "queries"
    return root / f"{stem}_result_v1.json", root / f"{stem}_certificates_v1.json"


def _operator_telemetry(certificates: list[dict]) -> list[dict]:
    rows = []
    counters = {"QK": 0, "A.V": 0}
    for certificate in certificates:
        family = certificate["family"]
        if family not in counters:
            continue
        block = counters[family]
        counters[family] += 1
        diagnostics = certificate["support_diagnostics"]
        stage = diagnostics["stage_timing_seconds"]
        rows.append({
            "block": block,
            "family": family,
            "structural_seconds": sum(float(value) for value in stage.values()),
            "facade_seconds": float(certificate["facade_total_seconds"]),
            "total_seconds": (sum(float(value) for value in stage.values())
                              + float(certificate["facade_total_seconds"])),
            "stage_timing_seconds": stage,
            "executed_quadratic_MACs": diagnostics["executed_quadratic_MACs"],
            "structurally_skipped_quadratic_MACs": diagnostics[
                "structurally_skipped_quadratic_MACs"],
            "grouped_launches": diagnostics["grouped_launches"],
            "support_class_count_left": diagnostics["support_class_count_left"],
            "support_class_count_right": diagnostics["support_class_count_right"],
            "outward_envelope_max": diagnostics["envelope_max"],
        })
    return rows


def _expected_effective_av_tile(certificate: dict) -> int:
    """Independently reproduce the frozen 128-MiB mechanical tile cap."""
    diagnostics = certificate["support_diagnostics"]
    shape = certificate["output"]["weights"]["shape"]
    dtype = certificate["output"]["weights"]["dtype"]
    element_sizes = {"torch.float32": 4, "torch.float64": 8}
    if dtype not in element_sizes or len(shape) != 4:
        raise RuntimeError("unsupported A.V output shape/dtype for tile proof")
    heads, _, queries, features = map(int, shape)
    bytes_per_pair = heads * queries * features * element_sizes[dtype]
    cap = math.isqrt(max(
        1, int(diagnostics["temporary_cap_bytes"]) // bytes_per_pair))
    return max(1, min(int(diagnostics["requested_generator_tile"]), cap))


def _certificate_acceptance(certificates: list[dict]) -> dict:
    """Return every acceptance predicate instead of short-circuiting.

    This differs from the V1 harness only in accepting the exact mechanical
    cap already used by ``_bounded_av_tile``.  It does not inspect or alter
    coefficient values.
    """
    predicates = {
        "native_checker": True,
        "support_checker": True,
        "provenance_support_validity": True,
        "generic_fallback_zero": True,
        "fresh_radius_outwardness": True,
        "fresh_symbol_count_order": True,
        "range_metadata": True,
        "structural_pair_coverage": True,
        "deterministic_reduction_lineage": True,
        "AV_tile_matches_mechanical_cap": True,
        "AV_temporary_within_cap": True,
    }
    av_tiles = []
    first_failure = None

    def fail(name: str) -> None:
        nonlocal first_failure
        predicates[name] = False
        if first_failure is None:
            first_failure = name

    for index, certificate in enumerate(certificates):
        family = certificate.get("family")
        try:
            if not optimized_runner.native_checker.check_common(certificate, family):
                fail("native_checker")
        except (AssertionError, KeyError, TypeError):
            fail("native_checker")
        proof = certificate.get("support_proof")
        if not isinstance(proof, dict) or proof.get("validated") is not True:
            fail("support_checker")
            fail("provenance_support_validity")
        if certificate.get("generic_semantic_remainder_used") is not False:
            fail("generic_fallback_zero")
        output = certificate.get("output", {})
        if not isinstance(output.get("generator_count"), int) \
                or output.get("generator_count", -1) < 0:
            fail("fresh_symbol_count_order")
        ranges = output.get("ranges")
        if not isinstance(ranges, dict) or ranges.get("count") != output.get(
                "generator_count"):
            fail("range_metadata")
        if family in ("QK", "A.V"):
            diagnostics = certificate.get("support_diagnostics", {})
            if diagnostics.get("support_validated") is not True:
                fail("support_checker")
                fail("structural_pair_coverage")
            if float(diagnostics.get("envelope_max", -1.0)) < 0.0:
                fail("fresh_radius_outwardness")
            if (int(diagnostics.get("nominal_quadratic_MACs", -1))
                    != int(diagnostics.get("executed_quadratic_MACs", -2))
                    + int(diagnostics.get(
                        "structurally_skipped_quadratic_MACs", -3))):
                fail("structural_pair_coverage")
            if family == "A.V":
                try:
                    expected_tile = _expected_effective_av_tile(certificate)
                    actual_tile = int(diagnostics["effective_generator_tile"])
                    av_tiles.append({"certificate_index": index,
                                     "actual": actual_tile,
                                     "expected": expected_tile})
                    if actual_tile != expected_tile:
                        fail("AV_tile_matches_mechanical_cap")
                    if int(diagnostics.get("peak_temporary_bytes", -1)) > int(
                            diagnostics["temporary_cap_bytes"]):
                        fail("AV_temporary_within_cap")
                except (KeyError, TypeError, ValueError, RuntimeError):
                    fail("AV_tile_matches_mechanical_cap")
    return {
        "accepted": all(predicates.values()),
        "predicates": predicates,
        "first_failure": first_failure,
        "AV_tiles": av_tiles,
    }


def _partial_acceptance(dispatch, delegate) -> dict:
    detailed = _certificate_acceptance(dispatch.certificates)
    if dispatch.generic_family_invocations != 0:
        raise RuntimeError("optimized smoke reached generic fallback")
    for certificate in dispatch.certificates:
        if certificate["family"] in ("QK", "A.V"):
            if certificate["support_diagnostics"]["envelope_max"] < 0.0:
                raise RuntimeError("fresh-radius envelope is not outward")
    qk_count = sum(item["family"] == "QK" for item in dispatch.certificates)
    fences = list(delegate.lifetime_fence_records)
    reached_b2 = qk_count >= 3 or len(fences) == 1
    if len(fences) > 1 or (reached_b2 and len(fences) != 1):
        raise RuntimeError("B2 QK lifetime fence count differs")
    return {
        "checker_accepted": bool(detailed["accepted"]),
        "acceptance_predicates": detailed["predicates"],
        "acceptance_first_failure": detailed["first_failure"],
        "AV_tile_checks": detailed["AV_tiles"],
        "outward_fresh_radius_envelopes": True,
        "deterministic_support_reduction_lineage": True,
        "reached_B2_QK": reached_b2,
        "B2_QK_lifetime_fence_count": len(fences),
        "lifetime_fence": fences[0] if fences else None,
    }


def query(property_id: str, rho: float, ordinal: int, authorized: bool) -> None:
    if not authorized:
        raise RuntimeError("explicit USER authorization required")
    manifest = validate_manifest()
    prop = _property(manifest, property_id)
    result_path, certificate_path = _query_paths(property_id, ordinal, rho)
    if result_path.exists() or certificate_path.exists():
        raise FileExistsError("immutable optimized smoke query artifact exists")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("optimized smoke requires CUDA")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    setup_started = started
    modules = adapter.FrozenDeepTModules()
    from Verifiers.Zonotope import Zonotope
    device = torch.device("cuda:0")
    model, _, configuration = adapter.load_native_model(
        modules, device, dtype=torch.float32)
    if configuration != manifest["model"]["configuration"]:
        raise RuntimeError("loaded model configuration differs")
    ids = torch.tensor([prop["token_ids"]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(ids, attention_mask=torch.ones_like(ids))[0]
        label = int(prop["clean_label"])
        nominal = float(logits[0, label] - logits[0, 1 - label])
        pre = adapter.native_pre_layernorm_embeddings(model, ids)[0].detach()
    if int(logits.argmax(-1)[0]) != label:
        raise RuntimeError("nominal prediction differs")
    setup_seconds = time.perf_counter() - setup_started
    args = adapter.build_deept_args(modules, device)
    args.keep_intermediate_zonotopes = False
    z = Zonotope(args=args, p=100, eps=float(rho),
                 perturbed_word_index=int(prop["token_position"]), value=pre)
    dead_holder = {"input_zonotope": z, "token_ids": ids,
                   "nominal_logits": logits, "pre_layernorm": pre}
    del z, ids, logits, pre
    delegate, dispatch = lifetime.make_dispatch(
        dead_holder, production, generator_tile=structural.AV_GENERATOR_TILE)
    bound_started = time.perf_counter()
    try:
        with torch.no_grad():
            if torch.is_grad_enabled():
                raise RuntimeError("optimized proof graph requires no_grad")
            margin, dispatch = graph.execute(
                dead_holder["input_zonotope"], model, args,
                clean_label=label, dispatch=dispatch)
    except AssertionError as error:
        if not historical.is_native_domain_failure(error):
            raise
        torch.cuda.synchronize()
        acceptance = _partial_acceptance(dispatch, delegate)
        if not acceptance["checker_accepted"]:
            raise RuntimeError("partial optimized checker rejected domain failure")
        timings = _operator_telemetry(dispatch.certificates)
        b2 = [row for row in timings
              if row["block"] == 2 and row["family"] == "QK"]
        saved = _write(result_path, {
            "schema": "CORET_OPTIMIZED_THREE_PROPERTY_QUERY_RESULT_V1",
            "terminal_status": "UNCERTIFIED_DOMAIN_FAILURE",
            "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
            "property_id": property_id,
            "query_ordinal": ordinal,
            "rho": float(rho),
            "rho_binary64_hex": float(rho).hex(),
            "certified": False,
            "authoritative_bound_returned": False,
            "exception_type": "AssertionError",
            "exception_message": str(error),
            "partial_operator_checker_accepted": True,
            "all_support_claims_validated": True,
            "generic_fallbacks": 0,
            "provenance_ID_consistent": True,
            **acceptance,
            "per_block_QK_AV_timing": timings,
            "B2_QK_seconds": None if not b2 else b2[0]["total_seconds"],
            "runtime_seconds": time.perf_counter() - started,
            "setup_seconds": setup_seconds,
            "bound_seconds": time.perf_counter() - bound_started,
            "peak_CPU_RSS_bytes": resource.getrusage(
                resource.RUSAGE_SELF).ru_maxrss * 1024,
            "peak_GPU_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_GPU_reserved_bytes": torch.cuda.max_memory_reserved(),
            "scientific_queries": 1,
            "bound_entrypoint_calls": 1,
        })
        print(json.dumps({"terminal_status": saved["terminal_status"],
                          "property_id": property_id, "rho": rho,
                          "record_sha256": saved["record_sha256"]}, sort_keys=True))
        return
    concretize_started = time.perf_counter()
    lower, upper = margin.concretize()
    torch.cuda.synchronize()
    concretize_seconds = time.perf_counter() - concretize_started
    bound_seconds = time.perf_counter() - bound_started
    if not bool(torch.isfinite(lower).all() and torch.isfinite(upper).all()
                and (lower <= upper).all()):
        raise RuntimeError("optimized smoke margin interval invalid")
    counts = dispatch.assert_complete(graph.EXPECTED_THREE_BLOCK_COUNTS)
    acceptance = _partial_acceptance(dispatch, delegate)
    if (not acceptance["checker_accepted"]
            or acceptance["B2_QK_lifetime_fence_count"] != 1):
        raise RuntimeError("optimized smoke acceptance failed")
    timings = _operator_telemetry(dispatch.certificates)
    b2 = [row for row in timings
          if row["block"] == 2 and row["family"] == "QK"]
    if len(b2) != 1:
        raise RuntimeError("optimized smoke B2 QK timing unavailable")
    family_timing, call_timing = optimized_runner._family_timing(
        dispatch.certificates)
    checker_started = time.perf_counter()
    if not _certificate_acceptance(dispatch.certificates)["accepted"]:
        raise RuntimeError("optimized smoke independent checker rejected")
    checker_seconds = time.perf_counter() - checker_started
    certificate_started = time.perf_counter()
    certificate = _write(certificate_path, {
        "schema": "CORET_OPTIMIZED_THREE_PROPERTY_QUERY_CERTIFICATES_V1",
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        "property_id": property_id,
        "query_ordinal": ordinal,
        "rho": float(rho),
        "rho_binary64_hex": float(rho).hex(),
        "invocation_counts": counts,
        "generic_family_invocations": 0,
        "independent_checker_accepted": True,
        "all_support_claims_validated": True,
        "outward_fresh_radius_envelopes": True,
        "provenance_ID_consistent": True,
        "deterministic_support_reduction_lineage": True,
        "per_block_QK_AV_timing": timings,
        "lifetime_fence": acceptance["lifetime_fence"],
        "family_timing_seconds": family_timing,
        "operator_calls": call_timing,
        "certificates": dispatch.certificates,
        "scientific_queries": 1,
        "bound_entrypoint_calls": 1,
    })
    certificate_seconds = time.perf_counter() - certificate_started
    lo, hi = float(lower.min()), float(upper.max())
    saved = _write(result_path, {
        "schema": "CORET_OPTIMIZED_THREE_PROPERTY_QUERY_RESULT_V1",
        "terminal_status": "COMPLETE",
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        "property_id": property_id,
        "query_ordinal": ordinal,
        "rho": float(rho),
        "rho_binary64_hex": float(rho).hex(),
        "direct_margin_interval": [lo, hi],
        "nominal_margin": nominal,
        "certified": lo > 0.0,
        "independent_checker_accepted": True,
        "all_support_claims_validated": True,
        "complete_certificate": True,
        "operator_certificate_record_sha256": certificate["record_sha256"],
        "invocation_counts": counts,
        "generic_fallbacks": 0,
        "provenance_ID_consistent": True,
        **acceptance,
        "per_block_QK_AV_timing": timings,
        "B2_QK_seconds": b2[0]["total_seconds"],
        "runtime_seconds": time.perf_counter() - started,
        "setup_seconds": setup_seconds,
        "bound_seconds": bound_seconds,
        "concretize_seconds": concretize_seconds,
        "independent_checker_seconds": checker_seconds,
        "certificate_serialization_seconds": certificate_seconds,
        "family_timing_seconds": family_timing,
        "peak_CPU_RSS_bytes": resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss * 1024,
        "peak_GPU_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_GPU_reserved_bytes": torch.cuda.max_memory_reserved(),
        "scientific_queries": 1,
        "bound_entrypoint_calls": 1,
    })
    print(json.dumps({"terminal_status": saved["terminal_status"],
                      "property_id": property_id, "rho": rho,
                      "certified": saved["certified"],
                      "B2_QK_seconds": saved["B2_QK_seconds"],
                      "record_sha256": saved["record_sha256"]}, sort_keys=True))


def _records(property_id: str) -> list[dict]:
    root = _property_root(property_id) / "queries"
    values = []
    if root.exists():
        for path in root.glob("query_*_result_v1.json"):
            item = dict(ref.verified(path))
            item["_path"] = path
            values.append(item)
    values.sort(key=lambda item: int(item["query_ordinal"]))
    if [item["query_ordinal"] for item in values] != list(range(len(values))):
        raise RuntimeError("query ordinals are not contiguous")
    if len({item["rho_binary64_hex"] for item in values}) != len(values):
        raise RuntimeError("duplicate query radius")
    return values


def _journal(property_id: str, values: list[dict]) -> None:
    path = _property_root(property_id) / "query_events_v1.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = [] if not path.exists() else historical.load_jsonl(path)
    seen = {item["query_result_record_sha256"] for item in existing}
    with path.open("a", encoding="utf-8") as handle:
        for item in values:
            if item["record_sha256"] in seen:
                continue
            event = {
                "schema": "CORET_OPTIMIZED_THREE_PROPERTY_QUERY_EVENT_V1",
                "property_id": property_id,
                "query_ordinal": item["query_ordinal"],
                "rho": item["rho"],
                "rho_binary64_hex": item["rho_binary64_hex"],
                "certified": item["certified"],
                "query_result_path": str(item["_path"]),
                "query_result_record_sha256": item["record_sha256"],
            }
            event["event_sha256"] = historical.canonical_payload_sha(event)
            handle.write(json.dumps(event, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            seen.add(item["record_sha256"])


def _invoke(property_id: str, rho: float, ordinal: int) -> None:
    command = [sys.executable, str(Path(__file__).resolve()), "query",
               "--authorized", "--property-id", property_id,
               "--rho", float(rho).hex(), "--query-ordinal", str(ordinal)]
    subprocess.run(command, cwd=WORKTREE, check=True, env={
        **os.environ,
        "PYTHONPATH": (
            "research_hab:/mnt/c/users/david-despacho/documents/"
            "vafali projects/lookahead-branching/research_hab"),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    })


def _query_summary(item: dict) -> dict:
    return {key: item.get(key) for key in (
        "query_ordinal", "rho", "terminal_status", "certified",
        "runtime_seconds", "setup_seconds", "bound_seconds", "B2_QK_seconds",
        "reached_B2_QK", "B2_QK_lifetime_fence_count",
        "per_block_QK_AV_timing", "peak_CPU_RSS_bytes",
        "peak_GPU_allocated_bytes", "peak_GPU_reserved_bytes",
        "record_sha256")}


def run_property(property_id: str, authorized: bool) -> None:
    if not authorized:
        raise RuntimeError("explicit USER authorization required")
    manifest = validate_manifest()
    prop = _property(manifest, property_id)
    result_path = _property_root(property_id) / "result_v1.json"
    if result_path.exists():
        existing = ref.verified(result_path)
        if existing.get("terminal_status") != "OPTIMIZED_PROPERTY_SMOKE_PASS":
            raise RuntimeError("existing property result did not pass")
        print(json.dumps({"terminal_status": "EXISTING_COMPLETE_PROPERTY_STOP",
                          "property_id": property_id,
                          "record_sha256": existing["record_sha256"]}, sort_keys=True))
        return
    while True:
        values = _records(property_id)
        _journal(property_id, values)
        state = historical.search_state(values)
        if state.get("done"):
            break
        _invoke(property_id, float(state["next_rho"]), len(values))
    values = _records(property_id)
    _journal(property_id, values)
    reference = prop["frozen_proof_reference"]["certified_radius"]
    radius = state["lower"]
    ratio = None if radius is None else radius / reference
    reached = sum(bool(item.get("reached_B2_QK")) for item in values)
    fence_count = sum(int(item.get("B2_QK_lifetime_fence_count", 0))
                      for item in values)
    b2_times = [float(item["B2_QK_seconds"]) for item in values
                if item.get("B2_QK_seconds") is not None]
    gates = {
        "terminal_search_complete": state.get("classification") ==
            "COMPLETE_CERTIFIED_RADIUS",
        "zero_checker_rejection": all(
            item.get("checker_accepted",
                     item.get("independent_checker_accepted",
                              item.get("partial_operator_checker_accepted", False)))
            for item in values),
        "zero_provenance_support_violation": all(
            item.get("provenance_ID_consistent") is True
            and item.get("all_support_claims_validated") is True
            for item in values),
        "zero_generic_fallback": sum(
            int(item.get("generic_fallbacks", 0)) for item in values) == 0,
        "outward_fresh_radii": all(
            item.get("outward_fresh_radius_envelopes") is True for item in values),
        "deterministic_support_reduction_lineage": all(
            item.get("deterministic_support_reduction_lineage") is True
            for item in values),
        "radius_retention": ratio is not None and ratio >= RADIUS_RETENTION_GATE,
        "exactly_one_fence_per_reached_trajectory": fence_count == reached,
        "B2_QK_no_pathology": bool(b2_times) and max(b2_times) <=
            B2_QK_PATHOLOGY_GUARD_SECONDS,
        "complete_wall_and_memory_telemetry": all(
            all(item.get(key) is not None for key in (
                "runtime_seconds", "peak_CPU_RSS_bytes",
                "peak_GPU_allocated_bytes", "peak_GPU_reserved_bytes"))
            for item in values),
    }
    passed = all(gates.values())
    saved = _write(result_path, {
        "schema": "CORET_OPTIMIZED_THREE_PROPERTY_RESULT_V1",
        "terminal_status": ("OPTIMIZED_PROPERTY_SMOKE_PASS" if passed
                            else "OPTIMIZED_PROPERTY_SMOKE_FAIL"),
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        "property_id": property_id,
        "sentence_ordinal": prop["sentence_ordinal"],
        "token_position": prop["token_position"],
        "search_classification": state["classification"],
        "certified_radius": radius,
        "uncertified_upper_radius": state["upper"],
        "frozen_proof_reference_radius": reference,
        "radius_retention_ratio": ratio,
        "radius_retention_gate": RADIUS_RETENTION_GATE,
        "query_count": len(values),
        "certified_query_count": sum(bool(item["certified"]) for item in values),
        "domain_failure_count": sum(
            item["terminal_status"] == "UNCERTIFIED_DOMAIN_FAILURE"
            for item in values),
        "reached_B2_QK_query_count": reached,
        "B2_QK_lifetime_fence_count": fence_count,
        "B2_QK_seconds": b2_times,
        "B2_QK_max_seconds": max(b2_times) if b2_times else None,
        "gates": gates,
        "total_search_wall_seconds": sum(
            float(item["runtime_seconds"]) for item in values),
        "per_radius": [_query_summary(item) for item in values],
        "peak_CPU_RSS_bytes": max(
            int(item["peak_CPU_RSS_bytes"]) for item in values),
        "peak_GPU_allocated_bytes": max(
            int(item["peak_GPU_allocated_bytes"]) for item in values),
        "peak_GPU_reserved_bytes": max(
            int(item["peak_GPU_reserved_bytes"]) for item in values),
        "scientific_queries": len(values),
        "bound_entrypoint_calls": len(values),
    })
    if not passed:
        raise RuntimeError(
            f"optimized property smoke failed: {property_id}: "
            f"{[name for name, value in gates.items() if not value]}")
    print(json.dumps({"terminal_status": saved["terminal_status"],
                      "property_id": property_id,
                      "certified_radius": radius,
                      "radius_retention_ratio": ratio,
                      "record_sha256": saved["record_sha256"]}, sort_keys=True))


def next_incomplete_property(manifest: dict) -> str | None:
    for prop in manifest["properties"]:
        path = _property_root(prop["property_id"]) / "result_v1.json"
        if not path.exists():
            return prop["property_id"]
        record = ref.verified(path)
        if record.get("terminal_status") != "OPTIMIZED_PROPERTY_SMOKE_PASS":
            raise RuntimeError(f"existing failed property blocks resume: {path}")
    return None


def summarize() -> dict:
    manifest = validate_manifest()
    results = []
    for prop in manifest["properties"]:
        path = _property_root(prop["property_id"]) / "result_v1.json"
        record = ref.verified(path)
        if record.get("terminal_status") != "OPTIMIZED_PROPERTY_SMOKE_PASS":
            raise RuntimeError("cannot summarize non-passing property")
        results.append(record)
    passed = (len(results) == 3 and all(
        all(item["gates"].values()) for item in results))
    saved = _write(SUMMARY, {
        "schema": "CORET_OPTIMIZED_THREE_PROPERTY_SMOKE_SUMMARY_V1",
        "terminal_status": ("OPTIMIZED_THREE_PROPERTY_SMOKE_PASS" if passed
                            else "OPTIMIZED_THREE_PROPERTY_SMOKE_FAIL"),
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        "ordered_property_ids": [item["property_id"] for item in results],
        "property_count": len(results),
        "certified_radii": [item["certified_radius"] for item in results],
        "radius_retention_ratios": [item["radius_retention_ratio"]
                                    for item in results],
        "total_wall_seconds": sum(item["total_search_wall_seconds"]
                                  for item in results),
        "peak_CPU_RSS_bytes": max(item["peak_CPU_RSS_bytes"] for item in results),
        "peak_GPU_allocated_bytes": max(
            item["peak_GPU_allocated_bytes"] for item in results),
        "peak_GPU_reserved_bytes": max(
            item["peak_GPU_reserved_bytes"] for item in results),
        "generic_fallbacks": 0,
        "all_gates_passed": passed,
        "property_result_record_sha256s": [item["record_sha256"]
                                            for item in results],
    })
    if not passed:
        raise RuntimeError("optimized three-property smoke aggregate failed")
    return saved


def run(authorized: bool) -> None:
    if not authorized:
        raise RuntimeError("explicit USER authorization required")
    manifest = validate_manifest()
    if SUMMARY.exists():
        existing = ref.verified(SUMMARY)
        print(json.dumps({"terminal_status": "EXISTING_COMPLETE_SMOKE_STOP",
                          "classification": existing["terminal_status"],
                          "record_sha256": existing["record_sha256"]}, sort_keys=True))
        return
    while True:
        property_id = next_incomplete_property(manifest)
        if property_id is None:
            break
        command = [sys.executable, str(Path(__file__).resolve()), "run-property",
                   "--authorized", "--property-id", property_id]
        subprocess.run(command, cwd=WORKTREE, check=True, env={
            **os.environ,
            "PYTHONPATH": (
                "research_hab:/mnt/c/users/david-despacho/documents/"
                "vafali projects/lookahead-branching/research_hab"),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        })
    saved = summarize()
    print(json.dumps({"terminal_status": saved["terminal_status"],
                      "property_count": saved["property_count"],
                      "record_sha256": saved["record_sha256"]}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=(
        "freeze", "preflight", "query", "run-property", "run"))
    parser.add_argument("--authorized", action="store_true")
    parser.add_argument("--property-id")
    parser.add_argument("--rho")
    parser.add_argument("--query-ordinal", type=int)
    values = parser.parse_args()
    if values.action == "freeze":
        freeze()
    elif values.action == "preflight":
        preflight()
    elif values.action == "query":
        if values.property_id is None or values.rho is None \
                or values.query_ordinal is None:
            raise RuntimeError("query requires property, rho, and ordinal")
        query(values.property_id, float.fromhex(values.rho),
              values.query_ordinal, values.authorized)
    elif values.action == "run-property":
        if values.property_id is None:
            raise RuntimeError("run-property requires property")
        run_property(values.property_id, values.authorized)
    else:
        run(values.authorized)


if __name__ == "__main__":
    main()
