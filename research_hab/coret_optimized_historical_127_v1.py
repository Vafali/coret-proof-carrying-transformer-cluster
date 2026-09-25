#!/usr/bin/env python3
"""Final optimized, proof-carrying historical 127-property benchmark.

The scientific property list and search protocol are inherited verbatim from
the immutable historical manifest.  Only the accepted optimized execution
path is used: native semantics, provenance-derived structural support,
mechanically capped A.V tiles, and the deterministic pre-B2-QK cache fence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

import coret_bounded_native_production_graph_v1 as graph
import coret_deept_exact_standard_ln_adapter as adapter
import coret_native_semantics_production_graph_v1 as production
import coret_optimized_three_property_smoke_v1 as optimized_smoke
import coret_proof_carrying_historical_127_v1 as historical_127
import coret_proof_carrying_historical_smoke_v1 as historical_smoke
import coret_sound_reference_v1 as ref
import coret_structural_support_lifetime_v1 as lifetime
import coret_structural_support_precise_dot_v1 as structural


WORKTREE = Path(__file__).resolve().parents[1]
ROOT = ref.ROOT
HERE = Path(__file__).resolve()
OUT = WORKTREE / (
    "research_hab/results/coret_optimized_historical_127_v1_20260924")
PRIOR_MANIFEST = OUT / "coret_optimized_historical_127_manifest_v1.json"
MANIFEST = OUT / "coret_optimized_historical_127_manifest_v2.json"
PREFLIGHT = OUT / "coret_optimized_historical_127_preflight_v2.json"
SUMMARY = OUT / "coret_optimized_historical_127_summary_v2.json"
LAUNCHER = WORKTREE / "research_hab/run_coret_optimized_historical_127_v1.sh"
TEST = WORKTREE / "research_hab/tests/test_coret_optimized_historical_127_v1.py"

HISTORICAL_MANIFEST = historical_127.HISTORICAL_MANIFEST
DEEPT_POSITIONS = historical_127.DEEPT_POSITIONS
DEEPT_RESULT = historical_127.DEEPT_RESULT
PROOF_SMOKE_MANIFEST = historical_127.SMOKE_MANIFEST
PROOF_SMOKE_SUMMARY = historical_127.SMOKE_SUMMARY
PROOF_SMOKE_ROOT = historical_127.SMOKE_ROOT
OPTIMIZED_SMOKE_MANIFEST = optimized_smoke.MANIFEST
OPTIMIZED_SMOKE_SUMMARY = optimized_smoke.SUMMARY
OPTIMIZED_SMOKE_ROOT = optimized_smoke.OUT

EXPECTED_HISTORICAL_CANONICAL_SHA = (
    "7e4b2fea94424e554f07272aa8a246da7cda1212be83af57bbcdae7c870ed9dd")
EXPECTED_DEEPT_CACHE_SHA = (
    "67e80d74cf83e5f810726405a6f6f0cf9928f4dda84d7f87decf59f354c8181d")
EXPECTED_PROOF_SMOKE_SUMMARY_SHA = (
    "b24126783f1d7370524d005003d003234bd135fb0f5458dcd69a87723d7042fd")
EXPECTED_OPTIMIZED_SMOKE_MANIFEST_SHA = (
    "98868b6ba6a5be2985001115f256a84c260f9cd83259865621701f59b340d449")
EXPECTED_OPTIMIZED_SMOKE_STATUS = "OPTIMIZED_THREE_PROPERTY_SMOKE_PASS"
INITIAL_RHO = 0.000625
MAX_FACTOR_TWO_REFINEMENTS = 12
MIDPOINT_ITERATIONS = 10
B2_QK_PATHOLOGY_GUARD_SECONDS = 30.0
PRIOR_MANIFEST_CANONICAL_SHA = (
    "7237f1e6d642bf0d4d10e5a4d0739a4a350223277a05f14f584207f084093399")
RECIPROCAL_NAN_NATIVE_ASSERTION = (
    "Reciprocal: there are NaNs in the new COEFFS, pre-condition not met")
PINNED_ZONOTOPE_BLOB_SHA256 = (
    "08b502ea409170e184f6dbeaad3318d3decfb58cabb6417f3c27d62308d57ad9")
PINNED_VERIFIER_BLOB_SHA256 = (
    "64ea76bdf7322c526290cd973525dbe49debcd0bdbe8450d9615ed36ecd45fe2")


def _is_native_domain_failure(error: BaseException) -> bool:
    """Mirror native DeepT's fail-closed handling, without broadening it.

    The historical predicates retain their exact existing type/prefix checks.
    This adds only the one pinned reciprocal assertion, by exact type and exact
    message equality.  Near matches and every other AssertionError stay fatal.
    """
    return historical_smoke.is_native_domain_failure(error) or (
        type(error) is AssertionError
        and str(error) == RECIPROCAL_NAN_NATIVE_ASSERTION
    )


def _canonical(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


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
        HERE, LAUNCHER, TEST,
        WORKTREE / "research_hab/coret_structural_support_lifetime_v1.py",
        WORKTREE / "research_hab/coret_structural_support_precise_dot_v1.py",
        WORKTREE / "research_hab/coret_optimized_three_property_smoke_v1.py",
        Path(graph.__file__).resolve(), Path(production.__file__).resolve(),
        Path(adapter.__file__).resolve(), Path(historical_smoke.__file__).resolve(),
        ROOT / "research_hab/coret_native_semantics_checker_v1.py",
        ROOT / "research_hab/coret_sound_reference_v1.py",
    )


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _cached_deept() -> dict[str, dict]:
    rows = {}
    for row in _load_jsonl(DEEPT_POSITIONS):
        property_id = row["property_id"]
        if property_id in rows:
            raise RuntimeError(f"duplicate cached DeepT ID: {property_id}")
        rows[property_id] = row
    return rows


def _proof_references(proof_manifest: dict) -> dict[str, dict]:
    references = {}
    for selected in proof_manifest["properties"]:
        property_id = selected["property_id"]
        path = PROOF_SMOKE_ROOT / "properties" / property_id / "result_v1.json"
        result = ref.verified(path)
        if (result.get("terminal_status") != "COMPLETE"
                or result.get("classification") != "COMPLETE_CERTIFIED_RADIUS"
                or not result.get("complete_certificate")
                or not result.get("independent_checker_accepted")):
            raise RuntimeError(f"proof reference is not accepted: {property_id}")
        references[property_id] = {
            "path": str(path),
            "file_sha256": ref.sha(path),
            "record_sha256": result["record_sha256"],
            "certified_radius": result["certified_radius"],
            "certified_radius_binary64_hex": float(
                result["certified_radius"]).hex(),
        }
    if len(references) != 10:
        raise RuntimeError("frozen proof-reference count differs")
    return references


def _optimized_reuse(optimized_manifest: dict,
                     optimized_summary: dict) -> list[dict]:
    expected_ids = optimized_summary["ordered_property_ids"]
    if (optimized_summary.get("terminal_status") !=
            EXPECTED_OPTIMIZED_SMOKE_STATUS
            or optimized_summary.get("property_count") != 3
            or optimized_summary.get("all_gates_passed") is not True):
        raise RuntimeError("optimized smoke is not reusable")
    records = []
    for ordinal, property_id in enumerate(expected_ids):
        result_path = (OPTIMIZED_SMOKE_ROOT / "properties" / property_id /
                       "result_v1.json")
        result = ref.verified(result_path)
        if (result.get("terminal_status") != "OPTIMIZED_PROPERTY_SMOKE_PASS"
                or not all(result.get("gates", {}).values())
                or result.get("generic_fallbacks", 0) != 0):
            raise RuntimeError(f"optimized result is not reusable: {property_id}")
        query_root = result_path.parent / "queries"
        query_paths = sorted(query_root.glob("query_*_result_v1.json"))
        queries = [ref.verified(path) for path in query_paths]
        if len(queries) != int(result["query_count"]):
            raise RuntimeError(f"optimized query inventory differs: {property_id}")
        records.append({
            "reuse_ordinal": ordinal,
            "property_id": property_id,
            "result_path": str(result_path),
            "result_file_sha256": ref.sha(result_path),
            "result_record_sha256": result["record_sha256"],
            "certified_radius": result["certified_radius"],
            "query_count": len(queries),
            "query_inventory_sha256": _canonical([
                {"path": str(path), "file_sha256": ref.sha(path),
                 "record_sha256": query["record_sha256"]}
                for path, query in zip(query_paths, queries)
            ]),
        })
    manifest_ids = [item["property_id"] for item in optimized_manifest["properties"]]
    if manifest_ids != expected_ids or len(records) != 3:
        raise RuntimeError("optimized reuse order differs")
    return records


def _duplicate_entries(properties: list[dict]) -> list[dict]:
    entries = [item for item in properties
               if int(item["source_test_line"]) == 1172
               and int(item["token_position"]) == 10]
    if len(entries) != 2 or len({item["property_id"] for item in entries}) != 2:
        raise RuntimeError("frozen historical duplicate was not preserved")
    return [{"benchmark_ordinal": item["benchmark_ordinal"],
             "property_id": item["property_id"],
             "sentence_ordinal": item["sentence_ordinal"],
             "source_test_line": item["source_test_line"],
             "token_position": item["token_position"]} for item in entries]


def expected_manifest() -> dict:
    prior = ref.verified(PRIOR_MANIFEST, "canonical_manifest_sha256")
    if prior["canonical_manifest_sha256"] != PRIOR_MANIFEST_CANONICAL_SHA:
        raise RuntimeError("prior optimized historical-127 manifest differs")
    parent = ref.verified(HISTORICAL_MANIFEST, "canonical_manifest_sha256")
    if parent["canonical_manifest_sha256"] != EXPECTED_HISTORICAL_CANONICAL_SHA:
        raise RuntimeError("historical scientific manifest identity differs")
    if ref.sha(DEEPT_POSITIONS) != EXPECTED_DEEPT_CACHE_SHA:
        raise RuntimeError("cached native DeepT identity differs")
    if ref.sha(PROOF_SMOKE_SUMMARY) != EXPECTED_PROOF_SMOKE_SUMMARY_SHA:
        raise RuntimeError("accepted proof smoke summary identity differs")
    proof_manifest = ref.verified(PROOF_SMOKE_MANIFEST,
                                  "canonical_manifest_sha256")
    optimized_manifest = ref.verified(OPTIMIZED_SMOKE_MANIFEST,
                                      "canonical_manifest_sha256")
    if optimized_manifest["canonical_manifest_sha256"] != \
            EXPECTED_OPTIMIZED_SMOKE_MANIFEST_SHA:
        raise RuntimeError("optimized smoke manifest identity differs")
    optimized_summary = ref.verified(OPTIMIZED_SMOKE_SUMMARY)
    if optimized_summary.get("canonical_manifest_sha256") != \
            optimized_manifest["canonical_manifest_sha256"]:
        raise RuntimeError("optimized smoke summary/manifest linkage differs")
    deept_summary = ref.verified(DEEPT_RESULT)
    cache = _cached_deept()
    proof_refs = _proof_references(proof_manifest)
    optimized_reuse = _optimized_reuse(optimized_manifest, optimized_summary)
    parent_properties = historical_127.frozen_properties(parent)
    properties = []
    for ordinal, item in enumerate(parent_properties):
        row = cache.get(item["property_id"])
        if row is None or row.get("manifest_sha256") != \
                EXPECTED_HISTORICAL_CANONICAL_SHA:
            raise RuntimeError(f'missing cached DeepT row: {item["property_id"]}')
        properties.append({
            "benchmark_ordinal": ordinal,
            **item,
            "cached_DeepT_reference": row,
            "frozen_proof_reference": proof_refs.get(item["property_id"]),
        })
    if len(properties) != 127 or len(cache) != 127:
        raise RuntimeError("historical property/cache count differs")
    if optimized_manifest["model"] != parent["model"]:
        raise RuntimeError("optimized model identity differs from historical model")
    if optimized_manifest["checkpoint_sha256"] != parent["checkpoint_sha256"]:
        raise RuntimeError("optimized checkpoint identity differs")
    duplicate = _duplicate_entries(properties)
    return {
        "schema": "CORET_OPTIMIZED_HISTORICAL_127_MANIFEST_V2",
        "status": "FROZEN_BEFORE_FINAL_BENCHMARK",
        "method_status": "OPTIMIZED_THREE_PROPERTY_SMOKE_PASS",
        "historical_scientific_manifest": {
            "path": str(HISTORICAL_MANIFEST),
            "canonical_sha256": parent["canonical_manifest_sha256"],
            "file_sha256": ref.sha(HISTORICAL_MANIFEST),
            "ordered_property_ids_sha256": _canonical(
                [item["property_id"] for item in properties]),
        },
        "properties": properties,
        "property_count": 127,
        "historical_duplicate_entries": duplicate,
        "preserve_duplicate_historical_property": True,
        "cached_native_DeepT": {
            "rerun_permitted": False,
            "positions_path": str(DEEPT_POSITIONS),
            "positions_file_sha256": ref.sha(DEEPT_POSITIONS),
            "positions_count": len(cache),
            "summary_path": str(DEEPT_RESULT),
            "summary_file_sha256": ref.sha(DEEPT_RESULT),
            "summary_record_sha256": deept_summary["record_sha256"],
        },
        "frozen_proof_references": {
            "count": len(proof_refs),
            "accepted_smoke_manifest_path": str(PROOF_SMOKE_MANIFEST),
            "accepted_smoke_manifest_canonical_sha256": proof_manifest[
                "canonical_manifest_sha256"],
            "accepted_smoke_summary_path": str(PROOF_SMOKE_SUMMARY),
            "accepted_smoke_summary_file_sha256": ref.sha(PROOF_SMOKE_SUMMARY),
        },
        "optimized_smoke_reuse": {
            "count": len(optimized_reuse),
            "scientific_reexecution": False,
            "manifest_path": str(OPTIMIZED_SMOKE_MANIFEST),
            "manifest_canonical_sha256": optimized_manifest[
                "canonical_manifest_sha256"],
            "summary_path": str(OPTIMIZED_SMOKE_SUMMARY),
            "summary_file_sha256": ref.sha(OPTIMIZED_SMOKE_SUMMARY),
            "summary_record_sha256": optimized_summary["record_sha256"],
            "records": optimized_reuse,
        },
        "model": parent["model"],
        "checkpoint_sha256": parent["checkpoint_sha256"],
        "pinned_DeepT_revision": optimized_manifest["pinned_DeepT_revision"],
        "threat_model": {
            "norm_p": 100,
            "single_token_embedding_Linf": True,
            "source_dimension": 128,
            "perturbed_tokens": 1,
        },
        "search": {
            "initial_rho": INITIAL_RHO,
            "initial_rho_binary64_hex": INITIAL_RHO.hex(),
            "factor_two_bracketing": True,
            "maximum_factor_two_refinements": MAX_FACTOR_TWO_REFINEMENTS,
            "midpoint_iterations": MIDPOINT_ITERATIONS,
            "typed_native_domain_failure_prefixes": list(
                historical_smoke.NATIVE_DOMAIN_FAILURE_PREFIXES),
            "typed_native_domain_failure_exact_assertions": [
                RECIPROCAL_NAN_NATIVE_ASSERTION],
            "unexpected_exception_policy": "fatal",
            "reported_radius": "largest observed certified lower endpoint",
        },
        "orchestration_correction": {
            "prior_manifest_path": str(PRIOR_MANIFEST),
            "prior_manifest_canonical_sha256": PRIOR_MANIFEST_CANONICAL_SHA,
            "prior_manifest_file_sha256": ref.sha(PRIOR_MANIFEST),
            "completed_prior_results_are_immutable_and_reused": True,
            "scientific_semantics_changed": False,
            "native_behavior": (
                "pinned VerifierZonotope.get_bounds_difference_in_scores catches "
                "AssertionError and returns None; verify_safety maps None to False"),
            "translation_scope": (
                "exact AssertionError type and exact reciprocal-NaN message only"),
            "pinned_source_evidence": {
                "revision": "16ffe4075f1f8a7c87fa2a187d8c46cfd51e07bf",
                "Zonotope_py_blob_sha256": PINNED_ZONOTOPE_BLOB_SHA256,
                "VerifierZonotope_py_blob_sha256": PINNED_VERIFIER_BLOB_SHA256,
                "reciprocal_assertion_line": 1923,
                "verifier_try_lines": [102, 143],
                "verifier_AssertionError_catch_lines": [144, 149],
                "verify_safety_None_to_False_lines": [75, 83],
            },
        },
        "optimized_execution": {
            "proof_carrying_native_semantics": True,
            "provenance_derived_structural_support": True,
            "support_and_reduction_recomputed_each_radius": True,
            "requested_AV_generator_tile": structural.AV_GENERATOR_TILE,
            "grouped_temporary_cap_bytes": structural.AV_TEMPORARY_CAP_BYTES,
            "effective_AV_tile_rule": (
                "min(requested_tile, isqrt(cap_bytes // "
                "(heads*queries*features*element_size)))"),
            "deterministic_pre_B2_QK_lifetime_cache_fence": True,
            "lifetime_fence_count_per_trajectory_reaching_B2_QK": 1,
            "B2_QK_pathology_guard_seconds": B2_QK_PATHOLOGY_GUARD_SECONDS,
            "independent_checker_required": True,
            "generic_fallback_allowed": False,
            "autograd_enabled": False,
            "deterministic_algorithms": True,
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        },
        "outputs": {
            "root": str(OUT),
            "per_property": str(OUT / "properties/<property_id>/result_v1.json"),
            "per_query": str(OUT / "properties/<property_id>/queries/query_*_result_v1.json"),
            "per_query_certificate": str(OUT / "properties/<property_id>/queries/query_*_certificates_v1.json"),
            "append_only_journal": str(OUT / "properties/<property_id>/query_events_v1.jsonl"),
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
        raise RuntimeError("optimized historical-127 manifest mismatch")
    return stored


def freeze() -> None:
    saved = _write(MANIFEST, expected_manifest(), "canonical_manifest_sha256")
    print(json.dumps({
        "terminal_status": "OPTIMIZED_HISTORICAL_127_FROZEN_NO_SOLVE",
        "canonical_manifest_sha256": saved["canonical_manifest_sha256"],
        "property_count": saved["property_count"],
        "cached_DeepT_count": saved["cached_native_DeepT"]["positions_count"],
        "optimized_reuse_count": saved["optimized_smoke_reuse"]["count"],
        "scientific_query_count": 0,
        "bound_entrypoint_call_count": 0,
    }, sort_keys=True))


def _resume_self_check() -> dict:
    rows = [
        {"rho_binary64_hex": INITIAL_RHO.hex(), "certified": True},
        {"rho_binary64_hex": (2 * INITIAL_RHO).hex(), "certified": False},
        {"rho_binary64_hex": (1.5 * INITIAL_RHO).hex(), "certified": True},
    ]
    state = historical_smoke.search_state(rows)
    expected_next = 0.00109375
    if (state.get("stage") != "midpoint"
            or state.get("midpoint_index") != 1
            or state.get("next_rho") != expected_next):
        raise RuntimeError("resume scheduler self-check differs")
    return state


def preflight() -> None:
    manifest = validate_manifest()
    parent = ref.verified(HISTORICAL_MANIFEST, "canonical_manifest_sha256")
    parent_ids = [item["property_id"] for item in parent["properties"]]
    manifest_ids = [item["property_id"] for item in manifest["properties"]]
    if parent_ids != manifest_ids or len(manifest_ids) != 127:
        raise RuntimeError("ordered 127-property identity differs")
    if manifest["cached_native_DeepT"]["positions_count"] != 127:
        raise RuntimeError("DeepT cache coverage differs")
    if len(manifest["historical_duplicate_entries"]) != 2:
        raise RuntimeError("historical duplicate is absent")
    resume = _resume_self_check()
    record = {
        "schema": "CORET_OPTIMIZED_HISTORICAL_127_PREFLIGHT_V2",
        "terminal_status": "PASS_NO_SOLVE",
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        "historical_manifest_canonical_sha256": manifest[
            "historical_scientific_manifest"]["canonical_sha256"],
        "ordered_property_ids_sha256": manifest[
            "historical_scientific_manifest"]["ordered_property_ids_sha256"],
        "property_count": 127,
        "cached_DeepT_reference_count": 127,
        "optimized_reusable_property_count": 3,
        "frozen_proof_reference_count": 10,
        "historical_duplicate_entries": manifest["historical_duplicate_entries"],
        "model_identity_matches": True,
        "checkpoint_identity_matches": True,
        "threat_model_identity_matches": True,
        "resume_scheduler_self_check": resume,
        "checkpoint_loaded": False,
        "model_forward_calls": 0,
        "scientific_query_count": 0,
        "bound_entrypoint_call_count": 0,
    }
    if PREFLIGHT.exists():
        existing = ref.verified(PREFLIGHT)
        expected = dict(record)
        expected["record_sha256"] = ref.canonical(record)
        if existing != expected:
            raise RuntimeError("existing optimized-127 preflight differs")
        status = "PREFLIGHT_REUSED_VERIFIED"
        saved = existing
    else:
        saved = _write(PREFLIGHT, record)
        status = "PASS_NO_SOLVE"
    print(json.dumps({
        "terminal_status": status,
        "record_sha256": saved["record_sha256"],
        "property_count": 127,
        "cached_DeepT_reference_count": 127,
        "optimized_reuse_count": 3,
        "scientific_query_count": 0,
        "bound_entrypoint_call_count": 0,
    }, sort_keys=True))


def _property(manifest: dict, property_id: str) -> dict:
    rows = [item for item in manifest["properties"]
            if item["property_id"] == property_id]
    if len(rows) != 1:
        raise RuntimeError("property is not unique in frozen benchmark")
    return rows[0]


def _example(prop: dict) -> dict:
    parent = ref.verified(HISTORICAL_MANIFEST, "canonical_manifest_sha256")
    rows = [item for item in parent["examples"]
            if int(item["sentence_ordinal"]) == int(prop["sentence_ordinal"])]
    if len(rows) != 1:
        raise RuntimeError("historical example identity differs")
    return rows[0]


def _tag(rho: float) -> str:
    return float(rho).hex().replace("+", "p").replace("-", "m").replace(".", "d")


def _property_root(property_id: str) -> Path:
    return OUT / "properties" / property_id


def _query_paths(property_id: str, ordinal: int, rho: float) -> tuple[Path, Path]:
    stem = f"query_{ordinal:02d}_{_tag(rho)}"
    root = _property_root(property_id) / "queries"
    return root / f"{stem}_result_v1.json", root / f"{stem}_certificates_v1.json"


def _query_telemetry(dispatch, delegate) -> dict:
    acceptance = optimized_smoke._partial_acceptance(dispatch, delegate)
    if not acceptance["checker_accepted"]:
        raise RuntimeError(
            f'optimized acceptance failed: {acceptance["acceptance_first_failure"]}')
    timings = optimized_smoke._operator_telemetry(dispatch.certificates)
    b2 = [row for row in timings
          if row["block"] == 2 and row["family"] == "QK"]
    if acceptance["reached_B2_QK"] and len(b2) > 1:
        raise RuntimeError("multiple B2 QK timings")
    b2_seconds = None if not b2 else b2[0]["total_seconds"]
    if b2_seconds is not None and b2_seconds > B2_QK_PATHOLOGY_GUARD_SECONDS:
        raise RuntimeError("B2 QK allocator pathology returned")
    return {
        **acceptance,
        "per_block_QK_AV_timing": timings,
        "B2_QK_seconds": b2_seconds,
    }


def generate_query(property_id: str, rho: float, ordinal: int,
                   authorized: bool) -> None:
    if not authorized:
        raise RuntimeError("explicit USER authorization required")
    manifest = validate_manifest()
    prop = _property(manifest, property_id)
    result_path, certificate_path = _query_paths(property_id, ordinal, rho)
    if result_path.exists() or certificate_path.exists():
        raise FileExistsError("immutable optimized-127 query artifact exists")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("optimized historical-127 query requires CUDA")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda:0")
    started = time.perf_counter()
    setup_started = started
    modules = adapter.FrozenDeepTModules()
    from Verifiers.Zonotope import Zonotope
    model, _, configuration = adapter.load_native_model(
        modules, device, dtype=torch.float32)
    if configuration != manifest["model"]["configuration"]:
        raise RuntimeError("loaded model configuration differs")
    example = _example(prop)
    ids = torch.tensor([example["token_ids"]], dtype=torch.long, device=device)
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
                raise RuntimeError("optimized benchmark requires no_grad")
            margin, dispatch = graph.execute(
                dead_holder["input_zonotope"], model, args,
                clean_label=label, dispatch=dispatch)
    except AssertionError as error:
        if not _is_native_domain_failure(error):
            raise
        torch.cuda.synchronize()
        telemetry = _query_telemetry(dispatch, delegate)
        saved = _write(result_path, {
            "schema": "CORET_OPTIMIZED_HISTORICAL_127_QUERY_RESULT_V1",
            "terminal_status": "UNCERTIFIED_DOMAIN_FAILURE",
            "reason_code": "UNCERTIFIED_DOMAIN_FAILURE",
            "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
            "property_id": property_id,
            "query_ordinal": ordinal,
            "rho": float(rho),
            "rho_binary64_hex": float(rho).hex(),
            "direct_margin_interval": None,
            "nominal_margin": nominal,
            "certified": False,
            "authoritative_bound_returned": False,
            "complete_certificate": False,
            "exception_type": "AssertionError",
            "exception_message": str(error),
            "independent_checker_accepted": True,
            "all_support_claims_validated": True,
            "generic_fallback_count": 0,
            "provenance_ID_consistent": True,
            "outward_fresh_radius_envelopes": True,
            "deterministic_support_reduction_lineage": True,
            **telemetry,
            "setup_seconds": setup_seconds,
            "proof_generation_time_seconds": time.perf_counter() - bound_started,
            "independent_checker_time_seconds": 0.0,
            "certificate_serialization_seconds": 0.0,
            "bound_runtime_seconds": time.perf_counter() - bound_started,
            "total_wall_time_seconds": time.perf_counter() - started,
            "peak_CPU_RSS_bytes": resource.getrusage(
                resource.RUSAGE_SELF).ru_maxrss * 1024,
            "peak_GPU_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_GPU_reserved_bytes": torch.cuda.max_memory_reserved(),
            "fresh_verifier_evaluation_count": 1,
            "bound_entrypoint_call_count": 1,
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
        raise RuntimeError("optimized authoritative interval invalid")
    counts = dispatch.assert_complete(graph.EXPECTED_THREE_BLOCK_COUNTS)
    checker_started = time.perf_counter()
    telemetry = _query_telemetry(dispatch, delegate)
    checker_seconds = time.perf_counter() - checker_started
    if dispatch.generic_family_invocations != 0:
        raise RuntimeError("optimized benchmark reached generic fallback")
    family_timing, call_timing = optimized_smoke.optimized_runner._family_timing(
        dispatch.certificates)
    certificate_started = time.perf_counter()
    certificate = _write(certificate_path, {
        "schema": "CORET_OPTIMIZED_HISTORICAL_127_QUERY_CERTIFICATES_V1",
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        "property_id": property_id,
        "query_ordinal": ordinal,
        "rho": float(rho),
        "rho_binary64_hex": float(rho).hex(),
        "invocation_counts": counts,
        "generic_family_invocations": 0,
        "complete_certificate": True,
        "independent_checker_accepted": True,
        "all_support_claims_validated": True,
        "provenance_ID_consistent": True,
        "outward_fresh_radius_envelopes": True,
        "deterministic_support_reduction_lineage": True,
        "acceptance_predicates": telemetry["acceptance_predicates"],
        "AV_tile_checks": telemetry["AV_tile_checks"],
        "lifetime_fence": telemetry["lifetime_fence"],
        "per_block_QK_AV_timing": telemetry["per_block_QK_AV_timing"],
        "family_timing_seconds": family_timing,
        "operator_calls": call_timing,
        "certificates": dispatch.certificates,
        "scientific_query_count": 1,
        "bound_entrypoint_call_count": 1,
    })
    certificate_seconds = time.perf_counter() - certificate_started
    lo, hi = float(lower.min()), float(upper.max())
    saved = _write(result_path, {
        "schema": "CORET_OPTIMIZED_HISTORICAL_127_QUERY_RESULT_V1",
        "terminal_status": "COMPLETE",
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        "property_id": property_id,
        "query_ordinal": ordinal,
        "rho": float(rho),
        "rho_binary64_hex": float(rho).hex(),
        "direct_margin_interval": [lo, hi],
        "nominal_margin": nominal,
        "certified": lo > 0.0,
        "authoritative_bound_returned": True,
        "complete_certificate": True,
        "independent_checker_accepted": True,
        "all_support_claims_validated": True,
        "generic_fallback_count": 0,
        "provenance_ID_consistent": True,
        "outward_fresh_radius_envelopes": True,
        "deterministic_support_reduction_lineage": True,
        "operator_certificate_path": str(certificate_path),
        "operator_certificate_record_sha256": certificate["record_sha256"],
        "invocation_counts": counts,
        **telemetry,
        "setup_seconds": setup_seconds,
        "proof_generation_time_seconds": bound_seconds,
        "independent_checker_time_seconds": checker_seconds,
        "concretize_seconds": concretize_seconds,
        "certificate_serialization_seconds": certificate_seconds,
        "bound_runtime_seconds": bound_seconds,
        "total_wall_time_seconds": time.perf_counter() - started,
        "peak_CPU_RSS_bytes": resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss * 1024,
        "peak_GPU_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_GPU_reserved_bytes": torch.cuda.max_memory_reserved(),
        "fresh_verifier_evaluation_count": 1,
        "bound_entrypoint_call_count": 1,
    })
    print(json.dumps({"terminal_status": saved["terminal_status"],
                      "property_id": property_id, "rho": rho,
                      "certified": saved["certified"],
                      "B2_QK_seconds": saved["B2_QK_seconds"],
                      "record_sha256": saved["record_sha256"]}, sort_keys=True))


def _query_records(property_id: str) -> list[dict]:
    directory = _property_root(property_id) / "queries"
    if not directory.exists():
        return []
    allowed_manifests = {
        PRIOR_MANIFEST_CANONICAL_SHA,
        validate_manifest()["canonical_manifest_sha256"],
    }
    rows = []
    for path in directory.glob("query_*_result_v1.json"):
        row = dict(ref.verified(path))
        if row.get("property_id") != property_id:
            raise RuntimeError("query property identity differs")
        if row.get("canonical_manifest_sha256") not in allowed_manifests:
            raise RuntimeError("query manifest identity is not an allowed resume identity")
        row["_path"] = path
        rows.append(row)
    rows.sort(key=lambda item: int(item["query_ordinal"]))
    if [item["query_ordinal"] for item in rows] != list(range(len(rows))):
        raise RuntimeError("query sequence is not contiguous")
    if len({item["rho_binary64_hex"] for item in rows}) != len(rows):
        raise RuntimeError("duplicate query radius")
    return rows


def _sync_journal(property_id: str, rows: list[dict]) -> None:
    path = _property_root(property_id) / "query_events_v1.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = [] if not path.exists() else _load_jsonl(path)
    seen = {item["query_result_record_sha256"] for item in existing}
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            if row["record_sha256"] in seen:
                continue
            event = {
                "schema": "CORET_OPTIMIZED_HISTORICAL_127_QUERY_EVENT_V1",
                "property_id": property_id,
                "query_ordinal": row["query_ordinal"],
                "rho": row["rho"],
                "rho_binary64_hex": row["rho_binary64_hex"],
                "certified": row["certified"],
                "terminal_status": row["terminal_status"],
                "query_result_path": str(row["_path"]),
                "query_result_record_sha256": row["record_sha256"],
            }
            event["event_sha256"] = _canonical(event)
            handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            seen.add(row["record_sha256"])


def _invoke_query(property_id: str, rho: float, ordinal: int) -> None:
    subprocess.run([
        sys.executable, str(HERE), "query", "--authorized",
        "--property-id", property_id, "--rho", float(rho).hex(),
        "--query-ordinal", str(ordinal),
    ], cwd=WORKTREE, check=True, env={
        **os.environ,
        "PYTHONPATH": (
            "research_hab:/mnt/c/users/david-despacho/documents/"
            "vafali projects/lookahead-branching/research_hab"),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    })


def _reuse_record(manifest: dict, property_id: str) -> dict | None:
    rows = [item for item in manifest["optimized_smoke_reuse"]["records"]
            if item["property_id"] == property_id]
    if len(rows) > 1:
        raise RuntimeError("duplicate optimized reuse record")
    return rows[0] if rows else None


def _reuse_property(manifest: dict, prop: dict, reuse: dict,
                    result_path: Path) -> dict:
    source_path = Path(reuse["result_path"])
    if ref.sha(source_path) != reuse["result_file_sha256"]:
        raise RuntimeError("optimized reusable result file changed")
    source = ref.verified(source_path)
    if source["record_sha256"] != reuse["result_record_sha256"]:
        raise RuntimeError("optimized reusable result record changed")
    query_paths = sorted(source_path.parent.joinpath("queries").glob(
        "query_*_result_v1.json"))
    queries = [ref.verified(path) for path in query_paths]
    inventory = _canonical([
        {"path": str(path), "file_sha256": ref.sha(path),
         "record_sha256": row["record_sha256"]}
        for path, row in zip(query_paths, queries)
    ])
    if inventory != reuse["query_inventory_sha256"]:
        raise RuntimeError("optimized reusable query inventory changed")
    deep_radius = float(prop["cached_DeepT_reference"][
        "certified_lower_endpoint_binary64"])
    proof = prop["frozen_proof_reference"]
    return _write(result_path, {
        "schema": "CORET_OPTIMIZED_HISTORICAL_127_PROPERTY_RESULT_V1",
        "terminal_status": "COMPLETE",
        "classification": "COMPLETE_CERTIFIED_RADIUS",
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        "property_id": prop["property_id"],
        "benchmark_ordinal": prop["benchmark_ordinal"],
        "sentence_ordinal": prop["sentence_ordinal"],
        "token_position": prop["token_position"],
        "certified_radius": source["certified_radius"],
        "uncertified_upper_radius": source["uncertified_upper_radius"],
        "cached_DeepT_certified_radius": deep_radius,
        "radius_ratio_to_cached_DeepT": source["certified_radius"] / deep_radius,
        "frozen_proof_carrying_reference_radius": None if proof is None else
            proof["certified_radius"],
        "radius_ratio_to_frozen_proof_carrying": None if proof is None else
            source["certified_radius"] / proof["certified_radius"],
        "verifier_evaluation_count": len(queries),
        "certified_query_count": source["certified_query_count"],
        "domain_failure_count": source["domain_failure_count"],
        "complete_certificate": True,
        "independent_checker_accepted": True,
        "checker_failure_count": 0,
        "soundness_or_proof_failure_count": 0,
        "generic_fallback_count": 0,
        "all_support_claims_validated": True,
        "all_provenance_consistent": True,
        "outward_fresh_radius_envelopes": True,
        "deterministic_support_reduction_lineage": True,
        "B2_QK_lifetime_fence_count": source["B2_QK_lifetime_fence_count"],
        "B2_QK_max_seconds": source["B2_QK_max_seconds"],
        "proof_generation_time_seconds": sum(
            float(row.get("bound_seconds") or 0.0) for row in queries),
        "independent_checker_time_seconds": sum(
            float(row.get("independent_checker_seconds") or 0.0) for row in queries),
        "total_wall_time_seconds": source["total_search_wall_seconds"],
        "peak_CPU_RSS_bytes": source["peak_CPU_RSS_bytes"],
        "peak_GPU_allocated_bytes": source["peak_GPU_allocated_bytes"],
        "peak_GPU_reserved_bytes": source["peak_GPU_reserved_bytes"],
        "reused_from_optimized_smoke": True,
        "optimized_smoke_result_path": str(source_path),
        "optimized_smoke_result_record_sha256": source["record_sha256"],
        "fresh_verifier_evaluations_this_run": 0,
    })


def run_property(property_id: str, authorized: bool) -> None:
    if not authorized:
        raise RuntimeError("explicit USER authorization required")
    manifest = validate_manifest()
    prop = _property(manifest, property_id)
    result_path = _property_root(property_id) / "result_v1.json"
    if result_path.exists():
        existing = ref.verified(result_path)
        if existing.get("terminal_status") != "COMPLETE":
            raise RuntimeError("existing property result is not complete")
        print(json.dumps({"terminal_status": "EXISTING_COMPLETE_PROPERTY_STOP",
                          "property_id": property_id,
                          "record_sha256": existing["record_sha256"]}, sort_keys=True))
        return
    reuse = _reuse_record(manifest, property_id)
    if reuse is not None:
        saved = _reuse_property(manifest, prop, reuse, result_path)
        print(json.dumps({"terminal_status": "COMPLETE_OPTIMIZED_SMOKE_REUSED",
                          "property_id": property_id,
                          "record_sha256": saved["record_sha256"]}, sort_keys=True))
        return
    while True:
        rows = _query_records(property_id)
        _sync_journal(property_id, rows)
        state = historical_smoke.search_state(rows)
        if state.get("done"):
            break
        _invoke_query(property_id, float(state["next_rho"]), len(rows))
    rows = _query_records(property_id)
    _sync_journal(property_id, rows)
    lower, upper = state["lower"], state["upper"]
    if lower is None:
        raise RuntimeError("property completed without certified radius")
    final = next(item for item in rows
                 if item["rho_binary64_hex"] == float(lower).hex())
    certificate = ref.verified(Path(final["operator_certificate_path"]))
    if (not final.get("complete_certificate")
            or not final.get("independent_checker_accepted")
            or certificate.get("generic_family_invocations") != 0):
        raise RuntimeError("final certificate is not accepted")
    if not all(item.get("independent_checker_accepted") for item in rows):
        raise RuntimeError("query checker rejection in property trajectory")
    if any(int(item.get("generic_fallback_count", 0)) for item in rows):
        raise RuntimeError("generic fallback in property trajectory")
    reached = sum(bool(item.get("reached_B2_QK")) for item in rows)
    fences = sum(int(item.get("B2_QK_lifetime_fence_count", 0)) for item in rows)
    if reached != fences:
        raise RuntimeError("B2 lifetime-fence trajectory count differs")
    b2_times = [float(item["B2_QK_seconds"]) for item in rows
                if item.get("B2_QK_seconds") is not None]
    if b2_times and max(b2_times) > B2_QK_PATHOLOGY_GUARD_SECONDS:
        raise RuntimeError("B2 QK allocator pathology returned")
    deep_radius = float(prop["cached_DeepT_reference"][
        "certified_lower_endpoint_binary64"])
    proof = prop["frozen_proof_reference"]
    saved = _write(result_path, {
        "schema": "CORET_OPTIMIZED_HISTORICAL_127_PROPERTY_RESULT_V1",
        "terminal_status": "COMPLETE",
        "classification": state["classification"],
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        "property_id": property_id,
        "benchmark_ordinal": prop["benchmark_ordinal"],
        "sentence_ordinal": prop["sentence_ordinal"],
        "token_position": prop["token_position"],
        "certified_radius": lower,
        "uncertified_upper_radius": upper,
        "cached_DeepT_certified_radius": deep_radius,
        "radius_ratio_to_cached_DeepT": lower / deep_radius,
        "frozen_proof_carrying_reference_radius": None if proof is None else
            proof["certified_radius"],
        "radius_ratio_to_frozen_proof_carrying": None if proof is None else
            lower / proof["certified_radius"],
        "verifier_evaluation_count": len(rows),
        "certified_query_count": sum(bool(item["certified"]) for item in rows),
        "domain_failure_count": sum(
            item["terminal_status"] == "UNCERTIFIED_DOMAIN_FAILURE" for item in rows),
        "complete_certificate": True,
        "independent_checker_accepted": True,
        "checker_failure_count": 0,
        "soundness_or_proof_failure_count": 0,
        "generic_fallback_count": 0,
        "all_support_claims_validated": True,
        "all_provenance_consistent": True,
        "outward_fresh_radius_envelopes": True,
        "deterministic_support_reduction_lineage": True,
        "B2_QK_lifetime_fence_count": fences,
        "B2_QK_max_seconds": max(b2_times) if b2_times else None,
        "proof_generation_time_seconds": sum(
            float(item.get("proof_generation_time_seconds") or 0.0)
            for item in rows),
        "independent_checker_time_seconds": sum(
            float(item.get("independent_checker_time_seconds") or 0.0)
            for item in rows),
        "total_wall_time_seconds": sum(
            float(item["total_wall_time_seconds"]) for item in rows),
        "peak_CPU_RSS_bytes": max(item["peak_CPU_RSS_bytes"] for item in rows),
        "peak_GPU_allocated_bytes": max(
            item["peak_GPU_allocated_bytes"] for item in rows),
        "peak_GPU_reserved_bytes": max(
            item["peak_GPU_reserved_bytes"] for item in rows),
        "reused_from_optimized_smoke": False,
        "fresh_verifier_evaluations_this_run": len(rows),
        "final_certificate_path": final["operator_certificate_path"],
        "final_certificate_record_sha256": final[
            "operator_certificate_record_sha256"],
    })
    print(json.dumps({"terminal_status": saved["terminal_status"],
                      "property_id": property_id,
                      "certified_radius": saved["certified_radius"],
                      "record_sha256": saved["record_sha256"]}, sort_keys=True))


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def summarize() -> None:
    manifest = validate_manifest()
    if SUMMARY.exists():
        existing = ref.verified(SUMMARY)
        print(json.dumps({"terminal_status": "EXISTING_COMPLETE_SUMMARY_STOP",
                          "record_sha256": existing["record_sha256"]}, sort_keys=True))
        return
    results = []
    for prop in manifest["properties"]:
        path = _property_root(prop["property_id"]) / "result_v1.json"
        if not path.exists():
            raise RuntimeError(f'incomplete property: {prop["property_id"]}')
        row = ref.verified(path)
        if (row.get("terminal_status") != "COMPLETE"
                or not row.get("independent_checker_accepted")
                or row.get("generic_fallback_count") != 0):
            raise RuntimeError(f'nonaccepted property: {prop["property_id"]}')
        results.append(row)
    ratios = [float(item["radius_ratio_to_cached_DeepT"]) for item in results]
    proof_ratios = [float(item["radius_ratio_to_frozen_proof_carrying"])
                    for item in results
                    if item["radius_ratio_to_frozen_proof_carrying"] is not None]
    optimized_radii = [float(item["certified_radius"]) for item in results]
    deep_radii = [float(item["cached_DeepT_certified_radius"]) for item in results]
    runtimes = [float(item["total_wall_time_seconds"]) for item in results]
    ranking = [{"property_id": item["property_id"],
                "ratio": item["radius_ratio_to_cached_DeepT"],
                "optimized_radius": item["certified_radius"],
                "cached_DeepT_radius": item["cached_DeepT_certified_radius"]}
               for item in results]
    saved = _write(SUMMARY, {
        "schema": "CORET_OPTIMIZED_HISTORICAL_127_SUMMARY_V1",
        "terminal_status": "OPTIMIZED_HISTORICAL_127_COMPLETE",
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        "completed_properties": len(results),
        "property_count": 127,
        "checker_failures": sum(item["checker_failure_count"] for item in results),
        "soundness_or_proof_failures": sum(
            item["soundness_or_proof_failure_count"] for item in results),
        "generic_fallbacks": sum(item["generic_fallback_count"] for item in results),
        "minimum_ratio_to_cached_DeepT": min(ratios),
        "median_ratio_to_cached_DeepT": statistics.median(ratios),
        "mean_ratio_to_cached_DeepT": statistics.fmean(ratios),
        "geometric_mean_ratio_to_cached_DeepT": math.exp(
            statistics.fmean(math.log(value) for value in ratios)),
        "ratios_to_cached_DeepT_ge_1.00": sum(value >= 1.0 for value in ratios),
        "ratios_to_cached_DeepT_ge_0.99": sum(value >= 0.99 for value in ratios),
        "ratios_to_cached_DeepT_ge_0.95": sum(value >= 0.95 for value in ratios),
        "frozen_proof_reference_count": len(proof_ratios),
        "minimum_ratio_to_frozen_proof": min(proof_ratios),
        "median_ratio_to_frozen_proof": statistics.median(proof_ratios),
        "mean_ratio_to_frozen_proof": statistics.fmean(proof_ratios),
        "optimized_mean_radius": statistics.fmean(optimized_radii),
        "optimized_median_radius": statistics.median(optimized_radii),
        "cached_DeepT_mean_radius": statistics.fmean(deep_radii),
        "cached_DeepT_median_radius": statistics.median(deep_radii),
        "runtime_mean_seconds": statistics.fmean(runtimes),
        "runtime_median_seconds": statistics.median(runtimes),
        "runtime_p95_seconds": _p95(runtimes),
        "total_wall_seconds": sum(runtimes),
        "worst_10_ratios": sorted(
            ranking, key=lambda item: (item["ratio"], item["property_id"]))[:10],
        "best_10_ratios": sorted(
            ranking, key=lambda item: (-item["ratio"], item["property_id"]))[:10],
        "reused_optimized_smoke_properties": sum(
            bool(item["reused_from_optimized_smoke"]) for item in results),
        "fresh_verifier_evaluations": sum(
            int(item["fresh_verifier_evaluations_this_run"]) for item in results),
        "DeepT_rerun_count": 0,
        "historical_duplicate_entries": manifest["historical_duplicate_entries"],
        "property_results": [{
            "benchmark_ordinal": item["benchmark_ordinal"],
            "property_id": item["property_id"],
            "result_record_sha256": item["record_sha256"],
            "certified_radius": item["certified_radius"],
            "cached_DeepT_certified_radius": item["cached_DeepT_certified_radius"],
            "ratio_to_cached_DeepT": item["radius_ratio_to_cached_DeepT"],
            "ratio_to_frozen_proof": item["radius_ratio_to_frozen_proof_carrying"],
        } for item in results],
    })
    print(json.dumps({"terminal_status": saved["terminal_status"],
                      "completed_properties": saved["completed_properties"],
                      "record_sha256": saved["record_sha256"]}, sort_keys=True))


def next_incomplete_property(manifest: dict) -> str | None:
    for prop in manifest["properties"]:
        path = _property_root(prop["property_id"]) / "result_v1.json"
        if not path.exists():
            return prop["property_id"]
        result = ref.verified(path)
        if result.get("terminal_status") != "COMPLETE":
            raise RuntimeError(f"noncomplete property blocks resume: {path}")
    return None


def benchmark(authorized: bool) -> None:
    if not authorized:
        raise RuntimeError("explicit USER authorization required")
    manifest = validate_manifest()
    preflight()
    while True:
        property_id = next_incomplete_property(manifest)
        if property_id is None:
            break
        subprocess.run([
            sys.executable, str(HERE), "property", "--authorized",
            "--property-id", property_id,
        ], cwd=WORKTREE, check=True, env={
            **os.environ,
            "PYTHONPATH": (
                "research_hab:/mnt/c/users/david-despacho/documents/"
                "vafali projects/lookahead-branching/research_hab"),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        })
    summarize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=(
        "freeze", "preflight", "query", "property", "summarize", "benchmark"))
    parser.add_argument("--authorized", action="store_true")
    parser.add_argument("--property-id")
    parser.add_argument("--rho")
    parser.add_argument("--query-ordinal", type=int)
    values = parser.parse_args()
    if values.mode == "freeze":
        freeze()
    elif values.mode == "preflight":
        preflight()
    elif values.mode == "query":
        if values.property_id is None or values.rho is None \
                or values.query_ordinal is None:
            parser.error("query requires property, rho, and query ordinal")
        generate_query(values.property_id, float.fromhex(values.rho),
                       values.query_ordinal, values.authorized)
    elif values.mode == "property":
        if values.property_id is None:
            parser.error("property requires property ID")
        run_property(values.property_id, values.authorized)
    elif values.mode == "summarize":
        summarize()
    else:
        benchmark(values.authorized)


if __name__ == "__main__":
    main()
