#!/usr/bin/env python3
"""Resumable proof-carrying evaluation of all 127 frozen historical properties.

This module is orchestration only.  The accepted bounded native-semantics
production graph and independent checker are imported unchanged.
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

import coret_bounded_native_execution_v1 as bounded
import coret_bounded_native_production_graph_v1 as graph
import coret_deept_exact_standard_ln_adapter as adapter
import coret_native_semantics_production_graph_v1 as native_graph
import coret_native_semantics_proof_v1 as native_proof
import coret_proof_carrying_historical_smoke_v1 as smoke
import coret_sound_reference_v1 as ref


ROOT = ref.ROOT
HERE = Path(__file__).resolve()
HISTORICAL_MANIFEST = ROOT / "research_hab/results/coret_deept_paper_benchmark_v5_20260917/coret_deept_paper_benchmark_execution_manifest_v5.json"
DEEPT_POSITIONS = ROOT / "research_hab/results/coret_deept_paper_benchmark_v5_20260917/deept_positions_v5.jsonl"
DEEPT_RESULT = ROOT / "research_hab/results/coret_deept_paper_benchmark_v5_20260917/deept_result_v5.json"
SMOKE_ROOT = ROOT / "research_hab/results/coret_proof_carrying_historical_smoke_v4_20260922"
SMOKE_MANIFEST = SMOKE_ROOT / "coret_proof_carrying_historical_smoke_manifest_v4.json"
SMOKE_SUMMARY = SMOKE_ROOT / "coret_proof_carrying_historical_smoke_summary_v4.json"
SMOKE_V3_ROOT = ROOT / "research_hab/results/coret_proof_carrying_historical_smoke_v3_20260922"
OUT = ROOT / "research_hab/results/coret_proof_carrying_historical_127_v1_20260923"
MANIFEST = OUT / "coret_proof_carrying_historical_127_manifest_v1.json"
PREFLIGHT = OUT / "coret_proof_carrying_historical_127_preflight_v1.json"
SUMMARY = OUT / "coret_proof_carrying_historical_127_summary_v1.json"
LAUNCHER = ROOT / "research_hab/run_coret_proof_carrying_historical_127_v1.sh"
TESTS = ROOT / "research_hab/tests/test_coret_proof_carrying_historical_127_v1.py"

EXPECTED_PARENT_CANONICAL_SHA = "7e4b2fea94424e554f07272aa8a246da7cda1212be83af57bbcdae7c870ed9dd"
EXPECTED_DEEPT_CACHE_SHA = "67e80d74cf83e5f810726405a6f6f0cf9928f4dda84d7f87decf59f354c8181d"
EXPECTED_SMOKE_SUMMARY_SHA = "b24126783f1d7370524d005003d003234bd135fb0f5458dcd69a87723d7042fd"
EXPECTED_SMOKE_CLASSIFICATION = "HISTORICAL_10_PROPERTY_SMOKE_COMPLETE"
INITIAL_RHO = 0.000625
MIDPOINT_ITERATIONS = 10
MAX_FACTOR_TWO_REFINEMENTS = 12


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def canonical_sha(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def cached_deept() -> dict[str, dict]:
    records = {}
    for row in load_jsonl(DEEPT_POSITIONS):
        property_id = row["property_id"]
        if property_id in records:
            raise RuntimeError(f"duplicate cached DeepT property: {property_id}")
        records[property_id] = row
    return records


def frozen_properties(parent: dict) -> list[dict]:
    properties = [dict(item) for item in parent["properties"]]
    if len(properties) != 127 or len({p["property_id"] for p in properties}) != 127:
        raise RuntimeError("historical property identity/count differs")
    if [p["sentence_ordinal"] for p in properties] != sorted(
            p["sentence_ordinal"] for p in properties):
        raise RuntimeError("historical property ordering differs")
    return properties


def smoke_query_records(property_id: str) -> list[dict]:
    records = []
    for root in (SMOKE_V3_ROOT, SMOKE_ROOT):
        directory = root / "properties" / property_id / "queries"
        if not directory.exists():
            continue
        for path in directory.glob("query_*_result_v1.json"):
            row = dict(ref.verified(path))
            row["_path"] = path
            records.append(row)
    records.sort(key=lambda x: int(x["query_ordinal"]))
    if records and [r["query_ordinal"] for r in records] != list(range(len(records))):
        raise RuntimeError(f"smoke query sequence differs for {property_id}")
    return records


def smoke_reuse_records(smoke_manifest: dict) -> list[dict]:
    reusable = []
    for selected in smoke_manifest["properties"]:
        property_id = selected["property_id"]
        result_path = SMOKE_ROOT / "properties" / property_id / "result_v1.json"
        result = ref.verified(result_path)
        if (result.get("terminal_status") != "COMPLETE"
                or result.get("classification") != "COMPLETE_CERTIFIED_RADIUS"
                or not result.get("complete_certificate")
                or not result.get("independent_checker_accepted")):
            raise RuntimeError(f"smoke result is not reusable: {property_id}")
        queries = smoke_query_records(property_id)
        if len(queries) != int(result["query_count"]):
            raise RuntimeError(f"smoke query count differs: {property_id}")
        final = result["final_certified_query"]
        certificate = ref.verified(ROOT / final["operator_certificate_path"])
        if (certificate.get("generic_family_invocations") != 0
                or not certificate.get("independent_checker_accepted")
                or not certificate.get("complete_certificate")):
            raise RuntimeError(f"smoke certificate is not reusable: {property_id}")
        reusable.append({
            "property_id": property_id,
            "result_path": str(result_path.relative_to(ROOT)),
            "result_file_sha256": ref.sha(result_path),
            "result_record_sha256": result["record_sha256"],
            "query_count": len(queries),
            "query_inventory_sha256": canonical_sha([
                {"path": str(q["_path"].relative_to(ROOT)),
                 "record_sha256": q["record_sha256"],
                 "file_sha256": ref.sha(q["_path"])} for q in queries
            ]),
            "certified_query_count": sum(bool(q["certified"]) for q in queries),
            "domain_failure_count": sum(
                q.get("reason_code") == "UNCERTIFIED_DOMAIN_FAILURE" for q in queries),
            "authoritative_uncertified_count": sum(
                not q["certified"] and q.get("authoritative_bound_returned") is not False
                for q in queries),
        })
    if len(reusable) != 10 or len({x["property_id"] for x in reusable}) != 10:
        raise RuntimeError("smoke reusable property count differs")
    return reusable


def source_files() -> tuple[Path, ...]:
    return (
        HERE, LAUNCHER, TESTS, Path(smoke.__file__).resolve(),
        Path(bounded.__file__).resolve(), Path(graph.__file__).resolve(),
        Path(adapter.__file__).resolve(), Path(native_graph.__file__).resolve(),
        Path(native_proof.__file__).resolve(),
        ROOT / "research_hab/coret_native_semantics_checker_v1.py",
        ROOT / "research_hab/coret_sound_reference_v1.py",
    )


def accepted_method_source_hashes(smoke_manifest: dict) -> dict[str, str]:
    paths = (
        Path(bounded.__file__).resolve(), Path(graph.__file__).resolve(),
        Path(adapter.__file__).resolve(), Path(native_graph.__file__).resolve(),
        Path(native_proof.__file__).resolve(),
        ROOT / "research_hab/coret_native_semantics_checker_v1.py",
        ROOT / "research_hab/coret_sound_reference_v1.py",
    )
    accepted = smoke_manifest["source_hashes"]
    result = {}
    for path in paths:
        relative = str(path.relative_to(ROOT))
        actual = ref.sha(path)
        if accepted.get(relative) != actual:
            raise RuntimeError(f"accepted method source changed: {relative}")
        result[relative] = actual
    return result


def resume_self_check() -> dict:
    records = [
        {"rho_binary64_hex": (0.000625).hex(), "certified": True},
        {"rho_binary64_hex": (0.00125).hex(), "certified": False},
        {"rho_binary64_hex": (0.0009375).hex(), "certified": True},
    ]
    state = smoke.search_state(records)
    expected = {"stage": "midpoint", "midpoint_index": 1,
                "lower": 0.0009375, "upper": 0.00125,
                "next_rho": 0.00109375}
    for key, value in expected.items():
        if state.get(key) != value:
            raise RuntimeError(f"resume scheduler self-check differs at {key}")
    return expected


def expected_manifest() -> dict:
    parent = ref.verified(HISTORICAL_MANIFEST, "canonical_manifest_sha256")
    if parent["canonical_manifest_sha256"] != EXPECTED_PARENT_CANONICAL_SHA:
        raise RuntimeError("frozen historical manifest canonical identity differs")
    if ref.sha(DEEPT_POSITIONS) != EXPECTED_DEEPT_CACHE_SHA:
        raise RuntimeError("frozen DeepT positions cache differs")
    if ref.sha(SMOKE_SUMMARY) != EXPECTED_SMOKE_SUMMARY_SHA:
        raise RuntimeError("accepted smoke summary differs")
    smoke_summary = ref.verified(SMOKE_SUMMARY)
    smoke_manifest = ref.verified(SMOKE_MANIFEST, "canonical_manifest_sha256")
    if (smoke_summary.get("classification") != EXPECTED_SMOKE_CLASSIFICATION
            or smoke_summary.get("completed_properties") != 10):
        raise RuntimeError("accepted smoke completion differs")
    deept_summary = ref.verified(DEEPT_RESULT)
    cache = cached_deept()
    properties = frozen_properties(parent)
    enriched = []
    for ordinal, prop in enumerate(properties):
        row = cache.get(prop["property_id"])
        if row is None or row.get("manifest_sha256") != EXPECTED_PARENT_CANONICAL_SHA:
            raise RuntimeError(f'missing frozen DeepT cache: {prop["property_id"]}')
        enriched.append({"benchmark_ordinal": ordinal, **prop,
                         "cached_DeepT_reference": row})
    reusable = smoke_reuse_records(smoke_manifest)
    method_hashes = accepted_method_source_hashes(smoke_manifest)
    return {
        "schema": "CORET_PROOF_CARRYING_HISTORICAL_127_MANIFEST_V1",
        "status": "FROZEN_BEFORE_PRODUCTION",
        "accepted_method_classification": "HISTORICAL_10_PROPERTY_SMOKE_PASS",
        "accepted_smoke_summary": {
            "path": str(SMOKE_SUMMARY.relative_to(ROOT)),
            "file_sha256": ref.sha(SMOKE_SUMMARY),
            "record_sha256": smoke_summary["record_sha256"],
            "manifest_path": str(SMOKE_MANIFEST.relative_to(ROOT)),
            "manifest_canonical_sha256": smoke_manifest["canonical_manifest_sha256"],
        },
        "historical_manifest": {
            "path": str(HISTORICAL_MANIFEST.relative_to(ROOT)),
            "canonical_sha256": parent["canonical_manifest_sha256"],
            "file_sha256": ref.sha(HISTORICAL_MANIFEST),
        },
        "cached_DeepT": {
            "rerun_permitted": False,
            "positions_path": str(DEEPT_POSITIONS.relative_to(ROOT)),
            "positions_file_sha256": ref.sha(DEEPT_POSITIONS),
            "positions_count": len(cache),
            "summary_path": str(DEEPT_RESULT.relative_to(ROOT)),
            "summary_file_sha256": ref.sha(DEEPT_RESULT),
            "summary_record_sha256": deept_summary["record_sha256"],
        },
        "properties": enriched,
        "property_count": len(enriched),
        "preserve_duplicate_sentence_structure": True,
        "smoke_reuse": {
            "property_count": len(reusable),
            "records": reusable,
            "scientific_reexecution": False,
        },
        "threat_model": {
            "norm_p": 100,
            "single_token_embedding_Linf": True,
            "source_dimension": 128,
        },
        "search": {
            "implementation": "accepted smoke.search_state",
            "initial_rho": INITIAL_RHO,
            "factor_two_bracketing": True,
            "maximum_factor_two_refinements": MAX_FACTOR_TWO_REFINEMENTS,
            "midpoint_iterations": MIDPOINT_ITERATIONS,
            "native_domain_failure_prefixes": list(smoke.NATIVE_DOMAIN_FAILURE_PREFIXES),
            "unexpected_exception_policy": "fatal",
            "reported_radius": "largest observed certified lower endpoint",
        },
        "production": {
            "implementation_revision": "BOUNDED_MEMORY_NATIVE_EXECUTION_V1",
            "pinned_DeepT_revision": native_proof.PINNED_REVISION,
            "model": parent["model"],
            "checkpoint_sha256": parent["checkpoint_sha256"],
            "required_invocation_counts": graph.EXPECTED_THREE_BLOCK_COUNTS,
            "generic_fallback_allowed": False,
            "autograd_enabled": False,
            "deterministic_algorithms": True,
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "accepted_method_source_hashes": method_hashes,
        },
        "telemetry": {
            "checker_timing": "timed wrapper around unchanged check_common call",
            "reused_smoke_checker_timing": "null because accepted smoke included checker in bound runtime but did not time separately",
            "runtime_p95": "nearest-rank ceil(0.95*n)-1 on sorted per-property total wall times",
            "memory": "existing non-invasive resource/CUDA peak counters",
        },
        "outputs": {
            "root": str(OUT.relative_to(ROOT)),
            "per_property": "properties/<property_id>/result_v1.json",
            "per_query": "properties/<property_id>/queries/query_<ordinal>_<rhohex>_result_v1.json",
            "per_query_certificate": "properties/<property_id>/queries/query_<ordinal>_<rhohex>_certificates_v1.json",
            "append_only_journal": "properties/<property_id>/query_events_v1.jsonl",
            "summary": str(SUMMARY.relative_to(ROOT)),
        },
        "source_hashes": {
            str(path.relative_to(ROOT)): ref.sha(path) for path in source_files()
        },
        "preparation_scientific_queries": 0,
        "preparation_bound_entrypoint_calls": 0,
    }


def validate_manifest() -> dict:
    stored = ref.verified(MANIFEST, "canonical_manifest_sha256")
    payload = dict(stored)
    payload.pop("canonical_manifest_sha256")
    if payload != expected_manifest():
        raise RuntimeError("historical-127 manifest differs from frozen inputs")
    return stored


def freeze() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        raise FileExistsError("immutable historical-127 manifest exists")
    saved = ref.write(MANIFEST, expected_manifest(), key="canonical_manifest_sha256")
    print(json.dumps({
        "terminal_status": "HISTORICAL_127_FROZEN_NO_SOLVE",
        "canonical_manifest_sha256": saved["canonical_manifest_sha256"],
        "property_count": saved["property_count"],
        "cached_DeepT_count": saved["cached_DeepT"]["positions_count"],
        "smoke_reuse_count": saved["smoke_reuse"]["property_count"],
        "scientific_query_count": 0,
        "bound_entrypoint_call_count": 0,
    }, sort_keys=True))


def preflight() -> None:
    manifest = validate_manifest()
    if (manifest["property_count"] != 127
            or manifest["cached_DeepT"]["positions_count"] != 127
            or manifest["smoke_reuse"]["property_count"] != 10):
        raise RuntimeError("historical-127 preflight count differs")
    duplicate_pair = [p for p in manifest["properties"]
                      if p["source_test_line"] == 1172 and p["token_position"] == 10]
    if len(duplicate_pair) != 2 or len({p["property_id"] for p in duplicate_pair}) != 2:
        raise RuntimeError("historical duplicate structure was not preserved")
    resume_state = resume_self_check()
    record = {
        "schema": "CORET_PROOF_CARRYING_HISTORICAL_127_PREFLIGHT_V1",
        "terminal_status": "PASS_NO_SOLVE",
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        "property_count": 127,
        "cached_DeepT_reference_count": 127,
        "smoke_reusable_property_count": 10,
        "historical_duplicate_structure_preserved": True,
        "resume_scheduler_self_check": resume_state,
        "checkpoint_loaded": False,
        "model_forward_calls": 0,
        "scientific_query_count": 0,
        "bound_entrypoint_call_count": 0,
    }
    if PREFLIGHT.exists():
        existing = ref.verified(PREFLIGHT)
        payload = dict(existing)
        payload.pop("record_sha256")
        if payload != record:
            raise RuntimeError("existing historical-127 preflight differs")
        status = "PREFLIGHT_REUSED_VERIFIED"
    else:
        existing = ref.write(PREFLIGHT, record)
        status = "PASS_NO_SOLVE"
    print(json.dumps({"terminal_status": status,
                      "record_sha256": existing["record_sha256"],
                      "property_count": 127,
                      "cached_DeepT_reference_count": 127,
                      "smoke_reuse_count": 10,
                      "scientific_query_count": 0,
                      "bound_entrypoint_call_count": 0}, sort_keys=True))


def property_record(manifest: dict, property_id: str) -> dict:
    rows = [p for p in manifest["properties"] if p["property_id"] == property_id]
    if len(rows) != 1:
        raise RuntimeError("property is not unique in frozen historical benchmark")
    return rows[0]


def historical_example(prop: dict) -> dict:
    parent = ref.verified(HISTORICAL_MANIFEST, "canonical_manifest_sha256")
    rows = [e for e in parent["examples"]
            if e["sentence_ordinal"] == prop["sentence_ordinal"]]
    if len(rows) != 1:
        raise RuntimeError("historical example is not unique")
    return rows[0]


def rho_tag(rho: float) -> str:
    return float(rho).hex().replace("+", "p").replace("-", "m").replace(".", "d")


def query_paths(property_id: str, ordinal: int, rho: float) -> tuple[Path, Path]:
    root = OUT / "properties" / property_id / "queries"
    stem = f"query_{ordinal:02d}_{rho_tag(rho)}"
    return root / f"{stem}_result_v1.json", root / f"{stem}_certificates_v1.json"


def domain_failure_result(manifest: dict, prop: dict, ordinal: int, rho: float,
                          error: AssertionError, started: float, bound_started: float,
                          checker_seconds: float, dispatch) -> dict:
    total = time.perf_counter() - started
    bound = time.perf_counter() - bound_started
    return {
        "schema": "CORET_PROOF_CARRYING_HISTORICAL_127_QUERY_RESULT_V1",
        "terminal_status": "UNCERTIFIED_DOMAIN_FAILURE",
        "reason_code": "UNCERTIFIED_DOMAIN_FAILURE",
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        "property_id": prop["property_id"],
        "query_ordinal": ordinal,
        "rho": float(rho),
        "rho_binary64_hex": float(rho).hex(),
        "direct_margin_interval": None,
        "nominal_margin": None,
        "certified": False,
        "authoritative_bound_returned": False,
        "complete_certificate": False,
        "independent_checker_accepted": False,
        "generic_fallback_count": 0,
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "partial_invocation_counts": {
            family: int(dispatch.counts[family]) for family in graph.FAMILIES
        },
        "proof_generation_time_seconds": max(0.0, bound - checker_seconds),
        "independent_checker_time_seconds": checker_seconds,
        "bound_runtime_seconds": bound,
        "total_wall_time_seconds": total,
        "peak_CPU_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "peak_GPU_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_GPU_reserved_bytes": torch.cuda.max_memory_reserved(),
        "fresh_verifier_evaluation_count": 1,
    }


def generate_query(property_id: str, rho: float, ordinal: int, authorized: bool) -> None:
    if not authorized:
        raise RuntimeError("explicit USER authorization required")
    manifest = validate_manifest()
    prop = property_record(manifest, property_id)
    result_path, certificate_path = query_paths(property_id, ordinal, rho)
    if result_path.exists() or certificate_path.exists():
        raise FileExistsError("immutable historical-127 query artifact exists")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("historical-127 query requires frozen CUDA execution")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda:0")
    started = time.perf_counter()
    modules = adapter.FrozenDeepTModules()
    from Verifiers.Zonotope import Zonotope
    model, _, configuration = adapter.load_native_model(
        modules, device, dtype=torch.float32)
    if configuration != manifest["production"]["model"]["configuration"]:
        raise RuntimeError("loaded architecture differs")
    example = historical_example(prop)
    ids = torch.tensor([example["token_ids"]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(ids, attention_mask=torch.ones_like(ids))[0]
        label = int(prop["clean_label"])
        nominal = float(logits[0, label] - logits[0, 1-label])
        pre = adapter.native_pre_layernorm_embeddings(model, ids)[0].detach()
    if int(logits.argmax(-1)[0]) != label:
        raise RuntimeError("nominal prediction differs")
    args = adapter.build_deept_args(modules, device)
    args.keep_intermediate_zonotopes = False
    z = Zonotope(args=args, p=100, eps=float(rho),
                 perturbed_word_index=int(prop["token_position"]), value=pre)
    checker_seconds = 0.0
    original_check = native_graph.check_common

    def timed_check(*values, **keywords):
        nonlocal checker_seconds
        tick = time.perf_counter()
        try:
            return original_check(*values, **keywords)
        finally:
            checker_seconds += time.perf_counter() - tick

    dispatch = graph.new_dispatch()
    bound_started = time.perf_counter()
    native_graph.check_common = timed_check
    try:
        try:
            with torch.no_grad():
                if torch.is_grad_enabled():
                    raise RuntimeError("historical-127 graph requires autograd disabled")
                margin, dispatch = graph.execute(
                    z, model, args, clean_label=label, dispatch=dispatch)
        except AssertionError as error:
            if not smoke.is_native_domain_failure(error):
                raise
            torch.cuda.synchronize()
            saved = ref.write(result_path, domain_failure_result(
                manifest, prop, ordinal, rho, error, started, bound_started,
                checker_seconds, dispatch))
            print(json.dumps({"terminal_status": saved["terminal_status"],
                              "property_id": property_id, "rho": rho,
                              "record_sha256": saved["record_sha256"]}, sort_keys=True))
            return
    finally:
        native_graph.check_common = original_check
    if margin.zonotope_w.requires_grad or margin.zonotope_w.grad_fn is not None:
        raise RuntimeError("historical-127 result retained autograd lineage")
    lower, upper = margin.concretize()
    torch.cuda.synchronize()
    bound_seconds = time.perf_counter() - bound_started
    if not bool(torch.isfinite(lower).all() and torch.isfinite(upper).all()
                and (lower <= upper).all()):
        raise RuntimeError("historical-127 authoritative interval invalid")
    counts = dispatch.assert_complete(graph.EXPECTED_THREE_BLOCK_COUNTS)
    certificate = ref.write(certificate_path, {
        "schema": "CORET_PROOF_CARRYING_HISTORICAL_127_QUERY_CERTIFICATES_V1",
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        "property_id": property_id,
        "query_ordinal": ordinal,
        "rho": float(rho),
        "rho_binary64_hex": float(rho).hex(),
        "invocation_counts": counts,
        "generic_family_invocations": dispatch.generic_family_invocations,
        "complete_certificate": True,
        "independent_checker_accepted": True,
        "certificates": dispatch.certificates,
        "scientific_query_count": 1,
        "bound_entrypoint_call_count": 1,
    })
    lo, hi = float(lower.min()), float(upper.max())
    saved = ref.write(result_path, {
        "schema": "CORET_PROOF_CARRYING_HISTORICAL_127_QUERY_RESULT_V1",
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
        "generic_fallback_count": dispatch.generic_family_invocations,
        "operator_certificate_path": str(certificate_path.relative_to(ROOT)),
        "operator_certificate_record_sha256": certificate["record_sha256"],
        "invocation_counts": counts,
        "proof_generation_time_seconds": max(0.0, bound_seconds - checker_seconds),
        "independent_checker_time_seconds": checker_seconds,
        "bound_runtime_seconds": bound_seconds,
        "total_wall_time_seconds": time.perf_counter() - started,
        "peak_CPU_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "peak_GPU_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_GPU_reserved_bytes": torch.cuda.max_memory_reserved(),
        "fresh_verifier_evaluation_count": 1,
    })
    print(json.dumps({"terminal_status": saved["terminal_status"],
                      "property_id": property_id, "rho": rho,
                      "certified": saved["certified"],
                      "record_sha256": saved["record_sha256"]}, sort_keys=True))


def query_records(property_id: str) -> list[dict]:
    directory = OUT / "properties" / property_id / "queries"
    if not directory.exists():
        return []
    rows = []
    for path in directory.glob("query_*_result_v1.json"):
        row = dict(ref.verified(path))
        if row.get("property_id") != property_id:
            raise RuntimeError("query property identity differs")
        row["_path"] = path
        rows.append(row)
    rows.sort(key=lambda x: int(x["query_ordinal"]))
    if [r["query_ordinal"] for r in rows] != list(range(len(rows))):
        raise RuntimeError("query sequence is not contiguous")
    if len({r["rho_binary64_hex"] for r in rows}) != len(rows):
        raise RuntimeError("duplicate radius in query sequence")
    return rows


def sync_journal(property_id: str, records: list[dict]) -> None:
    path = OUT / "properties" / property_id / "query_events_v1.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = [] if not path.exists() else load_jsonl(path)
    seen = {x["query_result_record_sha256"] for x in existing}
    with path.open("a", encoding="utf-8") as stream:
        for row in records:
            if row["record_sha256"] in seen:
                continue
            event = {
                "schema": "CORET_PROOF_CARRYING_HISTORICAL_127_QUERY_EVENT_V1",
                "property_id": property_id,
                "query_ordinal": row["query_ordinal"],
                "rho": row["rho"],
                "rho_binary64_hex": row["rho_binary64_hex"],
                "certified": row["certified"],
                "terminal_status": row["terminal_status"],
                "query_result_path": str(row["_path"].relative_to(ROOT)),
                "query_result_record_sha256": row["record_sha256"],
            }
            event["event_sha256"] = canonical_sha(event)
            stream.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            seen.add(row["record_sha256"])


def invoke_query(property_id: str, rho: float, ordinal: int) -> None:
    subprocess.run([
        sys.executable, str(HERE), "query", "--authorized",
        "--property-id", property_id, "--rho", float(rho).hex(),
        "--query-ordinal", str(ordinal),
    ], cwd=ROOT, check=True,
       env={**os.environ, "PYTHONPATH": "research_hab",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})


def reusable_smoke(manifest: dict, property_id: str) -> dict | None:
    rows = [x for x in manifest["smoke_reuse"]["records"]
            if x["property_id"] == property_id]
    if len(rows) > 1:
        raise RuntimeError("duplicate smoke reuse record")
    return rows[0] if rows else None


def reuse_smoke_property(manifest: dict, prop: dict, reuse: dict,
                         result_path: Path) -> dict:
    smoke_path = ROOT / reuse["result_path"]
    if ref.sha(smoke_path) != reuse["result_file_sha256"]:
        raise RuntimeError("smoke result file changed")
    source = ref.verified(smoke_path)
    if source["record_sha256"] != reuse["result_record_sha256"]:
        raise RuntimeError("smoke result record changed")
    queries = smoke_query_records(prop["property_id"])
    inventory = canonical_sha([
        {"path": str(q["_path"].relative_to(ROOT)),
         "record_sha256": q["record_sha256"],
         "file_sha256": ref.sha(q["_path"])} for q in queries
    ])
    if inventory != reuse["query_inventory_sha256"]:
        raise RuntimeError("smoke query inventory changed")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    return ref.write(result_path, {
        "schema": "CORET_PROOF_CARRYING_HISTORICAL_127_PROPERTY_RESULT_V1",
        "terminal_status": "COMPLETE",
        "classification": "COMPLETE_CERTIFIED_RADIUS",
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        "property_id": prop["property_id"],
        "benchmark_ordinal": prop["benchmark_ordinal"],
        "sentence_ordinal": prop["sentence_ordinal"],
        "token_position": prop["token_position"],
        "certified_radius": source["certified_radius"],
        "uncertified_upper_radius": source["uncertified_upper_radius"],
        "final_bracket_width": source["final_bracket_width"],
        "cached_DeepT_certified_radius": source["cached_DeepT_certified_radius"],
        "radius_ratio_to_cached_DeepT": source["radius_ratio_to_cached_DeepT"],
        "verifier_evaluation_count": len(queries),
        "certified_query_count": reuse["certified_query_count"],
        "domain_failure_count": reuse["domain_failure_count"],
        "authoritative_uncertified_count": reuse["authoritative_uncertified_count"],
        "complete_certificate": True,
        "independent_checker_accepted": True,
        "checker_failure_count": 0,
        "soundness_or_proof_failure_count": 0,
        "generic_fallback_count": 0,
        "proof_generation_time_seconds": sum(
            float(q.get("bound_runtime_seconds") or 0.0) for q in queries),
        "independent_checker_time_seconds": None,
        "independent_checker_time_separately_recorded": False,
        "total_wall_time_seconds": source["runtime_seconds"],
        "peak_CPU_RSS_bytes": source["peak_CPU_RSS_bytes"],
        "peak_GPU_allocated_bytes": source["peak_GPU_allocated_bytes"],
        "peak_GPU_reserved_bytes": source["peak_GPU_reserved_bytes"],
        "reused_from_accepted_smoke": True,
        "smoke_result_path": reuse["result_path"],
        "smoke_result_record_sha256": source["record_sha256"],
        "fresh_verifier_evaluations_this_run": 0,
    })


def run_property(property_id: str, authorized: bool) -> None:
    if not authorized:
        raise RuntimeError("explicit USER authorization required")
    manifest = validate_manifest()
    prop = property_record(manifest, property_id)
    result_path = OUT / "properties" / property_id / "result_v1.json"
    if result_path.exists():
        existing = ref.verified(result_path)
        print(json.dumps({"terminal_status": "EXISTING_COMPLETE_PROPERTY_STOP",
                          "property_id": property_id,
                          "record_sha256": existing["record_sha256"]}, sort_keys=True))
        return
    reuse = reusable_smoke(manifest, property_id)
    if reuse is not None:
        saved = reuse_smoke_property(manifest, prop, reuse, result_path)
        print(json.dumps({"terminal_status": "COMPLETE_SMOKE_RESULT_REUSED",
                          "property_id": property_id,
                          "record_sha256": saved["record_sha256"]}, sort_keys=True))
        return
    while True:
        records = query_records(property_id)
        sync_journal(property_id, records)
        state = smoke.search_state(records)
        if state.get("done"):
            break
        invoke_query(property_id, float(state["next_rho"]), len(records))
    records = query_records(property_id)
    sync_journal(property_id, records)
    lower_rho, upper_rho = state["lower"], state["upper"]
    if lower_rho is None:
        raise RuntimeError("property completed without a certified radius")
    final = next(q for q in records
                 if q["rho_binary64_hex"] == float(lower_rho).hex())
    certificate = ref.verified(ROOT / final["operator_certificate_path"])
    if (not final["complete_certificate"]
            or not final["independent_checker_accepted"]
            or certificate.get("generic_family_invocations") != 0):
        raise RuntimeError("final claimed certificate is not independently accepted")
    deept_radius = float(prop["cached_DeepT_reference"]["certified_lower_endpoint_binary64"])
    checker_values = [float(q.get("independent_checker_time_seconds") or 0.0)
                      for q in records]
    saved = ref.write(result_path, {
        "schema": "CORET_PROOF_CARRYING_HISTORICAL_127_PROPERTY_RESULT_V1",
        "terminal_status": "COMPLETE",
        "classification": state["classification"],
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        "property_id": property_id,
        "benchmark_ordinal": prop["benchmark_ordinal"],
        "sentence_ordinal": prop["sentence_ordinal"],
        "token_position": prop["token_position"],
        "certified_radius": lower_rho,
        "uncertified_upper_radius": upper_rho,
        "final_bracket_width": None if upper_rho is None else upper_rho - lower_rho,
        "cached_DeepT_certified_radius": deept_radius,
        "radius_ratio_to_cached_DeepT": lower_rho / deept_radius,
        "verifier_evaluation_count": len(records),
        "certified_query_count": sum(bool(q["certified"]) for q in records),
        "domain_failure_count": sum(
            q.get("reason_code") == "UNCERTIFIED_DOMAIN_FAILURE" for q in records),
        "authoritative_uncertified_count": sum(
            not q["certified"] and q.get("authoritative_bound_returned") is True
            for q in records),
        "complete_certificate": True,
        "independent_checker_accepted": True,
        "checker_failure_count": 0,
        "soundness_or_proof_failure_count": 0,
        "generic_fallback_count": sum(int(q.get("generic_fallback_count") or 0)
                                      for q in records),
        "proof_generation_time_seconds": sum(
            float(q.get("proof_generation_time_seconds") or 0.0) for q in records),
        "independent_checker_time_seconds": sum(checker_values),
        "independent_checker_time_separately_recorded": True,
        "total_wall_time_seconds": sum(
            float(q.get("total_wall_time_seconds") or 0.0) for q in records),
        "peak_CPU_RSS_bytes": max(q["peak_CPU_RSS_bytes"] for q in records),
        "peak_GPU_allocated_bytes": max(q["peak_GPU_allocated_bytes"] for q in records),
        "peak_GPU_reserved_bytes": max(q["peak_GPU_reserved_bytes"] for q in records),
        "reused_from_accepted_smoke": False,
        "fresh_verifier_evaluations_this_run": len(records),
        "final_certificate_path": final["operator_certificate_path"],
        "final_certificate_record_sha256": final["operator_certificate_record_sha256"],
    })
    print(json.dumps({"terminal_status": saved["terminal_status"],
                      "property_id": property_id,
                      "certified_radius": saved["certified_radius"],
                      "record_sha256": saved["record_sha256"]}, sort_keys=True))


def percentile95(values: list[float]) -> float:
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
        path = OUT / "properties" / prop["property_id"] / "result_v1.json"
        if not path.exists():
            raise RuntimeError(f'incomplete property: {prop["property_id"]}')
        result = ref.verified(path)
        if result.get("terminal_status") != "COMPLETE":
            raise RuntimeError(f'noncomplete property: {prop["property_id"]}')
        results.append(result)
    ratios = [float(x["radius_ratio_to_cached_DeepT"]) for x in results]
    proof_radii = [float(x["certified_radius"]) for x in results]
    deept_radii = [float(x["cached_DeepT_certified_radius"]) for x in results]
    runtimes = [float(x["total_wall_time_seconds"]) for x in results]
    ranking = [{"property_id": x["property_id"],
                "ratio": x["radius_ratio_to_cached_DeepT"],
                "proof_carrying_radius": x["certified_radius"],
                "cached_DeepT_radius": x["cached_DeepT_certified_radius"]}
               for x in results]
    worst = sorted(ranking, key=lambda x: (x["ratio"], x["property_id"]))[:10]
    best = sorted(ranking, key=lambda x: (-x["ratio"], x["property_id"]))[:10]
    saved = ref.write(SUMMARY, {
        "schema": "CORET_PROOF_CARRYING_HISTORICAL_127_SUMMARY_V1",
        "terminal_status": "COMPLETE",
        "classification": "HISTORICAL_127_BENCHMARK_COMPLETE",
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        "completed_properties": len(results),
        "property_count": 127,
        "checker_failures": sum(x["checker_failure_count"] for x in results),
        "soundness_or_proof_failures": sum(
            x["soundness_or_proof_failure_count"] for x in results),
        "generic_fallbacks": sum(x["generic_fallback_count"] for x in results),
        "minimum_paired_radius_ratio": min(ratios),
        "median_paired_radius_ratio": statistics.median(ratios),
        "mean_paired_radius_ratio": statistics.fmean(ratios),
        "geometric_mean_paired_radius_ratio": math.exp(
            statistics.fmean(math.log(x) for x in ratios)),
        "ratios_ge_1.00": sum(x >= 1.00 for x in ratios),
        "ratios_ge_0.99": sum(x >= 0.99 for x in ratios),
        "ratios_ge_0.95": sum(x >= 0.95 for x in ratios),
        "proof_carrying_mean_radius": statistics.fmean(proof_radii),
        "proof_carrying_median_radius": statistics.median(proof_radii),
        "cached_DeepT_mean_radius": statistics.fmean(deept_radii),
        "cached_DeepT_median_radius": statistics.median(deept_radii),
        "runtime_mean_seconds": statistics.fmean(runtimes),
        "runtime_median_seconds": statistics.median(runtimes),
        "runtime_p95_seconds": percentile95(runtimes),
        "worst_10_paired_ratios": worst,
        "best_10_paired_ratios": best,
        "reused_smoke_properties": sum(x["reused_from_accepted_smoke"] for x in results),
        "fresh_verifier_evaluations": sum(
            x["fresh_verifier_evaluations_this_run"] for x in results),
        "DeepT_rerun_count": 0,
        "property_results": [{
            "property_id": x["property_id"],
            "result_record_sha256": x["record_sha256"],
            "certified_radius": x["certified_radius"],
            "cached_DeepT_certified_radius": x["cached_DeepT_certified_radius"],
            "ratio": x["radius_ratio_to_cached_DeepT"],
        } for x in results],
    })
    print(json.dumps({"terminal_status": saved["terminal_status"],
                      "classification": saved["classification"],
                      "record_sha256": saved["record_sha256"]}, sort_keys=True))


def benchmark(authorized: bool) -> None:
    if not authorized:
        raise RuntimeError("explicit USER authorization required")
    manifest = validate_manifest()
    preflight()
    for prop in manifest["properties"]:
        subprocess.run([sys.executable, str(HERE), "property", "--authorized",
                        "--property-id", prop["property_id"]],
                       cwd=ROOT, check=True,
                       env={**os.environ, "PYTHONPATH": "research_hab",
                            "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    summarize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("freeze", "preflight", "query", "property",
                                         "summarize", "benchmark"))
    parser.add_argument("--authorized", action="store_true")
    parser.add_argument("--property-id")
    parser.add_argument("--rho")
    parser.add_argument("--query-ordinal", type=int)
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze()
    elif args.mode == "preflight":
        preflight()
    elif args.mode == "query":
        if args.property_id is None or args.rho is None or args.query_ordinal is None:
            parser.error("query requires --property-id, --rho, --query-ordinal")
        generate_query(args.property_id, float.fromhex(args.rho),
                       args.query_ordinal, args.authorized)
    elif args.mode == "property":
        if args.property_id is None:
            parser.error("property requires --property-id")
        run_property(args.property_id, args.authorized)
    elif args.mode == "summarize":
        summarize()
    else:
        benchmark(args.authorized)


if __name__ == "__main__":
    main()
