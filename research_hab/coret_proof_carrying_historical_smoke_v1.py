#!/usr/bin/env python3
"""Frozen ten-property proof-carrying historical smoke benchmark.

The benchmark is orchestration around the accepted bounded native-semantics
production graph.  It never invokes DeepT: all DeepT comparisons are loaded
from the immutable V5 cache.
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

import coret_bounded_native_execution_v1 as bounded
import coret_bounded_native_production_graph_v1 as graph
import coret_deept_exact_standard_ln_adapter as adapter
import coret_native_semantics_proof_v1 as native_proof
import coret_sound_reference_v1 as ref


ROOT = ref.ROOT
HERE = Path(__file__).resolve()
HISTORICAL_MANIFEST = ROOT / "research_hab/results/coret_deept_paper_benchmark_v5_20260917/coret_deept_paper_benchmark_execution_manifest_v5.json"
DEEPT_POSITIONS = ROOT / "research_hab/results/coret_deept_paper_benchmark_v5_20260917/deept_positions_v5.jsonl"
DEEPT_RESULT = ROOT / "research_hab/results/coret_deept_paper_benchmark_v5_20260917/deept_result_v5.json"
ACCEPTED_TOK10_MANIFEST = ROOT / "research_hab/results/coret_bounded_native_tok10_v1_20260922/coret_bounded_native_tok10_manifest_recenter_fix_v1.json"
ACCEPTED_TOK10_RESULT = ROOT / "research_hab/results/coret_bounded_native_tok10_v1_20260922/coret_bounded_native_tok10_result_v1.json"
ACCEPTED_TOK10_CERTIFICATE = ROOT / "research_hab/results/coret_bounded_native_tok10_v1_20260922/coret_bounded_native_tok10_certificates_v1.json"
PRIOR_OUT = ROOT / "research_hab/results/coret_proof_carrying_historical_smoke_v3_20260922"
PRIOR_MANIFEST = PRIOR_OUT / "coret_proof_carrying_historical_smoke_manifest_v3.json"
OUT = ROOT / "research_hab/results/coret_proof_carrying_historical_smoke_v4_20260922"
MANIFEST = OUT / "coret_proof_carrying_historical_smoke_manifest_v4.json"
PREFLIGHT = OUT / "coret_proof_carrying_historical_smoke_preflight_v4.json"
SUMMARY = OUT / "coret_proof_carrying_historical_smoke_summary_v4.json"
LAUNCHER = ROOT / "research_hab/run_coret_proof_carrying_historical_smoke_v2.sh"
TESTS = ROOT / "research_hab/tests/test_coret_proof_carrying_historical_smoke_v1.py"

SELECTION_RULE = "upper_median_eligible_token_position"
INITIAL_RHO = 1.0 / 1600.0
MAX_FACTOR_TWO_REFINEMENTS = 12
MIDPOINT_ITERATIONS = 10
EXPECTED_PARENT_SHA = "7e4b2fea94424e554f07272aa8a246da7cda1212be83af57bbcdae7c870ed9dd"
EXPECTED_ACCEPTED_TOK10_SHA = "d9a3e24f4789db58a7e78e0d95ebcd2869c7db08ed2a1816397f8a300bb460f6"
NATIVE_DOMAIN_FAILURE_PREFIXES = (
    "sqrt: Bounds must be positive",
    "reciprocal: Bounds must be positive",
)
OBSERVED_DOMAIN_FAILURES = (
    {
        "property_id": "deept_table7_stdln3_s000_line504_tok10",
        "rho": 0.00125,
        "exception_type": "AssertionError",
        "exception_message": "sqrt: Bounds must be positive",
        "origin": "user-observed V2 smoke query before authoritative bound",
    },
    {
        "property_id": "deept_table7_stdln3_s003_line2031_tok08",
        "rho": 0.001875,
        "exception_type": "AssertionError",
        "exception_message": (
            "reciprocal: Bounds must be positive but 6 elements out of 1156 "
            "were < 1e-12 (min value = 0.000000000000, iszero = True)"
        ),
        "origin": "user-observed V3 smoke query before authoritative bound",
    },
)


def canonical_payload_sha(value: dict) -> str:
    import hashlib
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def frozen_selection(parent: dict) -> list[dict]:
    """Select one property per sentence without consulting any result."""
    properties = {p["property_id"]: p for p in parent["properties"]}
    selected = []
    examples = sorted(parent["examples"], key=lambda x: x["sentence_ordinal"])
    if [e["sentence_ordinal"] for e in examples] != list(range(10)):
        raise RuntimeError("historical manifest does not contain sentence ordinals 0..9")
    for example in examples:
        eligible = list(example["eligible_token_positions"])
        if not eligible or eligible != sorted(eligible):
            raise RuntimeError("eligible token positions are empty or noncanonical")
        token_position = int(eligible[len(eligible) // 2])
        property_id = f'{example["sentence_id"]}_tok{token_position:02d}'
        prop = properties.get(property_id)
        if prop is None:
            raise RuntimeError(f"selected property absent: {property_id}")
        if int(prop["sentence_ordinal"]) != int(example["sentence_ordinal"]):
            raise RuntimeError("selected property sentence mismatch")
        selected.append({
            "selection_ordinal": len(selected),
            "sentence_ordinal": int(example["sentence_ordinal"]),
            "sentence_id": example["sentence_id"],
            "property_id": property_id,
            "token_position": token_position,
            "token": prop["token"],
            "token_id": int(prop["token_id"]),
            "sequence_length": int(prop["sequence_length"]),
            "source_dimension": int(prop["source_dimension"]),
            "source_test_line": int(prop["source_test_line"]),
            "clean_label": int(prop["clean_label"]),
            "nominal_prediction": int(prop["nominal_prediction"]),
            "raw_sentence_sha256": prop["raw_sentence_sha256"],
        })
    if len({x["sentence_ordinal"] for x in selected}) != 10:
        raise RuntimeError("selection is not one property per sentence")
    return selected


def cached_deept_records() -> dict[str, dict]:
    records = {}
    for record in load_jsonl(DEEPT_POSITIONS):
        property_id = record["property_id"]
        if property_id in records:
            raise RuntimeError(f"duplicate cached DeepT record: {property_id}")
        records[property_id] = record
    return records


def source_files() -> tuple[Path, ...]:
    return (
        HERE, LAUNCHER, TESTS,
        Path(bounded.__file__).resolve(), Path(graph.__file__).resolve(),
        Path(adapter.__file__).resolve(), Path(native_proof.__file__).resolve(),
        ROOT / "research_hab/coret_native_semantics_checker_v1.py",
        ROOT / "research_hab/coret_native_semantics_production_graph_v1.py",
        ROOT / "research_hab/coret_sound_reference_v1.py",
    )


def prior_state_inventory() -> dict:
    prior_manifest = ref.verified(PRIOR_MANIFEST, "canonical_manifest_sha256")
    if prior_manifest["canonical_manifest_sha256"] != (
            "775380b854917149f2c9570942fc3783164fad6001c6eb7612d62663f0cd0a26"):
        raise RuntimeError("prior smoke manifest identity differs")
    files = []
    for path in sorted(PRIOR_OUT.rglob("*")):
        if path.is_file():
            files.append({"path": str(path.relative_to(ROOT)), "sha256": ref.sha(path)})
    completed = []
    partial = []
    properties_root = PRIOR_OUT / "properties"
    if properties_root.exists():
        for directory in sorted(x for x in properties_root.iterdir() if x.is_dir()):
            result = directory / "result_v1.json"
            queries = sorted((directory / "queries").glob("query_*_result_v1.json"))
            item = {"property_id": directory.name, "query_count": len(queries)}
            if result.exists():
                record = ref.verified(result)
                item["result_record_sha256"] = record["record_sha256"]
                completed.append(item)
            else:
                item["query_record_sha256s"] = [ref.verified(q)["record_sha256"]
                                                  for q in queries]
                partial.append(item)
    return {
        "manifest_canonical_sha256": prior_manifest["canonical_manifest_sha256"],
        "file_count": len(files),
        "inventory_sha256": canonical_payload_sha({"files": files}),
        "completed_properties": completed,
        "partial_properties": partial,
    }


def expected_manifest() -> dict:
    parent = ref.verified(HISTORICAL_MANIFEST, "canonical_manifest_sha256")
    if parent["canonical_manifest_sha256"] != EXPECTED_PARENT_SHA:
        raise RuntimeError("historical manifest identity differs")
    deept_summary = ref.verified(DEEPT_RESULT)
    accepted_manifest = ref.verified(ACCEPTED_TOK10_MANIFEST,
                                     "canonical_manifest_sha256")
    accepted_result = ref.verified(ACCEPTED_TOK10_RESULT)
    accepted_certificate = ref.verified(ACCEPTED_TOK10_CERTIFICATE)
    if accepted_manifest["canonical_manifest_sha256"] != EXPECTED_ACCEPTED_TOK10_SHA:
        raise RuntimeError("accepted tok10 execution identity differs")
    if (accepted_result.get("terminal_status") != "COMPLETE"
            or not accepted_result.get("certified")
            or accepted_result.get("rho") != INITIAL_RHO
            or accepted_result.get("property_id") != "deept_table7_stdln3_s000_line504_tok10"):
        raise RuntimeError("accepted tok10 seed is not reusable")
    if accepted_certificate.get("record_sha256") != accepted_result.get(
            "operator_certificate_record_sha256"):
        raise RuntimeError("accepted tok10 certificate/result linkage differs")
    selection = frozen_selection(parent)
    cached = cached_deept_records()
    for item in selection:
        record = cached.get(item["property_id"])
        if record is None or record.get("manifest_sha256") != EXPECTED_PARENT_SHA:
            raise RuntimeError(f'missing cached DeepT reference: {item["property_id"]}')
        item["cached_DeepT_reference"] = record
    return {
        "schema": "CORET_PROOF_CARRYING_HISTORICAL_SMOKE_MANIFEST_V4",
        "status": "FROZEN_BEFORE_BENCHMARK",
        "frozen_method_status": "END_TO_END_PROOF_CARRYING_GO",
        "selection_rule": {
            "name": SELECTION_RULE,
            "definition": "for each sentence, sort the frozen eligible positions and select positions[len(positions)//2]",
            "uses_scientific_results": False,
        },
        "properties": selection,
        "property_count": 10,
        "threat_model": {
            "norm_p": 100,
            "perturbed_tokens": 1,
            "source_dimension": 128,
            "embedding_Linf": True,
        },
        "radius_search": {
            "initial_rho": INITIAL_RHO,
            "factor_two_bracketing": True,
            "maximum_factor_two_refinements": MAX_FACTOR_TWO_REFINEMENTS,
            "midpoint_iterations": MIDPOINT_ITERATIONS,
            "reported_value": "largest observed certified lower endpoint",
            "oracle_certified_iff": "finite authoritative direct-margin lower bound > 0",
            "unexpected_exception_policy": "fail_closed",
        },
        "accepted_tok10_seed_reuse": {
            "scientifically_identical_property_and_radius": True,
            "property_id": accepted_result["property_id"],
            "rho": accepted_result["rho"],
            "manifest_path": str(ACCEPTED_TOK10_MANIFEST.relative_to(ROOT)),
            "manifest_canonical_sha256": accepted_manifest["canonical_manifest_sha256"],
            "manifest_file_sha256": ref.sha(ACCEPTED_TOK10_MANIFEST),
            "result_path": str(ACCEPTED_TOK10_RESULT.relative_to(ROOT)),
            "result_record_sha256": accepted_result["record_sha256"],
            "result_file_sha256": ref.sha(ACCEPTED_TOK10_RESULT),
            "certificate_path": str(ACCEPTED_TOK10_CERTIFICATE.relative_to(ROOT)),
            "certificate_record_sha256": accepted_certificate["record_sha256"],
            "certificate_file_sha256": ref.sha(ACCEPTED_TOK10_CERTIFICATE),
        },
        "recovered_native_domain_failures": [
            {
                **failure,
                "oracle_status": "UNCERTIFIED_DOMAIN_FAILURE",
                "authoritative_bound_returned": False,
                "native_semantics": (
                    "pinned operator raises AssertionError; pinned "
                    "VerifierZonotope.get_bounds_difference_in_scores catches it and "
                    "returns None; verify_safety maps None to False"
                ),
                "rerun_required": False,
            }
            for failure in OBSERVED_DOMAIN_FAILURES
        ],
        "prior_smoke_state_reuse": prior_state_inventory(),
        "cached_DeepT_only": {
            "rerun_permitted": False,
            "manifest_path": str(HISTORICAL_MANIFEST.relative_to(ROOT)),
            "manifest_canonical_sha256": parent["canonical_manifest_sha256"],
            "manifest_file_sha256": ref.sha(HISTORICAL_MANIFEST),
            "positions_path": str(DEEPT_POSITIONS.relative_to(ROOT)),
            "positions_file_sha256": ref.sha(DEEPT_POSITIONS),
            "summary_path": str(DEEPT_RESULT.relative_to(ROOT)),
            "summary_record_sha256": deept_summary["record_sha256"],
            "summary_file_sha256": ref.sha(DEEPT_RESULT),
        },
        "production": {
            "implementation_revision": "BOUNDED_MEMORY_NATIVE_EXECUTION_V1",
            "pinned_DeepT_revision": native_proof.PINNED_REVISION,
            "model": parent["model"],
            "checkpoint_sha256": parent["checkpoint_sha256"],
            "required_invocation_counts": graph.EXPECTED_THREE_BLOCK_COUNTS,
            "generic_semantic_remainder_allowed": False,
            "autograd_enabled": False,
            "deterministic_algorithms": True,
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        },
        "result_schema": {
            "per_query": [
                "property_id", "rho", "direct_margin_interval", "nominal_margin",
                "certified", "independent_checker_accepted", "complete_certificate",
                "runtime_seconds", "peak_CPU_RSS_bytes", "peak_GPU_allocated_bytes",
                "peak_GPU_reserved_bytes", "cached_DeepT_reference",
            ],
            "per_property": [
                "certified_radius", "final_certified_query", "final_uncertified_query",
                "complete_certificate", "independent_checker_accepted",
                "runtime_seconds", "peak_CPU_RSS_bytes", "peak_GPU_allocated_bytes",
                "peak_GPU_reserved_bytes", "cached_DeepT_certified_radius",
                "radius_ratio_to_cached_DeepT",
            ],
            "aggregate": [
                "completed_properties", "certified_properties", "fresh_proof_evaluations",
                "reused_proof_evaluations", "total_runtime_seconds",
            ],
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
        raise RuntimeError("historical smoke manifest differs from frozen inputs")
    return stored


def freeze() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        raise FileExistsError("immutable historical smoke manifest exists")
    saved = ref.write(MANIFEST, expected_manifest(), key="canonical_manifest_sha256")
    print(json.dumps({
        "terminal_status": "HISTORICAL_10_PROPERTY_SMOKE_FROZEN_NO_SOLVE",
        "canonical_manifest_sha256": saved["canonical_manifest_sha256"],
        "property_count": 10,
        "scientific_query_count": 0,
        "bound_entrypoint_call_count": 0,
    }, sort_keys=True))


def preflight() -> None:
    manifest = validate_manifest()
    properties = manifest["properties"]
    if len(properties) != 10 or [p["sentence_ordinal"] for p in properties] != list(range(10)):
        raise RuntimeError("frozen smoke selection differs")
    record = {
        "schema": "CORET_PROOF_CARRYING_HISTORICAL_SMOKE_PREFLIGHT_V1",
        "terminal_status": "PASS_NO_SOLVE",
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        "property_ids": [p["property_id"] for p in properties],
        "property_count": 10,
        "cached_DeepT_records_verified": 10,
        "accepted_tok10_seed_verified": True,
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
            raise RuntimeError("existing smoke preflight differs")
        status = "PREFLIGHT_REUSED_VERIFIED"
    else:
        existing = ref.write(PREFLIGHT, record)
        status = "PASS_NO_SOLVE"
    print(json.dumps({
        "terminal_status": status,
        "record_sha256": existing["record_sha256"],
        "property_count": 10,
        "scientific_query_count": 0,
        "bound_entrypoint_call_count": 0,
    }, sort_keys=True))


def property_manifest_record(manifest: dict, property_id: str) -> dict:
    matches = [x for x in manifest["properties"] if x["property_id"] == property_id]
    if len(matches) != 1:
        raise RuntimeError("property is not in frozen smoke selection")
    return matches[0]


def historical_example(property_record: dict) -> dict:
    parent = ref.verified(HISTORICAL_MANIFEST, "canonical_manifest_sha256")
    matches = [x for x in parent["examples"]
               if int(x["sentence_ordinal"]) == int(property_record["sentence_ordinal"])]
    if len(matches) != 1:
        raise RuntimeError("historical example identity is not unique")
    return matches[0]


def rho_tag(rho: float) -> str:
    return float(rho).hex().replace("+", "p").replace("-", "m").replace(".", "d")


def query_paths(property_id: str, ordinal: int, rho: float) -> tuple[Path, Path]:
    base = OUT / "properties" / property_id / "queries"
    stem = f"query_{ordinal:02d}_{rho_tag(rho)}"
    return base / f"{stem}_result_v1.json", base / f"{stem}_certificates_v1.json"


def accepted_seed_record(manifest: dict, prop: dict, ordinal: int) -> dict:
    source = ref.verified(ACCEPTED_TOK10_RESULT)
    certificate = ref.verified(ACCEPTED_TOK10_CERTIFICATE)
    return {
        "schema": "CORET_PROOF_CARRYING_SMOKE_QUERY_RESULT_V1",
        "terminal_status": "COMPLETE_REUSED_ACCEPTED_PROOF",
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        "property_id": prop["property_id"],
        "query_ordinal": ordinal,
        "rho": INITIAL_RHO,
        "rho_binary64_hex": INITIAL_RHO.hex(),
        "direct_margin_interval": source["direct_margin_interval"],
        "nominal_margin": source["nominal_margin"],
        "certified": bool(source["certified"]),
        "independent_checker_accepted": True,
        "complete_certificate": True,
        "operator_certificate_record_sha256": certificate["record_sha256"],
        "operator_certificate_path": str(ACCEPTED_TOK10_CERTIFICATE.relative_to(ROOT)),
        "invocation_counts": source["invocation_counts"],
        "runtime_seconds": source["total_runtime_seconds"],
        "bound_runtime_seconds": source["bound_runtime_seconds"],
        "peak_CPU_RSS_bytes": source["peak_CPU_RSS_bytes"],
        "peak_GPU_allocated_bytes": source["peak_GPU_allocated_bytes"],
        "peak_GPU_reserved_bytes": source["peak_GPU_reserved_bytes"],
        "all_outputs_finite": source["all_outputs_finite"],
        "cached_DeepT_reference": prop["cached_DeepT_reference"],
        "fresh_proof_evaluation_count": 0,
        "reused_proof_evaluation_count": 1,
        "scientific_query_already_consumed_before_smoke": 1,
        "bound_entrypoint_calls_during_smoke": 0,
        "reused_result_record_sha256": source["record_sha256"],
    }


def is_native_domain_failure(error: BaseException) -> bool:
    return (type(error) is AssertionError
            and any(str(error).startswith(prefix)
                    for prefix in NATIVE_DOMAIN_FAILURE_PREFIXES))


def domain_failure_record(manifest: dict, prop: dict, ordinal: int, rho: float,
                          *, exception_message: str, runtime_seconds: float,
                          bound_runtime_seconds: float | None,
                          peak_cpu: int, peak_allocated: int, peak_reserved: int,
                          fresh: int, reused: int, origin: str,
                          nominal_margin: float | None = None,
                          partial_invocation_counts: dict | None = None) -> dict:
    return {
        "schema": "CORET_PROOF_CARRYING_SMOKE_QUERY_RESULT_V1",
        "terminal_status": "UNCERTIFIED_DOMAIN_FAILURE",
        "reason_code": "UNCERTIFIED_DOMAIN_FAILURE",
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        "property_id": prop["property_id"],
        "query_ordinal": ordinal,
        "rho": float(rho),
        "rho_binary64_hex": float(rho).hex(),
        "direct_margin_interval": None,
        "nominal_margin": nominal_margin,
        "certified": False,
        "authoritative_bound_returned": False,
        "independent_checker_accepted": False,
        "complete_certificate": False,
        "operator_certificate_record_sha256": None,
        "operator_certificate_path": None,
        "partial_invocation_counts": partial_invocation_counts,
        "exception_type": "AssertionError",
        "exception_message": exception_message,
        "failure_origin": origin,
        "runtime_seconds": runtime_seconds,
        "bound_runtime_seconds": bound_runtime_seconds,
        "peak_CPU_RSS_bytes": peak_cpu,
        "peak_GPU_allocated_bytes": peak_allocated,
        "peak_GPU_reserved_bytes": peak_reserved,
        "all_outputs_finite": None,
        "cached_DeepT_reference": prop["cached_DeepT_reference"],
        "fresh_proof_evaluation_count": fresh,
        "reused_proof_evaluation_count": reused,
        "scientific_query_already_consumed_before_smoke_v3": reused,
        "bound_entrypoint_calls_during_smoke_v3": fresh,
    }


def generate_query(property_id: str, rho: float, ordinal: int, authorized: bool) -> None:
    if not authorized:
        raise RuntimeError("explicit USER authorization required")
    manifest = validate_manifest()
    prop = property_manifest_record(manifest, property_id)
    result_path, certificate_path = query_paths(property_id, ordinal, rho)
    if result_path.exists() or certificate_path.exists():
        raise FileExistsError("immutable smoke query artifact exists")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if property_id == "deept_table7_stdln3_s000_line504_tok10" and rho == INITIAL_RHO:
        saved = ref.write(result_path, accepted_seed_record(manifest, prop, ordinal))
        print(json.dumps({"terminal_status": saved["terminal_status"],
                          "property_id": property_id, "rho": rho,
                          "record_sha256": saved["record_sha256"]}, sort_keys=True))
        return
    observed_matches = [
        item for item in manifest["recovered_native_domain_failures"]
        if (property_id == item["property_id"]
            and float(rho).hex() == float(item["rho"]).hex())
    ]
    if len(observed_matches) > 1:
        raise RuntimeError("duplicate recovered native domain failure")
    if observed_matches:
        observed = observed_matches[0]
        saved = ref.write(result_path, domain_failure_record(
            manifest, prop, ordinal, rho,
            exception_message=observed["exception_message"],
            runtime_seconds=0.0, bound_runtime_seconds=None,
            peak_cpu=0, peak_allocated=0, peak_reserved=0,
            fresh=0, reused=1, origin=observed["origin"]))
        print(json.dumps({"terminal_status": saved["terminal_status"],
                          "property_id": property_id, "rho": rho,
                          "record_sha256": saved["record_sha256"]}, sort_keys=True))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("proof-carrying smoke query requires frozen CUDA execution")
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
    bound_started = time.perf_counter()
    dispatch = graph.new_dispatch()
    try:
        with torch.no_grad():
            if torch.is_grad_enabled():
                raise RuntimeError("production proof graph requires autograd disabled")
            margin, dispatch = graph.execute(
                z, model, args, clean_label=label, dispatch=dispatch)
    except AssertionError as error:
        if not is_native_domain_failure(error):
            raise
        torch.cuda.synchronize()
        saved = ref.write(result_path, domain_failure_record(
            manifest, prop, ordinal, rho,
            exception_message=str(error),
            runtime_seconds=time.perf_counter() - started,
            bound_runtime_seconds=time.perf_counter() - bound_started,
            peak_cpu=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            peak_allocated=torch.cuda.max_memory_allocated(),
            peak_reserved=torch.cuda.max_memory_reserved(),
            fresh=1, reused=0,
            origin="live pinned-native LayerNorm sqrt domain failure",
            nominal_margin=nominal,
            partial_invocation_counts={
                family: int(dispatch.counts[family]) for family in graph.FAMILIES
            }))
        print(json.dumps({"terminal_status": saved["terminal_status"],
                          "property_id": property_id, "rho": rho,
                          "exception_message": saved["exception_message"],
                          "record_sha256": saved["record_sha256"]}, sort_keys=True))
        return
    if margin.zonotope_w.requires_grad or margin.zonotope_w.grad_fn is not None:
        raise RuntimeError("production proof result retained autograd lineage")
    lower, upper = margin.concretize()
    torch.cuda.synchronize()
    bound_seconds = time.perf_counter() - bound_started
    if not bool(torch.isfinite(lower).all() and torch.isfinite(upper).all()
                and (lower <= upper).all()):
        raise RuntimeError("proof-carrying smoke interval invalid")
    counts = dispatch.assert_complete(graph.EXPECTED_THREE_BLOCK_COUNTS)
    certificate = ref.write(certificate_path, {
        "schema": "CORET_PROOF_CARRYING_SMOKE_QUERY_CERTIFICATES_V1",
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        "property_id": property_id,
        "query_ordinal": ordinal,
        "rho": float(rho),
        "rho_binary64_hex": float(rho).hex(),
        "invocation_counts": counts,
        "generic_family_invocations": dispatch.generic_family_invocations,
        "independent_checker_accepted": True,
        "complete_certificate": True,
        "certificates": dispatch.certificates,
        "scientific_query_count": 1,
        "bound_entrypoint_call_count": 1,
    })
    lo, hi = float(lower.min()), float(upper.max())
    saved = ref.write(result_path, {
        "schema": "CORET_PROOF_CARRYING_SMOKE_QUERY_RESULT_V1",
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
        "complete_certificate": True,
        "operator_certificate_record_sha256": certificate["record_sha256"],
        "operator_certificate_path": str(certificate_path.relative_to(ROOT)),
        "invocation_counts": counts,
        "runtime_seconds": time.perf_counter() - started,
        "bound_runtime_seconds": bound_seconds,
        "peak_CPU_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "peak_GPU_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_GPU_reserved_bytes": torch.cuda.max_memory_reserved(),
        "all_outputs_finite": True,
        "cached_DeepT_reference": prop["cached_DeepT_reference"],
        "fresh_proof_evaluation_count": 1,
        "reused_proof_evaluation_count": 0,
        "scientific_query_already_consumed_before_smoke": 0,
        "bound_entrypoint_calls_during_smoke": 1,
    })
    print(json.dumps({"terminal_status": saved["terminal_status"],
                      "property_id": property_id, "rho": rho,
                      "certified": saved["certified"],
                      "direct_margin_interval": saved["direct_margin_interval"],
                      "record_sha256": saved["record_sha256"]}, sort_keys=True))


def query_records(property_id: str) -> list[dict]:
    records = []
    for root in (PRIOR_OUT, OUT):
        directory = root / "properties" / property_id / "queries"
        if not directory.exists():
            continue
        for path in directory.glob("query_*_result_v1.json"):
            record = ref.verified(path)
            if record.get("property_id") != property_id:
                raise RuntimeError("query artifact property mismatch")
            record = dict(record)
            record["_path"] = path
            records.append(record)
    records.sort(key=lambda x: int(x["query_ordinal"]))
    if [x["query_ordinal"] for x in records] != list(range(len(records))):
        raise RuntimeError("query ordinals are not a contiguous append-only sequence")
    if len({x["rho_binary64_hex"] for x in records}) != len(records):
        raise RuntimeError("duplicate radius in query history")
    return records


def sync_journal(property_id: str, records: list[dict]) -> None:
    journal = OUT / "properties" / property_id / "query_events_v1.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    existing = [] if not journal.exists() else load_jsonl(journal)
    seen = {x["query_result_record_sha256"] for x in existing}
    with journal.open("a", encoding="utf-8") as handle:
        for record in records:
            if record["record_sha256"] in seen:
                continue
            event = {
                "schema": "CORET_PROOF_CARRYING_SMOKE_QUERY_EVENT_V1",
                "property_id": property_id,
                "query_ordinal": record["query_ordinal"],
                "rho": record["rho"],
                "rho_binary64_hex": record["rho_binary64_hex"],
                "certified": record["certified"],
                "query_result_path": str(record["_path"].relative_to(ROOT)),
                "query_result_record_sha256": record["record_sha256"],
            }
            event["event_sha256"] = canonical_payload_sha(event)
            handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            seen.add(record["record_sha256"])


def search_state(records: list[dict]) -> dict:
    by_rho = {x["rho_binary64_hex"]: x for x in records}

    def get(rho: float):
        return by_rho.get(float(rho).hex())

    initial = get(INITIAL_RHO)
    if initial is None:
        return {"next_rho": INITIAL_RHO, "stage": "initial"}
    lower = upper = None
    if initial["certified"]:
        lower = INITIAL_RHO
        for refinement in range(1, MAX_FACTOR_TWO_REFINEMENTS + 1):
            candidate = INITIAL_RHO * (2.0 ** refinement)
            observation = get(candidate)
            if observation is None:
                return {"next_rho": candidate, "stage": "factor_two_expansion"}
            if observation["certified"]:
                lower = candidate
            else:
                upper = candidate
                break
        if upper is None:
            return {"done": True, "classification": "CERTIFIED_THROUGH_MAX_REFINEMENT",
                    "lower": lower, "upper": None, "midpoints": 0}
    else:
        upper = INITIAL_RHO
        for refinement in range(1, MAX_FACTOR_TWO_REFINEMENTS + 1):
            candidate = INITIAL_RHO / (2.0 ** refinement)
            observation = get(candidate)
            if observation is None:
                return {"next_rho": candidate, "stage": "factor_two_halving"}
            if observation["certified"]:
                lower = candidate
                break
            upper = candidate
        if lower is None:
            return {"done": True, "classification": "NO_CERTIFIED_RADIUS_FOUND",
                    "lower": None, "upper": upper, "midpoints": 0}
    for midpoint_index in range(MIDPOINT_ITERATIONS):
        midpoint = (lower + upper) / 2.0
        observation = get(midpoint)
        if observation is None:
            return {"next_rho": midpoint, "stage": "midpoint",
                    "midpoint_index": midpoint_index, "lower": lower, "upper": upper}
        if observation["certified"]:
            lower = midpoint
        else:
            upper = midpoint
    return {"done": True, "classification": "COMPLETE_CERTIFIED_RADIUS",
            "lower": lower, "upper": upper, "midpoints": MIDPOINT_ITERATIONS}


def invoke_query(property_id: str, rho: float, ordinal: int) -> None:
    command = [sys.executable, str(HERE), "query", "--authorized",
               "--property-id", property_id, "--rho", float(rho).hex(),
               "--query-ordinal", str(ordinal)]
    subprocess.run(command, cwd=ROOT, check=True,
                   env={**os.environ, "PYTHONPATH": "research_hab",
                        "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})


def run_property(property_id: str, authorized: bool) -> None:
    if not authorized:
        raise RuntimeError("explicit USER authorization required")
    manifest = validate_manifest()
    prop = property_manifest_record(manifest, property_id)
    result_path = OUT / "properties" / property_id / "result_v1.json"
    if result_path.exists():
        existing = ref.verified(result_path)
        print(json.dumps({"terminal_status": "EXISTING_COMPLETE_PROPERTY_STOP",
                          "property_id": property_id,
                          "record_sha256": existing["record_sha256"]}, sort_keys=True))
        return
    prior_result_path = PRIOR_OUT / "properties" / property_id / "result_v1.json"
    if prior_result_path.exists():
        prior = ref.verified(prior_result_path)
        migrated = dict(prior)
        migrated.pop("record_sha256")
        migrated["canonical_manifest_sha256"] = manifest["canonical_manifest_sha256"]
        migrated["reused_completed_prior_smoke_result"] = True
        migrated["prior_result_path"] = str(prior_result_path.relative_to(ROOT))
        migrated["prior_result_record_sha256"] = prior["record_sha256"]
        result_path.parent.mkdir(parents=True, exist_ok=True)
        saved = ref.write(result_path, migrated)
        print(json.dumps({"terminal_status": "COMPLETE_PRIOR_RESULT_REUSED",
                          "property_id": property_id,
                          "record_sha256": saved["record_sha256"]}, sort_keys=True))
        return
    while True:
        records = query_records(property_id)
        sync_journal(property_id, records)
        state = search_state(records)
        if state.get("done"):
            break
        invoke_query(property_id, float(state["next_rho"]), len(records))
    records = query_records(property_id)
    sync_journal(property_id, records)
    lower_rho = state["lower"]
    upper_rho = state["upper"]
    final_certified = None if lower_rho is None else next(
        x for x in records if x["rho_binary64_hex"] == float(lower_rho).hex())
    final_uncertified = None if upper_rho is None else next(
        x for x in records if x["rho_binary64_hex"] == float(upper_rho).hex())
    cached_radius = float(prop["cached_DeepT_reference"]["certified_lower_endpoint_binary64"])
    runtime = sum(float(x["runtime_seconds"]) for x in records)
    saved = ref.write(result_path, {
        "schema": "CORET_PROOF_CARRYING_HISTORICAL_SMOKE_PROPERTY_RESULT_V1",
        "terminal_status": "COMPLETE",
        "classification": state["classification"],
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        "property_id": property_id,
        "sentence_ordinal": prop["sentence_ordinal"],
        "token_position": prop["token_position"],
        "certified_radius": lower_rho,
        "uncertified_upper_radius": upper_rho,
        "final_bracket_width": None if upper_rho is None or lower_rho is None else upper_rho - lower_rho,
        "final_certified_query": None if final_certified is None else {
            "rho": final_certified["rho"],
            "direct_margin_interval": final_certified["direct_margin_interval"],
            "query_result_record_sha256": final_certified["record_sha256"],
            "query_result_path": str(final_certified["_path"].relative_to(ROOT)),
            "operator_certificate_record_sha256": final_certified["operator_certificate_record_sha256"],
            "operator_certificate_path": final_certified["operator_certificate_path"],
        },
        "final_uncertified_query": None if final_uncertified is None else {
            "rho": final_uncertified["rho"],
            "direct_margin_interval": final_uncertified["direct_margin_interval"],
            "query_result_record_sha256": final_uncertified["record_sha256"],
            "query_result_path": str(final_uncertified["_path"].relative_to(ROOT)),
        },
        "complete_certificate": final_certified is not None and final_certified["complete_certificate"],
        "independent_checker_accepted": final_certified is not None and final_certified["independent_checker_accepted"],
        "query_count": len(records),
        "fresh_proof_evaluations": sum(x["fresh_proof_evaluation_count"] for x in records),
        "reused_proof_evaluations": sum(x["reused_proof_evaluation_count"] for x in records),
        "runtime_seconds": runtime,
        "peak_CPU_RSS_bytes": max(x["peak_CPU_RSS_bytes"] for x in records),
        "peak_GPU_allocated_bytes": max(x["peak_GPU_allocated_bytes"] for x in records),
        "peak_GPU_reserved_bytes": max(x["peak_GPU_reserved_bytes"] for x in records),
        "cached_DeepT_reference": prop["cached_DeepT_reference"],
        "cached_DeepT_certified_radius": cached_radius,
        "radius_ratio_to_cached_DeepT": None if lower_rho is None else lower_rho / cached_radius,
    })
    print(json.dumps({"terminal_status": saved["terminal_status"],
                      "property_id": property_id,
                      "certified_radius": saved["certified_radius"],
                      "record_sha256": saved["record_sha256"]}, sort_keys=True))


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
            raise RuntimeError(f'incomplete smoke property: {prop["property_id"]}')
        results.append(ref.verified(path))
    saved = ref.write(SUMMARY, {
        "schema": "CORET_PROOF_CARRYING_HISTORICAL_SMOKE_SUMMARY_V1",
        "terminal_status": "COMPLETE",
        "classification": "HISTORICAL_10_PROPERTY_SMOKE_COMPLETE",
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        "completed_properties": 10,
        "certified_properties": sum(x["certified_radius"] is not None for x in results),
        "fresh_proof_evaluations": sum(x["fresh_proof_evaluations"] for x in results),
        "reused_proof_evaluations": sum(x["reused_proof_evaluations"] for x in results),
        "total_runtime_seconds": sum(x["runtime_seconds"] for x in results),
        "peak_CPU_RSS_bytes": max(x["peak_CPU_RSS_bytes"] for x in results),
        "peak_GPU_allocated_bytes": max(x["peak_GPU_allocated_bytes"] for x in results),
        "peak_GPU_reserved_bytes": max(x["peak_GPU_reserved_bytes"] for x in results),
        "property_results": [{
            "property_id": x["property_id"],
            "certified_radius": x["certified_radius"],
            "cached_DeepT_certified_radius": x["cached_DeepT_certified_radius"],
            "radius_ratio_to_cached_DeepT": x["radius_ratio_to_cached_DeepT"],
            "result_record_sha256": x["record_sha256"],
        } for x in results],
        "DeepT_rerun_count": 0,
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
            parser.error("query requires --property-id, --rho and --query-ordinal")
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
