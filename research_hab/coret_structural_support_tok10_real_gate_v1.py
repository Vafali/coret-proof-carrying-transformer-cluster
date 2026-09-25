#!/usr/bin/env python3
"""USER-only end-to-end tok10 structural-support scientific gate."""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import torch

import coret_bounded_native_production_graph_v1 as graph
import coret_deept_exact_standard_ln_adapter as adapter
import coret_native_semantics_checker_v1 as native_checker
import coret_native_semantics_production_graph_v1 as production
import coret_proof_carrying_historical_smoke_v1 as smoke
import coret_sound_reference_v1 as ref
import coret_structural_support_precise_dot_v1 as structural


WORKTREE = Path(__file__).resolve().parents[1]
ROOT = ref.ROOT
OUT = WORKTREE / "research_hab/results/coret_structural_support_tok10_real_gate_v2_20260923"
MANIFEST = OUT / "coret_structural_support_tok10_real_gate_manifest_v2.json"
PREFLIGHT = OUT / "coret_structural_support_tok10_real_gate_preflight_v2.json"
RESULT = OUT / "coret_structural_support_tok10_real_gate_result_v2.json"
JOURNAL = OUT / "query_events_v1.jsonl"
QUERIES = OUT / "queries"
LAUNCHER = WORKTREE / "research_hab/run_coret_structural_support_tok10_real_gate_v1.sh"
PROPERTY = "deept_table7_stdln3_s000_line504_tok10"
RHO0 = 1.0 / 1600.0
MAX_DOUBLINGS = 12
MIDPOINTS = 10
TILE_RESULT = WORKTREE / "research_hab/results/coret_structural_support_av_tile112_gate_v1_20260923/coret_structural_support_av_tile112_result_v1.json"
REFERENCE_RESULT = ROOT / "research_hab/results/coret_proof_carrying_historical_smoke_v4_20260922/properties/deept_table7_stdln3_s000_line504_tok10/result_v1.json"
REFERENCE_INITIAL = ROOT / "research_hab/results/coret_bounded_native_tok10_v1_20260922/coret_bounded_native_tok10_result_v1.json"


def _write(path, payload, key="record_sha256"):
    value = dict(payload); value[key] = ref.canonical(value)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return value


def _sources():
    return (
        Path(__file__).resolve(), LAUNCHER,
        WORKTREE / "research_hab/coret_structural_support_precise_dot_v1.py",
        WORKTREE / "research_hab/coret_structural_support_checker_v1.py",
        WORKTREE / "research_hab/tests/test_coret_structural_support_precise_dot_v1.py",
        Path(graph.__file__).resolve(), Path(production.__file__).resolve(),
        Path(adapter.__file__).resolve(), Path(native_checker.__file__).resolve(),
        Path(smoke.__file__).resolve(),
    )


def _property_and_example():
    parent = ref.verified(smoke.HISTORICAL_MANIFEST,
                          "canonical_manifest_sha256")
    prop = [x for x in parent["properties"] if x["property_id"] == PROPERTY]
    example = [x for x in parent["examples"] if x["sentence_ordinal"] == 0]
    if len(prop) != 1 or len(example) != 1:
        raise RuntimeError("frozen tok10 identity unavailable")
    return parent, prop[0], example[0]


def expected_manifest():
    parent, prop, example = _property_and_example()
    tile = ref.verified(TILE_RESULT)
    reference = ref.verified(REFERENCE_RESULT)
    initial = ref.verified(REFERENCE_INITIAL)
    if (tile["terminal_status"] !=
            "STRUCTURAL_SUPPORT_PRECISE_DOT_READY_FOR_REAL_GATE"):
        raise RuntimeError("structural runtime prerequisite not accepted")
    if tile["combined_QK_AV_speedup"] < 3.0:
        raise RuntimeError("structural runtime prerequisite misses frozen gate")
    if (reference["terminal_status"] != "COMPLETE"
            or reference["classification"] != "COMPLETE_CERTIFIED_RADIUS"):
        raise RuntimeError("accepted tok10 search reference differs")
    if (initial["property_id"] != PROPERTY or initial["rho"] != RHO0
            or not initial["certified"]):
        raise RuntimeError("accepted tok10 initial result differs")
    return {
        "schema": "CORET_STRUCTURAL_SUPPORT_TOK10_REAL_GATE_MANIFEST_V2",
        "status": "FROZEN_BEFORE_SCIENTIFIC_GATE",
        "property": prop, "example": example,
        "threat_model": {
            "p": 100, "one_token": True, "token_position": 10,
            "source_dimension": 128, "embedding_Linf": True,
        },
        "model": parent["model"],
        "checkpoint_sha256": parent["checkpoint_sha256"],
        "pinned_DeepT_revision": structural.native_proof.PINNED_REVISION,
        "search": {
            "initial_rho": RHO0,
            "factor_two_bracketing": True,
            "maximum_factor_two_refinements": MAX_DOUBLINGS,
            "midpoint_iterations": MIDPOINTS,
            "reported_value": "largest observed certified lower endpoint",
            "typed_uncertified_assertion_prefixes": list(
                smoke.NATIVE_DOMAIN_FAILURE_PREFIXES),
            "unexpected_exception_policy": "fail_closed",
        },
        "optimized_execution": {
            "native_DeepT_mathematics_authoritative": True,
            "support_source": "provenance_and_operator_topology_only",
            "magnitude_pruning": False,
            "QK": "exact structural-support execution",
            "A.V": "exact structural-support execution",
            "requested_generator_tile": structural.AV_GENERATOR_TILE,
            "temporary_cap_bytes": structural.AV_TEMPORARY_CAP_BYTES,
            "support_recomputed_per_query": True,
            "generic_fallback_allowed": False,
        },
        "runtime_prerequisite": {
            "path": str(TILE_RESULT),
            "record_sha256": tile["record_sha256"],
            "file_sha256": ref.sha(TILE_RESULT),
            "combined_QK_AV_speedup": tile["combined_QK_AV_speedup"],
        },
        "accepted_reference": {
            "initial_margin_interval": initial["direct_margin_interval"],
            "initial_certified": initial["certified"],
            "initial_result_record_sha256": initial["record_sha256"],
            "search_certified_radius": reference["certified_radius"],
            "search_result_record_sha256": reference["record_sha256"],
            "search_result_file_sha256": ref.sha(REFERENCE_RESULT),
        },
        "acceptance": {
            "initial_certified_boolean_must_match": True,
            "final_certified_boolean_and_radius_reported_against_reference": True,
            "all_23_native_family_invocations_required": True,
            "all_support_claims_validated": True,
            "all_operator_checkers_required": True,
            "zero_generic_fallback": True,
            "fresh_radius_outward_evidence": tile["record_sha256"],
        },
        "outputs": {
            "queries": str(QUERIES), "journal": str(JOURNAL),
            "result": str(RESULT),
        },
        "source_hashes": {str(path): ref.sha(path) for path in _sources()},
        "preparation_scientific_queries": 0,
        "preparation_bound_entrypoint_calls": 0,
    }


def validate_manifest():
    stored = ref.verified(MANIFEST, "canonical_manifest_sha256")
    payload = dict(stored); payload.pop("canonical_manifest_sha256")
    if payload != expected_manifest():
        raise RuntimeError("optimized tok10 manifest mismatch")
    return stored


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    saved = _write(MANIFEST, expected_manifest(), "canonical_manifest_sha256")
    print(json.dumps({"terminal_status": "FROZEN_NO_SOLVE",
                      "canonical_manifest_sha256": saved[
                          "canonical_manifest_sha256"],
                      "scientific_queries": 0,
                      "bound_entrypoint_calls": 0}, sort_keys=True))


def preflight():
    manifest = validate_manifest()
    if RESULT.exists():
        raise FileExistsError("immutable optimized tok10 result exists")
    record = {
        "schema": "CORET_STRUCTURAL_SUPPORT_TOK10_REAL_GATE_PREFLIGHT_V2",
        "terminal_status": "PASS_NO_SOLVE",
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        "property_id": PROPERTY, "initial_rho": RHO0,
        "midpoint_iterations": MIDPOINTS,
        "requested_generator_tile": structural.AV_GENERATOR_TILE,
        "temporary_cap_bytes": structural.AV_TEMPORARY_CAP_BYTES,
        "checkpoint_loaded": False, "model_forward_calls": 0,
        "scientific_queries": 0, "bound_entrypoint_calls": 0,
    }
    saved = ref.verified(PREFLIGHT) if PREFLIGHT.exists() else _write(
        PREFLIGHT, record)
    payload = dict(saved); payload.pop("record_sha256")
    if payload != record:
        raise RuntimeError("optimized tok10 preflight differs")
    print(json.dumps({"terminal_status": "PASS_NO_SOLVE",
                      "record_sha256": saved["record_sha256"],
                      "scientific_queries": 0,
                      "bound_entrypoint_calls": 0}, sort_keys=True))


def _tag(rho):
    return float(rho).hex().replace("+", "p").replace("-", "m").replace(".", "d")


def _paths(ordinal, rho):
    stem = f"query_{ordinal:02d}_{_tag(rho)}"
    return QUERIES / f"{stem}_result_v1.json", QUERIES / f"{stem}_certificates_v1.json"


def _family_timing(certificates):
    totals = {}
    calls = []
    for certificate in certificates:
        family = certificate["family"]
        item = totals.setdefault(family, {
            "calls": 0, "native_operator_seconds": 0.0,
            "facade_total_seconds": 0.0, "numerical_overhead_seconds": 0.0,
        })
        item["calls"] += 1
        for key in ("native_operator_seconds", "facade_total_seconds",
                    "numerical_overhead_seconds"):
            item[key] += float(certificate.get(key, 0.0))
        call = {"family": family,
                "native_operator_seconds": certificate.get(
                    "native_operator_seconds"),
                "facade_total_seconds": certificate.get("facade_total_seconds")}
        if "support_diagnostics" in certificate:
            call["support_diagnostics"] = certificate["support_diagnostics"]
        calls.append(call)
    return totals, calls


def _support_acceptance(certificates):
    for certificate in certificates:
        if not native_checker.check_common(certificate, certificate["family"]):
            return False
        proof = certificate.get("support_proof")
        if not isinstance(proof, dict) or proof.get("validated") is not True:
            return False
        if certificate.get("generic_semantic_remainder_used") is not False:
            return False
        if certificate["family"] in ("QK", "A.V"):
            diagnostics = certificate.get("support_diagnostics", {})
            if diagnostics.get("support_validated") is not True:
                return False
            if diagnostics.get("mode") == "A.V" and (
                    diagnostics.get("effective_generator_tile") !=
                    structural.AV_GENERATOR_TILE or
                    diagnostics.get("peak_temporary_bytes", 0) >
                    structural.AV_TEMPORARY_CAP_BYTES):
                return False
    return True


def query(rho, ordinal, authorized):
    if not authorized:
        raise RuntimeError("explicit USER authorization required")
    manifest = validate_manifest()
    result_path, certificate_path = _paths(ordinal, rho)
    if result_path.exists() or certificate_path.exists():
        raise FileExistsError("immutable optimized query artifact exists")
    QUERIES.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter(); setup_started = started
    modules = adapter.FrozenDeepTModules()
    from Verifiers.Zonotope import Zonotope
    model, _, configuration = adapter.load_native_model(
        modules, torch.device("cuda:0"), dtype=torch.float32)
    if configuration != manifest["model"]["configuration"]:
        raise RuntimeError("loaded model configuration differs")
    ids = torch.tensor([manifest["example"]["token_ids"]],
                       dtype=torch.long, device="cuda:0")
    with torch.no_grad():
        logits = model(ids, attention_mask=torch.ones_like(ids))[0]
        label = int(manifest["property"]["clean_label"])
        nominal = float(logits[0, label] - logits[0, 1-label])
        pre = adapter.native_pre_layernorm_embeddings(model, ids)[0].detach()
    setup_seconds = time.perf_counter() - setup_started
    args = adapter.build_deept_args(modules, torch.device("cuda:0"))
    args.keep_intermediate_zonotopes = False
    z = Zonotope(args=args, p=100, eps=float(rho),
                 perturbed_word_index=10, value=pre)
    delegate = structural.StructuralNativeSemanticOperators(
        generator_tile=structural.AV_GENERATOR_TILE)
    dispatch = production.NativeProductionDispatch(delegate=delegate)
    bound_started = time.perf_counter()
    try:
        with torch.no_grad():
            margin, dispatch = graph.execute(
                z, model, args, clean_label=label, dispatch=dispatch)
    except AssertionError as error:
        if not smoke.is_native_domain_failure(error):
            raise
        torch.cuda.synchronize()
        saved = _write(result_path, {
            "schema": "CORET_STRUCTURAL_SUPPORT_TOK10_QUERY_RESULT_V1",
            "terminal_status": "UNCERTIFIED_DOMAIN_FAILURE",
            "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
            "property_id": PROPERTY, "query_ordinal": ordinal,
            "rho": float(rho), "rho_binary64_hex": float(rho).hex(),
            "certified": False, "authoritative_bound_returned": False,
            "exception_type": "AssertionError",
            "exception_message": str(error),
            "runtime_seconds": time.perf_counter() - started,
            "setup_seconds": setup_seconds,
            "bound_seconds": time.perf_counter() - bound_started,
            "peak_CPU_RSS_bytes": resource.getrusage(
                resource.RUSAGE_SELF).ru_maxrss * 1024,
            "peak_GPU_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_GPU_reserved_bytes": torch.cuda.max_memory_reserved(),
            "scientific_queries": 1, "bound_entrypoint_calls": 1,
        })
        print(json.dumps({"terminal_status": saved["terminal_status"],
                          "rho": rho, "record_sha256": saved[
                              "record_sha256"]}, sort_keys=True))
        return
    concretize_started = time.perf_counter()
    lower, upper = margin.concretize(); torch.cuda.synchronize()
    concretize_seconds = time.perf_counter() - concretize_started
    bound_seconds = time.perf_counter() - bound_started
    if not bool(torch.isfinite(lower).all() and torch.isfinite(upper).all()
                and (lower <= upper).all()):
        raise RuntimeError("optimized tok10 margin interval invalid")
    counts = dispatch.assert_complete(graph.EXPECTED_THREE_BLOCK_COUNTS)
    if dispatch.generic_family_invocations != 0:
        raise RuntimeError("optimized tok10 reached generic fallback")
    checker_started = time.perf_counter()
    if not _support_acceptance(dispatch.certificates):
        raise RuntimeError("optimized tok10 support/checker acceptance failed")
    checker_seconds = time.perf_counter() - checker_started
    family_timing, call_timing = _family_timing(dispatch.certificates)
    certificate_started = time.perf_counter()
    certificate = _write(certificate_path, {
        "schema": "CORET_STRUCTURAL_SUPPORT_TOK10_CERTIFICATES_V1",
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        "property_id": PROPERTY, "query_ordinal": ordinal,
        "rho": float(rho), "rho_binary64_hex": float(rho).hex(),
        "invocation_counts": counts, "generic_family_invocations": 0,
        "independent_checker_accepted": True,
        "all_support_claims_validated": True,
        "complete_certificate": True,
        "offline_center_retained_and_fresh_outward_evidence": manifest[
            "runtime_prerequisite"]["record_sha256"],
        "provenance_ID_consistent": True,
        "family_timing_seconds": family_timing,
        "operator_calls": call_timing,
        "certificates": dispatch.certificates,
        "scientific_queries": 1, "bound_entrypoint_calls": 1,
    })
    certificate_seconds = time.perf_counter() - certificate_started
    lo, hi = float(lower.min()), float(upper.max())
    initial_reference = manifest["accepted_reference"]
    saved = _write(result_path, {
        "schema": "CORET_STRUCTURAL_SUPPORT_TOK10_QUERY_RESULT_V1",
        "terminal_status": "COMPLETE",
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        "property_id": PROPERTY, "query_ordinal": ordinal,
        "rho": float(rho), "rho_binary64_hex": float(rho).hex(),
        "direct_margin_interval": [lo, hi], "nominal_margin": nominal,
        "certified": lo > 0.0,
        "independent_checker_accepted": True,
        "all_support_claims_validated": True,
        "complete_certificate": True,
        "operator_certificate_record_sha256": certificate["record_sha256"],
        "invocation_counts": counts, "generic_fallbacks": 0,
        "provenance_ID_consistent": True,
        "runtime_seconds": time.perf_counter() - started,
        "setup_seconds": setup_seconds, "bound_seconds": bound_seconds,
        "concretize_seconds": concretize_seconds,
        "independent_checker_seconds": checker_seconds,
        "certificate_serialization_seconds": certificate_seconds,
        "family_timing_seconds": family_timing,
        "peak_CPU_RSS_bytes": resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss * 1024,
        "peak_GPU_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_GPU_reserved_bytes": torch.cuda.max_memory_reserved(),
        "accepted_initial_comparison": (
            {"reference_interval": initial_reference["initial_margin_interval"],
             "reference_certified": initial_reference["initial_certified"],
             "certified_boolean_matches": (
                 (lo > 0.0) == initial_reference["initial_certified"])}
            if float(rho).hex() == RHO0.hex() else None),
        "scientific_queries": 1, "bound_entrypoint_calls": 1,
    })
    if float(rho).hex() == RHO0.hex() and not saved[
            "accepted_initial_comparison"]["certified_boolean_matches"]:
        raise RuntimeError("optimized initial certified Boolean differs")
    print(json.dumps({"terminal_status": "COMPLETE", "rho": rho,
                      "certified": saved["certified"],
                      "direct_margin_interval": saved["direct_margin_interval"],
                      "record_sha256": saved["record_sha256"]}, sort_keys=True))


def records():
    found = []
    if QUERIES.exists():
        for path in QUERIES.glob("query_*_result_v1.json"):
            item = ref.verified(path); item = dict(item); item["_path"] = path
            found.append(item)
    found.sort(key=lambda x: int(x["query_ordinal"]))
    if [x["query_ordinal"] for x in found] != list(range(len(found))):
        raise RuntimeError("optimized query ordinals are not contiguous")
    return found


def _journal(found):
    existing = [] if not JOURNAL.exists() else smoke.load_jsonl(JOURNAL)
    seen = {x["query_result_record_sha256"] for x in existing}
    with JOURNAL.open("a", encoding="utf-8") as handle:
        for item in found:
            if item["record_sha256"] in seen:
                continue
            event = {
                "schema": "CORET_STRUCTURAL_SUPPORT_TOK10_QUERY_EVENT_V1",
                "property_id": PROPERTY,
                "query_ordinal": item["query_ordinal"], "rho": item["rho"],
                "rho_binary64_hex": item["rho_binary64_hex"],
                "certified": item["certified"],
                "query_result_path": str(item["_path"]),
                "query_result_record_sha256": item["record_sha256"],
            }
            event["event_sha256"] = smoke.canonical_payload_sha(event)
            handle.write(json.dumps(event, sort_keys=True) + "\n")
            handle.flush(); os.fsync(handle.fileno())
            seen.add(item["record_sha256"])


def _invoke(rho, ordinal):
    command = [sys.executable, str(Path(__file__).resolve()), "query",
               "--authorized", "--rho", float(rho).hex(),
               "--query-ordinal", str(ordinal)]
    subprocess.run(command, cwd=WORKTREE, check=True, env={
        **os.environ,
        "PYTHONPATH": (
            "research_hab:/mnt/c/users/david-despacho/documents/"
            "vafali projects/lookahead-branching/research_hab"),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    })


def run(authorized):
    if not authorized:
        raise RuntimeError("explicit USER authorization required")
    manifest = validate_manifest()
    if RESULT.exists():
        existing = ref.verified(RESULT)
        print(json.dumps({"terminal_status": "EXISTING_COMPLETE_RESULT_STOP",
                          "record_sha256": existing["record_sha256"]},
                         sort_keys=True))
        return
    while True:
        found = records(); _journal(found)
        state = smoke.search_state(found)
        if state.get("done"):
            certified = [x for x in found if x["certified"]]
            lower_record = max(certified, key=lambda x: x["rho"])
            reference = manifest["accepted_reference"][
                "search_certified_radius"]
            saved = _write(RESULT, {
                "schema": "CORET_STRUCTURAL_SUPPORT_TOK10_REAL_GATE_RESULT_V2",
                "terminal_status": "STRUCTURAL_SUPPORT_END_TO_END_REAL_GATE_COMPLETE",
                "canonical_manifest_sha256": manifest[
                    "canonical_manifest_sha256"],
                "property_id": PROPERTY, "search_classification": state[
                    "classification"],
                "certified_radius": state["lower"],
                "uncertified_upper": state["upper"],
                "midpoint_iterations": state["midpoints"],
                "accepted_reference_certified_radius": reference,
                "radius_ratio_to_reference": state["lower"] / reference,
                "final_certified_boolean": lower_record["certified"],
                "query_count": len(found),
                "certified_query_count": len(certified),
                "domain_failure_count": sum(
                    x["terminal_status"] == "UNCERTIFIED_DOMAIN_FAILURE"
                    for x in found),
                "all_checkers_accepted": all(
                    x.get("independent_checker_accepted", True)
                    for x in found),
                "generic_fallbacks": sum(x.get("generic_fallbacks", 0)
                                           for x in found),
                "total_wall_seconds": sum(x["runtime_seconds"] for x in found),
                "peak_CPU_RSS_bytes": max(x["peak_CPU_RSS_bytes"] for x in found),
                "peak_GPU_allocated_bytes": max(
                    x["peak_GPU_allocated_bytes"] for x in found),
                "peak_GPU_reserved_bytes": max(
                    x["peak_GPU_reserved_bytes"] for x in found),
                "scientific_queries": len(found),
                "bound_entrypoint_calls": len(found),
            })
            print(json.dumps({"terminal_status": saved["terminal_status"],
                              "certified_radius": saved["certified_radius"],
                              "radius_ratio_to_reference": saved[
                                  "radius_ratio_to_reference"],
                              "record_sha256": saved["record_sha256"]},
                             sort_keys=True))
            return
        _invoke(state["next_rho"], len(found))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "preflight", "query", "run"))
    parser.add_argument("--authorized", action="store_true")
    parser.add_argument("--rho")
    parser.add_argument("--query-ordinal", type=int)
    values = parser.parse_args()
    if values.action == "freeze": freeze()
    elif values.action == "preflight": preflight()
    elif values.action == "query":
        if values.rho is None or values.query_ordinal is None:
            parser.error("query requires --rho and --query-ordinal")
        query(float.fromhex(values.rho), values.query_ordinal,
              values.authorized)
    else: run(values.authorized)
