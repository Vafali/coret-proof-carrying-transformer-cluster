#!/usr/bin/env python3
"""End-to-end tok10 search with the validated B2-QK allocator fence."""
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
import coret_native_semantics_production_graph_v1 as production
import coret_proof_carrying_historical_smoke_v1 as smoke
import coret_sound_reference_v1 as ref
import coret_structural_support_lifetime_v1 as lifetime
import coret_structural_support_precise_dot_v1 as structural
import coret_structural_support_tok10_real_gate_v1 as frozen_runner


WORKTREE = Path(__file__).resolve().parents[1]
OUT = WORKTREE / (
    "research_hab/results/coret_structural_support_tok10_lifetime_v1_20260923")
MANIFEST = OUT / "coret_structural_support_tok10_lifetime_manifest_v1.json"
PREFLIGHT = OUT / "coret_structural_support_tok10_lifetime_preflight_v1.json"
RESULT = OUT / "coret_structural_support_tok10_lifetime_result_v1.json"
JOURNAL = OUT / "query_events_v1.jsonl"
QUERIES = OUT / "queries"
LAUNCHER = WORKTREE / "research_hab/run_coret_structural_support_tok10_lifetime_v1.sh"
TEST = WORKTREE / "research_hab/tests/test_coret_structural_support_lifetime_v1.py"
LIFETIME_GATE = WORKTREE / (
    "research_hab/results/coret_b2_qk_lifetime_gate_v2_20260923/"
    "coret_b2_qk_lifetime_gate_result_v2.json")
PREVIOUS_RESULT = WORKTREE / (
    "research_hab/results/coret_structural_support_tok10_real_gate_v2_20260923/"
    "coret_structural_support_tok10_real_gate_result_v2.json")
PROPERTY = frozen_runner.PROPERTY
RHO0 = frozen_runner.RHO0
MAX_DOUBLINGS = frozen_runner.MAX_DOUBLINGS
MIDPOINTS = frozen_runner.MIDPOINTS


def _write(path, payload, key="record_sha256"):
    value = dict(payload); value[key] = ref.canonical(value)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return value


def _sources():
    return [Path(__file__).resolve(), LAUNCHER, TEST,
            WORKTREE / "research_hab/coret_structural_support_lifetime_v1.py",
            WORKTREE / "research_hab/coret_structural_support_precise_dot_v1.py",
            Path(graph.__file__).resolve()]


def expected_manifest():
    scientific = frozen_runner.validate_manifest()
    prior = ref.verified(PREVIOUS_RESULT)
    gate = ref.verified(LIFETIME_GATE)
    if gate["terminal_status"] != "B2_QK_LIFETIME_OPTIMIZATION_READY":
        raise RuntimeError("accepted B2 QK lifetime gate differs")
    if prior["terminal_status"] != \
            "STRUCTURAL_SUPPORT_END_TO_END_REAL_GATE_COMPLETE":
        raise RuntimeError("previous optimized tok10 result differs")
    if prior["certified_radius"] != 0.0011090087890625:
        raise RuntimeError("previous optimized tok10 radius differs")
    return {
        "schema": "CORET_STRUCTURAL_SUPPORT_TOK10_LIFETIME_MANIFEST_V1",
        "status": "FROZEN_BEFORE_SCIENTIFIC_GATE",
        "scientific_parent_manifest_canonical_sha256": scientific[
            "canonical_manifest_sha256"],
        "property": scientific["property"],
        "example": scientific["example"],
        "threat_model": scientific["threat_model"],
        "model": scientific["model"],
        "checkpoint_sha256": scientific["checkpoint_sha256"],
        "pinned_DeepT_revision": scientific["pinned_DeepT_revision"],
        "search": scientific["search"],
        "optimized_execution": {
            **scientific["optimized_execution"],
            "B2_QK_lifetime_fence": (
                "GC+synchronize+trim unused CUDA allocator cache immediately "
                "before third QK"),
            "lifetime_fence_count_per_reached_query": 1,
        },
        "accepted_lifetime_gate": {
            "path": str(LIFETIME_GATE),
            "record_sha256": gate["record_sha256"],
            "file_sha256": ref.sha(LIFETIME_GATE),
            "baseline_seconds": gate["baseline_QK"]["wall_seconds"],
            "optimized_seconds": gate["optimized_QK"]["wall_seconds"],
            "speedup": gate["speedup"],
            "parity": gate["parity"],
        },
        "previous_optimized_reference": {
            "path": str(PREVIOUS_RESULT),
            "record_sha256": prior["record_sha256"],
            "file_sha256": ref.sha(PREVIOUS_RESULT),
            "certified_radius": prior["certified_radius"],
        },
        "acceptance": {
            "all_23_native_family_invocations_required": True,
            "all_support_claims_validated": True,
            "all_operator_checkers_required": True,
            "zero_generic_fallback": True,
            "exactly_one_B2_QK_lifetime_fence": True,
            "radius_reported_against_previous_optimized_reference": True,
        },
        "outputs": {"queries": str(QUERIES), "journal": str(JOURNAL),
                    "result": str(RESULT)},
        "source_hashes": {str(path): ref.sha(path) for path in _sources()},
        "preparation_scientific_queries": 0,
        "preparation_bound_entrypoint_calls": 0,
    }


def validate_manifest():
    stored = ref.verified(MANIFEST, "canonical_manifest_sha256")
    payload = dict(stored); payload.pop("canonical_manifest_sha256")
    if payload != expected_manifest():
        raise RuntimeError("tok10 lifetime manifest mismatch")
    return stored


def freeze():
    saved = _write(MANIFEST, expected_manifest(), "canonical_manifest_sha256")
    print(json.dumps({"terminal_status": "FROZEN_NO_SOLVE",
                      "canonical_manifest_sha256": saved[
                          "canonical_manifest_sha256"],
                      "scientific_queries": 0, "bound_entrypoint_calls": 0},
                     sort_keys=True))


def preflight():
    manifest = validate_manifest()
    if RESULT.exists() or QUERIES.exists() or JOURNAL.exists():
        raise FileExistsError("immutable tok10 lifetime artifact exists")
    record = {
        "schema": "CORET_STRUCTURAL_SUPPORT_TOK10_LIFETIME_PREFLIGHT_V1",
        "terminal_status": "PASS_NO_SOLVE",
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        "property_id": PROPERTY, "initial_rho": RHO0,
        "midpoint_iterations": MIDPOINTS,
        "previous_optimized_certified_radius": manifest[
            "previous_optimized_reference"]["certified_radius"],
        "scientific_queries": 0, "bound_entrypoint_calls": 0,
    }
    saved = _write(PREFLIGHT, record)
    print(json.dumps({"terminal_status": "PASS_NO_SOLVE",
                      "record_sha256": saved["record_sha256"]}, sort_keys=True))


def _tag(rho):
    return float(rho).hex().replace("+", "p").replace("-", "m").replace(".", "d")


def _paths(ordinal, rho):
    stem = f"query_{ordinal:02d}_{_tag(rho)}"
    return (QUERIES / f"{stem}_result_v1.json",
            QUERIES / f"{stem}_certificates_v1.json")


def _operator_timing(certificates):
    rows = []
    counters = {"QK": 0, "A.V": 0}
    for certificate in certificates:
        family = certificate["family"]
        if family not in counters:
            continue
        block = counters[family]; counters[family] += 1
        diagnostics = certificate["support_diagnostics"]
        stage = diagnostics["stage_timing_seconds"]
        rows.append({
            "block": block, "family": family,
            "structural_seconds": sum(float(value) for value in stage.values()),
            "facade_seconds": float(certificate["facade_total_seconds"]),
            "total_seconds": (sum(float(value) for value in stage.values()) +
                              float(certificate["facade_total_seconds"])),
            "stage_timing_seconds": stage,
            "executed_quadratic_MACs": diagnostics[
                "executed_quadratic_MACs"],
            "grouped_launches": diagnostics["grouped_launches"],
        })
    return rows


def _query_telemetry(delegate, dispatch):
    timings = _operator_timing(dispatch.certificates)
    b2 = [row for row in timings
          if row["block"] == 2 and row["family"] == "QK"]
    fences = list(delegate.lifetime_fence_records)
    if len(fences) != 1 or len(b2) != 1:
        raise RuntimeError("query did not record exactly one B2 QK fence/timing")
    return {"per_block_QK_AV_timing": timings,
            "B2_QK_seconds": b2[0]["total_seconds"],
            "B2_QK_structural_seconds": b2[0]["structural_seconds"],
            "lifetime_fence": fences[0]}


def query(rho, ordinal, authorized):
    if not authorized:
        raise RuntimeError("explicit USER authorization required")
    manifest = validate_manifest()
    result_path, certificate_path = _paths(ordinal, rho)
    if result_path.exists() or certificate_path.exists():
        raise FileExistsError("immutable tok10 lifetime query artifact exists")
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
    dead_holder = {"input_zonotope": z, "token_ids": ids,
                   "nominal_logits": logits, "pre_layernorm": pre}
    del z, ids, logits, pre
    delegate, dispatch = lifetime.make_dispatch(
        dead_holder, production, generator_tile=structural.AV_GENERATOR_TILE)
    bound_started = time.perf_counter()
    try:
        with torch.no_grad():
            margin, dispatch = graph.execute(
                dead_holder["input_zonotope"], model, args,
                clean_label=label, dispatch=dispatch)
    except AssertionError as error:
        if not smoke.is_native_domain_failure(error):
            raise
        torch.cuda.synchronize()
        telemetry = _query_telemetry(delegate, dispatch)
        partial_checker = frozen_runner._support_acceptance(
            dispatch.certificates)
        saved = _write(result_path, {
            "schema": "CORET_STRUCTURAL_SUPPORT_TOK10_LIFETIME_QUERY_V1",
            "terminal_status": "UNCERTIFIED_DOMAIN_FAILURE",
            "canonical_manifest_sha256": manifest[
                "canonical_manifest_sha256"],
            "property_id": PROPERTY, "query_ordinal": ordinal,
            "rho": float(rho), "rho_binary64_hex": float(rho).hex(),
            "certified": False, "authoritative_bound_returned": False,
            "exception_type": "AssertionError", "exception_message": str(error),
            **telemetry,
            "partial_operator_checkers_accepted": partial_checker,
            "generic_fallbacks": int(dispatch.generic_family_invocations),
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
        raise RuntimeError("lifetime tok10 margin interval invalid")
    counts = dispatch.assert_complete(graph.EXPECTED_THREE_BLOCK_COUNTS)
    checker_started = time.perf_counter()
    if not frozen_runner._support_acceptance(dispatch.certificates):
        raise RuntimeError("lifetime tok10 checker acceptance failed")
    checker_seconds = time.perf_counter() - checker_started
    telemetry = _query_telemetry(delegate, dispatch)
    family_timing, call_timing = frozen_runner._family_timing(
        dispatch.certificates)
    certificate_started = time.perf_counter()
    certificate = _write(certificate_path, {
        "schema": "CORET_STRUCTURAL_SUPPORT_TOK10_LIFETIME_CERTIFICATES_V1",
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        "property_id": PROPERTY, "query_ordinal": ordinal,
        "rho": float(rho), "rho_binary64_hex": float(rho).hex(),
        "invocation_counts": counts, "generic_family_invocations": 0,
        "independent_checker_accepted": True,
        "all_support_claims_validated": True,
        "complete_certificate": True,
        "provenance_ID_consistent": True,
        "family_timing_seconds": family_timing,
        "operator_calls": call_timing,
        "per_block_QK_AV_timing": telemetry["per_block_QK_AV_timing"],
        "lifetime_fence": telemetry["lifetime_fence"],
        "certificates": dispatch.certificates,
        "scientific_queries": 1, "bound_entrypoint_calls": 1,
    })
    certificate_seconds = time.perf_counter() - certificate_started
    lo, hi = float(lower.min()), float(upper.max())
    saved = _write(result_path, {
        "schema": "CORET_STRUCTURAL_SUPPORT_TOK10_LIFETIME_QUERY_V1",
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
        **telemetry,
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
        "scientific_queries": 1, "bound_entrypoint_calls": 1,
    })
    print(json.dumps({"terminal_status": "COMPLETE", "rho": rho,
                      "certified": saved["certified"],
                      "B2_QK_seconds": saved["B2_QK_seconds"],
                      "record_sha256": saved["record_sha256"]}, sort_keys=True))


def records():
    found = []
    if QUERIES.exists():
        for path in QUERIES.glob("query_*_result_v1.json"):
            item = ref.verified(path); item = dict(item); item["_path"] = path
            found.append(item)
    found.sort(key=lambda item: int(item["query_ordinal"]))
    if [item["query_ordinal"] for item in found] != list(range(len(found))):
        raise RuntimeError("tok10 lifetime query ordinals are not contiguous")
    return found


def _journal(found):
    existing = [] if not JOURNAL.exists() else smoke.load_jsonl(JOURNAL)
    seen = {item["query_result_record_sha256"] for item in existing}
    with JOURNAL.open("a", encoding="utf-8") as handle:
        for item in found:
            if item["record_sha256"] in seen:
                continue
            event = {
                "schema": "CORET_STRUCTURAL_SUPPORT_TOK10_LIFETIME_EVENT_V1",
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


def _summary(item):
    return {
        "query_ordinal": item["query_ordinal"], "rho": item["rho"],
        "terminal_status": item["terminal_status"],
        "certified": item["certified"],
        "runtime_seconds": item["runtime_seconds"],
        "setup_seconds": item["setup_seconds"],
        "bound_seconds": item["bound_seconds"],
        "B2_QK_seconds": item["B2_QK_seconds"],
        "per_block_QK_AV_timing": item["per_block_QK_AV_timing"],
        "peak_CPU_RSS_bytes": item["peak_CPU_RSS_bytes"],
        "peak_GPU_allocated_bytes": item["peak_GPU_allocated_bytes"],
        "peak_GPU_reserved_bytes": item["peak_GPU_reserved_bytes"],
        "lifetime_fence": item["lifetime_fence"],
        "record_sha256": item["record_sha256"],
    }


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
            certified = [item for item in found if item["certified"]]
            reference = manifest["previous_optimized_reference"][
                "certified_radius"]
            summaries = [_summary(item) for item in found]
            saved = _write(RESULT, {
                "schema": "CORET_STRUCTURAL_SUPPORT_TOK10_LIFETIME_RESULT_V1",
                "terminal_status": "B2_QK_LIFETIME_FIX_END_TO_END_GATE_COMPLETE",
                "canonical_manifest_sha256": manifest[
                    "canonical_manifest_sha256"],
                "property_id": PROPERTY,
                "certified_radius": state["lower"],
                "uncertified_upper": state["upper"],
                "midpoint_iterations": state["midpoints"],
                "previous_optimized_certified_radius": reference,
                "radius_ratio_to_previous_optimized": state["lower"] / reference,
                "query_count": len(found),
                "certified_query_count": len(certified),
                "domain_failure_count": sum(
                    item["terminal_status"] == "UNCERTIFIED_DOMAIN_FAILURE"
                    for item in found),
                "total_search_wall_seconds": sum(
                    item["runtime_seconds"] for item in found),
                "per_radius": summaries,
                "all_checkers_accepted": all(
                    item.get("independent_checker_accepted",
                             item.get("partial_operator_checkers_accepted", False))
                    for item in found),
                "all_provenance_consistent": all(
                    item.get("provenance_ID_consistent", True)
                    for item in found),
                "generic_fallbacks": sum(
                    item.get("generic_fallbacks", 0) for item in found),
                "lifetime_fence_count": sum(
                    1 for item in found if item.get("lifetime_fence")),
                "peak_CPU_RSS_bytes": max(
                    item["peak_CPU_RSS_bytes"] for item in found),
                "peak_GPU_allocated_bytes": max(
                    item["peak_GPU_allocated_bytes"] for item in found),
                "peak_GPU_reserved_bytes": max(
                    item["peak_GPU_reserved_bytes"] for item in found),
                "scientific_queries": len(found),
                "bound_entrypoint_calls": len(found),
            })
            print(json.dumps({"terminal_status": saved["terminal_status"],
                              "certified_radius": saved["certified_radius"],
                              "radius_ratio_to_previous_optimized": saved[
                                  "radius_ratio_to_previous_optimized"],
                              "total_search_wall_seconds": saved[
                                  "total_search_wall_seconds"],
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
            raise RuntimeError("query requires rho and ordinal")
        query(float.fromhex(values.rho), values.query_ordinal, values.authorized)
    else: run(values.authorized)
