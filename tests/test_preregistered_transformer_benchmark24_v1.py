"""Metadata and fake-backend tests ONLY. No verifier/Torch/CUDA import."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import cluster_common as C
import preregister_transformer_benchmark24_v1 as P
import run_transformer_benchmark24_v1 as R


LOCAL_PRODUCTION = Path('/home/vafali_ubuntu/worktrees/lookahead-branching-runtime-opt-v1/research_hab/results/coret_optimized_historical_127_v1_20260924/coret_optimized_historical_127_manifest_v2.json')


def population():
    # Nine unique sentences and historical rank diversity, no outcomes required.
    rows = []
    for i in range(127):
        group, position = i % 9, i // 9 + 1
        rows.append({"property_id": f"example_{group}_tok{position:02}", "benchmark_ordinal": i,
            "sentence_id": f"example_{group}", "sentence_ordinal": group,
            "source_test_line": group + 100, "canonical_binary_test_index": group,
            "raw_sentence_sha256": f"{group:064x}", "sequence_length": 18,
            "source_dimension": 128, "token_position": position, "token": "x",
            "token_id": position, "clean_label": 1, "nominal_prediction": 1,
            "cached_DeepT_reference": {"certified_lower_endpoint_binary64": .0008 + i * 1e-6}})
    return rows


def fake_manifest():
    rows = P.select(population())
    for row in rows:
        row.update(token_ids=[1] * 18, radius_source=P.RADIUS_SOURCE,
                   sequence_length_stratum="medium_13to20", measurement_inventory=P.capabilities())
    manifest = {"schema": P.SCHEMA, "property_count": 24, "properties": rows,
                "source_population_sha256": C.canonical(P.population_projection(population())),
                "frozen_execution_source_hashes": {}, "source_artifacts": {}}
    manifest["manifest_sha256"] = C.canonical(manifest)
    return manifest


def raw(row, *, lower=1., status="COMPLETE", seconds=10.):
    return {"property_id": row["property_id"], "historical_candidate_radius_hex": row["tested_radius_hex"],
            "candidate_source": P.RADIUS_SOURCE, "clean_label": row["clean_label"],
            "terminal_status": status, "binary_search_performed": False,
            "runtime_seconds": seconds, "final_sound_lower_margin": lower,
            "certified_at_historical_radius": lower > 0 if status == "COMPLETE" else False,
            "scientific_evaluation_complete": status in ("COMPLETE", "UNCERTIFIED_DOMAIN_FAILURE"),
            "generic_fallback_count": 0, "final_generator_count": 14000,
            "numerical_widening": .001, "max_numerical_native_ratio": .01}


def test_selection_deterministic_under_reordering():
    assert P.select(population()) == P.select(list(reversed(population())))
    rows = P.select(population())
    assert len(rows) == len({r["property_id"] for r in rows}) == 24
    counts = __import__('collections').Counter(r["raw_sentence_sha256"] for r in rows)
    assert len(counts) == 9 and set(counts.values()) <= {2, 3}
    assert all(sum(r["radius_stratum"] == s for r in rows) == 8 for s in P.STRATA)


def test_no_outcome_fields_can_influence_membership_or_population_digest():
    before = population()
    mutated = deepcopy(before)
    for i, row in enumerate(mutated):
        row.update(certified=i % 2 == 0, margin=999., failure_stage="block2", certificate_status="PASS", runtime_seconds=i)
        row["cached_DeepT_reference"].update(binary_search_wall_seconds=9999., current_verifier_certified=True)
    assert P.select(before) == P.select(mutated)
    assert C.canonical(P.population_projection(before)) == C.canonical(P.population_projection(mutated))


def test_radius_not_shrunk_and_manifest_digest_is_stable():
    manifest = fake_manifest()
    assert manifest == fake_manifest()
    source = {r["property_id"]: P.radius(r) for r in population()}
    assert all(r["tested_radius"] == source[r["property_id"]] for r in manifest["properties"])
    payload = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    assert C.canonical(payload) == manifest["manifest_sha256"]


def test_modified_digest_or_wrong_expected_pin_rejects(tmp_path):
    manifest = fake_manifest()
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest))
    assert R.read_protocol(p, expected_sha=manifest["manifest_sha256"]) == manifest
    with pytest.raises(RuntimeError):
        R.read_protocol(p, expected_sha="wrong")
    manifest["properties"][0]["tested_radius"] *= .5
    p.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError):
        P.load_manifest(p)


def test_single_property_mock_executes_exactly_once_at_frozen_radius_and_resumes(monkeypatch, tmp_path):
    manifest = fake_manifest()
    row = manifest["properties"][0]
    calls = []
    monkeypatch.setattr(R, "artifact_errors", lambda *_: [])
    def backend(selected, _workspace, _artifact, device):
        assert selected == row and selected["tested_radius"] == row["tested_radius"] and device == "cpu"
        calls.append(selected["property_id"])
        return raw(selected), {"reductions": [{"support_inflation": 1e-13}], "representative_mpfr_checks_performed": False}
    result_root = tmp_path / "results"
    result = R.run_one(manifest, row["property_id"], tmp_path / "inputs", result_root, "cpu", backend=backend)
    assert calls == [row["property_id"]]
    assert result["producer_proof_status"] == "CERTIFIED_MARGIN"
    assert result["final_proof_status"] == "INCONCLUSIVE"
    assert result["independently_checked_certificate_status"] == "NOT_AVAILABLE"
    assert result["reduction_count"] == 1
    assert R.run_one(manifest, row["property_id"], tmp_path / "inputs", result_root, "cpu", backend=backend) == result
    assert len(calls) == 1


def test_missing_artifact_records_error_without_backend_or_replacement(monkeypatch, tmp_path):
    manifest = fake_manifest()
    monkeypatch.setattr(R, "artifact_errors", lambda *_: [{"kind": "FROZEN_ARTIFACT_MISSING"}])
    row = manifest["properties"][0]
    result = R.run_one(manifest, row["property_id"], tmp_path / "input", tmp_path / "result", "cpu",
                       backend=lambda *_: pytest.fail("missing input must not execute"))
    assert result["final_proof_status"] == "REJECTED_ERROR"
    assert "FROZEN_ARTIFACT_MISSING" in result["failure_reason"]
    assert len(manifest["properties"]) == 24


def test_unknown_id_cannot_run_or_replace_a_member(tmp_path):
    with pytest.raises(RuntimeError, match="no replacement"):
        R.run_one(fake_manifest(), "not_a_member", tmp_path / "input", tmp_path / "result", "cpu")


def test_cpu_execution_unsupported_is_explicit_and_import_free(tmp_path):
    with pytest.raises(RuntimeError, match="CPU_EXECUTION_UNSUPPORTED"):
        R._existing_backend({}, tmp_path, tmp_path, "cpu")


def test_aggregation_synthetic_outcomes_with_fixed_denominator_and_strata():
    manifest = fake_manifest()
    results = []
    for i, row in enumerate(manifest["properties"]):
        value = R.normalize(manifest, row, raw(row, lower=-1. if i % 3 == 1 else 1., seconds=float(i + 1)))
        if i % 3 == 0:
            # Synthetic observation from a hypothetical COMPLETE independent
            # checker. Production adapter NEVER emits this invented PASS.
            value.update(final_proof_status="CERTIFIED", independently_checked_certificate_status="PASS")
        elif i % 3 == 2:
            value.update(final_proof_status="REJECTED_ERROR", failure_stage="block0")
        results.append(value)
    summary = R.aggregate(manifest, results)
    assert summary["denominator"] == 24 and summary["not_run"] == 0
    assert summary["certified"] == summary["inconclusive"] == summary["rejected_error"] == 8
    assert summary["independent_certificate_check_success"] == 8
    assert summary["runtime_median_seconds"] == 12.5 and summary["runtime_max_seconds"] == 24
    assert summary["failure_stage_histogram"] == {"block0": 8}
    assert all(group["denominator"] == 8 for group in summary["by_radius_stratum"].values())


def test_unrun_results_are_not_reported_as_scientific_failures():
    summary = R.aggregate(fake_manifest(), [])
    assert summary["not_run"] == 24
    assert summary["certified"] == summary["inconclusive"] == summary["rejected_error"] == 0


def test_aggregation_rejects_duplicates_radii_and_false_certificate_claims():
    manifest = fake_manifest()
    row = manifest["properties"][0]
    result = R.normalize(manifest, row, raw(row))
    with pytest.raises(RuntimeError, match="duplicate"):
        R.aggregate(manifest, [result, result])
    bad = {**result, "tested_radius": 0.}
    with pytest.raises(RuntimeError):
        R.aggregate(manifest, [bad])
    bad = {**result, "final_proof_status": "CERTIFIED"}
    with pytest.raises(RuntimeError, match="independent"):
        R.aggregate(manifest, [bad])


def test_domain_miss_is_inconclusive_not_infrastructure():
    manifest = fake_manifest()
    row = manifest["properties"][0]
    result = R.normalize(manifest, row, raw(row, status="UNCERTIFIED_DOMAIN_FAILURE"))
    assert result["final_proof_status"] == "INCONCLUSIVE" and result["inconclusive_reason"] == "SOUND_DOMAIN_FAILURE"


def test_raw_property_or_radius_substitution_rejects():
    manifest = fake_manifest()
    row = manifest["properties"][0]
    with pytest.raises(RuntimeError):
        R.normalize(manifest, row, {**raw(row), "property_id": "wrong"})
    with pytest.raises(RuntimeError):
        R.normalize(manifest, row, {**raw(row), "historical_candidate_radius_hex": "0x0.0p+0"})


def test_real_frozen_population_selection_is_metadata_only_and_distinct():
    if not LOCAL_PRODUCTION.exists():
        pytest.skip("local historical metadata unavailable")
    properties = json.loads(LOCAL_PRODUCTION.read_text())["properties"]
    selected = P.select(properties)
    assert len({(r["raw_sentence_sha256"], r["token_position"]) for r in selected}) == 24
    assert len({r["raw_sentence_sha256"] for r in selected}) == 9
    assert {r["sequence_length"] for r in selected} == {8, 12, 16, 17, 18, 20, 22, 27}


def test_actual_frozen_manifest_rebuild_is_identical_without_evaluation():
    historical_root = Path('/mnt/c/users/david-despacho/documents/vafali projects/lookahead-branching/research_hab/results/coret_deept_paper_benchmark_v5_20260917')
    scientific = historical_root / 'coret_deept_paper_benchmark_execution_manifest_v5.json'
    positions = historical_root / 'deept_positions_v5.jsonl'
    if not all(path.exists() for path in (LOCAL_PRODUCTION, scientific, positions)):
        pytest.skip("local frozen metadata unavailable")
    frozen = R.read_protocol()
    rebuilt = P.build_manifest(LOCAL_PRODUCTION, scientific, positions)
    assert rebuilt == frozen
    assert frozen['manifest_sha256'] == R.FROZEN_MANIFEST_SHA
    assert frozen['preparation_scientific_queries'] == frozen['preparation_bound_calls'] == 0
