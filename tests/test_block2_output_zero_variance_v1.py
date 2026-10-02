from __future__ import annotations

import copy
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "zero_variance_decision",
    ROOT / "scripts/decide_block2_output_zero_variance_v1.py")
DECIDE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DECIDE)


def problem(center, generators, low, high):
    return DECIDE.centered_problem(center, generators, low, high)


def test_exact_synthetic_zero_variance_is_certified():
    value = problem([1.0, -1.0], [[-1.0, 1.0]], [0.0], [2.0])
    upper = DECIDE.exact_candidate_upper(value, np.array([1.0]))
    assert upper["variance_upper_outward_binary64"] == 0.0
    certificate, status = DECIDE.construct_exact_zero_certificate(
        value, np.array([1.0]), 2)
    assert status["verified"] is True
    assert DECIDE.verify_exact_zero_certificate(
        value, certificate)["exact_variance"] == "0"


def test_strictly_positive_state_has_positive_outward_dual():
    value = problem([1.0, -1.0], [[0.0, 0.0]], [0.0], [0.0])
    candidate, primal = DECIDE.numerical_primal_search(value)
    assert primal["numerical_variance"] == 1.0
    dual = DECIDE.dual_scaling_search(value, candidate)
    assert dual["best"]["outward_safe_lower"] > 0.999999999999


def test_asymmetric_ranges_are_respected_by_exact_candidate():
    value = problem([1.0, -1.0], [[-2.0, 2.0]], [0.25], [0.75])
    certificate, status = DECIDE.construct_exact_zero_certificate(
        value, np.array([0.5]), 2)
    assert status["verified"] is True
    assert DECIDE.verify_exact_zero_certificate(
        value, certificate)["maximum_exact_residual"] == "0"


def test_zero_dual_baseline_is_exactly_zero():
    baseline = DECIDE.exact_zero_dual_baseline(128)
    assert baseline["exact_objective"] == "0"
    assert baseline["outward_safe_lower"] == 0.0


def test_exact_rational_candidate_outside_range_is_rejected():
    value = problem([1.0, -1.0], [[-1.0, 1.0]], [0.0], [2.0])
    certificate, _status = DECIDE.construct_exact_zero_certificate(
        value, np.array([1.0]), 2)
    mutated = copy.deepcopy(certificate)
    mutated["xi_rationals"][0] = {"numerator": "3", "denominator": "1"}
    with pytest.raises(RuntimeError, match="outside range"):
        DECIDE.verify_exact_zero_certificate(value, mutated)


def test_exact_rational_equality_mutation_is_rejected():
    value = problem([1.0, -1.0], [[-1.0, 1.0]], [0.0], [2.0])
    certificate, _status = DECIDE.construct_exact_zero_certificate(
        value, np.array([1.0]), 2)
    mutated = copy.deepcopy(certificate)
    mutated["xi_rationals"][0] = {"numerator": "1", "denominator": "2"}
    with pytest.raises(RuntimeError, match="equality"):
        DECIDE.verify_exact_zero_certificate(value, mutated)


def _write_json(path: Path, value: dict):
    payload = dict(value)
    payload["record_sha256"] = DECIDE.cluster_common.canonical(payload)
    path.write_text(json.dumps(payload, sort_keys=True))


def _capture_fixture(tmp_path: Path):
    def state(generators):
        weights = torch.zeros(generators + 1, 1, 128, dtype=torch.float64)
        weights[0, 0, 0], weights[0, 0, 1] = 1.0, -1.0
        if generators:
            weights[1, 0, 0], weights[1, 0, 1] = -1.0, 1.0
        reasons = (["native_semantic"]
                   + ["fp64_roundoff_coordinate_box"] * (generators - 1))
        return {
            "weights": weights,
            "range_low": torch.full((generators,), -1.0, dtype=torch.float64),
            "range_high": torch.ones(generators, dtype=torch.float64),
            "proof": {"ids": [f"g{index}" for index in range(generators)],
                      "masks": [1] * generators, "reasons": reasons,
                      "num_tokens": 1},
        }

    pre, post = state(3), state(2)
    identity = {
        "property_id": DECIDE.PROPERTY_ID, "multiplier": DECIDE.MULTIPLIER,
        "tested_radius": 0.00060791015625,
        "tested_radius_hex": float(0.00060791015625).hex(),
        "stage_label": DECIDE.STAGE,
        "layernorm_index": DECIDE.LAYERNORM_INDEX,
        "pinned_deept_revision": DECIDE.oracle.PINNED_REVISION,
        "scientific_manifest_sha256":
            DECIDE.cluster_common.SCIENTIFIC_MANIFEST_SHA,
        "production_manifest_sha256":
            DECIDE.cluster_common.PRODUCTION_MANIFEST_SHA,
        "source_set_model": DECIDE.oracle.SOURCE_SET_MODEL,
    }
    artifact = tmp_path / "states.pt"
    torch.save({
        "schema": DECIDE.CAPTURE_SCHEMA,
        "pinned_revision": DECIDE.oracle.PINNED_REVISION,
        "identity": identity,
        "states": {"pre_last_reduction": pre,
                   "post_last_reduction": post},
        "complete_layernorm_input_alias": "post_last_reduction",
        "reduction_label": DECIDE.REDUCTION_LABEL,
    }, artifact)
    diagnostic = {
        "label": DECIDE.STAGE, "domain_admissible": False,
        "minimum_token_index": 0, "generator_count": 2,
    }
    result = tmp_path / "result.json"
    _write_json(result, {
        "property_id": DECIDE.PROPERTY_ID,
        "terminal_status": "UNCERTIFIED_DOMAIN_FAILURE",
        "generic_fallback_count": 0,
        "domain_failure_diagnostic": diagnostic,
    })
    variants = []
    for name, key in (
            ("pre_last_reduction", "pre_last_reduction"),
            ("post_last_reduction", "post_last_reduction"),
            ("complete_layernorm_input", "post_last_reduction")):
        variants.append({"capture_variant": name, "state_key": key,
                         **DECIDE._state_hashes({
                             "pre_last_reduction": pre,
                             "post_last_reduction": post}[key])})
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {
        "schema": DECIDE.CAPTURE_MANIFEST_SCHEMA, **identity,
        "tensor_artifact_path": artifact.name,
        "tensor_artifact_sha256": DECIDE.oracle.sha256(artifact),
        "artifact_identity": identity,
        "result_path": result.name,
        "result_sha256": DECIDE.oracle.sha256(result),
        "reduction_label": DECIDE.REDUCTION_LABEL,
        "variants": variants,
    })
    return manifest, artifact


def test_authenticated_capture_and_tensor_mutation_rejection(tmp_path):
    manifest, artifact = _capture_fixture(tmp_path)
    variants, identity = DECIDE._load_authenticated_variants(manifest)
    assert set(variants) == {
        "complete_post_reduction", "native_only",
        "authenticated_pre_reduction"}
    assert identity["artifact_sha256"] == DECIDE.oracle.sha256(artifact)
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    payload["states"]["post_last_reduction"]["weights"][0, 0, 0] += 1.0
    torch.save(payload, artifact)
    with pytest.raises(RuntimeError, match="artifact SHA"):
        DECIDE._load_authenticated_variants(manifest)


def test_end_to_end_synthetic_capture_decision_is_exact(tmp_path):
    manifest, _artifact = _capture_fixture(tmp_path)
    report = DECIDE.execute(manifest, tmp_path / "decision.json", 128)
    assert report["complete_state_decision"] == DECIDE.DECISION_ZERO
    assert all(row["decision"] == DECIDE.DECISION_ZERO
               and row["exact_zero_certificate_status"]["verified"] is True
               for row in report["results"])
    assert report["scientific_queries"] == report["bound_calls"] == 0
    assert (tmp_path / "decision.partial.json").is_file()


def _variant(center=(1.0, -1.0), generator=(-1.0, 1.0),
             low=0.0, high=2.0):
    return {
        "center": np.array(center),
        "generators": np.array([generator]),
        "low": np.array([low]), "high": np.array([high]),
        "ids": ["g0"], "reasons": ["native_semantic"],
        "native_generator_count": 1, "numerical_generator_count": 0,
    }


def test_positive_dual_stops_before_exact_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(
        DECIDE, "construct_exact_zero_certificate",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("exact called")))
    row = DECIDE.decide_variant(
        "positive", _variant(generator=(0.0, 0.0), low=0.0, high=0.0),
        128, tmp_path)
    assert row["decision"] == DECIDE.DECISION_POSITIVE
    assert row["exact_zero_certificate_status"]["attempted"] is False


def test_non_near_candidate_stops_without_claiming_infeasibility(
        tmp_path, monkeypatch):
    monkeypatch.setattr(DECIDE, "dual_scaling_search", lambda *_a, **_k: {
        "zero_baseline": DECIDE.exact_zero_dual_baseline(2),
        "candidates": [], "best": {"outward_safe_lower": 0.0}})
    monkeypatch.setattr(
        DECIDE, "construct_exact_zero_certificate",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("exact called")))
    row = DECIDE.decide_variant(
        "non_near", _variant(generator=(0.0, 0.0), low=0.0, high=0.0),
        128, tmp_path)
    assert row["near_zero_gate"]["is_compelling_near_zero"] is False
    assert row["decision"] == DECIDE.DECISION_UNRESOLVED
    assert "did not pass near-zero" in row[
        "exact_zero_certificate_status"]["reason"]


def test_skip_exact_leaves_near_zero_candidate_unresolved(tmp_path, monkeypatch):
    monkeypatch.setattr(DECIDE, "dual_scaling_search", lambda *_a, **_k: {
        "zero_baseline": DECIDE.exact_zero_dual_baseline(2),
        "candidates": [], "best": {"outward_safe_lower": 0.0}})
    monkeypatch.setattr(
        DECIDE, "construct_exact_zero_certificate",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("exact called")))
    row = DECIDE.decide_variant(
        "skip", _variant(), 128, tmp_path, skip_exact=True)
    assert row["near_zero_gate"]["is_compelling_near_zero"] is True
    assert row["decision"] == DECIDE.DECISION_UNRESOLVED
    assert row["exact_zero_certificate_status"]["reason"] == (
        "exact stage disabled by --skip-exact")


def test_progress_is_flushed_after_each_numerical_stage(capsys):
    value = problem([1.0, -1.0], [[-1.0, 1.0]], [0.0], [2.0])
    DECIDE.numerical_primal_search(value, "progress_fixture")
    output = capsys.readouterr().out
    assert "highs_feasibility_complete" in output
    assert "bounded_least_squares_complete" in output
    assert "numerical_primal_stage_complete" in output


def test_authenticated_range_hash_mutation_rejection(tmp_path):
    manifest, artifact = _capture_fixture(tmp_path)
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    payload["states"]["post_last_reduction"]["range_high"][0] = 0.5
    torch.save(payload, artifact)
    changed = DECIDE.cluster_common.verified_json(manifest)
    changed.pop("record_sha256")
    changed["tensor_artifact_sha256"] = DECIDE.oracle.sha256(artifact)
    _write_json(manifest, changed)
    with pytest.raises(RuntimeError, match="state hash"):
        DECIDE._load_authenticated_variants(manifest)


def test_diagnostic_does_not_import_or_modify_production_verifier():
    source = inspect.getsource(DECIDE)
    assert "run_sound_fp64_finish_3l_v1" not in source
    assert "coret_sound_fp64_block0_feasibility_v1" not in source
