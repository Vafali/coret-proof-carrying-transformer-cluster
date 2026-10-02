import copy
import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


if "gmpy2" not in sys.modules:
    sys.modules["gmpy2"] = types.SimpleNamespace()


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CAPTURE = load("psd_capture", "scripts/run_sound_fp64_3l_psd_state_capture_v1.py")
ORACLE = load("psd_oracle_capture_test",
              "scripts/diagnose_psd_layernorm_variance_v1.py")
RUNNER = load(
    "psd_next_layernorm_capture_runner_test",
    "scripts/run_sound_fp64_3l_psd_layernorm_experiment_v1.py")


def state_payload(generators=2):
    weights = torch.zeros(generators + 1, 1, 128, dtype=torch.float64)
    weights[0, 0, 0], weights[0, 0, 1] = 2.0, -2.0
    if generators:
        weights[1, 0, 0], weights[1, 0, 1] = 1.5, -1.5
    if generators > 1:
        weights[2, 0, 2] = 0.01
    reasons = (["native_semantic"]
               + ["fp64_roundoff_coordinate_box"] * (generators - 1))
    return {
        "weights": weights,
        "range_low": torch.full((generators,), -1.0, dtype=torch.float64),
        "range_high": torch.ones(generators, dtype=torch.float64),
        "proof": {
            "ids": [f"g{index}" for index in range(generators)],
            "masks": [1] * generators, "reasons": reasons,
            "num_tokens": 1,
        },
    }


@dataclass(frozen=True)
class FakeProof:
    ids: tuple


def test_capture_disabled_leaves_functions_unmodified():
    original_reduce = CAPTURE.sound.sound_reduce
    original_layernorm = CAPTURE.finish3l._layernorm_variance_state
    capture = CAPTURE.PassiveCapture()
    with capture.installed(enabled=False):
        assert CAPTURE.sound.sound_reduce is original_reduce
        assert CAPTURE.finish3l._layernorm_variance_state is original_layernorm
    assert CAPTURE.sound.sound_reduce is original_reduce
    assert CAPTURE.finish3l._layernorm_variance_state is original_layernorm


def test_interposer_preserves_results_and_only_records_references(monkeypatch):
    before, after = object(), object()
    proof, post_proof = FakeProof(("a", "b")), FakeProof(("a",))
    witness = {"label": CAPTURE.REDUCTION_LABEL}

    def reduction(state, state_proof, cap, label):
        assert state is before and state_proof is proof
        return after, post_proof, witness

    def layernorm(state, state_proof, label):
        return "centered", "variance", {"domain_admissible": False}

    monkeypatch.setattr(CAPTURE.sound, "sound_reduce", reduction)
    monkeypatch.setattr(CAPTURE.finish3l, "_layernorm_variance_state", layernorm)
    capture = CAPTURE.PassiveCapture()
    with capture.installed():
        result = CAPTURE.sound.sound_reduce(
            before, proof, 14000, CAPTURE.REDUCTION_LABEL)
        assert result == (after, post_proof, witness)
        ln = CAPTURE.finish3l._layernorm_variance_state(
            after, post_proof, CAPTURE.LAYERNORM_LABEL)
        assert ln == ("centered", "variance", {"domain_admissible": False})
    assert capture.pre == (before, proof)
    assert capture.post == capture.complete == (after, post_proof)
    assert capture.reduction_witness == witness
    assert CAPTURE.sound.sound_reduce is reduction
    assert CAPTURE.finish3l._layernorm_variance_state is layernorm


def write_json(path, value):
    payload = dict(value)
    payload["record_sha256"] = CAPTURE.cluster_common.canonical(payload)
    path.write_text(json.dumps(payload, sort_keys=True))


def captured_fixture(tmp_path):
    pre, post = state_payload(3), state_payload(2)
    result_dir = tmp_path / "scientific_execution" / "properties" / CAPTURE.PROPERTY_ID
    result_dir.mkdir(parents=True)
    diagnostic = {
        "reason_code": CAPTURE.finish3l.LAYERNORM_DOMAIN_REASON,
        "domain_admissible": False, "minimum_token_index": 0,
        "sound_variance_lower": -2.0, "token_count": 1,
        "generator_count": 2,
    }
    result_path = result_dir / "result.json"
    write_json(result_path, {
        "schema": CAPTURE.campaign.RESULT_SCHEMA,
        "terminal_status": "UNCERTIFIED_DOMAIN_FAILURE",
        "property_id": CAPTURE.PROPERTY_ID,
        "historical_candidate_radius": CAPTURE.TESTED_RADIUS,
        "scientific_evaluation_complete": True,
        "certified_at_historical_radius": False,
        "classification": "FAILED_AT_HISTORICAL_RADIUS",
        "failure_category": "SOUND_LAYERNORM_DOMAIN_FAILURE",
        "generic_fallback_count": 0,
        "domain_failure_diagnostic": diagnostic,
    })
    witness_path = tmp_path / "reduction_witness.json"
    witness = {
        "label": CAPTURE.REDUCTION_LABEL,
        "input_generator_count": 3, "output_generator_count": 2,
        "retained_ids": ["g0"], "dropped_ids": ["g1", "g2"],
    }
    write_json(witness_path, witness)
    identity = {
        "property_id": CAPTURE.PROPERTY_ID, "multiplier": CAPTURE.MULTIPLIER,
        "tested_radius": CAPTURE.TESTED_RADIUS,
        "tested_radius_hex": CAPTURE.TESTED_RADIUS_HEX,
        "stage_label": CAPTURE.LAYERNORM_LABEL,
        "pinned_deept_revision": CAPTURE.prefix.PINNED_REVISION,
        "scientific_manifest_sha256": "1" * 64,
        "production_manifest_sha256": "2" * 64,
        "source_set_model": CAPTURE.SOURCE_SET_MODEL,
    }
    artifact_path = tmp_path / "states.pt"
    torch.save({
        "schema": CAPTURE.ARTIFACT_SCHEMA,
        "pinned_revision": CAPTURE.prefix.PINNED_REVISION,
        "identity": identity,
        "states": {"pre_last_reduction": pre,
                   "post_last_reduction": post},
        "complete_layernorm_input_alias": "post_last_reduction",
        "reduction_witness_sha256": CAPTURE.sha256(witness_path),
    }, artifact_path)
    variants = []
    for name, key in (("pre_last_reduction", "pre_last_reduction"),
                      ("post_last_reduction", "post_last_reduction"),
                      ("complete_layernorm_input", "post_last_reduction")):
        variants.append({"capture_variant": name, "state_key": key,
                         **CAPTURE._state_hashes(
                             {"pre_last_reduction": pre,
                              "post_last_reduction": post}[key])})
    manifest_path = tmp_path / "capture_manifest.json"
    write_json(manifest_path, {
        "schema": CAPTURE.MANIFEST_SCHEMA, **identity,
        "tensor_artifact_path": artifact_path.name,
        "tensor_artifact_sha256": CAPTURE.sha256(artifact_path),
        "artifact_identity": identity,
        "result_path": str(result_path.relative_to(tmp_path)),
        "result_sha256": CAPTURE.sha256(result_path),
        "reduction_witness_path": witness_path.name,
        "reduction_witness_sha256": CAPTURE.sha256(witness_path),
        "variants": variants,
    })
    return manifest_path, artifact_path, result_path, diagnostic


def test_capture_authentication_and_label_swap_rejection(tmp_path):
    manifest, _artifact, _result, _diagnostic = captured_fixture(tmp_path)
    assert CAPTURE.verify_capture(manifest)["property_id"] == CAPTURE.PROPERTY_ID
    changed = CAPTURE.cluster_common.verified_json(manifest)
    changed.pop("record_sha256")
    changed["variants"][0]["capture_variant"] = "post_last_reduction"
    changed["variants"][1]["capture_variant"] = "pre_last_reduction"
    write_json(manifest, changed)
    with pytest.raises(RuntimeError, match="labels"):
        CAPTURE.verify_capture(manifest)


def test_tensor_or_metadata_mutation_rejects(tmp_path):
    manifest, artifact, _result, _diagnostic = captured_fixture(tmp_path)
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    payload["states"]["post_last_reduction"]["weights"][0, 0, 0] += 1.0
    torch.save(payload, artifact)
    with pytest.raises(RuntimeError, match="artifact SHA"):
        CAPTURE.verify_capture(manifest)


def test_ordered_ids_ranges_and_provenance_are_hash_bound():
    state = state_payload(2)
    first = CAPTURE._state_hashes(state)
    changed = copy.deepcopy(state)
    changed["proof"]["ids"] = list(reversed(changed["proof"]["ids"]))
    second = CAPTURE._state_hashes(changed)
    assert first["generator_sha256"] == second["generator_sha256"]
    assert first["generator_ids_sha256"] != second["generator_ids_sha256"]
    changed = copy.deepcopy(state)
    changed["range_high"][0] = 0.5
    assert (CAPTURE._state_hashes(changed)["ranges_sha256"]
            != first["ranges_sha256"])


def test_oracle_consumes_synthetic_capture_and_native_projection_is_provenance_only(tmp_path):
    manifest, artifact, result, diagnostic = captured_fixture(tmp_path)
    oracle_input = tmp_path / "oracle.json"
    write_json(oracle_input, {
        "schema": ORACLE.INPUT_SCHEMA, "property_id": ORACLE.PROPERTY_ID,
        "pinned_revision": ORACLE.PINNED_REVISION,
        "source_set_model": ORACLE.SOURCE_SET_MODEL,
        "capture_manifest": {"path": manifest.name,
                             "sha256": ORACLE.sha256(manifest)},
        "evaluations": [{
            "property_id": ORACLE.PROPERTY_ID, "multiplier": "0.75",
            "tested_radius": CAPTURE.TESTED_RADIUS, "token_index": 0,
            "source_result": {
                "path": str(result.relative_to(tmp_path)),
                "sha256": ORACLE.sha256(result)},
            "complete_state": {"path": artifact.name,
                               "sha256": ORACLE.sha256(artifact),
                               "schema": CAPTURE.ARTIFACT_SCHEMA,
                               "state_key": "post_last_reduction"},
            "pre_reduction_state": {"path": artifact.name,
                                    "sha256": ORACLE.sha256(artifact),
                                    "schema": CAPTURE.ARTIFACT_SCHEMA,
                                    "state_key": "pre_last_reduction"},
        }],
    })
    report = ORACLE.evaluate_manifest(oracle_input)
    names = [item["variant"] for item in report["results"]]
    assert names == ["complete", "native_only", "authenticated_pre_reduction"]
    native = report["results"][1]
    assert native["generator_count"] == 1
    assert native["numerical_generator_count"] == 0
    assert native["claim_scope"] == "diagnostic_native_only_subset_not_complete_state"


def test_tests_do_not_call_scientific_evaluator(monkeypatch, tmp_path):
    monkeypatch.setattr(
        CAPTURE.campaign, "execute_property",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("called")))
    state = state_payload(1)
    CAPTURE.validate_snapshot(state)
    assert CAPTURE._state_hashes(state)["generator_count"] == 1


def test_next_layernorm_capture_is_oracle_compatible_and_hash_bound(tmp_path):
    payload = state_payload(2)
    state = SimpleNamespace(
        zonotope_w=payload["weights"],
        error_term_range_low=payload["range_low"],
        error_term_range_high=payload["range_high"],
        num_error_terms=2, num_words=1, word_embedding_size=128,
        device=torch.device("cpu"))
    proof = RUNNER.experiment.structural.SupportProof(
        tuple(payload["proof"]["masks"]), tuple(payload["proof"]["ids"]),
        tuple(payload["proof"]["reasons"]), 1)
    diagnostic = {
        "reason_code": RUNNER.finish3l.LAYERNORM_DOMAIN_REASON,
        "label": RUNNER.NEXT_STAGE,
        "domain_admissible": False,
        "minimum_token_index": 0,
        "minimum_coordinate_index": 0,
        "token_count": 1,
        "hidden_dimension": 128,
        "generator_count": 2,
        "native_generator_count": 1,
        "numerical_generator_count": 1,
        "nominal_centered_second_moment": 1.0,
        "variance_affine_center": 0.5,
        "variance_lower_support": 2.5,
        "variance_upper_support": 2.5,
        "sound_variance_lower": -2.0,
        "sound_variance_upper_at_minimum": 3.0,
        "layernorm_epsilon": 1e-12,
        "sqrt_input_lower": -2.0,
        "native_sqrt_threshold": 1e-12,
        "sqrt_safety_margin": -2.0,
        "plain_relational_variance_lower_bound": 0.1,
        "numerical_variance_widening_upper_bound": 2.1,
    }
    captured = RUNNER.NextLayerNormFailureCapture()
    captured(
        state=state, proof=proof, label=RUNNER.NEXT_STAGE,
        layernorm_index=RUNNER.NEXT_LAYERNORM_INDEX,
        diagnostics=diagnostic, pre_reduction_state=state,
        pre_reduction_proof=proof,
        reduction_label=RUNNER.NEXT_REDUCTION_LABEL)
    result_path = tmp_path / "scientific_execution" / "properties" / \
        RUNNER.PROPERTY_ID / "result.json"
    result_path.parent.mkdir(parents=True)
    write_json(result_path, {
        "schema": RUNNER.campaign.RESULT_SCHEMA,
        "terminal_status": "UNCERTIFIED_DOMAIN_FAILURE",
        "property_id": RUNNER.PROPERTY_ID,
        "historical_candidate_radius": RUNNER.TESTED_RADIUS,
        "scientific_evaluation_complete": True,
        "certified_at_historical_radius": False,
        "classification": "FAILED_AT_HISTORICAL_RADIUS",
        "failure_category": "SOUND_LAYERNORM_DOMAIN_FAILURE",
        "failure_stage": "block2_to_margin",
        "failure_reason": RUNNER.finish3l.LAYERNORM_DOMAIN_REASON,
        "generic_fallback_count": 0,
        "domain_failure_diagnostic": diagnostic,
    })
    source = {
        "pinned_deept_revision": ORACLE.PINNED_REVISION,
        "scientific_manifest_sha256":
            RUNNER.cluster_common.SCIENTIFIC_MANIFEST_SHA,
        "production_manifest_sha256":
            RUNNER.cluster_common.PRODUCTION_MANIFEST_SHA,
        "source_set_model": ORACLE.SOURCE_SET_MODEL,
    }
    record = RUNNER._persist_next_capture(
        tmp_path, captured, source, result_path)
    assert record["verified_identity"]["stage_label"] == "block2_output"
    report = ORACLE.evaluate_manifest(Path(record["oracle_input_path"]))
    assert [row["variant"] for row in report["results"]] == [
        "complete", "native_only", "authenticated_pre_reduction"]
    assert all(row["stage_label"] == "block2_output"
               and row["layernorm_index"] == 6
               for row in report["results"])

    manifest_path = Path(record["manifest_path"])
    manifest = RUNNER.cluster_common.verified_json(manifest_path)
    manifest.pop("record_sha256")
    manifest["layernorm_index"] = 5
    write_json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="identity"):
        RUNNER._verify_next_capture(manifest_path)
