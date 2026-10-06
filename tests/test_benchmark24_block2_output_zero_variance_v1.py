"""CPU-only tiny fixtures. No scientific property or large zonotope execution."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import cluster_common as C
import run_transformer_benchmark24_v1 as B
import capture_benchmark24_block2_output_zero_variance_v1 as CAP
import decide_block2_output_zero_variance_v1 as D


def write_json(path, value):
    value = {k: v for k, v in value.items() if k != "record_sha256"}
    path.write_text(json.dumps({**value, "record_sha256": C.canonical(value)}, sort_keys=True))


@pytest.fixture
def capture_fixture(tmp_path, monkeypatch):
    # Production enforces exactly 14k. Unit fixtures deliberately use 2 rows
    # and never instantiate the production-sized state or a native Zonotope.
    assert CAP.EXPECTED_GENERATORS == 14000
    monkeypatch.setattr(CAP, "EXPECTED_GENERATORS", 2)
    expected = CAP.identity(B.read_protocol())
    weights = torch.zeros(3, 12, 128, dtype=torch.float64)
    weights[0, 4, 0], weights[0, 4, -1] = 1., -1.
    weights[1, 4, 0], weights[1, 4, -1] = -1., 1.
    state = {"weights": weights, "range_low": torch.full((2,), -1., dtype=torch.float64),
             "range_high": torch.ones(2, dtype=torch.float64),
             "proof": {"ids": ["native0", "numerical0"], "masks": [1 << 4, 0],
                       "reasons": ["native_semantic", "fp64_roundoff_coordinate_box"], "num_tokens": 12}}
    diagnostic = {"reason_code": "SOUND_FP64_LAYERNORM_VARIANCE_DOMAIN_FAILURE", "label": CAP.STAGE,
                  "domain_admissible": False, "minimum_token_index": 4, "minimum_coordinate_index": 0,
                  "generator_count": 2, "input_shape": [3, 12, 128], "token_count": 12,
                  "hidden_dimension": 128, "sound_variance_lower": -.3120541308930289}
    payload = {"schema": CAP.ARTIFACT_SCHEMA, "identity": expected,
               "pinned_revision": expected["source_artifacts"]["pinned_DeepT_revision"],
               "states": {"post_last_reduction": state}, "diagnostics": diagnostic}
    artifact = tmp_path / "state.pt"
    torch.save(payload, artifact)
    result = tmp_path / "result.json"
    write_json(result, {"schema": "CORET_SOUND_FP64_3L_PROPERTY_RESULT_V1", "property_id": CAP.PROPERTY_ID,
        "historical_candidate_radius": expected["tested_radius"], "historical_candidate_radius_hex": expected["tested_radius_hex"],
        "candidate_source": "cached_DeepT_reference.certified_lower_endpoint_binary64", "clean_label": expected["clean_label"],
        "terminal_status": "UNCERTIFIED_DOMAIN_FAILURE", "scientific_evaluation_complete": True,
        "certified_at_historical_radius": False, "binary_search_performed": False,
        "generic_fallback_count": 0, "domain_failure_diagnostic": diagnostic})
    manifest = tmp_path / "capture_manifest.json"
    write_json(manifest, {"schema": CAP.MANIFEST_SCHEMA, "identity": expected,
        "tensor_artifact_path": artifact.name, "tensor_artifact_sha256": C.sha256(artifact),
        "state_hashes": CAP.component_hashes(state), "diagnostics": diagnostic,
        "result_path": result.name, "result_sha256": C.sha256(result)})
    return manifest, artifact, payload


def rewrite_artifact(manifest, artifact, payload, *, component_hashes=False):
    torch.save(payload, artifact)
    value = C.verified_json(manifest)
    value["tensor_artifact_sha256"] = C.sha256(artifact)
    if component_hashes:
        value["state_hashes"] = CAP.component_hashes(payload["states"]["post_last_reduction"])
    write_json(manifest, value)


def run_capture_oracle(manifest, output, **kwargs):
    return D.execute(manifest, output, 127, expected_property_id=CAP.PROPERTY_ID,
                     variant="complete_post_reduction", exact_solve_timeout_seconds=5, **kwargs)


def test_authenticated_capture_uses_full_state_ranges_and_numerical_rows(capture_fixture):
    manifest, _, payload = capture_fixture
    state, authenticated = CAP.verify_capture(manifest)
    assert torch.equal(state["weights"], payload["states"]["post_last_reduction"]["weights"])
    variant = D._capture_variant(state, 4)
    assert len(variant["ids"]) == 2 and variant["numerical_generator_count"] == 1
    assert authenticated["token_index"] == 4
    assert authenticated["identity"]["tested_radius_hex"] == "0x1.4dc28f5c28f5bp-10"


@pytest.mark.parametrize("field", ["center", "generator", "id", "low", "high", "mask", "reason", "topology"])
def test_component_tampering_rejected_even_with_updated_file_hash(capture_fixture, field):
    manifest, artifact, payload = capture_fixture
    state = payload["states"]["post_last_reduction"]
    if field == "center": state["weights"][0, 4, 0] += .25
    elif field == "generator": state["weights"][1, 4, 0] += .25
    elif field == "id": state["proof"]["ids"][0] = "replaced_id"
    elif field == "low": state["range_low"][0] = -.5
    elif field == "high": state["range_high"][0] = .5
    elif field == "mask": state["proof"]["masks"][0] = 0
    elif field == "reason": state["proof"]["reasons"][0] = "different_lineage"
    elif field == "topology": state["weights"] = state["weights"][:-1]
    rewrite_artifact(manifest, artifact, payload)
    with pytest.raises(RuntimeError): CAP.verify_capture(manifest)


@pytest.mark.parametrize("field", ["property_id", "tested_radius_hex", "token_ids", "frozen_execution_source_hashes", "source_artifacts"])
def test_frozen_identity_tampering_rejected(capture_fixture, field):
    manifest, _, _ = capture_fixture
    value = C.verified_json(manifest)
    value["identity"][field] = "wrong"
    write_json(manifest, value)
    with pytest.raises(RuntimeError, match="frozen property/radius/tokens/source"):
        CAP.verify_capture(manifest)


def test_raw_artifact_hash_and_source_result_hash_remain_mandatory(capture_fixture):
    manifest, artifact, payload = capture_fixture
    payload["states"]["post_last_reduction"]["weights"][0, 4, 0] += .5
    torch.save(payload, artifact)
    with pytest.raises(RuntimeError, match="artifact SHA"): CAP.verify_capture(manifest)
    rewrite_artifact(manifest, artifact, payload, component_hashes=True)
    (manifest.parent / "result.json").write_text("{}")
    with pytest.raises(RuntimeError, match="source-result SHA"): CAP.verify_capture(manifest)


def test_wrong_failing_token_rejected(capture_fixture):
    manifest, artifact, payload = capture_fixture
    payload["diagnostics"]["minimum_token_index"] = 3
    rewrite_artifact(manifest, artifact, payload)
    with pytest.raises(RuntimeError, match="diagnostic/stage/token"): CAP.verify_capture(manifest)


def test_passive_callback_forwarding_preserves_bitwise_values_and_restores_function():
    captured = CAP.PassiveFailureCapture()
    tensor = torch.tensor([1., -0., 2.], dtype=torch.float64)
    original_bits = tensor.view(torch.int64).clone()
    proof = object()
    def original(*args, **kwargs):
        assert args == ("unchanged argument",)
        kwargs["experimental_layernorm_failure_capture"](state=tensor, proof=proof,
            label=CAP.STAGE, layernorm_index=6, diagnostics={"marker": "same"},
            pre_reduction_state=None, pre_reduction_proof=None, reduction_label="b2_ffn_residual")
        return tensor
    finish = SimpleNamespace(execute=original)
    with CAP.installed_callback(finish, captured):
        assert finish.execute("unchanged argument") is tensor
    assert finish.execute is original
    assert captured.state is tensor and captured.proof is proof
    assert torch.equal(tensor.view(torch.int64), original_bits)
    with pytest.raises(RuntimeError, match="twice"):
        captured(state=tensor, proof=proof, label=CAP.STAGE, layernorm_index=6,
                 diagnostics={}, pre_reduction_state=None, pre_reduction_proof=None, reduction_label="b2_ffn_residual")


def test_token4_only_feasible_requires_persisted_exact_rational_replay(capture_fixture):
    manifest, _, _ = capture_fixture
    report = run_capture_oracle(manifest, manifest.parent / "token4.json")  # default is token 4
    assert report["final_status"] == D.FEASIBLE
    assert [r["token_index"] for r in report["results"]] == [4]
    assert report["decision_scope"] == "token_4_only"
    row = report["results"][0]
    certificate = C.verified_json(Path(row["exact_zero_certificate"]["path"]))
    state, _ = CAP.verify_capture(manifest)
    problem = D._variant_problem(D._capture_variant(state, 4))
    assert D.verify_exact_zero_certificate(problem, certificate)["exact_equalities"] == 127
    bad = deepcopy(certificate)
    bad["xi_rationals"][0] = {"numerator": "1", "denominator": "2"}
    with pytest.raises(RuntimeError, match="equality"): D.verify_exact_zero_certificate(problem, bad)


def test_exact_exclusion_replays_original_equations_and_asymmetric_box():
    p = D.centered_problem([1., -1.], [[-1., 1.]], [0.], [.5], ["g"])
    certificate = D.construct_exact_exclusion_certificate(p, np.array([1., -1.]))
    assert certificate is not None
    checked = D.verify_exact_exclusion_certificate(p, certificate)
    assert checked["verified"] and checked["exact_cancellation_verified"]
    assert Fraction(**{k: int(v) for k, v in checked["exact_farkas_rhs"].items()}) < 0
    bad = deepcopy(certificate)
    bad["equality_multipliers"][0]["numerator"] = "0"
    with pytest.raises(RuntimeError, match="RHS"): D.verify_exact_exclusion_certificate(p, bad)
    changed = D.centered_problem([1., -1.], [[-1., 1.]], [0.], [2.], ["g"])
    bad = deepcopy(certificate)
    bad["problem_sha256"] = changed["problem_sha256"]
    with pytest.raises(RuntimeError, match="RHS"): D.verify_exact_exclusion_certificate(changed, bad)


def test_exact_replay_uses_original_dyadics_not_rounded_difference_matrix():
    # Numerical subtraction loses the last bit; rational replay must retain it.
    p = D.centered_problem([1., 2.**-54], [[1., 2.**-54]], [0.], [1.], ["g"])
    assert p["b"][0] == 1.
    assert D._exact_center_difference(p, 0) == Fraction(1) - Fraction(1, 2**54)
    certificate = D.construct_exact_exclusion_certificate(p, np.array([1., -1.]))
    assert D.verify_exact_exclusion_certificate(p, certificate)["verified"]


def test_numerical_positive_status_has_no_exclusion_authority(tmp_path, monkeypatch):
    state = {"center": np.array([1., -1.]), "generators": np.array([[-1., 1.]]),
             "low": np.array([0.]), "high": np.array([2.]), "ids": ["g"],
             "native_generator_count": 1, "numerical_generator_count": 0}
    monkeypatch.setattr(D, "dual_scaling_search", lambda *_: {"best": {
        "outward_safe_lower": 999., "witness_binary64_hex": [(1.).hex(), (-1.).hex()]}})
    row = D.decide_variant("untrusted", state, 127, tmp_path,
                           require_exact_exclusion=True, skip_exact=True)
    assert row["decision"] == D.INCONCLUSIVE and row["exact_exclusion_certificate"] is None


def test_all_tokens_requires_replayed_token4_exclusion_and_does_not_overclaim(capture_fixture):
    manifest, artifact, payload = capture_fixture
    with pytest.raises(RuntimeError, match="token 4 first"):
        run_capture_oracle(manifest, manifest.parent / "unauthorized.json", token_index="all")
    # Token 4 cannot reach zero, but token 0 is identically constant.
    payload["states"]["post_last_reduction"]["range_high"][0] = .5
    rewrite_artifact(manifest, artifact, payload, component_hashes=True)
    token4_path = manifest.parent / "token4.json"
    single = run_capture_oracle(manifest, token4_path, token_index="4")
    assert single["final_status"] == D.EXCLUDED
    assert single["decision_scope"] == "token_4_only"
    all_report = run_capture_oracle(manifest, manifest.parent / "all.json", token_index="all",
                                   token4_exclusion_report=token4_path)
    assert all_report["final_status"] == D.FEASIBLE
    assert [r["token_index"] for r in all_report["results"]] == [4, 0]
    path = Path(single["results"][0]["exact_exclusion_certificate"]["path"])
    path.write_text("{}")
    with pytest.raises(RuntimeError, match="certificate SHA"):
        run_capture_oracle(manifest, manifest.parent / "corrupt_all.json", token_index="all",
                           token4_exclusion_report=token4_path)


def test_feasible_token4_report_cannot_authorize_all_tokens(capture_fixture):
    manifest, _, _ = capture_fixture
    path = manifest.parent / "token4.json"
    run_capture_oracle(manifest, path)
    with pytest.raises(RuntimeError, match="not an authenticated token-4 exclusion"):
        run_capture_oracle(manifest, manifest.parent / "all.json", token_index="all", token4_exclusion_report=path)


def test_wrong_property_or_native_only_variant_rejected(capture_fixture):
    manifest, _, _ = capture_fixture
    with pytest.raises(RuntimeError, match="expected-property-id"):
        D.execute(manifest, manifest.parent / "bad.json", expected_property_id="different")
    with pytest.raises(RuntimeError, match="ALL authenticated"):
        D.execute(manifest, manifest.parent / "bad.json", expected_property_id=CAP.PROPERTY_ID,
                  variant="native_only")


def test_whole_layernorm_exclusion_requires_all_twelve_exact_certificates(capture_fixture):
    manifest, artifact, payload = capture_fixture
    state = payload["states"]["post_last_reduction"]
    for token in range(12):
        state["weights"][0, token, 0], state["weights"][0, token, -1] = 1., -1.
        state["weights"][1, token, 0], state["weights"][1, token, -1] = -1., 1.
    state["proof"]["masks"][0] = (1 << 12) - 1
    state["range_high"][0] = .5
    rewrite_artifact(manifest, artifact, payload, component_hashes=True)
    single_path = manifest.parent / "token4.json"
    single = run_capture_oracle(manifest, single_path)
    assert single["final_status"] == D.EXCLUDED
    report = run_capture_oracle(manifest, manifest.parent / "all.json", token_index="all",
                               token4_exclusion_report=single_path)
    assert report["final_status"] == D.EXCLUDED
    assert {r["token_index"] for r in report["results"]} == set(range(12))
    assert all(r["exact_exclusion_certificate"] for r in report["results"])


def test_capture_wrapper_end_to_end_with_mock_property_only(capture_fixture, monkeypatch):
    manifest, _, payload = capture_fixture
    state = payload["states"]["post_last_reduction"]
    raw = {k: v for k, v in C.verified_json(manifest.parent / "result.json").items() if k != "record_sha256"}
    raw.update(runtime_seconds=0., final_generator_count=2)
    original_bits = state["weights"].view(torch.int64).clone()
    def mock_finish(*args, **kwargs):
        kwargs["experimental_layernorm_failure_capture"](state=state, proof=state["proof"],
            label=CAP.STAGE, layernorm_index=6, diagnostics=payload["diagnostics"],
            pre_reduction_state=None, pre_reduction_proof=None, reduction_label="b2_ffn_residual")
        return raw
    finish = SimpleNamespace(execute=mock_finish)
    def mock_backend(row, workspace, artifact_root, device):
        result = finish.execute()
        B.atomic_record(workspace / "properties" / row["property_id"] / "result.json", result)
        return result, None
    def snapshot(reference, proof):
        assert reference is state and proof is state["proof"]
        return deepcopy(reference)
    monkeypatch.setitem(sys.modules, "run_sound_fp64_finish_3l_v1", finish)
    monkeypatch.setitem(sys.modules, "run_sound_fp64_3l_psd_state_capture_v1", SimpleNamespace(
        _snapshot=snapshot, _write_torch_atomic=lambda path, value: torch.save(value, path)))
    monkeypatch.setattr(B, "_existing_backend", mock_backend)
    monkeypatch.setattr(B, "artifact_errors", lambda *_: [])
    # This mocked backend executes no frozen producer. Keep the real legacy
    # source guard unchanged (it correctly rejects the new producer revision).
    monkeypatch.setattr(B, "execution_source_errors", lambda *_: [])
    output = manifest.parent / "new-capture"
    authenticated = CAP.execute(CAP.PROPERTY_ID, manifest.parent / "mock-inputs", output, "cuda:0")
    assert authenticated["identity"]["property_id"] == CAP.PROPERTY_ID
    assert finish.execute is mock_finish
    assert torch.equal(state["weights"].view(torch.int64), original_bits)
    assert set(torch.load(output / "pre_block2_output_layernorm_state.pt", weights_only=False)["states"]) == {"post_last_reduction"}
    with pytest.raises(RuntimeError, match="NEW isolated"):
        CAP.execute(CAP.PROPERTY_ID, manifest.parent / "mock-inputs", output, "cuda:0")


def test_finite_near_zero_without_exact_witness_remains_inconclusive(tmp_path, monkeypatch):
    variant = {"center": np.array([1., 2.**-52, 0.]),
               "generators": np.array([[-1., 0., 0.]]), "low": np.array([0.]), "high": np.array([2.]),
               "ids": ["g"], "native_generator_count": 1, "numerical_generator_count": 0}
    # No numerical dual candidate is given certificate authority.
    monkeypatch.setattr(D, "dual_scaling_search", lambda *_: {"best": {
        "outward_safe_lower": 0., "witness_binary64_hex": [(0.).hex()] * 3}})
    result = D.decide_variant("near", variant, 127, tmp_path, require_exact_exclusion=True,
                              exact_only_if_near_zero=False)
    assert result["decision"] == D.INCONCLUSIVE
    assert result["exact_zero_certificate_status"]["status_code"] == D.EXACT_REPLAY_FAILED
    assert result["exact_zero_certificate"] is None and result["exact_exclusion_certificate"] is None


@pytest.mark.parametrize("stage", ["bounded_least_squares_start", "exact_farkas_replay_start",
                                  "exact_reconstruction_start", "exact_exclusion_persist_and_replay_start"])
def test_hard_exclusion_watchdog_stops_slow_path_without_certificate(tmp_path, monkeypatch, stage):
    def slow(*args, **kwargs):
        D._progress(args[0], stage, time.perf_counter())
        time.sleep(2)
        pytest.fail("watchdog must terminate before this point")
    monkeypatch.setattr(D, "_decide_variant_unbounded", slow)
    variant = {"ids": ["g"], "native_generator_count": 1, "numerical_generator_count": 0}
    started = time.perf_counter()
    row = D.decide_variant("slow", variant, 127, tmp_path,
                           require_exact_exclusion=True, exact_solve_timeout_seconds=.08)
    elapsed = time.perf_counter() - started
    assert .08 <= elapsed < .6  # Fixed bounded cleanup/scheduling allowance.
    assert row["decision"] == D.INCONCLUSIVE and row["reason"] == D.EXACT_EXCLUSION_TIMEOUT
    assert row["timeout_diagnostics"]["stage"] == stage
    assert row["timeout_diagnostics"]["worker_terminated"] is True
    assert row["exact_zero_certificate"] is None and row["exact_exclusion_certificate"] is None


def test_timeout_writes_final_inconclusive_decision_file(capture_fixture, monkeypatch):
    manifest, _, _ = capture_fixture
    def slow(*args, **kwargs):
        D._progress(args[0], "exact_farkas_replay_start", time.perf_counter())
        time.sleep(2)
        pytest.fail("slow exact path must be stopped")
    monkeypatch.setattr(D, "_decide_variant_unbounded", slow)
    output = manifest.parent / "timed_decision.json"
    report = D.execute(manifest, output, expected_property_id=CAP.PROPERTY_ID,
                       variant="complete_post_reduction", token_index="4",
                       exact_solve_timeout_seconds=.08)
    persisted = C.verified_json(output)
    assert persisted == report
    assert persisted["final_status"] == D.INCONCLUSIVE
    assert persisted["reason"] == D.EXACT_EXCLUSION_TIMEOUT
    assert persisted["timeout_diagnostics"]["elapsed_seconds"] >= .08
    assert persisted["timeout_diagnostics"]["stage"] == "exact_farkas_replay_start"
    assert (manifest.parent / "timed_decision.partial.json").is_file()


def test_watchdog_does_not_swallow_programming_errors(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise AssertionError("unrelated invariant failure")
    monkeypatch.setattr(D, "_decide_variant_unbounded", broken)
    with pytest.raises(AssertionError, match="unrelated invariant failure"):
        D.decide_variant("broken", {"ids": []}, 127, tmp_path,
                         require_exact_exclusion=True, exact_solve_timeout_seconds=1)


@pytest.mark.parametrize("slow_function,expected_stage", [
    ("lsq_linear", "bounded_least_squares_start"),
    ("_farkas_replay", "exact_farkas_replay_start"),
])
def test_actual_attempt_bounds_slow_lsq_and_exact_replay(tmp_path, monkeypatch, slow_function, expected_stage):
    def stalled(*args, **kwargs):
        time.sleep(2)
        pytest.fail("stalled proposal/replay must be terminated")
    monkeypatch.setattr(D, slow_function, stalled)
    # Actual tiny infeasible LP, NOT a captured property or Transformer state.
    variant = {"center": np.array([1., -1.]), "generators": np.array([[0., 0.]]),
               "low": np.array([0.]), "high": np.array([1. if slow_function == "lsq_linear" else 0.]),
               "ids": ["g"], "native_generator_count": 1, "numerical_generator_count": 0}
    started = time.perf_counter()
    row = D.decide_variant("stalled", variant, 127, tmp_path,
                           require_exact_exclusion=True, exact_solve_timeout_seconds=.2)
    assert time.perf_counter() - started < .8
    assert row["decision"] == D.INCONCLUSIVE and row["reason"] == D.EXACT_EXCLUSION_TIMEOUT
    assert row["timeout_diagnostics"]["stage"] == expected_stage
    assert row["timeout_diagnostics"]["worker_terminated"] is True
