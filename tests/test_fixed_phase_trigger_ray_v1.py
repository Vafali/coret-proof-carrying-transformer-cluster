"""Bounded CPU-only ray acquisition fixtures; no scientific artifacts loaded."""
from fractions import Fraction as F
from dataclasses import replace

import numpy as np
import pytest

from test_fixed_phase_certificate_repair_v1 import O, triangle_lp, fake_dedicated_solver
from test_fixed_phase_layernorm_witness_v1 import fixture, infeasible_fixed_phase_fixture
import highspy


def trigger(lp, method="SIMPLEX_FEASIBILITY"):
    return O.solve_highspy(lp, proposal_only=True, proposal_method=method, time_limit_seconds=1.)


def forbidden_dedicated(*_):
    pytest.fail("usable exact trigger certificate must bypass dedicated solve")


@pytest.mark.parametrize("method", ["PRIMARY", "SIMPLEX_FEASIBILITY"])
def test_real_trigger_ray_goes_directly_to_existing_exact_repair_and_full_replay(method, monkeypatch):
    lp = triangle_lp(F(1, 2**47))
    # Keep the scaled contradictory RHS above HiGHS feasibility tolerance;
    # tiny coefficients still exercise the reversible power-of-two mapping.
    lp = replace(lp, rows=tuple(replace(row, upper=row.upper * 1024) for row in lp.rows))
    proposal = trigger(lp, method)
    assert proposal["model_status"] == "Infeasible"
    assert proposal["trigger_ray_original_space_mapping_verified"]
    assert proposal["trigger_ray_raw_dimension"] == len(lp.rows)
    assert not proposal["infeasible"]  # Proposal-only numerical status is not proof.
    if method == "SIMPLEX_FEASIBILITY":
        assert proposal["options"]["simplex_strategy"] == 1
        assert proposal["options"]["presolve"] == "on"
    assert proposal["solver_scaling"]["rows_scaled"] == len(lp.rows)
    scales = np.asarray([float(F(v)) for v in proposal["solver_scaling"]["row_scales_exact"]])
    np.testing.assert_array_equal(proposal["original_row_dual_ray"], proposal["raw_dual_ray"] * scales)
    repair, calls = O.repair_fixed_phase_dual_ray, []
    def recorded(model, ray, seconds, **kwargs):
        assert model is lp
        np.testing.assert_array_equal(ray, proposal["original_row_dual_ray"])
        calls.append(ray)
        return repair(model, ray, seconds, **kwargs)
    monkeypatch.setattr(O, "repair_fixed_phase_dual_ray", recorded)
    report = O.attempt_fixed_phase_certificate(lp, method, 2., trigger_proposal=proposal,
        certificate_solver=forbidden_dedicated, factor_cache={})
    assert len(calls) == 1 and report["exact_farkas_replay_verified"]
    assert report["fixed_phase_farkas_verified"]
    assert report["certificate_ray_source"] == "TRIGGER_" + method
    assert not report["dedicated_certificate_fallback_used"]
    assert report["normalized_ray_candidate_sha256"]
    assert O.verify_exact_lp_farkas(lp, report["fixed_phase_farkas_certificate"])["verified"]
    assert not report["permits_node_exclusion"]


@pytest.mark.parametrize("method", ["PRIMARY", "IPM_FEASIBILITY", "SIMPLEX_FEASIBILITY"])
def test_each_trigger_source_preserves_phase_local_acceptance_only(method, monkeypatch):
    p, lp, pattern, _ = infeasible_fixed_phase_fixture()
    candidate = trigger(lp, "PRIMARY")
    # Ray candidates have no solver proof authority; rebinding a mock method
    # simulates an exposed IPM/crossover ray, replay still uses the original LP.
    candidate["solver_method"] = method
    def primary(*_):
        return candidate if method == "PRIMARY" else {"model_status": "Unknown", "column_values": None}
    def fallback(model, seconds, current, role):
        return candidate if current == method else {"solver_method": current,
            "model_status": "Unknown", "column_values": None}
    monkeypatch.setattr(O, "FixedPhaseAnchorWorkspace", lambda *_: pytest.fail("no anchor on Infeasible"))
    _, report = O.search_fixed_phase_exact_witness(p, lp, pattern, primary, 3.,
        feasibility_propose=fallback, certificate_solver=forbidden_dedicated)
    assert report["certificate_ray_source"] == "TRIGGER_" + method
    assert report["search_status"] == "FIXED_PHASE_LINEARLY_INFEASIBLE"
    assert report["exact_farkas_replay_verified"]
    assert not report["permits_infeasibility_claim"] and not report["permits_node_exclusion"]


@pytest.mark.parametrize("defect", ["wrong_dimension", "missing_mapping", "raw_bits",
    "mapped_bits", "wrong_lp", "wrong_scaled_lp", "wrong_scale", "mapping_twice", "nonfinite"])
def test_ambiguous_trigger_ray_rejected_and_dedicated_fallback_used(defect):
    lp = triangle_lp()
    proposal = trigger(lp)
    if defect == "wrong_dimension":
        proposal["raw_dual_ray"] = np.ones(2)
        proposal["original_row_dual_ray"] = np.ones(2)
    elif defect == "missing_mapping":
        del proposal["row_ray_mapping"]
    elif defect == "raw_bits":
        proposal["raw_dual_ray"][0] += 1
    elif defect == "mapped_bits":
        proposal["original_row_dual_ray"][0] += 1
    elif defect == "wrong_lp":
        proposal["row_ray_mapping"]["canonical_lp_sha256"] = "wrong"
    elif defect == "wrong_scaled_lp":
        proposal["row_ray_mapping"]["solver_scaled_lp_sha256"] = "wrong"
    elif defect == "wrong_scale":
        proposal["row_ray_mapping"]["row_scaling_sha256"] = "wrong"
    elif defect == "mapping_twice":
        proposal["row_ray_mapping"]["mapping_count"] = 2
    elif defect == "nonfinite":
        proposal["original_row_dual_ray"][0] = np.nan
    calls = []
    def dedicated(model, seconds):
        calls.append(model)
        return fake_dedicated_solver(model)
    report = O.attempt_fixed_phase_certificate(lp, "SIMPLEX_FEASIBILITY", 2.,
        trigger_proposal=proposal, certificate_solver=dedicated, factor_cache={})
    assert len(calls) == 1 and report["dedicated_certificate_fallback_used"]
    assert report["trigger_ray_rejected_reason"]
    assert not report["trigger_ray_original_space_mapping_verified"]
    assert report["certificate_ray_source"] == "DEDICATED_CERTIFICATE"
    assert report["exact_farkas_replay_verified"]


def test_absent_trigger_ray_falls_back_and_no_ray_remains_inconclusive():
    lp = triangle_lp()
    calls = []
    def dedicated(model, seconds):
        calls.append(model)
        return fake_dedicated_solver(model, infeasible=False)
    report = O.attempt_fixed_phase_certificate(lp, "IPM_FEASIBILITY", 2.,
        trigger_proposal={"model_status": "Infeasible"}, certificate_solver=dedicated)
    assert len(calls) == 1 and report["dedicated_certificate_fallback_used"]
    assert report["trigger_ray_rejected_reason"] == "NO_EXPOSED_TRIGGER_RAY"
    assert report["direct_ray_repair_status"] == "NO_USABLE_DEDICATED_DUAL_RAY"
    assert not report["fixed_phase_farkas_verified"] and not report["exact_farkas_replay_verified"]


def test_duplicate_trigger_and_dedicated_direction_repaired_once_even_scaled_and_reversed(monkeypatch):
    lp = triangle_lp()
    proposal = trigger(lp)
    repairs, solves = [], []
    def failed_repair(model, ray, seconds, **kwargs):
        repairs.append(ray.copy())
        return None, [], "DIRECT_RAY_SUPPORT_REPAIR_FAILED"
    def dedicated(model, seconds):
        solves.append(model)
        result = fake_dedicated_solver(model)
        result["original_row_dual_ray"] = -8. * proposal["original_row_dual_ray"]
        return result
    monkeypatch.setattr(O, "repair_fixed_phase_dual_ray", failed_repair)
    report = O.attempt_fixed_phase_certificate(lp, "SIMPLEX_FEASIBILITY", 2.,
        trigger_proposal=proposal, certificate_solver=dedicated)
    assert len(repairs) == len(solves) == 1
    assert report["duplicate_ray_candidate_skipped"]
    assert len(report["ray_candidate_sha256s"]) == len(report["ray_candidate_attempts"]) == 1
    assert not report["exact_farkas_replay_verified"]


def test_trigger_infeasible_and_claimed_certificate_cannot_bypass_original_lp_replay(monkeypatch):
    p, lp, pattern, certificate = infeasible_fixed_phase_fixture()
    candidate = trigger(lp, "PRIMARY")
    certificate["multipliers"][0]["multiplier"] = "-1/1"
    monkeypatch.setattr(O, "repair_fixed_phase_dual_ray", lambda *args, **kwargs: (
        certificate, [{"exact_farkas_replay_verified": True}], "EXACT_FARKAS_VERIFIED"))
    _, report = O.search_fixed_phase_exact_witness(p, lp, pattern, lambda *_: candidate, 3.,
        certificate_solver=lambda model, _: fake_dedicated_solver(model, infeasible=False))
    assert report["search_status"] == "INCONCLUSIVE"
    assert not report["exact_farkas_replay_verified"] and not report["fixed_phase_farkas_verified"]
    assert "negative" in report["exact_certificate_rejection"]
    assert not report["permits_infeasibility_claim"]


def test_materially_new_trigger_after_cached_no_ray_gets_repair_without_repeating_dedicated_solve():
    lp, cache, calls = triangle_lp(), {}, []
    def dedicated(model, seconds):
        calls.append(model)
        return fake_dedicated_solver(model, infeasible=False)
    first = O.attempt_fixed_phase_certificate(lp, "IPM_FEASIBILITY", 2.,
        certificate_solver=dedicated, attempt_cache=cache)
    assert not first["fixed_phase_farkas_verified"]
    candidate = trigger(lp)
    second = O.attempt_fixed_phase_certificate(lp, "SIMPLEX_FEASIBILITY", 2.,
        trigger_proposal=candidate, certificate_solver=dedicated, attempt_cache=cache, factor_cache={})
    assert second["certificate_attempt_cache_hit"] and second["exact_farkas_replay_verified"]
    assert len(calls) == 1
    assert cache[lp.identity()]["audit"]["exact_farkas_replay_verified"]


def test_same_trigger_in_later_method_deduplicates_failed_repair_and_dedicated_solve(monkeypatch):
    lp, cache = triangle_lp(), {}
    proposal = trigger(lp)
    calls, solves = [], []
    def repair(model, ray, seconds, **kwargs):
        calls.append(ray)
        return None, [], "DIRECT_RAY_SUPPORT_REPAIR_FAILED"
    def dedicated(model, seconds):
        solves.append(model)
        return fake_dedicated_solver(model, infeasible=False)
    monkeypatch.setattr(O, "repair_fixed_phase_dual_ray", repair)
    for method in ("SIMPLEX_FEASIBILITY", "PRIMARY"):
        proposal["solver_method"] = method
        result = O.attempt_fixed_phase_certificate(lp, method, 2., trigger_proposal=proposal,
            certificate_solver=dedicated, attempt_cache=cache)
        assert not result["exact_farkas_replay_verified"]
    assert len(calls) == len(solves) == 1
    assert result["certificate_attempt_cache_hit"] and result["duplicate_ray_candidate_skipped"]


@pytest.mark.parametrize("method", ["IPM_FEASIBILITY", "SIMPLEX_FEASIBILITY"])
@pytest.mark.parametrize("exposed", [False, True])
def test_wrapper_harvests_only_unambiguous_incumbent_rows_from_same_instance(monkeypatch, method, exposed):
    lp, real, instances = triangle_lp(), highspy.Highs, []
    class Engine:
        def __init__(self):
            self.inner = real()
            self.get_ray_calls = 0
            instances.append(self)
        def __getattr__(self, name):
            return getattr(self.inner, name)
        def getDualRayExist(self):
            return highspy.HighsStatus.kOk, exposed
        def getDualRay(self):
            self.get_ray_calls += 1
            # A valid original-row candidate. Exact replay, not this assertion,
            # proves the resulting certificate against the original LP.
            return highspy.HighsStatus.kOk, True, -np.ones(len(lp.rows))
    monkeypatch.setattr(highspy, "Highs", Engine)
    result = trigger(lp, method)
    assert len(instances) == 1
    assert instances[0].get_ray_calls == 1
    assert result["trigger_ray_cached_before_request"] is exposed
    assert result["trigger_ray_forced_request_attempted"] is (not exposed)
    assert result["trigger_ray_original_space_mapping_verified"]
    report = O.attempt_fixed_phase_certificate(lp, method, 2., trigger_proposal=result,
        certificate_solver=forbidden_dedicated, factor_cache={})
    assert report["exact_farkas_replay_verified"]
    assert report["trigger_ray_forced_request_attempted"] is (not exposed)
    if not exposed:
        assert report["trigger_ray_forced_request_status"] == str(highspy.HighsStatus.kOk)
        assert report["trigger_ray_forced_request_has_ray"] is True
        assert report["trigger_ray_forced_request_seconds"] >= 0.


def test_wrapper_wrong_dimensional_ray_is_rejected_without_mapping(monkeypatch):
    lp, real = triangle_lp(), highspy.Highs
    class Engine:
        def __init__(self):
            self.inner = real()
        def __getattr__(self, name):
            return getattr(self.inner, name)
        def getDualRayExist(self):
            return highspy.HighsStatus.kOk, True
        def getDualRay(self):
            return highspy.HighsStatus.kOk, True, np.ones(2)
    monkeypatch.setattr(highspy, "Highs", Engine)
    result = trigger(lp)
    assert result["trigger_ray_raw_dimension"] == 2
    assert result["trigger_ray_expected_original_row_dimension"] == 3
    assert result["original_row_dual_ray"] is None and result["row_ray_mapping"] is None
    assert result["trigger_ray_rejected_reason"] == "TRIGGER_RAY_DIMENSION_AMBIGUOUS"


def test_presolved_ray_cannot_bypass_original_variable_bound_replay(monkeypatch):
    lp = O.ExactCanonicalLP(("x",), (F(1),), (F(2),),
        (O.ExactLPRow("x_le_zero", (0,), (F(1),), None, F(0)),))
    candidate = trigger(lp)
    original_verify, calls = O.verify_exact_lp_farkas, []
    def replay(model, certificate):
        assert model is lp and model.column_lower == (F(1),) and model.column_upper == (F(2),)
        calls.append(certificate)
        return original_verify(model, certificate)
    monkeypatch.setattr(O, "verify_exact_lp_farkas", replay)
    result = O.attempt_fixed_phase_certificate(lp, "SIMPLEX_FEASIBILITY", 2.,
        trigger_proposal=candidate, certificate_solver=forbidden_dedicated, factor_cache={})
    assert len(calls) == 1 and result["exact_farkas_replay_verified"]
    assert any(entry["kind"] == "column" for entry in calls[0]["multipliers"])


def test_real_presolve_infeasible_without_cached_ray_requests_values_on_same_instance(monkeypatch, tmp_path):
    lp = O.ExactCanonicalLP(("x",), (F(1),), (F(2),),
        (O.ExactLPRow("x_le_zero", (0,), (F(1),), None, F(0)),))
    real, instances = highspy.Highs, []
    class Engine:
        def __init__(self):
            self.inner = real()
            self.value_requests = 0
            instances.append(self)
        def __getattr__(self, name):
            return getattr(self.inner, name)
        def getDualRayExist(self):
            self.cached_result = self.inner.getDualRayExist()
            return self.cached_result
        def getDualRay(self):
            self.value_requests += 1
            return self.inner.getDualRay()
    monkeypatch.setattr(highspy, "Highs", Engine)
    result = O.solve_highspy(lp, tmp_path / "forced.log", proposal_only=True,
        proposal_method="SIMPLEX_FEASIBILITY", time_limit_seconds=1.)
    assert result["model_status"] == "Infeasible"
    assert len(instances) == 1 and instances[0].cached_result == (highspy.HighsStatus.kOk, False)
    assert instances[0].value_requests == 1
    assert result["trigger_ray_pre_request_reason"] == "NO_CACHED_TRIGGER_RAY"
    assert result["trigger_ray_forced_request_attempted"]
    assert result["trigger_ray_forced_request_has_ray"]
    assert result["trigger_ray_forced_request_status"] == str(highspy.HighsStatus.kOk)
    assert result["trigger_ray_raw_dimension"] == 1
    assert result["trigger_ray_original_space_mapping_verified"]
    import json
    persisted = json.loads((tmp_path / "forced.proposal.json").read_text())
    for key in ("trigger_ray_cached_before_request", "trigger_ray_forced_request_attempted",
                "trigger_ray_forced_request_status", "trigger_ray_forced_request_has_ray",
                "trigger_ray_forced_request_seconds"):
        assert persisted[key] == result[key]
    # Retrieval and numerical Infeasible status have zero proof authority.
    assert not result["infeasible"] and "exact_farkas_replay_verified" not in result
    report = O.attempt_fixed_phase_certificate(lp, "SIMPLEX_FEASIBILITY", 2., trigger_proposal=result,
        certificate_solver=forbidden_dedicated, factor_cache={})
    assert report["exact_farkas_replay_verified"] and report["full_original_lp_replay_completed"]
    assert report["certificate_ray_source"] == "TRIGGER_SIMPLEX_FEASIBILITY"
    assert not report["dedicated_certificate_fallback_used"]
    assert O.verify_exact_lp_farkas(lp, report["fixed_phase_farkas_certificate"])["verified"]


@pytest.mark.parametrize("defect,reason", [
    ("unavailable", "FORCED_TRIGGER_RAY_HAS_RAY_FALSE"),
    ("error", "FORCED_TRIGGER_RAY_STATUS_ERROR"),
    ("missing", "FORCED_TRIGGER_RAY_VECTOR_MISSING"),
    ("malformed", "FORCED_TRIGGER_RAY_VECTOR_MALFORMED"),
    ("zero", "FORCED_TRIGGER_RAY_ZERO"),
    ("dimension", "FORCED_TRIGGER_RAY_BAD_DIMENSION"),
    ("nonfinite", "FORCED_TRIGGER_RAY_NONFINITE"),
    ("provenance", "FORCED_TRIGGER_RAY_PROVENANCE_AMBIGUOUS"),
])
def test_forced_request_failure_is_explicit_and_uses_existing_dedicated_fallback(monkeypatch, defect, reason):
    lp, real, instances = triangle_lp(), highspy.Highs, []
    class Engine:
        def __init__(self):
            self.inner = real()
            self.value_requests = 0
            instances.append(self)
        def __getattr__(self, name):
            return getattr(self.inner, name)
        def getDualRayExist(self):
            return highspy.HighsStatus.kOk, False
        def getDualRay(self):
            self.value_requests += 1
            if defect == "unavailable":
                return highspy.HighsStatus.kWarning, False, np.zeros(3)
            if defect == "error":
                return highspy.HighsStatus.kError, True, -np.ones(3)
            if defect == "missing":
                return highspy.HighsStatus.kWarning, True, None
            if defect == "malformed":
                return highspy.HighsStatus.kWarning, True, ["not a number"]
            if defect == "zero":
                return highspy.HighsStatus.kWarning, True, np.zeros(3)
            if defect == "dimension":
                return highspy.HighsStatus.kWarning, True, -np.ones(2)
            if defect == "nonfinite":
                return highspy.HighsStatus.kWarning, True, np.full(3, np.nan)
            assert defect == "provenance"
            self.inner.changeRowBounds(0, -highspy.kHighsInf, 7.)
            return highspy.HighsStatus.kWarning, True, -np.ones(3)
    monkeypatch.setattr(highspy, "Highs", Engine)
    result = trigger(lp, "IPM_FEASIBILITY")
    assert len(instances) == 1 and instances[0].value_requests == 1
    assert result["trigger_ray_forced_request_attempted"]
    assert not result["trigger_ray_cached_before_request"]
    assert result["trigger_ray_forced_request_status"]
    assert result["trigger_ray_returned_vector_present"] is (defect != "missing")
    if defect not in ("missing", "malformed"):
        assert result["trigger_ray_raw_dimension"] == (2 if defect == "dimension" else 3)
        assert result["trigger_ray_returned_vector_finite"] is (defect != "nonfinite")
        assert result["trigger_ray_returned_vector_nonzero"] is (defect not in ("unavailable", "zero"))
    elif defect == "malformed":
        assert result["trigger_ray_raw_dimension"] == 1
        assert result["trigger_ray_returned_vector_finite"] is None
    else:
        assert result["trigger_ray_raw_dimension"] is None
    assert result["trigger_ray_rejected_reason"] == reason
    assert result["original_row_dual_ray"] is None
    assert not result["trigger_ray_original_space_mapping_verified"]
    calls = []
    def dedicated(model, seconds):
        calls.append(model)
        return fake_dedicated_solver(model, infeasible=False)
    report = O.attempt_fixed_phase_certificate(lp, "IPM_FEASIBILITY", 2., trigger_proposal=result,
        certificate_solver=dedicated)
    assert calls == [lp] and report["dedicated_certificate_fallback_used"]
    assert report["trigger_ray_rejected_reason"] == reason
    assert report["trigger_ray_forced_request_attempted"]
    assert report["trigger_ray_raw_dimension"] == result["trigger_ray_raw_dimension"]
    assert report["trigger_ray_returned_vector_present"] == result["trigger_ray_returned_vector_present"]
    assert report["trigger_ray_returned_vector_finite"] == result["trigger_ray_returned_vector_finite"]
    assert not report["fixed_phase_farkas_verified"] and not report["exact_farkas_replay_verified"]


def test_uncertain_existence_query_still_requests_values_for_infeasible_trigger(monkeypatch):
    lp, real, calls = triangle_lp(), highspy.Highs, []
    class Engine:
        def __init__(self):
            self.inner = real()
        def __getattr__(self, name):
            return getattr(self.inner, name)
        def getDualRayExist(self):
            return highspy.HighsStatus.kWarning, False
        def getDualRay(self):
            calls.append(self)
            return highspy.HighsStatus.kOk, True, -np.ones(3)
    monkeypatch.setattr(highspy, "Highs", Engine)
    result = trigger(lp)
    assert len(calls) == 1 and result["trigger_ray_forced_request_attempted"]
    assert result["trigger_ray_original_space_mapping_verified"]
    report = O.attempt_fixed_phase_certificate(lp, "SIMPLEX_FEASIBILITY", 2., trigger_proposal=result,
        certificate_solver=forbidden_dedicated, factor_cache={})
    assert report["exact_farkas_replay_verified"]


def test_forced_ray_without_successful_full_replay_cannot_authorize_phase_infeasibility(monkeypatch):
    p, lp, pattern, cert = infeasible_fixed_phase_fixture()
    real = highspy.Highs
    class Engine:
        def __init__(self):
            self.inner = real()
        def __getattr__(self, name):
            return getattr(self.inner, name)
        def getDualRayExist(self):
            return highspy.HighsStatus.kOk, False
        def getDualRay(self):
            _status, has_ray, values = self.inner.getDualRay()
            return highspy.HighsStatus.kWarning, has_ray, values
    monkeypatch.setattr(highspy, "Highs", Engine)
    candidate = trigger(lp, "PRIMARY")
    assert candidate["trigger_ray_forced_request_attempted"]
    assert candidate["trigger_ray_forced_request_status"] == str(highspy.HighsStatus.kWarning)
    assert candidate["trigger_ray_original_space_mapping_verified"]
    cert["multipliers"][0]["multiplier"] = "-1/1"
    monkeypatch.setattr(O, "repair_fixed_phase_dual_ray", lambda *args, **kwargs: (
        cert, [{"exact_farkas_replay_verified": True}], "EXACT_FARKAS_VERIFIED"))
    _, report = O.search_fixed_phase_exact_witness(p, lp, pattern, lambda *_: candidate, 3.,
        certificate_solver=lambda model, _: fake_dedicated_solver(model, infeasible=False))
    assert report["trigger_ray_forced_request_attempted"]
    assert report["full_original_lp_replay_started"]
    assert "negative" in report["exact_certificate_rejection"]
    assert report["search_status"] == "INCONCLUSIVE"
    assert not report["fixed_phase_farkas_verified"] and not report["exact_farkas_replay_verified"]
    assert not report["permits_node_exclusion"] and not report["permits_infeasibility_claim"]


@pytest.mark.parametrize("method", ["PRIMARY", "IPM_FEASIBILITY", "SIMPLEX_FEASIBILITY"])
@pytest.mark.parametrize("cached", [False, True])
def test_warning_true_valid_vector_enters_exact_repair_and_requires_full_original_replay(monkeypatch, tmp_path, method, cached):
    lp, real, instances, repairs = triangle_lp(), highspy.Highs, [], []
    class Engine:
        def __init__(self):
            self.inner = real()
            self.requests = 0
            instances.append(self)
        def __getattr__(self, name):
            return getattr(self.inner, name)
        def getDualRayExist(self):
            return highspy.HighsStatus.kOk, cached
        def getDualRay(self):
            self.requests += 1
            return highspy.HighsStatus.kWarning, True, -np.ones(len(lp.rows))
    monkeypatch.setattr(highspy, "Highs", Engine)
    result = O.solve_highspy(lp, tmp_path / "warning.log", proposal_only=True,
        proposal_method=method, time_limit_seconds=1.)
    assert len(instances) == 1 and instances[0].requests == 1
    assert result["ray_call_status"] == str(highspy.HighsStatus.kWarning)
    assert result["trigger_ray_forced_request_attempted"] is (not cached)
    if not cached:
        assert result["trigger_ray_forced_request_status"] == str(highspy.HighsStatus.kWarning)
        assert result["trigger_ray_forced_request_has_ray"] is True
    assert result["trigger_ray_returned_vector_present"] is True
    assert result["trigger_ray_raw_dimension"] == len(lp.rows)
    assert result["trigger_ray_returned_vector_finite"] is True
    assert result["trigger_ray_returned_vector_nonzero"] is True
    assert result["trigger_ray_returned_vector_support_size"] == len(lp.rows)
    assert result["trigger_ray_original_space_mapping_verified"]
    assert result["row_ray_mapping"]["mapping_count"] == 1
    np.testing.assert_array_equal(result["raw_dual_ray"], -np.ones(len(lp.rows)))
    assert result["trigger_ray_rejected_reason"] is None
    # This is still only an untrusted numerical proposal, not a certificate.
    assert not result["infeasible"] and "exact_farkas_replay_verified" not in result
    import json
    stored = json.loads((tmp_path / "warning.proposal.json").read_text())
    for key in ("trigger_ray_returned_vector_present", "trigger_ray_raw_dimension",
                "trigger_ray_returned_vector_finite", "trigger_ray_returned_vector_nonzero",
                "trigger_ray_returned_vector_support_size"):
        assert stored[key] == result[key]
    original_repair, original_replay, replays = O.repair_fixed_phase_dual_ray, O.verify_exact_lp_farkas, []
    def repair(model, ray, seconds, **kwargs):
        assert model is lp
        np.testing.assert_array_equal(ray, result["original_row_dual_ray"])
        repairs.append(ray)
        return original_repair(model, ray, seconds, **kwargs)
    def replay(model, certificate):
        assert model is lp and certificate["canonical_lp_sha256"] == lp.identity()
        replays.append(certificate)
        return original_replay(model, certificate)
    monkeypatch.setattr(O, "repair_fixed_phase_dual_ray", repair)
    monkeypatch.setattr(O, "verify_exact_lp_farkas", replay)
    report = O.attempt_fixed_phase_certificate(lp, method, 2., trigger_proposal=result,
        certificate_solver=forbidden_dedicated, factor_cache={})
    assert len(repairs) == len(replays) == 1
    assert report["direct_ray_repair_status"] == "EXACT_FARKAS_VERIFIED"
    assert report["exact_farkas_replay_verified"] and report["full_original_lp_replay_completed"]
    assert report["trigger_ray_raw_dimension"] == len(lp.rows)
    assert report["trigger_ray_returned_vector_finite"] is True
    assert report["certificate_ray_source"] == "TRIGGER_" + method
    assert not report["dedicated_certificate_fallback_used"] and not report["permits_node_exclusion"]
