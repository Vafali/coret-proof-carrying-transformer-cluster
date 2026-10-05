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
    if method == "IPM_FEASIBILITY" and not exposed:
        assert instances[0].get_ray_calls == 0
        assert result["original_row_dual_ray"] is None
    else:
        assert instances[0].get_ray_calls == 1
        assert result["trigger_ray_original_space_mapping_verified"]
        report = O.attempt_fixed_phase_certificate(lp, method, 2., trigger_proposal=result,
            certificate_solver=forbidden_dedicated, factor_cache={})
        assert report["exact_farkas_replay_verified"]


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
