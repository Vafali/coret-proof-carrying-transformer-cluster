"""CPU-only certificate fixtures; no captured scientific inputs are loaded."""
from dataclasses import replace
from fractions import Fraction as F
import json
import time

import numpy as np
import pytest

from test_fixed_phase_layernorm_witness_v1 import O, fixture, infeasible_fixed_phase_fixture


@pytest.mark.parametrize("method", ["PRIMARY", "IPM_FEASIBILITY", "SIMPLEX_FEASIBILITY"])
def test_each_infeasible_method_triggers_dedicated_original_lp_certificate(method, monkeypatch):
    p, lp, pattern, _ = infeasible_fixed_phase_fixture()
    calls = []
    def certificate_solve(model, seconds):
        assert model is lp and model.identity() == lp.identity() and seconds <= 30
        calls.append(model)
        return O.solve_highspy(model, proposal_only=True,
            proposal_method="FIXED_PHASE_CERTIFICATE", time_limit_seconds=1.)
    def primary(*_):
        return {"model_status": "Infeasible" if method == "PRIMARY" else "Unknown", "column_values": None}
    def fallback(model, seconds, current, role):
        return {"solver_method": current, "model_status": "Infeasible" if current == method else "Unknown",
                "column_values": None}
    monkeypatch.setattr(O, "FixedPhaseAnchorWorkspace", lambda *_: pytest.fail("no anchor on infeasible state"))
    _, report = O.search_fixed_phase_exact_witness(p, lp, pattern, primary, 3.,
        feasibility_propose=fallback, certificate_solver=certificate_solve)
    assert len(calls) == 1
    assert report["numerical_infeasible_methods"] == [method]
    assert report["certificate_trigger_solver_method"] == method
    assert report["fixed_phase_farkas_verified"] and report["exact_farkas_replay_verified"]
    assert F(report["exact_farkas_rhs"]) < 0
    assert report["search_status"] == "FIXED_PHASE_LINEARLY_INFEASIBLE"
    assert not report["permits_node_exclusion"] and not report["permits_infeasibility_claim"]
    assert O.verify_exact_lp_farkas(lp, report["fixed_phase_farkas_certificate"])["verified"]


def triangle_lp(scale=F(1)):
    return O.ExactCanonicalLP(("x", "y"), (None, None), (None, None), tuple(
        O.ExactLPRow(f"triangle[{i}]", (0, 1), tuple(scale * F(x) for x in row), None, -scale)
        for i, row in enumerate(((1, 2), (-3, 1), (2, -3)))))


def test_reduced_integer_matrix_cache_hits_and_full_original_replay_remains_required():
    lp = triangle_lp(F(1, 2**47))
    cache = {}
    certificate, attempts, status = O.repair_fixed_phase_dual_ray(lp, np.ones(3), 2., factor_cache=cache)
    assert status == "EXACT_FARKAS_VERIFIED"
    assert attempts[-1]["exact_repair_matrix_dimension"] == 3
    assert attempts[-1]["numerical_support_rank"] == 3
    assert not attempts[-1]["normalized_matrix_cache_hit"]
    first = attempts[-1]
    assert first["integer_coefficient_max_bits"] < first["coefficient_denominator_max_bits"]
    second, attempts, _ = O.repair_fixed_phase_dual_ray(lp, np.ones(3), 2., factor_cache=cache)
    assert attempts[-1]["normalized_matrix_cache_hit"]
    assert attempts[-1]["exact_integer_solve"]["elimination_count"] == 0
    assert certificate == second
    assert O.verify_exact_lp_farkas(lp, second)["verified"]


def test_cache_reused_across_different_original_lps_but_rhs_and_identity_replayed():
    lp = triangle_lp()
    altered = replace(lp, rows=tuple(replace(row, upper=F(-2)) for row in lp.rows))
    cache = {}
    original, _, _ = O.repair_fixed_phase_dual_ray(lp, np.ones(3), 2., factor_cache=cache)
    certificate, attempts, _ = O.repair_fixed_phase_dual_ray(altered, np.ones(3), 2., factor_cache=cache)
    assert attempts[-1]["normalized_matrix_cache_hit"]
    assert original["canonical_lp_sha256"] != certificate["canonical_lp_sha256"]
    assert F(O.verify_exact_lp_farkas(altered, certificate)["exact_lambda_b"]) == -2
    with pytest.raises(RuntimeError, match="identity"):
        O.verify_exact_lp_farkas(lp, certificate)


def test_corrupted_cached_integer_factor_never_authorizes_a_certificate():
    lp, cache = triangle_lp(), {}
    O.repair_fixed_phase_dual_ray(lp, np.ones(3), 2., factor_cache=cache)
    for factor in cache.values():
        factor["upper"][0][0] += 1
    certificate, attempts, _ = O.repair_fixed_phase_dual_ray(lp, np.ones(3), 2., factor_cache=cache)
    assert certificate is None
    assert any("factorization identity differs" in row.get("failure", "") for row in attempts)
    assert not any(row["exact_farkas_replay_verified"] for row in attempts)


@pytest.mark.parametrize("upper", [None, F(2)])
def test_original_variable_bound_inequalities_are_part_of_full_certificate(upper):
    lp = O.ExactCanonicalLP(("x",), (F(1),), (upper,),
        (O.ExactLPRow("x_le_zero", (0,), (F(1),), None, F(0)),))
    certificate, attempts, _ = O.repair_fixed_phase_dual_ray(lp, np.ones(1), 2.)
    assert certificate is not None and attempts[-1]["support_selection_includes_original_bounds"]
    assert any(entry["kind"] == "column" for entry in certificate["multipliers"])
    assert O.verify_exact_lp_farkas(lp, certificate)["exact_lambda_b"] == "-1/1"


def test_large_sparse_cycle_ray_avoids_naive_512_square_bareiss_and_replays_every_original_coordinate(monkeypatch):
    cycle, extra = 512, 2048
    rows = []
    for i in range(cycle):
        j = (i + 1) % cycle
        entries = sorted(((i, F(1)), (j, F(-1))))
        rows.append(O.ExactLPRow(f"cycle[{i}]", tuple(k for k, _ in entries),
            tuple(v for _, v in entries), None, F(-1, cycle)))
    rows.extend(O.ExactLPRow(f"irrelevant[{i}]", (cycle + i,), (F(1),), None, F(1))
                for i in range(extra))
    lp = O.ExactCanonicalLP(tuple(f"x{i}" for i in range(cycle + extra)),
        (None,) * cycle + (F(-1),) * extra, (None,) * cycle + (F(1),) * extra, tuple(rows))
    started = time.perf_counter()
    solver = O.solve_highspy(lp, proposal_only=True, proposal_method="FIXED_PHASE_CERTIFICATE", time_limit_seconds=2.)
    assert solver["model_status"] == "Infeasible"
    assert solver["options"]["presolve"] == "off"
    ray = solver["original_row_dual_ray"].copy()
    ray[cycle:] = 1e-14  # Candidate noise is not proof and must be dropped.
    monkeypatch.setattr(O, "_bareiss_solve", lambda *_: pytest.fail("legacy dense repair must not run"))
    monkeypatch.setattr(O, "_exact_multi_rhs_solve", lambda *_args, **_kwargs: pytest.fail("sparse chain needs no dense elimination"))
    certificate, attempts, status = O.repair_fixed_phase_dual_ray(lp, ray, 3.)
    assert status == "EXACT_FARKAS_VERIFIED" and certificate is not None
    best = attempts[-1]
    assert best["raw_ray_support_size"] == cycle + extra
    assert best["truncated_candidate_support_size"] == cycle
    assert best["selected_independent_support_size"] == cycle
    assert best["sparse_links_eliminated"] == cycle - 1
    assert best["numerical_support_rank"] == cycle
    assert best["exact_repair_matrix_dimension"] == 0
    assert F(O.verify_exact_lp_farkas(lp, certificate)["exact_lambda_b"]) < 0
    print(json.dumps({"synthetic_sparse_lp_columns": lp.column_count, "rows": len(rows),
        "raw_support": best["raw_ray_support_size"], "selected_support": best["selected_independent_support_size"],
        "avoided_naive_dimension": cycle, "remaining_exact_dimension": best["exact_repair_matrix_dimension"],
        "seconds": time.perf_counter() - started}), flush=True)


def test_exact_replay_failure_and_repair_timeout_are_inconclusive(monkeypatch):
    p, lp, pattern, _ = infeasible_fixed_phase_fixture()
    def timed_out(*args, **kwargs):
        raise O.ExactSolveFailure("exact repair Bareiss timeout")
    monkeypatch.setattr(O, "repair_fixed_phase_dual_ray", timed_out)
    _, report = O.search_fixed_phase_exact_witness(p, lp, pattern,
        lambda *_: {"model_status": "Infeasible", "column_values": None}, 3.)
    assert report["search_status"] == "INCONCLUSIVE"
    assert not report["fixed_phase_farkas_verified"] and not report["exact_farkas_replay_verified"]
    assert not report["permits_infeasibility_claim"]
    assert report["direct_ray_repair_status"] == "DIRECT_RAY_REPAIR_TIMEOUT"
    assert report["exact_repair_timeout_stage"] == "reduced_support_exact_repair"


def test_dedicated_certificate_rejects_original_lp_substitution():
    _, lp, _, _ = fixture()
    def corrupted(model, seconds):
        result = O.solve_highspy(model, proposal_only=True, proposal_method="FIXED_PHASE_CERTIFICATE", time_limit_seconds=1.)
        result["construction_diagnostic"]["canonical_lp_sha256"] = "different"
        return result
    with pytest.raises(O.FixedPhaseInvariantError, match="original LP identity differs"):
        O.attempt_fixed_phase_certificate(lp, "IPM_FEASIBILITY", 2., certificate_solver=corrupted)


def test_claimed_repair_success_cannot_bypass_boundary_full_original_replay(monkeypatch):
    _, lp, _, cert = infeasible_fixed_phase_fixture()
    cert["multipliers"][0]["multiplier"] = "0/1"
    monkeypatch.setattr(O, "repair_fixed_phase_dual_ray", lambda *args, **kwargs: (
        cert, [{"exact_farkas_replay_verified": True, "exact_farkas_rhs": "-1/1"}], "EXACT_FARKAS_VERIFIED"))
    report = O.attempt_fixed_phase_certificate(lp, "PRIMARY", 2.)
    assert not report["fixed_phase_farkas_verified"] and not report["exact_farkas_replay_verified"]
    assert report["exact_farkas_rhs"] is None and report["exact_certificate_rejection"]


def test_noncontradictory_stationary_ray_and_negative_multiplier_do_not_certify():
    lp = replace(triangle_lp(), rows=tuple(replace(row, upper=F(1)) for row in triangle_lp().rows))
    certificate, attempts, _ = O.repair_fixed_phase_dual_ray(lp, np.ones(3), 2.)
    assert certificate is None and not any(row["exact_farkas_replay_verified"] for row in attempts)
    bad = {"schema": "CORET_EXACT_LP_FARKAS_CERTIFICATE_V1", "canonical_lp_sha256": lp.identity(),
           "multipliers": [{"kind": "row", "index": 0, "orientation": 1, "multiplier": "-1/1"}]}
    with pytest.raises(RuntimeError, match="negative"):
        O.verify_exact_lp_farkas(lp, bad)


def fake_dedicated_solver(lp, *, infeasible=True):
    """Only solver plumbing is mocked; exact Farkas replay stays authoritative."""
    return {"construction_diagnostic": {"canonical_lp_sha256": lp.identity(),
                "solver_scaled_lp_sha256": "test_solver_copy"},
            "solver_method": "FIXED_PHASE_CERTIFICATE",
            "rational_objective_sha256": O._sha_json([O._fs(F(0))] * lp.column_count),
            "options": {"solver": "simplex", "presolve": "off"},
            "solver_scaling": {"original_canonical_lp_sha256": lp.identity(),
                "solver_scaled_lp_sha256": "test_solver_copy"},
            "run_status": "OK", "model_status": "Infeasible" if infeasible else "Optimal",
            "original_row_dual_ray": np.ones(len(lp.rows)) if infeasible else None}


def fake_certificate_clock(monkeypatch):
    """Deterministically exercise 90/105-second boundaries without sleeping."""
    clock = {"now": 0., "timer_deadline": None, "restore_credits": [], "timers": []}
    monkeypatch.setattr(O.time, "perf_counter", lambda: clock["now"])
    def set_timer(_which, seconds, *_):
        clock["timers"].append(seconds)
        clock["timer_deadline"] = clock["now"] + seconds if seconds else None
        return (0., 0.)
    monkeypatch.setattr(O.signal, "setitimer", set_timer)
    def alarm(seconds):
        set_timer(None, seconds)
        def restore(completion_extension_seconds=0.):
            clock["restore_credits"].append(completion_extension_seconds)
        return restore
    monkeypatch.setattr(O, "_start_reconstruction_alarm", alarm)
    return clock


@pytest.mark.parametrize("outcome", ["valid", "negative_multiplier", "timeout"])
def test_near_deadline_candidate_gets_only_bounded_full_replay_and_proof_gates_acceptance(monkeypatch, outcome):
    lp = triangle_lp()
    clock = fake_certificate_clock(monkeypatch)
    original_verify = O.verify_exact_lp_farkas
    replay_calls = []
    def verify(model, certificate):
        replay_calls.append(certificate)
        clock["now"] += 2. if outcome != "timeout" else 16.
        if clock["now"] >= clock["timer_deadline"]:
            raise O.ExactSolveFailure("reconstruction family deadline")
        return original_verify(model, certificate)
    monkeypatch.setattr(O, "verify_exact_lp_farkas", verify)
    def selected_solution(model, support, deadline, cache, audit):
        assert deadline == 90.
        clock["now"] = 89.9
        audit.update(selected_system_replay_verified=True, candidate_constructed_before_deadline=True)
        return [F(-1, 3) if outcome == "negative_multiplier" else F(1, 3)] * len(support)
    monkeypatch.setattr(O, "_reduced_support_vertex", selected_solution)
    report = O.attempt_fixed_phase_certificate(lp, "PRIMARY",
        certificate_solver=lambda model, _: fake_dedicated_solver(model), factor_cache={})
    assert report["certificate_family_budget_seconds"] == 90.
    assert report["candidate_constructed_before_deadline"] and report["selected_system_replay_verified"]
    assert report["full_original_lp_replay_started"] and report["final_replay_extension_used"]
    assert clock["timers"] == pytest.approx([90., 15.1])
    assert clock["restore_credits"] == [15.]
    assert len(replay_calls) == 1  # No repeated expensive boundary replay.
    assert report["full_original_lp_replay_completed"] is (outcome != "timeout")
    assert report["fixed_phase_farkas_verified"] is (outcome == "valid")
    assert report["exact_farkas_replay_verified"] is (outcome == "valid")
    assert not report["permits_node_exclusion"]
    if outcome == "valid":
        assert original_verify(lp, report["fixed_phase_farkas_certificate"])["verified"]
        assert F(report["exact_farkas_rhs"]) < 0
        assert report["final_replay_seconds"] == pytest.approx(2.)
    else:
        assert report["exact_farkas_rhs"] is None
        assert report["direct_ray_repair_status"] == "DIRECT_RAY_REPAIR_TIMEOUT"


@pytest.mark.parametrize("selected,before", [(False, True), (True, False), (False, False)])
def test_completion_extension_requires_both_eligibility_invariants(monkeypatch, selected, before):
    lp = triangle_lp()
    clock = fake_certificate_clock(monkeypatch)
    def selected_solution(model, support, deadline, cache, audit):
        clock["now"] = 89.9
        audit.update(selected_system_replay_verified=selected, candidate_constructed_before_deadline=before)
        return [F(1, 3)] * len(support)
    monkeypatch.setattr(O, "_reduced_support_vertex", selected_solution)
    def verify(*_):
        clock["now"] = 90.01
        assert clock["timer_deadline"] == 90.
        raise O.ExactSolveFailure("reconstruction family deadline")
    monkeypatch.setattr(O, "verify_exact_lp_farkas", verify)
    result = O.attempt_fixed_phase_certificate(lp, "PRIMARY",
        certificate_solver=lambda model, _: fake_dedicated_solver(model), factor_cache={})
    assert not result["fixed_phase_farkas_verified"]
    assert not result["final_replay_extension_used"]
    assert clock["restore_credits"] == [0.]


def test_family_timeout_before_exact_candidate_never_authorizes_infeasibility(monkeypatch):
    lp = triangle_lp()
    clock = fake_certificate_clock(monkeypatch)
    def timeout(*_):
        clock["now"] = 90.
        raise O.ExactSolveFailure("exact repair Bareiss timeout")
    monkeypatch.setattr(O, "_reduced_support_vertex", timeout)
    result = O.attempt_fixed_phase_certificate(lp, "PRIMARY",
        certificate_solver=lambda model, _: fake_dedicated_solver(model), factor_cache={})
    assert not result["fixed_phase_farkas_verified"] and not result["exact_farkas_replay_verified"]
    assert not result["candidate_constructed_before_deadline"]
    assert not result["full_original_lp_replay_started"]
    assert clock["restore_credits"] == [0.]


def test_duplicate_fallback_triggers_reuse_one_dedicated_attempt_and_90_second_budget(monkeypatch):
    p, lp, pattern, _ = fixture()
    calls, cache = [], {}
    monkeypatch.setattr(O, "FixedPhaseAnchorWorkspace", lambda *_: pytest.fail("no admissible anchor"))
    def dedicated(model, seconds):
        calls.append((model.identity(), seconds))
        return fake_dedicated_solver(model, infeasible=False)
    def fallback(model, seconds, method, role):
        return {"solver_method": method, "model_status": "Infeasible", "column_values": None}
    _, report = O.search_fixed_phase_exact_witness(p, lp, pattern,
        lambda *_: {"model_status": "Unknown", "column_values": None}, 120.,
        feasibility_propose=fallback, certificate_solver=dedicated, certificate_attempt_cache=cache)
    assert len(calls) == 1 and len(report["certificate_attempts"]) == 2
    assert [a["certificate_attempt_cache_hit"] for a in report["certificate_attempts"]] == [False, True]
    assert [a["certificate_family_budget_seconds"] for a in report["certificate_attempts"]] == [90., 90.]
    assert report["numerical_infeasible_methods"] == ["IPM_FEASIBILITY", "SIMPLEX_FEASIBILITY"]
    assert report["search_status"] == "INCONCLUSIVE" and not report["fixed_phase_farkas_verified"]
    # The oracle shares this cache across searches of the same canonical phase.
    _, again = O.search_fixed_phase_exact_witness(p, lp, pattern,
        lambda *_: {"model_status": "Infeasible", "column_values": None}, 120.,
        certificate_solver=dedicated, certificate_attempt_cache=cache)
    assert len(calls) == 1 and again["certificate_attempts"][0]["certificate_attempt_cache_hit"]


def test_verified_attempt_cache_replays_original_certificate_without_new_solver(monkeypatch):
    lp, cache = triangle_lp(), {}
    calls = []
    def dedicated(model, seconds):
        calls.append(model.identity())
        return fake_dedicated_solver(model)
    first = O.attempt_fixed_phase_certificate(lp, "IPM_FEASIBILITY", 2.,
        certificate_solver=dedicated, attempt_cache=cache, factor_cache={})
    assert first["fixed_phase_farkas_verified"]
    replays = []
    original = O.verify_exact_lp_farkas
    def replay(model, certificate):
        replays.append(certificate)
        return original(model, certificate)
    monkeypatch.setattr(O, "verify_exact_lp_farkas", replay)
    again = O.attempt_fixed_phase_certificate(lp, "SIMPLEX_FEASIBILITY", 2.,
        certificate_solver=dedicated, attempt_cache=cache)
    assert len(calls) == 1 and len(replays) == 1
    assert again["certificate_attempt_cache_hit"] and again["fixed_phase_farkas_verified"]
    assert again["full_original_lp_replay_completed"]
    assert again["certificate_trigger_solver_method"] == "SIMPLEX_FEASIBILITY"
    assert first["certificate_trigger_solver_method"] == "IPM_FEASIBILITY"


@pytest.mark.parametrize("corruption", ["audit_checksum", "certificate"])
def test_attempt_cache_cannot_authorize_corrupt_or_unreplayed_certificate(monkeypatch, corruption):
    lp, cache = triangle_lp(), {}
    first = O.attempt_fixed_phase_certificate(lp, "PRIMARY", 2.,
        certificate_solver=lambda model, _: fake_dedicated_solver(model), attempt_cache=cache, factor_cache={})
    assert first["fixed_phase_farkas_verified"]
    entry = cache[lp.identity()]
    entry["audit"]["fixed_phase_farkas_certificate"]["multipliers"][0]["multiplier"] = "-1/1"
    if corruption == "certificate":
        entry["audit_sha256"] = O._sha_json(entry["audit"])
    def no_solve(*_):
        pytest.fail("dedup must not repeat dedicated solve")
    if corruption == "audit_checksum":
        with pytest.raises(O.FixedPhaseInvariantError, match="cache identity/integrity"):
            O.attempt_fixed_phase_certificate(lp, "IPM_FEASIBILITY", 2.,
                certificate_solver=no_solve, attempt_cache=cache)
    else:
        result = O.attempt_fixed_phase_certificate(lp, "IPM_FEASIBILITY", 2.,
            certificate_solver=no_solve, attempt_cache=cache)
        assert result["certificate_attempt_cache_hit"] and not result["fixed_phase_farkas_verified"]
        assert not result["exact_farkas_replay_verified"]
        assert "negative" in result["exact_certificate_rejection"]


def test_attempt_cache_key_is_full_original_canonical_lp_identity():
    lp, cache, calls = triangle_lp(), {}, []
    changed = replace(lp, rows=tuple(replace(row, upper=F(2)) for row in lp.rows))
    def dedicated(model, seconds):
        calls.append(model.identity())
        return fake_dedicated_solver(model, infeasible=False)
    for model in (lp, changed):
        result = O.attempt_fixed_phase_certificate(model, "PRIMARY", 2.,
            certificate_solver=dedicated, attempt_cache=cache)
        assert not result["certificate_attempt_cache_hit"] and not result["fixed_phase_farkas_verified"]
    assert len(calls) == len(cache) == 2


def test_nested_alarm_restoration_preserves_only_bounded_completion_credit(monkeypatch):
    clock, calls = {"now": 0.}, []
    monkeypatch.setattr(O.time, "perf_counter", lambda: clock["now"])
    monkeypatch.setattr(O.signal, "getsignal", lambda *_: "previous_handler")
    monkeypatch.setattr(O.signal, "signal", lambda *_: None)
    def timer(_which, seconds, interval=0.):
        calls.append((seconds, interval))
        return (90., 0.)
    monkeypatch.setattr(O.signal, "setitimer", timer)
    restore = O._start_reconstruction_alarm(90.)
    clock["now"] = 92.
    restore(completion_extension_seconds=15.)
    assert calls[-1] == (13., 0.)
    with pytest.raises(O.FixedPhaseInvariantError, match="completion extension"):
        restore(completion_extension_seconds=15.01)
