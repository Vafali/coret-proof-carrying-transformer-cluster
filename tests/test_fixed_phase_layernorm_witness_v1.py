from dataclasses import replace
from fractions import Fraction as F
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/analyze_block2_exact_layernorm_perspective_causal_v1.py"
SPEC = importlib.util.spec_from_file_location("fixed_phase_ln_test_oracle", PATH)
O = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = O
SPEC.loader.exec_module(O)


def fixture(epsilon=F(1), cancellation_anywhere=False, sources=1):
    zero = (F(0), F(0))
    identity = ((F(1), F(0)), (F(0), F(1)))
    negative = ((F(-1), F(0)), (F(0), F(-1)))
    p = O.ExactPerspectiveProblem(
        x0=zero, X=tuple((F(i + 1), F(-i - 1)) for i in range(sources)),
        low=(F(-4),) * sources, high=(F(4),) * sources,
        gamma=(F(1), F(1)), beta=(F(2), F(2)) if cancellation_anywhere else zero,
        epsilon=epsilon, W1=identity if cancellation_anywhere else (zero, zero),
        b1=zero, W2=negative if cancellation_anywhere else (zero, zero), b2=zero)
    pattern = [cancellation_anywhere] * 2
    lp, audit = O.build_fixed_phase_linear_lp(p, pattern)
    return p, lp, pattern, audit


def point(p, pattern, sources, t):
    return O._fixed_phase_point_from_sources(p, pattern, list(map(F, sources)), F(t))


def unusable_feasibility_proposal(*_):
    return {"model_status": "Unknown", "column_values": None}


def test_unknown_unusable_primary_uses_ipm_then_exact_original_replay(monkeypatch):
    p, lp, pattern, _ = fixture()
    identity = lp.identity()
    calls, replayed = [], []
    replay = O.replay_exact_linear_point
    def counted_replay(model, candidate):
        assert model is lp and model.identity() == identity
        replayed.append(model)
        return replay(model, candidate)
    monkeypatch.setattr(O, "replay_exact_linear_point", counted_replay)
    def fallback(model, seconds, method, role):
        assert model is lp and model.identity() == identity and 0 < seconds <= 30
        calls.append((method, role))
        return {"model_status": "Unknown", "run_status": "HighsStatus.kWarning",
                "solver_method": method, "options": {"presolve": "on"},
                "column_values": np.asarray(point(p, pattern, [0], 1), dtype=float),
                "construction_diagnostic": {"canonical_lp_sha256": identity}}
    found, report = O.search_fixed_phase_exact_witness(p, lp, pattern,
        unusable_feasibility_proposal, 3., feasibility_propose=fallback)
    assert calls == [("IPM_FEASIBILITY", "anchor")] and replayed
    assert found is not None and report["exact_anchor_replay_verified"]
    assert report["authenticated_anchor_solver_method"] == "IPM_FEASIBILITY"
    assert report["fallback_attempts"] == report["fallback_usable_primals"] == 1
    assert report["exact_anchor_reconstruction_attempts"] == report["exact_anchors_verified"] == 1
    attempts = report["proposal_attempts"]
    assert not attempts[0]["admitted_to_exact_reconstruction"]
    assert attempts[1]["admitted_to_exact_reconstruction"]
    assert attempts[1]["within_original_bounds"] and attempts[1]["within_original_rows"]
    assert attempts[1]["max_bound_violation"] == attempts[1]["max_row_violation"] == 0
    assert {a["canonical_lp_sha256"] for a in attempts} == {identity}
    assert lp.identity() == identity and not report["permits_infeasibility_claim"]


@pytest.mark.parametrize("defect", ["bounds", "rows", "nonfinite", "dimensions"])
def test_invalid_ipm_is_discarded_before_repair_and_simplex_is_next(monkeypatch, defect):
    p, lp, pattern, _ = fixture()
    good = np.asarray(point(p, pattern, [0], 1), dtype=float)
    bad = good.copy()
    if defect == "bounds":
        bad[0] = 5
    elif defect == "rows":
        bad[1] = .01
    elif defect == "nonfinite":
        bad[0] = np.nan
    else:
        bad = bad[:-1]
    calls, repaired = [], []
    reconstruct = O.FixedPhaseAnchorWorkspace.reconstruct_anchor
    def checked_reconstruct(workspace, candidate, seconds):
        assert np.array_equal(candidate, good)
        repaired.append(candidate)
        return reconstruct(workspace, candidate, seconds)
    monkeypatch.setattr(O.FixedPhaseAnchorWorkspace, "reconstruct_anchor", checked_reconstruct)
    def fallback(model, seconds, method, role):
        assert model is lp
        calls.append(method)
        return {"model_status": "Optimal", "solver_method": method,
                "column_values": bad if method == "IPM_FEASIBILITY" else good}
    found, report = O.search_fixed_phase_exact_witness(p, lp, pattern,
        unusable_feasibility_proposal, 3., feasibility_propose=fallback)
    assert calls == ["IPM_FEASIBILITY", "SIMPLEX_FEASIBILITY"]
    assert len(repaired) == 1 and found is not None
    assert report["fallback_attempts"] == 2 and report["fallback_usable_primals"] == 1
    assert not report["proposal_attempts"][1]["admitted_to_exact_reconstruction"]
    assert report["proposal_attempts"][2]["admitted_to_exact_reconstruction"]


def test_fallback_exact_replay_rejection_remains_inconclusive(monkeypatch):
    p, lp, pattern, _ = fixture()
    def reject(*_):
        raise RuntimeError("fallback complete original LP replay rejected")
    monkeypatch.setattr(O, "replay_exact_linear_point", reject)
    def fallback(model, seconds, method, role):
        return {"model_status": "Optimal", "solver_method": method,
                "column_values": np.asarray(point(p, pattern, [0], 1), dtype=float)}
    found, report = O.search_fixed_phase_exact_witness(p, lp, pattern,
        unusable_feasibility_proposal, 3., feasibility_propose=fallback)
    assert found is None and report["search_status"] == "INCONCLUSIVE"
    assert report["fallback_attempts"] == report["fallback_usable_primals"] == 2
    assert report["exact_anchor_reconstruction_attempts"] == 2
    assert report["exact_anchors_verified"] == 0 and not report["verified"]
    assert report["first_exact_replay_failure"] == "fallback complete original LP replay rejected"
    assert not report["permits_infeasibility_claim"]


def test_infeasible_fallback_has_no_farkas_or_exclusion_authority(monkeypatch):
    p, lp, pattern, _ = fixture()
    monkeypatch.setattr(O, "attempt_fixed_phase_farkas", lambda *_: pytest.fail("fallback ray forbidden"))
    monkeypatch.setattr(O, "FixedPhaseAnchorWorkspace", lambda *_: pytest.fail("infeasible anchor forbidden"))
    def fallback(model, seconds, method, role):
        return {"model_status": "Infeasible", "solver_method": method,
                "column_values": point(p, pattern, [0], 1),
                "original_row_dual_ray": np.ones(len(lp.rows))}
    found, report = O.search_fixed_phase_exact_witness(p, lp, pattern,
        unusable_feasibility_proposal, 3., feasibility_propose=fallback)
    assert found is None and report["search_status"] == "INCONCLUSIVE"
    assert report["fallback_attempts"] == 2 and report["exact_anchor_reconstruction_attempts"] == 0
    assert not report["fixed_phase_farkas_attempted"] and not report["fixed_phase_farkas_verified"]
    assert not report["permits_infeasibility_claim"]


def test_primary_infeasible_bounded_repair_timeout_is_recorded(monkeypatch):
    p, lp, pattern, _ = fixture()
    def timeout(*_):
        raise O.ExactSolveFailure("exact repair Bareiss timeout")
    monkeypatch.setattr(O, "attempt_fixed_phase_farkas", timeout)
    found, report = O.search_fixed_phase_exact_witness(p, lp, pattern,
        lambda *_: {"model_status": "Infeasible", "column_values": None}, 3.,
        feasibility_propose=lambda *_: pytest.fail("primary Infeasible remains Farkas-only"))
    assert found is None and report["search_status"] == "INCONCLUSIVE"
    assert report["fixed_phase_farkas_attempted"] and not report["fixed_phase_farkas_verified"]
    assert "Bareiss timeout" in report["failure"] and report["fallback_attempts"] == 0


@pytest.mark.parametrize("method,solver", [("IPM_FEASIBILITY", "ipm"),
                                         ("SIMPLEX_FEASIBILITY", "simplex")])
def test_real_feasibility_solver_copy_preserves_original_canonical_lp(method, solver, tmp_path):
    p, lp, pattern, _ = fixture()
    identity = lp.identity()
    result = O.solve_highspy(lp, tmp_path / f"{method}.log", proposal_only=True,
        proposal_method=method, time_limit_seconds=2.)
    assert lp.identity() == result["construction_diagnostic"]["canonical_lp_sha256"] == identity
    assert result["options"]["solver"] == solver and result["options"]["presolve"] == "on"
    assert result["options"]["threads"] == 1 and result["options"]["parallel"] == "off"
    assert result["exact_replay_required"] and not result["feasible"] and not result["infeasible"]
    assert not result["direct_dual_ray_available"] and result["raw_dual_ray"] is None
    screen = O.screen_fixed_phase_anchor_proposal(lp, result)
    assert screen["anchor_reconstruction_skipped_reason"] is None
    anchor, profile = O.FixedPhaseAnchorWorkspace(p, pattern, lp).reconstruct_anchor(result["column_values"], 2.)
    assert O.replay_exact_linear_point(lp, anchor)["verified"]
    assert profile["exact_linear_replay"]["rows_replayed"] == len(lp.rows)


def test_fallback_original_row_not_ignored_by_presolve_proposal(monkeypatch):
    p, lp, pattern, _ = fixture()
    lp = replace(lp, rows=lp.rows + (O.ExactLPRow("original_extra_bound", (3,), (F(1),), F(2), None),))
    monkeypatch.setattr(O, "FixedPhaseAnchorWorkspace", lambda *_: pytest.fail("original row was violated"))
    found, report = O.search_fixed_phase_exact_witness(p, lp, pattern,
        unusable_feasibility_proposal, 3., feasibility_propose=lambda *_: {
            "model_status": "Optimal", "options": {"presolve": "on"},
            "column_values": point(p, pattern, [0], 1)})
    assert found is None and report["exact_anchor_reconstruction_attempts"] == 0
    assert all(a["first_numerical_row_failure"]["name"] == "original_extra_bound"
               for a in report["anchors"][1:])


@pytest.mark.parametrize("method", ["IPM_FEASIBILITY", "SIMPLEX_FEASIBILITY"])
def test_fallback_cannot_be_requested_in_strict_proof_mode(method):
    _, lp, _, _ = fixture()
    with pytest.raises(O.FixedPhaseInvariantError, match="no proof authority"):
        O.solve_highspy(lp, proposal_method=method)
    with pytest.raises(O.FixedPhaseInvariantError, match="zero objective"):
        O.solve_highspy(lp, proposal_method=method, proposal_only=True,
                        objective=[F(1)] * lp.column_count)


def test_fallback_structural_corruption_is_fatal():
    _, lp, _, _ = fixture()
    # Simulate corruption after the canonical constructor's own validation.
    object.__setattr__(lp, "rows", lp.rows + (
        O.ExactLPRow("bad_index", (lp.column_count,), (F(1),), F(0), F(0)),))
    with pytest.raises(O.HighsCanonicalLPDiagnosticError):
        O.solve_highspy(lp, proposal_only=True, proposal_method="IPM_FEASIBILITY", time_limit_seconds=1.)


def test_default_portfolio_executes_real_ipm_then_authenticates_original_lp():
    p, lp, pattern, _ = fixture()
    found, report = O.search_fixed_phase_exact_witness(p, lp, pattern,
        unusable_feasibility_proposal, 3.)
    assert found is not None and report["exact_anchor_replay_verified"]
    assert report["authenticated_anchor_solver_method"] == "IPM_FEASIBILITY"
    assert report["anchor_reconstruction_skipped_reason"] is None
    assert O.replay_fixed_phase_semantic_witness(p, lp, found)["verified"]


def test_fallback_invariant_defect_is_not_heuristic_failure():
    p, lp, pattern, _ = fixture()
    def corrupt_identity(*_):
        return {"model_status": "Optimal", "column_values": point(p, pattern, [0], 1),
                "construction_diagnostic": {"canonical_lp_sha256": "changed"}}
    with pytest.raises(O.FixedPhaseInvariantError, match="canonical LP identity differs"):
        O.search_fixed_phase_exact_witness(p, lp, pattern,
            unusable_feasibility_proposal, 3., feasibility_propose=corrupt_identity)


def line(p, lp, pattern, q0, q1):
    return O.evaluate_fixed_phase_direction(p, lp, pattern, q0,
                                           [b - a for a, b in zip(q0, q1)], q1)


def test_exact_feasible_anchor_accepted_and_every_row_replayed():
    p, lp, pattern, _ = fixture()
    q0 = point(p, pattern, [0], 2)
    workspace = O.FixedPhaseAnchorWorkspace(p, pattern, lp)
    anchor, profile = workspace.reconstruct_anchor(np.asarray(q0, dtype=float), 2.)
    assert anchor == q0
    assert profile["exact_anchor_replay_verified"]
    assert profile["exact_linear_replay"]["rows_replayed"] == len(lp.rows)
    assert profile["exact_linear_replay"]["column_bounds_replayed"] == lp.column_count
    assert profile["equality_nullity"] == 1


def test_anchor_zero_slack_is_allowed():
    p, lp, pattern, _ = fixture()
    q0 = point(p, pattern, [0], 1)
    workspace = O.FixedPhaseAnchorWorkspace(p, pattern, lp)
    anchor, profile = workspace.reconstruct_anchor(np.asarray(q0, dtype=float), 2.)
    assert anchor == q0 and profile["exact_anchor_minimum_slack"] == "0/1"


def test_two_exact_anchors_preserve_full_segment_and_zero_interval():
    p, lp, pattern, _ = fixture(cancellation_anywhere=True)
    q0, q1 = point(p, pattern, [0], 2), point(p, pattern, [2], 2)
    witness, report = line(p, lp, pattern, q0, q1)
    assert report["alpha_interval_contains_zero"]
    assert report["segment_0_1_exactly_feasible"]
    assert report["second_anchor_replay_verified"]
    assert O._interval_contains_exact(report["alpha_interval"], F(1))
    for alpha in (F(0), F(1, 3), F(1, 2), F(1)):
        q = [a + alpha * (b - a) for a, b in zip(q0, q1)]
        assert O.replay_exact_linear_point(lp, q)["verified"]
    assert report["endpoint_residual_sign_change"]
    assert witness is not None
    assert "algebraic_root" in witness
    assert report["exact_replay_result"]["fixed_phase_linear_replay"]["verified"]


def test_singleton_zero_interval_is_supported_not_exclusion():
    p, lp, pattern, _ = fixture(cancellation_anywhere=True)
    lower = list(lp.column_lower)
    lower[3] = F(2)  # n=1, d=2 -> t index 3.
    lp = replace(lp, column_lower=tuple(lower), rows=lp.rows + (
        O.ExactLPRow("scale_upper_guard", (3,), (F(1),), None, F(2)),))
    q0, q1 = point(p, pattern, [0], 2), point(p, pattern, [0], 3)
    witness, report = O.evaluate_fixed_phase_direction(
        p, lp, pattern, q0, [b - a for a, b in zip(q0, q1)])
    assert witness is None
    assert report["alpha_interval_contains_zero"]
    assert report["alpha_interval_lower"] == report["alpha_interval_upper"] == "0/1"
    assert report["alpha_interval_width"] == "0/1"


def test_inconsistent_interval_builder_is_an_implementation_defect(monkeypatch):
    p, lp, pattern, _ = fixture()
    q0 = point(p, pattern, [0], 2)
    monkeypatch.setattr(O, "exact_affine_parameter_interval", lambda _: {
        "alpha_interval_lower": "1/1", "alpha_interval_upper": "2/1",
        "alpha_interval_empty": False})
    with pytest.raises(O.FixedPhaseInvariantError, match="alpha=0"):
        O.evaluate_fixed_phase_direction(p, lp, pattern, q0, [F(0)] * len(q0))


def test_defect_is_not_swallowed_by_search(monkeypatch):
    p, lp, pattern, _ = fixture()
    q0 = point(p, pattern, [0], 2)
    def defect(*_):
        raise O.FixedPhaseInvariantError("authenticated interval broken")
    monkeypatch.setattr(O, "evaluate_fixed_phase_direction", defect)
    propose = lambda *_: {"model_status": "Optimal", "feasible": True, "column_values": q0}
    with pytest.raises(O.FixedPhaseInvariantError):
        O.search_fixed_phase_exact_witness(p, lp, pattern, propose, 2.)


@pytest.mark.parametrize("coefficients,case,count", [
    ((0, 0, 0), "IDENTICALLY_ZERO", None),
    ((0, 0, 1), "NONZERO_CONSTANT", 0),
    ((0, 1, -1), "LINEAR", 1),
    ((1, 0, -1), "QUADRATIC", 2),
    ((1, -2, 1), "QUADRATIC", 1),
    ((1, 0, 1), "QUADRATIC", 0),
])
def test_all_exact_polynomial_cases(coefficients, case, count):
    interval = O.exact_affine_parameter_interval([("box", F(0), F(1), F(-1), F(1))])
    _polynomial, roots, report = O.exact_polynomial_roots_in_interval(coefficients, interval)
    assert report["polynomial_case"] == case and report["roots_total"] == count
    if case == "IDENTICALLY_ZERO":
        assert roots == [("rational", F(0), None)]


@pytest.mark.parametrize("coefficients", [(1, 0, -2), (-1, 0, 2)])
def test_irrational_roots_are_isolated_exactly_without_float(coefficients, monkeypatch):
    monkeypatch.setattr(O, "_isolate_quadratic_root", lambda *_: pytest.fail("float isolation used"))
    interval = O.exact_affine_parameter_interval([("box", F(0), F(1), F(-2), F(2))])
    polynomial, roots, report = O.exact_polynomial_roots_in_interval(coefficients, interval)
    assert len(roots) == report["roots_total"] == 2
    for kind, root, isolation in roots:
        assert kind == "algebraic" and root is None
        assert O.verify_quadratic_isolating_interval(polynomial, isolation)["verified"]


def test_closed_endpoint_linear_root_semantic_replay():
    p, lp, pattern, _ = fixture(cancellation_anywhere=True)
    q0, q1 = point(p, pattern, [1], 2), point(p, pattern, [2], 3)
    witness, report = O.evaluate_fixed_phase_direction(
        p, lp, pattern, q0, [b - a for a, b in zip(q0, q1)])
    assert report["polynomial_case"] == "LINEAR"
    assert report["alpha_interval_lower"] == "-1/1"
    assert witness["t"] == "1/1" and witness["source_values"] == ["0/1"]
    assert report["exact_replay_result"]["verified"]


def test_identically_zero_line_replays_zero_anchor():
    p, lp, pattern, _ = fixture()
    q0 = point(p, pattern, [0], 1)
    witness, report = O.evaluate_fixed_phase_direction(p, lp, pattern, q0, [F(0)] * len(q0))
    assert report["identically_zero"] and witness["t"] == "1/1"
    assert report["exact_replay_result"]["verified"]


def test_fully_fixed_phase_has_no_triangles_and_retains_source_equality():
    p, lp, pattern, _ = fixture()
    rows = lp.rows + (
        O.ExactLPRow("relu_triangle_nonnegative[0]", (6,), (F(-1),), None, F(0)),
        O.ExactLPRow("authenticated_source_equality", (0,), (F(1),), F(0), F(0)),)
    fixed, audit = O.build_fixed_phase_linear_lp(p, pattern, replace(lp, rows=rows))
    assert audit["triangle_rows_removed"] == 1
    assert not any(row.name.startswith("relu_triangle") for row in fixed.rows)
    assert any(row.name == "authenticated_source_equality" for row in fixed.rows)
    q0 = point(p, pattern, [0], 2)
    assert O.replay_exact_linear_point(fixed, q0)["verified"]
    corrupted = list(q0)
    corrupted[0] = F(1, 100)
    with pytest.raises(RuntimeError, match="constraint replay"):
        O.replay_exact_linear_point(fixed, corrupted)


def test_128_phases_are_fully_fixed_without_triangle_relaxation():
    d, n = 128, 1
    zero = (F(0),) * d
    p = O.ExactPerspectiveProblem(x0=zero, X=(zero,), low=(F(-1),), high=(F(1),),
        gamma=(F(1),) * d, beta=zero, epsilon=F(1),
        W1=(zero,) * d, b1=zero, W2=(zero,) * d, b2=zero)
    lp, audit = O.build_fixed_phase_linear_lp(p, [False] * d)
    assert audit["all_phases_fixed"] and not audit["triangle_relaxation_present"]
    assert len([row for row in lp.rows if row.name.startswith("fixed_inactive_value")]) == 128
    assert len([row for row in lp.rows if row.name.startswith("cancellation")]) == 127
    q0 = point(p, [False] * d, [0], 1)
    assert O.replay_exact_linear_point(lp, q0)["verified"]
    propose = lambda *_: {"model_status": "Optimal", "feasible": True, "column_values": q0}
    witness, report = O.search_fixed_phase_exact_witness(p, lp, [False] * d, propose, 5.)
    assert witness is not None and report["verified"]
    assert report["exact_anchor_replay_verified"]
    assert O.replay_fixed_phase_semantic_witness(p, lp, witness)["verified"]


def test_raw_nonzero_branch_threshold_rejected():
    p, lp, pattern, _ = fixture()
    bad = O.ExactLPRow("branch_inactive_sign[0]", (4,), (F(1),), None, F(1))
    with pytest.raises(O.FixedPhaseInvariantError, match="homogeneous perspective"):
        O.build_fixed_phase_linear_lp(p, pattern, replace(lp, rows=lp.rows + (bad,)))


def test_homogeneous_scaled_g_branch_is_retained():
    p, lp, pattern, _ = fixture()
    branch = O.ExactLPRow("branch_inactive_sign[0]", (4,), (F(1),), None, F(0))
    fixed, audit = O.build_fixed_phase_linear_lp(p, pattern, replace(lp, rows=lp.rows + (branch,)))
    assert branch in fixed.rows
    assert audit["inherited_branch_audit"] == "HOMOGENEOUS_SCALED_G_U_BRANCHES_VERIFIED"


def test_nullity_greater_than_one_failed_portfolio_is_inconclusive():
    p, lp, pattern, _ = fixture(cancellation_anywhere=True, sources=2)
    q0 = point(p, pattern, [0, 0], 2)
    propose = lambda *_: {"model_status": "Optimal", "feasible": True, "column_values": q0}
    found, report = O.search_fixed_phase_exact_witness(p, lp, pattern, propose, 2.)
    assert found is None and report["equality_nullity"] == 3
    assert report["search_status"] == "INCONCLUSIVE"
    assert not report["permits_infeasibility_claim"]
    assert O.scientific_status_from_proof(open_nodes=1) == O.INCONCLUSIVE


def test_numerical_infeasibility_without_exact_farkas_is_inconclusive():
    p, lp, pattern, _ = fixture()
    propose = lambda *_: {"model_status": "Infeasible", "feasible": False, "column_values": None}
    found, report = O.search_fixed_phase_exact_witness(p, lp, pattern, propose, 2.)
    assert found is None and report["search_status"] == "INCONCLUSIVE"
    assert not report["permits_infeasibility_claim"]


def test_anchor_reconstruction_failure_is_inconclusive():
    p, lp, pattern, _ = fixture()
    # An extra incompatible source equality cannot be repaired by the selected
    # cancellation-only heuristic. Failure says nothing about other phases.
    lp = replace(lp, rows=lp.rows + (
        O.ExactLPRow("extra_source", (0,), (F(1),), F(1), F(1)),))
    q0 = point(p, pattern, [0], 2)
    propose = lambda *_: {"model_status": "Optimal", "feasible": True, "column_values": q0}
    found, report = O.search_fixed_phase_exact_witness(p, lp, pattern, propose, 2.)
    assert found is None and not report["exact_anchor_replay_verified"]
    assert report["search_status"] == "INCONCLUSIVE"


def test_gradient_second_anchor_sign_change_returns_exact_irrational_witness():
    p, lp, pattern, _ = fixture(F(2))
    tau = lp.column_lower[3]
    q0, q1 = point(p, pattern, [0], tau), point(p, pattern, [0], 2)
    objectives = []
    def propose(model, objective, seconds, role):
        assert model.identity() == lp.identity()
        objectives.append((role, objective))
        return {"model_status": "Optimal", "feasible": True,
                "column_values": q0 if role == "anchor" else q1}
    witness, report = O.search_fixed_phase_exact_witness(p, lp, pattern, propose, 3.)
    assert witness is not None and "algebraic_root" in witness
    assert report["exact_anchor_replay_verified"] and report["second_anchor_replay_verified"]
    assert objectives[1][0] == "MAXIMIZE_PHI_GRADIENT"
    assert objectives[1][1][3] < 0
    direction = report["directions"][-1]
    assert direction["endpoint_residual_sign_change"]
    assert direction["alpha_interval_contains_zero"]
    assert direction["exact_replay_result"]["fixed_phase_linear_replay"]["verified"]
    assert O.replay_fixed_phase_semantic_witness(p, lp, witness)["verified"]


def test_small_real_highspy_proposes_but_exact_replay_certifies():
    p, lp, pattern, _ = fixture(F(2))
    upper = list(lp.column_upper)
    upper[3] = F(2)
    lp = replace(lp, column_upper=tuple(upper))
    def propose(model, objective, seconds, _role):
        return O.solve_highspy(model, objective=objective,
                              time_limit_seconds=min(2., seconds), proposal_only=True)
    witness, report = O.search_fixed_phase_exact_witness(p, lp, pattern, propose, 5.)
    assert witness is not None and report["verified"]
    assert O.replay_fixed_phase_semantic_witness(p, lp, witness)["verified"]


def test_float_point_cannot_discharge_exact_anchor_replay():
    p, lp, pattern, _ = fixture()
    with pytest.raises(RuntimeError, match="floating"):
        O.replay_exact_linear_point(lp, [float(v) for v in point(p, pattern, [0], 2)])


def _proposal_warning_engine(monkeypatch, log_path, *, api="run", mutation=None,
                             warning="WARNING: Problem has some excessively small costs",
                             model_status=None, solution_factory=None,
                             run_status=None, pass_status=None, run_mutation=None):
    import highspy
    real = highspy.Highs
    class Engine:
        def __init__(self):
            self.inner = real()
        def __getattr__(self, name):
            return getattr(self.inner, name)
        def passModel(self, model):
            result = self.inner.passModel(model)
            if mutation is not None:
                mutation(self.inner)
            if api == "passModel":
                log_path.write_text(warning + "\n")
                return highspy.HighsStatus.kWarning if pass_status is None else pass_status
            return result
        def run(self):
            self.inner.run()
            if run_mutation is not None:
                run_mutation(self.inner)
            if api == "run":
                log_path.write_text(warning + "\n")
                return highspy.HighsStatus.kWarning if run_status is None else run_status
            return highspy.HighsStatus.kOk
        def getModelStatus(self):
            return self.inner.getModelStatus() if model_status is None else model_status
        def getSolution(self):
            solution = self.inner.getSolution()
            return solution if solution_factory is None else solution_factory(solution)
    monkeypatch.setattr(highspy, "Highs", Engine)


@pytest.mark.parametrize("api", ["passModel", "run"])
def test_proposal_warning_scaled_matrix_tiny_cost_and_exact_replay(monkeypatch, tmp_path, api):
    p, lp, pattern, _ = fixture()
    lp = replace(lp, rows=lp.rows + (O.ExactLPRow(
        "tiny_matrix_row", (0,), (F(1, 2**70),), None, F(1, 2**70)),))
    identity = lp.identity()
    objective = [F(0)] * lp.column_count
    objective[0], objective[1] = F(3), F(1, 2**80)
    expected_objective_sha = O._sha_json([O._fs(v) for v in objective])
    log = tmp_path / "highs.log"
    _proposal_warning_engine(monkeypatch, log, api=api)
    result = O.solve_highspy(lp, log, objective=objective, proposal_only=True)
    assert result["proposal_available"] and result["exact_replay_required"]
    assert lp.identity() == identity
    assert result["rational_objective_sha256"] == expected_objective_sha
    audit = result["proposal_objective_audit"]
    assert audit["normalization_exact"] == "3/1"
    assert [v["column"] for v in audit["dropped"]] == [1]
    assert result["solver_scaling"]["sub_threshold_entries_before"] == 1
    assert result["solver_scaling"]["sub_threshold_entries_after"] == 0
    assert all(result["construction_diagnostic"]["proposal_retained_model_audit"].values())
    persisted = O.cluster_common.verified_json(Path(result["proposal_audit_path"]))
    assert persisted["rational_objective_sha256"] == expected_objective_sha
    assert persisted["proposal_objective_audit"] == audit
    assert persisted["exact_replay_required"]
    assert not persisted["authorizes_witness_or_exclusion"]
    workspace = O.FixedPhaseAnchorWorkspace(p, pattern, lp)
    anchor, evidence = workspace.reconstruct_anchor(result["column_values"], 2.)
    assert evidence["exact_anchor_replay_verified"]
    assert O.replay_exact_linear_point(lp, anchor)["verified"]
    corrupted = list(anchor)
    corrupted[0] = F(2)  # Violates the original exact tiny row, not an objective.
    with pytest.raises(RuntimeError):
        O.replay_exact_linear_point(lp, corrupted)
    assert not result["direct_dual_ray_available"]


@pytest.mark.parametrize("api", ["passModel", "run"])
def test_warning_remains_fatal_without_explicit_proposal_mode(monkeypatch, tmp_path, api):
    _, lp, _, _ = fixture()
    log = tmp_path / "strict.log"
    _proposal_warning_engine(monkeypatch, log, api=api)
    with pytest.raises(O.HighsCanonicalLPDiagnosticError):
        O.solve_highspy(lp, log, objective=[F(1)] * lp.column_count)


@pytest.mark.parametrize("mutation", [
    lambda h: h.changeColBounds(0, -3., 4.),
    lambda h: h.changeRowBounds(0, -1., 1.),
    lambda h: h.changeCoeff(0, 0, 0.5),
])
def test_proposal_warning_rejects_solver_changed_constraints(monkeypatch, tmp_path, mutation):
    _, lp, _, _ = fixture()
    log = tmp_path / "changed.log"
    _proposal_warning_engine(monkeypatch, log, mutation=mutation)
    with pytest.raises(O.HighsCanonicalLPDiagnosticError):
        O.solve_highspy(lp, log, proposal_only=True)


@pytest.mark.parametrize("warning", ["WARNING: no useful basis was obtained",
                                     "WARNING: unknown solver warning", ""])
def test_unclassified_execution_warning_is_heuristic_if_model_intact(monkeypatch, tmp_path, warning):
    _, lp, _, _ = fixture()
    log = tmp_path / "unknown.log"
    _proposal_warning_engine(monkeypatch, log, warning=warning)
    result = O.solve_highspy(lp, log, proposal_only=True)
    assert result["proposal_warning"] and result["proposal_usable_primal"]
    assert all(result["construction_diagnostic"]["proposal_retained_model_audit"].values())


@pytest.mark.parametrize("values,reason", [
    (None, "NO_PRIMAL_ARRAY"),
    ([], "PRIMAL_DIMENSION_MISMATCH"),
    ([0.] * 7, "PRIMAL_DIMENSION_MISMATCH"),
    ([float("nan")] * 8, "NONFINITE_PRIMAL_VALUES"),
    ([float("inf")] * 8, "NONFINITE_PRIMAL_VALUES"),
])
def test_unknown_warning_without_usable_primal_is_audited_inconclusive(monkeypatch, tmp_path,
                                                                   values, reason):
    import highspy
    p, lp, pattern, _ = fixture()
    log = tmp_path / "unknown.log"
    _proposal_warning_engine(monkeypatch, log,
        model_status=highspy.HighsModelStatus.kUnknown, warning="",
        solution_factory=lambda _: SimpleNamespace(col_value=values, value_valid=False))
    result = O.solve_highspy(lp, log, proposal_only=True)
    assert result["model_status"] == "Unknown"
    assert result["proposal_warning"]
    assert not result["proposal_usable_primal"] and not result["proposal_available"]
    assert result["proposal_failure_reason"] == reason
    assert result["column_values"] is None
    assert not result["infeasible"] and not result["direct_dual_ray_available"]
    persisted = O.cluster_common.verified_json(Path(result["proposal_audit_path"]))
    assert persisted["proposal_failure_reason"] == reason
    assert persisted["model_status"] == "Unknown"
    assert not persisted["authorizes_witness_or_exclusion"]
    found, report = O.search_fixed_phase_exact_witness(p, lp, pattern,
                                                      lambda *_: result, 2.)
    assert found is None and report["search_status"] == "INCONCLUSIVE"
    assert not report["exact_anchor_replay_verified"]
    assert not report["permits_infeasibility_claim"]
    assert report["anchors"][0]["heuristic_failure"]
    assert report["anchors"][0]["proposal_failure_reason"] == reason


@pytest.mark.parametrize("execution_status", ["kWarning", "kError", "kOk"])
def test_unknown_finite_proposal_attempts_exact_reconstruction_and_replay(monkeypatch, tmp_path,
                                                                      execution_status):
    import highspy
    p, lp, pattern, _ = fixture()
    log = tmp_path / "finite_unknown.log"
    _proposal_warning_engine(monkeypatch, log,
        model_status=highspy.HighsModelStatus.kUnknown, warning="",
        run_status=getattr(highspy.HighsStatus, execution_status),
        solution_factory=lambda s: SimpleNamespace(col_value=s.col_value, value_valid=False))
    result = O.solve_highspy(lp, log, proposal_only=True)
    assert result["proposal_usable_primal"] and result["proposal_available"]
    assert result["proposal_failure_reason"] is None
    assert not result["feasible"] and not result["infeasible"]
    calls = []
    replay = O.replay_exact_linear_point
    def record_replay(*args):
        calls.append(args)
        return replay(*args)
    monkeypatch.setattr(O, "replay_exact_linear_point", record_replay)
    found, report = O.search_fixed_phase_exact_witness(p, lp, pattern,
                                                      lambda *_: result, 2.)
    assert calls and report["exact_anchor_replay_verified"]
    assert found is not None
    assert O.replay_fixed_phase_semantic_witness(p, lp, found)["verified"]


@pytest.mark.parametrize("api", ["passModel", "run"])
def test_identical_unknown_warning_in_strict_mode_is_fatal(monkeypatch, tmp_path, api):
    import highspy
    _, lp, _, _ = fixture()
    log = tmp_path / "strict_unknown.log"
    _proposal_warning_engine(monkeypatch, log, api=api,
                             model_status=highspy.HighsModelStatus.kUnknown, warning="")
    with pytest.raises(O.HighsCanonicalLPDiagnosticError):
        O.solve_highspy(lp, log)


def test_pass_model_error_remains_fatal_in_proposal_mode(monkeypatch, tmp_path):
    import highspy
    _, lp, _, _ = fixture()
    log = tmp_path / "construction_error.log"
    _proposal_warning_engine(monkeypatch, log, api="passModel",
                             pass_status=highspy.HighsStatus.kError)
    with pytest.raises(O.HighsCanonicalLPDiagnosticError):
        O.solve_highspy(lp, log, proposal_only=True)


@pytest.mark.parametrize("api", ["passModel", "run"])
def test_matrix_corruption_rejected_even_with_ok_api_status(monkeypatch, tmp_path, api):
    import highspy
    _, lp, _, _ = fixture()
    log = tmp_path / "corrupt_ok.log"
    mutate = lambda h: h.changeCoeff(0, 0, 0.5)
    _proposal_warning_engine(monkeypatch, log, api=api,
        mutation=mutate if api == "passModel" else None,
        run_mutation=mutate if api == "run" else None,
        pass_status=highspy.HighsStatus.kOk, run_status=highspy.HighsStatus.kOk)
    with pytest.raises(O.HighsCanonicalLPDiagnosticError) as failure:
        O.solve_highspy(lp, log, proposal_only=True)
    assert failure.value.diagnostic["failure_classification"] == "HIGHS_LOADED_CANONICAL_MODEL_DIFFERS"


def test_missing_primal_anchor_does_not_stop_phase_bab(monkeypatch, tmp_path):
    import highspy
    p, lp, pattern, _ = fixture()
    log = tmp_path / "bab_unknown.log"
    _proposal_warning_engine(monkeypatch, log,
        model_status=highspy.HighsModelStatus.kUnknown,
        solution_factory=lambda _: None, warning="")
    result = O.solve_highspy(lp, log, proposal_only=True)
    monkeypatch.setattr(O, "DIMENSION", 2)
    root = np.zeros(8)
    root[6] = .25
    bounds = {"stable_active": [], "stable_inactive": [1], "unstable": [0]}
    attempts, nodes = [], []
    def witness(_pattern, _candidate, node_id):
        found, report = O.search_fixed_phase_exact_witness(p, lp, pattern,
                                                         lambda *_: result, 2.)
        attempts.append(node_id)
        assert report["search_status"] == "INCONCLUSIVE"
        return found, report
    def solve_node(_phases, node_id):
        nodes.append(node_id)
        return {"feasible": True, "solution": root, "infeasible": False}
    found, branch = O.run_phase_continuation(root, bounds, 1, 3, 4,
        O.time.perf_counter() + 2., solve_node, witness)
    assert found is None and len(attempts) == 3 and len(nodes) == 2
    assert branch["nodes_closed_by_certificate"] == 0 and branch["nodes_open"] == 2


def test_proposal_execution_programming_error_still_propagates(monkeypatch, tmp_path):
    _, lp, _, _ = fixture()
    log = tmp_path / "implementation_error.log"
    def implementation_error(_):
        raise O.FixedPhaseInvariantError("actual implementation defect")
    _proposal_warning_engine(monkeypatch, log, run_mutation=implementation_error)
    with pytest.raises(O.FixedPhaseInvariantError, match="actual implementation defect"):
        O.solve_highspy(lp, log, proposal_only=True)


def test_limited_proposal_is_not_optimality_or_feasibility_proof(monkeypatch, tmp_path):
    import highspy
    p, lp, pattern, _ = fixture()
    log = tmp_path / "limit.log"
    _proposal_warning_engine(monkeypatch, log, model_status=highspy.HighsModelStatus.kTimeLimit,
                             warning="")
    result = O.solve_highspy(lp, log, proposal_only=True)
    assert not result["feasible"] and result["proposal_available"]
    assert result["exact_replay_required"]
    assert result["construction_diagnostic"]["model_status_at_warning"] == "Time limit reached"
    def propose(*_):
        return result
    witness, report = O.search_fixed_phase_exact_witness(p, lp, pattern, propose, 3.)
    assert report["exact_anchor_replay_verified"]
    if witness is not None:
        assert O.replay_fixed_phase_semantic_witness(p, lp, witness)["verified"]


def test_limited_proposal_cannot_bypass_failed_exact_replay(monkeypatch, tmp_path):
    import highspy
    p, lp, pattern, _ = fixture()
    log = tmp_path / "limit.log"
    _proposal_warning_engine(monkeypatch, log, model_status=highspy.HighsModelStatus.kTimeLimit,
                             warning="")
    result = O.solve_highspy(lp, log, proposal_only=True)
    result["column_values"][:] = 1e10
    calls = []
    def reject_replay(*args):
        calls.append(args)
        raise RuntimeError("exact constraint replay failed")
    monkeypatch.setattr(O, "replay_exact_linear_point", reject_replay)
    witness, report = O.search_fixed_phase_exact_witness(p, lp, pattern,
        lambda *_: result, 2., feasibility_propose=unusable_feasibility_proposal)
    assert not calls and witness is None and not report["verified"]
    assert report["anchor_reconstruction_skipped_reason"] == "UNUSABLE_PROPOSAL"
    assert not report["anchor_reconstruction_attempted"]


def test_real_highs_time_limit_warning_has_proposal_only_handling(tmp_path):
    import highspy
    rng = np.random.default_rng(0)
    n = 50
    lp = O.ExactCanonicalLP(tuple(f"x{i}" for i in range(n)),
        (F(0),) * n, (F(1),) * n, tuple(O.ExactLPRow(
            f"r{i}", tuple(range(n)), tuple(map(F.from_float, row)), F(1), None)
            for i, row in enumerate(rng.uniform(0.01, 1, (n, n)))))
    objective = [F(1)] + [F(1, 2**80)] * (n - 1)
    result = O.solve_highspy(lp, tmp_path / "real.log", objective=objective,
                            time_limit_seconds=1e-10, proposal_only=True)
    assert result["run_status"] == str(highspy.HighsStatus.kWarning)
    assert result["model_status"] == "Time limit reached"
    assert result["proposal_available"] and result["exact_replay_required"]
    with pytest.raises(O.HighsCanonicalLPDiagnosticError) as failure:
        O.solve_highspy(lp, objective=objective, time_limit_seconds=1e-10)
    assert failure.value.diagnostic["model_status_at_run"] == "Time limit reached"


def test_direction_must_preserve_every_equality():
    p, lp, pattern, _ = fixture()
    q0 = point(p, pattern, [0], 2)
    direction = [F(0)] * len(q0)
    direction[1] = F(1)
    with pytest.raises(O.FixedPhaseInvariantError, match="preserve every"):
        O.evaluate_fixed_phase_direction(p, lp, pattern, q0, direction)


def test_complete_fixed_phase_witness_rejects_node_identity_mutation():
    p, lp, pattern, _ = fixture()
    q0 = point(p, pattern, [0], 1)
    witness, _ = O.evaluate_fixed_phase_direction(p, lp, pattern, q0, [F(0)] * len(q0))
    witness["fixed_phase_linear_lp_sha256"] = "wrong"
    with pytest.raises(RuntimeError, match="LP identity"):
        O.replay_fixed_phase_semantic_witness(p, lp, witness)


def test_opposite_inherited_branch_is_not_discarded_by_fixed_pattern():
    p, parent, pattern, _ = fixture(cancellation_anywhere=True)
    inherited = O.ExactLPRow("branch_active_sign[0]", (4,), (F(-1),), None, F(0))
    parent = replace(parent, rows=parent.rows + (inherited,))
    changed_pattern = [False, True]
    fixed, _audit = O.build_fixed_phase_linear_lp(p, changed_pattern, parent)
    assert inherited in fixed.rows
    assert any(row.name == "fixed_inactive_sign[0]" for row in fixed.rows)
    q = point(p, changed_pattern, [0], 2)
    with pytest.raises(RuntimeError, match="constraint replay"):
        O.replay_exact_linear_point(fixed, q)


def test_fixed_phase_infeasibility_needs_exact_farkas_replay():
    p, lp, pattern, _ = fixture()
    lp = replace(lp, rows=lp.rows + (
        O.ExactLPRow("authenticated_source_equality", (0,), (F(1),), F(1), F(1)),))
    result = O.solve_highspy(lp, time_limit_seconds=2.)
    assert result["infeasible"]
    # A solver status alone never closes a phase or declares scientific exclusion.
    found, report = O.search_fixed_phase_exact_witness(p, lp, pattern, lambda *_: result, 2.)
    assert found is None and not report["permits_infeasibility_claim"]
    # Supply an independently derived exact separator, rather than requiring
    # the existing untrusted numerical ray-repair heuristic to succeed.
    by_name = {row.name: i for i, row in enumerate(lp.rows)}
    cert = {"schema": "CORET_EXACT_LP_FARKAS_CERTIFICATE_V1",
            "canonical_lp_sha256": lp.identity(), "multipliers": [
                {"kind": "row", "index": by_name[name], "orientation": orientation,
                 "multiplier": str(multiplier)} for name, orientation, multiplier in (
                    ("centered[0]", -1, 1), ("centered[1]", 1, 1),
                    ("cancellation[1]", -1, 1), ("authenticated_source_equality", -1, 2))]}
    assert O.verify_exact_lp_farkas(lp, cert)["verified"]
    assert O.scientific_status_from_proof(root_certificate_verified=True) == O.EXCLUDED


def infeasible_fixed_phase_fixture():
    p, lp, pattern, _ = fixture()
    lp = replace(lp, rows=lp.rows + (
        O.ExactLPRow("authenticated_source_equality", (0,), (F(1),), F(1), F(1)),))
    names = {row.name: i for i, row in enumerate(lp.rows)}
    certificate = {"schema": "CORET_EXACT_LP_FARKAS_CERTIFICATE_V1",
        "canonical_lp_sha256": lp.identity(), "multipliers": [
            {"kind": "row", "index": names[name], "orientation": orientation,
             "multiplier": str(multiplier)} for name, orientation, multiplier in (
                ("centered[0]", -1, 1), ("centered[1]", 1, 1),
                ("cancellation[1]", -1, 1), ("authenticated_source_equality", -1, 2))]}
    assert O.verify_exact_lp_farkas(lp, certificate)["verified"]
    return p, lp, pattern, certificate


def test_infeasible_primal_never_reconstructed_and_exact_farkas_is_phase_local(monkeypatch):
    p, lp, pattern, certificate = infeasible_fixed_phase_fixture()
    calls = []
    def repair(model, ray, seconds):
        calls.append(model.identity())
        return certificate, [], "EXACT_FARKAS_VERIFIED"
    monkeypatch.setattr(O, "repair_direct_dual_ray", repair)
    monkeypatch.setattr(O, "FixedPhaseAnchorWorkspace",
                        lambda *_: pytest.fail("infeasible LP must not manufacture an anchor"))
    proposal = {"model_status": "Infeasible", "column_values": point(p, pattern, [0], 2),
                "proposal_available": True, "original_row_dual_ray": np.ones(len(lp.rows))}
    found, report = O.search_fixed_phase_exact_witness(p, lp, pattern, lambda *_: proposal, 2.)
    assert found is None and calls == [lp.identity()]
    assert not report["anchor_reconstruction_attempted"]
    assert report["fixed_phase_farkas_attempted"] and report["fixed_phase_farkas_verified"]
    assert report["search_status"] == "FIXED_PHASE_LINEARLY_INFEASIBLE"
    assert report["certificate_scope"] == "THIS_FIXED_PHASE_LINEAR_LP_ONLY"
    assert not report["permits_infeasibility_claim"] and not report["permits_node_exclusion"]
    assert O.scientific_status_from_proof(open_nodes=1) == O.INCONCLUSIVE


def test_infeasible_without_exact_certificate_is_inconclusive(monkeypatch):
    p, lp, pattern, _ = fixture()
    calls = []
    def fallback(model, _seconds):
        calls.append(model.identity())
        return None, {"certificate_verified": False}
    monkeypatch.setattr(O, "phase1_exact_farkas_fallback", fallback)
    monkeypatch.setattr(O, "FixedPhaseAnchorWorkspace", lambda *_: pytest.fail("no anchor"))
    found, report = O.search_fixed_phase_exact_witness(p, lp, pattern, lambda *_: {
        "model_status": "Infeasible", "column_values": point(p, pattern, [0], 1)}, 2.)
    assert calls == [lp.identity()] and found is None
    assert report["fixed_phase_farkas_attempted"] and not report["fixed_phase_farkas_verified"]
    assert report["search_status"] == "INCONCLUSIVE" and not report["permits_infeasibility_claim"]


def test_unreplayed_farkas_claim_never_authorizes_fixed_phase_exclusion(monkeypatch):
    p, lp, pattern, certificate = infeasible_fixed_phase_fixture()
    certificate["canonical_lp_sha256"] = "different_lp"
    monkeypatch.setattr(O, "repair_direct_dual_ray", lambda *_: (
        certificate, [], "EXACT_FARKAS_VERIFIED"))
    found, report = O.search_fixed_phase_exact_witness(p, lp, pattern, lambda *_: {
        "model_status": "Infeasible", "column_values": None,
        "original_row_dual_ray": np.ones(len(lp.rows))}, 2.)
    assert found is None and not report["fixed_phase_farkas_verified"]
    assert report["search_status"] == "INCONCLUSIVE"
    assert "exact_certificate_rejection" in report
    assert not report["permits_infeasibility_claim"]


def test_unknown_out_of_box_discarded_before_correction(monkeypatch):
    p, lp, pattern, _ = fixture()
    monkeypatch.setattr(O, "FixedPhaseAnchorWorkspace", lambda *_: pytest.fail("no blind correction"))
    proposal = {"model_status": "Unknown", "column_values": np.asarray(
        point(p, pattern, [5], 2), dtype=float), "proposal_available": True}
    found, report = O.search_fixed_phase_exact_witness(p, lp, pattern, lambda *_: proposal, 2.,
        feasibility_propose=unusable_feasibility_proposal)
    assert found is None and report["search_status"] == "INCONCLUSIVE"
    assert report["proposal_primal_present"]
    assert not report["proposal_primal_numerically_within_bounds"]
    assert not report["anchor_reconstruction_attempted"]
    assert report["anchor_reconstruction_skipped_reason"] == "UNUSABLE_PROPOSAL"
    assert not report["fixed_phase_farkas_attempted"]


def test_unknown_in_box_but_row_violation_is_discarded(monkeypatch):
    p, lp, pattern, _ = fixture()
    candidate = np.asarray(point(p, pattern, [0], 2), dtype=float)
    candidate[1] = .01
    monkeypatch.setattr(O, "FixedPhaseAnchorWorkspace", lambda *_: pytest.fail("no blind correction"))
    found, report = O.search_fixed_phase_exact_witness(p, lp, pattern, lambda *_: {
        "model_status": "Unknown", "column_values": candidate}, 2.,
        feasibility_propose=unusable_feasibility_proposal)
    assert found is None and report["proposal_primal_numerically_within_bounds"]
    assert not report["proposal_primal_numerically_within_rows"]
    assert report["first_numerical_row_failure"]["name"] == "centered[0]"
    assert report["anchor_reconstruction_skipped_reason"] == "UNUSABLE_PROPOSAL"


@pytest.mark.parametrize("status", ["Unknown", "Time limit reached", "Iteration limit reached",
                                   "Optimal", "Feasible"])
def test_numerically_admissible_status_enters_exact_reconstruction(monkeypatch, status):
    p, lp, pattern, _ = fixture()
    calls = []
    solve = O._solve_exact_correction_rhs
    def counted_solve(*args):
        calls.append(args)
        return solve(*args)
    monkeypatch.setattr(O, "_solve_exact_correction_rhs", counted_solve)
    found, report = O.search_fixed_phase_exact_witness(p, lp, pattern, lambda *_: {
        "model_status": status, "column_values": np.asarray(point(p, pattern, [0], 1), dtype=float)}, 2.)
    assert calls and found is not None
    assert report["proposal_primal_numerically_within_bounds"]
    assert report["proposal_primal_numerically_within_rows"]
    assert report["anchor_reconstruction_attempted"] and report["exact_anchor_replay_verified"]
    assert report["anchors"][0]["correction_intervals_derived_before_solve"]
    assert report["anchors"][0]["exact_correction_constraints_verified"]
    assert not report["fixed_phase_farkas_attempted"]
    assert O.replay_fixed_phase_semantic_witness(p, lp, found)["verified"]


def test_correction_basis_cannot_leave_source_box_before_solving(monkeypatch):
    p, lp, pattern, _ = fixture()
    lower, upper = list(lp.column_lower), list(lp.column_upper)
    lower[0], upper[0] = F(1), F(2)
    lp = replace(lp, column_lower=tuple(lower), column_upper=tuple(upper))
    workspace = O.FixedPhaseAnchorWorkspace(p, pattern, lp)
    monkeypatch.setattr(O, "_solve_exact_correction_rhs", lambda *_: pytest.fail("inadmissible basis must be skipped"))
    with pytest.raises(O.ExactSolveFailure, match="correction basis") as failure:
        workspace.reconstruct_anchor(np.asarray(point(p, pattern, [1], 2), dtype=float), 2.)
    assert not failure.value.anchor_diagnostic["exact_correction_applied"]


def test_retained_inequality_participates_in_pre_solve_correction_intervals(monkeypatch):
    p, lp, pattern, _ = fixture(sources=2)
    lp = replace(lp, rows=lp.rows + (O.ExactLPRow(
        "retained_source_cut", (1,), (F(1),), F(0), None),))
    workspace = O.FixedPhaseAnchorWorkspace(p, pattern, lp)
    monkeypatch.setattr(O, "_solve_exact_correction_rhs", lambda *_: pytest.fail("retained cut must constrain basis"))
    with pytest.raises(O.ExactSolveFailure, match="retained_source_cut"):
        workspace.reconstruct_anchor(np.asarray(point(p, pattern, [1, 1], 2), dtype=float), 2.)


def test_bad_exact_correction_rejected_before_application(monkeypatch):
    p, lp, pattern, _ = fixture()
    workspace = O.FixedPhaseAnchorWorkspace(p, pattern, lp)
    monkeypatch.setattr(O, "_solve_exact_correction_rhs", lambda *_: [[F(100)]])
    calls, make_point = [], O._fixed_phase_point_from_sources
    def checked_point(*args):
        calls.append(args[2])
        assert all(F(-4) <= value <= F(4) for value in args[2])
        return make_point(*args)
    monkeypatch.setattr(O, "_fixed_phase_point_from_sources", checked_point)
    with pytest.raises(O.ExactSolveFailure, match="outside admissible interval") as failure:
        workspace.reconstruct_anchor(np.asarray(point(p, pattern, [0], 2), dtype=float), 2.)
    assert len(calls) == 2  # Construct proposal and base, never an escaped anchor.
    diagnostic = failure.value.anchor_diagnostic
    assert not diagnostic["exact_correction_applied"]
    assert diagnostic["first_exact_replay_failure"] == "correction[0]: source/inequality interval"


def test_first_exact_replay_failure_is_reported_for_admitted_proposal(monkeypatch):
    p, lp, pattern, _ = fixture()
    def reject(*_):
        raise RuntimeError("deliberate complete replay rejection")
    monkeypatch.setattr(O, "replay_exact_linear_point", reject)
    found, report = O.search_fixed_phase_exact_witness(p, lp, pattern, lambda *_: {
        "model_status": "Unknown", "column_values": np.asarray(point(p, pattern, [0], 1), dtype=float)}, 2.)
    assert found is None and report["anchor_reconstruction_attempted"]
    assert report["first_exact_replay_failure"] == "deliberate complete replay rejection"
    assert report["search_status"] == "INCONCLUSIVE" and not report["permits_infeasibility_claim"]


def test_correction_interval_projection_matches_actual_source_and_scale_changes():
    p, lp, pattern, _ = fixture(cancellation_anywhere=True, sources=2)
    workspace = O.FixedPhaseAnchorWorkspace(p, pattern, lp)
    q = point(p, pattern, [F(1, 2), F(1, 3)], F(2))
    selected = (0, 1, 2)  # Includes scale/t.
    conditions, intervals = workspace.correction_admissibility(q, selected)
    delta = [F(1, 7), F(-1, 11), F(1, 13)]
    shifted = point(p, pattern, [q[0] + delta[0], q[1] + delta[1]], q[4] + delta[2])
    by_name = {row.name: row for row in lp.rows}
    for label, base, coefficients, _lo, _hi in conditions:
        reconstructed = base + sum((value * delta[j] for j, value in coefficients), F(0))
        actual = (shifted[int(label[7:-1])] if label.startswith("column[") else
                  O._dot(by_name[label].coefficients, [shifted[i] for i in by_name[label].indices]))
        assert reconstructed == actual
    assert len(intervals) == 3


@pytest.mark.parametrize("bad_coefficient", [1.0, "c*t"])
def test_claimed_linear_system_rejects_nonexact_or_nonlinear_coefficient(bad_coefficient):
    p, lp, pattern, _ = fixture()
    bad = O.ExactLPRow("extra_untrusted", (0,), (bad_coefficient,), None, F(0))
    with pytest.raises(O.FixedPhaseInvariantError, match="nonlinear"):
        O.build_fixed_phase_linear_lp(p, pattern, replace(lp, rows=lp.rows + (bad,)))


def test_one_sided_interval_and_zero_inequality_slope_are_supported():
    interval = O.exact_affine_parameter_interval([
        ("active_face", F(0), F(1), F(0), None),
        ("constant_valid", F(1), F(0), F(0), F(2))])
    assert interval["alpha_interval_lower"] == "0/1"
    assert interval["alpha_interval_upper"] is None
    assert O._interval_contains_exact(interval, F(0))
    _, roots, report = O.exact_polynomial_roots_in_interval((0, 1, 0), interval)
    assert roots == [("rational", F(0), None)] and report["roots_inside_interval"] == 1


def test_gradient_minimization_is_tried_after_maximum_fails_to_find_root():
    p, lp, pattern, _ = fixture()
    q0, q1 = point(p, pattern, [0], 2), point(p, pattern, [0], 1)
    calls = []
    def propose(_model, objective, _seconds, role):
        calls.append(role)
        if role == "MINIMIZE_PHI_GRADIENT":
            assert objective[3] > 0
            values = q1
        else:
            values = q0
        return {"model_status": "Optimal", "feasible": True, "column_values": values}
    witness, report = O.search_fixed_phase_exact_witness(p, lp, pattern, propose, 2.)
    assert calls == ["anchor", "MAXIMIZE_PHI_GRADIENT", "MINIMIZE_PHI_GRADIENT"]
    assert witness is not None and report["verified"]
    assert report["directions"][-1]["family"] == "MINIMIZE_PHI_GRADIENT"


def test_inherited_node_cut_filters_roots_and_is_replayed_at_certification():
    p, lp, pattern, _ = fixture(cancellation_anywhere=True)
    lp = replace(lp, rows=lp.rows + (
        O.ExactLPRow("authenticated_perspective_source_cut", (0,), (F(1),), None, F(1)),))
    q0, q1 = point(p, pattern, [0], 2), point(p, pattern, [1], 2)
    witness, report = line(p, lp, pattern, q0, q1)
    assert witness is not None and report["roots_inside_interval"] == 1
    assert report["alpha_interval_upper"] == "1/1"
    # sqrt(3) is not silently accepted outside the inherited cut. The negative
    # root remains available, so explicitly cut that side as well.
    lp = replace(lp, rows=lp.rows + (
        O.ExactLPRow("authenticated_perspective_source_lower", (0,), (F(1),), F(0), None),))
    witness, report = line(p, lp, pattern, q0, q1)
    assert witness is None and report["roots_inside_interval"] == 0


def test_stable_plus_branched_neurons_form_a_complete_phase_without_128_branches():
    bounds = {"stable_active": [0], "stable_inactive": [1]}
    assert O._complete_phase_map(bounds, {2: True}, 3) == [True, False, True]
    assert O._complete_phase_map(bounds, {}, 3) is None
    with pytest.raises(O.FixedPhaseInvariantError, match="stable phase"):
        O._complete_phase_map(bounds, {0: False, 2: True}, 3)
