"""Exact model-construction tests only: no SDP backend, captured state, or GPU."""
from dataclasses import replace
from fractions import Fraction as F
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/build_block2_partial_shor_sdp_v1.py"
SPEC = importlib.util.spec_from_file_location("partial_shor_model_test", PATH)
S = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = S
SPEC.loader.exec_module(S)
O = S.oracle


def fixture(monkeypatch, *, sources=1, infeasible=False):
    monkeypatch.setattr(O, "DIMENSION", 2)
    center = np.array([1., -1.])
    generators = np.zeros((sources, 2))
    gamma = np.zeros(2) if infeasible else np.ones(2)
    beta = np.zeros(2)
    W1 = np.eye(2)
    b1 = np.zeros(2)
    W2 = np.zeros((2, 2))
    b2 = np.array([0., 1.])
    lo, hi = -np.ones(sources), np.ones(sources)
    bounds = O._derive_exact_bounds(center, generators, lo, hi,
                                   gamma, beta, W1, b1, F(3))
    lp = O.build_exact_perspective_lp(lo, hi, gamma, beta, W1, b1, W2, b2, bounds)
    return S.build_partial_shor(lp, F(3), sources, 2), lp


def exact_linear_value(row, point):
    return sum((value * point[index] for index, value in zip(row.indices, row.coefficients)), F(0))


def test_A_exact_feasible_point_satisfies_every_shor_constraint(monkeypatch):
    model, _ = fixture(monkeypatch)
    q = [F(1), F(-1), F(2), F(1), F(0)]
    z = [F(0)] + q
    assert O.replay_exact_linear_point(model.linear_lp, z)["verified"]
    for row in model.moment_equalities:
        assert sum((value * q[i] * q[j] for i, j, value in row.terms), F(0)) == row.rhs
    # M=[1;q][1;q]^T is exactly PSD/rank-one by construction, not a solver claim.
    Q = np.outer(np.array(q, dtype=float), np.array(q, dtype=float))
    telemetry = S.numerical_telemetry(model, [0.], np.array(q, dtype=float), Q)
    assert telemetry["numerical_rank_estimate"] == 1
    assert telemetry["independently_measured_moment_equality_residual"] == 0
    assert telemetry["independently_measured_linear_violation"] == 0
    assert telemetry["independently_measured_bound_violation"] == 0
    assert not telemetry["proof_authority"]


def test_B_known_infeasible_qcqp_shor_has_same_exact_linear_contradiction(monkeypatch):
    model, _ = fixture(monkeypatch, infeasible=True)
    cancellation = next(row for row in model.linear_lp.rows if row.name == "cancellation[1]")
    # gamma=0, W2=0, beta+b2 difference=1: cancellation forces t=0.
    assert cancellation.indices == (3,) and cancellation.coefficients == (F(1),)
    assert cancellation.lower == cancellation.upper == 0
    assert model.linear_lp.column_lower[3] > 0
    # Adding PSD/moment equations cannot resolve this linear contradiction.
    # NO numerical SDP status or production exclusion is asserted.


def test_C_high_rank_relaxed_point_is_not_promoted_to_exact_witness(monkeypatch):
    model, _ = fixture(monkeypatch)
    q = np.array([1., -1., 2., 1., 0.])
    Q = np.outer(q, q)
    # PSD diagonal increment: d*delta_tt=delta_c0c0+delta_c1c1;
    # u moments are unchanged, so each ReLU complementarity is preserved.
    Q += np.diag([1., 1., 1., 0., 0.])
    telemetry = S.numerical_telemetry(model, [0.], q, Q)
    assert telemetry["numerical_rank_estimate"] == 4
    assert telemetry["minimum_moment_eigenvalue"] >= -1e-12
    assert telemetry["independently_measured_moment_equality_residual"] == 0
    assert telemetry["independently_measured_linear_violation"] == 0
    assert telemetry["q_reconstruction_consistency"]["rank_one_gap_frobenius"] > 0
    assert telemetry["final_status"] == "NUMERICAL_PROBE_ONLY_NO_SCIENTIFIC_AUTHORITY"


def test_only_c_t_u_are_lifted_not_sources(monkeypatch):
    model, _ = fixture(monkeypatch, sources=13)
    summary = model.summary()
    assert summary["psd_block_dimension"] == 6 and summary["q_dimension"] == 5
    assert summary["scalar_variable_count"] == 13 + 5 + 15
    assert len(model.linear_lp.variable_names) == 13 + 5
    assert summary["xi_lifted"] is False
    assert all(0 <= i <= j < 5 for row in model.moment_equalities for i, j, _ in row.terms)


def test_substitution_exactly_replays_every_original_linear_row_and_source_box(monkeypatch):
    model, root = fixture(monkeypatch)
    q = [F(1, 3), F(-2, 5), F(7, 4), F(3, 8), F(-1, 9)]
    g = [sum((coefficient * q[i] for i, coefficient in row), F(0)) for row in model.g_affine]
    original = [F(2, 3)] + q[:3] + g + q[3:]
    transformed = [F(2, 3)] + q
    rows = {row.name: row for row in model.linear_lp.rows}
    for row in root.rows:
        if row.name.startswith("preactivation["):
            assert exact_linear_value(row, original) == 0
        else:
            assert exact_linear_value(row, original) == exact_linear_value(rows[row.name], transformed)
            assert (row.lower, row.upper) == (rows[row.name].lower, rows[row.name].upper)
    assert model.linear_lp.column_lower[0] == root.column_lower[0]
    assert model.linear_lp.column_upper[0] == root.column_upper[0]


def test_relu_offdiagonal_moment_terms_have_no_factor_two_error(monkeypatch):
    model, _ = fixture(monkeypatch)
    q = [F(1, 3), F(-2, 5), F(7, 4), F(3, 8), F(-1, 9)]
    for i, row in enumerate(model.moment_equalities[1:]):
        g = sum((coefficient * q[j] for j, coefficient in model.g_affine[i]), F(0))
        u = q[3 + i]
        assert sum((value * q[a] * q[b] for a, b, value in row.terms), F(0)) == u * (u - g)


def test_nonzero_beta_bias_and_gamma_are_exactly_encoded_in_moments(monkeypatch):
    monkeypatch.setattr(O, "DIMENSION", 2)
    center, generators = np.array([1., -1.]), np.zeros((1, 2))
    lo, hi, gamma, beta = np.array([-1.]), np.array([1.]), np.array([2., 3.]), np.ones(2)
    W1, b1 = np.array([[1., 2.], [-3., 4.]]), np.array([.5, .5])
    W2, b2 = np.zeros((2, 2)), np.zeros(2)
    bounds = O._derive_exact_bounds(center, generators, lo, hi, gamma, beta, W1, b1, F(3))
    root = O.build_exact_perspective_lp(lo, hi, gamma, beta, W1, b1, W2, b2, bounds)
    model = S.build_partial_shor(root, F(3), 1, 2)
    assert model.g_affine == (((0, F(2)), (1, F(6)), (2, F(7, 2))),
                              ((0, F(-6)), (1, F(12)), (2, F(3, 2))))
    q = [F(1, 3), F(-2, 5), F(7, 4), F(3, 8), F(-1, 9)]
    for i, row in enumerate(model.moment_equalities[1:]):
        g = sum((a * q[j] for j, a in model.g_affine[i]), F(0))
        assert sum((a * q[j] * q[k] for j, k, a in row.terms), F(0)) == q[3 + i] * (q[3 + i] - g)


def test_g_bounds_are_preserved_as_affine_constraints(monkeypatch):
    _, root = fixture(monkeypatch)
    # original g0 is column 4, eliminated but its bound must remain present.
    root = replace(root, column_lower=(*root.column_lower[:4], F(-2), *root.column_lower[5:]))
    model = S.build_partial_shor(root, F(3), 1, 2)
    row = next(row for row in model.linear_lp.rows if row.name == "g_bound[0]")
    assert row.lower == -2 and row.indices == (1,) and row.coefficients == (F(1),)


@pytest.mark.parametrize("defect", ["g_rhs", "g_order", "missing_cancellation", "t_domain"])
def test_malformed_canonical_structure_rejected(monkeypatch, defect):
    _, root = fixture(monkeypatch)
    rows = list(root.rows)
    if defect == "g_rhs":
        i = next(i for i, row in enumerate(rows) if row.name == "preactivation[0]")
        rows[i] = replace(rows[i], lower=F(1), upper=F(1))
        root = replace(root, rows=tuple(rows))
    elif defect == "g_order":
        root = replace(root, variable_names=(*root.variable_names[:4], "g[1]", "g[0]", *root.variable_names[6:]))
    elif defect == "missing_cancellation":
        root = replace(root, rows=tuple(row for row in rows if not row.name.startswith("cancellation[")))
    else:
        root = replace(root, column_lower=(*root.column_lower[:3], F(0), *root.column_lower[4:]))
    with pytest.raises(RuntimeError):
        S.build_partial_shor(root, F(3), 1, 2)


def test_model_identity_and_exact_coefficients_are_deterministic(monkeypatch):
    first, _ = fixture(monkeypatch)
    second, _ = fixture(monkeypatch)
    # Authentication is synthetic, no wall-clock-dependent fields enter identity.
    assert first.record() == second.record()
    assert O._sha_json(first.record()) == O._sha_json(second.record())
    assert all(isinstance(value, F) for row in first.moment_equalities for _, _, value in row.terms)


@pytest.mark.parametrize("status", ["infeasible", "optimal", "unknown"])
def test_numerical_status_never_authorizes_science(monkeypatch, status):
    model, _ = fixture(monkeypatch)
    telemetry = S.numerical_telemetry(model, None, None, None, solver="SYNTHETIC_NOT_RUN", status=status)
    assert not telemetry["proof_authority"] and not telemetry["primal_present"]
    assert telemetry["final_status"] == "NUMERICAL_PROBE_ONLY_NO_SCIENTIFIC_AUTHORITY"


def test_nonfinite_and_underflowed_solver_copy_coefficients_rejected():
    for value in (F(1, 2**2000), F(2**2000)):
        with pytest.raises((RuntimeError, OverflowError)):
            S._proposal_float(value)


def test_inconsistent_source_reconstruction_is_visible(monkeypatch):
    model, _ = fixture(monkeypatch)
    q = np.array([2., -1., 2., 1., 0.])
    telemetry = S.numerical_telemetry(model, [0.], q, np.outer(q, q))
    assert telemetry["q_reconstruction_consistency"]["source_relation_max_error"] == 1
    assert telemetry["independently_measured_linear_violation"] >= 1
