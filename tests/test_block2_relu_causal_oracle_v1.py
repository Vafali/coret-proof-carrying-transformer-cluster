from __future__ import annotations

import copy
from fractions import Fraction
import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "relu_causal_oracle",
    ROOT / "scripts/analyze_block2_relu_causal_oracle_v1.py")
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


def state(center, generators=(), low=(), high=(), ids=None):
    center = np.asarray(center, dtype=np.float64)
    generators = np.asarray(generators, dtype=np.float64)
    if generators.size == 0:
        generators = np.empty((0, center.size), dtype=np.float64)
    count = len(generators)
    return {
        "center": center, "generators": generators,
        "low": np.asarray(low, dtype=np.float64),
        "high": np.asarray(high, dtype=np.float64),
        "ids": list(ids or [f"g{i}" for i in range(count)]),
        "masks": [1] * count, "reasons": ["native_semantic"] * count,
        "num_tokens": 1,
    }


def problem(h, r=None, weight=None, bias=None):
    r = r or state(np.zeros(128), np.empty((0, 128)), (), ())
    weight = (np.zeros((128, len(h["center"])), dtype=np.float64)
              if weight is None else weight)
    bias = np.zeros(128) if bias is None else bias
    return ORACLE.ReluCancellationProblem(h, r, weight, bias, "toy")


def _exact_pattern(p, candidate, pattern, tmp_path):
    witness, status = ORACLE.exact_fixed_pattern_witness(
        p, np.asarray(candidate, dtype=np.float64), pattern,
        tmp_path / "witness.json", 5.0)
    assert status["verified"] is True
    assert witness["maximum_exact_residual"] == "0"
    return witness


def test_stable_active_only_exact_feasible(tmp_path):
    h = state([0.0], [[1.0]], [0.0], [1.0])
    w = np.zeros((128, 1)); w[1, 0] = 1.0
    p = problem(h, weight=w)
    assert (len(p.active), len(p.inactive), len(p.unstable)) == (1, 0, 0)
    lp, _row = ORACLE.solve_hull(p)
    assert lp.success
    _exact_pattern(p, lp.x[:p.n], [True], tmp_path)


def test_stable_inactive_only_exact_feasible(tmp_path):
    p = problem(state([-1.0]))
    assert (len(p.active), len(p.inactive), len(p.unstable)) == (0, 1, 0)
    lp, _row = ORACLE.solve_hull(p)
    assert lp.success
    _exact_pattern(p, [], [False], tmp_path)


def test_unstable_hull_feasible_but_exact_relu_milp_infeasible():
    # Cancellation forces xi=0 and y=1/2.  This is in the triangle hull of
    # ReLU(xi), xi in [-1,1], but is not on the exact ReLU graph.
    h = state([0.0], [[1.0]], [-1.0], [1.0], ["shared"])
    r_center = np.zeros(128); r_center[2] = -0.5
    r_gen = np.zeros((1, 128)); r_gen[0, 1] = 1.0
    r = state(r_center, r_gen, [-1.0], [1.0], ["shared"])
    w = np.zeros((128, 1)); w[2, 0] = 1.0
    p = problem(h, r, w)
    hull, _ = ORACLE.solve_hull(p)
    assert hull.success
    exact, record = ORACLE.solve_exact_relu_milp(p, 5.0)
    assert exact.success is False
    assert record["solver_status"] == 2


def test_hull_infeasible_toy_has_exact_farkas_replay():
    r_center = np.zeros(128); r_center[1] = 1.0
    p = problem(state([-1.0]), state(r_center))
    hull, _ = ORACLE.solve_hull(p)
    assert hull.status == 2
    certificate, _diagnostics = ORACLE.propose_farkas(p)
    assert certificate is not None
    checked = ORACLE._verify_farkas(
        p, ORACLE._ratios(certificate["lambda"]),
        ORACLE._ratios(certificate["mu"]))
    assert checked["exact_stationarity"] is True


def test_valid_milp_activation_pattern_exact_witness_replay(tmp_path):
    h = state([0.0], [[1.0]], [-1.0], [1.0])
    p = problem(h)
    witness = _exact_pattern(p, [0.25], [True], tmp_path)
    assert ORACLE.verify_exact_relu_witness(p, witness)["exact_variance"] == "0"


def test_floating_candidate_with_wrong_activation_sign_rejected(tmp_path):
    h = state([0.0], [[1.0]], [-1.0], [1.0])
    p = problem(h)
    witness, status = ORACLE.exact_fixed_pattern_witness(
        p, np.array([-0.5]), [True], tmp_path / "bad.json", 5.0)
    assert witness is None
    assert any("activation sign" in reason
               for reason in status["failure_reasons"])


def test_corrupted_farkas_certificate_rejected():
    r_center = np.zeros(128); r_center[1] = 1.0
    p = problem(state([-1.0]), state(r_center))
    certificate, _ = ORACLE.propose_farkas(p)
    lambdas, mus = ORACLE._ratios(certificate["lambda"]), ORACLE._ratios(certificate["mu"])
    mus = [Fraction(0) for _ in mus]
    with pytest.raises(RuntimeError, match="contradiction"):
        ORACLE._verify_farkas(p, lambdas, mus)


def test_corrupted_exact_witness_rejected(tmp_path):
    p = problem(state([0.0], [[1.0]], [-1.0], [1.0]))
    witness = _exact_pattern(p, [0.25], [True], tmp_path)
    corrupted = copy.deepcopy(witness)
    corrupted["xi_rationals"][0] = {"numerator": "-1", "denominator": "2"}
    with pytest.raises(RuntimeError, match="activation sign"):
        ORACLE.verify_exact_relu_witness(p, corrupted)


def test_equality_witness_outside_source_box_is_rejected(tmp_path):
    p = problem(state([0.0], [[1.0]], [-1.0], [1.0]))
    witness = _exact_pattern(p, [0.25], [True], tmp_path)
    corrupted = copy.deepcopy(witness)
    corrupted["xi_rationals"][0] = {"numerator": "2", "denominator": "1"}
    with pytest.raises(RuntimeError, match="source box"):
        ORACLE.verify_exact_relu_witness(p, corrupted)


def test_corrupted_activation_pattern_identity_is_rejected(tmp_path):
    p = problem(state([0.0], [[1.0]], [-1.0], [1.0]))
    witness = _exact_pattern(p, [0.25], [True], tmp_path)
    corrupted = copy.deepcopy(witness)
    corrupted["activation_pattern"] = [False]
    with pytest.raises(RuntimeError, match="identity differs"):
        ORACLE.verify_exact_relu_witness(p, corrupted)


def test_shared_source_correlation_is_not_duplicated():
    h = state([0.0], [[1.0]], [-1.0], [1.0], ["shared"])
    r = state(np.zeros(128), np.zeros((1, 128)), [-1.0], [1.0], ["shared"])
    p = problem(h, r)
    assert p.n == 1
    assert p.source["ids"] == ["shared"]


def test_parameter_or_manifest_identity_mismatch_is_hard_failure():
    identity = {
        "pinned_deept_revision": "wrong",
        "scientific_manifest_sha256": ORACLE.cluster_common.SCIENTIFIC_MANIFEST_SHA,
        "production_manifest_sha256": ORACLE.cluster_common.PRODUCTION_MANIFEST_SHA,
    }
    with pytest.raises(RuntimeError, match="source identity differs"):
        ORACLE.frontier._load_ffn_second_parameters(identity)


def _two_source_correction_problem():
    h = state([-1.0], [[0.0], [0.0]], [0.0, 0.0], [1.0, 1.0],
              ["x0", "x1"])
    r_center = np.zeros(128)
    r_generators = np.zeros((2, 128))
    r_generators[:, 1] = 1.0
    r = state(r_center, r_generators, [0.0, 0.0], [1.0, 1.0],
              ["x0", "x1"])
    bias = np.zeros(128); bias[1] = -1.0
    return problem(h, r, bias=bias)


def test_slack_aware_alternative_basis_recovers_after_first_leaves_box(tmp_path):
    p = _two_source_correction_problem()
    witness, status = ORACLE.exact_fixed_pattern_witness(
        p, np.array([0.0, 2.0]), [False], tmp_path / "retry.json", 5.0,
        basis_override=[np.array([0]), np.array([1])])
    assert witness is not None
    assert status["attempted_basis_count"] == 2
    assert status["basis_attempts"][0]["failure_reason"] == (
        "SOURCE_BOX_VIOLATION")
    assert status["basis_attempts"][1]["verified"] is True


def test_fixed_pattern_max_slack_finds_strict_interior_candidate():
    p = _two_source_correction_problem()
    candidate, record = ORACLE.fixed_pattern_interior_candidate(p, [False])
    assert record["feasible"] is True
    assert record["best_common_slack_t"] > 0.0
    assert np.all(candidate > 0.0) and np.all(candidate < 1.0)


def test_exact_witness_on_source_and_relu_boundary_is_accepted(tmp_path):
    h = state([0.0], [[1.0]], [0.0], [1.0])
    w = np.zeros((128, 1)); w[1, 0] = 1.0
    p = problem(h, weight=w)
    witness = _exact_pattern(p, [0.0], [True], tmp_path)
    assert witness["source_box_check"] is True
    assert witness["activation_sign_check"] is True


def test_deterministic_basis_order_and_all_fail_remains_unresolved(tmp_path):
    p = _two_source_correction_problem()
    model_problem = p.fixed_pattern_model([False]).problem()
    candidate = np.array([0.5, 0.5])
    first = ORACLE._basis_candidates(p, model_problem, candidate, [False], 8)
    second = ORACLE._basis_candidates(p, model_problem, candidate, [False], 8)
    assert [row.tolist() for row in first] == [row.tolist() for row in second]
    witness, status = ORACLE.exact_fixed_pattern_witness(
        p, np.array([0.0, 2.0]), [False], tmp_path / "none.json", 5.0,
        basis_override=[np.array([0])])
    assert witness is None
    assert status["terminal_recovery_status"] == (
        "FIXED_PATTERN_EXACT_WITNESS_NOT_FOUND")
    assert status["verified"] is False


def test_deterministic_no_good_cut_finds_alternative_pattern(tmp_path):
    # Cancellation is automatic, so both boundary patterns at h=0 are valid.
    p = problem(state([0.0], [[1.0]], [-1.0], [1.0]))
    first, _ = ORACLE.solve_exact_relu_milp(p, 5.0)
    assert first.success
    first_pattern = [bool(round(first.x[p.n + len(p.unstable)]))]
    second, _ = ORACLE.solve_exact_relu_milp(p, 5.0, [first_pattern])
    repeated, _ = ORACLE.solve_exact_relu_milp(p, 5.0, [first_pattern])
    assert second.success and repeated.success
    second_pattern = [bool(round(second.x[p.n + len(p.unstable)]))]
    repeated_pattern = [bool(round(repeated.x[p.n + len(p.unstable)]))]
    assert second_pattern != first_pattern
    assert repeated_pattern == second_pattern
    witness = _exact_pattern(p, second.x[:p.n], second_pattern, tmp_path)
    assert witness["activation_pattern_sha256"] == ORACLE._sha_json(second_pattern)
