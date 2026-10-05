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
