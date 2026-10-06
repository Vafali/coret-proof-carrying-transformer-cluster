"""Bounded CPU-only partial-phase proofs; no scientific artifact is executed."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as F
import time

import numpy as np
import pytest

from test_fixed_phase_layernorm_witness_v1 import O


def partial_fixture(monkeypatch, dimension=3):
    monkeypatch.setattr(O, "DIMENSION", dimension)
    sources = 1
    g0, u0 = sources + dimension + 1, sources + 2 * dimension + 1
    names = ("xi[0]", *(f"c[{i}]" for i in range(dimension)), "t",
             *(f"g[{i}]" for i in range(dimension)),
             *(f"u[{i}]" for i in range(dimension)))
    lower = (F(-1), *([F(-1)] * dimension), F(1),
             *([F(-1)] * dimension), *([F(0)] * dimension))
    upper = (F(1), *([F(1)] * dimension), F(1),
             *([F(1)] * dimension), *([F(1)] * dimension))
    # At g[0]=0 a triangle admits u[0]=1/4, but neither exact ReLU phase does.
    rows = [O.ExactLPRow("g_zero", (g0,), (F(1),), F(0), F(0)),
            O.ExactLPRow("cancellation", (u0,), (F(1),), F(1, 4), F(1, 4)),
            O.ExactLPRow("centering", tuple(range(1, 1 + dimension)),
                         (F(1),) * dimension, F(0), F(0))]
    for i in range(dimension):
        rows.extend((
            O.ExactLPRow(f"triangle_lower[{i}]", (g0 + i, u0 + i),
                         (F(1), F(-1)), None, F(0)),
            O.ExactLPRow(f"triangle_upper[{i}]", (dimension + 1, g0 + i, u0 + i),
                         (F(-1), F(-1), F(2)), None, F(0))))
    lp = O.ExactCanonicalLP(names, lower, upper, tuple(rows))
    point = np.zeros(len(names))
    point[dimension + 1], point[u0] = 1., .25
    bounds = {"stable_active": [], "stable_inactive": [],
              "unstable": list(range(dimension))}
    assert O.replay_exact_linear_point(lp, list(map(F, point)))["verified"]
    return lp, point, bounds


def exact_node_certificate(root, phases):
    lp = O.lp_with_relu_phases(root, phases, 1)
    assert 0 in phases
    entries = [{"kind": "row", "index": 1, "orientation": -1, "multiplier": "1/1"}]
    branch_name = f"branch_{'active' if phases[0] else 'inactive'}_value[0]"
    index = next(i for i, row in enumerate(lp.rows) if row.name == branch_name)
    entries.append({"kind": "row", "index": index, "orientation": 1, "multiplier": "1/1"})
    if phases[0]:
        entries.append({"kind": "row", "index": 0, "orientation": 1, "multiplier": "1/1"})
    certificate = {"schema": "CORET_EXACT_LP_FARKAS_CERTIFICATE_V1",
                   "canonical_lp_sha256": lp.identity(), "multipliers": entries}
    assert O.verify_exact_lp_farkas(lp, certificate)["verified"]
    return certificate


def continuation(lp, point, bounds, solve, *, maximum_nodes=32, deadline=None, witness=None):
    return O.run_phase_continuation(point, bounds, 1, maximum_nodes, 1,
        time.perf_counter() + 4. if deadline is None else deadline, solve,
        witness or (lambda *_: (None, {"verified": False})), root_lp=lp)


def certifying_solver(lp, *, require_complete=False, open_active=False):
    def solve(phases, _node_id):
        cert = None
        if (not require_complete or len(phases) == O.DIMENSION) and not (
                open_active and phases[0]):
            cert = exact_node_certificate(lp, phases)
        return {"solver_status": "Infeasible", "solution": None, "certificate": cert,
                "node_lp_sha256": O.lp_with_relu_phases(lp, phases, 1).identity()}
    return solve


@pytest.mark.parametrize("active", [False, True])
def test_partial_lp_preserves_every_original_row_bound_and_unfixed_triangle(monkeypatch, active):
    root, _, _ = partial_fixture(monkeypatch)
    node = O.lp_with_relu_phases(root, {0: active}, 1)
    assert node.rows[:len(root.rows)] == root.rows
    assert node.variable_names == root.variable_names
    assert node.column_lower == root.column_lower and node.column_upper == root.column_upper
    assert len(node.rows) == len(root.rows) + 2
    assert node.rows[-1].coefficients == (F(-1) if active else F(1),)
    assert node.rows[-2].lower == node.rows[-2].upper == 0


def test_partial_certificates_prune_whole_subtrees_and_both_children_close_parent(monkeypatch):
    lp, point, bounds = partial_fixture(monkeypatch)
    found, audit = continuation(lp, point, bounds, certifying_solver(lp))
    assert found is None
    assert audit["phase_bab_nodes_created"] == audit["phase_bab_nodes_processed"] == 3
    assert audit["phase_bab_nodes_closed_exact_farkas"] == 2
    assert audit["phase_bab_open_nodes"] == 0 and audit["phase_bab_max_depth"] == 1
    assert audit["termination_reason"] == "EXHAUSTIVE_EXACT_FARKAS_CLOSURE"
    replay = O.verify_partial_phase_tree(audit["tree"], lp, bounds, 1)
    assert replay["permits_excluded"]
    assert O.scientific_status_from_proof(complete_tree_verified=True, open_nodes=0) == O.EXCLUDED
    for node in audit["tree"]["nodes"].values():
        if node.get("certificate"):
            assert node["phase_bab_fixed_phase_count"] == 1 < O.DIMENSION
            assert F(node["node_exact_farkas_rhs"]) < 0


def test_real_highspy_partial_node_uses_existing_trigger_repair_full_original_replay(monkeypatch):
    lp, _, _ = partial_fixture(monkeypatch)
    checked = []
    original = O.verify_exact_lp_farkas
    def check(model, certificate):
        checked.append(model.identity())
        return original(model, certificate)
    monkeypatch.setattr(O, "verify_exact_lp_farkas", check)
    def propose(model):
        return O.solve_highspy(model, proposal_only=True,
                              proposal_method="SIMPLEX_FEASIBILITY", time_limit_seconds=1.)
    result = O.solve_partial_phase_node(lp, {0: False}, 1, propose, certificate_seconds=2.)
    expected = O.lp_with_relu_phases(lp, {0: False}, 1)
    assert result["solver_status"] == "Infeasible" and result["certificate"] is not None
    assert expected.identity() in checked
    assert original(expected, result["certificate"])["verified"]
    assert result["certificate_attempt"]["exact_farkas_replay_verified"]


@pytest.mark.parametrize("status", ["Unknown", "Infeasible"])
def test_no_primal_or_exact_certificate_still_branches_to_exhaustion(monkeypatch, status):
    lp, point, bounds = partial_fixture(monkeypatch, dimension=2)
    def solve(*_):
        return {"solver_status": status, "infeasible": status == "Infeasible",
                "feasible": False, "solution": None, "certificate": None}
    _, audit = continuation(lp, point, bounds, solve)
    assert audit["phase_bab_nodes_created"] == 7
    assert audit["phase_bab_nodes_closed_exact_farkas"] == 0
    assert audit["phase_bab_open_nodes"] == 4
    assert not O.verify_partial_phase_tree(audit["tree"], lp, bounds, 1)["permits_excluded"]


def test_exhaustive_full_leaf_certificates_and_unresolved_child(monkeypatch):
    lp, point, bounds = partial_fixture(monkeypatch, dimension=2)
    _, closed = continuation(lp, point, bounds, certifying_solver(lp, require_complete=True))
    assert closed["phase_bab_nodes_created"] == 7 and closed["phase_bab_nodes_closed_exact_farkas"] == 4
    assert O.verify_partial_phase_tree(closed["tree"], lp, bounds, 1)["permits_excluded"]
    _, partial = continuation(lp, point, bounds, certifying_solver(lp, open_active=True))
    assert partial["phase_bab_nodes_closed_exact_farkas"] == 1 and partial["phase_bab_open_nodes"] == 2
    assert not O.verify_partial_phase_tree(partial["tree"], lp, bounds, 1)["permits_excluded"]


@pytest.mark.parametrize("limit", ["node", "wall"])
def test_limits_are_inconclusive_never_proof(monkeypatch, limit):
    lp, point, bounds = partial_fixture(monkeypatch)
    _, audit = continuation(lp, point, bounds, certifying_solver(lp), maximum_nodes=1,
                            deadline=0. if limit == "wall" else None)
    assert audit["phase_bab_open_nodes"] == 1
    assert audit["termination_reason"] == ("WALL_CLOCK_LIMIT" if limit == "wall" else "MAXIMUM_NODES")
    assert not O.verify_partial_phase_tree(audit["tree"], lp, bounds, 1)["permits_excluded"]


def test_verified_witness_still_immediately_wins(monkeypatch):
    lp, point, bounds = partial_fixture(monkeypatch)
    witness = {"verified": True}
    found, audit = continuation(lp, point, bounds, lambda *_: pytest.fail("no child solve"),
                              witness=lambda *_: (witness, {"verified": True}))
    assert found is witness and audit["phase_bab_nodes_created"] == 1
    assert O.scientific_status_from_proof(exact_witness_verified=True) == O.FEASIBLE


def test_complete_pattern_certificate_cannot_close_partial_parent(monkeypatch):
    lp, point, bounds = partial_fixture(monkeypatch, dimension=2)
    leaf = exact_node_certificate(lp, {0: False, 1: False})
    def solve(*_):
        return {"solver_status": "Infeasible", "certificate": leaf, "solution": None}
    _, audit = continuation(lp, point, bounds, solve, maximum_nodes=3)
    assert audit["phase_bab_nodes_closed_exact_farkas"] == 0
    assert not O.verify_partial_phase_tree(audit["tree"], lp, bounds, 1)["permits_excluded"]
    assert any("identity differs" in node.get("node_certificate_replay_failure", "")
               for node in audit["tree"]["nodes"].values())


def test_full_original_replay_failure_cannot_close_node(monkeypatch):
    lp, point, bounds = partial_fixture(monkeypatch, dimension=1)
    def solve(phases, _node_id):
        cert = exact_node_certificate(lp, phases)
        cert["multipliers"][0]["multiplier"] = "0/1"
        return {"solver_status": "Infeasible", "certificate": cert, "solution": None,
                "exact_farkas_replay_verified": True}
    _, audit = continuation(lp, point, bounds, solve)
    assert audit["phase_bab_nodes_closed_exact_farkas"] == 0
    assert audit["phase_bab_open_nodes"] == 2


@pytest.mark.parametrize("mutation", ["certificate", "bound", "phase", "extra_phase", "row", "identity"])
def test_tree_replay_binds_exact_node_context_and_full_original_system(monkeypatch, mutation):
    lp, point, bounds = partial_fixture(monkeypatch)
    _, audit = continuation(lp, point, bounds, certifying_solver(lp))
    tree = deepcopy(audit["tree"])
    inactive = tree["nodes"]["root.i0"]
    if mutation == "certificate":
        inactive["certificate"] = exact_node_certificate(lp, {0: True})
    elif mutation == "bound":
        lp = replace(lp, column_lower=(F(-2), *lp.column_lower[1:]))
    elif mutation == "row":
        lp = replace(lp, rows=(replace(lp.rows[0], upper=F(1)), *lp.rows[1:]))
    elif mutation == "phase":
        inactive["phases"]["0"] = "False"
    elif mutation == "extra_phase":
        inactive["phases"]["1"] = False
    else:
        inactive["node_lp_sha256"] = "0" * 64
    with pytest.raises(RuntimeError):
        O.verify_partial_phase_tree(tree, lp, bounds, 1)


def test_solver_lp_substitution_is_invariant_failure(monkeypatch):
    lp, point, bounds = partial_fixture(monkeypatch)
    with pytest.raises(O.FixedPhaseInvariantError, match="LP identity"):
        continuation(lp, point, bounds, lambda *_: {"node_lp_sha256": "wrong"})


def test_speculative_complete_phase_proof_has_no_global_exclusion_authority(monkeypatch):
    lp, point, bounds = partial_fixture(monkeypatch)
    def pattern_search(*_):
        return None, {"search_status": "FIXED_PHASE_LINEARLY_INFEASIBLE",
                      "fixed_phase_farkas_verified": True,
                      "fixed_phase_farkas_certificate": exact_node_certificate(lp, {0: False, 1: False, 2: False})}
    _, audit = continuation(lp, point, bounds, lambda *_: {}, maximum_nodes=1, witness=pattern_search)
    assert audit["nodes_open"] == 1 and audit["nodes_closed_by_certificate"] == 0


def test_without_authenticated_root_lp_numerical_or_certificate_flags_cannot_close(monkeypatch):
    lp, point, bounds = partial_fixture(monkeypatch, dimension=1)
    _, audit = O.run_phase_continuation(point, bounds, 1, 3, 0, time.perf_counter() + 2.,
        lambda phases, _: {"certificate": exact_node_certificate(lp, phases), "infeasible": True},
        lambda *_: pytest.fail("witness disabled"))
    assert audit["nodes_closed_by_certificate"] == 0 and audit["nodes_open"] == 2


@pytest.mark.parametrize("defect", ["topology", "nonpositive_t"])
def test_node_builder_rejects_malformed_root_not_a_scientific_closure(monkeypatch, defect):
    lp, _, _ = partial_fixture(monkeypatch)
    if defect == "topology":
        lp = replace(lp, variable_names=("wrong_source", *lp.variable_names[1:]))
    else:
        lp = replace(lp, column_lower=(*lp.column_lower[:4], F(0), *lp.column_lower[5:]))
    with pytest.raises(O.FixedPhaseInvariantError):
        O.solve_partial_phase_node(lp, {0: False}, 1, lambda *_: pytest.fail("solver must not run"))


def test_unknown_root_and_children_branch_without_any_primal(monkeypatch):
    lp, _, bounds = partial_fixture(monkeypatch, dimension=1)
    found, audit = continuation(lp, None, bounds,
        lambda *_: {"solver_status": "Unknown", "solution": None})
    assert found is None and audit["phase_bab_nodes_processed"] == 3
    assert audit["phase_bab_open_nodes"] == 2 and audit["phase_bab_nodes_closed_exact_farkas"] == 0


def test_real_solver_and_existing_repair_close_partial_tree_end_to_end(monkeypatch):
    lp, point, bounds = partial_fixture(monkeypatch)
    def propose(model):
        return O.solve_highspy(model, proposal_only=True,
                              proposal_method="SIMPLEX_FEASIBILITY", time_limit_seconds=1.)
    def solve(phases, _node_id):
        return O.solve_partial_phase_node(lp, phases, 1, propose, certificate_seconds=2.)
    found, audit = continuation(lp, point, bounds, solve)
    assert found is None and audit["phase_bab_nodes_closed_exact_farkas"] == 2
    assert audit["phase_bab_nodes_created"] == 3
    assert O.verify_partial_phase_tree(audit["tree"], lp, bounds, 1)["permits_excluded"]
    for node in audit["tree"]["nodes"].values():
        if node.get("certificate"):
            assert node["node_ray_source"] in ("TRIGGER_SIMPLEX_FEASIBILITY", "DEDICATED_CERTIFICATE")
            assert node["certificate_attempt"]["exact_farkas_replay_verified"]
