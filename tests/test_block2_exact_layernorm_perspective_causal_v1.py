from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_block2_exact_layernorm_perspective_causal_v1.py"
SPEC = importlib.util.spec_from_file_location("exact_ln_perspective", SCRIPT)
ORACLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ORACLE
SPEC.loader.exec_module(ORACLE)
F = Fraction


def problem():
    # The one shared source translates both coordinates identically.  Exact
    # centering therefore cancels it, exercising correlation preservation.
    return ORACLE.ExactPerspectiveProblem(
        x0=(F(1), F(-1)), X=((F(1), F(1)),),
        low=(F(-1),), high=(F(1),),
        gamma=(F(1), F(1)), beta=(F(0), F(0)), epsilon=F(0),
        W1=((F(1), F(0)), (F(0), F(1))), b1=(F(0), F(0)),
        W2=((F(0), F(0)), (F(2), F(0))), b2=(F(0), F(0)))


def witness(**updates):
    value = {
        "schema": ORACLE.WITNESS_SCHEMA,
        "source_values": ["1/2"], "t": "1/1",
        "relu_active": [True, False],
    }
    value.update(updates)
    return value


def test_perspective_transformation_matches_direct_layernorm():
    x = [F(3, 2), F(-1, 2)]
    gamma, beta, epsilon = [F(2), F(3)], [F(1), F(-2)], F(1, 4)
    c, _variance, t, y = ORACLE.perspective_direct(x, gamma, beta, epsilon)
    perspective = [float(g * centered) / t + float(b)
                   for g, centered, b in zip(gamma, c, beta)]
    assert perspective == y


def test_positive_homogeneity_for_positive_scale():
    for h in (-3.0, 0.0, 2.5):
        for t in (0.25, 2.0):
            assert t * max(0.0, h) == max(0.0, t * h)


def test_rational_lorentz_outer_contains_exact_layernorm_point():
    centered = [F(1), F(-1)]
    t_squared = sum((x * x for x in centered), F(0)) / 2 + F(1, 4)
    upper = ORACLE._exact_sqrt_upper(t_squared)
    assert upper * upper >= t_squared
    assert ORACLE.rational_lorentz_contains(centered, upper)


def infeasible_cone_program():
    # With no primal variables, s=h=-1 in the nonnegative cone is impossible.
    return ORACLE.CanonicalConeProgram(
        A=(), b=(), G=((),), h=(F(-1),),
        cones=(("nonnegative", 1),))


def certificate(value="1/1"):
    return {"schema": ORACLE.CERTIFICATE_SCHEMA,
            "equality_dual": [], "cone_dual": [value]}


def test_corrupted_conic_dual_certificate_rejected():
    with pytest.raises(RuntimeError, match="dual cone"):
        ORACLE.verify_rational_conic_certificate(
            infeasible_cone_program(), certificate("-1/1"))


def test_valid_synthetic_conic_infeasibility_certificate_accepted():
    result = ORACLE.verify_rational_conic_certificate(
        infeasible_cone_program(), certificate())
    assert result["verified"] is True
    assert result["strict_separator"] == "-1/1"


def test_lorentz_dual_membership_is_replayed_exactly():
    program = ORACLE.CanonicalConeProgram(
        A=(), b=(), G=((), ()), h=(F(-1), F(0)),
        cones=(("lorentz", 2),))
    valid = {"schema": ORACLE.CERTIFICATE_SCHEMA,
             "equality_dual": [], "cone_dual": ["1/1", "0/1"]}
    assert ORACLE.verify_rational_conic_certificate(
        program, valid)["dual_cone_membership"] is True
    invalid = {**valid, "cone_dual": ["1/1", "2/1"]}
    with pytest.raises(RuntimeError, match="dual cone"):
        ORACLE.verify_rational_conic_certificate(program, invalid)


def test_exact_fixed_pattern_quadratic_witness_reconstructed():
    result = ORACLE.replay_exact_perspective_witness(problem(), witness())
    assert result["verified"] is True
    assert result["maximum_exact_residual"] == "0"


def test_quadratic_root_isolating_interval_replay():
    result = ORACLE.verify_quadratic_isolating_interval(
        ["1/1", "0/1", "-2/1"], ["7/5", "3/2"])
    assert result["verified"] and result["degree"] == 2
    with pytest.raises(RuntimeError, match="does not strictly isolate"):
        ORACLE.verify_quadratic_isolating_interval(
            ["1/1", "0/1", "-2/1"], ["3/2", "2/1"])


def test_degree_two_algebraic_witness_replays_exactly():
    # epsilon=1 changes the exact scale to sqrt(2); all downstream equations
    # remain exact because beta=b1=b2=0 in this fixture.
    changed = replace(problem(), epsilon=F(1))
    algebraic = {
        "schema": ORACLE.WITNESS_SCHEMA,
        "algebraic_root": {
            "polynomial": ["1/1", "0/1", "-2/1"],
            "isolating_interval": ["7/5", "3/2"],
        },
        "source_affine": [{"constant": "1/2", "slope": "0/1"}],
        "t_affine": {"constant": "0/1", "slope": "1/1"},
        "relu_active": [True, False],
    }
    replay = ORACLE.replay_algebraic_perspective_witness(changed, algebraic)
    assert replay["verified"] is True
    assert replay["polynomial_degree"] == 2


def test_source_box_violation_rejected():
    with pytest.raises(RuntimeError, match="source box"):
        ORACLE.replay_exact_perspective_witness(
            problem(), witness(source_values=["2/1"]))


def test_relu_sign_violation_rejected():
    with pytest.raises(RuntimeError, match="inactive ReLU sign"):
        ORACLE.replay_exact_perspective_witness(
            problem(), witness(relu_active=[False, False]))


def test_layernorm_quadratic_equality_mutation_rejected():
    with pytest.raises(RuntimeError, match="quadratic equality"):
        ORACLE.replay_exact_perspective_witness(problem(), witness(t="2/1"))


def test_cancellation_equality_mutation_rejected():
    changed = replace(problem(), W2=((F(0), F(0)), (F(1), F(0))))
    with pytest.raises(RuntimeError, match="cancellation equality"):
        ORACLE.replay_exact_perspective_witness(changed, witness())


def test_shared_source_correlation_is_preserved():
    left = ORACLE.replay_exact_perspective_witness(
        problem(), witness(source_values=["-1/1"]))
    right = ORACLE.replay_exact_perspective_witness(
        problem(), witness(source_values=["1/1"]))
    assert left["shared_source_correlation_preserved"] is True
    assert right["shared_source_correlation_preserved"] is True


def complete_tree(open_active=False):
    nodes = {
        "root": {"phases": {}, "branch_neuron": 7,
                 "children": {"inactive": "i", "active": "a"}},
        "i": {"phases": {"7": False}, "certificate": {"ok": True}},
        "a": {"phases": {"7": True},
              "certificate": None if open_active else {"ok": True}},
    }
    return {"schema": ORACLE.TREE_SCHEMA, "root": "root", "nodes": nodes}


def certificate_checker(value):
    if value != {"ok": True}:
        raise RuntimeError("bad certificate")


def test_branching_covers_both_relu_phases_exactly():
    result = ORACLE.verify_branch_tree(complete_tree(), certificate_checker)
    assert result == {"verified": True, "nodes": 3, "leaves": 2,
                      "closed_leaves": 2, "open_leaves": 0,
                      "permits_excluded": True}


def test_open_bab_leaf_forces_inconclusive():
    result = ORACLE.verify_branch_tree(
        complete_tree(open_active=True), certificate_checker)
    assert result["open_leaves"] == 1
    assert result["permits_excluded"] is False


def test_complete_certified_tree_permits_excluded():
    assert ORACLE.verify_branch_tree(
        complete_tree(), certificate_checker)["permits_excluded"] is True


def valid_authentication():
    return {
        "property_id": ORACLE.PROPERTY_ID,
        "tested_radius": ORACLE.RADIUS,
        "tested_radius_hex": ORACLE.RADIUS_HEX,
        "pinned_deept_revision": "rev",
        "canonical_state_identity": {
            "canonical_state_identity_sha256":
                ORACLE.EXPECTED_CANONICAL_IDENTITY},
    }


def valid_parameters():
    return {name: {"parameter_identity_authenticated": True,
                   "pinned_revision": "rev"}
            for name in ("layernorm", "ffn_first", "ffn_second")}


@pytest.mark.parametrize("mutation", ["token", "state", "parameter"])
def test_authenticated_token_state_parameter_mismatch_hard_fails(mutation):
    auth = valid_authentication()
    downstream = {"analysis_token_index": ORACLE.EXPECTED_TOKEN,
                  "authenticated": True}
    parameters = valid_parameters()
    if mutation == "token":
        downstream["analysis_token_index"] = 1
    elif mutation == "state":
        auth["canonical_state_identity"] = {
            "canonical_state_identity_sha256": "0" * 64}
    else:
        parameters["ffn_second"]["parameter_identity_authenticated"] = False
    with pytest.raises(RuntimeError):
        ORACLE._verify_cross_authentication(auth, downstream, parameters)


def test_exact_shared_source_authentication_accepts_complete_identity():
    assert ORACLE._verify_cross_authentication(
        valid_authentication(),
        {"analysis_token_index": 0, "authenticated": True},
        valid_parameters()) is True


def test_backend_without_certificate_adapter_is_never_decisive():
    backend = ORACLE._backend_inventory()
    assert backend["selected"] == "NONE_CERTIFICATE_CAPABLE_INSTALLED"
    assert backend["certificate_extraction_supported"] is False
