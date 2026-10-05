from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
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


def test_bareiss_fixed_pattern_reconstruction_produces_exact_witness():
    fixture = ORACLE.ExactPerspectiveProblem(
        x0=(F(0), F(0)), X=((F(1), F(-1)),),
        low=(F(-1),), high=(F(1),),
        gamma=(F(1), F(1)), beta=(F(0), F(0)), epsilon=F(1),
        W1=((F(1), F(0)), (F(0), F(1))), b1=(F(0), F(0)),
        W2=((F(0), F(0)), (F(0), F(0))), b2=(F(0), F(0)))
    # Layout needed here is xi, c[0], c[1], t; derived g/u are irrelevant to
    # reconstruction because the exact fixture is supplied explicitly.
    proposal = np.asarray([0.25, 0.0, 0.0, 2.0])
    recovered, evidence = ORACLE.reconstruct_exact_fixed_pattern_witness(
        fixture, proposal, [True, False], 1, 2.0)
    assert recovered is not None
    assert evidence["verified"] is True
    assert evidence["method"] == \
        "BAREISS_127_CORRECTIONS_PLUS_QUADRATIC_ROOT"
    if "source_values" in recovered:
        replay = ORACLE.replay_exact_perspective_witness(fixture, recovered)
    else:
        replay = ORACLE.replay_algebraic_perspective_witness(fixture, recovered)
    assert replay["verified"] is True


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


def synthetic_infeasible_lp():
    # x >= 1 and x <= 0.
    return ORACLE.ExactCanonicalLP(
        ("x",), (F(1),), (None,),
        (ORACLE.ExactLPRow("x_le_zero", (0,), (F(1),), None, F(0)),))


def test_highspy_synthetic_infeasible_lp_direct_ray_extraction():
    result = ORACLE.solve_highspy(synthetic_infeasible_lp())
    assert result["infeasible"] is True
    assert result["direct_dual_ray_available"] is True
    certificate, attempts, status = ORACLE.repair_direct_dual_ray(
        synthetic_infeasible_lp(), result["raw_dual_ray"])
    assert status == "EXACT_FARKAS_VERIFIED"
    assert any(row["result"] == "EXACT_FARKAS_VERIFIED" for row in attempts)
    assert ORACLE.verify_exact_lp_farkas(
        synthetic_infeasible_lp(), certificate)["verified"] is True


def test_canonicalization_covers_row_senses_and_variable_bounds():
    lp = ORACLE.ExactCanonicalLP(
        ("x",), (F(-2),), (F(3),), (
            ORACLE.ExactLPRow("le", (0,), (F(1),), None, F(1)),
            ORACLE.ExactLPRow("ge", (0,), (F(1),), F(-1), None),
            ORACLE.ExactLPRow("eq", (0,), (F(2),), F(0), F(0)),
        ))
    refs = ORACLE.canonicalize_all_inequalities(lp)
    assert [(ref.kind, ref.index, ref.orientation) for ref in refs] == [
        ("row", 0, 1), ("row", 1, -1),
        ("row", 2, 1), ("row", 2, -1),
        ("column", 0, 1), ("column", 0, -1)]
    assert ORACLE._inequality(lp, refs[1]) == ((0,), (F(-1),), F(1))


def exact_certificate():
    lp = synthetic_infeasible_lp()
    return lp, {
        "schema": "CORET_EXACT_LP_FARKAS_CERTIFICATE_V1",
        "canonical_lp_sha256": lp.identity(),
        "multipliers": [
            {"kind": "row", "index": 0, "orientation": 1,
             "multiplier": "1/1"},
            {"kind": "column", "index": 0, "orientation": -1,
             "multiplier": "1/1"},
        ],
    }


def test_valid_exact_rational_farkas_replay():
    lp, certificate = exact_certificate()
    assert ORACLE.verify_exact_lp_farkas(
        lp, certificate)["exact_lambda_b"] == "-1/1"


def test_farkas_multiplier_sign_mutation_rejected():
    lp, certificate = exact_certificate()
    certificate["multipliers"][0]["multiplier"] = "-1/1"
    with pytest.raises(RuntimeError, match="negative"):
        ORACLE.verify_exact_lp_farkas(lp, certificate)


def test_farkas_stationarity_mutation_rejected():
    lp, certificate = exact_certificate()
    certificate["multipliers"].pop()
    with pytest.raises(RuntimeError, match="stationarity"):
        ORACLE.verify_exact_lp_farkas(lp, certificate)


def test_non_strict_farkas_contradiction_rejected():
    lp = ORACLE.ExactCanonicalLP(
        ("x",), (F(0),), (None,),
        (ORACLE.ExactLPRow("x_le_zero", (0,), (F(1),), None, F(0)),))
    certificate = {
        "schema": "CORET_EXACT_LP_FARKAS_CERTIFICATE_V1",
        "canonical_lp_sha256": lp.identity(),
        "multipliers": [
            {"kind": "row", "index": 0, "orientation": 1,
             "multiplier": "1/1"},
            {"kind": "column", "index": 0, "orientation": -1,
             "multiplier": "1/1"},
        ]}
    with pytest.raises(RuntimeError, match="not strict"):
        ORACLE.verify_exact_lp_farkas(lp, certificate)


def test_direct_float_ray_is_repaired_to_exact_stationarity():
    lp = synthetic_infeasible_lp()
    certificate, _attempts, status = ORACLE.repair_direct_dual_ray(
        lp, [-0.9999999999997])
    assert status == "EXACT_FARKAS_VERIFIED"
    assert ORACLE.verify_exact_lp_farkas(lp, certificate)["verified"] is True


def test_support_repair_failure_remains_unresolved():
    certificate, attempts, status = ORACLE.repair_direct_dual_ray(
        synthetic_infeasible_lp(), [0.0])
    assert certificate is None
    assert status == "DIRECT_RAY_SUPPORT_REPAIR_FAILED"
    assert attempts


def test_phase_i_exact_dual_fallback_synthetic():
    certificate, record = ORACLE.phase1_exact_farkas_fallback(
        synthetic_infeasible_lp())
    assert record["certificate_verified"] is True
    assert ORACLE.verify_exact_lp_farkas(
        synthetic_infeasible_lp(), certificate)["verified"] is True


def test_lp_feasible_never_implies_exact_feasibility():
    lp = ORACLE.ExactCanonicalLP(
        ("x",), (F(0),), (F(1),),
        (ORACLE.ExactLPRow("loose", (0,), (F(1),), None, F(2)),))
    assert ORACLE.solve_highspy(lp)["feasible"] is True
    assert ORACLE.scientific_status_from_proof() == ORACLE.INCONCLUSIVE


def test_lp_infeasible_without_verified_certificate_is_inconclusive():
    assert ORACLE.scientific_status_from_proof(
        root_certificate_verified=False) == ORACLE.INCONCLUSIVE


def test_certified_root_infeasible_is_excluded():
    assert ORACLE.scientific_status_from_proof(
        root_certificate_verified=True) == ORACLE.EXCLUDED


def test_certified_tree_and_open_tree_statuses():
    assert ORACLE.scientific_status_from_proof(
        complete_tree_verified=True, open_nodes=0) == ORACLE.EXCLUDED
    assert ORACLE.scientific_status_from_proof(
        complete_tree_verified=True, open_nodes=1) == ORACLE.INCONCLUSIVE


def test_highspy_backend_is_primary_and_gurobi_forbidden():
    backend = ORACLE._backend_inventory()
    assert backend["selected"] == "HIGHSPY_DIRECT_DUAL_RAY"
    assert backend["certificate_extraction_supported"] is True
    assert backend["priority"][0] == "highspy.getDualRay"
    assert backend["gurobi_permitted"] is False


def numeric_model(**updates):
    value = {
        "column_count": 2, "row_count": 1,
        "objective": [0.0, 0.0],
        "column_lower": [-float("inf"), -float("inf")],
        "column_upper": [float("inf"), float("inf")],
        "row_lower": [-float("inf")], "row_upper": [1.0],
        "starts": [0, 2], "indices": [0, 1], "values": [1.0, 2.0],
    }
    value.update(updates)
    return value


def diagnose(**updates):
    return ORACLE.diagnose_highs_arrays(**numeric_model(**updates))


def test_nan_coefficient_classified_and_located():
    result = diagnose(values=[float("nan"), 2.0])
    assert result["preflight_failure_classification"] == \
        "NONFINITE_CANONICAL_COEFFICIENT"
    assert result["numeric_fields"]["matrix_coefficients"]["nan_count"] == 1
    assert result["numeric_fields"]["matrix_coefficients"][
        "first_nonfinite"]["offset"] == 0


def test_infinite_improper_coefficient_classified():
    result = diagnose(values=[float("inf"), 2.0])
    assert result["preflight_failure_classification"] == \
        "NONFINITE_CANONICAL_COEFFICIENT"
    assert result["numeric_fields"]["matrix_coefficients"][
        "positive_infinity_count"] == 1


def test_nonfinite_objective_and_improper_bound_are_classified():
    objective = diagnose(objective=[float("nan"), 0.0])
    assert objective["preflight_failure_classification"] == \
        "NONFINITE_CANONICAL_COEFFICIENT"
    bound = diagnose(column_lower=[float("nan"), -float("inf")])
    assert bound["preflight_failure_classification"] == \
        "NONFINITE_CANONICAL_BOUND"
    assert bound["first_improper_nonfinite_bound"]["field"] == \
        "column_lower"


def test_highs_infinity_sentinel_is_not_a_user_bound_magnitude_failure():
    model = numeric_model(
        column_lower=[-1e30, -1e30], column_upper=[1e30, 1e30],
        row_lower=[-1e30], row_upper=[1.0])
    model["finite_bound_masks"] = (
        [False, False], [False, False], [False], [True])
    result = ORACLE.diagnose_highs_arrays(**model)
    assert result["over_infinite_bound_magnitude_count"] == 0
    assert result["preflight_failure_classification"] is None


def test_invalid_matrix_index_classified_and_located():
    result = diagnose(indices=[0, 2])
    assert result["preflight_failure_classification"] == \
        "HIGHS_MATRIX_INDEX_INVALID"
    assert result["first_outside_index"] == {"offset": 1, "value": 2}


def test_malformed_sparse_pointer_classified():
    result = diagnose(starts=[0, 1])
    assert result["preflight_failure_classification"] == \
        "HIGHS_MATRIX_STRUCTURE_INVALID"
    assert result["pointer_structure_valid"] is False
    assert result["pointer_terminal_nnz"] == 1


@pytest.mark.parametrize("indices", [[0, 0], [1, 0]])
def test_duplicate_or_unsorted_indices_classified(indices):
    result = diagnose(indices=indices)
    assert result["preflight_failure_classification"] == \
        "HIGHS_MATRIX_STRUCTURE_INVALID"
    assert (result["duplicate_column_index_count"]
            + result["unsorted_index_row_count"]) > 0


@pytest.mark.parametrize("coefficient", [1e-12, 1e16])
def test_extreme_coefficient_classified_with_distribution(coefficient):
    result = diagnose(values=[coefficient, 2.0])
    assert result["preflight_failure_classification"] == \
        "HIGHS_COEFFICIENT_MAGNITUDE_REJECTED"
    assert result["minimum_nonzero_absolute_finite_coefficient"] == \
        min(abs(coefficient), 2.0)


def test_highspy_api_rejection_classification_is_persisted(tmp_path):
    diagnostic = diagnose(values=[1e-12, 2.0])
    log = tmp_path / "highs.log"
    log.write_text("WARNING: ignored tiny coefficient\n")
    with pytest.raises(ORACLE.HighsCanonicalLPDiagnosticError) as captured:
        ORACLE._check_highs_status(
            "passModel", "HighsStatus.kWarning", "HighsStatus.kOk",
            diagnostic, log)
    output = tmp_path / "diagnostic.json"
    record = ORACLE.persist_highs_rejection(output, captured.value)
    assert record["diagnostic"]["failure_classification"] == \
        "HIGHS_COEFFICIENT_MAGNITUDE_REJECTED"
    assert record["diagnostic"]["failed_api_call"] == "passModel"
    assert "ignored tiny coefficient" in record["diagnostic"]["highs_log_text"]
    assert ORACLE.cluster_common.verified_json(output) == record


def tiny_lp(rows=None):
    return ORACLE.ExactCanonicalLP(
        ("xi[0]",), (F(0),), (F(1),),
        tuple(rows or (ORACLE.ExactLPRow(
            "tiny[0]", (0,),
            (F.from_float(4.113201943316126e-14),), None, F(1)),)))


def scale(lp, **kwargs):
    return ORACLE.build_highs_row_scaled_lp(
        lp, small_matrix_value=1e-9, large_matrix_value=1e15,
        infinite_bound=1e20, **kwargs)


@pytest.mark.parametrize("lower,upper", [
    (None, F(3)), (F(-2), None), (F(1), F(1))])
def test_positive_row_scaling_preserves_all_row_senses(lower, upper):
    lp = ORACLE.ExactCanonicalLP(
        ("xi[0]",), (None,), (None,),
        (ORACLE.ExactLPRow("sense[0]", (0,), (F(1, 10 ** 12),),
                           lower, upper),))
    result = scale(lp)
    original, scaled = lp.rows[0], result.lp.rows[0]
    factor = result.scales[0]
    assert scaled.coefficients == tuple(value * factor for value in
                                        original.coefficients)
    assert scaled.lower == (None if lower is None else lower * factor)
    assert scaled.upper == (None if upper is None else upper * factor)
    assert (scaled.lower is None) == (original.lower is None)
    assert (scaled.upper is None) == (original.upper is None)


def test_tiny_exact_coefficient_survives_real_highspy_export(tmp_path):
    lp = tiny_lp()
    original_identity = lp.identity()
    result = ORACLE.solve_highspy(
        lp, tmp_path / "highs.log", tmp_path / "scaling.json")
    report = result["solver_scaling"]
    diagnostic = result["construction_diagnostic"]
    assert report["sub_threshold_entries_before"] == 1
    assert report["sub_threshold_entries_after"] == 0
    assert diagnostic["original_diagnostic"][
        "sub_small_matrix_value_count"] == 1
    assert diagnostic["scaled_diagnostic"][
        "sub_small_matrix_value_count"] == 0
    assert diagnostic["sub_small_matrix_value_count"] == 0
    assert report["minimum_post_scale_nonzero_abs"] > 1e-9
    assert lp.identity() == original_identity
    assert ORACLE.cluster_common.verified_json(
        tmp_path / "scaling.json")["row_scaling_sha256"] == \
        report["row_scaling_sha256"]


def test_pre_passmodel_gate_rejects_deliberate_scaling_bypass():
    lp = tiny_lp()
    arrays = ORACLE._highs_numeric_arrays(lp, 1e30)
    thresholds = {"small_matrix_value": 1e-9,
                  "large_matrix_value": 1e15, "infinite_bound": 1e20}
    original = ORACLE._diagnose_highs_lp(lp, arrays, thresholds)
    with pytest.raises(ORACLE.HighsCanonicalLPDiagnosticError) as captured:
        ORACLE._require_solver_scaling_applied(
            lp, lp, None, original, original, None)
    assert captured.value.diagnostic["failure_classification"] == \
        "HIGHS_SOLVER_SCALING_NOT_APPLIED"
    assert captured.value.diagnostic["failed_api_call"] == \
        "pre_passModel_scaling_gate"


def test_all_128_style_tiny_coefficients_survive_scaling():
    rows = tuple(ORACLE.ExactLPRow(
        f"preactivation[{index}]", (0,),
        (F(1 + index, 10 ** 14),), None, F(1)) for index in range(128))
    result = scale(tiny_lp(rows))
    assert result.report["sub_threshold_entries_before"] == 128
    assert result.report["sub_threshold_entries_after"] == 0
    assert result.report["sub_threshold_distribution_by_row_family"] == {
        "preactivation": 128}
    assert result.report["sub_threshold_distribution_by_variable_family"] == {
        "xi": 128}


def test_minimal_deterministic_power_of_two_is_selected():
    result = scale(tiny_lp())
    exponent = result.exponents[0]
    original = abs(tiny_lp().rows[0].coefficients[0])
    target = F.from_float(1e-8)
    assert original * (2 ** exponent) >= target
    assert exponent > 0
    assert original * (2 ** (exponent - 1)) < target
    assert scale(tiny_lp()).exponents == result.exponents


def test_reported_real_minimum_requires_scale_exponent_18():
    coefficient = F.from_float(4.113201943316126e-14)
    lp = ORACLE.ExactCanonicalLP(
        ("xi[0]",), (None,), (None,),
        (ORACLE.ExactLPRow(
            "centered[0]", (0,), (coefficient,), None, F(1)),))
    result = scale(lp)
    assert result.exponents == (18,)
    assert float(coefficient * result.scales[0]) >= 1e-8
    assert float(coefficient * (result.scales[0] // 2)) < 1e-8


def test_unsafe_overflow_row_scaling_fails_closed():
    lp = ORACLE.ExactCanonicalLP(
        ("xi[0]", "c[0]"), (None, None), (None, None),
        (ORACLE.ExactLPRow(
            "mixed[0]", (0, 1), (F(1, 10 ** 14), F(10 ** 14)),
            None, F(1)),))
    with pytest.raises(ORACLE.HighsCanonicalLPDiagnosticError) as captured:
        scale(lp)
    assert captured.value.diagnostic["failure_classification"] == \
        "HIGHS_SAFE_ROW_SCALING_IMPOSSIBLE"


def test_solver_scaling_preserves_original_identity_and_has_own_identity():
    lp = tiny_lp()
    original = lp.identity()
    result = scale(lp)
    assert lp.identity() == original
    assert result.report["original_canonical_lp_sha256"] == original
    assert result.report["solver_scaled_lp_sha256"] == result.lp.identity()
    assert result.lp.identity() != original
    assert result.report["row_scaling_sha256"] == ORACLE._sha_json(
        result.report["row_scales_exact"])


def test_exact_dual_ray_back_mapping_multiplies_scale_once():
    assert ORACLE.map_solver_row_multipliers_exact(
        (F(3, 5), F(7, 11)), (8, 4)) == (F(24, 5), F(28, 11))


def scaled_infeasible_lp():
    # The tiny row is x <= 0; x >= 100 makes the scaled violation exceed
    # HiGHS' feasibility tolerance while retaining a sub-threshold source row.
    return ORACLE.ExactCanonicalLP(
        ("xi[0]",), (F(100),), (None,),
        (ORACLE.ExactLPRow(
            "tiny_upper[0]", (0,), (F(1, 10 ** 12),),
            None, F(0)),))


def certificate_from_row_multiplier(lp, multiplier):
    return {
        "schema": "CORET_EXACT_LP_FARKAS_CERTIFICATE_V1",
        "canonical_lp_sha256": lp.identity(),
        "multipliers": [
            {"kind": "row", "index": 0, "orientation": 1,
             "multiplier": ORACLE._fs(multiplier)},
            {"kind": "column", "index": 0, "orientation": -1,
             "multiplier": ORACLE._fs(multiplier * F(1, 10 ** 12))},
        ],
    }


def test_scale_omission_and_double_scaling_reject_but_once_accepts():
    lp = scaled_infeasible_lp()
    factor = scale(lp).scales[0]
    solver_mu = F(1, factor)
    mapped = ORACLE.map_solver_row_multipliers_exact(
        (solver_mu,), (factor,))[0]
    assert ORACLE.verify_exact_lp_farkas(
        lp, certificate_from_row_multiplier(lp, mapped))["verified"]
    # Keeping the original bound multiplier exposes omission/double scaling.
    for wrong in (solver_mu, solver_mu * factor * factor):
        certificate = certificate_from_row_multiplier(lp, mapped)
        certificate["multipliers"][0]["multiplier"] = ORACLE._fs(wrong)
        with pytest.raises(RuntimeError, match="stationarity"):
            ORACLE.verify_exact_lp_farkas(lp, certificate)


def test_scaled_synthetic_infeasible_ray_exact_replays_original(tmp_path):
    lp = scaled_infeasible_lp()
    result = ORACLE.solve_highspy(lp, tmp_path / "highs.log")
    assert result["infeasible"] and result["direct_dual_ray_available"]
    certificate, _attempts, status = ORACLE.repair_direct_dual_ray(
        lp, result["original_row_dual_ray"])
    assert status == "EXACT_FARKAS_VERIFIED"
    assert ORACLE.verify_exact_lp_farkas(lp, certificate)["verified"]


def test_highs_warning_about_ignored_coefficient_remains_fatal(tmp_path):
    diagnostic = diagnose(values=[1e-12, 2.0])
    with pytest.raises(ORACLE.HighsCanonicalLPDiagnosticError) as captured:
        ORACLE._check_highs_status(
            "passModel", "HighsStatus.kWarning", "HighsStatus.kOk",
            diagnostic, tmp_path / "absent.log")
    assert captured.value.diagnostic["failure_classification"] == \
        "HIGHS_COEFFICIENT_MAGNITUDE_REJECTED"


def continuation_fixture():
    source_count = 1
    solution = np.zeros(source_count + 3 * ORACLE.DIMENSION + 1)
    g0 = source_count + ORACLE.DIMENSION + 1
    u0 = g0 + ORACLE.DIMENSION
    solution[g0] = 0.0       # deterministic active tie
    solution[u0] = 0.25      # largest relaxation violation
    solution[g0 + 1] = -0.5
    solution[u0 + 1] = 0.0
    bounds = {
        "stable_active": [],
        "stable_inactive": list(range(2, ORACLE.DIMENSION)),
        "unstable": [0, 1],
    }
    return source_count, solution, bounds


def test_feasible_root_attempts_exact_witness_before_branching():
    source_count, solution, bounds = continuation_fixture()
    calls = []

    def witness(pattern, candidate, node_id):
        calls.append((pattern, node_id))
        return {"verified": True}, {"attempted": True, "verified": True}

    found, branch = ORACLE.run_phase_continuation(
        solution, bounds, source_count, 8, 4,
        time.perf_counter() + 2.0,
        lambda *_: pytest.fail("root witness should precede node solves"),
        witness)
    assert found == {"verified": True}
    assert calls[0][1] == "root"
    assert calls[0][0][0] is True  # g == 0 tie is active
    assert branch["nodes_created"] == 1
    assert branch["phase_patterns_attempted"] == 1


def test_failed_root_witness_initializes_phase_bab():
    source_count, solution, bounds = continuation_fixture()
    calls = []

    def witness(pattern, candidate, node_id):
        calls.append(node_id)
        return None, {"attempted": True, "verified": False,
                      "failure": "synthetic exact replay failure"}

    def solve_node(_phases, _node_id):
        return {"solver_status": "Optimal", "feasible": True,
                "infeasible": False, "solution": solution,
                "certificate": None}

    found, branch = ORACLE.run_phase_continuation(
        solution, bounds, source_count, 3, 4,
        time.perf_counter() + 2.0, solve_node, witness)
    assert found is None
    assert calls[0] == "root"
    assert branch["attempted"] is True
    assert branch["nodes_created"] >= 1
    assert branch["phase_patterns_attempted"] > 0
    assert branch["limit_reason"] == "MAXIMUM_NODES"


def test_feasible_root_zero_continuation_state_is_forbidden_by_flow():
    source_count, solution, bounds = continuation_fixture()
    found, branch = ORACLE.run_phase_continuation(
        solution, bounds, source_count, 1, 1,
        time.perf_counter() + 2.0,
        lambda *_: pytest.fail("no child solve expected"),
        lambda *_: (None, {"attempted": True, "verified": False}))
    assert found is None
    assert not (branch["nodes_created"] == 0
                and branch["phase_patterns_attempted"] == 0)


def test_exact_positive_dyadic_scale_bound_without_float_sqrt(monkeypatch):
    monkeypatch.setattr(ORACLE.math, "sqrt", lambda *_: pytest.fail("float sqrt"))
    epsilon = F.from_float(1e-12)
    lower, proof = ORACLE.exact_dyadic_sqrt_lower(epsilon)
    assert lower > 0
    assert lower * lower <= epsilon < (lower + F(1, 2 ** 64)) ** 2
    assert lower == F(18446744073709, 2 ** 64)
    assert proof["epsilon_exact"] == ORACLE._fs(epsilon)
    assert proof["t_lower_bound_denominator_bits"] == 64
    assert proof["proof_check_L_squared_le_epsilon"]


@pytest.mark.parametrize("epsilon", [F(0), F(-1), F(1, 2 ** 200)])
def test_uncertifiable_positive_scale_bound_fails_closed(epsilon):
    with pytest.raises(RuntimeError):
        ORACLE.exact_dyadic_sqrt_lower(epsilon)


def test_every_exact_layernorm_witness_has_implied_lower_bound():
    fixture = replace(problem(), epsilon=F(3))
    assert ORACLE.replay_exact_perspective_witness(
        fixture, witness(t="2/1"))["verified"]
    lower, _proof = ORACLE.exact_dyadic_sqrt_lower(fixture.epsilon)
    assert F(2) >= lower


def test_canonical_and_scaled_lp_exclude_zero_scale():
    d = ORACLE.DIMENSION
    zero = np.zeros(d)
    matrix = np.zeros((d, d))
    bounds = ORACLE._derive_exact_bounds(
        zero, np.zeros((1, d)), [F(-1)], [F(1)], np.ones(d), zero,
        matrix, zero, F(1, 4))
    lp = ORACLE.build_exact_perspective_lp(
        [F(-1)], [F(1)], np.ones(d), zero, matrix, zero, matrix, zero, bounds)
    t = lp.variable_names.index("t")
    assert lp.column_lower[t] == F(1, 2) > 0
    scaled = scale(lp)
    assert scaled.lp.column_lower[t] == lp.column_lower[t]
    assert scaled.lp.column_upper[t] == lp.column_upper[t]


@pytest.mark.parametrize("conditions,lower,upper", [
    ([("source", F(1, 2), F(-2), F(-1), F(1))], F(-1, 4), F(3, 4)),
    ([("t", F(0), F(1), F(1, 2), F(2))], F(1, 2), F(2)),
    ([("active", F(-1), F(2), F(0), None)], F(1, 2), None),
    ([("inactive", F(1), F(-2), None, F(0))], F(1, 2), None),
    ([("source", F(0), F(1), F(-1), F(1)),
      ("t", F(0), F(1), F(1, 2), F(2)),
      ("inactive", F(-3, 4), F(1), None, F(0))], F(1, 2), F(3, 4)),
])
def test_exact_source_scale_relu_interval_intersections(conditions, lower, upper):
    interval = ORACLE.exact_affine_parameter_interval(conditions)
    assert not interval["alpha_interval_empty"]
    assert interval["alpha_interval_lower"] == (ORACLE._fs(lower)
                                                if lower is not None else None)
    assert interval["alpha_interval_upper"] == (ORACLE._fs(upper)
                                                if upper is not None else None)


def test_exact_interval_empty_and_endpoint_constraint_reporting():
    interval = ORACLE.exact_affine_parameter_interval([
        ("box", F(0), F(1), F(-1), F(1)),
        ("phase", F(-2), F(1), F(0), None)])
    assert interval["alpha_interval_empty"]
    assert interval["active_constraint_at_lower"] == ["phase:lower"]
    assert interval["active_constraint_at_upper"] == ["box:upper"]


def test_empty_family_stops_before_polynomial_construction(monkeypatch):
    fixture = replace(problem(), epsilon=F(1))
    monkeypatch.setattr(ORACLE, "_fraction_square_root",
                        lambda *_: pytest.fail("polynomial root work"))
    found, evidence = ORACLE._replay_affine_family(
        fixture, [True, False], [F(2)], [F(0)], [F(1), F(-1)],
        [F(0), F(0)], F(0), F(1), F(2))
    assert found is None and evidence["alpha_interval_empty"]
    assert not evidence["polynomial_constructed"]


def test_outside_roots_are_not_exact_replayed(monkeypatch):
    fixture = replace(problem(), epsilon=F(1))
    monkeypatch.setattr(ORACLE, "replay_exact_perspective_witness",
                        lambda *_: pytest.fail("outside root replayed"))
    monkeypatch.setattr(ORACLE, "replay_algebraic_perspective_witness",
                        lambda *_: pytest.fail("outside root replayed"))
    monkeypatch.setattr(ORACLE, "_isolate_quadratic_root",
                        lambda *_: pytest.fail("outside root isolated"))
    found, evidence = ORACLE._replay_affine_family(
        fixture, [True, False], [F(0)], [F(0)], [F(1), F(-1)],
        [F(0), F(0)], F(0), F(1), F(1))
    assert found is None
    assert evidence["roots_total"] == 2
    assert evidence["roots_inside_interval"] == 0


def test_inside_algebraic_root_exact_replays():
    fixture = replace(problem(), epsilon=F(1))
    found, evidence = ORACLE._replay_affine_family(
        fixture, [True, False], [F(0)], [F(0)], [F(1), F(-1)],
        [F(0), F(0)], F(0), F(1), F(2))
    assert found is not None
    assert evidence["roots_inside_interval"] == 1
    assert evidence["exact_replay_result"]["verified"]
    assert ORACLE.replay_algebraic_perspective_witness(fixture, found)["verified"]


def test_slack_aware_basis_order_is_deterministic():
    sensitivity = np.asarray([[1.0, 1.0, 1.0]])
    arguments = (sensitivity, [0.99, 0.0, 0.8], [F(-1)] * 3, [F(1)] * 3)
    first, _slack = ORACLE.box_aware_correction_bases(*arguments)
    second, _slack = ORACLE.box_aware_correction_bases(*arguments)
    assert first == second
    assert first[0]["columns"] == (1,)
    assert first[0]["minimum_selected_source_slack"] == 1.0


def test_rank_deficient_basis_rejected_before_bareiss(monkeypatch):
    monkeypatch.setattr(ORACLE, "_bareiss_solve", lambda *_: pytest.fail("Bareiss"))
    with pytest.raises(ORACLE.ExactSolveFailure, match="rank deficient"):
        ORACLE.box_aware_correction_bases(
            np.zeros((2, 3)), [0.0] * 3, [F(-1)] * 3, [F(1)] * 3)


def test_alternate_basis_succeeds_after_out_of_box_family(monkeypatch):
    fixture = ORACLE.ExactPerspectiveProblem(
        x0=(F(2), F(-2)), X=((F(1), F(-1)), (F(4), F(-4))),
        low=(F(-1), F(-1)), high=(F(1), F(1)),
        gamma=(F(1), F(1)), beta=(F(0), F(0)), epsilon=F(1),
        W1=((F(1), F(0)), (F(0), F(1))), b1=(F(0), F(0)),
        W2=((F(0), F(0)), (F(0), F(0))), b2=(F(0), F(0)))
    bases = [{"columns": (i,), "basis_sha256": ORACLE._sha_json((i,)),
              "minimum_selected_source_slack": 1.0, "numerical_rank": 1}
             for i in (0, 1)]
    monkeypatch.setattr(ORACLE, "box_aware_correction_bases",
                        lambda *_: (bases, np.ones(2)))
    found, evidence = ORACLE.reconstruct_exact_fixed_pattern_witness(
        fixture, np.zeros(5), [True, False], 2, 2.0)
    assert found is not None
    assert evidence["families"][0]["alpha_interval_empty"]
    assert evidence["families"][-1]["basis_sha256"] == bases[1]["basis_sha256"]
    assert ORACLE.replay_exact_perspective_witness(fixture, found)["verified"]


def test_failed_portfolio_does_not_prove_infeasibility(monkeypatch):
    fixture = replace(problem(), X=((F(1), F(-1)),), epsilon=F(1),
                      W2=((F(0), F(0)), (F(0), F(0))))
    monkeypatch.setattr(ORACLE, "_replay_affine_family",
                        lambda *_: (None, {
                            "alpha_interval_lower": "0/1",
                            "alpha_interval_upper": "1/1",
                            "alpha_interval_empty": False,
                            "active_constraint_at_lower": [],
                            "active_constraint_at_upper": []}))
    found, evidence = ORACLE.reconstruct_exact_fixed_pattern_witness(
        fixture, np.zeros(4), [True, False], 1, 2.0)
    assert found is None and not evidence["permits_infeasibility_claim"]
    assert ORACLE.scientific_status_from_proof(
        exact_witness_verified=False, root_certificate_verified=False,
        complete_tree_verified=False, open_nodes=1) == ORACLE.INCONCLUSIVE


@pytest.mark.parametrize("sign,lower,upper,inside", [
    (-1, F(-2), F(-1), True), (-1, F(0), F(2), False),
    (1, F(1), F(2), True), (1, F(-2), F(-1), False),
    (1, F(0), F(1), False), (-1, F(-1), F(0), False),
])
def test_irrational_root_filter_uses_exact_endpoint_comparisons(sign, lower, upper, inside):
    interval = ORACLE.exact_affine_parameter_interval([
        ("alpha", F(0), F(1), lower, upper)])
    assert ORACLE._irrational_quadratic_branch_inside(
        F(1), F(0), F(8), sign, interval) is inside


def test_family_alarm_interrupts_and_restores_handler():
    import signal
    previous = signal.getsignal(signal.SIGALRM)
    restore = ORACLE._start_reconstruction_alarm(0.01)
    try:
        with pytest.raises(ORACLE.ExactSolveFailure, match="family deadline"):
            time.sleep(0.1)
    finally:
        restore()
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0


def test_nonbasic_source_direction_is_attempted(monkeypatch):
    fixture = replace(problem(), X=((F(1), F(-1)), (F(2), F(-2))),
                      low=(F(-1), F(-1)), high=(F(1), F(1)), epsilon=F(1),
                      W2=((F(0), F(0)), (F(0), F(0))))
    monkeypatch.setattr(ORACLE, "_replay_affine_family", lambda *_: (None, {
        "alpha_interval_lower": "0/1", "alpha_interval_upper": "1/1",
        "alpha_interval_empty": False, "active_constraint_at_lower": [],
        "active_constraint_at_upper": []}))
    found, evidence = ORACLE.reconstruct_exact_fixed_pattern_witness(
        fixture, np.asarray([0.0, 0.0, 0.0, 0.0, 1.0]), [True, False], 2, 2.0)
    assert found is None
    assert any(row["free_parameter"].startswith("source_delta[")
               for row in evidence["families"])
    assert len(evidence["families"]) <= 8


def test_existing_linear_node_rows_intersect_family_before_root_work(monkeypatch):
    fixture = replace(problem(), epsilon=F(1))
    node = ORACLE.ExactCanonicalLP(
        tuple(str(i) for i in range(8)), (None,) * 8, (None,) * 8,
        (ORACLE.ExactLPRow("extra_t_cut", (3,), (F(1),), None, F(1)),))
    monkeypatch.setattr(ORACLE, "_isolate_quadratic_root",
                        lambda *_: pytest.fail("root outside node interval"))
    found, evidence = ORACLE._replay_affine_family(
        fixture, [True, False], [F(0)], [F(0)], [F(1), F(-1)],
        [F(0), F(0)], F(0), F(1), F(2), node)
    assert found is None and evidence["roots_inside_interval"] == 0
    assert "node:extra_t_cut:upper" in evidence["active_constraint_at_upper"]


def test_portfolio_attempts_at_most_eight_families(monkeypatch):
    n = 9
    fixture = replace(problem(), x0=(F(0), F(0)),
                      X=tuple((F(i + 1), F(-i - 1)) for i in range(n)),
                      low=(F(-1),) * n, high=(F(1),) * n, epsilon=F(1),
                      W2=((F(0), F(0)), (F(0), F(0))))
    bases = [{"columns": (i,), "basis_sha256": ORACLE._sha_json((i,)),
              "minimum_selected_source_slack": 1.0, "numerical_rank": 1}
             for i in range(8)]
    monkeypatch.setattr(ORACLE, "box_aware_correction_bases",
                        lambda *_: (bases, np.ones(n)))
    monkeypatch.setattr(ORACLE, "_replay_affine_family", lambda *_: (None, {
        "alpha_interval_lower": "0/1", "alpha_interval_upper": "1/1",
        "alpha_interval_empty": False, "active_constraint_at_lower": [],
        "active_constraint_at_upper": []}))
    candidate = np.zeros(n + 3)
    candidate[n + 2] = 1.0
    found, evidence = ORACLE.reconstruct_exact_fixed_pattern_witness(
        fixture, candidate, [True, False], n, 2.0)
    assert found is None
    assert len(evidence["families"]) == 8
    assert not evidence["permits_infeasibility_claim"]


def independent_fraction_solve(matrix, rhs):
    """Small test oracle: ordinary exact rational elimination, no producer code."""
    n = len(matrix)
    a = [[F(value) for value in row] + [F(value)] for row, value in zip(matrix, rhs)]
    for k in range(n):
        pivot = next(i for i in range(k, n) if a[i][k])
        a[k], a[pivot] = a[pivot], a[k]
        divisor = a[k][k]
        a[k] = [value / divisor for value in a[k]]
        for i in range(n):
            if i == k:
                continue
            multiplier = a[i][k]
            a[i] = [value - multiplier * other for value, other in zip(a[i], a[k])]
    return [row[-1] for row in a]


def test_exact_multi_rhs_matches_independent_fraction_solves():
    matrix = [[0, 2, 3], [4, 5, 6], [7, 8, 10]]
    columns = [[1, 2, 3], [9, -2, 0], [-1, 7, 11], [0, 0, 0]]
    solutions, report = ORACLE._exact_multi_rhs_solve(matrix, columns, 2.0)
    assert solutions == [independent_fraction_solve(matrix, rhs) for rhs in columns]
    assert report["elimination_count"] == 1
    assert report["selected_system_replay_verified"]


def test_identical_matrix_reuses_authenticated_elimination_for_new_rhs():
    matrix = [[0, 2, 3], [4, 5, 6], [7, 8, 10]]
    cache = {}
    _first, first = ORACLE._exact_multi_rhs_solve(matrix, [[1, 2, 3]], 2.0, cache)
    solved, second = ORACLE._exact_multi_rhs_solve(matrix, [[9, -2, 0], [4, 5, 6]], 2.0, cache)
    assert second["elimination_reused"] and second["elimination_count"] == 0
    assert second["factorization_sha256"] == first["factorization_sha256"]
    assert solved == [independent_fraction_solve(matrix, rhs) for rhs in ([9, -2, 0], [4, 5, 6])]


def test_corrupted_factorization_is_rejected():
    matrix = [[2, 1], [1, 2]]
    cache = {}
    ORACLE._exact_multi_rhs_solve(matrix, [[3, 3]], 2.0, cache)
    factor = next(iter(cache.values()))
    factor["upper"][0][0] += 1
    with pytest.raises(ORACLE.ExactSolveFailure, match="factorization identity"):
        ORACLE._exact_multi_rhs_solve(matrix, [[3, 3]], 2.0, cache)


def test_row_gcd_and_sign_normalization_preserves_exact_solutions():
    matrix = [[F(-2), F(-1)], [F(-2, 3), F(-4, 3)]]
    rhs = [[F(-3), F(-2)], [F(2, 3), F(1, 5)]]
    integers, columns, denominators, report = ORACLE._integerize_correction_system(matrix, rhs)
    assert integers == [[2, 1], [1, 2]]
    assert report["row_signs"] == [-1, -1]
    solved, _report = ORACLE._exact_multi_rhs_solve(integers, columns, 2.0)
    actual = [[value / divisor for value in solution]
              for solution, divisor in zip(solved, denominators)]
    assert actual == [independent_fraction_solve(matrix, column) for column in rhs]


def test_rhs_denominators_do_not_inflate_coefficient_matrix():
    matrix = [[F(3, 2), F(1, 4)], [F(1, 8), F(5, 16)]]
    rhs = [[F(1, 2 ** 100), F(3, 2 ** 120)], [F(2, 3), F(1, 7)]]
    integers, columns, denominators, profile = ORACLE._integerize_correction_system(matrix, rhs)
    assert integers == [[6, 1], [2, 5]]
    assert profile["integerized_matrix_maximum_bit_length"] == 3
    old_rows = [ORACLE._integerize([*row, rhs[0][i], rhs[1][i]])
                for i, row in enumerate(matrix)]
    assert max(abs(value).bit_length() for row in old_rows for value in row[:2]) >= 100
    solved, _report = ORACLE._exact_multi_rhs_solve(integers, columns, 2.0)
    actual = [[value / divisor for value in solution]
              for solution, divisor in zip(solved, denominators)]
    assert actual == [independent_fraction_solve(matrix, column) for column in rhs]


def test_exact_permutations_and_integer_growth_diagnostics_are_deterministic():
    matrix = [[0, 0, 100], [2, 500, 7], [13, 17, 19]]
    first, report = ORACLE._exact_multi_rhs_solve(matrix, [[1, 2, 3]], 2.0)
    second, repeat = ORACLE._exact_multi_rhs_solve(matrix, [[1, 2, 3]], 2.0)
    assert first == second == [independent_fraction_solve(matrix, [1, 2, 3])]
    assert report["row_permutation"] == repeat["row_permutation"]
    assert report["column_permutation"] == repeat["column_permutation"]
    assert report["factorization_sha256"] == repeat["factorization_sha256"]
    assert report["peak_intermediate_integer_bit_length"] > 0
    assert report["bareiss_elimination_seconds"] >= 0
    assert report["back_substitution_seconds"] >= 0


def test_positive_slack_basis_beats_large_boundary_columns():
    sensitivity = np.asarray([[1e9, 1.0, 0.0], [0.0, 0.0, 1.0]])
    bases, _slack = ORACLE.box_aware_correction_bases(
        sensitivity, [1.0, 0.0, 0.0], [F(-1)] * 3, [F(1)] * 3)
    assert all(set(basis["columns"]) == {1, 2} for basis in bases)
    audit = bases[0]["slack_diagnostics"]
    assert audit["source_variables_with_positive_slack"] == 2
    assert audit["positive_slack_full_rank_candidate_found"]
    assert audit["maximum_minimum_slack_among_deterministic_full_rank_candidates"] == 1.0
    assert all(count == 2 for count in audit["source_slack_threshold_counts"].values())


def test_unavoidable_boundary_source_is_reported_honestly():
    bases, _slack = ORACLE.box_aware_correction_bases(
        np.eye(2), [0.0, 1.0], [F(-1)] * 2, [F(1)] * 2)
    audit = bases[0]["slack_diagnostics"]
    assert bases[0]["minimum_selected_source_slack"] == 0.0
    assert audit["source_variables_with_positive_slack"] == 1
    assert audit["positive_slack_basis_classification"] == \
        "POSITIVE_SLACK_BASIS_IMPOSSIBLE_TOO_FEW_SOURCES"
    assert not audit["positive_slack_full_rank_candidate_found"]


def test_positive_pool_numerical_rank_failure_is_not_nonexistence_proof():
    bases, _slack = ORACLE.box_aware_correction_bases(
        np.asarray([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        [0.0, 0.0, 1.0], [F(-1)] * 3, [F(1)] * 3)
    audit = bases[0]["slack_diagnostics"]
    assert audit["positive_slack_pool_numerical_rank"] == 1
    assert audit["rank_screen_is_not_an_exact_nonexistence_proof"]
    assert audit["positive_slack_basis_classification"] == \
        "POSITIVE_SLACK_FULL_RANK_NOT_FOUND_BY_NUMERICAL_SCREEN"


def test_multiple_free_directions_share_one_elimination_and_reach_interval(monkeypatch):
    fixture = replace(problem(), X=((F(1), F(-1)), (F(2), F(-2))),
                      low=(F(-1), F(-1)), high=(F(1), F(1)), epsilon=F(1),
                      W2=((F(0), F(0)), (F(0), F(0))))
    reached = []
    original = ORACLE._replay_affine_family
    def replay(*args):
        _found, evidence = original(*args)
        reached.append(evidence)
        return None, evidence
    monkeypatch.setattr(ORACLE, "_replay_affine_family", replay)
    found, report = ORACLE.reconstruct_exact_fixed_pattern_witness(
        fixture, np.asarray([0.0, 0.0, 0.0, 0.0, 1.0]), [True, False], 2, 2.0)
    assert found is None and len(reached) >= 2
    assert all("alpha_interval_empty" in evidence for evidence in reached)
    systems = report["linear_systems"]
    assert all(row["rhs_count"] == 4 for row in systems)
    assert sum(row["elimination_count"] for row in systems) == 1
    assert all(row["original_rational_system_replay_verified"] for row in systems)
    assert not report["permits_infeasibility_claim"]


def test_integer_multi_rhs_singular_and_timeout_fail_closed():
    with pytest.raises(ORACLE.ExactSolveFailure, match="singular"):
        ORACLE._exact_multi_rhs_solve([[1, 2], [2, 4]], [[3, 6]], 2.0)
    with pytest.raises(ORACLE.ExactSolveFailure, match="timeout"):
        ORACLE._exact_multi_rhs_solve([[2, 1], [1, 2]], [[3, 3]], 0.0)


def test_dense_127_multi_rhs_exact_fixture_under_bounded_budget():
    import random
    random_source = random.Random(3011)
    n = 127
    matrix = [[random_source.randrange(-128, 128) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 65536 + i
    expected = [[(j + k) % 5 - 2 for j in range(n)] for k in range(4)]
    rhs = [[sum(value * x for value, x in zip(row, solution)) for row in matrix]
           for solution in expected]
    started = time.perf_counter()
    solved, report = ORACLE._exact_multi_rhs_solve(matrix, rhs, 18.0)
    elapsed = time.perf_counter() - started
    assert solved == expected
    assert report["elimination_count"] == 1 and report["rhs_count"] == 4
    assert report["peak_intermediate_integer_bit_length"] > 16
    print(json.dumps({"synthetic_dense_127_seconds": elapsed, **report}), flush=True)


def test_exact_column_scaling_reduces_growth_and_restores_original_variables():
    matrix = [[F(6 * 101), F(3 * 103)], [F(2 * 101), F(5 * 103)]]
    rhs = [[F(1, 2 ** 53), F(3, 2 ** 52)], [F(5, 7), F(-2, 3)]]
    report = {}
    solved = ORACLE._solve_exact_correction_rhs(matrix, rhs, 2.0, {}, report)
    assert solved == [independent_fraction_solve(matrix, column) for column in rhs]
    assert report["column_primitive_matrix_maximum_bit_length"] == 3
    assert min(report["column_gcd_bits_removed"]) >= 6
    assert report["original_rational_system_replay_verified"]


def test_corrupted_linear_answer_is_rejected_by_original_equation_replay(monkeypatch):
    matrix = [[F(2), F(1)], [F(1), F(2)]]
    monkeypatch.setattr(ORACLE, "_exact_multi_rhs_solve",
                        lambda *_: ([[F(0), F(0)]], {}))
    with pytest.raises(ORACLE.ExactSolveFailure, match="rational correction system replay"):
        ORACLE._solve_exact_correction_rhs(matrix, [[F(3), F(3)]], 2.0, {}, {})


def test_float_linear_solve_is_not_accepted_as_integer_evidence():
    with pytest.raises(ORACLE.ExactSolveFailure, match="type differs"):
        ORACLE._exact_multi_rhs_solve([[2.0, 1], [1, 2]], [[3, 3]], 2.0)


def test_127_correction_fixture_reaches_interval_and_exact_witness():
    d, n = 128, 127
    zeros = (F(0),) * d
    sources = tuple(tuple(F(int(j == i) - int(j == d - 1))
                          for j in range(d)) for i in range(n))
    fixture = ORACLE.ExactPerspectiveProblem(
        x0=zeros, X=sources, low=(F(-1),) * n, high=(F(1),) * n,
        gamma=(F(1),) * d, beta=zeros, epsilon=F(1),
        W1=(zeros,) * d, b1=zeros, W2=(zeros,) * d, b2=zeros)
    candidate = np.zeros(n + d + 1)
    candidate[-1] = 1.0
    found, report = ORACLE.reconstruct_exact_fixed_pattern_witness(
        fixture, candidate, [False] * d, n, 15.0)
    assert found is not None and report["verified"]
    assert report["linear_systems"][0]["dimension"] == 127
    assert report["linear_systems"][0]["elimination_count"] == 1
    assert report["families"][-1]["alpha_interval"] is not None
    assert not report["families"][-1]["alpha_interval_empty"]
    assert ORACLE.replay_exact_perspective_witness(fixture, found)["verified"]


def _production_two_direction_fixture():
    d, n = 128, 128
    zeros = (F(0),) * d
    sources = tuple(tuple(F(int(j == i % 127) - int(j == 127))
                          for j in range(d)) for i in range(n))
    fixture = ORACLE.ExactPerspectiveProblem(
        x0=zeros, X=sources, low=(F(-1),) * n, high=(F(1),) * n,
        gamma=(F(1),) * d, beta=zeros, epsilon=F(1),
        W1=(zeros,) * d, b1=zeros, W2=(zeros,) * d, b2=zeros)
    columns = tuple(range(1, 128))
    basis = {"columns": columns, "basis_sha256": ORACLE._sha_json(columns),
             "minimum_selected_source_slack": 1.0, "numerical_rank": 127}
    candidate = np.zeros(n + d + 1)
    candidate[-1] = 1.0
    return fixture, basis, candidate


def test_production_127_basis_two_directions_one_setup_solve_and_interval(monkeypatch):
    fixture, basis, candidate = _production_two_direction_fixture()
    # Duplicate portfolio entries cannot restart either setup or elimination.
    monkeypatch.setattr(ORACLE, "box_aware_correction_bases",
                        lambda *_: ([basis, dict(basis)], np.ones(128)))
    counts = {"matrix": 0, "integerization": 0, "elimination": 0}
    for name, counter in (("_build_selected_exact_matrix", "matrix"),
                          ("_integerize_correction_system", "integerization"),
                          ("_exact_multi_rhs_solve", "elimination")):
        original = getattr(ORACLE, name)
        def counted(*args, _original=original, _counter=counter, **kwargs):
            counts[_counter] += 1
            assert counts[_counter] == 1, "same basis repeated exact work"
            return _original(*args, **kwargs)
        monkeypatch.setattr(ORACLE, name, counted)
    replay = ORACLE._replay_affine_family
    reached = []
    def collect(*args):
        _found, evidence = replay(*args)
        reached.append(evidence)
        return None, evidence
    monkeypatch.setattr(ORACLE, "_replay_affine_family", collect)
    started = time.perf_counter()
    found, report = ORACLE.reconstruct_exact_fixed_pattern_witness(
        fixture, candidate, [False] * 128, 128, 15.0)
    elapsed = time.perf_counter() - started
    assert found is None and counts == {"matrix": 1, "integerization": 1, "elimination": 1}
    assert len(reached) == len(report["families"]) == 2
    assert all(row["alpha_interval"] is not None for row in report["families"])
    system, = report["linear_systems"]
    assert system["rhs_count"] == 4 and system["dimension"] == 127
    assert system["shared_free_directions"] == ["t", "source_delta[0]"]
    assert system["matrix_assembly_count"] == system["integerization_count"] == 1
    assert system["elimination_count"] == system["basis_solve_count"] == 1
    first, second = report["families"]
    assert first["basis_solve_id"] == second["basis_solve_id"]
    assert not first["elimination_reused"] and second["elimination_reused"]
    assert first["elimination_count"] == 1 and second["elimination_count"] == 0
    assert all(row["solved_affine_data_reused"] for row in report["families"])
    assert all(row["exact_replay_result"]["verified"] for row in report["families"])
    assert all(row["interval_seconds"] >= 0 and row["polynomial_seconds"] >= 0
               and row["replay_seconds"] >= 0 for row in report["families"])
    assert not report["permits_infeasibility_claim"]
    print(json.dumps({"production_127_two_direction_fixture_seconds": elapsed,
                      "counts": counts, "basis_profile": system}), flush=True)


def test_near_old_timeout_setup_and_solve_have_separate_budgets(monkeypatch):
    fixture = replace(problem(), X=((F(1), F(-1)), (F(2), F(-2))),
                      low=(F(-1), F(-1)), high=(F(1), F(1)), epsilon=F(1),
                      W2=((F(0), F(0)), (F(0), F(0))))
    columns = (1,)
    basis = {"columns": columns, "basis_sha256": ORACLE._sha_json(columns),
             "minimum_selected_source_slack": 1.0, "numerical_rank": 1}
    monkeypatch.setattr(ORACLE, "box_aware_correction_bases",
                        lambda *_: ([basis], np.ones(2)))
    real_clock = time.perf_counter
    offset = [0.0]
    monkeypatch.setattr(ORACLE.time, "perf_counter", lambda: real_clock() + offset[0])
    alarms = []
    monkeypatch.setattr(ORACLE, "_start_reconstruction_alarm",
                        lambda seconds: (alarms.append(seconds) or (lambda: None)))
    build, solve = ORACLE._build_selected_exact_matrix, ORACLE._solve_exact_correction_rhs
    counts = {"build": 0, "solve": 0}
    def slow_build(*args):
        counts["build"] += 1
        result = build(*args)
        offset[0] += 17.9  # Simulated job3012 assembly, no sleeping.
        return result
    def slow_solve(*args):
        counts["solve"] += 1
        assert args[2] > 29.0  # Setup did not consume the solve budget.
        result = solve(*args)
        offset[0] += 11.64
        return result
    monkeypatch.setattr(ORACLE, "_build_selected_exact_matrix", slow_build)
    monkeypatch.setattr(ORACLE, "_solve_exact_correction_rhs", slow_solve)
    replay = ORACLE._replay_affine_family
    def collect(*args):
        result, evidence = replay(*args)
        offset[0] += 1.0  # Both directions must finish past the old 30s cutoff.
        return None, evidence
    monkeypatch.setattr(ORACLE, "_replay_affine_family", collect)
    found, report = ORACLE.reconstruct_exact_fixed_pattern_witness(
        fixture, np.asarray([0., 0., 0., 0., 1.]), [True, False], 2, 120.)
    assert found is None and counts == {"build": 1, "solve": 1}
    assert len(report["families"]) == 2
    assert all(row["alpha_interval"] is not None for row in report["families"])
    assert all("failure" not in row for row in report["families"])
    assert len(alarms) == 4 and alarms[1] > 29 and alarms[2] > 14
    assert report["runtime_seconds"] > 30


def test_lazy_exact_column_assembly_matches_fraction_dots_and_reuses_columns():
    rows = [[F(1, 3), F(-2, 5)], [F(7, 8), F(9, 16)]]
    columns = [[F(3, 7), F(5, 2)], [F(1, 2), F(-1, 2)], [F(11, 3), F(7, 9)]]
    requests = []
    def source(i):
        requests.append(i)
        return columns[i]
    cache = {}
    integer_rows = [ORACLE._exact_integer_vector(row) for row in rows]
    first = ORACLE._build_selected_exact_matrix(rows, (0, 1), source, cache, integer_rows)
    second = ORACLE._build_selected_exact_matrix(rows, (1, 2), source, cache, integer_rows)
    assert first == [[ORACLE._dot(row, columns[i]) for i in (0, 1)] for row in rows]
    assert second == [[ORACLE._dot(row, columns[i]) for i in (1, 2)] for row in rows]
    assert requests == [0, 1, 2] and len(cache) == 3


@pytest.mark.parametrize("mutation", ["lp", "phase", "basis", "matrix", "directions", "solution"])
def test_solved_basis_cache_binds_context_and_exact_answers(mutation):
    identity = {"canonical_lp_identity_sha256": "lp", "phase_pattern_identity_sha256": "phase",
                "basis_sha256": "basis", "normalized_exact_matrix_sha256": "matrix",
                "ordered_rhs_direction_identities": ["t", "source_delta[0]"]}
    cache = {}
    key = ORACLE._cache_solved_basis(cache, identity, [[F(1)], [F(2)], [F(3)], [F(4)]])
    assert ORACLE._read_solved_basis_direction(cache, key, identity, 1) == [[F(3)], [F(4)]]
    if mutation == "solution":
        cache[key]["solutions"][0][0] = "5/1"
    else:
        field = {"lp": "canonical_lp_identity_sha256", "phase": "phase_pattern_identity_sha256",
                 "basis": "basis_sha256", "matrix": "normalized_exact_matrix_sha256",
                 "directions": "ordered_rhs_direction_identities"}[mutation]
        identity = {**identity, field: "changed"}
    with pytest.raises(ORACLE.ExactSolveFailure, match="cache integrity"):
        ORACLE._read_solved_basis_direction(cache, key, identity, 1)


def test_basis_hash_mismatch_rejects_before_any_exact_solve():
    basis = {"columns": (0,), "basis_sha256": "wrong"}
    with pytest.raises(RuntimeError, match="basis identity"):
        ORACLE._group_correction_bases([basis], 1, 2)


def test_direction_failure_does_not_discard_other_cached_direction(monkeypatch):
    fixture = replace(problem(), X=((F(1), F(-1)), (F(2), F(-2))),
                      low=(F(-1), F(-1)), high=(F(1), F(1)), epsilon=F(1),
                      W2=((F(0), F(0)), (F(0), F(0))))
    basis = {"columns": (1,), "basis_sha256": ORACLE._sha_json((1,)),
             "minimum_selected_source_slack": 1.0, "numerical_rank": 1}
    monkeypatch.setattr(ORACLE, "box_aware_correction_bases",
                        lambda *_: ([basis], np.ones(2)))
    replay = ORACLE._replay_affine_family
    calls = []
    def first_fails(*args):
        calls.append(1)
        if len(calls) == 1:
            raise ORACLE.ExactSolveFailure("direction replay timeout")
        _found, evidence = replay(*args)
        return None, evidence
    monkeypatch.setattr(ORACLE, "_replay_affine_family", first_fails)
    found, report = ORACLE.reconstruct_exact_fixed_pattern_witness(
        fixture, np.asarray([0., 0., 0., 0., 1.]), [True, False], 2, 2.)
    assert found is None and len(calls) == 2
    assert "failure" in report["families"][0]
    assert report["families"][1]["alpha_interval"] is not None
    assert report["linear_systems"][0]["basis_solve_count"] == 1
    assert not report["permits_infeasibility_claim"]


def test_67_positive_sources_cannot_supply_127_interior_correction_columns():
    sensitivity = np.column_stack((np.eye(127), np.ones(127)))
    candidate = [0.0] * 67 + [1.0] * 61
    bases, _slack = ORACLE.box_aware_correction_bases(
        sensitivity, candidate, [F(-1)] * 128, [F(1)] * 128, maximum=2)
    audit = bases[0]["slack_diagnostics"]
    assert audit["source_variables_with_positive_slack"] == 67
    assert audit["positive_slack_pool_numerical_rank"] == 67
    assert audit["required_correction_columns"] == 127
    assert audit["zero_minimum_slack_expected_by_cardinality"]
    assert audit["slack_threshold_rank_searches"] == 0
    assert all(row["minimum_selected_source_slack"] == 0.0 for row in bases)
    slack_vectors = [tuple(sorted(map(F, row["selected_source_slacks_exact"])))
                     for row in bases]
    assert slack_vectors == sorted(slack_vectors, reverse=True)


def test_production_solved_basis_identity_includes_canonical_node_lp(monkeypatch):
    fixture = replace(problem(), X=((F(1), F(-1)), (F(2), F(-2))),
                      low=(F(-1), F(-1)), high=(F(1), F(1)), epsilon=F(1),
                      W2=((F(0), F(0)), (F(0), F(0))))
    lp = ORACLE.ExactCanonicalLP(tuple(str(i) for i in range(9)),
                                (None,) * 9, (None,) * 9, ())
    found, report = ORACLE.reconstruct_exact_fixed_pattern_witness(
        fixture, np.asarray([0., 0., 0., 0., 1.]), [True, False], 2, 2., lp)
    assert found is not None and report["verified"]
    system = report["linear_systems"][0]
    identity = system["solved_basis_identity"]
    assert identity["canonical_lp_identity_sha256"] == lp.identity()
    assert not identity["standalone_fixture_without_lp"]
    assert identity["phase_pattern_identity_sha256"] == ORACLE._sha_json([True, False])
    assert identity["normalized_exact_matrix_sha256"] == system["column_primitive_matrix_sha256"]
    assert report["families"][0]["basis_work_shared"]
    assert report["families"][0]["basis_work_record_id"] == system["basis_work_record_id"]
