#!/usr/bin/env python3
"""CPU-only zero-variance decision for the authenticated Block-2 output state.

Numerical LP/least-squares routines are candidate generators only.  A zero
classification requires exact rational replay against the persisted IEEE-754
coefficients and ranges.  A positive classification requires the existing
directed/outward dual checker.  Otherwise the result is explicitly unresolved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy
from scipy.linalg import qr
from scipy.optimize import linprog, lsq_linear, minimize


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import cluster_common
import diagnose_psd_layernorm_variance_v1 as oracle


SCHEMA = "CORET_BLOCK2_OUTPUT_ZERO_VARIANCE_DECISION_V1"
EXACT_SCHEMA = "CORET_BLOCK2_OUTPUT_ZERO_VARIANCE_EXACT_WITNESS_V1"
CAPTURE_SCHEMA = "CORET_PSD_NEXT_LAYERNORM_STATE_CAPTURE_V1"
CAPTURE_MANIFEST_SCHEMA = "CORET_PSD_NEXT_LAYERNORM_CAPTURE_MANIFEST_V1"
PROPERTY_ID = "deept_table7_stdln3_s001_line1794_tok11"
MULTIPLIER = "0.75"
STAGE = "block2_output"
LAYERNORM_INDEX = 6
REDUCTION_LABEL = "b2_ffn_residual"
DECISION_ZERO = "ZERO_VARIANCE_FEASIBLE_EXACT"
DECISION_POSITIVE = "ZERO_VARIANCE_NOT_ESTABLISHED_DUAL_POSITIVE"
DECISION_UNRESOLVED = "ZERO_VARIANCE_UNRESOLVED"


def _atomic_json(path: Path, value: dict) -> dict:
    payload = dict(value)
    payload["record_sha256"] = cluster_common.canonical(payload)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return cluster_common.verified_json(path)


def _json_sha(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _tensor_sha(tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    header = json.dumps({"dtype": str(array.dtype), "shape": list(array.shape)},
                        sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def _state_hashes(state: dict) -> dict:
    weights = state["weights"].detach().cpu().contiguous()
    low = state["range_low"].detach().cpu().contiguous()
    high = state["range_high"].detach().cpu().contiguous()
    proof = state["proof"]
    ranges = hashlib.sha256(
        low.numpy().tobytes() + high.numpy().tobytes()).hexdigest()
    ids = _json_sha(proof["ids"])
    provenance = _json_sha({
        "masks": proof["masks"], "reasons": proof["reasons"],
        "num_tokens": proof["num_tokens"],
    })
    return {
        "center_sha256": _tensor_sha(weights[0]),
        "generator_sha256": _tensor_sha(weights[1:]),
        "generator_ids_sha256": ids,
        "ranges_sha256": ranges,
        "provenance_sha256": provenance,
        "generator_id_range_provenance_sha256": _json_sha({
            "generator_ids_sha256": ids, "ranges_sha256": ranges,
            "provenance_sha256": provenance,
        }),
        "generator_count": int(weights.shape[0] - 1),
        "native_generator_count": sum(
            reason not in oracle.NUMERICAL_REASONS
            for reason in proof["reasons"]),
        "numerical_generator_count": sum(
            reason in oracle.NUMERICAL_REASONS
            for reason in proof["reasons"]),
        "token_count": int(weights.shape[1]),
        "hidden_dimension": int(weights.shape[2]),
        "dtype": str(weights.dtype).replace("torch.", ""),
    }


def _load_authenticated_variants(manifest_path: Path) -> tuple[dict, dict]:
    manifest = cluster_common.verified_json(manifest_path)
    if (manifest.get("schema") != CAPTURE_MANIFEST_SCHEMA
            or manifest.get("property_id") != PROPERTY_ID
            or manifest.get("multiplier") != MULTIPLIER
            or manifest.get("stage_label") != STAGE
            or int(manifest.get("layernorm_index", -1)) != LAYERNORM_INDEX
            or manifest.get("reduction_label") != REDUCTION_LABEL
            or manifest.get("pinned_deept_revision") != oracle.PINNED_REVISION
            or manifest.get("scientific_manifest_sha256") !=
            cluster_common.SCIENTIFIC_MANIFEST_SHA
            or manifest.get("production_manifest_sha256") !=
            cluster_common.PRODUCTION_MANIFEST_SHA
            or manifest.get("source_set_model") != oracle.SOURCE_SET_MODEL):
        raise RuntimeError("zero-variance capture manifest identity differs")
    artifact_path = (manifest_path.parent
                     / manifest["tensor_artifact_path"]).resolve()
    if oracle.sha256(artifact_path) != manifest["tensor_artifact_sha256"]:
        raise RuntimeError("zero-variance capture artifact SHA differs")
    import torch  # CPU artifact decoding only.
    payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if (payload.get("schema") != CAPTURE_SCHEMA
            or payload.get("pinned_revision") != oracle.PINNED_REVISION
            or payload.get("identity") != manifest.get("artifact_identity")
            or payload.get("complete_layernorm_input_alias") !=
            "post_last_reduction"
            or payload.get("reduction_label") != REDUCTION_LABEL):
        raise RuntimeError("zero-variance capture artifact identity differs")
    states = payload.get("states")
    if set(states or {}) != {"pre_last_reduction", "post_last_reduction"}:
        raise RuntimeError("zero-variance capture state inventory differs")
    expected = (
        ("pre_last_reduction", "pre_last_reduction"),
        ("post_last_reduction", "post_last_reduction"),
        ("complete_layernorm_input", "post_last_reduction"),
    )
    variants = manifest.get("variants")
    if (not isinstance(variants, list)
            or [(row.get("capture_variant"), row.get("state_key"))
                for row in variants] != list(expected)):
        raise RuntimeError("zero-variance capture variant inventory differs")
    for row, (_name, key) in zip(variants, expected):
        actual = _state_hashes(states[key])
        for field, value in actual.items():
            if row.get(field) != value:
                raise RuntimeError(
                    f"zero-variance capture state hash differs: {key}/{field}")
    result_path = (manifest_path.parent / manifest["result_path"]).resolve()
    if oracle.sha256(result_path) != manifest["result_sha256"]:
        raise RuntimeError("zero-variance source-result SHA differs")
    result = cluster_common.verified_json(result_path)
    diagnostic = result.get("domain_failure_diagnostic")
    if (result.get("property_id") != PROPERTY_ID
            or result.get("terminal_status") != "UNCERTIFIED_DOMAIN_FAILURE"
            or result.get("generic_fallback_count") != 0
            or not isinstance(diagnostic, dict)
            or diagnostic.get("label") != STAGE
            or diagnostic.get("domain_admissible") is not False
            or int(diagnostic.get("generator_count", -1)) !=
            int(states["post_last_reduction"]["weights"].shape[0] - 1)):
        raise RuntimeError("zero-variance source-result semantics differ")

    def decoded(key: str) -> dict:
        spec = {
            "path": artifact_path, "sha256": manifest["tensor_artifact_sha256"],
            "schema": CAPTURE_SCHEMA, "state_key": key,
        }
        return oracle._load_torch_state(
            spec["path"], spec["sha256"], spec["schema"], spec["state_key"])

    post = decoded("post_last_reduction")
    pre = decoded("pre_last_reduction")
    token = int(diagnostic["minimum_token_index"])
    return ({
        "complete_post_reduction": oracle._variant(post, token, False),
        "native_only": oracle._variant(post, token, True),
        "authenticated_pre_reduction": oracle._variant(pre, token, False),
    }, {
        "manifest_path": str(manifest_path),
        "manifest_sha256": oracle.sha256(manifest_path),
        "artifact_path": str(artifact_path),
        "artifact_sha256": manifest["tensor_artifact_sha256"],
        "result_path": str(result_path),
        "result_sha256": manifest["result_sha256"],
        "token_index": token,
    })


def _problem_sha(center: np.ndarray, generators: np.ndarray,
                 low: np.ndarray, high: np.ndarray, ids: list[str]) -> str:
    digest = hashlib.sha256()
    for value in (center, generators, low, high):
        array = np.ascontiguousarray(value, dtype="<f8")
        digest.update(json.dumps(
            list(array.shape), separators=(",", ":")).encode())
        digest.update(array.tobytes())
    digest.update(json.dumps(ids, separators=(",", ":")).encode())
    return digest.hexdigest()


def centered_problem(center, generators, low, high, ids=None) -> dict:
    center = np.asarray(center, dtype=np.float64)
    generators = np.asarray(generators, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    if (center.ndim != 1 or center.size < 2 or generators.ndim != 2
            or generators.shape[1] != center.size
            or low.shape != (len(generators),) or high.shape != low.shape
            or not all(np.isfinite(item).all() for item in
                       (center, generators, low, high))
            or np.any(low > high)):
        raise RuntimeError("zero-variance affine problem is malformed")
    # Equality of every coordinate to the last is exactly equivalent to zero
    # centered variance and avoids any floating mean in the exact certificate.
    ids = list(ids) if ids is not None else [f"g{index}" for index in range(
        len(generators))]
    if len(ids) != len(generators) or len(set(ids)) != len(ids):
        raise RuntimeError("zero-variance generator identity differs")
    return {
        "center": center, "generators": generators, "low": low, "high": high,
        "ids": ids,
        "b": center[:-1] - center[-1],
        "A": (generators[:, :-1] - generators[:, [-1]]).T,
        "dimension": int(center.size), "variable_count": int(len(generators)),
        "problem_sha256": _problem_sha(center, generators, low, high, ids),
    }


def _box_metrics(candidate: np.ndarray, low: np.ndarray,
                 high: np.ndarray) -> dict:
    lower = candidate - low
    upper = high - candidate
    scale = np.maximum(1.0, np.maximum(np.abs(low), np.abs(high)))
    tolerance = 1e-9 * scale
    at_lower = lower <= tolerance
    at_upper = upper <= tolerance
    return {
        "minimum_lower_slack": float(lower.min()) if len(lower) else None,
        "minimum_upper_slack": float(upper.min()) if len(upper) else None,
        "minimum_distance_to_box_boundary": float(
            np.minimum(lower, upper).min()) if len(lower) else None,
        "variables_at_lower_bound": int(at_lower.sum()),
        "variables_at_upper_bound": int(at_upper.sum()),
        "variables_at_either_bound": int((at_lower | at_upper).sum()),
        "maximum_box_violation": float(max(
            0.0, -float(lower.min(initial=0.0)),
            -float(upper.min(initial=0.0)))),
    }


def numerical_primal_search(problem: dict) -> tuple[np.ndarray, dict]:
    A, b = problem["A"], problem["b"]
    low, high = problem["low"], problem["high"]
    scale = np.maximum(
        np.maximum(np.linalg.norm(A, axis=1), np.abs(b)),
        np.finfo(np.float64).tiny)
    scaled_A, scaled_b = A / scale[:, None], -b / scale
    started = time.perf_counter()
    lp = linprog(
        np.zeros(len(low), dtype=np.float64), A_eq=scaled_A, b_eq=scaled_b,
        bounds=list(zip(low, high)), method="highs",
        options={"presolve": True})
    searches = []
    candidates = []
    if lp.x is not None and np.isfinite(lp.x).all():
        candidates.append(("highs_feasibility", np.asarray(lp.x)))
    searches.append({
        "solver": "scipy.optimize.linprog/highs", "success": bool(lp.success),
        "status": int(lp.status), "message": str(lp.message),
        "iterations": int(getattr(lp, "nit", 0)),
    })
    fixed = low == high
    fixed_value = low.copy()
    adjusted_b = b + (A[:, fixed] @ fixed_value[fixed]
                      if fixed.any() else 0.0)
    free = ~fixed
    if free.any():
        lsq = lsq_linear(
            A[:, free] / scale[:, None], -adjusted_b / scale,
            bounds=(low[free], high[free]), method="trf", lsq_solver="lsmr",
            tol=1e-12, lsmr_tol=1e-12, max_iter=500, verbose=0)
        candidate = fixed_value.copy()
        candidate[free] = lsq.x
        candidates.append(("bounded_least_squares", candidate))
        searches.append({
            "solver": "scipy.optimize.lsq_linear/trf-lsmr",
            "success": bool(lsq.success), "status": int(lsq.status),
            "message": str(lsq.message), "iterations": int(lsq.nit),
            "optimality": float(lsq.optimality),
        })
    if not candidates:
        candidate = low + 0.5 * (high - low)
        candidates.append(("box_midpoint_fallback", candidate))
    evaluated = []
    for name, candidate in candidates:
        candidate = np.minimum(np.maximum(candidate, low), high)
        vector = problem["center"] + candidate @ problem["generators"]
        centered = vector - vector.mean()
        equality = b + A @ candidate
        evaluated.append((float(np.dot(centered, centered) / len(centered)),
                          name, candidate, equality, centered))
    variance, selected, candidate, equality, centered = min(
        evaluated, key=lambda row: row[0])
    return candidate, {
        "selected_candidate": selected,
        "solver_runs": searches,
        "max_equality_residual": float(np.abs(equality).max(initial=0.0)),
        "centered_residual_l2": float(np.linalg.norm(centered)),
        "numerical_variance": variance,
        "row_scaling_min": float(scale.min()),
        "row_scaling_max": float(scale.max()),
        "runtime_seconds": time.perf_counter() - started,
        **_box_metrics(candidate, low, high),
    }


def _fraction(value: float) -> Fraction:
    return Fraction.from_float(float(value))


def _exact_coefficient(problem: dict, row: int, column: int) -> Fraction:
    generator = problem["generators"][column]
    return _fraction(generator[row]) - _fraction(generator[-1])


def _exact_center_difference(problem: dict, row: int) -> Fraction:
    center = problem["center"]
    return _fraction(center[row]) - _fraction(center[-1])


def verify_exact_zero_certificate(problem: dict, certificate: dict) -> dict:
    if (certificate.get("schema") != EXACT_SCHEMA
            or certificate.get("problem_sha256") != problem["problem_sha256"]
            or certificate.get("generator_ids_sha256") !=
            _json_sha(problem["ids"])
            or len(certificate.get("xi_rationals", ())) !=
            problem["variable_count"]):
        raise RuntimeError("exact zero certificate shape differs")
    values = []
    for item in certificate["xi_rationals"]:
        try:
            values.append(Fraction(int(item["numerator"]),
                                   int(item["denominator"])))
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
            raise RuntimeError("exact zero certificate value differs") from error
    for index, value in enumerate(values):
        if not (_fraction(problem["low"][index]) <= value
                <= _fraction(problem["high"][index])):
            raise RuntimeError(f"exact zero certificate is outside range: {index}")
    residuals = []
    for row in range(problem["dimension"] - 1):
        residual = _exact_center_difference(problem, row)
        residual += sum(
            (_exact_coefficient(problem, row, column) * value
             for column, value in enumerate(values)), Fraction(0))
        residuals.append(residual)
    if any(residual != 0 for residual in residuals):
        raise RuntimeError("exact zero certificate equality differs")
    return {
        "exact_equalities": len(residuals), "exact_box_constraints": len(values),
        "maximum_exact_residual": "0", "exact_variance": "0",
    }


def construct_exact_zero_certificate(problem: dict, candidate: np.ndarray,
                                     maximum_rank: int) -> tuple[dict | None, dict]:
    A = problem["A"]
    free = np.flatnonzero(problem["low"] < problem["high"])
    slack = np.minimum(candidate - problem["low"],
                       problem["high"] - candidate)
    range_width = problem["high"] - problem["low"]
    interior = np.flatnonzero(
        (range_width > 0.0) & (slack > 1e-10 * range_width))
    correction_pool = interior if len(interior) else free
    if not len(free):
        selected_columns = np.empty(0, dtype=np.int64)
        rank = 0
    else:
        _q, r, pivots = qr(
            A[:, correction_pool], mode="economic", pivoting=True)
        threshold = (max(A.shape) * np.finfo(np.float64).eps
                     * (abs(r[0, 0]) if r.size else 0.0))
        rank = int(np.sum(np.abs(np.diag(r)) > threshold))
        selected_columns = correction_pool[
            np.asarray(pivots[:rank], dtype=np.int64)]
        # An interior-only pool may not span every exact equality.  Retry with
        # every non-singleton variable before declaring a rank limitation.
        full_rank = int(np.linalg.matrix_rank(A[:, free]))
        if rank < full_rank:
            _q, r, pivots = qr(A[:, free], mode="economic", pivoting=True)
            threshold = (max(A.shape) * np.finfo(np.float64).eps
                         * (abs(r[0, 0]) if r.size else 0.0))
            rank = int(np.sum(np.abs(np.diag(r)) > threshold))
            selected_columns = free[
                np.asarray(pivots[:rank], dtype=np.int64)]
    diagnostics = {
        "numerical_rank": rank,
        "equation_count": int(A.shape[0]),
        "selected_correction_variables": int(len(selected_columns)),
        "interior_candidate_variables": int(len(interior)),
        "attempted": rank <= maximum_rank,
    }
    if rank > maximum_rank:
        diagnostics["reason"] = "exact correction rank exceeds configured cap"
        return None, diagnostics
    if rank:
        _q, _r, row_pivots = qr(
            A[:, selected_columns].T, mode="economic", pivoting=True)
        selected_rows = np.asarray(row_pivots[:rank], dtype=np.int64)
    else:
        selected_rows = np.empty(0, dtype=np.int64)
    selected_set = set(int(index) for index in selected_columns)
    values = [_fraction(value) for value in candidate]
    try:
        if rank:
            matrix = sympy.Matrix([
                [sympy.Rational(_exact_coefficient(problem, int(row), int(col)).numerator,
                                _exact_coefficient(problem, int(row), int(col)).denominator)
                 for col in selected_columns]
                for row in selected_rows])
            rhs = []
            for row in selected_rows:
                value = -_exact_center_difference(problem, int(row))
                value -= sum(
                    (_exact_coefficient(problem, int(row), column) * values[column]
                     for column in range(problem["variable_count"])
                     if column not in selected_set), Fraction(0))
                rhs.append(sympy.Rational(value.numerator, value.denominator))
            solution = matrix.inv().multiply(sympy.Matrix(rhs))
            for column, value in zip(selected_columns, solution):
                values[int(column)] = Fraction(int(value.p), int(value.q))
        certificate = {
            "schema": EXACT_SCHEMA,
            "problem_sha256": problem["problem_sha256"],
            "generator_ids_sha256": _json_sha(problem["ids"]),
            "construction": (
                "exact_rational_correction_on_qr_selected_full_rank_subsystem"),
            "reference_coordinate": problem["dimension"] - 1,
            "selected_rows": [int(value) for value in selected_rows],
            "selected_columns": [int(value) for value in selected_columns],
            "xi_rationals": [
                {"numerator": str(value.numerator),
                 "denominator": str(value.denominator)} for value in values],
        }
        checked = verify_exact_zero_certificate(problem, certificate)
        diagnostics.update({"verified": True, **checked})
        return certificate, diagnostics
    except Exception as error:
        diagnostics.update({
            "verified": False,
            "reason": f"{type(error).__name__}: {error}",
        })
        return None, diagnostics


def exact_candidate_upper(problem: dict, candidate: np.ndarray) -> dict:
    values = [_fraction(value) for value in candidate]
    for index, value in enumerate(values):
        if not (_fraction(problem["low"][index]) <= value
                <= _fraction(problem["high"][index])):
            raise RuntimeError("candidate upper-bound point is outside range")
    coordinates = []
    for coordinate in range(problem["dimension"]):
        value = _fraction(problem["center"][coordinate])
        value += sum(
            (_fraction(problem["generators"][index, coordinate]) * xi
             for index, xi in enumerate(values)), Fraction(0))
        coordinates.append(value)
    mean = sum(coordinates, Fraction(0)) / len(coordinates)
    variance = sum(((value - mean) ** 2 for value in coordinates),
                   Fraction(0)) / len(coordinates)
    outward = float(variance)
    if Fraction.from_float(outward) < variance:
        outward = math.nextafter(outward, math.inf)
    return {
        "exact_feasible_binary64_candidate": True,
        "exact_variance_numerator": str(variance.numerator),
        "exact_variance_denominator": str(variance.denominator),
        "variance_upper_outward_binary64": outward,
    }


def exact_zero_dual_baseline(dimension: int) -> dict:
    return {
        "witness": "y=0", "exact_objective": "0",
        "outward_safe_lower": 0.0, "dimension": int(dimension),
    }


def _optimize_from_start(problem: dict, start: np.ndarray) -> tuple[np.ndarray, dict]:
    center, generators = problem["center"], problem["generators"]
    low, high = problem["low"], problem["high"]
    b = oracle.center_vector(center)

    def objective_gradient(candidate):
        y = oracle.center_vector(np.asarray(candidate, dtype=np.float64))
        q = generators @ y if len(generators) else np.empty(0)
        minimizing = np.where(q >= 0.0, low, high)
        support = (generators.T @ minimizing
                   if len(generators) else np.zeros_like(y))
        support = oracle.center_vector(support)
        numerator = (2.0 * np.dot(y, b) - np.dot(y, y)
                     + 2.0 * (np.dot(q, minimizing) if len(q) else 0.0))
        gradient = 2.0 * (b - y + support)
        return -float(numerator / len(center)), -gradient / len(center)

    result = minimize(
        objective_gradient, oracle.center_vector(start), method="L-BFGS-B",
        jac=True, options={"maxiter": 2000, "ftol": 1e-15,
                           "gtol": 1e-12, "maxls": 100})
    return oracle.center_vector(np.asarray(result.x)), {
        "success": bool(result.success), "status": int(result.status),
        "message": str(result.message), "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "untrusted_objective": -float(result.fun),
    }


def dual_scaling_search(problem: dict, primal_candidate: np.ndarray) -> dict:
    vector = problem["center"] + primal_candidate @ problem["generators"]
    residual = oracle.center_vector(vector)
    b = oracle.center_vector(problem["center"])
    b_norm = max(float(np.linalg.norm(b)), np.finfo(np.float64).tiny)
    starts = [("primal_residual", residual), ("centered_center", b)]
    for exponent in (-20, -12, -6, 0, 6, 12, 20):
        starts.append((f"normalized_center_2^{exponent}",
                       (b / b_norm) * (2.0 ** exponent)))
    rows = []
    for name, start in starts:
        witness, optimizer = _optimize_from_start(problem, start)
        checked = oracle.recheck_dual_witness(
            problem["center"], problem["generators"], problem["low"],
            problem["high"], witness)
        rows.append({
            "start": name,
            "start_l2": float(np.linalg.norm(start)),
            "witness_l2": float(np.linalg.norm(witness)),
            "optimizer": optimizer,
            "outward_safe_lower":
                checked["psd_dual_candidate_lower_outward_safe"],
            "directed_checker": checked,
            "witness_binary64_hex": [float(value).hex() for value in witness],
            "witness_sha256": hashlib.sha256(
                np.ascontiguousarray(witness, dtype="<f8").tobytes()).hexdigest(),
        })
    best = max(rows, key=lambda row: row["outward_safe_lower"])
    return {
        "zero_baseline": exact_zero_dual_baseline(problem["dimension"]),
        "candidates": rows, "best": best,
    }


def decide_variant(name: str, variant: dict, exact_max_rank: int,
                   certificate_dir: Path) -> dict:
    started = time.perf_counter()
    problem = centered_problem(
        variant["center"], variant["generators"],
        variant["low"], variant["high"], variant["ids"])
    candidate, primal = numerical_primal_search(problem)
    upper = exact_candidate_upper(problem, candidate)
    exact, exact_status = construct_exact_zero_certificate(
        problem, candidate, exact_max_rank)
    certificate_record = None
    if exact is not None:
        certificate_path = certificate_dir / f"{name}_exact_zero_witness.json"
        persisted = _atomic_json(certificate_path, exact)
        # Re-read and recheck the serialized rational witness.
        verify_exact_zero_certificate(problem, persisted)
        certificate_record = {
            "path": str(certificate_path),
            "sha256": oracle.sha256(certificate_path),
            "record_sha256": persisted["record_sha256"],
        }
    dual = dual_scaling_search(problem, candidate)
    if exact is not None:
        decision = DECISION_ZERO
    elif dual["best"]["outward_safe_lower"] > 0.0:
        decision = DECISION_POSITIVE
    else:
        decision = DECISION_UNRESOLVED
    return {
        "variant": name, "decision": decision,
        "generator_count": problem["variable_count"],
        "native_generator_count": variant["native_generator_count"],
        "numerical_generator_count": variant["numerical_generator_count"],
        "primal_numerical_search": primal,
        "primal_exact_binary64_candidate_upper": upper,
        "exact_zero_certificate_status": exact_status,
        "exact_zero_certificate": certificate_record,
        "dual_search": dual,
        "runtime_seconds": time.perf_counter() - started,
    }


def execute(manifest_path: Path, output_path: Path,
            exact_max_rank: int = 128) -> dict:
    variants, identity = _load_authenticated_variants(manifest_path)
    certificate_dir = output_path.parent / "exact_zero_witnesses"
    results = [decide_variant(name, variant, exact_max_rank, certificate_dir)
               for name, variant in variants.items()]
    return _atomic_json(output_path, {
        "schema": SCHEMA, "property_id": PROPERTY_ID,
        "multiplier": MULTIPLIER, "stage_label": STAGE,
        "layernorm_index": LAYERNORM_INDEX,
        "authenticated_capture": identity,
        "results": results,
        "complete_state_decision": results[0]["decision"],
        "optimizer_is_untrusted": True,
        "zero_decision_requires_exact_rational_witness": True,
        "positive_decision_requires_outward_safe_dual_lower": True,
        "scientific_queries": 0, "bound_calls": 0,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exact-max-rank", type=int, default=128)
    args = parser.parse_args()
    if not 0 <= args.exact_max_rank <= 128:
        raise RuntimeError("exact rank cap is outside [0,128]")
    report = execute(args.capture_manifest.resolve(), args.output.resolve(),
                     args.exact_max_rank)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
