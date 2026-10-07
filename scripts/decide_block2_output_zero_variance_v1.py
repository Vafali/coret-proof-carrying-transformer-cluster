#!/usr/bin/env python3
"""CPU-only zero-variance decision for the authenticated Block-2 output state.

Numerical LP/least-squares routines are candidate generators only.  A zero
classification requires exact rational replay against the persisted IEEE-754
coefficients and ranges.  A positive classification requires the existing
directed/outward dual checker.  Otherwise the result is explicitly unresolved.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import multiprocessing
import os
import signal
import sys
import threading
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
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
EXACT_BAREISS_TIMEOUT = "EXACT_BAREISS_TIMEOUT"
EXACT_BAREISS_SINGULAR = "EXACT_BAREISS_SINGULAR_OR_PIVOT_FAILURE"
EXACT_BAREISS_NONEXACT = "EXACT_BAREISS_NONEXACT_DIVISION"
EXACT_SOLUTION_OUTSIDE_BOX = "EXACT_SOLUTION_OUTSIDE_AUTHENTICATED_BOX"
EXACT_REPLAY_FAILED = "EXACT_REPLAY_FAILED"
EXACT_ZERO_VERIFIED = "EXACT_AUTHENTICATED_ZERO_WITNESS_VERIFIED"
VARIANT_NAMES = (
    "complete_post_reduction", "native_only",
    "authenticated_pre_reduction")
EXCLUSION_SCHEMA = "CORET_BLOCK2_OUTPUT_ZERO_VARIANCE_EXACT_FARKAS_V1"
FEASIBLE = "EXACT_ZERO_VARIANCE_FEASIBLE"
EXCLUDED = "EXACT_ZERO_VARIANCE_EXCLUDED"
INCONCLUSIVE = "INCONCLUSIVE"
EXACT_EXCLUSION_TIMEOUT = "EXACT_EXCLUSION_TIMEOUT"


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


def _progress(variant: str, stage: str, started: float, **fields) -> None:
    print(json.dumps({
        "event": "ZERO_VARIANCE_PROGRESS",
        "variant": variant,
        "stage": stage,
        "elapsed_seconds": time.perf_counter() - started,
        **fields,
    }, sort_keys=True), flush=True)


def numerical_primal_search(problem: dict,
                            variant_name: str = "synthetic") -> tuple[np.ndarray, dict]:
    A, b = problem["A"], problem["b"]
    low, high = problem["low"], problem["high"]
    scale = np.maximum(
        np.maximum(np.linalg.norm(A, axis=1), np.abs(b)),
        np.finfo(np.float64).tiny)
    scaled_A, scaled_b = A / scale[:, None], -b / scale
    started = time.perf_counter()
    stage_started = time.perf_counter()
    lp = linprog(
        np.zeros(len(low), dtype=np.float64), A_eq=scaled_A, b_eq=scaled_b,
        bounds=list(zip(low, high)), method="highs",
        options={"presolve": True})
    _progress(
        variant_name, "highs_feasibility_complete", stage_started,
        success=bool(lp.success), status=int(lp.status),
        iterations=int(getattr(lp, "nit", 0)))
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
        stage_started = time.perf_counter()
        _progress(variant_name, "bounded_least_squares_start", stage_started)
        lsq = lsq_linear(
            A[:, free] / scale[:, None], -adjusted_b / scale,
            bounds=(low[free], high[free]), method="trf", lsq_solver="lsmr",
            tol=1e-12, lsmr_tol=1e-12, max_iter=500, verbose=0)
        _progress(
            variant_name, "bounded_least_squares_complete", stage_started,
            success=bool(lsq.success), status=int(lsq.status),
            iterations=int(lsq.nit), optimality=float(lsq.optimality))
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
    maximum_equality_residual = float(
        np.abs(equality).max(initial=0.0))
    maximum_scaled_equality_residual = float(
        (np.abs(equality) / scale).max(initial=0.0))
    result = {
        "selected_candidate": selected,
        "solver_runs": searches,
        "max_equality_residual": maximum_equality_residual,
        "max_scaled_equality_residual": maximum_scaled_equality_residual,
        "centered_residual_l2": float(np.linalg.norm(centered)),
        "centered_residual_rms": float(np.sqrt(variance)),
        "numerical_variance": variance,
        "candidate_state_scale": float(max(
            1.0, np.abs(vector).max(initial=0.0))),
        "row_scaling_min": float(scale.min()),
        "row_scaling_max": float(scale.max()),
        "runtime_seconds": time.perf_counter() - started,
        **_box_metrics(candidate, low, high),
    }
    _progress(
        variant_name, "numerical_primal_stage_complete", started,
        selected_candidate=selected,
        max_equality_residual=maximum_equality_residual,
        max_scaled_equality_residual=maximum_scaled_equality_residual,
        numerical_variance=variance)
    return candidate, result


def _fraction(value: float) -> Fraction:
    return Fraction.from_float(float(value))


def _exact_coefficient(problem: dict, row: int, column: int) -> Fraction:
    exact = problem.get("exact_coefficient")
    if exact is not None:
        return exact(row, column)
    generator = problem["generators"][column]
    return _fraction(generator[row]) - _fraction(generator[-1])


def _exact_center_difference(problem: dict, row: int) -> Fraction:
    exact = problem.get("exact_center_difference")
    if exact is not None:
        return exact(row)
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
    exact_replay = problem.get("exact_replay")
    if exact_replay is not None:
        checked = exact_replay(values)
        if (not isinstance(checked, dict)
                or checked.get("maximum_exact_residual") != "0"
                or checked.get("exact_variance") != "0"):
            raise RuntimeError("exact zero certificate equality differs")
        return checked
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


def _rational_json(value: Fraction) -> dict:
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def _read_rational(value: dict) -> Fraction:
    try:
        return Fraction(int(value["numerator"]), int(value["denominator"]))
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise RuntimeError("exact certificate rational is malformed") from error


def _farkas_replay(problem: dict, direction: list[Fraction]) -> dict:
    """Full ORIGINAL coordinate-difference equations and box inequalities.

    h + A xi = 0. Equality multiplier y is signed (equivalently split
    between the two canonical inequality orientations). For a=y*A, bound
    multipliers are lower=max(a,0), upper=max(-a,0). They cancel a exactly.
    RHS = -y*h - sum min(a*low,a*high) must be STRICTLY negative.
    No numerical LP coefficient or rounded centered matrix is used here.
    """
    rhs = -sum((y * _exact_center_difference(problem, row)
                for row, y in enumerate(direction)), Fraction(0))
    lower_count = upper_count = 0
    for column in range(problem["variable_count"]):
        a = sum((y * _exact_coefficient(problem, row, column)
                 for row, y in enumerate(direction) if y), Fraction(0))
        lower, upper = max(a, Fraction(0)), max(-a, Fraction(0))
        if lower < 0 or upper < 0 or a - lower + upper != 0:
            raise RuntimeError("exact Farkas cancellation/nonnegativity failed")
        rhs -= lower * _fraction(problem["low"][column])
        rhs += upper * _fraction(problem["high"][column])
        lower_count += lower > 0
        upper_count += upper > 0
    return {"exact_farkas_rhs": _rational_json(rhs),
            "exact_equalities": problem["dimension"] - 1,
            "exact_box_constraints": 2 * problem["variable_count"],
            "positive_lower_bound_multipliers": lower_count,
            "positive_upper_bound_multipliers": upper_count,
            "exact_cancellation_verified": True, "exact_nonnegativity_verified": True}


def verify_exact_exclusion_certificate(problem: dict, certificate: dict) -> dict:
    if (certificate.get("schema") != EXCLUSION_SCHEMA or
            certificate.get("problem_sha256") != problem["problem_sha256"] or
            certificate.get("generator_ids_sha256") != _json_sha(problem["ids"]) or
            certificate.get("reference_coordinate") != problem["dimension"] - 1 or
            certificate.get("bound_multiplier_rule") != "exact_sign_of_yA" or
            len(certificate.get("equality_multipliers", [])) != problem["dimension"] - 1):
        raise RuntimeError("exact exclusion certificate identity/shape differs")
    direction = [_read_rational(x) for x in certificate["equality_multipliers"]]
    checked = _farkas_replay(problem, direction)
    if (_read_rational(checked["exact_farkas_rhs"]) >= 0 or
            checked["exact_farkas_rhs"] != certificate.get("exact_farkas_rhs")):
        raise RuntimeError("exact exclusion certificate RHS is not negative or differs")
    return {**checked, "verified": True}


def construct_exact_exclusion_certificate(problem: dict, witness) -> dict | None:
    # A floating dual is ONLY a direction proposal. Recenter it EXACTLY to
    # obtain a sum-zero functional; all subsequent operations are rational.
    if len(witness) != problem["dimension"] or not np.isfinite(witness).all():
        return None
    values = [_fraction(x) for x in witness]
    mean = sum(values, Fraction(0)) / problem["dimension"]
    direction = [x - mean for x in values[:-1]]
    checked = _farkas_replay(problem, direction)
    if _read_rational(checked["exact_farkas_rhs"]) >= 0:
        return None
    certificate = {"schema": EXCLUSION_SCHEMA, "problem_sha256": problem["problem_sha256"],
                   "generator_ids_sha256": _json_sha(problem["ids"]),
                   "reference_coordinate": problem["dimension"] - 1,
                   "equality_multipliers": [_rational_json(x) for x in direction],
                   "bound_multiplier_rule": "exact_sign_of_yA",
                   "exact_farkas_rhs": checked["exact_farkas_rhs"]}
    verify_exact_exclusion_certificate(problem, certificate)
    return certificate


class ExactBareissTimeoutError(RuntimeError):
    pass


class ExactBareissSingularError(RuntimeError):
    pass


class ExactBareissNonexactDivisionError(RuntimeError):
    pass


class ExactIntegerSystemError(RuntimeError):
    pass


class ExactReplayError(RuntimeError):
    pass


def _dyadic_row_to_integers(values: list[Fraction]) -> tuple[list[int], dict]:
    """Clear a dyadic row denominator and remove its integer content."""
    exponents = []
    for value in values:
        denominator = value.denominator
        if denominator <= 0 or denominator & (denominator - 1):
            raise ExactIntegerSystemError(
                "selected correction system contains a non-dyadic value")
        exponents.append(denominator.bit_length() - 1)
    common_exponent = max(exponents, default=0)
    integers = [
        value.numerator << (common_exponent - exponent)
        for value, exponent in zip(values, exponents)
    ]
    content = 0
    for value in integers:
        content = math.gcd(content, abs(value))
    content = max(content, 1)
    integers = [value // content for value in integers]
    first = next((value for value in integers if value), 0)
    if first < 0:
        integers = [-value for value in integers]
    return integers, {
        "common_denominator_power_of_two": common_exponent,
        "removed_row_gcd_bits": content.bit_length() - 1,
        "maximum_integer_bits": max(
            (abs(value).bit_length() for value in integers), default=0),
    }


@contextmanager
def _exact_solve_guard(timeout_seconds: float):
    """Interrupt a silent exact solve on POSIX when run in the main thread."""
    enabled = (timeout_seconds > 0 and hasattr(signal, "SIGALRM")
               and threading.current_thread() is threading.main_thread())
    if not enabled:
        yield False
        return
    def timed_out(_signum, _frame):
        raise ExactBareissTimeoutError(
            f"exact integer solve exceeded {timeout_seconds:g} seconds")

    previous_handler = signal.signal(signal.SIGALRM, timed_out)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0.0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _bareiss_eliminate(
        integer_rows: list[list[int]], integer_rhs: list[int]
) -> tuple[list[list[int]], dict]:
    """Deterministic fraction-free elimination of an integer system."""
    size = len(integer_rows)
    augmented = [list(row) + [int(rhs)] for row, rhs in
                 zip(integer_rows, integer_rhs)]
    initial_bits = max(
        (abs(value).bit_length() for row in augmented for value in row),
        default=0)
    maximum_bits = initial_bits
    previous_pivot = 1
    row_swaps = []
    exact_divisions = 0
    for pivot_column in range(size - 1):
        pivot_row = next(
            (row for row in range(pivot_column, size)
             if augmented[row][pivot_column] != 0), None)
        if pivot_row is None:
            raise ExactBareissSingularError(
                f"no nonzero pivot in column {pivot_column}")
        if pivot_row != pivot_column:
            augmented[pivot_column], augmented[pivot_row] = (
                augmented[pivot_row], augmented[pivot_column])
            row_swaps.append([pivot_column, pivot_row])
        pivot = augmented[pivot_column][pivot_column]
        pivot_values = augmented[pivot_column]
        for row_index in range(pivot_column + 1, size):
            row_values = augmented[row_index]
            multiplier = row_values[pivot_column]
            for column in range(pivot_column + 1, size + 1):
                numerator = (pivot * row_values[column]
                             - multiplier * pivot_values[column])
                maximum_bits = max(maximum_bits, abs(numerator).bit_length())
                quotient, remainder = divmod(numerator, previous_pivot)
                if remainder:
                    raise ExactBareissNonexactDivisionError(
                        "Bareiss division is nonexact at "
                        f"({row_index},{column})")
                row_values[column] = quotient
                maximum_bits = max(maximum_bits, abs(quotient).bit_length())
                exact_divisions += 1
            row_values[pivot_column] = 0
        previous_pivot = pivot
    if augmented[-1][-2] == 0:
        raise ExactBareissSingularError("final Bareiss pivot is zero")
    return augmented, {
        "initial_maximum_integer_bits": initial_bits,
        "elimination_maximum_integer_bits": maximum_bits,
        "row_swaps": row_swaps,
        "exact_division_count": exact_divisions,
    }


def _bareiss_back_substitution(
        augmented: list[list[int]]) -> list[Fraction]:
    size = len(augmented)
    solution = [Fraction(0) for _ in range(size)]
    for row_index in range(size - 1, -1, -1):
        pivot = augmented[row_index][row_index]
        if pivot == 0:
            raise ExactBareissSingularError(
                f"zero triangular pivot at row {row_index}")
        residual = Fraction(augmented[row_index][size])
        residual -= sum(
            (augmented[row_index][column] * solution[column]
             for column in range(row_index + 1, size)), Fraction(0))
        solution[row_index] = residual / pivot
    return solution


def _solve_exact_integer_system(
        integer_rows: list[list[int]], integer_rhs: list[int],
        timeout_seconds: float, variant_name: str = "synthetic"
) -> tuple[list[Fraction], dict]:
    """Pure-Python Bareiss solve followed by exact selected-system replay."""
    size = len(integer_rows)
    if (not integer_rows or any(len(row) != size for row in integer_rows)):
        raise ExactIntegerSystemError(
            "selected exact integer system is not nonempty and square")
    if len(integer_rhs) != size:
        raise ExactIntegerSystemError("exact integer RHS shape differs")
    with _exact_solve_guard(timeout_seconds) as guard_enabled:
        elimination_started = time.perf_counter()
        augmented, elimination = _bareiss_eliminate(
            integer_rows, integer_rhs)
        elimination_seconds = time.perf_counter() - elimination_started
        _progress(
            variant_name, "bareiss_elimination", elimination_started,
            maximum_integer_bits_before=
            elimination["initial_maximum_integer_bits"],
            maximum_integer_bits_encountered=
            elimination["elimination_maximum_integer_bits"],
            row_swap_count=len(elimination["row_swaps"]))

        back_started = time.perf_counter()
        solution = _bareiss_back_substitution(augmented)
        back_seconds = time.perf_counter() - back_started
        _progress(
            variant_name, "rational_back_substitution", back_started,
            solution_count=len(solution))

        replay_started = time.perf_counter()
        for row, rhs in zip(integer_rows, integer_rhs):
            if sum((coefficient * value for coefficient, value in
                    zip(row, solution)), Fraction(0)) != rhs:
                raise ExactReplayError(
                    "selected exact integer system replay is nonzero")
        replay_seconds = time.perf_counter() - replay_started
        _progress(
            variant_name, "selected_system_replay", replay_started,
            equation_count=size)
    return solution, {
        "solver_api": "pure_python_fraction_free_bareiss",
        "solver_domain": "ZZ",
        "selected_system_shape": [size, size],
        "exact_selected_system_replay": True,
        "timeout_guard_enabled": guard_enabled,
        "timeout_seconds": timeout_seconds,
        "bareiss_elimination_seconds": elimination_seconds,
        "rational_back_substitution_seconds": back_seconds,
        "selected_system_replay_seconds": replay_seconds,
        **elimination,
    }


def construct_exact_zero_certificate(
        problem: dict, candidate: np.ndarray, maximum_rank: int,
        *, variant_name: str = "synthetic",
        solve_timeout_seconds: float = 300.0) -> tuple[dict | None, dict]:
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
        if len(interior) and rank < min(A.shape[0], len(free)):
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
        "status_code": None,
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
            build_started = time.perf_counter()
            integer_rows = []
            integer_rhs = []
            row_metadata = []
            exact_residual = problem.get("exact_residual")
            candidate_residuals = (exact_residual(values)
                                   if exact_residual is not None else None)
            for row in selected_rows:
                if candidate_residuals is None:
                    rhs = -_exact_center_difference(problem, int(row))
                    rhs -= sum(
                        (_exact_coefficient(problem, int(row), column)
                         * values[column]
                         for column in range(problem["variable_count"])
                         if column not in selected_set), Fraction(0))
                else:
                    rhs = -candidate_residuals[int(row)]
                    rhs += sum(
                        (_exact_coefficient(problem, int(row), int(column))
                         * values[int(column)]
                         for column in selected_columns), Fraction(0))
                dyadic_row = [
                    _exact_coefficient(problem, int(row), int(column))
                    for column in selected_columns]
                integers, metadata = _dyadic_row_to_integers(
                    [*dyadic_row, rhs])
                integer_rows.append(integers[:-1])
                integer_rhs.append(integers[-1])
                row_metadata.append(metadata)
            build_seconds = time.perf_counter() - build_started
            _progress(
                variant_name, "integer_system_build", build_started,
                row_count=len(integer_rows), column_count=len(selected_columns),
                maximum_row_denominator_power=max(
                    item["common_denominator_power_of_two"]
                    for item in row_metadata),
                maximum_integer_bits=max(
                    item["maximum_integer_bits"] for item in row_metadata))
            solve_started = time.perf_counter()
            solution, solver_evidence = _solve_exact_integer_system(
                integer_rows, integer_rhs, solve_timeout_seconds,
                variant_name)
            solve_seconds = time.perf_counter() - solve_started
            _progress(
                variant_name, "exact_integer_solve_complete", solve_started,
                solver_api=solver_evidence["solver_api"],
                solver_domain=solver_evidence["solver_domain"])
            solver_evidence.update({
                "integer_system_build_seconds": build_seconds,
                "exact_integer_solve_seconds": solve_seconds,
                "row_denominator_power_min": min(
                    item["common_denominator_power_of_two"]
                    for item in row_metadata),
                "row_denominator_power_max": max(
                    item["common_denominator_power_of_two"]
                    for item in row_metadata),
                "maximum_integer_bits": max(
                    item["maximum_integer_bits"] for item in row_metadata),
            })
            diagnostics["exact_solver"] = solver_evidence
            for column, value in zip(selected_columns, solution):
                values[int(column)] = value
        else:
            diagnostics["exact_solver"] = {
                "solver_api": "empty_exact_system",
                "solver_domain": "ZZ",
                "selected_system_shape": [0, 0],
                "exact_selected_system_replay": True,
                "preferred_solver_failures": [],
            }
        box_started = time.perf_counter()
        outside = [
            index for index, value in enumerate(values)
            if not (_fraction(problem["low"][index]) <= value
                    <= _fraction(problem["high"][index]))
        ]
        box_seconds = time.perf_counter() - box_started
        diagnostics["exact_box_check_seconds"] = box_seconds
        _progress(
            variant_name, "exact_box_check", box_started,
            outside_box_variable_count=len(outside))
        if outside:
            diagnostics.update({
                "verified": False,
                "status_code": EXACT_SOLUTION_OUTSIDE_BOX,
                "reason": "exact algebraic solution violates authenticated box",
                "outside_box_variable_count": len(outside),
                "first_outside_box_variable": int(outside[0]),
            })
            return None, diagnostics
        certificate = {
            "schema": EXACT_SCHEMA,
            "problem_sha256": problem["problem_sha256"],
            "generator_ids_sha256": _json_sha(problem["ids"]),
            "construction": (
                "dyadic_row_cleared_bareiss_correction_on_"
                "qr_selected_full_rank_subsystem"),
            "reference_coordinate": problem["dimension"] - 1,
            "selected_rows": [int(value) for value in selected_rows],
            "selected_columns": [int(value) for value in selected_columns],
            "xi_rationals": [
                {"numerator": str(value.numerator),
                 "denominator": str(value.denominator)} for value in values],
        }
        replay_started = time.perf_counter()
        try:
            checked = verify_exact_zero_certificate(problem, certificate)
        except Exception as error:
            raise ExactReplayError(
                f"full authenticated replay failed: {error}") from error
        replay_seconds = time.perf_counter() - replay_started
        diagnostics["exact_full_replay_seconds"] = replay_seconds
        _progress(
            variant_name, "full_127_equation_replay", replay_started,
            exact_equalities=checked["exact_equalities"],
            exact_box_constraints=checked["exact_box_constraints"])
        diagnostics.update({
            "verified": True, "status_code": EXACT_ZERO_VERIFIED, **checked})
        return certificate, diagnostics
    except ExactReplayError as error:
        diagnostics.update({
            "verified": False,
            "status_code": EXACT_REPLAY_FAILED,
            "reason": f"{type(error).__name__}: {error}",
        })
        return None, diagnostics
    except ExactBareissTimeoutError as error:
        diagnostics.update({
            "verified": False,
            "status_code": EXACT_BAREISS_TIMEOUT,
            "reason": f"{type(error).__name__}: {error}",
        })
        return None, diagnostics
    except ExactBareissSingularError as error:
        diagnostics.update({
            "verified": False,
            "status_code": EXACT_BAREISS_SINGULAR,
            "reason": f"{type(error).__name__}: {error}",
        })
        return None, diagnostics
    except ExactBareissNonexactDivisionError as error:
        diagnostics.update({
            "verified": False,
            "status_code": EXACT_BAREISS_NONEXACT,
            "reason": f"{type(error).__name__}: {error}",
        })
        return None, diagnostics
    except ExactIntegerSystemError as error:
        diagnostics.update({
            "verified": False,
            "status_code": EXACT_BAREISS_SINGULAR,
            "reason": f"{type(error).__name__}: {error}",
        })
        return None, diagnostics
    except Exception as error:
        diagnostics.update({
            "verified": False,
            "status_code": EXACT_REPLAY_FAILED,
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


def dual_scaling_search(problem: dict, primal_candidate: np.ndarray,
                        variant_name: str = "synthetic") -> dict:
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
        started = time.perf_counter()
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
        _progress(
            variant_name, "dual_start_complete", started,
            start=name, optimizer_success=optimizer["success"],
            outward_safe_lower=
                checked["psd_dual_candidate_lower_outward_safe"],
            witness_l2=float(np.linalg.norm(witness)))
    best = max(rows, key=lambda row: row["outward_safe_lower"])
    return {
        "zero_baseline": exact_zero_dual_baseline(problem["dimension"]),
        "candidates": rows, "best": best,
    }


def near_zero_gate(problem: dict, primal: dict) -> dict:
    relative_tolerance = 1e-9
    state_tolerance = relative_tolerance * primal["candidate_state_scale"]
    scaled_residual = primal["max_scaled_equality_residual"]
    centered_rms = primal["centered_residual_rms"]
    near = (scaled_residual <= relative_tolerance
            and centered_rms <= state_tolerance
            and primal["maximum_box_violation"] == 0.0)
    return {
        "is_compelling_near_zero": bool(near),
        "relative_equality_tolerance": relative_tolerance,
        "state_residual_tolerance": state_tolerance,
        "observed_scaled_equality_residual": scaled_residual,
        "observed_centered_residual_rms": centered_rms,
        "requires_exact_reconstruction_for_zero_claim": True,
    }


def _decide_variant_unbounded(name: str, variant: dict, exact_max_rank: int,
                   certificate_dir: Path, *, skip_exact: bool = False,
                   exact_only_if_near_zero: bool = True,
                   fast_first: bool = True,
                   exact_solve_timeout_seconds: float = 300.0,
                   require_exact_exclusion: bool = False) -> dict:
    started = time.perf_counter()
    problem = centered_problem(
        variant["center"], variant["generators"],
        variant["low"], variant["high"], variant["ids"])
    _progress(name, "variant_start", started,
              generator_count=problem["variable_count"])
    if not fast_first:
        raise RuntimeError("non-fast-first execution is intentionally unsupported")
    candidate, primal = numerical_primal_search(problem, name)
    gate = near_zero_gate(problem, primal)
    dual_started = time.perf_counter()
    _progress(name, "dual_stage_start", dual_started)
    dual = dual_scaling_search(problem, candidate, name)
    _progress(
        name, "dual_stage_complete", dual_started,
        best_outward_safe_lower=dual["best"]["outward_safe_lower"])
    exact = None
    upper = None
    certificate_record = None
    exclusion = None
    exclusion_record = None
    if require_exact_exclusion:
        exclusion_started = time.perf_counter()
        _progress(name, "exact_farkas_replay_start", exclusion_started)
        exclusion = construct_exact_exclusion_certificate(
            problem, [float.fromhex(x) for x in dual["best"]["witness_binary64_hex"]])
        _progress(name, "exact_farkas_replay_complete", exclusion_started,
                  verified=exclusion is not None)
    if exclusion is not None or (not require_exact_exclusion and dual["best"]["outward_safe_lower"] > 0.0):
        exact_status = {
            "attempted": False,
            "reason": ("exact original-equation/box Farkas replay verified" if require_exact_exclusion
                       else "positive outward-safe dual lower established first"),
        }
        decision = EXCLUDED if require_exact_exclusion else DECISION_POSITIVE
    elif skip_exact:
        exact_status = {
            "attempted": False, "reason": "exact stage disabled by --skip-exact"}
        decision = DECISION_UNRESOLVED
    elif exact_only_if_near_zero and not gate["is_compelling_near_zero"]:
        exact_status = {
            "attempted": False,
            "reason": "numerical candidate did not pass near-zero triage",
        }
        decision = DECISION_UNRESOLVED
    else:
        exact_started = time.perf_counter()
        _progress(name, "exact_reconstruction_start", exact_started,
                  numerical_rank_limit=exact_max_rank)
        exact, exact_status = construct_exact_zero_certificate(
            problem, candidate, exact_max_rank, variant_name=name,
            solve_timeout_seconds=exact_solve_timeout_seconds)
        _progress(
            name, "exact_reconstruction_complete", exact_started,
            verified=bool(exact_status.get("verified", False)),
            reason=exact_status.get("reason"))
        if exact is not None:
            decision = FEASIBLE if require_exact_exclusion else DECISION_ZERO
        else:
            upper_started = time.perf_counter()
            _progress(name, "exact_candidate_upper_start", upper_started)
            upper = exact_candidate_upper(problem, candidate)
            _progress(
                name, "exact_candidate_upper_complete", upper_started,
                variance_upper_outward_binary64=
                    upper["variance_upper_outward_binary64"])
            decision = DECISION_UNRESOLVED
    if require_exact_exclusion and decision == DECISION_UNRESOLVED:
        decision = INCONCLUSIVE
    if exact is not None:
        _progress(name, "exact_witness_persist_and_replay_start", time.perf_counter())
        certificate_path = certificate_dir / f"{name}_exact_zero_witness.json"
        persisted = _atomic_json(certificate_path, exact)
        # Re-read and recheck the serialized rational witness.
        verify_exact_zero_certificate(problem, persisted)
        certificate_record = {
            "path": str(certificate_path),
            "sha256": oracle.sha256(certificate_path),
            "record_sha256": persisted["record_sha256"],
        }
    if exclusion is not None:
        _progress(name, "exact_exclusion_persist_and_replay_start", time.perf_counter())
        exclusion_path = certificate_dir / f"{name}_exact_exclusion.json"
        persisted = _atomic_json(exclusion_path, exclusion)
        verify_exact_exclusion_certificate(problem, persisted)
        exclusion_record = {"path": str(exclusion_path), "sha256": oracle.sha256(exclusion_path),
                            "record_sha256": persisted["record_sha256"]}
    record = {
        "variant": name, "decision": decision,
        "generator_count": problem["variable_count"],
        "native_generator_count": variant["native_generator_count"],
        "numerical_generator_count": variant["numerical_generator_count"],
        "primal_numerical_search": primal,
        "near_zero_gate": gate,
        "primal_exact_binary64_candidate_upper": upper,
        "exact_zero_certificate_status": exact_status,
        "exact_zero_certificate": certificate_record,
        "exact_exclusion_certificate": exclusion_record,
        "dual_search": dual,
        "runtime_seconds": time.perf_counter() - started,
    }
    _progress(
        name, "variant_complete", started, decision=decision,
        exact_attempted=bool(exact_status.get("attempted", False)))
    return record


def _exclusion_attempt_worker(connection, args, kwargs):
    """Same computation, supervised for wall time; no numerical changes."""
    original_progress = globals()["_progress"]
    def progress(variant, stage, started, **fields):
        connection.send(("stage", {"stage": stage, **fields}))
        original_progress(variant, stage, started, **fields)
    globals()["_progress"] = progress
    try:
        connection.send(("result", _decide_variant_unbounded(*args, **kwargs)))
    except Exception as error:
        # Transport/re-raise errors, NEVER reinterpret them as numerical failure.
        connection.send(("error", error))
    finally:
        globals()["_progress"] = original_progress
        connection.close()


def decide_variant(name: str, variant: dict, exact_max_rank: int,
                   certificate_dir: Path, *, skip_exact: bool = False,
                   exact_only_if_near_zero: bool = True,
                   fast_first: bool = True,
                   exact_solve_timeout_seconds: float = 300.0,
                   require_exact_exclusion: bool = False) -> dict:
    kwargs = dict(skip_exact=skip_exact, exact_only_if_near_zero=exact_only_if_near_zero,
                  fast_first=fast_first, exact_solve_timeout_seconds=exact_solve_timeout_seconds,
                  require_exact_exclusion=require_exact_exclusion)
    args = (name, variant, exact_max_rank, certificate_dir)
    if not require_exact_exclusion:
        return _decide_variant_unbounded(*args, **kwargs)  # Legacy policy unchanged.
    if not math.isfinite(exact_solve_timeout_seconds) or exact_solve_timeout_seconds <= 0:
        raise RuntimeError("exact exclusion wall-time budget must be finite and positive")
    if "fork" not in multiprocessing.get_all_start_methods():
        raise RuntimeError("hard CPU exclusion watchdog requires POSIX fork; cannot silently disable it")
    context = multiprocessing.get_context("fork")
    receiving, sending = context.Pipe(duplex=False)
    process = context.Process(target=_exclusion_attempt_worker, args=(sending, args, kwargs))
    started = time.perf_counter()
    deadline = started + exact_solve_timeout_seconds
    stage = {"stage": "exclusion_attempt_start"}
    try:
        process.start()
        sending.close()
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0 or not receiving.poll(remaining):
                break
            try:
                kind, value = receiving.recv()
            except EOFError as error:
                raise RuntimeError(f"exclusion worker exited without result at {stage['stage']}") from error
            # A result arriving after the deadline cannot acquire authority.
            if time.perf_counter() >= deadline:
                break
            if kind == "stage":
                stage = value
            elif kind == "error":
                raise value
            elif kind == "result":
                return value
            else:
                raise RuntimeError("exclusion watchdog received an invalid message")
        # Terminate before publishing INCONCLUSIVE, including native/C work.
        process.terminate()
        process.join(.1)
        if process.is_alive():
            process.kill()
            process.join(.1)
        elapsed = time.perf_counter() - started
        diagnostics = {"timeout_seconds": exact_solve_timeout_seconds,
                       "elapsed_seconds": elapsed, "stage": stage["stage"],
                       "last_stage_diagnostics": stage, "worker_terminated": not process.is_alive(),
                       "scope": "entire_proposal_exclusion_reconstruction_persistence_and_replay_attempt"}
        _progress(name, "exact_exclusion_timeout", started,
                  timeout_stage=stage["stage"], timeout_seconds=exact_solve_timeout_seconds,
                  worker_terminated=diagnostics["worker_terminated"])
        return {"variant": name, "decision": INCONCLUSIVE, "reason": EXACT_EXCLUSION_TIMEOUT,
                "generator_count": len(variant["ids"]),
                "native_generator_count": variant["native_generator_count"],
                "numerical_generator_count": variant["numerical_generator_count"],
                "exact_zero_certificate": None, "exact_exclusion_certificate": None,
                "exact_zero_certificate_status": {"verified": False, "reason": EXACT_EXCLUSION_TIMEOUT},
                "timeout_diagnostics": diagnostics, "runtime_seconds": elapsed}
    finally:
        sending.close()
        receiving.close()
        if process.pid is not None:
            if process.is_alive():
                process.kill()
            process.join(.1)


def _capture_variant(state: dict, token: int) -> dict:
    return oracle._variant({"weights": state["weights"].numpy(),
                            "low": state["range_low"].numpy(), "high": state["range_high"].numpy(),
                            "ids": state["proof"]["ids"], "reasons": state["proof"]["reasons"],
                            "num_tokens": state["proof"]["num_tokens"]}, token, False)


def _variant_problem(variant):
    return centered_problem(variant["center"], variant["generators"],
                            variant["low"], variant["high"], variant["ids"])


def _authorize_all_tokens(report_path: Path, identity: dict, state: dict) -> dict:
    if report_path is None:
        raise RuntimeError("--token-index all requires --token4-exclusion-report; token 4 first")
    report = cluster_common.verified_json(report_path)
    if (report.get("schema") != SCHEMA or report.get("property_id") != identity["identity"]["property_id"] or
            report.get("requested_token_scope") != "4" or report.get("final_status") != EXCLUDED or
            report.get("authenticated_capture") != identity or len(report.get("results", [])) != 1):
        raise RuntimeError("all-token authorization is not an authenticated token-4 exclusion")
    row = report["results"][0]
    if row.get("token_index") != 4 or row.get("decision") != EXCLUDED:
        raise RuntimeError("token-4 exclusion result differs")
    certificate = row.get("exact_exclusion_certificate") or {}
    path = Path(certificate.get("path", ""))
    if not path.is_file() or oracle.sha256(path) != certificate.get("sha256"):
        raise RuntimeError("token-4 exclusion certificate SHA differs")
    # Report status itself has no authority: independently replay again.
    verify_exact_exclusion_certificate(_variant_problem(_capture_variant(state, 4)),
                                       cluster_common.verified_json(path))
    return row


def _execute_benchmark_capture(manifest_path, output_path, exact_max_rank, *,
                               fast_first, skip_exact, exact_only_if_near_zero,
                               exact_solve_timeout_seconds, variant,
                               expected_property_id, token_index, token4_exclusion_report):
    import capture_benchmark24_block2_output_zero_variance_v1 as capture
    if expected_property_id not in capture.TARGET_TOKENS:
        raise RuntimeError("new capture requires its exact --expected-property-id")
    if variant not in ("complete_post_reduction", "all"):
        raise RuntimeError("new capture oracle must use ALL authenticated generators/ranges")
    target_token = capture.TARGET_TOKENS[expected_property_id]
    token_index = str(target_token) if token_index is None else str(token_index)
    if expected_property_id == capture.SEPARATOR_PROPERTY_ID:
        if token_index != "8" or token4_exclusion_report is not None:
            raise RuntimeError("s003 oracle permits native token 8 only; no all-token mode")
        # The existing whole-attempt watchdog is unchanged; cap this target at
        # the preregistered 180 seconds, including proposals and exact replay.
        exact_solve_timeout_seconds = min(exact_solve_timeout_seconds, 180.)
    elif token_index not in ("4", "all"):
        raise RuntimeError("first oracle target is native tensor token 4; then authorized all")
    if output_path.exists() or output_path.with_suffix(".partial.json").exists():
        raise RuntimeError("refusing to overwrite a zero-variance diagnostic")
    started = time.perf_counter()
    state, identity = capture.verify_capture(manifest_path)
    if identity["identity"]["property_id"] != expected_property_id:
        raise RuntimeError("capture differs from exact --expected-property-id")
    _progress("complete_post_reduction", "loading_authentication_complete", started,
              artifact_sha256=identity["artifact_sha256"], token_index=token_index)
    rows = []
    if token_index == "all":
        rows.append(_authorize_all_tokens(token4_exclusion_report, identity, state))
        tokens = [i for i in range(state["proof"]["num_tokens"]) if i != 4]
    else:
        tokens = [target_token]
    for token in tokens:
        row = decide_variant(
            f"complete_post_reduction_token_{token}", _capture_variant(state, token),
            exact_max_rank, output_path.parent / f"{output_path.stem}_certificates",
            skip_exact=skip_exact, exact_only_if_near_zero=exact_only_if_near_zero,
            fast_first=fast_first, exact_solve_timeout_seconds=exact_solve_timeout_seconds,
            require_exact_exclusion=True)
        row["token_index"] = token
        rows.append(row)
        print(json.dumps({"event": "ZERO_VARIANCE_VARIANT_RESULT", "result": row}, sort_keys=True), flush=True)
        _atomic_json(output_path.with_suffix(".partial.json"), {
            "schema": SCHEMA, "property_id": expected_property_id, "authenticated_capture": identity,
            "requested_token_scope": token_index, "results": rows, "partial": True,
            "scientific_queries": 0, "bound_calls": 0})
        if row["decision"] == FEASIBLE:
            break  # One exact token witness already answers the existential question.
    final_status = (FEASIBLE if any(r["decision"] == FEASIBLE for r in rows) else
                    EXCLUDED if len(rows) == (state["proof"]["num_tokens"] if token_index == "all" else 1)
                    and all(r["decision"] == EXCLUDED for r in rows) else INCONCLUSIVE)
    timeouts = [r for r in rows if r.get("reason") == EXACT_EXCLUSION_TIMEOUT]
    return _atomic_json(output_path, {
        "schema": SCHEMA, "property_id": expected_property_id, "stage_label": STAGE,
        "authenticated_capture": identity, "requested_token_scope": token_index,
        "token_index_convention": "zero_based_native_tensor", "results": rows,
        "final_status": final_status, "complete_state_decision": final_status,
        "reason": EXACT_EXCLUSION_TIMEOUT if final_status == INCONCLUSIVE and timeouts else None,
        "timeout_diagnostics": timeouts[0]["timeout_diagnostics"] if timeouts else None,
        "decision_scope": "any_token_zero_variance" if token_index == "all" else f"token_{target_token}_only",
        "exact_solve_timeout_seconds": exact_solve_timeout_seconds,
        "interpretation": {FEASIBLE: "Captured abstract state admits a constant vector; tightening alone cannot exclude it.",
                           EXCLUDED: "Exact zero excluded only for the stated token scope; no production variance repair implemented.",
                           INCONCLUSIVE: "No exact decision for the stated scope; numerical statuses have no authority."}[final_status],
        "optimizer_is_untrusted": True, "all_numerical_generators_included": True,
        "zero_decision_requires_exact_rational_witness": True,
        "exclusion_requires_full_original_equation_and_box_farkas_replay": True,
        "runtime_seconds": time.perf_counter() - started, "scientific_queries": 0, "bound_calls": 0})


def execute(manifest_path: Path, output_path: Path,
            exact_max_rank: int = 128, *, fast_first: bool = True,
            skip_exact: bool = False,
            exact_only_if_near_zero: bool = True,
            exact_solve_timeout_seconds: float = 300.0,
            variant: str = "all", expected_property_id: str | None = None,
            token_index: str | None = None, token4_exclusion_report: Path | None = None) -> dict:
    import capture_benchmark24_block2_output_zero_variance_v1 as capture
    if cluster_common.verified_json(manifest_path).get("schema") == capture.MANIFEST_SCHEMA:
        return _execute_benchmark_capture(
            manifest_path, output_path, exact_max_rank, fast_first=fast_first,
            skip_exact=skip_exact, exact_only_if_near_zero=exact_only_if_near_zero,
            exact_solve_timeout_seconds=exact_solve_timeout_seconds, variant=variant,
            expected_property_id=expected_property_id, token_index=token_index,
            token4_exclusion_report=token4_exclusion_report)
    if expected_property_id not in (None, PROPERTY_ID) or token_index is not None or token4_exclusion_report is not None:
        raise RuntimeError("legacy capture property/token interface differs")
    selected_variant = variant
    load_started = time.perf_counter()
    variants, identity = _load_authenticated_variants(manifest_path)
    if selected_variant != "all":
        variants = {selected_variant: variants[selected_variant]}
    _progress("all", "loading_authentication_complete", load_started,
              variant_count=len(variants),
              artifact_sha256=identity["artifact_sha256"])
    certificate_dir = output_path.parent / "exact_zero_witnesses"
    results = []
    for name, variant_state in variants.items():
        row = decide_variant(
            name, variant_state, exact_max_rank, certificate_dir,
            skip_exact=skip_exact,
            exact_only_if_near_zero=exact_only_if_near_zero,
            fast_first=fast_first,
            exact_solve_timeout_seconds=exact_solve_timeout_seconds)
        results.append(row)
        print(json.dumps({
            "event": "ZERO_VARIANCE_VARIANT_RESULT",
            "result": row,
        }, sort_keys=True), flush=True)
        # Persist each completed variant immediately, so diagnostics survive
        # interruption of a later exact stage.
        _atomic_json(output_path.with_suffix(".partial.json"), {
            "schema": SCHEMA, "property_id": PROPERTY_ID,
            "authenticated_capture": identity, "results": results,
            "partial": True, "scientific_queries": 0, "bound_calls": 0,
        })
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
        "execution_policy": {
            "fast_first": fast_first, "skip_exact": skip_exact,
            "exact_only_if_near_zero": exact_only_if_near_zero,
            "exact_solve_timeout_seconds": exact_solve_timeout_seconds,
            "variant": selected_variant,
        },
        "scientific_queries": 0, "bound_calls": 0,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-property-id")
    parser.add_argument("--token-index", choices=("4", "8", "all"),
                        help="Native zero-based token: s003 permits token 8 only; legacy s004 uses token 4")
    parser.add_argument("--token4-exclusion-report", type=Path)
    parser.add_argument("--exact-max-rank", type=int, default=128)
    parser.add_argument(
        "--exact-solve-timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--variant", choices=(*VARIANT_NAMES, "all"), default="all")
    parser.add_argument(
        "--fast-first", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-exact", action="store_true")
    parser.add_argument(
        "--exact-only-if-near-zero", action=argparse.BooleanOptionalAction,
        default=True)
    args = parser.parse_args()
    if not 0 <= args.exact_max_rank <= 128:
        raise RuntimeError("exact rank cap is outside [0,128]")
    if not 0 < args.exact_solve_timeout_seconds <= 3600:
        raise RuntimeError("exact solve timeout is outside (0,3600]")
    report = execute(args.capture_manifest.resolve(), args.output.resolve(),
                     args.exact_max_rank, fast_first=args.fast_first,
                     skip_exact=args.skip_exact,
                     exact_only_if_near_zero=args.exact_only_if_near_zero,
                     exact_solve_timeout_seconds=
                     args.exact_solve_timeout_seconds,
                     variant=args.variant, expected_property_id=args.expected_property_id,
                     token_index=args.token_index,
                     token4_exclusion_report=args.token4_exclusion_report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
