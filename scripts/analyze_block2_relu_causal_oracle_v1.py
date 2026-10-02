#!/usr/bin/env python3
"""Proof-carrying CPU oracle for Block-2 exact-ReLU cancellation.

Numerical HiGHS LP/MILP results are proposals only.  Scientific conclusions
require either an exact rational Farkas replay or an exact rational shared-
source ReLU witness replay.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.linalg import qr
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, csr_matrix, eye, hstack, vstack


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


frontier = _module(
    "block2_ffn_frontier_authenticated",
    REPO / "scripts/analyze_block2_ffn_joint_cancellation_frontier_v1.py")
zero = frontier.zero
cluster_common = frontier.cluster_common

SCHEMA = "CORET_BLOCK2_RELU_CAUSAL_ORACLE_V1"
FINAL_HULL_EXCLUDED = "RELU_HULL_CANCELLATION_EXCLUDED_EXACT"
FINAL_EXACT_FEASIBLE = "EXACT_RELU_CANCELLATION_FEASIBLE"
FINAL_MILP_UNCERTIFIED = "EXACT_RELU_MILP_INFEASIBLE_UNCERTIFIED"
FINAL_INCONCLUSIVE = "RELU_CAUSAL_ORACLE_INCONCLUSIVE"
FARKAS_SCHEMA = "CORET_BLOCK2_RELU_FARKAS_CERTIFICATE_V1"
WITNESS_SCHEMA = "CORET_BLOCK2_EXACT_RELU_CANCELLATION_WITNESS_V1"


def _fraction(value) -> Fraction:
    return Fraction.from_float(float(value))


def _sha_json(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _atomic_json(path: Path, value: dict) -> dict:
    payload = dict(value)
    payload["record_sha256"] = cluster_common.canonical(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return cluster_common.verified_json(path)


def _exact_affine_bounds(center: np.ndarray, generators: np.ndarray,
                         low: np.ndarray, high: np.ndarray
                         ) -> tuple[list[Fraction], list[Fraction]]:
    lowers, uppers = [], []
    for coordinate in range(center.size):
        lo = hi = _fraction(center[coordinate])
        for row, lower, upper in zip(generators, low, high):
            coefficient = _fraction(row[coordinate])
            first, second = coefficient * _fraction(lower), coefficient * _fraction(upper)
            lo += min(first, second)
            hi += max(first, second)
        lowers.append(lo)
        uppers.append(hi)
    return lowers, uppers


class ReluCancellationProblem:
    def __init__(self, preactivation: dict, residual: dict,
                 weight: np.ndarray, bias: np.ndarray, label="block2_relu"):
        h, r, source = frontier._align_sources(preactivation, residual)
        self.h, self.r, self.source = h, r, source
        self.weight = np.asarray(weight, dtype=np.float64)
        self.bias = np.asarray(bias, dtype=np.float64)
        self.label = label
        if (h["center"].ndim != 1 or r["center"].shape != (128,)
                or self.weight.shape != (128, h["center"].size)
                or self.bias.shape != (128,)):
            raise RuntimeError("authenticated ReLU cancellation dimensions differ")
        self.n = len(source["ids"])
        self.m = h["center"].size
        started = time.perf_counter()
        self.lower_exact, self.upper_exact = _exact_affine_bounds(
            h["center"], h["generators"], source["low"], source["high"])
        self.bound_seconds = time.perf_counter() - started
        self.inactive = [index for index, upper in enumerate(self.upper_exact)
                         if upper <= 0]
        self.active = [index for index, (lower, upper) in enumerate(
            zip(self.lower_exact, self.upper_exact))
            if upper > 0 and lower >= 0]
        self.unstable = [index for index, (lower, upper) in enumerate(
            zip(self.lower_exact, self.upper_exact)) if lower < 0 < upper]
        if len(self.inactive) + len(self.active) + len(self.unstable) != self.m:
            raise RuntimeError("exact ReLU bound partition is incomplete")
        self._build_numeric_hull()
        self.identity_sha256 = _sha_json({
            "ids": source["ids"],
            "low_hex": [float(value).hex() for value in source["low"]],
            "high_hex": [float(value).hex() for value in source["high"]],
            "h_center": frontier._array_sha(h["center"]),
            "h_generators": frontier._array_sha(h["generators"]),
            "r_center": frontier._array_sha(r["center"]),
            "r_generators": frontier._array_sha(r["generators"]),
            "weight": frontier._array_sha(self.weight),
            "bias": frontier._array_sha(self.bias),
        })

    def _build_numeric_hull(self):
        active = np.asarray(self.active, dtype=np.int64)
        unstable = np.asarray(self.unstable, dtype=np.int64)
        if len(active):
            center = self.r["center"] + self.bias + (
                self.weight[:, active] @ self.h["center"][active])
            generators = self.r["generators"] + (
                self.h["generators"][:, active] @ self.weight[:, active].T)
        else:
            center = self.r["center"] + self.bias
            generators = self.r["generators"].copy()
        ref_center = center[1:] - center[0]
        ref_generators = (generators[:, 1:] - generators[:, [0]]).T
        unstable_weight = (self.weight[1:, unstable]
                           - self.weight[[0], :][:, unstable])
        self.E = np.concatenate((ref_generators, unstable_weight), axis=1)
        self.f = -ref_center

        rows, columns, values, rhs = [], [], [], []
        for local, neuron in enumerate(self.unstable):
            # h-y <= 0.
            row = len(rhs)
            nz = np.flatnonzero(self.h["generators"][:, neuron])
            rows.extend([row] * len(nz)); columns.extend(nz.tolist())
            values.extend(self.h["generators"][nz, neuron].tolist())
            rows.append(row); columns.append(self.n + local); values.append(-1.0)
            rhs.append(-float(self.h["center"][neuron]))
            # y-alpha*h <= -alpha*l.
            lower = float(self.lower_exact[neuron])
            upper = float(self.upper_exact[neuron])
            alpha = upper / (upper - lower)
            row = len(rhs)
            rows.extend([row] * len(nz)); columns.extend(nz.tolist())
            values.extend((-alpha * self.h["generators"][nz, neuron]).tolist())
            rows.append(row); columns.append(self.n + local); values.append(1.0)
            rhs.append(alpha * (float(self.h["center"][neuron]) - lower))
        self.A_triangle = coo_matrix(
            (values, (rows, columns)),
            shape=(len(rhs), self.n + len(self.unstable))).tocsr()
        self.b_triangle = np.asarray(rhs, dtype=np.float64)

    def hull_bounds(self) -> list[tuple[float | None, float | None]]:
        bounds = list(zip(self.source["low"], self.source["high"]))
        bounds.extend((0.0, float(self.upper_exact[index]))
                      for index in self.unstable)
        return bounds

    def exact_h(self, values: list[Fraction]) -> list[Fraction]:
        return [
            _fraction(center) + sum(
                (_fraction(self.h["generators"][row, coordinate]) * xi
                 for row, xi in enumerate(values)), Fraction(0))
            for coordinate, center in enumerate(self.h["center"])]

    def fixed_pattern_model(self, active_pattern: list[bool]):
        if len(active_pattern) != self.m:
            raise RuntimeError("activation pattern length differs")
        masked = self.weight.copy()
        masked[:, np.logical_not(active_pattern)] = 0.0
        pre = {
            "center": self.h["center"], "generators": self.h["generators"],
            "low": self.source["low"], "high": self.source["high"],
            "ids": self.source["ids"], "masks": self.source["masks"],
            "reasons": self.source["reasons"], "num_tokens": 1,
        }
        residual = {
            "center": self.r["center"], "generators": self.r["generators"],
            "low": self.source["low"], "high": self.source["high"],
            "ids": self.source["ids"], "masks": self.source["masks"],
            "reasons": self.source["reasons"], "num_tokens": 1,
        }
        return frontier.ExactJointAffine(
            pre, residual, weight=masked, bias=self.bias,
            label="exact_fixed_relu_pattern")

    def verify_pattern(self, values: list[Fraction], pattern: list[bool]) -> dict:
        h = self.exact_h(values)
        violations = [index for index, (value, active) in enumerate(zip(h, pattern))
                      if (active and value < 0) or (not active and value > 0)]
        if violations:
            raise RuntimeError(
                f"exact activation sign check differs: {violations[0]}")
        return {"activation_sign_check": True,
                "checked_preactivations": len(h)}

    def exact_lp(self):
        """Materialize exact sparse A,b,E,f only for Farkas replay."""
        variable_count = self.n + len(self.unstable)
        A, b = [], []
        # Source and y bounds are explicit inequalities in the Farkas system.
        for index in range(self.n):
            A.append({index: Fraction(1)}); b.append(_fraction(self.source["high"][index]))
            A.append({index: Fraction(-1)}); b.append(-_fraction(self.source["low"][index]))
        for local, neuron in enumerate(self.unstable):
            y = self.n + local
            A.append({y: Fraction(-1)}); b.append(Fraction(0))
            row = {y: Fraction(-1)}
            for source in range(self.n):
                value = _fraction(self.h["generators"][source, neuron])
                if value: row[source] = value
            A.append(row); b.append(-_fraction(self.h["center"][neuron]))
            lower, upper = self.lower_exact[neuron], self.upper_exact[neuron]
            alpha = upper / (upper - lower)
            row = {y: Fraction(1)}
            for source in range(self.n):
                value = -alpha * _fraction(self.h["generators"][source, neuron])
                if value: row[source] = value
            A.append(row)
            b.append(alpha * (_fraction(self.h["center"][neuron]) - lower))
        E, f = [], []
        active = self.active
        for output in range(1, 128):
            row = {}
            for source in range(self.n):
                value = (_fraction(self.r["generators"][source, output])
                         - _fraction(self.r["generators"][source, 0]))
                for neuron in active:
                    value += ((_fraction(self.weight[output, neuron])
                               - _fraction(self.weight[0, neuron]))
                              * _fraction(self.h["generators"][source, neuron]))
                if value: row[source] = value
            for local, neuron in enumerate(self.unstable):
                value = (_fraction(self.weight[output, neuron])
                         - _fraction(self.weight[0, neuron]))
                if value: row[self.n + local] = value
            constant = (_fraction(self.r["center"][output])
                        - _fraction(self.r["center"][0])
                        + _fraction(self.bias[output]) - _fraction(self.bias[0]))
            for neuron in active:
                constant += ((_fraction(self.weight[output, neuron])
                              - _fraction(self.weight[0, neuron]))
                             * _fraction(self.h["center"][neuron]))
            E.append(row); f.append(-constant)
        return A, b, E, f, variable_count


def solve_hull(problem: ReluCancellationProblem,
               timeout: float = 300.0) -> tuple[object, dict]:
    started = time.perf_counter()
    variable_count = problem.n + len(problem.unstable)
    if variable_count == 0:
        feasible = bool(np.array_equal(problem.f, np.zeros_like(problem.f)))
        result = SimpleNamespace(
            success=feasible, status=0 if feasible else 2,
            message=("constant equality system is feasible" if feasible
                     else "constant equality system is infeasible"),
            x=np.empty(0, dtype=np.float64))
        return result, {
            "solver_backend": "exact_constant_system_precheck",
            "solver_status": int(result.status),
            "solver_message": result.message,
            "runtime_seconds": time.perf_counter() - started,
            "feasible": feasible, "variable_count": 0,
            "equality_count": int(problem.E.shape[0]),
            "inequality_count": 0,
        }
    result = linprog(
        np.zeros(variable_count),
        A_ub=problem.A_triangle, b_ub=problem.b_triangle,
        A_eq=problem.E, b_eq=problem.f, bounds=problem.hull_bounds(),
        method="highs", options={"presolve": True, "time_limit": timeout})
    return result, {
        "solver_backend": "scipy.optimize.linprog/highs",
        "solver_status": int(result.status), "solver_message": str(result.message),
        "runtime_seconds": time.perf_counter() - started,
        "feasible": bool(result.success),
        "variable_count": variable_count,
        "equality_count": int(problem.E.shape[0]),
        "inequality_count": int(2 * problem.n + 3 * len(problem.unstable)),
    }


def _verify_farkas(problem: ReluCancellationProblem,
                   lambdas: list[Fraction], mus: list[Fraction]) -> dict:
    A, b, E, f, variables = problem.exact_lp()
    if len(lambdas) != len(A) or len(mus) != len(E):
        raise RuntimeError("Farkas certificate dimensions differ")
    if any(value < 0 for value in lambdas):
        raise RuntimeError("Farkas lambda is negative")
    stationarity = [Fraction(0) for _ in range(variables)]
    for multiplier, row in zip(lambdas, A):
        for column, value in row.items(): stationarity[column] += multiplier * value
    for multiplier, row in zip(mus, E):
        for column, value in row.items(): stationarity[column] += multiplier * value
    if any(stationarity):
        raise RuntimeError("Farkas stationarity replay differs")
    contradiction = sum((x * value for x, value in zip(lambdas, b)), Fraction(0))
    contradiction += sum((x * value for x, value in zip(mus, f)), Fraction(0))
    if contradiction >= 0:
        raise RuntimeError("Farkas contradiction is not strict")
    return {"exact_stationarity": True, "exact_lambda_nonnegative": True,
            "contradiction_numerator": str(contradiction.numerator),
            "contradiction_denominator": str(contradiction.denominator)}


def _rational_row_to_integers(values: list[Fraction]) -> list[int]:
    denominator = 1
    for value in values:
        denominator = math.lcm(denominator, value.denominator)
    integers = [value.numerator * (denominator // value.denominator)
                for value in values]
    content = 0
    for value in integers:
        content = math.gcd(content, abs(value))
    content = max(content, 1)
    integers = [value // content for value in integers]
    first = next((value for value in integers if value), 0)
    return [-value for value in integers] if first < 0 else integers


def _repair_farkas(problem: ReluCancellationProblem, candidate: np.ndarray,
                   numeric_matrix: np.ndarray,
                   timeout: float = 30.0) -> tuple[list[Fraction], list[Fraction], dict]:
    """Repair a small numerical ray using exact fraction-free elimination."""
    A, b, E, f, variables = problem.exact_lp()
    equation_count = variables + 1
    column_count = len(A) + len(E)
    if equation_count > 512:
        raise RuntimeError("exact Farkas repair exceeds bounded equation cap")
    objective = float(numeric_matrix[-1] @ candidate)
    if not math.isfinite(objective) or objective >= 0:
        raise RuntimeError("numerical Farkas proposal lacks contradiction")
    candidate = np.asarray(candidate / (-objective), dtype=np.float64)
    _q, r, pivots = qr(numeric_matrix, mode="economic", pivoting=True)
    threshold = (max(numeric_matrix.shape) * np.finfo(np.float64).eps
                 * (abs(r[0, 0]) if r.size else 0.0))
    rank = int(np.sum(np.abs(np.diag(r)) > threshold))
    if rank != equation_count:
        raise RuntimeError("Farkas correction system lacks full row rank")
    selected_columns = np.asarray(pivots[:rank], dtype=np.int64)
    _q, _r, row_pivots = qr(
        numeric_matrix[:, selected_columns].T,
        mode="economic", pivoting=True)
    selected_rows = np.asarray(row_pivots[:rank], dtype=np.int64)
    selected = set(int(value) for value in selected_columns)
    values = [_fraction(value) for value in candidate]

    def coefficient(row, column):
        if row < variables:
            return (A[column].get(row, Fraction(0)) if column < len(A)
                    else E[column - len(A)].get(row, Fraction(0)))
        return b[column] if column < len(A) else f[column - len(A)]

    integer_rows, integer_rhs = [], []
    for row in selected_rows:
        target = Fraction(0) if row < variables else Fraction(-1)
        target -= sum(
            (coefficient(int(row), column) * values[column]
             for column in range(column_count) if column not in selected),
            Fraction(0))
        rational = [coefficient(int(row), int(column))
                    for column in selected_columns]
        integers = _rational_row_to_integers([*rational, target])
        integer_rows.append(integers[:-1]); integer_rhs.append(integers[-1])
    solution, evidence = zero._solve_exact_integer_system(
        integer_rows, integer_rhs, timeout, "relu_farkas_repair")
    for column, value in zip(selected_columns, solution):
        values[int(column)] = value
    lambdas, mus = values[:len(A)], values[len(A):]
    checked = _verify_farkas(problem, lambdas, mus)
    return lambdas, mus, {
        "repair_backend": "qr_selected_fraction_free_bareiss",
        "equation_count": equation_count, "column_count": column_count,
        "numerical_rank": rank, **evidence, **checked,
    }


def propose_farkas(problem: ReluCancellationProblem,
                   timeout: float = 300.0) -> tuple[dict | None, dict]:
    """Numerically propose, then directly replay a rational Farkas ray.

    Exact repair is deliberately bounded: large unsupported rays return
    inconclusive rather than invoking a giant rational LP solver.
    """
    started = time.perf_counter()
    # Build the certificate LP numerically, including source/y bounds.
    n, u = problem.n, len(problem.unstable)
    identity = eye(n, format="csr")
    source_A = vstack((identity, -identity), format="csr")
    source_A = hstack((source_A, csr_matrix((2 * n, u))), format="csr")
    y_nonnegative = hstack((csr_matrix((u, n)), -eye(u, format="csr")),
                           format="csr")
    A = vstack((source_A, y_nonnegative, problem.A_triangle), format="csr")
    b = np.concatenate((problem.source["high"], -problem.source["low"],
                        np.zeros(u), problem.b_triangle))
    E = csr_matrix(problem.E)
    stationarity = hstack((A.T, E.T), format="csr")
    objective = np.concatenate((b, problem.f))[None, :]
    result = linprog(
        np.zeros(A.shape[0] + E.shape[0]),
        A_ub=objective, b_ub=np.array([-1.0]),
        A_eq=stationarity, b_eq=np.zeros(A.shape[1]),
        bounds=[(0.0, None)] * A.shape[0] + [(None, None)] * E.shape[0],
        method="highs", options={"presolve": True, "time_limit": timeout})
    diagnostics = {
        "solver_backend": "scipy.optimize.linprog/highs_farkas_system",
        "solver_status": int(result.status), "solver_message": str(result.message),
        "runtime_seconds": time.perf_counter() - started,
        "exact_repair_supported": A.shape[1] <= 512,
    }
    if not result.success or A.shape[1] > 512:
        return None, diagnostics
    numeric_matrix = np.asarray(vstack((stationarity, csr_matrix(objective))).todense())
    try:
        lambdas, mus, checked = _repair_farkas(
            problem, np.asarray(result.x), numeric_matrix)
    except RuntimeError as error:
        diagnostics["exact_replay_error"] = str(error)
        return None, diagnostics
    diagnostics["exact_repair"] = checked
    certificate = {
        "schema": FARKAS_SCHEMA, "problem_sha256": problem.identity_sha256,
        "lambda": [{"numerator": str(x.numerator), "denominator": str(x.denominator)}
                   for x in lambdas],
        "mu": [{"numerator": str(x.numerator), "denominator": str(x.denominator)}
               for x in mus], **checked,
    }
    return certificate, diagnostics


def solve_exact_relu_milp(problem: ReluCancellationProblem,
                          timeout: float) -> tuple[object, dict]:
    started = time.perf_counter()
    n, u = problem.n, len(problem.unstable)
    total = n + u + u
    rows, columns, values, lower, upper = [], [], [], [], []
    for local, neuron in enumerate(problem.unstable):
        hrow = problem.h["generators"][:, neuron]
        nz = np.flatnonzero(hrow)
        # h-y <= 0.
        row = len(lower); rows.extend([row] * len(nz)); columns.extend(nz.tolist())
        values.extend(hrow[nz].tolist()); rows.append(row); columns.append(n + local); values.append(-1.0)
        lower.append(-np.inf); upper.append(-problem.h["center"][neuron])
        # y-u*a <= 0.
        row = len(lower); rows.extend((row, row)); columns.extend((n + local, n + u + local))
        values.extend((1.0, -float(problem.upper_exact[neuron])))
        lower.append(-np.inf); upper.append(0.0)
        # y-h-l*a <= h0-l.
        row = len(lower); rows.extend([row] * len(nz)); columns.extend(nz.tolist())
        values.extend((-hrow[nz]).tolist()); rows.extend((row, row))
        columns.extend((n + local, n + u + local)); values.extend((1.0, -float(problem.lower_exact[neuron])))
        lower.append(-np.inf); upper.append(
            problem.h["center"][neuron] - float(problem.lower_exact[neuron]))
    nonlinear = coo_matrix((values, (rows, columns)), shape=(len(lower), total)).tocsr()
    equality = hstack((csr_matrix(problem.E), csr_matrix((127, u))), format="csr")
    constraints = LinearConstraint(
        vstack((equality, nonlinear), format="csr"),
        np.concatenate((problem.f, np.asarray(lower))),
        np.concatenate((problem.f, np.asarray(upper))))
    lows = np.concatenate((problem.source["low"], np.zeros(u), np.zeros(u)))
    highs = np.concatenate((problem.source["high"],
                            np.array([float(problem.upper_exact[i]) for i in problem.unstable]),
                            np.ones(u)))
    integrality = np.concatenate((np.zeros(n + u), np.ones(u)))
    result = milp(np.zeros(total), integrality=integrality,
                  bounds=Bounds(lows, highs), constraints=constraints,
                  options={"time_limit": timeout, "presolve": True})
    return result, {
        "attempted": True, "solver_backend": "scipy.optimize.milp/highs",
        "solver_status": int(result.status), "solver_message": str(result.message),
        "runtime_seconds": time.perf_counter() - started,
        "activation_pattern_sha256": None,
    }


def _ratios(items) -> list[Fraction]:
    return [Fraction(int(item["numerator"]), int(item["denominator"]))
            for item in items]


def exact_fixed_pattern_witness(problem: ReluCancellationProblem,
                                candidate: np.ndarray, pattern: list[bool],
                                output: Path, timeout: float) -> tuple[dict | None, dict]:
    model = problem.fixed_pattern_model(pattern)
    exact_problem = model.problem()
    certificate, status = zero.construct_exact_zero_certificate(
        exact_problem, candidate, 127, variant_name="exact_relu_fixed_pattern",
        solve_timeout_seconds=timeout)
    if certificate is None:
        return None, status
    values = _ratios(certificate["xi_rationals"])
    signs = problem.verify_pattern(values, pattern)
    replay = model.exact_replay(values)
    witness = {
        "schema": WITNESS_SCHEMA,
        "problem_sha256": problem.identity_sha256,
        "activation_pattern": pattern,
        "activation_pattern_sha256": _sha_json(pattern),
        "xi_rationals": certificate["xi_rationals"],
        "source_ids_sha256": _sha_json(problem.source["ids"]),
        "source_box_check": True, **signs, **replay,
    }
    persisted = _atomic_json(output, witness)
    verify_exact_relu_witness(problem, persisted)
    return persisted, status


def verify_exact_relu_witness(problem: ReluCancellationProblem,
                              witness: dict) -> dict:
    if (witness.get("schema") != WITNESS_SCHEMA
            or witness.get("problem_sha256") != problem.identity_sha256
            or witness.get("source_ids_sha256") != _sha_json(problem.source["ids"])
            or witness.get("activation_pattern_sha256") !=
            _sha_json(witness.get("activation_pattern"))):
        raise RuntimeError("exact ReLU witness identity differs")
    values = _ratios(witness["xi_rationals"])
    if len(values) != problem.n:
        raise RuntimeError("exact ReLU witness source count differs")
    for index, value in enumerate(values):
        if not (_fraction(problem.source["low"][index]) <= value
                <= _fraction(problem.source["high"][index])):
            raise RuntimeError("exact ReLU witness source box differs")
    pattern = list(witness["activation_pattern"])
    problem.verify_pattern(values, pattern)
    model = problem.fixed_pattern_model(pattern)
    return model.exact_replay(values)


def execute(manifest: Path, output: Path, certificate_dir: Path,
            lp_timeout: float, milp_timeout: float,
            exact_timeout: float) -> dict:
    started = time.perf_counter()
    payload, identity, token = frontier._load_capture(manifest)
    weight, bias, parameter_identity = frontier._load_ffn_second_parameters(identity)
    states = payload["states"]
    pre = frontier._decode_state(states["ffn_first_post_reduction"], token)
    residual = frontier._decode_state(
        states["post_attention_ln_post_reduction"], token)
    problem = ReluCancellationProblem(pre, residual, weight, bias)
    hull_result, hull = solve_hull(problem, lp_timeout)
    hull.update({"exact_farkas_attempted": False,
                 "exact_farkas_verified": False,
                 "certificate_path": None, "certificate_sha256": None})
    milp_record = {"attempted": False, "solver_status": None,
                   "runtime_seconds": 0.0, "activation_pattern_sha256": None}
    witness_record = {"attempted": False, "verified": False,
                      "maximum_exact_residual": None,
                      "source_box_check": None, "activation_sign_check": None,
                      "witness_path": None, "witness_sha256": None}
    certificate_dir.mkdir(parents=True, exist_ok=True)
    if not hull_result.success:
        hull["exact_farkas_attempted"] = True
        certificate, diagnostics = propose_farkas(problem, lp_timeout)
        hull["farkas_search"] = diagnostics
        if certificate is None:
            final_status = FINAL_INCONCLUSIVE
        else:
            path = certificate_dir / "relu_hull_farkas_certificate.json"
            persisted = _atomic_json(path, certificate)
            lambdas, mus = _ratios(persisted["lambda"]), _ratios(persisted["mu"])
            _verify_farkas(problem, lambdas, mus)
            hull.update({"exact_farkas_verified": True,
                         "certificate_path": str(path),
                         "certificate_sha256": cluster_common.sha256(path)})
            final_status = FINAL_HULL_EXCLUDED
    else:
        result, milp_record = solve_exact_relu_milp(problem, milp_timeout)
        if result.success:
            pattern = [False] * problem.m
            for index in problem.active: pattern[index] = True
            for local, index in enumerate(problem.unstable):
                pattern[index] = bool(round(result.x[problem.n + len(problem.unstable) + local]))
            milp_record["activation_pattern_sha256"] = _sha_json(pattern)
            witness_record["attempted"] = True
            path = certificate_dir / "exact_relu_cancellation_witness.json"
            try:
                witness, status = exact_fixed_pattern_witness(
                    problem, np.asarray(result.x[:problem.n]), pattern,
                    path, exact_timeout)
            except RuntimeError as error:
                witness, status = None, {"reason": str(error)}
            witness_record["exact_reconstruction"] = status
            if witness is None:
                final_status = FINAL_INCONCLUSIVE
            else:
                witness_record.update({
                    "verified": True, "maximum_exact_residual": "0",
                    "source_box_check": True, "activation_sign_check": True,
                    "witness_path": str(path),
                    "witness_sha256": cluster_common.sha256(path),
                })
                final_status = FINAL_EXACT_FEASIBLE
        elif int(result.status) == 2:
            final_status = FINAL_MILP_UNCERTIFIED
        else:
            final_status = FINAL_INCONCLUSIVE
    interpretations = {
        FINAL_HULL_EXCLUDED: "RELU_RELAXATION_CAUSALLY_RESPONSIBLE",
        FINAL_EXACT_FEASIBLE: "RELU_RELAXATION_EXONERATED_SEND_SEARCH_UPSTREAM",
        FINAL_MILP_UNCERTIFIED: "RELU_CAUSAL_ORACLE_NEEDS_BAB_PROOF",
        FINAL_INCONCLUSIVE: "RELU_CAUSAL_BOUNDARY_UNRESOLVED",
    }
    report = {
        "schema": SCHEMA, "property_id": identity["property_id"],
        "tested_radius": payload["identity"]["tested_radius"],
        "tested_radius_hex": payload["identity"]["tested_radius_hex"],
        "manifest_authentication": identity,
        "parameter_authentication": parameter_identity,
        "problem_sha256": problem.identity_sha256,
        "analysis_token_index": token,
        "source_variable_count": problem.n,
        "relu_counts": {"stable_active": len(problem.active),
                        "stable_inactive": len(problem.inactive),
                        "unstable": len(problem.unstable),
                        "exact_bound_runtime_seconds": problem.bound_seconds},
        "hull_lp": hull, "exact_relu_milp": milp_record,
        "exact_relu_witness": witness_record,
        "final_status": final_status,
        "causal_interpretation": interpretations[final_status],
        "runtime_seconds": time.perf_counter() - started,
        "scientific_queries": 0, "bound_calls": 0, "gpu_jobs": 0,
    }
    return _atomic_json(output, report)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--certificate-dir", required=True, type=Path)
    parser.add_argument("--lp-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--milp-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--exact-solve-timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()
    report = execute(args.manifest.resolve(), args.output.resolve(),
                     args.certificate_dir.resolve(),
                     args.lp_timeout_seconds,
                     args.milp_timeout_seconds,
                     args.exact_solve_timeout_seconds)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
