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
                 weight: np.ndarray, bias: np.ndarray, label="block2_relu",
                 exact_preactivation=None, additional_equalities=None):
        h, r, source = frontier._align_sources(preactivation, residual)
        self.h, self.r, self.source = h, r, source
        self.weight = np.asarray(weight, dtype=np.float64)
        self.bias = np.asarray(bias, dtype=np.float64)
        self.label = label
        self.exact_preactivation = exact_preactivation
        self.additional_equalities = additional_equalities
        if (h["center"].ndim != 1 or r["center"].shape != (128,)
                or self.weight.shape != (128, h["center"].size)
                or self.bias.shape != (128,)):
            raise RuntimeError("authenticated ReLU cancellation dimensions differ")
        self.n = len(source["ids"])
        self.m = h["center"].size
        started = time.perf_counter()
        self.lower_exact, self.upper_exact = self._exact_bounds()
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
        self._append_additional_equalities()
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
            "exact_preactivation_identity": (
                None if exact_preactivation is None
                else exact_preactivation["identity_sha256"]),
            "additional_equalities_identity": (
                None if additional_equalities is None
                else additional_equalities["identity_sha256"]),
        })

    def _append_additional_equalities(self):
        if self.additional_equalities is None:
            self.additional_equality_count = 0
            return
        extra = self.additional_equalities
        matrix = np.asarray(extra["numeric_A"], dtype=np.float64)
        constant = np.asarray(extra["numeric_b"], dtype=np.float64)
        if (matrix.ndim != 2 or matrix.shape[1] != self.n
                or constant.shape != (matrix.shape[0],)
                or not np.isfinite(matrix).all()
                or not np.isfinite(constant).all()):
            raise RuntimeError("additional exact equality dimensions differ")
        padded = np.pad(matrix, ((0, 0), (0, len(self.unstable))))
        self.E = np.concatenate((self.E, padded), axis=0)
        self.f = np.concatenate((self.f, -constant), axis=0)
        self.additional_equality_count = int(matrix.shape[0])

    def additional_exact_residual(self, values: list[Fraction]):
        if self.additional_equalities is None:
            return []
        extra = self.additional_equalities
        return [
            extra["exact_constant"](row) + sum(
                (extra["exact_coefficient"](row, source) * xi
                 for source, xi in enumerate(values)), Fraction(0))
            for row in range(self.additional_equality_count)]

    def h_center_exact(self, coordinate: int) -> Fraction:
        if self.exact_preactivation is None:
            return _fraction(self.h["center"][coordinate])
        return self.exact_preactivation["center"](coordinate)

    def h_generator_exact(self, source: int, coordinate: int) -> Fraction:
        if self.exact_preactivation is None:
            return _fraction(self.h["generators"][source, coordinate])
        return self.exact_preactivation["generator"](source, coordinate)

    def _exact_bounds(self):
        if (self.exact_preactivation is not None
                and self.exact_preactivation.get("bounds") is not None):
            lower, upper = self.exact_preactivation["bounds"]
            if len(lower) != self.h["center"].size or len(upper) != len(lower):
                raise RuntimeError("exact preactivation bound dimensions differ")
            return list(lower), list(upper)
        lowers, uppers = [], []
        for coordinate in range(self.h["center"].size):
            lo = hi = self.h_center_exact(coordinate)
            for source, (lower, upper) in enumerate(zip(
                    self.source["low"], self.source["high"])):
                coefficient = self.h_generator_exact(source, coordinate)
                first = coefficient * _fraction(lower)
                second = coefficient * _fraction(upper)
                lo += min(first, second)
                hi += max(first, second)
            lowers.append(lo); uppers.append(hi)
        return lowers, uppers

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
            self.h_center_exact(coordinate) + sum(
                (self.h_generator_exact(row, coordinate) * xi
                 for row, xi in enumerate(values)), Fraction(0))
            for coordinate in range(self.h["center"].size)]

    def fixed_pattern_model(self, active_pattern: list[bool]):
        if len(active_pattern) != self.m:
            raise RuntimeError("activation pattern length differs")
        return ExactFixedReluModel(self, active_pattern)

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
        # Keep exact inequality order identical to propose_farkas(): all
        # source bounds, all y>=0 rows, then the two triangle rows per neuron.
        for local, _neuron in enumerate(self.unstable):
            y = self.n + local
            A.append({y: Fraction(-1)}); b.append(Fraction(0))
        for local, neuron in enumerate(self.unstable):
            y = self.n + local
            row = {y: Fraction(-1)}
            for source in range(self.n):
                value = self.h_generator_exact(source, neuron)
                if value: row[source] = value
            A.append(row); b.append(-self.h_center_exact(neuron))
            lower, upper = self.lower_exact[neuron], self.upper_exact[neuron]
            alpha = upper / (upper - lower)
            row = {y: Fraction(1)}
            for source in range(self.n):
                value = -alpha * self.h_generator_exact(source, neuron)
                if value: row[source] = value
            A.append(row)
            b.append(alpha * (self.h_center_exact(neuron) - lower))
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
                              * self.h_generator_exact(source, neuron))
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
                             * self.h_center_exact(neuron))
            E.append(row); f.append(-constant)
        if self.additional_equalities is not None:
            extra = self.additional_equalities
            for equality in range(self.additional_equality_count):
                row = {}
                for source in range(self.n):
                    value = extra["exact_coefficient"](equality, source)
                    if value:
                        row[source] = value
                E.append(row)
                f.append(-extra["exact_constant"](equality))
        return A, b, E, f, variable_count


class ExactFixedReluModel:
    """Exact shared-source r + W2*ReLU(h)+b2 for one fixed sign pattern."""

    def __init__(self, parent: ReluCancellationProblem, pattern: list[bool]):
        self.parent = parent
        self.pattern = list(pattern)
        masked = parent.weight.copy()
        masked[:, np.logical_not(pattern)] = 0.0
        self.masked = masked
        self.numeric_center = (parent.r["center"] + parent.bias
                               + masked @ parent.h["center"])
        self.numeric_generators = (parent.r["generators"]
                                   + parent.h["generators"] @ masked.T)
        self.source = parent.source

    def _coordinate(self, output: int, source: int | None) -> Fraction:
        if source is None:
            value = (_fraction(self.parent.r["center"][output])
                     + _fraction(self.parent.bias[output]))
            h_value = self.parent.h_center_exact
        else:
            value = _fraction(self.parent.r["generators"][source, output])
            h_value = lambda coordinate: self.parent.h_generator_exact(
                source, coordinate)
        for coordinate, active in enumerate(self.pattern):
            if active:
                value += (_fraction(self.parent.weight[output, coordinate])
                          * h_value(coordinate))
        return value

    def exact_center_difference(self, row: int) -> Fraction:
        if row >= 127:
            return self.parent.additional_equalities["exact_constant"](
                row - 127)
        return self._coordinate(row, None) - self._coordinate(-1, None)

    def exact_coefficient(self, row: int, column: int) -> Fraction:
        if row >= 127:
            return self.parent.additional_equalities["exact_coefficient"](
                row - 127, column)
        return self._coordinate(row, column) - self._coordinate(-1, column)

    def exact_residual(self, values: list[Fraction]) -> list[Fraction]:
        h = self.parent.exact_h(values)
        residual = [
            _fraction(center) + sum(
                (_fraction(self.parent.r["generators"][source, coordinate])
                 * xi for source, xi in enumerate(values)), Fraction(0))
            for coordinate, center in enumerate(self.parent.r["center"])]
        output = []
        for row in range(128):
            value = residual[row] + _fraction(self.parent.bias[row])
            value += sum(
                (_fraction(self.parent.weight[row, coordinate]) * h[coordinate]
                 for coordinate, active in enumerate(self.pattern) if active),
                Fraction(0))
            output.append(value)
        residuals = [value - output[-1] for value in output[:-1]]
        return residuals + self.parent.additional_exact_residual(values)

    def exact_replay(self, values: list[Fraction]) -> dict:
        residuals = self.exact_residual(values)
        cancellation = residuals[:127]
        additional = residuals[127:]
        if any(cancellation):
            raise RuntimeError("exact joint cancellation replay is nonzero")
        if any(additional):
            raise RuntimeError("exact additional equality replay is nonzero")
        return {
            "exact_equalities": len(residuals),
            "cancellation_equalities": 127,
            "additional_equalities": self.parent.additional_equality_count,
            "invariant_check": (
                self.parent.additional_equality_count > 0
                and not any(additional)),
            "exact_box_constraints": len(values),
            "maximum_exact_residual": "0", "exact_variance": "0",
            "joint_residual_ffn_correlation_preserved": True,
            "exact_preactivation_composition_replayed": True,
        }

    def problem(self) -> dict:
        problem = zero.centered_problem(
            self.numeric_center, self.numeric_generators,
            self.source["low"], self.source["high"], self.source["ids"])
        if self.parent.additional_equalities is not None:
            extra = self.parent.additional_equalities
            problem["A"] = np.concatenate(
                (problem["A"], np.asarray(extra["numeric_A"])), axis=0)
            problem["b"] = np.concatenate(
                (problem["b"], np.asarray(extra["numeric_b"])), axis=0)
        problem.update({
            "exact_center_difference": self.exact_center_difference,
            "exact_coefficient": self.exact_coefficient,
            "exact_residual": self.exact_residual,
            "exact_replay": self.exact_replay,
        })
        return problem


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


def _repair_large_farkas(problem: ReluCancellationProblem,
                         candidate: np.ndarray,
                         timeout: float = 180.0):
    """Repair a large-source Farkas ray through bound-multiplier elimination.

    Source-box multipliers are eliminated analytically after fixing their
    active sign.  Exact solving therefore has only ``unstable+1`` equations,
    independent of the often 14k+ source count.
    """
    started = time.perf_counter()
    A, b, E, f, variables = problem.exact_lp()
    n, u = problem.n, len(problem.unstable)
    source_rows = 2 * n
    if len(candidate) != len(A) + len(E):
        raise RuntimeError("large Farkas candidate dimensions differ")
    columns = [("lambda", index) for index in range(source_rows, len(A))]
    columns.extend(("mu", index) for index in range(len(E)))
    values = [_fraction(candidate[index])
              for index in range(source_rows, len(A))]
    values.extend(_fraction(candidate[len(A) + index])
                  for index in range(len(E)))

    def coefficient(column, variable):
        kind, index = columns[column]
        return (A[index].get(variable, Fraction(0)) if kind == "lambda"
                else E[index].get(variable, Fraction(0)))

    # Freeze only the sign of the analytically reconstructed source-bound
    # multiplier.  Exact replay below remains authoritative.
    source_signs = []
    for variable in range(n):
        high_multiplier = float(candidate[2 * variable])
        low_multiplier = float(candidate[2 * variable + 1])
        source_signs.append(1 if low_multiplier >= high_multiplier else -1)

    def contradiction_coefficient(column):
        kind, index = columns[column]
        value = b[index] if kind == "lambda" else f[index]
        for variable, sign in enumerate(source_signs):
            stationarity = coefficient(column, variable)
            bound = (_fraction(problem.source["low"][variable])
                     if sign > 0 else
                     _fraction(problem.source["high"][variable]))
            value -= bound * stationarity
        return value

    exact_rows = []
    for local in range(u):
        exact_rows.append([
            coefficient(column, n + local)
            for column in range(len(columns))])
    exact_rows.append([
        contradiction_coefficient(column)
        for column in range(len(columns))])
    numeric = np.array([[float(value) for value in row]
                        for row in exact_rows], dtype=np.float64)
    equation_count = u + 1
    _q, r, pivots = qr(numeric, mode="economic", pivoting=True)
    threshold = (max(numeric.shape) * np.finfo(np.float64).eps
                 * (abs(r[0, 0]) if r.size else 0.0))
    rank = int(np.sum(np.abs(np.diag(r)) > threshold))
    if rank != equation_count:
        raise RuntimeError("large Farkas reduced correction lacks full row rank")
    selected = np.asarray(pivots[:equation_count], dtype=np.int64)
    chosen = set(int(value) for value in selected)
    targets = [Fraction(0)] * u + [Fraction(-1)]
    integer_rows, integer_rhs = [], []
    for row, target in zip(exact_rows, targets):
        rhs = target - sum(
            (row[column] * values[column]
             for column in range(len(columns)) if column not in chosen),
            Fraction(0))
        integers = _rational_row_to_integers(
            [*[row[int(column)] for column in selected], rhs])
        integer_rows.append(integers[:-1]); integer_rhs.append(integers[-1])
    solution, evidence = zero._solve_exact_integer_system(
        integer_rows, integer_rhs, timeout, "relu_large_farkas_repair")
    for column, value in zip(selected, solution):
        values[int(column)] = value

    core_lambda_count = len(A) - source_rows
    core_lambdas = values[:core_lambda_count]
    mus = values[core_lambda_count:]
    if any(value < 0 for value in core_lambdas):
        raise RuntimeError("large Farkas repaired core lambda is negative")
    source_lambdas = [Fraction(0)] * source_rows
    for variable, sign in enumerate(source_signs):
        contribution = sum(
            (coefficient(column, variable) * values[column]
             for column in range(len(columns))), Fraction(0))
        if sign > 0:
            if contribution < 0:
                raise RuntimeError("large Farkas source sign changed")
            source_lambdas[2 * variable + 1] = contribution
        else:
            if contribution > 0:
                raise RuntimeError("large Farkas source sign changed")
            source_lambdas[2 * variable] = -contribution
    lambdas = source_lambdas + core_lambdas
    checked = _verify_farkas(problem, lambdas, mus)
    return lambdas, mus, {
        "repair_backend": "source_bound_elimination_plus_bareiss",
        "original_variable_count": variables,
        "reduced_equation_count": equation_count,
        "reduced_column_count": len(columns),
        "runtime_seconds": time.perf_counter() - started,
        **evidence, **checked,
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
        "exact_repair_supported": True,
        "exact_repair_mode": ("bounded_full_stationarity"
                              if A.shape[1] <= 512 else
                              "source_bound_elimination"),
    }
    if not result.success:
        return None, diagnostics
    try:
        if A.shape[1] <= 512:
            numeric_matrix = np.asarray(vstack(
                (stationarity, csr_matrix(objective))).todense())
            lambdas, mus, checked = _repair_farkas(
                problem, np.asarray(result.x), numeric_matrix)
        else:
            lambdas, mus, checked = _repair_large_farkas(
                problem, np.asarray(result.x))
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
                          timeout: float,
                          excluded_patterns: list[list[bool]] | None = None
                          ) -> tuple[object, dict]:
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
    for pattern in excluded_patterns or []:
        if len(pattern) != problem.m:
            raise RuntimeError("excluded activation pattern length differs")
        # sum(active a)-sum(inactive a) <= number_active-1 excludes exactly
        # this binary pattern while retaining every other pattern.
        row = len(lower)
        active_count = 0
        for local, neuron in enumerate(problem.unstable):
            active = bool(pattern[neuron])
            rows.append(row); columns.append(n + u + local)
            values.append(1.0 if active else -1.0)
            active_count += int(active)
        lower.append(-np.inf); upper.append(float(active_count - 1))
    nonlinear = coo_matrix((values, (rows, columns)), shape=(len(lower), total)).tocsr()
    equality = hstack((csr_matrix(problem.E),
                       csr_matrix((problem.E.shape[0], u))), format="csr")
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
        "excluded_pattern_count": len(excluded_patterns or []),
    }


def _ratios(items) -> list[Fraction]:
    return [Fraction(int(item["numerator"]), int(item["denominator"]))
            for item in items]


def fixed_pattern_interior_candidate(
        problem: ReluCancellationProblem, pattern: list[bool]) -> tuple[np.ndarray | None, dict]:
    """Maximize deterministic normalized box/sign slack for one pattern."""
    model = problem.fixed_pattern_model(pattern)
    exact_problem = model.problem()
    Aeq, beq = exact_problem["A"], -exact_problem["b"]
    n = problem.n
    rows, columns, values, rhs = [], [], [], []
    widths = problem.source["high"] - problem.source["low"]
    source_scales = np.where(widths > 0.0, widths, 0.0)
    for index, (low, high, scale) in enumerate(zip(
            problem.source["low"], problem.source["high"], source_scales)):
        row = len(rhs); rows.extend((row, row)); columns.extend((index, n))
        values.extend((-1.0, float(scale))); rhs.append(-float(low))
        row = len(rhs); rows.extend((row, row)); columns.extend((index, n))
        values.extend((1.0, float(scale))); rhs.append(float(high))
    bound_widths = np.array([
        float(upper - lower) for lower, upper in
        zip(problem.lower_exact, problem.upper_exact)])
    for neuron, active in enumerate(pattern):
        scale = max(bound_widths[neuron], np.finfo(np.float64).tiny)
        nz = np.flatnonzero(problem.h["generators"][:, neuron])
        row = len(rhs); rows.extend([row] * len(nz)); columns.extend(nz.tolist())
        if active:
            values.extend((-problem.h["generators"][nz, neuron]).tolist())
            rows.append(row); columns.append(n); values.append(scale)
            rhs.append(float(problem.h["center"][neuron]))
        else:
            values.extend(problem.h["generators"][nz, neuron].tolist())
            rows.append(row); columns.append(n); values.append(scale)
            rhs.append(-float(problem.h["center"][neuron]))
    Aub = coo_matrix((values, (rows, columns)),
                     shape=(len(rhs), n + 1)).tocsr()
    E = hstack((csr_matrix(Aeq), csr_matrix((Aeq.shape[0], 1))), format="csr")
    objective = np.zeros(n + 1); objective[-1] = -1.0
    started = time.perf_counter()
    result = linprog(
        objective, A_ub=Aub, b_ub=np.asarray(rhs), A_eq=E, b_eq=beq,
        bounds=list(zip(problem.source["low"], problem.source["high"]))
        + [(None, 1.0)], method="highs", options={"presolve": True})
    candidate = (np.asarray(result.x[:n]) if result.success else None)
    if result.success:
        source_lower_slack = candidate - problem.source["low"]
        source_upper_slack = problem.source["high"] - candidate
        h_value = problem.h["center"] + candidate @ problem.h["generators"]
        sign_slack = np.where(np.asarray(pattern), h_value, -h_value)
        source_violation = max(
            0.0,
            -float(source_lower_slack.min()) if source_lower_slack.size else 0.0,
            -float(source_upper_slack.min()) if source_upper_slack.size else 0.0)
        sign_violation = max(0.0, -float(sign_slack.min()))
    else:
        source_lower_slack = source_upper_slack = sign_slack = np.array([])
        source_violation = sign_violation = None
    metrics = {
        "solver_backend": "scipy.optimize.linprog/highs_max_common_slack",
        "feasible": bool(result.success), "solver_status": int(result.status),
        "solver_message": str(result.message),
        "runtime_seconds": time.perf_counter() - started,
        "best_common_slack_t": float(result.x[-1]) if result.success else None,
        "maximum_numerical_equality_residual": (
            float(np.abs(Aeq @ candidate - beq).max(initial=0.0))
            if result.success else None),
        "minimum_source_box_slack": (float(np.minimum(
            source_lower_slack, source_upper_slack).min())
            if result.success and source_lower_slack.size else None),
        "maximum_source_box_violation": source_violation,
        "minimum_activation_sign_slack": (
            float(sign_slack.min()) if result.success else None),
        "maximum_activation_sign_violation": sign_violation,
    }
    return candidate, metrics


def _basis_candidates(problem: ReluCancellationProblem,
                      model_problem: dict, candidate: np.ndarray,
                      pattern: list[bool], maximum: int = 32) -> list[np.ndarray]:
    A = model_problem["A"]
    widths = problem.source["high"] - problem.source["low"]
    box_slack = np.where(
        widths > 0.0,
        np.minimum(candidate - problem.source["low"],
                   problem.source["high"] - candidate) / widths,
        0.0)
    h = (problem.h["center"]
         + candidate @ problem.h["generators"])
    sign_margin = np.maximum(
        np.where(np.asarray(pattern), h, -h), 0.0)
    scale = np.maximum(
        np.array([float(upper - lower) for lower, upper in
                  zip(problem.lower_exact, problem.upper_exact)]),
        np.finfo(np.float64).tiny)
    normalized_margin = sign_margin / scale
    sensitivity = np.abs(problem.h["generators"]) @ (
        1.0 / np.maximum(normalized_margin, 2.0 ** -40))
    sign_quality = 1.0 / (1.0 + sensitivity * np.maximum(widths, 0.0))
    quality = np.maximum(box_slack, 0.0) * sign_quality
    free = np.flatnonzero(widths > 0.0)
    if not len(free):
        return [np.empty(0, dtype=np.int64)]
    bases, seen = [], set()
    exponents = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    for attempt in range(maximum * 2):
        exponent = exponents[attempt % len(exponents)]
        jitter = 1.0 + 1e-6 * np.sin(
            (free.astype(np.float64) + 1.0) * (attempt + 1.0))
        weights = np.maximum(quality[free], 2.0 ** -40) ** exponent * jitter
        _q, r, pivots = qr(
            A[:, free] * weights[None, :], mode="economic", pivoting=True)
        threshold = (max(A.shape) * np.finfo(np.float64).eps
                     * (abs(r[0, 0]) if r.size else 0.0))
        rank = int(np.sum(np.abs(np.diag(r)) > threshold))
        columns = free[np.asarray(pivots[:rank], dtype=np.int64)]
        key = tuple(sorted(int(value) for value in columns))
        if key not in seen:
            seen.add(key); bases.append(columns)
        if len(bases) >= maximum:
            break
    return bases


def _attempt_exact_basis(problem: ReluCancellationProblem, model,
                         candidate: np.ndarray, pattern: list[bool],
                         columns: np.ndarray, timeout: float) -> tuple[list[Fraction] | None, dict]:
    exact_problem = model.problem()
    A = exact_problem["A"]
    rank = len(columns)
    if rank == 0:
        selected_rows = np.empty(0, dtype=np.int64)
    else:
        _q, _r, pivots = qr(A[:, columns].T, mode="economic", pivoting=True)
        selected_rows = np.asarray(pivots[:rank], dtype=np.int64)
    values = [_fraction(value) for value in candidate]
    residual = exact_problem["exact_residual"](values)
    integer_rows, integer_rhs = [], []
    try:
        for row in selected_rows:
            rhs = -residual[int(row)] + sum(
                (exact_problem["exact_coefficient"](int(row), int(column))
                 * values[int(column)] for column in columns), Fraction(0))
            coefficients = [exact_problem["exact_coefficient"](
                int(row), int(column)) for column in columns]
            integers, _metadata = zero._dyadic_row_to_integers(
                [*coefficients, rhs])
            integer_rows.append(integers[:-1]); integer_rhs.append(integers[-1])
        if rank:
            solution, solver = zero._solve_exact_integer_system(
                integer_rows, integer_rhs, timeout, "relu_fixed_pattern_basis")
            for column, value in zip(columns, solution):
                values[int(column)] = value
        else:
            solver = {"solver_api": "empty_exact_correction_system",
                      "selected_system_shape": [0, 0]}
        outside = [index for index, value in enumerate(values)
                   if not (_fraction(problem.source["low"][index]) <= value
                           <= _fraction(problem.source["high"][index]))]
        if outside:
            return None, {"failure_reason": "SOURCE_BOX_VIOLATION",
                          "outside_count": len(outside),
                          "first_outside": outside[0], **solver}
        problem.verify_pattern(values, pattern)
        replay = model.exact_replay(values)
        return values, {"verified": True, **solver, **replay}
    except Exception as error:
        return None, {"failure_reason": f"{type(error).__name__}: {error}"}


def exact_fixed_pattern_witness(problem: ReluCancellationProblem,
                                candidate: np.ndarray, pattern: list[bool],
                                output: Path, timeout: float,
                                maximum_bases: int = 32,
                                basis_override=None) -> tuple[dict | None, dict]:
    model = problem.fixed_pattern_model(pattern)
    model_problem = model.problem()
    bases = (list(basis_override) if basis_override is not None else
             _basis_candidates(problem, model_problem, candidate, pattern,
                               maximum_bases))
    attempts = []
    values = None
    deadline = time.perf_counter() + timeout
    for ordinal, columns in enumerate(bases[:maximum_bases]):
        remaining = deadline - time.perf_counter()
        if remaining <= 0.0:
            attempts.append({"ordinal": ordinal,
                             "failure_reason": "EXACT_RECOVERY_TIMEOUT"})
            break
        recovered, evidence = _attempt_exact_basis(
            problem, model, candidate, pattern,
            np.asarray(columns, dtype=np.int64), remaining)
        attempts.append({
            "ordinal": ordinal,
            "selected_column_count": len(columns),
            "selected_columns_sha256": _sha_json(
                [int(value) for value in columns]),
            **evidence,
        })
        if recovered is not None:
            values = recovered
            break
    status = {
        "attempted_basis_count": len(attempts), "basis_attempts": attempts,
        "verified": values is not None,
        "failure_reasons": [row.get("failure_reason") for row in attempts
                            if row.get("failure_reason")],
        "best_failed_basis": (attempts[-1] if attempts and values is None
                              else None),
        "terminal_recovery_status": (
            "EXACT_AUTHENTICATED_RELU_WITNESS_VERIFIED" if values is not None
            else "FIXED_PATTERN_EXACT_WITNESS_NOT_FOUND"),
    }
    if values is None:
        return None, status
    signs = problem.verify_pattern(values, pattern)
    replay = model.exact_replay(values)
    witness = {
        "schema": WITNESS_SCHEMA,
        "problem_sha256": problem.identity_sha256,
        "activation_pattern": pattern,
        "activation_pattern_sha256": _sha_json(pattern),
        "xi_rationals": [
            {"numerator": str(value.numerator),
             "denominator": str(value.denominator)} for value in values],
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
            exact_timeout: float, maximum_patterns: int = 8,
            maximum_bases: int = 32) -> dict:
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
                   "runtime_seconds": 0.0, "activation_pattern_sha256": None,
                   "patterns": [], "alternative_pattern_fallback": True}
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
        excluded_patterns = []
        first_milp_infeasible = False
        final_status = FINAL_INCONCLUSIVE
        total_bases = 0
        failure_reasons = []
        best_common_slack = None
        best_failed_basis = None
        for pattern_ordinal in range(maximum_patterns):
            result, current_milp = solve_exact_relu_milp(
                problem, milp_timeout, excluded_patterns)
            milp_record["attempted"] = True
            milp_record["solver_status"] = current_milp["solver_status"]
            milp_record["solver_message"] = current_milp["solver_message"]
            milp_record["runtime_seconds"] += current_milp["runtime_seconds"]
            if not result.success:
                first_milp_infeasible = (pattern_ordinal == 0
                                         and int(result.status) == 2)
                break
            pattern = [False] * problem.m
            for index in problem.active: pattern[index] = True
            for local, index in enumerate(problem.unstable):
                pattern[index] = bool(round(result.x[problem.n + len(problem.unstable) + local]))
            pattern_sha = _sha_json(pattern)
            milp_record["activation_pattern_sha256"] = pattern_sha
            candidate, fixed_lp = fixed_pattern_interior_candidate(problem, pattern)
            pattern_row = {
                "ordinal": pattern_ordinal,
                "activation_pattern_sha256": pattern_sha,
                "fixed_pattern_lp": fixed_lp,
            }
            if fixed_lp["best_common_slack_t"] is not None:
                best_common_slack = max(
                    best_common_slack if best_common_slack is not None else -math.inf,
                    fixed_lp["best_common_slack_t"])
            witness_record["attempted"] = True
            witness = None
            status = {"verified": False, "failure_reasons": [
                "FIXED_PATTERN_NUMERICAL_LP_INFEASIBLE"]}
            if candidate is not None:
                path = certificate_dir / "exact_relu_cancellation_witness.json"
                try:
                    witness, status = exact_fixed_pattern_witness(
                        problem, candidate, pattern, path, exact_timeout,
                        maximum_bases=maximum_bases)
                except RuntimeError as error:
                    witness, status = None, {"verified": False,
                                             "failure_reasons": [str(error)]}
            pattern_row["exact_reconstruction"] = status
            milp_record["patterns"].append(pattern_row)
            total_bases += int(status.get("attempted_basis_count", 0))
            failure_reasons.extend(status.get("failure_reasons", []))
            if status.get("best_failed_basis") is not None:
                best_failed_basis = status["best_failed_basis"]
            if witness is not None:
                witness_record.update({
                    "verified": True, "maximum_exact_residual": "0",
                    "source_box_check": True, "activation_sign_check": True,
                    "witness_path": str(path),
                    "witness_sha256": cluster_common.sha256(path),
                })
                final_status = FINAL_EXACT_FEASIBLE
                break
            excluded_patterns.append(pattern)
        if first_milp_infeasible:
            final_status = FINAL_MILP_UNCERTIFIED
        witness_record.update({
            "number_of_patterns_attempted": len(milp_record["patterns"]),
            "number_of_bases_attempted": total_bases,
            "numerical_fixed_pattern_lp_feasible": any(
                row["fixed_pattern_lp"]["feasible"]
                for row in milp_record["patterns"]),
            "best_common_slack_t": best_common_slack,
            "best_failed_basis": best_failed_basis,
            "failure_reasons": failure_reasons,
            "maximum_numerical_equality_residual": min(
                (row["fixed_pattern_lp"]["maximum_numerical_equality_residual"]
                 for row in milp_record["patterns"]
                 if row["fixed_pattern_lp"]["feasible"]), default=None),
            "minimum_source_box_violation": min(
                (row["fixed_pattern_lp"]["maximum_source_box_violation"]
                 for row in milp_record["patterns"]
                 if row["fixed_pattern_lp"]["feasible"]), default=None),
            "minimum_activation_sign_violation": min(
                (row["fixed_pattern_lp"]["maximum_activation_sign_violation"]
                 for row in milp_record["patterns"]
                 if row["fixed_pattern_lp"]["feasible"]), default=None),
            "terminal_recovery_status": (
                "EXACT_AUTHENTICATED_RELU_WITNESS_VERIFIED"
                if witness_record["verified"] else
                "FIXED_PATTERN_EXACT_WITNESS_NOT_FOUND"),
        })
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
    parser.add_argument("--maximum-patterns", type=int, default=8)
    parser.add_argument("--maximum-bases", type=int, default=32)
    args = parser.parse_args()
    report = execute(args.manifest.resolve(), args.output.resolve(),
                     args.certificate_dir.resolve(),
                     args.lp_timeout_seconds,
                     args.milp_timeout_seconds,
                     args.exact_solve_timeout_seconds,
                     args.maximum_patterns, args.maximum_bases)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
