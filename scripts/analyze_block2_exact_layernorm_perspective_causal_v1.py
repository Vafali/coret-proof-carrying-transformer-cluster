#!/usr/bin/env python3
"""Proof-carrying exact-LayerNorm perspective causal oracle.

The numerical conic solver (when one is available) is only a proposal engine.
The only decisive outcomes are an exactly replayed shared-source witness or a
complete branch tree whose conic infeasibility rays all pass the rational
checker in this file.  In particular, solver status is never a proof.

This program is CPU-only and consumes existing authenticated captures.  It
does not call a verifier/bound entry point or alter producer artifacts.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
import gzip
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from typing import Iterable, Sequence

import numpy as np
from scipy.linalg import qr
from scipy.optimize import linprog
import torch


REPO = Path(__file__).resolve().parents[1]


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


runner = image = prefrontier = frontier = None


class _LocalIntegrity:
    """Tiny dependency-free copy of the canonical JSON integrity contract."""

    @staticmethod
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def canonical(value: dict) -> str:
        payload = {key: item for key, item in value.items()
                   if key != "record_sha256"}
        return hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @classmethod
    def verified_json(cls, path: Path) -> dict:
        value = json.loads(path.read_text())
        if value.get("record_sha256") != cls.canonical(value):
            raise RuntimeError(f"JSON record SHA differs: {path}")
        return value


cluster_common = _LocalIntegrity


def _ensure_production_modules():
    """Import the producer authentication stack only for real artifact use.

    Exact core tests intentionally do not import Torch/MPFR producer modules;
    this keeps the checker primitives independently testable in a CPU-only
    minimal environment.
    """
    global runner, image, prefrontier, frontier
    if runner is not None:
        return
    runner = _module(
        "perspective_capture_runner",
        REPO / "scripts/run_sound_fp64_3l_psd_layernorm_experiment_v1.py")
    image = _module(
        "perspective_layernorm_parameters",
        REPO / "scripts/analyze_block2_layernorm_image_invariant_causal_v1.py")
    prefrontier = image.prefrontier
    frontier = image.frontier

SCHEMA = "CORET_BLOCK2_EXACT_LAYERNORM_PERSPECTIVE_CAUSAL_V1"
CERTIFICATE_SCHEMA = "CORET_RATIONAL_CONIC_INFEASIBILITY_CERTIFICATE_V1"
WITNESS_SCHEMA = "CORET_EXACT_LAYERNORM_CANCELLATION_WITNESS_V1"
TREE_SCHEMA = "CORET_RELU_PHASE_CONIC_PROOF_TREE_V1"

FEASIBLE = "EXACT_LAYERNORM_CANCELLATION_FEASIBLE"
EXCLUDED = "EXACT_LAYERNORM_CANCELLATION_EXCLUDED"
INCONCLUSIVE = "EXACT_LAYERNORM_CAUSAL_ORACLE_INCONCLUSIVE"

PROPERTY_ID = "deept_table7_stdln3_s001_line1794_tok11"
RADIUS = 0.00060791015625
RADIUS_HEX = RADIUS.hex()
EXPECTED_CANONICAL_IDENTITY = (
    "c2fbf1d175157dfecca9c3da95573b593921a4dc05ca1b6a3e7eacaf730ea507")
EXPECTED_SOURCES = 14_000
EXPECTED_TOKEN = 0
DIMENSION = 128


def _fr(value) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, str):
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            return Fraction(int(numerator), int(denominator))
        return Fraction(value)
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    return Fraction.from_float(float(value))


def _fs(value: Fraction) -> str:
    value = _fr(value)
    return f"{value.numerator}/{value.denominator}"


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


def _dot(left: Sequence[Fraction], right: Sequence[Fraction]) -> Fraction:
    if len(left) != len(right):
        raise RuntimeError("exact dot-product dimensions differ")
    return sum((_fr(a) * _fr(b) for a, b in zip(left, right)), Fraction(0))


def _matvec(matrix, vector):
    return [_dot(row, vector) for row in matrix]


def _fraction_matrix(value: np.ndarray) -> list[list[Fraction]]:
    return [[_fr(item) for item in row] for row in np.asarray(value)]


def _fraction_vector(value: np.ndarray) -> list[Fraction]:
    return [_fr(item) for item in np.asarray(value)]


def _exact_sqrt_upper(value: Fraction) -> Fraction:
    """Return a finite binary64 rational U with U**2 >= value."""
    value = _fr(value)
    if value < 0:
        raise RuntimeError("cannot upper-bound the square root of a negative")
    candidate = math.sqrt(float(value))
    if not math.isfinite(candidate):
        raise RuntimeError("square-root upper bound is not finite")
    upper = Fraction.from_float(candidate)
    while upper * upper < value:
        candidate = math.nextafter(candidate, math.inf)
        upper = Fraction.from_float(candidate)
    return upper


def perspective_direct(x, gamma, beta, epsilon):
    """Exact-rational direct LayerNorm and its perspective representation."""
    d = len(x)
    mean = sum(map(_fr, x), Fraction(0)) / d
    centered = [_fr(item) - mean for item in x]
    variance_plus_epsilon = (
        sum((item * item for item in centered), Fraction(0)) / d
        + _fr(epsilon))
    t = math.sqrt(float(variance_plus_epsilon))
    y = [float(_fr(g) * c) / t + float(_fr(b))
         for g, c, b in zip(gamma, centered, beta)]
    return centered, variance_plus_epsilon, t, y


def rational_lorentz_contains(centered: Sequence[Fraction],
                              t: Fraction) -> bool:
    """Check ||[2c;(1-d)t]|| <= (1+d)t using exact arithmetic."""
    centered = list(map(_fr, centered))
    t = _fr(t)
    d = len(centered)
    lhs_squared = sum(((2 * item) ** 2 for item in centered), Fraction(0))
    lhs_squared += ((1 - d) * t) ** 2
    rhs = (1 + d) * t
    return t >= 0 and rhs >= 0 and lhs_squared <= rhs * rhs


@dataclass(frozen=True)
class CanonicalConeProgram:
    """Canonical A*x=b, G*x+s=h, s in a product cone."""

    A: tuple[tuple[Fraction, ...], ...]
    b: tuple[Fraction, ...]
    G: tuple[tuple[Fraction, ...], ...]
    h: tuple[Fraction, ...]
    cones: tuple[tuple[str, int], ...]

    @property
    def variables(self):
        rows = self.A or self.G
        return len(rows[0]) if rows else 0


@dataclass(frozen=True)
class ExactLPRow:
    name: str
    indices: tuple[int, ...]
    coefficients: tuple[Fraction, ...]
    lower: Fraction | None
    upper: Fraction | None

    def __post_init__(self):
        if (len(self.indices) != len(self.coefficients)
                or tuple(sorted(self.indices)) != self.indices
                or len(set(self.indices)) != len(self.indices)
                or (self.lower is not None and self.upper is not None
                    and self.lower > self.upper)):
            raise RuntimeError(f"invalid canonical LP row: {self.name}")


@dataclass(frozen=True)
class ExactCanonicalLP:
    variable_names: tuple[str, ...]
    column_lower: tuple[Fraction | None, ...]
    column_upper: tuple[Fraction | None, ...]
    rows: tuple[ExactLPRow, ...]

    def __post_init__(self):
        n = len(self.variable_names)
        if (len(self.column_lower) != n or len(self.column_upper) != n
                or len(set(self.variable_names)) != n):
            raise RuntimeError("canonical LP variable topology differs")
        for lo, hi in zip(self.column_lower, self.column_upper):
            if lo is not None and hi is not None and lo > hi:
                raise RuntimeError("canonical LP variable bounds differ")
        if any(index < 0 or index >= n for row in self.rows
               for index in row.indices):
            raise RuntimeError("canonical LP row index is outside variables")

    @property
    def column_count(self):
        return len(self.variable_names)

    @property
    def nnz(self):
        return sum(len(row.indices) for row in self.rows)

    @cached_property
    def identity_sha256(self):
        digest = hashlib.sha256()
        digest.update(b"CORET_EXACT_CANONICAL_LP_V1\n")
        for index, (name, lo, hi) in enumerate(zip(
                self.variable_names, self.column_lower, self.column_upper)):
            digest.update(
                f"V\t{index}\t{name}\t{_optional_fs(lo)}\t"
                f"{_optional_fs(hi)}\n".encode())
        for index, row in enumerate(self.rows):
            digest.update(
                f"R\t{index}\t{row.name}\t{_optional_fs(row.lower)}\t"
                f"{_optional_fs(row.upper)}\t".encode())
            for column, coefficient in zip(row.indices, row.coefficients):
                digest.update(f"{column}:{_fs(coefficient)},".encode())
            digest.update(b"\n")
        return digest.hexdigest()

    def identity(self):
        return self.identity_sha256


@dataclass(frozen=True)
class CanonicalInequalityRef:
    """A signed base row or a finite variable bound in a*x <= b form."""

    kind: str
    index: int
    orientation: int


def _optional_fs(value):
    return "inf" if value is None else _fs(value)


def _inequality(lp: ExactCanonicalLP,
                reference: CanonicalInequalityRef):
    if reference.orientation not in (-1, 1):
        raise RuntimeError("canonical inequality orientation differs")
    if reference.kind == "row":
        row = lp.rows[reference.index]
        if reference.orientation == 1:
            if row.upper is None:
                raise RuntimeError("canonical row has no upper side")
            return row.indices, row.coefficients, row.upper
        if row.lower is None:
            raise RuntimeError("canonical row has no lower side")
        return (row.indices, tuple(-value for value in row.coefficients),
                -row.lower)
    if reference.kind == "column":
        if reference.orientation == 1:
            value = lp.column_upper[reference.index]
            if value is None:
                raise RuntimeError("canonical column has no upper bound")
            return (reference.index,), (Fraction(1),), value
        value = lp.column_lower[reference.index]
        if value is None:
            raise RuntimeError("canonical column has no lower bound")
        return (reference.index,), (Fraction(-1),), -value
    raise RuntimeError("unknown canonical inequality kind")


def canonicalize_all_inequalities(lp: ExactCanonicalLP):
    result = []
    for index, row in enumerate(lp.rows):
        if row.upper is not None:
            result.append(CanonicalInequalityRef("row", index, 1))
        if row.lower is not None:
            result.append(CanonicalInequalityRef("row", index, -1))
    for index, (lo, hi) in enumerate(zip(
            lp.column_lower, lp.column_upper)):
        if hi is not None:
            result.append(CanonicalInequalityRef("column", index, 1))
        if lo is not None:
            result.append(CanonicalInequalityRef("column", index, -1))
    return tuple(result)


def verify_exact_lp_farkas(lp: ExactCanonicalLP, certificate: dict) -> dict:
    if certificate.get("schema") != "CORET_EXACT_LP_FARKAS_CERTIFICATE_V1":
        raise RuntimeError("LP Farkas certificate schema differs")
    if certificate.get("canonical_lp_sha256") != lp.identity():
        raise RuntimeError("LP Farkas certificate model identity differs")
    q = [Fraction(0) for _ in range(lp.column_count)]
    r = Fraction(0)
    entries = certificate.get("multipliers")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("LP Farkas certificate multipliers are absent")
    for entry in entries:
        multiplier = _fr(entry["multiplier"])
        if multiplier < 0:
            raise RuntimeError("LP Farkas multiplier is negative")
        reference = CanonicalInequalityRef(
            entry["kind"], int(entry["index"]), int(entry["orientation"]))
        indices, coefficients, rhs = _inequality(lp, reference)
        for index, coefficient in zip(indices, coefficients):
            q[index] += multiplier * coefficient
        r += multiplier * rhs
    nonzero = [index for index, value in enumerate(q) if value]
    if nonzero:
        raise RuntimeError(
            f"LP Farkas stationarity differs at column {nonzero[0]}")
    if r >= 0:
        raise RuntimeError("LP Farkas contradiction is not strict")
    return {"verified": True, "exact_stationarity": True,
            "exact_lambda_b": _fs(r), "strict_contradiction": True,
            "multiplier_count": len(entries)}


def _cone_dual_member(values: Sequence[Fraction], cones) -> bool:
    offset = 0
    for kind, size in cones:
        block = list(map(_fr, values[offset:offset + size]))
        if len(block) != size:
            return False
        if kind == "nonnegative":
            if any(value < 0 for value in block):
                return False
        elif kind == "lorentz":
            if size < 2 or block[0] < 0:
                return False
            if block[0] * block[0] < sum(
                    (item * item for item in block[1:]), Fraction(0)):
                return False
        elif kind == "zero":
            if any(value != 0 for value in block):
                return False
        else:
            raise RuntimeError(f"unsupported exact cone: {kind}")
        offset += size
    return offset == len(values)


def verify_rational_conic_certificate(program: CanonicalConeProgram,
                                      certificate: dict) -> dict:
    """Exact Farkas/conic replay; solver status is deliberately ignored."""
    if certificate.get("schema") != CERTIFICATE_SCHEMA:
        raise RuntimeError("conic certificate schema differs")
    y = tuple(_fr(item) for item in certificate.get("equality_dual", []))
    z = tuple(_fr(item) for item in certificate.get("cone_dual", []))
    if len(y) != len(program.b) or len(z) != len(program.h):
        raise RuntimeError("conic certificate dimensions differ")
    if not _cone_dual_member(z, program.cones):
        raise RuntimeError("dual value is outside the exact dual cone")
    residual = []
    for column in range(program.variables):
        value = sum((program.A[row][column] * y[row]
                     for row in range(len(y))), Fraction(0))
        value += sum((program.G[row][column] * z[row]
                      for row in range(len(z))), Fraction(0))
        residual.append(value)
    if any(residual):
        raise RuntimeError("exact conic dual stationarity differs")
    separator = _dot(program.b, y) + _dot(program.h, z)
    if separator >= 0:
        raise RuntimeError("conic certificate is not strictly separating")
    return {"verified": True, "maximum_exact_residual": "0",
            "strict_separator": _fs(separator),
            "dual_cone_membership": True}


class ExactSolveFailure(RuntimeError):
    pass


class HighsCanonicalLPDiagnosticError(RuntimeError):
    def __init__(self, message: str, diagnostic: dict):
        super().__init__(message)
        self.diagnostic = diagnostic


@dataclass(frozen=True)
class SolverScaledLP:
    lp: ExactCanonicalLP
    scales: tuple[int, ...]
    exponents: tuple[int, ...]
    report: dict


def _semantic_family(name: str) -> str:
    return name.split("[", 1)[0]


def build_highs_row_scaled_lp(
        lp: ExactCanonicalLP, *, small_matrix_value: float,
        large_matrix_value: float, infinite_bound: float,
        target_min_abs: float = 1e-8) -> SolverScaledLP:
    """Build an exactly equivalent power-of-two row-scaled solver copy."""
    if not (0 < small_matrix_value < target_min_abs < large_matrix_value):
        raise RuntimeError("HiGHS row-scaling thresholds are invalid")
    target = Fraction.from_float(float(target_min_abs))
    scales, exponents, scaled_rows = [], [], []
    tiny_entries, scaled_families = [], {}
    pre_nonzero, post_nonzero, post_bounds = [], [], []
    for side, bounds in (("column_lower", lp.column_lower),
                         ("column_upper", lp.column_upper)):
        for column, bound in enumerate(bounds):
            if bound is None:
                continue
            converted = float(bound)
            post_bounds.append(abs(converted))
            if (not math.isfinite(converted)
                    or abs(converted) >= infinite_bound):
                raise HighsCanonicalLPDiagnosticError(
                    "a finite column bound is unsafe for HiGHS", {
                        "failure_classification":
                            "HIGHS_SAFE_ROW_SCALING_IMPOSSIBLE",
                        "column_index": column, "bound_side": side,
                        "exact_bound": _fs(bound),
                        "infinite_bound": infinite_bound,
                    })
    for row_index, row in enumerate(lp.rows):
        exact_nonzero = [abs(value) for value in row.coefficients if value]
        if not exact_nonzero:
            scale, exponent = 1, 0
        else:
            minimum = min(exact_nonzero)
            problematic = any(
                0 < abs(float(value)) <= small_matrix_value
                for value in row.coefficients)
            scale, exponent = 1, 0
            if problematic:
                while minimum * scale < target:
                    scale <<= 1
                    exponent += 1
            family = _semantic_family(row.name)
            if exponent:
                scaled_families[family] = scaled_families.get(family, 0) + 1
        scaled_coefficients = tuple(value * scale for value in row.coefficients)
        scaled_lower = None if row.lower is None else row.lower * scale
        scaled_upper = None if row.upper is None else row.upper * scale
        for column, original, scaled in zip(
                row.indices, row.coefficients, scaled_coefficients):
            original_float, scaled_float = float(original), float(scaled)
            if original:
                pre_nonzero.append(abs(original_float))
                post_nonzero.append(abs(scaled_float))
            if original and abs(original_float) <= small_matrix_value:
                variable_name = lp.variable_names[column]
                tiny_entries.append({
                    "row_index": row_index, "row_name": row.name,
                    "row_family": _semantic_family(row.name),
                    "column_index": column, "variable_name": variable_name,
                    "variable_family": _semantic_family(variable_name),
                    "exact_rational_value": _fs(original),
                    "float_value": original_float,
                    "scaled_exact_rational_value": _fs(scaled),
                    "scaled_float_value": scaled_float,
                    "scale_exponent": exponent,
                })
            if original and (not math.isfinite(scaled_float)
                             or abs(scaled_float) <= small_matrix_value
                             or abs(scaled_float) >= large_matrix_value):
                raise HighsCanonicalLPDiagnosticError(
                    "no safe exact power-of-two row scaling exists", {
                        "failure_classification":
                            "HIGHS_SAFE_ROW_SCALING_IMPOSSIBLE",
                        "row_index": row_index, "row_name": row.name,
                        "column_index": column,
                        "exact_coefficient": _fs(original),
                        "scaled_exact_coefficient": _fs(scaled),
                        "scale_exponent": exponent,
                        "small_matrix_value": small_matrix_value,
                        "large_matrix_value": large_matrix_value,
                    })
        for side, bound in (("lower", scaled_lower),
                            ("upper", scaled_upper)):
            if bound is None:
                continue
            converted = float(bound)
            post_bounds.append(abs(converted))
            if (not math.isfinite(converted)
                    or abs(converted) >= infinite_bound):
                raise HighsCanonicalLPDiagnosticError(
                    "row scaling makes a finite bound unsafe", {
                        "failure_classification":
                            "HIGHS_SAFE_ROW_SCALING_IMPOSSIBLE",
                        "row_index": row_index, "row_name": row.name,
                        "bound_side": side, "exact_bound": _fs(bound),
                        "scale_exponent": exponent,
                        "infinite_bound": infinite_bound,
                    })
        scales.append(scale); exponents.append(exponent)
        scaled_rows.append(ExactLPRow(
            row.name, row.indices, scaled_coefficients,
            scaled_lower, scaled_upper))
    scaled_lp = ExactCanonicalLP(
        lp.variable_names, lp.column_lower, lp.column_upper,
        tuple(scaled_rows))
    scales_exact = [_fs(Fraction(value)) for value in scales]
    row_scaling_sha256 = _sha_json(scales_exact)
    tiny_by_row, tiny_by_variable = {}, {}
    for entry in tiny_entries:
        row_family = entry["row_family"]
        variable_family = entry["variable_family"]
        tiny_by_row[row_family] = tiny_by_row.get(row_family, 0) + 1
        tiny_by_variable[variable_family] = \
            tiny_by_variable.get(variable_family, 0) + 1
    report = {
        "schema": "CORET_HIGHS_EXACT_ROW_SCALING_V1",
        "method": "POSITIVE_POWER_OF_TWO_ROW_SCALING",
        "target_min_abs": target_min_abs,
        "target_min_abs_exact_binary64": _fs(target),
        "rows_scaled": sum(exponent > 0 for exponent in exponents),
        "maximum_scale_exponent": max(exponents, default=0),
        "minimum_pre_scale_nonzero_abs": min(pre_nonzero, default=None),
        "minimum_post_scale_nonzero_abs": min(post_nonzero, default=None),
        "maximum_pre_scale_abs": max(pre_nonzero, default=None),
        "maximum_post_scale_abs": max(post_nonzero, default=None),
        "maximum_post_scale_finite_bound": max(post_bounds, default=None),
        "original_canonical_lp_sha256": lp.identity(),
        "solver_scaled_lp_sha256": scaled_lp.identity(),
        "row_scaling_sha256": row_scaling_sha256,
        "row_scale_exponents": exponents,
        "row_scales_exact": scales_exact,
        "sub_threshold_entries_before": len(tiny_entries),
        "sub_threshold_entries_after": sum(
            value != 0 and abs(value) <= small_matrix_value
            for value in post_nonzero),
        "semantic_distribution_of_scaled_rows": scaled_families,
        "sub_threshold_distribution_by_row_family": tiny_by_row,
        "sub_threshold_distribution_by_variable_family": tiny_by_variable,
        "sub_threshold_entries": tiny_entries,
        "small_matrix_value": small_matrix_value,
        "large_matrix_value": large_matrix_value,
        "infinite_bound": infinite_bound,
    }
    if report["sub_threshold_entries_after"]:
        raise HighsCanonicalLPDiagnosticError(
            "scaled solver LP still has sub-threshold coefficients", {
                **report,
                "failure_classification":
                    "HIGHS_SAFE_ROW_SCALING_IMPOSSIBLE"})
    return SolverScaledLP(scaled_lp, tuple(scales), tuple(exponents), report)


def map_solver_row_ray_to_original(raw_ray, scales):
    raw = np.asarray(raw_ray, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    if raw.ndim != 1 or scales.shape != raw.shape:
        raise RuntimeError("HiGHS row ray/scaling topology differs")
    mapped = raw * scales
    if not np.isfinite(mapped).all():
        raise RuntimeError("mapped HiGHS row ray is nonfinite")
    return mapped


def map_solver_row_multipliers_exact(multipliers, scales):
    if len(multipliers) != len(scales):
        raise RuntimeError("solver multiplier/scaling topology differs")
    return tuple(_fr(value) * int(scale)
                 for value, scale in zip(multipliers, scales))


def persist_solver_row_scaling(scaling: SolverScaledLP, path: Path):
    return _atomic_json(path, scaling.report)


def _highs_numeric_arrays(lp: ExactCanonicalLP, infinity: float):
    starts, indices, values = [0], [], []
    for row in lp.rows:
        indices.extend(row.indices)
        values.extend(float(value) for value in row.coefficients)
        starts.append(len(indices))
    return {
        "objective": np.zeros(lp.column_count, dtype=np.float64),
        "column_lower": np.asarray([
            -infinity if value is None else float(value)
            for value in lp.column_lower], dtype=np.float64),
        "column_upper": np.asarray([
            infinity if value is None else float(value)
            for value in lp.column_upper], dtype=np.float64),
        "row_lower": np.asarray([
            -infinity if row.lower is None else float(row.lower)
            for row in lp.rows], dtype=np.float64),
        "row_upper": np.asarray([
            infinity if row.upper is None else float(row.upper)
            for row in lp.rows], dtype=np.float64),
        "starts": starts, "indices": indices, "values": values,
        "finite_bound_masks": (
            [value is not None for value in lp.column_lower],
            [value is not None for value in lp.column_upper],
            [row.lower is not None for row in lp.rows],
            [row.upper is not None for row in lp.rows]),
    }


def _diagnose_highs_lp(lp: ExactCanonicalLP, arrays: dict,
                       thresholds: dict):
    return diagnose_highs_arrays(
        column_count=lp.column_count, row_count=len(lp.rows),
        objective=arrays["objective"],
        column_lower=arrays["column_lower"],
        column_upper=arrays["column_upper"],
        row_lower=arrays["row_lower"], row_upper=arrays["row_upper"],
        starts=arrays["starts"], indices=arrays["indices"],
        values=arrays["values"],
        small_matrix_value=thresholds["small_matrix_value"],
        large_matrix_value=thresholds["large_matrix_value"],
        infinite_bound=thresholds["infinite_bound"],
        finite_bound_masks=arrays["finite_bound_masks"])


def _require_solver_scaling_applied(
        original_lp: ExactCanonicalLP, solver_lp: ExactCanonicalLP,
        scaling: SolverScaledLP | None, original_diagnostic: dict,
        scaled_diagnostic: dict, scaling_path: Path | None):
    report = None if scaling is None else scaling.report
    valid = bool(
        report
        and report.get("method") == "POSITIVE_POWER_OF_TWO_ROW_SCALING"
        and report.get("original_canonical_lp_sha256") ==
            original_lp.identity()
        and report.get("solver_scaled_lp_sha256") == solver_lp.identity()
        and report.get("row_scaling_sha256")
        and report.get("sub_threshold_entries_before") ==
            original_diagnostic.get("sub_small_matrix_value_count")
        and report.get("sub_threshold_entries_after") == 0
        and scaled_diagnostic.get("sub_small_matrix_value_count") == 0)
    persisted = True
    if scaling_path is not None:
        persisted = scaling_path.is_file()
        if persisted:
            persisted_record = cluster_common.verified_json(scaling_path)
            persisted = (
                persisted_record.get("row_scaling_sha256") ==
                    report.get("row_scaling_sha256")
                and persisted_record.get("solver_scaled_lp_sha256") ==
                    solver_lp.identity())
    if not valid or not persisted:
        raise HighsCanonicalLPDiagnosticError(
            "solver-scaled LP was not applied before passModel", {
                "failure_classification": "HIGHS_SOLVER_SCALING_NOT_APPLIED",
                "failed_api_call": "pre_passModel_scaling_gate",
                "original_canonical_lp_sha256": original_lp.identity(),
                "solver_lp_sha256": solver_lp.identity(),
                "solver_scaling": report,
                "original_diagnostic": original_diagnostic,
                "scaled_diagnostic": scaled_diagnostic,
                "scaling_path": (None if scaling_path is None
                                 else str(scaling_path)),
                "scaling_artifact_persisted": persisted,
            })
    return True


def _numeric_distribution(values, *, allow_infinity=False):
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    nonzero = finite & (array != 0.0)
    absolute = np.abs(array[finite])
    nonzero_absolute = np.abs(array[nonzero])
    buckets = {}
    if nonzero_absolute.size:
        exponents = np.floor(np.log10(nonzero_absolute)).astype(np.int64)
        unique, counts = np.unique(exponents, return_counts=True)
        buckets = {str(int(exponent)): int(count)
                   for exponent, count in zip(unique, counts)}
    result = {
        "count": int(array.size), "finite_count": int(finite.sum()),
        "nonfinite_count": int((~finite).sum()),
        "nan_count": int(np.isnan(array).sum()),
        "positive_infinity_count": int(np.isposinf(array).sum()),
        "negative_infinity_count": int(np.isneginf(array).sum()),
        "maximum_absolute_finite": (
            None if not absolute.size else float(absolute.max())),
        "minimum_nonzero_absolute_finite": (
            None if not nonzero_absolute.size else
            float(nonzero_absolute.min())),
        "nonzero_decimal_exponent_histogram": buckets,
        "infinity_allowed": bool(allow_infinity),
    }
    if (~finite).any():
        first = int(np.flatnonzero(~finite)[0])
        result["first_nonfinite"] = {"offset": first,
                                     "value": repr(float(array[first]))}
    return result


def diagnose_highs_arrays(*, column_count, row_count, objective,
                          column_lower, column_upper, row_lower, row_upper,
                          starts, indices, values,
                          small_matrix_value=1e-9,
                          large_matrix_value=1e15,
                          infinite_bound=1e20,
                          finite_bound_masks=None):
    starts = np.asarray(starts)
    indices = np.asarray(indices)
    values = np.asarray(values, dtype=np.float64)
    fields = {
        "objective": _numeric_distribution(objective),
        "column_lower_bounds": _numeric_distribution(
            column_lower, allow_infinity=True),
        "column_upper_bounds": _numeric_distribution(
            column_upper, allow_infinity=True),
        "row_lower_bounds": _numeric_distribution(
            row_lower, allow_infinity=True),
        "row_upper_bounds": _numeric_distribution(
            row_upper, allow_infinity=True),
        "matrix_coefficients": _numeric_distribution(values),
    }
    malformed, duplicates, unsorted = [], [], []
    duplicate_total = 0
    pointer_valid = (starts.ndim == 1 and len(starts) == row_count + 1
                     and len(starts) > 0 and int(starts[0]) == 0
                     and np.all(starts[1:] >= starts[:-1])
                     and int(starts[-1]) == len(indices) == len(values))
    if pointer_valid:
        for row in range(row_count):
            begin, end = int(starts[row]), int(starts[row + 1])
            row_indices = indices[begin:end]
            duplicate_count = len(row_indices) - len(
                set(int(value) for value in row_indices))
            if duplicate_count:
                duplicate_total += duplicate_count
                duplicates.append({"row": row, "duplicate_count": duplicate_count})
            if len(row_indices) > 1 and np.any(row_indices[1:] <= row_indices[:-1]):
                unsorted.append({"row": row,
                                 "indices": [int(value) for value in
                                             row_indices[:16]]})
    else:
        malformed.append({"kind": "csr_pointer",
                          "pointer_length": int(starts.size),
                          "expected_pointer_length": row_count + 1,
                          "terminal": (None if not starts.size
                                       else int(starts[-1])),
                          "expected_terminal": len(indices)})
    outside = np.flatnonzero((indices < 0) | (indices >= column_count)) \
        if indices.size else np.empty(0, dtype=np.int64)
    finite_coefficients = values[np.isfinite(values)]
    absolute_nonzero = np.abs(finite_coefficients[finite_coefficients != 0])
    small = np.flatnonzero(
        np.isfinite(values) & (values != 0)
        & (np.abs(values) < float(small_matrix_value)))
    large = np.flatnonzero(
        np.isfinite(values) & (np.abs(values) > float(large_matrix_value)))
    bound_arrays = [np.asarray(item, dtype=np.float64) for item in
                    (column_lower, column_upper, row_lower, row_upper)]
    if finite_bound_masks is None:
        finite_bound_masks = [
            ~np.isneginf(bound_arrays[0]), ~np.isposinf(bound_arrays[1]),
            ~np.isneginf(bound_arrays[2]), ~np.isposinf(bound_arrays[3])]
    finite_bound_masks = [np.asarray(mask, dtype=bool)
                          for mask in finite_bound_masks]
    if (len(finite_bound_masks) != 4
            or any(mask.shape != item.shape for mask, item in
                   zip(finite_bound_masks, bound_arrays))):
        raise ValueError("finite bound masks differ from bound arrays")
    finite_bounds = np.concatenate([item[np.isfinite(item)]
                                    for item in bound_arrays])
    large_bound_details = []
    bound_names = ("column_lower", "column_upper",
                   "row_lower", "row_upper")
    for name, array, finite_mask in zip(
            bound_names, bound_arrays, finite_bound_masks):
        for offset in np.flatnonzero(
                finite_mask & np.isfinite(array)
                & (np.abs(array) >= float(infinite_bound))):
            large_bound_details.append({
                "field": name, "offset": int(offset),
                "value": float(array[offset])})
    improper_nonfinite_bounds = []
    for name, array, finite_mask in zip(
            bound_names, bound_arrays, finite_bound_masks):
        for offset in np.flatnonzero(finite_mask & ~np.isfinite(array)):
            improper_nonfinite_bounds.append({
                "field": name, "offset": int(offset),
                "value": repr(float(array[offset]))})
    diagnostic = {
        "column_count": int(column_count), "row_count": int(row_count),
        "nnz": int(len(values)), "numeric_fields": fields,
        "maximum_absolute_finite_coefficient": (
            None if not finite_coefficients.size else
            float(np.max(np.abs(finite_coefficients)))),
        "minimum_nonzero_absolute_finite_coefficient": (
            None if not absolute_nonzero.size else
            float(np.min(absolute_nonzero))),
        "maximum_finite_bound_magnitude": (
            None if not finite_bounds.size else
            float(np.max(np.abs(finite_bounds)))),
        "malformed_row_count": len(malformed),
        "malformed_rows": malformed[:16],
        "duplicate_column_index_count": int(duplicate_total),
        "duplicate_column_index_row_count": len(duplicates),
        "duplicate_column_index_rows": duplicates[:16],
        "unsorted_index_row_count": len(unsorted),
        "unsorted_index_rows": unsorted[:16],
        "index_minimum": None if not indices.size else int(indices.min()),
        "index_maximum": None if not indices.size else int(indices.max()),
        "outside_index_count": int(outside.size),
        "first_outside_index": (None if not outside.size else {
            "offset": int(outside[0]), "value": int(indices[outside[0]])}),
        "sparse_encoding": "CSR_ROW_WISE",
        "pointer_length": int(starts.size),
        "expected_pointer_length": int(row_count + 1),
        "pointer_terminal_nnz": (None if not starts.size
                                 else int(starts[-1])),
        "expected_terminal_nnz": int(len(values)),
        "pointer_structure_valid": bool(pointer_valid),
        "highs_thresholds": {
            "small_matrix_value": float(small_matrix_value),
            "large_matrix_value": float(large_matrix_value),
            "infinite_bound": float(infinite_bound)},
        "sub_small_matrix_value_count": int(small.size),
        "first_sub_small_matrix_value": (None if not small.size else {
            "offset": int(small[0]), "value": float(values[small[0]])}),
        "over_large_matrix_value_count": int(large.size),
        "first_over_large_matrix_value": (None if not large.size else {
            "offset": int(large[0]), "value": float(values[large[0]])}),
        "over_infinite_bound_magnitude_count": len(large_bound_details),
        "first_over_infinite_bound_magnitude": (
            None if not large_bound_details else large_bound_details[0]),
        "improper_nonfinite_bound_count": len(improper_nonfinite_bounds),
        "first_improper_nonfinite_bound": (
            None if not improper_nonfinite_bounds
            else improper_nonfinite_bounds[0]),
    }
    classification = None
    if (fields["objective"]["nonfinite_count"]
            or fields["matrix_coefficients"]["nonfinite_count"]):
        classification = "NONFINITE_CANONICAL_COEFFICIENT"
    elif improper_nonfinite_bounds:
        classification = "NONFINITE_CANONICAL_BOUND"
    elif outside.size:
        classification = "HIGHS_MATRIX_INDEX_INVALID"
    elif not pointer_valid or duplicates or unsorted:
        classification = "HIGHS_MATRIX_STRUCTURE_INVALID"
    elif small.size or large.size:
        classification = "HIGHS_COEFFICIENT_MAGNITUDE_REJECTED"
    elif large_bound_details:
        classification = "HIGHS_BOUND_MAGNITUDE_REJECTED"
    diagnostic["preflight_failure_classification"] = classification
    diagnostic["preflight_passed"] = classification is None
    return diagnostic


def _api_failure_classification(api: str, diagnostic: dict):
    preflight = diagnostic.get("preflight_failure_classification")
    if preflight:
        return preflight
    return {
        "addCols": "HIGHS_ADD_COLS_REJECTED",
        "addRows": "HIGHS_ADD_ROWS_REJECTED",
        "passModel": "HIGHS_PASS_MODEL_REJECTED",
        "passLp": "HIGHS_PASS_MODEL_REJECTED",
        "changeColsBounds": "HIGHS_ADD_COLS_REJECTED",
        "changeRowsBounds": "HIGHS_ADD_ROWS_REJECTED",
        "run": "HIGHS_RUN_REJECTED",
    }.get(api, "UNKNOWN_HIGHS_MODEL_REJECTION")


def _check_highs_status(api: str, status, ok_status, diagnostic: dict,
                        log_path: Path | None):
    diagnostic.setdefault("api_statuses", []).append(
        {"api": api, "status": str(status)})
    if status == ok_status:
        return
    diagnostic.update({
        "failed_api_call": api,
        "failed_highs_status": str(status),
        "failure_classification": _api_failure_classification(api, diagnostic),
        "highs_log_path": None if log_path is None else str(log_path),
        "highs_log_text": (None if log_path is None or not log_path.is_file()
                           else log_path.read_text(errors="replace")[-20000:]),
    })
    raise HighsCanonicalLPDiagnosticError(
        f"{api} rejected canonical LP with {status}", diagnostic)


def _integerize(values: Sequence[Fraction]):
    denominators = [value.denominator for value in values]
    common = 1
    for denominator in denominators:
        common = math.lcm(common, denominator)
    integers = [value.numerator * (common // value.denominator)
                for value in values]
    divisor = 0
    for value in integers:
        divisor = math.gcd(divisor, abs(value))
    if divisor > 1:
        integers = [value // divisor for value in integers]
    return integers


def _bareiss_solve(matrix, rhs, timeout_seconds=30.0):
    """Deterministic exact square solve; Fraction only in back substitution."""
    n = len(matrix)
    if n != len(rhs) or any(len(row) != n for row in matrix):
        raise ExactSolveFailure("exact repair system is not square")
    augmented = [list(map(int, row)) + [int(value)]
                 for row, value in zip(matrix, rhs)]
    deadline = time.perf_counter() + timeout_seconds
    previous = 1
    for column in range(n - 1):
        if time.perf_counter() > deadline:
            raise ExactSolveFailure("exact repair Bareiss timeout")
        pivot = next((row for row in range(column, n)
                      if augmented[row][column]), None)
        if pivot is None:
            raise ExactSolveFailure("exact repair system is singular")
        if pivot != column:
            augmented[column], augmented[pivot] = \
                augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        for row in range(column + 1, n):
            factor = augmented[row][column]
            for target in range(column + 1, n + 1):
                numerator = (augmented[row][target] * pivot_value
                             - factor * augmented[column][target])
                if numerator % previous:
                    raise ExactSolveFailure("Bareiss division is nonexact")
                augmented[row][target] = numerator // previous
            augmented[row][column] = 0
        previous = pivot_value
    if n and augmented[-1][-2] == 0:
        raise ExactSolveFailure("exact repair system is singular")
    solution = [Fraction(0) for _ in range(n)]
    for row in range(n - 1, -1, -1):
        value = Fraction(augmented[row][-1]) - sum(
            (Fraction(augmented[row][column]) * solution[column]
             for column in range(row + 1, n)), Fraction(0))
        solution[row] = value / augmented[row][row]
    return solution


def _integer_rows_sha256(rows):
    """Unambiguous integer hash without Python's decimal digit limit."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(len(row).to_bytes(8, "big"))
        for value in row:
            value = int(value)
            payload = abs(value).to_bytes(max(1, (abs(value).bit_length() + 7) // 8), "big")
            digest.update(bytes([value < 0]))
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    return digest.hexdigest()


def _integerize_correction_system(matrix, rhs_columns):
    """Normalize A independently; clear each transformed RHS by column.

    Clearing a row using RHS denominators needlessly enlarges A and all its
    minors. A-only primitive rows permit the same elimination for every RHS.
    """
    started = time.perf_counter()
    n = len(matrix)
    if not n or any(len(row) != n for row in matrix) or any(
            len(rhs) != n for rhs in rhs_columns):
        raise ExactSolveFailure("correction system topology differs")
    integer_matrix, transformed = [], [[] for _ in rhs_columns]
    denominator_bits, gcd_bits, row_scales, signs = [], [], [], []
    for i, row in enumerate(matrix):
        common = math.lcm(*(value.denominator for value in row))
        integers = [value.numerator * (common // value.denominator) for value in row]
        divisor = math.gcd(*integers)
        if not divisor:
            raise ExactSolveFailure("correction matrix has a zero row")
        sign = 1 if next(value for value in integers if value) > 0 else -1
        scale = Fraction(sign * common, divisor)
        integer_matrix.append([sign * value // divisor for value in integers])
        for column, rhs in zip(transformed, rhs_columns):
            column.append(rhs[i] * scale)
        denominator_bits.append(common.bit_length())
        gcd_bits.append(divisor.bit_length() - 1)
        row_scales.append(scale)
        signs.append(sign)
    rhs_denominators = [math.lcm(*(value.denominator for value in column))
                        for column in transformed]
    integer_rhs = [[value.numerator * (denominator // value.denominator)
                    for value in column]
                   for column, denominator in zip(transformed, rhs_denominators)]
    diagnostics = {
        "dimension": n, "rhs_count": len(rhs_columns),
        "nonzero_count": sum(bool(value) for row in matrix for value in row),
        "maximum_coefficient_numerator_bit_length": max(
            abs(value.numerator).bit_length() for row in matrix for value in row),
        "maximum_coefficient_denominator_bit_length": max(
            value.denominator.bit_length() for row in matrix for value in row),
        "integerized_matrix_maximum_bit_length": max(
            abs(value).bit_length() for row in integer_matrix for value in row),
        "rhs_maximum_bit_length": max(
            (abs(value).bit_length() for column in integer_rhs for value in column), default=0),
        "row_common_denominator_bits": denominator_bits,
        "row_gcd_bits_removed": gcd_bits,
        "row_signs": signs,
        "row_scales_sha256": _sha_json([_fs(value) for value in row_scales]),
        "rhs_column_denominator_bits": [value.bit_length() for value in rhs_denominators],
        "normalized_integer_matrix_sha256": _integer_rows_sha256(integer_matrix),
        "integer_rhs_sha256": _integer_rows_sha256(integer_rhs),
        "integerization_seconds": time.perf_counter() - started,
    }
    return integer_matrix, integer_rhs, rhs_denominators, diagnostics


def _exact_multi_rhs_solve(matrix, rhs_columns, timeout_seconds=30.0,
                         factor_cache=None, diagnostics=None):
    """One deterministic Bareiss elimination, reusable for arbitrary RHS.

    Cache entries are untrusted: every answer is replayed against the original
    integer matrix and RHS before it can be used in an affine family.
    """
    started = time.perf_counter()
    deadline = started + timeout_seconds
    n, count = len(matrix), len(rhs_columns)
    if (not n or not count or any(len(row) != n for row in matrix)
            or any(len(rhs) != n for rhs in rhs_columns)
            or any(type(value) is not int for row in matrix for value in row)
            or any(type(value) is not int for rhs in rhs_columns for value in rhs)):
        raise ExactSolveFailure("integer multi-RHS system topology/type differs")
    report = diagnostics if diagnostics is not None else {}
    matrix_sha = _integer_rows_sha256(matrix)
    factor = (factor_cache or {}).get(matrix_sha)
    if factor is not None and factor["matrix"] != matrix:
        raise ExactSolveFailure("cached correction matrix identity differs")
    if factor is not None:
        identity = _sha_json({
            "matrix_sha256": matrix_sha, "rows": factor["rows"],
            "columns": factor["columns"],
            "upper_sha256": _integer_rows_sha256(factor["upper"]),
            "steps_sha256": _integer_rows_sha256([
                [k, chosen, pivot, previous, *values]
                for k, chosen, pivot, previous, values in factor["steps"]])})
        if identity != factor["factorization_sha256"]:
            raise ExactSolveFailure("cached exact factorization identity differs")
    report.update(matrix_sha256=matrix_sha, rhs_count=count,
                  elimination_reused=factor is not None,
                  elimination_count=0 if factor is not None else 1,
                  pre_product_common_factor_cancellations=0,
                  maximum_pre_product_cancelled_factor_bit_length=0,
                  peak_intermediate_integer_bit_length=max(
                      abs(value).bit_length() for row in [*matrix, *rhs_columns] for value in row))

    def check_time():
        if time.perf_counter() >= deadline:
            raise ExactSolveFailure("exact repair multi-RHS Bareiss timeout")

    def note(value):
        report["peak_intermediate_integer_bit_length"] = max(
            report["peak_intermediate_integer_bit_length"], abs(value).bit_length())

    def row_factors(pivot, factor_value, previous, entry_count):
        # Cancel a common factor BEFORE products, without changing Bareiss's
        # recurrence. Arbitrary row-content cancellation would break it.
        common = math.gcd(math.gcd(pivot, factor_value), previous)
        if common > 1:
            report["pre_product_common_factor_cancellations"] += entry_count
            report["maximum_pre_product_cancelled_factor_bit_length"] = max(
                report["maximum_pre_product_cancelled_factor_bit_length"], common.bit_length())
        return pivot // common, factor_value // common, previous // common

    def reduce_entry(value, pivot_entry, pivot, factor_value, denominator):
        first = value * pivot
        second = factor_value * pivot_entry
        numerator = first - second
        quotient, remainder = divmod(numerator, denominator)
        note(first); note(second); note(numerator); note(quotient)
        if remainder:
            raise ExactSolveFailure("multi-RHS Bareiss division is nonexact")
        return quotient

    if factor is None:
        # Permutations depend on A only, never on a RHS or a floating solve.
        column_order = sorted(range(n), key=lambda j: (
            sum(bool(row[j]) for row in matrix),
            max(abs(row[j]).bit_length() for row in matrix), j))
        row_order = sorted(range(n), key=lambda i: (
            sum(bool(value) for value in matrix[i]),
            max(abs(value).bit_length() for value in matrix[i]), i))
        work = [[matrix[i][j] for j in column_order] +
                [rhs[i] for rhs in rhs_columns] for i in row_order]
        previous, steps = 1, []
        elimination_started = time.perf_counter()
        try:
            for k in range(n - 1):
                check_time()
                candidates = [i for i in range(k, n) if work[i][k]]
                if not candidates:
                    raise ExactSolveFailure("exact repair system is singular")
                chosen = min(candidates, key=lambda i: (abs(work[i][k]).bit_length(),
                                                       abs(work[i][k]), i))
                if chosen != k:
                    work[k], work[chosen] = work[chosen], work[k]
                pivot = work[k][k]
                factors = tuple(work[i][k] for i in range(k + 1, n))
                steps.append((k, chosen, pivot, previous, factors))
                for i in range(k + 1, n):
                    check_time()
                    value = work[i][k]
                    p, f, denominator = row_factors(pivot, value, previous, n + count - k - 1)
                    for j in range(k + 1, n + count):
                        work[i][j] = reduce_entry(work[i][j], work[k][j],
                                                  p, f, denominator)
                    work[i][k] = 0
                previous = pivot
        finally:
            report["bareiss_elimination_seconds"] = time.perf_counter() - elimination_started
        if not work[-1][n - 1]:
            raise ExactSolveFailure("exact repair system is singular")
        factor = {"matrix": [row[:] for row in matrix], "rows": row_order,
                  "columns": column_order, "upper": [row[:n] for row in work],
                  "steps": steps,
                  "peak_intermediate_integer_bit_length": report["peak_intermediate_integer_bit_length"]}
        factor["factorization_sha256"] = _sha_json({
            "matrix_sha256": matrix_sha, "rows": row_order, "columns": column_order,
            "upper_sha256": _integer_rows_sha256(factor["upper"]),
            "steps_sha256": _integer_rows_sha256([
                [k, chosen, pivot, previous, *values]
                for k, chosen, pivot, previous, values in steps])})
        if factor_cache is not None:
            factor_cache[matrix_sha] = factor
        transformed_rhs = [row[n:] for row in work]
    else:
        transformed_rhs = [[rhs[i] for rhs in rhs_columns] for i in factor["rows"]]
        elimination_started = time.perf_counter()
        try:
            for k, chosen, pivot, previous, factors in factor["steps"]:
                check_time()
                if chosen != k:
                    transformed_rhs[k], transformed_rhs[chosen] = \
                        transformed_rhs[chosen], transformed_rhs[k]
                for i, value in zip(range(k + 1, n), factors):
                    p, f, denominator = row_factors(pivot, value, previous, count)
                    for j in range(count):
                        transformed_rhs[i][j] = reduce_entry(
                            transformed_rhs[i][j], transformed_rhs[k][j],
                            p, f, denominator)
        finally:
            report["rhs_elimination_replay_seconds"] = time.perf_counter() - elimination_started
        report["bareiss_elimination_seconds"] = 0.0
    report.update(factorization_sha256=factor["factorization_sha256"],
                  row_permutation=factor["rows"], column_permutation=factor["columns"])
    solutions = []
    back_started = time.perf_counter()
    try:
        for column in range(count):
            solution = [Fraction(0)] * n
            for i in range(n - 1, -1, -1):
                check_time()
                value = Fraction(transformed_rhs[i][column]) - sum(
                    (factor["upper"][i][j] * solution[j] for j in range(i + 1, n)), Fraction(0))
                solution[i] = value / factor["upper"][i][i]
            original_order = [Fraction(0)] * n
            for j, original in enumerate(factor["columns"]):
                original_order[original] = solution[j]
            solutions.append(original_order)
    finally:
        report["back_substitution_seconds"] = time.perf_counter() - back_started
    replay_started = time.perf_counter()
    for solution, rhs in zip(solutions, rhs_columns):
        for row, expected in zip(matrix, rhs):
            check_time()
            if sum((value * x for value, x in zip(row, solution)), Fraction(0)) != expected:
                raise ExactSolveFailure("selected multi-RHS system exact replay failed")
    report.update(selected_system_replay_seconds=time.perf_counter() - replay_started,
                  selected_system_replay_verified=True,
                  exact_linear_solve_seconds=time.perf_counter() - started)
    return solutions, report


def _solve_exact_correction_rhs(matrix, rhs_columns, timeout_seconds,
                                factor_cache, diagnostics):
    started = time.perf_counter()
    integers, rhs, denominators, profile = _integerize_correction_system(
        matrix, rhs_columns)
    diagnostics.update(profile)
    diagnostics["rational_matrix_sha256"] = _integer_rows_sha256([
        [part for value in row for part in (value.numerator, value.denominator)]
        for row in matrix])
    # A shared exact scalar in a generator column should not inflate every
    # determinant. Solve for y_j=gcd_j*x_j and undo that scaling exactly.
    column_divisors = [math.gcd(*(row[j] for row in integers))
                       for j in range(len(integers))]
    if any(value == 0 for value in column_divisors):
        raise ExactSolveFailure("correction matrix has a zero column")
    primitive = [[value // divisor for value, divisor in zip(row, column_divisors)]
                 for row in integers]
    diagnostics.update(
        column_gcd_bits_removed=[value.bit_length() - 1 for value in column_divisors],
        column_scaling_sha256=_integer_rows_sha256([column_divisors]),
        column_primitive_matrix_sha256=_integer_rows_sha256(primitive),
        column_primitive_matrix_maximum_bit_length=max(
            abs(value).bit_length() for row in primitive for value in row))
    print(json.dumps({"stage": "exact_correction_system_before_solve",
                      **diagnostics}), flush=True)
    remaining = timeout_seconds - (time.perf_counter() - started)
    if remaining <= 0:
        raise ExactSolveFailure("correction integerization deadline")
    solved, _report = _exact_multi_rhs_solve(
        primitive, rhs, remaining, factor_cache, diagnostics)
    solutions = [[value / (denominator * divisor)
                  for value, divisor in zip(column, column_divisors)]
                 for column, denominator in zip(solved, denominators)]
    replay_started = time.perf_counter()
    for solution, expected in zip(solutions, rhs_columns):
        for row, value in zip(matrix, expected):
            if _dot(row, solution) != value:
                raise ExactSolveFailure("original rational correction system replay failed")
    diagnostics.update(original_rational_system_replay_verified=True,
                       original_rational_system_replay_seconds=time.perf_counter() - replay_started)
    print(json.dumps({"stage": "exact_correction_system_solved",
                      **diagnostics}), flush=True)
    return solutions


def _oriented_ray_rows(lp: ExactCanonicalLP, raw_ray, convention: int,
                       threshold: float):
    result = []
    for index, raw in enumerate(raw_ray):
        value = float(raw)
        if not math.isfinite(value):
            raise RuntimeError("HiGHS dual ray contains a nonfinite value")
        if abs(value) <= threshold:
            continue
        orientation = 1 if convention * value > 0 else -1
        row = lp.rows[index]
        if ((orientation == 1 and row.upper is None)
                or (orientation == -1 and row.lower is None)):
            return None
        result.append((CanonicalInequalityRef("row", index, orientation),
                       abs(value)))
    return result


def _support_stationarity_matrix(lp: ExactCanonicalLP, support,
                                 include_bounded=False):
    bounded = [lo is not None or hi is not None for lo, hi in zip(
        lp.column_lower, lp.column_upper)]
    unbounded_columns = [index for index, value in enumerate(bounded)
                         if include_bounded or not value]
    sparse_columns = []
    for reference, _proposal in support:
        indices, coefficients, _rhs = _inequality(lp, reference)
        sparse_columns.append(dict(zip(indices, coefficients)))
    rows = []
    for column in unbounded_columns:
        values = [values.get(column, Fraction(0))
                  for values in sparse_columns]
        if any(values):
            rows.append(values)
    rows.append([Fraction(1) for _ in support])
    rhs = [Fraction(0) for _ in range(len(rows) - 1)] + [Fraction(1)]
    return rows, rhs


def _exact_vertex_from_support(equations, rhs, timeout_seconds):
    numeric = np.asarray([[float(value) for value in row]
                          for row in equations], dtype=np.float64)
    target = np.asarray([float(value) for value in rhs], dtype=np.float64)
    proposal = linprog(
        np.zeros(numeric.shape[1]), A_eq=numeric, b_eq=target,
        bounds=[(0.0, None)] * numeric.shape[1], method="highs",
        options={"presolve": False})
    if not proposal.success:
        raise ExactSolveFailure("ray support has no numerical nonnegative repair")
    scale = max(1.0, float(np.max(np.abs(proposal.x))))
    positive = np.flatnonzero(proposal.x > scale * 1e-10)
    if not len(positive):
        raise ExactSolveFailure("ray repair returned empty support")
    sub = numeric[:, positive]
    _q, r, row_pivots = qr(sub.T, mode="economic", pivoting=True)
    rank = int(np.linalg.matrix_rank(sub))
    if rank != len(positive):
        raise ExactSolveFailure("ray repair vertex is rank deficient")
    selected_rows = np.asarray(row_pivots[:rank], dtype=np.int64)
    integer_matrix, integer_rhs = [], []
    for row in selected_rows:
        integers = _integerize(
            [*(equations[int(row)][int(column)] for column in positive),
             rhs[int(row)]])
        integer_matrix.append(integers[:-1])
        integer_rhs.append(integers[-1])
    solved = _bareiss_solve(integer_matrix, integer_rhs, timeout_seconds)
    values = [Fraction(0) for _ in range(numeric.shape[1])]
    for column, value in zip(positive, solved):
        values[int(column)] = value
    if any(value < 0 for value in values):
        raise ExactSolveFailure("exact repaired ray has a negative multiplier")
    for row, expected in zip(equations, rhs):
        if _dot(row, values) != expected:
            raise ExactSolveFailure("exact repaired ray stationarity replay failed")
    return values, {
        "numerical_repair_status": int(proposal.status),
        "input_support_size": numeric.shape[1],
        "repaired_support_size": len(positive),
        "exact_repair_rank": rank,
    }


def _complete_farkas_with_bounds(lp: ExactCanonicalLP, support, multipliers):
    entries, q = [], [Fraction(0) for _ in range(lp.column_count)]
    for (reference, _proposal), multiplier in zip(support, multipliers):
        if not multiplier:
            continue
        indices, coefficients, _rhs = _inequality(lp, reference)
        entries.append({"kind": reference.kind, "index": reference.index,
                        "orientation": reference.orientation,
                        "multiplier": _fs(multiplier)})
        for index, coefficient in zip(indices, coefficients):
            q[index] += multiplier * coefficient
    for index, value in enumerate(q):
        if not value:
            continue
        if value > 0:
            reference = CanonicalInequalityRef("column", index, -1)
            multiplier = value
        else:
            reference = CanonicalInequalityRef("column", index, 1)
            multiplier = -value
        _inequality(lp, reference)  # require the needed finite side
        entries.append({"kind": "column", "index": index,
                        "orientation": reference.orientation,
                        "multiplier": _fs(multiplier)})
    certificate = {
        "schema": "CORET_EXACT_LP_FARKAS_CERTIFICATE_V1",
        "canonical_lp_sha256": lp.identity(), "multipliers": entries,
    }
    replay = verify_exact_lp_farkas(lp, certificate)
    return certificate, replay


def repair_direct_dual_ray(lp: ExactCanonicalLP, raw_ray,
                           timeout_seconds=60.0,
                           denominator_caps=(2**12, 2**16, 2**20, 2**24,
                                             2**28, 2**32)):
    """Use ray support/sign only; reconstruct an exact normalized Farkas ray."""
    raw = np.asarray(raw_ray, dtype=np.float64)
    if raw.shape != (len(lp.rows),):
        raise RuntimeError("HiGHS dual ray row count differs")
    maximum = float(np.max(np.abs(raw))) if raw.size else 0.0
    attempts = []
    thresholds = [0.0] + [maximum / cap for cap in denominator_caps]
    deadline = time.perf_counter() + timeout_seconds
    for convention in (1, -1):
        for ordinal, threshold in enumerate(thresholds):
            if time.perf_counter() >= deadline:
                return None, attempts, "DIRECT_RAY_REPAIR_TIMEOUT"
            support = _oriented_ray_rows(lp, raw, convention, threshold)
            row = {"convention": convention, "threshold": threshold,
                   "ordinal": ordinal,
                   "support_size": None if support is None else len(support)}
            if not support:
                row["result"] = "ORIENTATION_OR_SUPPORT_INVALID"
                attempts.append(row)
                continue
            try:
                equations, rhs = _support_stationarity_matrix(lp, support)
                multipliers, repair = _exact_vertex_from_support(
                    equations, rhs, max(0.1, deadline - time.perf_counter()))
                certificate, replay = _complete_farkas_with_bounds(
                    lp, support, multipliers)
                row.update({"result": "EXACT_FARKAS_VERIFIED", **repair,
                            "exact_lambda_b": replay["exact_lambda_b"]})
                attempts.append(row)
                return certificate, attempts, "EXACT_FARKAS_VERIFIED"
            except (RuntimeError, ExactSolveFailure) as error:
                row.update({"result": "EXACT_REPAIR_FAILED",
                            "failure": f"{type(error).__name__}: {error}"})
                attempts.append(row)
    return None, attempts, "DIRECT_RAY_SUPPORT_REPAIR_FAILED"


@dataclass(frozen=True)
class ExactPerspectiveProblem:
    """Small exact instance used for witness replay and unit tests.

    Production data stay content-addressed and are evaluated lazily; material-
    izing 14,000x128 Fraction objects would be needless and memory-hostile.
    """

    x0: tuple[Fraction, ...]
    X: tuple[tuple[Fraction, ...], ...]
    low: tuple[Fraction, ...]
    high: tuple[Fraction, ...]
    gamma: tuple[Fraction, ...]
    beta: tuple[Fraction, ...]
    epsilon: Fraction
    W1: tuple[tuple[Fraction, ...], ...]
    b1: tuple[Fraction, ...]
    W2: tuple[tuple[Fraction, ...], ...]
    b2: tuple[Fraction, ...]

    @property
    def d(self):
        return len(self.x0)


def _start_reconstruction_alarm(seconds):
    """Bound a CPU family, including exact matrix construction and replay."""
    previous_handler = signal.getsignal(signal.SIGALRM)
    def expired(_signum, _frame):
        raise ExactSolveFailure("reconstruction family deadline")
    signal.signal(signal.SIGALRM, expired)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    started = time.perf_counter()
    def restore():
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0]:
            signal.setitimer(signal.ITIMER_REAL, max(
                1e-6, previous_timer[0] - (time.perf_counter() - started)),
                previous_timer[1])
    return restore


def replay_exact_perspective_witness(problem: ExactPerspectiveProblem,
                                     witness: dict) -> dict:
    """Replay a rational exact witness. Algebraic witnesses use the same
    equations after their degree<=2 root has been independently isolated;
    this bounded implementation accepts a rational representative only.
    """
    if witness.get("schema") != WITNESS_SCHEMA:
        raise RuntimeError("perspective witness schema differs")
    xi = tuple(_fr(item) for item in witness.get("source_values", []))
    if len(xi) != len(problem.X):
        raise RuntimeError("perspective witness source count differs")
    if any(value < lo or value > hi for value, lo, hi in
           zip(xi, problem.low, problem.high)):
        raise RuntimeError("perspective witness violates the source box")
    t = _fr(witness["t"])
    if t <= 0:
        raise RuntimeError("perspective witness scale is not positive")
    x = [problem.x0[j] + sum(
        (problem.X[i][j] * xi[i] for i in range(len(xi))), Fraction(0))
         for j in range(problem.d)]
    mean = sum(x, Fraction(0)) / problem.d
    c = [value - mean for value in x]
    quadratic = problem.d * t * t - sum(
        (value * value for value in c), Fraction(0)) \
        - problem.d * problem.epsilon
    if quadratic:
        raise RuntimeError("LayerNorm quadratic equality differs")
    scaled = [problem.gamma[j] * c[j] for j in range(problem.d)]
    w1_scaled = _matvec(problem.W1, scaled)
    w1_beta = _matvec(problem.W1, problem.beta)
    g = [w1_scaled[i] + t * (w1_beta[i] + problem.b1[i])
         for i in range(len(problem.W1))]
    pattern = tuple(witness.get("relu_active", []))
    if len(pattern) != len(g) or any(type(value) is not bool for value in pattern):
        raise RuntimeError("perspective witness ReLU pattern differs")
    u = []
    for active, value in zip(pattern, g):
        if active:
            if value < 0:
                raise RuntimeError("perspective witness active ReLU sign differs")
            u.append(value)
        else:
            if value > 0:
                raise RuntimeError("perspective witness inactive ReLU sign differs")
            u.append(Fraction(0))
    residual = [scaled[j] + _dot(problem.W2[j], u)
                + t * (problem.beta[j] + problem.b2[j])
                for j in range(problem.d)]
    differences = [value - residual[0] for value in residual[1:]]
    if any(differences):
        raise RuntimeError("perspective witness cancellation equality differs")
    return {
        "verified": True, "source_box_check": True,
        "layernorm_equality_check": True, "relu_sign_check": True,
        "cancellation_check": True, "maximum_exact_residual": "0",
        "shared_source_correlation_preserved": True,
    }


def verify_quadratic_isolating_interval(coefficients, interval) -> dict:
    """Verify an isolating interval for a real root of degree one or two."""
    values = [_fr(item) for item in coefficients]
    while values and values[0] == 0:
        values.pop(0)
    if not 2 <= len(values) <= 3:
        raise RuntimeError("root polynomial degree is not one or two")
    lo, hi = map(_fr, interval)
    if lo >= hi:
        raise RuntimeError("root isolating interval is empty")

    def evaluate(x):
        result = Fraction(0)
        for coefficient in values:
            result = result * x + coefficient
        return result

    left, right = evaluate(lo), evaluate(hi)
    if left == 0 or right == 0 or left * right >= 0:
        raise RuntimeError("quadratic interval does not strictly isolate a root")
    if len(values) == 3:
        a, b, c = values
        if b * b - 4 * a * c <= 0:
            raise RuntimeError("quadratic polynomial has no distinct real roots")
        vertex = -b / (2 * a)
        # A sign-changing interval containing the vertex could contain both
        # roots only if an endpoint itself crosses twice, which cannot yield
        # opposite signs.  Record the derivative test explicitly nonetheless.
        derivative_lo = 2 * a * lo + b
        derivative_hi = 2 * a * hi + b
        if derivative_lo == 0 or derivative_hi == 0:
            raise RuntimeError("root interval endpoint is a stationary point")
    return {"verified": True, "degree": len(values) - 1,
            "interval": [_fs(lo), _fs(hi)],
            "endpoint_signs": [-1 if left < 0 else 1,
                               -1 if right < 0 else 1]}


@dataclass(frozen=True)
class QuadraticRootContext:
    """Exact Q(lambda) arithmetic for an isolated degree-two real root."""

    a: Fraction
    b: Fraction
    c: Fraction
    lower: Fraction
    upper: Fraction

    @classmethod
    def from_record(cls, record: dict):
        coefficients = tuple(_fr(item) for item in record["polynomial"])
        if len(coefficients) != 3 or coefficients[0] == 0:
            raise RuntimeError("algebraic witness polynomial is not quadratic")
        interval = tuple(_fr(item) for item in record["isolating_interval"])
        verify_quadratic_isolating_interval(coefficients, interval)
        a, b, c = coefficients
        discriminant = b * b - 4 * a * c
        # A square discriminant would make lambda rational; require the simpler
        # rational witness encoding rather than a non-minimal field extension.
        numerator_root = math.isqrt(discriminant.numerator)
        denominator_root = math.isqrt(discriminant.denominator)
        if (numerator_root * numerator_root == discriminant.numerator
                and denominator_root * denominator_root ==
                discriminant.denominator):
            raise RuntimeError("quadratic witness root is rational/non-minimal")
        return cls(a, b, c, interval[0], interval[1])

    def polynomial(self, value: Fraction) -> Fraction:
        return self.a * value * value + self.b * value + self.c

    def refine(self, lower: Fraction, upper: Fraction):
        left, right = self.polynomial(lower), self.polynomial(upper)
        if left == 0 or right == 0 or left * right >= 0:
            raise RuntimeError("lost exact quadratic root isolation")
        middle = (lower + upper) / 2
        value = self.polynomial(middle)
        if value == 0:
            raise RuntimeError("isolated algebraic root unexpectedly rational")
        return ((lower, middle) if left * value < 0 else (middle, upper))


@dataclass(frozen=True)
class QuadraticElement:
    context: QuadraticRootContext
    constant: Fraction = Fraction(0)
    slope: Fraction = Fraction(0)

    def _coerce(self, other):
        if isinstance(other, QuadraticElement):
            if other.context != self.context:
                raise RuntimeError("algebraic witness root contexts differ")
            return other
        return QuadraticElement(self.context, _fr(other), Fraction(0))

    def __add__(self, other):
        other = self._coerce(other)
        return QuadraticElement(self.context,
                                self.constant + other.constant,
                                self.slope + other.slope)

    __radd__ = __add__

    def __neg__(self):
        return QuadraticElement(self.context, -self.constant, -self.slope)

    def __sub__(self, other):
        return self + (-self._coerce(other))

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __mul__(self, other):
        other = self._coerce(other)
        product = self.slope * other.slope
        return QuadraticElement(
            self.context,
            self.constant * other.constant
            - product * self.context.c / self.context.a,
            self.constant * other.slope + self.slope * other.constant
            - product * self.context.b / self.context.a)

    __rmul__ = __mul__

    def is_zero(self):
        return self.constant == 0 and self.slope == 0

    def sign(self):
        if self.is_zero():
            return 0
        lo, hi = self.context.lower, self.context.upper
        for _ in range(1024):
            first = self.constant + self.slope * lo
            second = self.constant + self.slope * hi
            if min(first, second) > 0:
                return 1
            if max(first, second) < 0:
                return -1
            lo, hi = self.context.refine(lo, hi)
        raise RuntimeError("could not determine exact algebraic sign")


def replay_algebraic_perspective_witness(problem: ExactPerspectiveProblem,
                                         witness: dict) -> dict:
    """Exact replay in Q(lambda), lambda an isolated quadratic real root."""
    if witness.get("schema") != WITNESS_SCHEMA:
        raise RuntimeError("perspective witness schema differs")
    context = QuadraticRootContext.from_record(witness["algebraic_root"])

    def element(record):
        return QuadraticElement(context, _fr(record["constant"]),
                                _fr(record["slope"]))

    xi = tuple(element(item) for item in witness.get("source_affine", []))
    if len(xi) != len(problem.X):
        raise RuntimeError("perspective witness source count differs")
    for value, lo, hi in zip(xi, problem.low, problem.high):
        if (value - lo).sign() < 0 or (hi - value).sign() < 0:
            raise RuntimeError("perspective witness violates the source box")
    t = element(witness["t_affine"])
    if t.sign() <= 0:
        raise RuntimeError("perspective witness scale is not positive")
    x = [sum((problem.X[i][j] * xi[i] for i in range(len(xi))),
             QuadraticElement(context, problem.x0[j]))
         for j in range(problem.d)]
    mean = sum(x, QuadraticElement(context)) * Fraction(1, problem.d)
    c = [value - mean for value in x]
    quadratic = problem.d * t * t - sum(
        (value * value for value in c), QuadraticElement(context)) \
        - problem.d * problem.epsilon
    if not quadratic.is_zero():
        raise RuntimeError("LayerNorm quadratic equality differs")
    scaled = [problem.gamma[j] * c[j] for j in range(problem.d)]
    w1_scaled = [sum((weight * value for weight, value in zip(row, scaled)),
                     QuadraticElement(context)) for row in problem.W1]
    w1_beta = _matvec(problem.W1, problem.beta)
    g = [w1_scaled[i] + t * (w1_beta[i] + problem.b1[i])
         for i in range(len(problem.W1))]
    pattern = tuple(witness.get("relu_active", []))
    if len(pattern) != len(g):
        raise RuntimeError("perspective witness ReLU pattern differs")
    u = []
    for active, value in zip(pattern, g):
        sign = value.sign()
        if active and sign < 0:
            raise RuntimeError("perspective witness active ReLU sign differs")
        if not active and sign > 0:
            raise RuntimeError("perspective witness inactive ReLU sign differs")
        u.append(value if active else QuadraticElement(context))
    residual = [scaled[j] + sum(
        (problem.W2[j][i] * u[i] for i in range(len(u))),
        QuadraticElement(context)) + t * (problem.beta[j] + problem.b2[j])
        for j in range(problem.d)]
    if any(not (value - residual[0]).is_zero() for value in residual[1:]):
        raise RuntimeError("perspective witness cancellation equality differs")
    return {"verified": True, "source_box_check": True,
            "layernorm_equality_check": True, "relu_sign_check": True,
            "cancellation_check": True, "maximum_exact_residual": "0",
            "polynomial_degree": 2,
            "isolating_interval": [_fs(context.lower), _fs(context.upper)],
            "shared_source_correlation_preserved": True}


def verify_branch_tree(tree: dict, certificate_checker) -> dict:
    """Verify binary ReLU phase coverage and every exclusion leaf."""
    if tree.get("schema") != TREE_SCHEMA:
        raise RuntimeError("branch proof-tree schema differs")
    nodes = tree.get("nodes")
    if not isinstance(nodes, dict) or tree.get("root") not in nodes:
        raise RuntimeError("branch proof-tree root is absent")
    visited, leaves, open_leaves, closed = set(), 0, 0, 0

    def visit(node_id, inherited):
        nonlocal leaves, open_leaves, closed
        if node_id in visited:
            raise RuntimeError("branch proof-tree contains a cycle/alias")
        visited.add(node_id)
        node = nodes[node_id]
        phases = {int(key): bool(value)
                  for key, value in (node.get("phases") or {}).items()}
        if any(phases.get(key) != value for key, value in inherited.items()):
            raise RuntimeError("branch child dropped an inherited phase")
        children = node.get("children")
        if children is None:
            leaves += 1
            certificate = node.get("certificate")
            if certificate is None:
                open_leaves += 1
            else:
                certificate_checker(certificate)
                closed += 1
            return
        neuron = int(node["branch_neuron"])
        if (set(children) != {"inactive", "active"}
                or neuron in phases):
            raise RuntimeError("branch does not cover both fresh ReLU phases")
        for label, active in (("inactive", False), ("active", True)):
            child = nodes.get(children[label])
            if child is None:
                raise RuntimeError("branch child is absent")
            child_phases = {int(key): bool(value)
                            for key, value in child.get("phases", {}).items()}
            if child_phases.get(neuron) is not active:
                raise RuntimeError("branch child phase differs")
            visit(children[label], {**phases, neuron: active})

    visit(tree["root"], {})
    if visited != set(nodes):
        raise RuntimeError("branch proof-tree has unreachable nodes")
    return {"verified": True, "nodes": len(visited), "leaves": leaves,
            "closed_leaves": closed, "open_leaves": open_leaves,
            "permits_excluded": open_leaves == 0}


def scientific_status_from_proof(*, exact_witness_verified=False,
                                 root_certificate_verified=False,
                                 complete_tree_verified=False,
                                 open_nodes=0):
    if exact_witness_verified:
        return FEASIBLE
    if root_certificate_verified:
        return EXCLUDED
    if complete_tree_verified and open_nodes == 0:
        return EXCLUDED
    return INCONCLUSIVE


def _authenticate_downstream_token(path: Path) -> dict:
    report = cluster_common.verified_json(path)
    witness = report.get("exact_relu_witness") or {}
    if (report.get("schema") != "CORET_BLOCK2_RELU_CAUSAL_ORACLE_V1"
            or report.get("property_id") != PROPERTY_ID
            or float(report.get("tested_radius", float("nan"))) != RADIUS
            or report.get("final_status") != "EXACT_RELU_CANCELLATION_FEASIBLE"
            or witness.get("verified") is not True
            or witness.get("source_box_check") is not True
            or witness.get("activation_sign_check") is not True
            or witness.get("maximum_exact_residual") not in (0, "0")):
        raise RuntimeError("downstream exact cancellation result differs")
    token = report.get("analysis_token_index")
    if type(token) is not int or token != EXPECTED_TOKEN:
        raise RuntimeError("authenticated downstream analysis token differs")
    witness_path = witness.get("witness_path")
    witness_sha = witness.get("witness_sha256")
    if not isinstance(witness_path, str) or not isinstance(witness_sha, str):
        raise RuntimeError("downstream exact witness identity is absent")
    resolved = Path(witness_path).expanduser()
    if not resolved.is_absolute():
        resolved = (path.parent / resolved).resolve()
    if not resolved.is_file() or cluster_common.sha256(resolved) != witness_sha:
        raise RuntimeError("downstream exact witness artifact differs")
    return {"analysis_token_index": token,
            "report_path": str(path),
            "report_sha256": cluster_common.sha256(path),
            "witness_path": str(resolved), "witness_sha256": witness_sha,
            "authenticated": True}


def _load_authenticated_capture(capture_root: Path) -> tuple[dict, dict]:
    manifest_path = capture_root / "pre_layernorm_input_manifest.json"
    manifest = cluster_common.verified_json(manifest_path)
    artifact_path = (manifest_path.parent
                     / manifest["tensor_artifact_path"]).resolve()
    if cluster_common.sha256(artifact_path) != manifest["state"][
            "tensor_artifact_sha256"]:
        raise RuntimeError("pre-LayerNorm artifact SHA differs")
    payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    snapshot = (payload.get("states") or {}).get("pre_layernorm_input")
    if not isinstance(snapshot, dict):
        raise RuntimeError("pre-LayerNorm input state is absent")
    canonical = runner._snapshot_state_identity(snapshot)
    if canonical.get("canonical_state_identity_sha256") != \
            EXPECTED_CANONICAL_IDENTITY:
        raise RuntimeError("pre-LayerNorm canonical state identity differs")
    verified = runner._verify_pre_layernorm_input_capture(
        manifest_path, allow_legacy_missing_canonical=True,
        expected_canonical_identity=canonical)
    weights = snapshot["weights"].detach().cpu().contiguous()
    low = snapshot["range_low"].detach().cpu().contiguous()
    high = snapshot["range_high"].detach().cpu().contiguous()
    proof = snapshot["proof"]
    if (weights.dtype != torch.float64
            or tuple(weights.shape) != (EXPECTED_SOURCES + 1, 27, DIMENSION)
            or tuple(low.shape) != (EXPECTED_SOURCES,)
            or tuple(high.shape) != (EXPECTED_SOURCES,)
            or len(proof["ids"]) != EXPECTED_SOURCES
            or len(proof["masks"]) != EXPECTED_SOURCES
            or len(proof["reasons"]) != EXPECTED_SOURCES
            or len(set(proof["ids"])) != EXPECTED_SOURCES
            or not bool(torch.isfinite(weights).all())
            or not bool(torch.isfinite(low).all())
            or not bool(torch.isfinite(high).all())
            or bool((low > high).any())):
        raise RuntimeError("pre-LayerNorm state topology/ranges differ")
    model = manifest.get("model_authentication") or {}
    identity = {
        "property_id": manifest.get("property_id"),
        "tested_radius": manifest.get("tested_radius"),
        "tested_radius_hex": manifest.get("tested_radius_hex"),
        "pinned_deept_revision": model.get("pinned_revision"),
        "scientific_manifest_sha256": model.get("scientific_manifest_sha256"),
        "production_manifest_sha256": model.get("production_manifest_sha256"),
        "capture_manifest_path": str(manifest_path.resolve()),
        "capture_manifest_sha256": cluster_common.sha256(manifest_path),
        "artifact_path": str(artifact_path),
        "artifact_sha256": cluster_common.sha256(artifact_path),
        "canonical_state_identity": canonical,
        "strict_capture_verification": verified,
    }
    if (identity["property_id"] != PROPERTY_ID
            or float(identity["tested_radius"]) != RADIUS
            or identity["tested_radius_hex"] != RADIUS_HEX):
        raise RuntimeError("pre-LayerNorm scientific instance differs")
    return snapshot, identity


def _load_parameters(capture_identity: dict):
    gamma, beta, ln_identity = image._load_layernorm_parameters(capture_identity)
    W1, b1, w1_identity = prefrontier._load_ffn_first_parameters(capture_identity)
    W2, b2, w2_identity = frontier._load_ffn_second_parameters(capture_identity)
    source = frontier.DEFAULT_PARAMETER_SOURCE
    epsilon_path = (
        "Robustness-Verification-for-Transformers/Verifiers/Zonotope.py")
    epsilon_raw = frontier._git_blob_at_revision(
        source["pinned_revision"], epsilon_path)
    syntax = ast.parse(epsilon_raw.decode("utf-8"), filename=epsilon_path)
    assignments = [node for node in syntax.body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name)
                           and target.id == "epsilon" for target in node.targets)]
    if (len(assignments) != 1
            or not isinstance(assignments[0].value, ast.Constant)
            or not isinstance(assignments[0].value.value, (int, float))):
        raise RuntimeError("pinned native LayerNorm epsilon is ambiguous")
    epsilon = _fr(assignments[0].value.value)
    if epsilon <= 0:
        raise RuntimeError("authenticated LayerNorm epsilon is not positive")
    return (gamma, beta, W1, b1, W2, b2, epsilon,
            {"layernorm": ln_identity, "ffn_first": w1_identity,
             "ffn_second": w2_identity,
             "epsilon": _fs(epsilon),
             "epsilon_source": (
                 f"git:{source['pinned_revision']}:{epsilon_path}"),
             "epsilon_source_sha256": hashlib.sha256(epsilon_raw).hexdigest(),
             "config_sha256": source["config_sha256"]})


def _centered_coordinate_bounds(center, generators, low, high):
    """Exact dyadic bounds for P(x0+X*xi), retaining shared xi per row."""
    d = center.size
    center_exact = [_fr(value) for value in center]
    centered_center = [value - sum(center_exact, Fraction(0)) / d
                       for value in center_exact]
    lower = list(centered_center)
    upper = list(centered_center)
    coefficients_by_coordinate = [[] for _ in range(d)]
    for row, lo_raw, hi_raw in zip(generators, low, high):
        values = [_fr(value) for value in row]
        mean = sum(values, Fraction(0)) / d
        lo, hi = _fr(lo_raw), _fr(hi_raw)
        for coordinate, value in enumerate(values):
            coefficient = value - mean
            coefficients_by_coordinate[coordinate].append(coefficient)
            first, second = coefficient * lo, coefficient * hi
            lower[coordinate] += min(first, second)
            upper[coordinate] += max(first, second)
    return centered_center, lower, upper, coefficients_by_coordinate


def _linear_interval(weights, lower, upper, bias=Fraction(0)):
    lo = hi = _fr(bias)
    for weight, left, right in zip(weights, lower, upper):
        weight = _fr(weight)
        first, second = weight * left, weight * right
        lo += min(first, second)
        hi += max(first, second)
    return lo, hi


def exact_dyadic_sqrt_lower(epsilon, bits=64):
    """Certify a positive implied scale bound using integers only."""
    epsilon = _fr(epsilon)
    if epsilon <= 0 or bits <= 0:
        raise RuntimeError("positive epsilon and denominator bits required")
    m = math.isqrt((epsilon.numerator << (2 * bits)) // epsilon.denominator)
    lower = Fraction(m, 1 << bits)
    if lower <= 0 or lower * lower > epsilon:
        raise RuntimeError("dyadic scale lower bound certification failed")
    return lower, {
        "epsilon_exact": _fs(epsilon),
        "t_lower_bound_exact": _fs(lower),
        "t_lower_bound_float": float(lower),
        "t_lower_bound_denominator_bits": bits,
        "proof_check_L_squared_le_epsilon": True,
    }


def _derive_exact_bounds(center, generators, low, high, gamma, beta,
                         W1, b1, epsilon):
    started = time.perf_counter()
    c0, c_lower, c_upper, c_coefficients = _centered_coordinate_bounds(
        center, generators, low, high)
    max_norm_squared = sum((max(abs(lo), abs(hi)) ** 2
                            for lo, hi in zip(c_lower, c_upper)), Fraction(0))
    upper = _exact_sqrt_upper(max_norm_squared / DIMENSION + epsilon)
    lower, lower_proof = exact_dyadic_sqrt_lower(epsilon)
    gamma_exact = [_fr(value) for value in gamma]
    beta_exact = [_fr(value) for value in beta]
    scaled_lower, scaled_upper = [], []
    for gamma_value, lo, hi in zip(gamma_exact, c_lower, c_upper):
        first, second = gamma_value * lo, gamma_value * hi
        scaled_lower.append(min(first, second))
        scaled_upper.append(max(first, second))
    w1_beta = []
    g_lower, g_upper = [], []
    for row, bias in zip(W1, b1):
        exact_row = [_fr(value) for value in row]
        affine_lo, affine_hi = _linear_interval(
            exact_row, scaled_lower, scaled_upper)
        q = _dot(exact_row, beta_exact) + _fr(bias)
        t_first, t_second = lower * q, upper * q
        g_lower.append(affine_lo + min(t_first, t_second))
        g_upper.append(affine_hi + max(t_first, t_second))
        w1_beta.append(q)
    active = [index for index, lo in enumerate(g_lower) if lo >= 0]
    inactive = [index for index, hi in enumerate(g_upper) if hi <= 0]
    unstable = [index for index in range(len(g_lower))
                if index not in set(active) and index not in set(inactive)]
    bounds_payload = [[_fs(lo), _fs(hi)]
                      for lo, hi in zip(g_lower, g_upper)]
    return {
        "centered_center": c0,
        "centered_coefficients": c_coefficients,
        "centered_lower": c_lower, "centered_upper": c_upper,
        "max_norm_squared": max_norm_squared, "t_upper": upper,
        "t_lower": lower, "t_lower_proof": lower_proof,
        "g_lower": g_lower, "g_upper": g_upper,
        "stable_active": active, "stable_inactive": inactive,
        "unstable": unstable, "bounds_sha256": _sha_json(bounds_payload),
        "runtime_seconds": time.perf_counter() - started,
    }


def _backend_inventory() -> dict:
    packages = {}
    for name in ("highspy", "scipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    selected = "HIGHSPY_DIRECT_DUAL_RAY" if packages["highspy"] else \
        "SCIPY_HIGHS_PHASE_I"
    return {
        "packages": packages,
        "selected": selected,
        "certificate_extraction_supported": bool(packages["highspy"]),
        "priority": ["highspy.getDualRay", "scipy_highs_phase_i"],
        "gurobi_permitted": False,
    }


def _coalesced(items):
    values = {}
    for index, coefficient in items:
        values[int(index)] = values.get(int(index), Fraction(0)) + _fr(coefficient)
    pairs = [(index, value) for index, value in sorted(values.items()) if value]
    return tuple(index for index, _ in pairs), tuple(value for _, value in pairs)


def build_exact_perspective_lp(low, high, gamma, beta, W1, b1, W2, b2,
                               bounds: dict) -> ExactCanonicalLP:
    n, d = len(low), DIMENSION
    xi0 = 0
    c0 = n
    t_index = c0 + d
    g0 = t_index + 1
    u0 = g0 + d
    names = ([f"xi[{index}]" for index in range(n)]
             + [f"c[{index}]" for index in range(d)] + ["t"]
             + [f"g[{index}]" for index in range(d)]
             + [f"u[{index}]" for index in range(d)])
    column_lower = [_fr(value) for value in low] + [None] * d + [bounds["t_lower"]] \
        + [None] * (2 * d)
    column_upper = [_fr(value) for value in high] + [None] * d \
        + [bounds["t_upper"]] + [None] * (2 * d)
    rows = []
    source_indices = tuple(range(n))
    for coordinate in range(d):
        coefficients = bounds["centered_coefficients"][coordinate]
        indices, values = _coalesced([
            *((source_indices[index], -value)
              for index, value in enumerate(coefficients)),
            (c0 + coordinate, Fraction(1)),
        ])
        rhs = bounds["centered_center"][coordinate]
        rows.append(ExactLPRow(f"centered[{coordinate}]", indices, values,
                               rhs, rhs))
    gamma_exact = [_fr(value) for value in gamma]
    beta_exact = [_fr(value) for value in beta]
    b1_exact = [_fr(value) for value in b1]
    b2_exact = [_fr(value) for value in b2]
    W1_exact = _fraction_matrix(W1)
    W2_exact = _fraction_matrix(W2)
    w1_beta = _matvec(W1_exact, beta_exact)
    for output in range(d):
        items = [(c0 + feature,
                  -W1_exact[output][feature] * gamma_exact[feature])
                 for feature in range(d)]
        items.extend(((t_index, -(w1_beta[output] + b1_exact[output])),
                      (g0 + output, Fraction(1))))
        indices, values = _coalesced(items)
        rows.append(ExactLPRow(f"preactivation[{output}]", indices, values,
                               Fraction(0), Fraction(0)))
    for coordinate in range(1, d):
        items = [(c0, -gamma_exact[0]),
                 (c0 + coordinate, gamma_exact[coordinate]),
                 (t_index, beta_exact[coordinate] + b2_exact[coordinate]
                  - beta_exact[0] - b2_exact[0])]
        items.extend((u0 + feature,
                      W2_exact[coordinate][feature] - W2_exact[0][feature])
                     for feature in range(d))
        indices, values = _coalesced(items)
        rows.append(ExactLPRow(f"cancellation[{coordinate}]", indices, values,
                               Fraction(0), Fraction(0)))
    active = set(bounds["stable_active"])
    inactive = set(bounds["stable_inactive"])
    for neuron in range(d):
        if neuron in active:
            indices, values = _coalesced(
                [(g0 + neuron, -1), (u0 + neuron, 1)])
            rows.append(ExactLPRow(f"relu_active_value[{neuron}]",
                                   indices, values, Fraction(0), Fraction(0)))
            rows.append(ExactLPRow(f"relu_active_sign[{neuron}]",
                                   (g0 + neuron,), (Fraction(-1),), None,
                                   Fraction(0)))
        elif neuron in inactive:
            rows.append(ExactLPRow(f"relu_inactive_value[{neuron}]",
                                   (u0 + neuron,), (Fraction(1),),
                                   Fraction(0), Fraction(0)))
            rows.append(ExactLPRow(f"relu_inactive_sign[{neuron}]",
                                   (g0 + neuron,), (Fraction(1),), None,
                                   Fraction(0)))
        else:
            lower, upper = bounds["g_lower"][neuron], bounds["g_upper"][neuron]
            if not lower < 0 < upper:
                raise RuntimeError("unstable ReLU exact bounds differ")
            slope = upper / (upper - lower)
            rows.append(ExactLPRow(f"relu_triangle_nonnegative[{neuron}]",
                                   (u0 + neuron,), (Fraction(-1),), None,
                                   Fraction(0)))
            indices, values = _coalesced(
                [(g0 + neuron, 1), (u0 + neuron, -1)])
            rows.append(ExactLPRow(f"relu_triangle_above_input[{neuron}]",
                                   indices, values, None, Fraction(0)))
            indices, values = _coalesced(
                [(g0 + neuron, -slope), (u0 + neuron, 1)])
            rows.append(ExactLPRow(f"relu_triangle_upper[{neuron}]",
                                   indices, values, None, -slope * lower))
    return ExactCanonicalLP(tuple(names), tuple(column_lower),
                            tuple(column_upper), tuple(rows))


def persist_exact_lp(lp: ExactCanonicalLP, path: Path):
    """Deterministic gzip JSONL serialization of every exact coefficient."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            header = {"schema": "CORET_EXACT_CANONICAL_LP_V1",
                      "canonical_lp_sha256": lp.identity(),
                      "column_count": lp.column_count,
                      "row_count": len(lp.rows), "nnz": lp.nnz,
                      "variables": [{"name": name,
                                     "lower": _optional_fs(lo),
                                     "upper": _optional_fs(hi)}
                                    for name, lo, hi in zip(
                                        lp.variable_names, lp.column_lower,
                                        lp.column_upper)]}
            stream.write((json.dumps(header, sort_keys=True,
                                     separators=(",", ":")) + "\n").encode())
            for index, row in enumerate(lp.rows):
                record = {"row": index, "name": row.name,
                          "lower": _optional_fs(row.lower),
                          "upper": _optional_fs(row.upper),
                          "entries": [[column, _fs(value)] for column, value
                                      in zip(row.indices, row.coefficients)]}
                stream.write((json.dumps(record, sort_keys=True,
                                         separators=(",", ":")) + "\n").encode())
    os.replace(temporary, path)
    return {"path": str(path), "sha256": cluster_common.sha256(path),
            "canonical_lp_sha256": lp.identity()}


def solve_highspy(lp: ExactCanonicalLP, log_path: Path | None = None,
                  scaling_path: Path | None = None):
    import highspy
    highs = highspy.Highs()
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    options = {"output_flag": log_path is not None,
               "log_to_console": False,
               "presolve": "off", "threads": 1,
               "parallel": "off", "random_seed": 0,
               "solver": "simplex"}
    if log_path is not None:
        options["log_file"] = str(log_path)
    option_statuses = []
    for name, value in options.items():
        status = highs.setOptionValue(name, value)
        option_statuses.append({"option": name, "value": value,
                                "status": str(status)})
        if status != highspy.HighsStatus.kOk:
            diagnostic = {
                "failure_classification": "UNKNOWN_HIGHS_MODEL_REJECTION",
                "failed_api_call": f"setOptionValue({name})",
                "failed_highs_status": str(status),
                "option_statuses": option_statuses,
                "column_count": lp.column_count,
                "row_count": len(lp.rows), "nnz": lp.nnz,
                "highs_log_path": None if log_path is None else str(log_path),
            }
            raise HighsCanonicalLPDiagnosticError(
                f"HiGHS rejected deterministic option {name}", diagnostic)
    threshold_statuses = {}
    thresholds = {}
    for name in ("small_matrix_value", "large_matrix_value", "infinite_bound"):
        status, value = highs.getOptionValue(name)
        threshold_statuses[name] = str(status)
        if status != highspy.HighsStatus.kOk:
            raise HighsCanonicalLPDiagnosticError(
                f"HiGHS rejected threshold query {name}", {
                    "failure_classification":
                        "UNKNOWN_HIGHS_MODEL_REJECTION",
                    "failed_api_call": f"getOptionValue({name})",
                    "failed_highs_status": str(status),
                    "option_statuses": option_statuses,
                    "threshold_option_statuses": threshold_statuses,
                    "column_count": lp.column_count,
                    "row_count": len(lp.rows), "nnz": lp.nnz,
                })
        thresholds[name] = float(value)
    infinity = highspy.kHighsInf
    original_arrays = _highs_numeric_arrays(lp, infinity)
    original_diagnostic = _diagnose_highs_lp(
        lp, original_arrays, thresholds)
    original_structural_failure = original_diagnostic[
        "preflight_failure_classification"] in {
            "NONFINITE_CANONICAL_COEFFICIENT", "NONFINITE_CANONICAL_BOUND",
            "HIGHS_MATRIX_INDEX_INVALID", "HIGHS_MATRIX_STRUCTURE_INVALID",
            "HIGHS_BOUND_MAGNITUDE_REJECTED"}
    if original_structural_failure:
        original_diagnostic.update({
            "failure_classification": original_diagnostic[
                "preflight_failure_classification"],
            "failed_api_call": "original_pre_scaling_validation",
            "failed_highs_status": None,
        })
        raise HighsCanonicalLPDiagnosticError(
            "original canonical LP failed structural validation",
            original_diagnostic)
    scaling = build_highs_row_scaled_lp(
        lp, small_matrix_value=thresholds["small_matrix_value"],
        large_matrix_value=thresholds["large_matrix_value"],
        infinite_bound=thresholds["infinite_bound"])
    if scaling_path is not None:
        persist_solver_row_scaling(scaling, scaling_path)
    solver_lp = scaling.lp
    solver_arrays = _highs_numeric_arrays(solver_lp, infinity)
    scaled_diagnostic = _diagnose_highs_lp(
        solver_lp, solver_arrays, thresholds)
    _require_solver_scaling_applied(
        lp, solver_lp, scaling, original_diagnostic,
        scaled_diagnostic, scaling_path)
    model = highspy.HighsLp()
    model.num_col_ = solver_lp.column_count
    model.num_row_ = len(solver_lp.rows)
    model.col_cost_ = solver_arrays["objective"]
    model.col_lower_ = solver_arrays["column_lower"]
    model.col_upper_ = solver_arrays["column_upper"]
    model.row_lower_ = solver_arrays["row_lower"]
    model.row_upper_ = solver_arrays["row_upper"]
    starts, indices, values = (solver_arrays["starts"],
                               solver_arrays["indices"],
                               solver_arrays["values"])
    diagnostic = dict(scaled_diagnostic)
    diagnostic.update({
        "schema": "CORET_HIGHSPY_CANONICAL_LP_DIAGNOSTIC_V1",
        "canonical_lp_sha256": lp.identity(),
        "original_canonical_lp_sha256": lp.identity(),
        "solver_scaled_lp_sha256": solver_lp.identity(),
        "row_scaling_sha256": scaling.report["row_scaling_sha256"],
        "solver_scaling": scaling.report,
        "original_diagnostic": original_diagnostic,
        "scaled_diagnostic": dict(scaled_diagnostic),
        "highspy_version": importlib.metadata.version("highspy"),
        "highs_version": highs.version(), "options": options,
        "option_statuses": option_statuses,
        "threshold_option_statuses": threshold_statuses,
        "api_statuses": [],
        "highs_log_path": None if log_path is None else str(log_path),
    })
    scaled_failure = diagnostic["preflight_failure_classification"] is not None
    if scaled_failure:
        diagnostic.update({
            "failure_classification": diagnostic[
                "preflight_failure_classification"],
            "failed_api_call": "scaled_pre_passModel_validation",
            "failed_highs_status": None,
        })
        raise HighsCanonicalLPDiagnosticError(
            "scaled LP failed pre-passModel validation", diagnostic)
    model.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    model.a_matrix_.start_ = np.asarray(starts, dtype=np.int64)
    model.a_matrix_.index_ = np.asarray(indices, dtype=np.int32)
    model.a_matrix_.value_ = np.asarray(values, dtype=np.float64)
    pass_status = highs.passModel(model)
    _check_highs_status("passModel", pass_status, highspy.HighsStatus.kOk,
                        diagnostic, log_path)
    started = time.perf_counter()
    run_status = highs.run()
    runtime = time.perf_counter() - started
    _check_highs_status("run", run_status, highspy.HighsStatus.kOk,
                        diagnostic, log_path)
    status = highs.getModelStatus()
    status_name = highs.modelStatusToString(status)
    ray_status, ray_exists = highs.getDualRayExist()
    raw_ray = None
    ray_call_status = ray_status
    if ray_status == highspy.HighsStatus.kOk and ray_exists:
        ray_call_status, returned, values = highs.getDualRay()
        if ray_call_status == highspy.HighsStatus.kOk and returned:
            raw_ray = np.asarray(values, dtype=np.float64)
    original_row_ray = (None if raw_ray is None else
                        map_solver_row_ray_to_original(raw_ray,
                                                       scaling.scales))
    solution = highs.getSolution()
    return {
        "run_status": str(run_status), "model_status": status_name,
        "infeasible": status == highspy.HighsModelStatus.kInfeasible,
        "feasible": status in (highspy.HighsModelStatus.kOptimal,
                               highspy.HighsModelStatus.kObjectiveBound),
        "direct_dual_ray_available": raw_ray is not None,
        "raw_dual_ray": raw_ray,
        "original_row_dual_ray": original_row_ray,
        "column_values": (np.asarray(solution.col_value, dtype=np.float64)
                          if solution.value_valid else None),
        "runtime_seconds": runtime, "highs_runtime_seconds": highs.getRunTime(),
        "highspy_version": importlib.metadata.version("highspy"),
        "highs_version": highs.version(), "options": options,
        "ray_exist_status": str(ray_status),
        "ray_call_status": str(ray_call_status),
        "solver_scaling": scaling.report,
        "construction_diagnostic": diagnostic,
    }


def phase1_exact_farkas_fallback(lp: ExactCanonicalLP,
                                 timeout_seconds=60.0):
    """SciPy/HiGHS L1-violation Phase-I proposal.

    This is invoked only when an infeasible HiGHS model has no direct row ray.
    It minimizes sum(s) subject to C*x-s<=b, s>=0 for the fully canonicalized
    inequalities.  A positive floating optimum is not proof: inequality
    marginals merely select support for exact repair/replay.
    """
    from scipy.sparse import coo_matrix, eye, hstack
    references = canonicalize_all_inequalities(lp)
    row_indices, column_indices, data, rhs = [], [], [], []
    for row, reference in enumerate(references):
        indices, coefficients, bound = _inequality(lp, reference)
        rhs.append(float(bound))
        for index, coefficient in zip(indices, coefficients):
            row_indices.append(row); column_indices.append(index)
            data.append(float(coefficient))
    canonical = coo_matrix(
        (data, (row_indices, column_indices)),
        shape=(len(references), lp.column_count)).tocsr()
    phase_matrix = hstack([
        canonical, -eye(len(references), format="csr")], format="csr")
    objective = np.concatenate([
        np.zeros(lp.column_count), np.ones(len(references))])
    result = linprog(
        objective, A_ub=phase_matrix, b_ub=np.asarray(rhs),
        bounds=[(None, None)] * lp.column_count
               + [(0.0, None)] * len(references), method="highs",
        options={"presolve": False, "time_limit": timeout_seconds})
    record = {"solver": "scipy.optimize.linprog/highs",
              "formulation": "L1 canonical-inequality violation Phase-I",
              "status": int(result.status), "message": result.message,
              "numerical_objective": (float(result.fun)
                                      if result.fun is not None else None)}
    if not result.success or result.fun is None or result.fun <= 0:
        return None, record
    marginals = -np.asarray(result.ineqlin.marginals, dtype=np.float64)
    maximum = max(1.0, float(np.max(np.abs(marginals))))
    selected = np.flatnonzero(marginals > maximum * 1e-10)
    support = [(references[int(index)], float(marginals[int(index)]))
               for index in selected]
    try:
        equations, target = _support_stationarity_matrix(
            lp, support, include_bounded=True)
        multipliers, repair = _exact_vertex_from_support(
            equations, target, timeout_seconds)
        certificate, replay = _complete_farkas_with_bounds(
            lp, support, multipliers)
        record.update({"certificate_verified": True, **repair,
                       "exact_lambda_b": replay["exact_lambda_b"]})
        return certificate, record
    except (RuntimeError, ExactSolveFailure) as error:
        record.update({"certificate_verified": False,
                       "exact_repair_failure":
                           f"{type(error).__name__}: {error}"})
        return None, record


def _verify_cross_authentication(authentication: dict, downstream: dict,
                                 parameters: dict) -> bool:
    """Fail closed on any state/token/parameter identity disagreement."""
    canonical = authentication.get("canonical_state_identity") or {}
    if (authentication.get("property_id") != PROPERTY_ID
            or float(authentication.get("tested_radius", float("nan"))) != RADIUS
            or authentication.get("tested_radius_hex") != RADIUS_HEX
            or canonical.get("canonical_state_identity_sha256") !=
            EXPECTED_CANONICAL_IDENTITY
            or downstream.get("analysis_token_index") != EXPECTED_TOKEN
            or downstream.get("authenticated") is not True):
        raise RuntimeError("cross-artifact state/token identity differs")
    for name in ("layernorm", "ffn_first", "ffn_second"):
        record = parameters.get(name) or {}
        if record.get("parameter_identity_authenticated") is not True:
            raise RuntimeError(f"{name} parameter identity is unauthenticated")
    revisions = {
        parameters[name].get("pinned_revision")
        for name in ("layernorm", "ffn_first", "ffn_second")
        if parameters[name].get("pinned_revision") is not None}
    if revisions and revisions != {authentication.get("pinned_deept_revision")}:
        raise RuntimeError("cross-artifact pinned revision differs")
    return True


def _root_model_summary(source_count: int, bounds: dict) -> dict:
    active = len(bounds["stable_active"])
    inactive = len(bounds["stable_inactive"])
    unstable = len(bounds["unstable"])
    variables = source_count + 3 * DIMENSION + 1
    equalities = 2 * DIMENSION + (DIMENSION - 1) + active + inactive
    inequalities = (2 * source_count + 2 + active + inactive
                    + 3 * unstable)
    formulation = {
        "source_variables": source_count, "centered_variables": DIMENSION,
        "scale_variables": 1, "preactivation_variables": DIMENSION,
        "relu_output_variables": DIMENSION,
        "variable_count": variables,
        "equality_count": equalities,
        "inequality_count": inequalities,
        "cone_dimension": 0,
        "dropped_outer_constraint": (
            "sum(c_j^2) <= 128*t^2 (dropping a constraint remains an outer relaxation)"),
        "proof_backend_form": "rational polyhedral outer relaxation",
        "epsilon_restored_only_in_exact_witness_replay": False,
        "epsilon_used_for_implied_scale_lower_bound": True,
        "source_correlation_preserved": True,
        "cancellation_equations": DIMENSION - 1,
    }
    formulation["canonical_model_sha256"] = _sha_json(formulation)
    return formulation


def _empty_exact_witness_record():
    return {
        "attempted": False, "verified": False,
        "phase_pattern_sha256": None, "bases_attempted": 0,
        "polynomial_degree": None, "polynomial_sha256": None,
        "isolating_interval": None, "source_box_check": None,
        "layernorm_equality_check": None, "relu_sign_check": None,
        "maximum_exact_residual": None,
        "witness_path": None, "witness_sha256": None,
        "representation": (
            "dyadic fixed sources + <=127 Bareiss correction variables + "
            "degree<=2 rational polynomial and rational isolating interval"),
    }


def candidate_relu_phase_pattern(solution, bounds: dict, source_count: int):
    solution = np.asarray(solution, dtype=np.float64)
    expected = source_count + 3 * DIMENSION + 1
    if solution.shape != (expected,) or not np.isfinite(solution).all():
        raise RuntimeError("root LP primal solution topology differs")
    g0 = source_count + DIMENSION + 1
    u0 = g0 + DIMENSION
    g = solution[g0:g0 + DIMENSION]
    u = solution[u0:u0 + DIMENSION]
    pattern = [False] * DIMENSION
    for index in bounds["stable_active"]:
        pattern[index] = True
    for index in bounds["stable_inactive"]:
        pattern[index] = False
    for index in bounds["unstable"]:
        pattern[index] = bool(g[index] >= 0.0)  # exact deterministic zero tie
    violations = {
        int(index): max(0.0, float(u[index] - max(0.0, g[index])))
        for index in bounds["unstable"]}
    return pattern, {
        "phase_pattern_sha256": _sha_json(pattern),
        "stable_active_count": len(bounds["stable_active"]),
        "stable_inactive_count": len(bounds["stable_inactive"]),
        "unstable_count": len(bounds["unstable"]),
        "zero_tie_rule": "g_i >= 0 selects active",
        "relu_hull_violation_scores": {
            str(key): value for key, value in violations.items()},
    }


def lp_with_relu_phases(lp: ExactCanonicalLP, phases: dict[int, bool],
                        source_count: int) -> ExactCanonicalLP:
    g0 = source_count + DIMENSION + 1
    u0 = g0 + DIMENSION
    rows = list(lp.rows)
    for neuron, active in sorted(phases.items()):
        if not 0 <= neuron < DIMENSION or type(active) is not bool:
            raise RuntimeError("branch ReLU phase differs")
        if active:
            indices, values = _coalesced(
                [(g0 + neuron, -1), (u0 + neuron, 1)])
            rows.append(ExactLPRow(
                f"branch_active_value[{neuron}]", indices, values,
                Fraction(0), Fraction(0)))
            rows.append(ExactLPRow(
                f"branch_active_sign[{neuron}]", (g0 + neuron,),
                (Fraction(-1),), None, Fraction(0)))
        else:
            rows.append(ExactLPRow(
                f"branch_inactive_value[{neuron}]", (u0 + neuron,),
                (Fraction(1),), Fraction(0), Fraction(0)))
            rows.append(ExactLPRow(
                f"branch_inactive_sign[{neuron}]", (g0 + neuron,),
                (Fraction(1),), None, Fraction(0)))
    return ExactCanonicalLP(
        lp.variable_names, lp.column_lower, lp.column_upper, tuple(rows))


def _branch_neuron(solution, bounds: dict, phases: dict, source_count: int):
    pattern, proposal = candidate_relu_phase_pattern(
        solution, bounds, source_count)
    scores = proposal["relu_hull_violation_scores"]
    remaining = [int(index) for index in bounds["unstable"]
                 if int(index) not in phases]
    if not remaining:
        return None, pattern, proposal
    # Largest violation first; lower neuron index wins an exact tie.
    selected = min(remaining, key=lambda index: (-scores[str(index)], index))
    return selected, pattern, proposal


def run_phase_continuation(
        root_solution, bounds: dict, source_count: int,
        maximum_nodes: int, maximum_patterns: int, deadline: float,
        solve_node, attempt_witness):
    """Minimal deterministic ReLU-phase BaB with proof-carrying leaves."""
    nodes = {"root": {"phases": {}, "children": None,
                      "certificate": None, "status": "OPEN"}}
    queue = [("root", {}, root_solution, 0)]
    patterns, closed, maximum_depth = 0, 0, 0
    witness = None
    attempts = []
    limit_reason = None
    while queue and witness is None:
        if time.perf_counter() >= deadline:
            limit_reason = "WALL_CLOCK_LIMIT"
            break
        node_id, phases, inherited_solution, depth = queue.pop(0)
        maximum_depth = max(maximum_depth, depth)
        if inherited_solution is None:
            solved = solve_node(phases, node_id)
            nodes[node_id]["solver_status"] = solved.get("solver_status")
            if solved.get("infeasible"):
                certificate = solved.get("certificate")
                if certificate is not None:
                    nodes[node_id].update(
                        {"status": "CLOSED_EXACT_FARKAS",
                         "certificate": certificate})
                    closed += 1
                else:
                    nodes[node_id]["status"] = "OPEN_UNCERTIFIED_INFEASIBLE"
                continue
            if not solved.get("feasible") or solved.get("solution") is None:
                nodes[node_id]["status"] = "OPEN_SOLVER_UNRESOLVED"
                continue
            solution = solved["solution"]
        else:
            solution = inherited_solution
            nodes[node_id]["solver_status"] = "ROOT_OPTIMAL"
        neuron, pattern, proposal = _branch_neuron(
            solution, bounds, phases, source_count)
        nodes[node_id]["phase_proposal"] = proposal
        if patterns < maximum_patterns:
            patterns += 1
            found, evidence = attempt_witness(pattern, solution, node_id)
            attempts.append({"node_id": node_id, **evidence})
            if found is not None:
                witness = found
                nodes[node_id]["status"] = "EXACT_WITNESS_VERIFIED"
                break
        if neuron is None:
            nodes[node_id]["status"] = "OPEN_FIXED_PHASE_WITHOUT_PROOF"
            continue
        if len(nodes) + 2 > maximum_nodes:
            nodes[node_id]["status"] = "OPEN_NODE_LIMIT"
            limit_reason = "MAXIMUM_NODES"
            break
        children = {}
        nodes[node_id].update({"branch_neuron": neuron,
                               "children": children,
                               "status": "BRANCHED"})
        for label, active in (("inactive", False), ("active", True)):
            child_id = f"{node_id}.{label[0]}{neuron}"
            child_phases = {**phases, neuron: active}
            children[label] = child_id
            nodes[child_id] = {"phases": {
                str(key): value for key, value in child_phases.items()},
                "children": None, "certificate": None, "status": "QUEUED"}
            queue.append((child_id, child_phases, None, depth + 1))
    open_nodes = sum(
        node.get("children") is None and node.get("certificate") is None
        and node.get("status") != "EXACT_WITNESS_VERIFIED"
        for node in nodes.values())
    tree = {"schema": TREE_SCHEMA, "root": "root", "nodes": nodes}
    return witness, {
        "attempted": True, "nodes_created": len(nodes),
        "nodes_closed_by_certificate": closed,
        "nodes_open": open_nodes, "maximum_depth": maximum_depth,
        "phase_patterns_attempted": patterns,
        "witness_attempts": attempts, "limit_reason": limit_reason,
        "tree": tree,
    }


def _exact_problem_from_arrays(weights, token, low, high, gamma, beta,
                               W1, b1, W2, b2, epsilon):
    return ExactPerspectiveProblem(
        x0=tuple(_fr(value) for value in weights[0, token]),
        X=tuple(tuple(_fr(value) for value in row)
                for row in weights[1:, token]),
        low=tuple(_fr(value) for value in low),
        high=tuple(_fr(value) for value in high),
        gamma=tuple(_fr(value) for value in gamma),
        beta=tuple(_fr(value) for value in beta),
        epsilon=_fr(epsilon), W1=tuple(tuple(_fr(value) for value in row)
                                      for row in W1),
        b1=tuple(_fr(value) for value in b1),
        W2=tuple(tuple(_fr(value) for value in row) for row in W2),
        b2=tuple(_fr(value) for value in b2))


def attempt_exact_lp_candidate_witness(problem: ExactPerspectiveProblem,
                                       solution, pattern, source_count):
    """Exact replay of the dyadic LP proposal; failure is never exclusion."""
    solution = np.asarray(solution, dtype=np.float64)
    t_index = source_count + DIMENSION
    witness = {
        "schema": WITNESS_SCHEMA,
        "source_values": [_fs(_fr(value))
                          for value in solution[:source_count]],
        "t": _fs(_fr(solution[t_index])),
        "relu_active": [bool(value) for value in pattern],
    }
    try:
        replay = replay_exact_perspective_witness(problem, witness)
        return witness, {"attempted": True, "verified": True,
                         "method": "EXACT_DYADIC_LP_CANDIDATE_REPLAY",
                         **replay}
    except RuntimeError as error:
        return None, {"attempted": True, "verified": False,
                      "method": "EXACT_DYADIC_LP_CANDIDATE_REPLAY",
                      "failure": f"{type(error).__name__}: {error}"}


def _fraction_square_root(value: Fraction):
    if value < 0:
        return None
    numerator, denominator = (math.isqrt(value.numerator),
                              math.isqrt(value.denominator))
    if (numerator * numerator == value.numerator
            and denominator * denominator == value.denominator):
        return Fraction(numerator, denominator)
    return None


def _isolate_quadratic_root(coefficients, approximation: float):
    a, b, c = coefficients

    def evaluate(value):
        return a * value * value + b * value + c

    left_float = math.nextafter(approximation, -math.inf)
    right_float = math.nextafter(approximation, math.inf)
    for _ in range(256):
        left, right = (_fr(left_float), _fr(right_float))
        left_value, right_value = evaluate(left), evaluate(right)
        if left_value and right_value and left_value * right_value < 0:
            return left, right
        left_float = math.nextafter(left_float, -math.inf)
        right_float = math.nextafter(right_float, math.inf)
    raise ExactSolveFailure("could not isolate the quadratic witness root")


def exact_affine_parameter_interval(conditions):
    """Intersect lo <= a+b*alpha <= hi; None denotes an infinite endpoint."""
    lower = upper = None
    lower_labels, upper_labels = [], []
    impossible = []
    for label, a, b, lo, hi in conditions:
        a, b = _fr(a), _fr(b)
        for side, bound in (("lower", lo), ("upper", hi)):
            if bound is None:
                continue
            bound = _fr(bound)
            name = f"{label}:{side}"
            if not b:
                if (side == "lower" and a < bound
                        or side == "upper" and a > bound):
                    impossible.append(name)
                continue
            endpoint = (bound - a) / b
            is_lower = (side == "lower") == (b > 0)
            if is_lower:
                if lower is None or endpoint > lower:
                    lower, lower_labels = endpoint, [name]
                elif endpoint == lower:
                    lower_labels.append(name)
            else:
                if upper is None or endpoint < upper:
                    upper, upper_labels = endpoint, [name]
                elif endpoint == upper:
                    upper_labels.append(name)
    return {
        "alpha_interval_lower": _fs(lower) if lower is not None else None,
        "alpha_interval_upper": _fs(upper) if upper is not None else None,
        "alpha_interval_empty": bool(impossible or (
            lower is not None and upper is not None and lower > upper)),
        "active_constraint_at_lower": lower_labels,
        "active_constraint_at_upper": upper_labels,
        "constant_constraint_violations": impossible,
        "linear_constraint_count": len(conditions),
    }


def _root_inside_parameter_interval(kind, root, isolation, polynomial, interval):
    if interval["alpha_interval_empty"]:
        return False
    lo = interval["alpha_interval_lower"]
    hi = interval["alpha_interval_upper"]
    lo, hi = (_fr(lo) if lo is not None else None,
              _fr(hi) if hi is not None else None)
    if kind == "rational":
        return (lo is None or root >= lo) and (hi is None or root <= hi)
    context = QuadraticRootContext.from_record({
        "polynomial": [_fs(value) for value in polynomial],
        "isolating_interval": [_fs(value) for value in isolation]})
    alpha = QuadraticElement(context, Fraction(0), Fraction(1))
    return (lo is None or (alpha - lo).sign() >= 0) and (
        hi is None or (hi - alpha).sign() >= 0)


def _irrational_quadratic_branch_inside(a, b, discriminant, sign, interval):
    """Compare (-b +/- sqrt(D))/(2a), a>0, with rational endpoints exactly."""
    if a <= 0 or sign not in (-1, 1):
        raise RuntimeError("quadratic branch normalization differs")
    lo, hi = interval["alpha_interval_lower"], interval["alpha_interval_upper"]
    if lo is not None:
        h = 2 * a * _fr(lo) + b
        above = ((h < 0 and discriminant <= h * h) if sign == -1
                 else (h <= 0 or discriminant >= h * h))
        if not above:
            return False
    if hi is not None:
        h = 2 * a * _fr(hi) + b
        below = ((h >= 0 or discriminant >= h * h) if sign == -1
                 else (h > 0 and discriminant <= h * h))
        if not below:
            return False
    return not interval["alpha_interval_empty"]


def box_aware_correction_bases(sensitivity, candidate, low, high, maximum=8):
    """Prefer positive-slack spanning pools; numerical rank remains untrusted."""
    rows = sensitivity.shape[0]
    exact_slack = [max(Fraction(0), min(_fr(value) - lo, hi - _fr(value)))
                   for value, lo, hi in zip(candidate, low, high)]
    slack = np.asarray([float(value) for value in exact_slack])
    free = np.asarray([i for i, (lo, hi) in enumerate(zip(low, high)) if lo < hi],
                      dtype=np.int64)
    positive = np.asarray([i for i in free if exact_slack[int(i)] > 0], dtype=np.int64)
    if len(free) < rows:
        raise ExactSolveFailure("too few free correction sources")
    tolerance = max(sensitivity.shape) * np.finfo(np.float64).eps * max(
        (float(np.linalg.norm(sensitivity[:, i])) for i in free), default=0.0)

    def propose(pool, weights=None):
        if not len(pool):
            return None, 0
        values = sensitivity[:, pool]
        if weights is not None:
            values = values * weights
        _q, r, pivots = qr(values, mode="economic", pivoting=True)
        # A selected basis is screened again on the unweighted original columns.
        columns = tuple(int(i) for i in pool[pivots[:rows]])
        _q, unweighted = qr(sensitivity[:, columns], mode="economic")
        rank = int(np.sum(np.abs(np.diag(unweighted)) > tolerance))
        return (columns if len(pool) >= rows and rank == rows else None), rank

    positive_basis, positive_rank = propose(positive)
    best_pool = positive if positive_basis is not None else free
    threshold_searches = 0
    if positive_basis is not None:
        thresholds = sorted(set(float(exact_slack[int(i)]) for i in positive))
        left, right = 0, len(thresholds)
        while left < right:
            middle = (left + right) // 2
            pool = positive[slack[positive] >= thresholds[middle]]
            basis, _rank = propose(pool)
            threshold_searches += 1
            if basis is not None:
                best_pool = pool
                left = middle + 1
            else:
                right = middle
    audit = {
        "source_count": len(candidate),
        "source_variables_with_positive_slack": len(positive),
        "source_slack_threshold_counts": {
            value: sum(s >= Fraction(value) for s in exact_slack)
            for value in ("1e-12", "1e-9", "1e-6", "1e-3")},
        "positive_slack_pool_numerical_rank": positive_rank,
        "positive_slack_full_rank_candidate_found": positive_basis is not None,
        "positive_slack_basis_classification": (
            "POSITIVE_SLACK_FULL_RANK_CANDIDATE_FOUND" if positive_basis is not None
            else "POSITIVE_SLACK_BASIS_IMPOSSIBLE_TOO_FEW_SOURCES" if len(positive) < rows
            else "POSITIVE_SLACK_FULL_RANK_NOT_FOUND_BY_NUMERICAL_SCREEN"),
        "rank_screen_is_not_an_exact_nonexistence_proof": True,
        "slack_threshold_rank_searches": threshold_searches,
    }
    bases, seen = [], set()
    pools = [best_pool] if positive_basis is None else [best_pool, positive]
    for pool in pools:
        scale = max(float(slack[pool].max()), np.finfo(np.float64).tiny)
        weights = np.maximum(slack[pool] / scale, 1e-6)
        for ordinal in range(maximum * 2):
            modulation = 0.5 + ((pool * (2 * ordinal + 1) + ordinal) % 17) / 16.0
            columns, rank = propose(pool, weights ** (
                1.0 if ordinal % 2 == 0 else 0.5) * modulation)
            if columns is None or columns in seen:
                continue
            seen.add(columns)
            selected_slack = [exact_slack[i] for i in columns]
            bases.append({
                "columns": columns, "basis_sha256": _sha_json(columns),
                "minimum_selected_source_slack": float(min(selected_slack)),
                "minimum_selected_source_slack_exact": _fs(min(selected_slack)),
                "selected_source_slacks_exact": [_fs(value) for value in selected_slack],
                "selected_source_slacks": [float(value) for value in selected_slack],
                "numerical_rank": rank})
            if len(bases) == maximum:
                break
        if len(bases) == maximum:
            break
    if not bases:
        raise ExactSolveFailure("fixed-pattern cancellation correction rank deficient")
    bases.sort(key=lambda row: (-row["minimum_selected_source_slack"], row["columns"]))
    audit["maximum_minimum_slack_among_deterministic_full_rank_candidates"] = max(
        row["minimum_selected_source_slack"] for row in bases)
    for basis in bases:
        basis["slack_diagnostics"] = audit
    return bases, slack


def _replay_affine_family(problem, pattern, source_constant, source_slope,
                          c_constant, c_slope, t_constant, t_slope, t_upper,
                          node_lp=None):
    conditions = [(f"source[{i}]", a, b, lo, hi) for i, (a, b, lo, hi)
                  in enumerate(zip(source_constant, source_slope,
                                   problem.low, problem.high))]
    lower, _proof = exact_dyadic_sqrt_lower(problem.epsilon)
    conditions.append(("t", t_constant, t_slope, lower, t_upper))
    scaled_a = [g * value for g, value in zip(problem.gamma, c_constant)]
    scaled_b = [g * value for g, value in zip(problem.gamma, c_slope)]
    w1_beta = _matvec(problem.W1, problem.beta)
    g_constant, g_slope = [], []
    for i, (row, active) in enumerate(zip(problem.W1, pattern)):
        q = w1_beta[i] + problem.b1[i]
        g_constant.append(_dot(row, scaled_a) + q * t_constant)
        g_slope.append(_dot(row, scaled_b) + q * t_slope)
        conditions.append((f"relu[{i}]", g_constant[-1], g_slope[-1],
                           Fraction(0) if active else None,
                           None if active else Fraction(0)))
    if node_lp is not None:
        # c, g and u are constructed from the source family and fixed pattern;
        # their defining equalities hold by construction. Cancellation was
        # replayed identically before this function. Check every remaining node
        # row (triangles, stable signs/values, branch phases, any extra rows).
        u_constant = [value if active else Fraction(0)
                      for value, active in zip(g_constant, pattern)]
        u_slope = [value if active else Fraction(0)
                   for value, active in zip(g_slope, pattern)]
        constants = [*source_constant, *c_constant, t_constant,
                     *g_constant, *u_constant]
        slopes = [*source_slope, *c_slope, t_slope, *g_slope, *u_slope]
        if len(constants) != len(node_lp.variable_names):
            raise RuntimeError("affine family/node LP topology differs")
        for i, (lo, hi) in enumerate(zip(node_lp.column_lower, node_lp.column_upper)):
            if lo is not None or hi is not None:
                conditions.append((f"node_column[{i}]", constants[i], slopes[i], lo, hi))
        for row in node_lp.rows:
            if row.name.split("[")[0] in ("centered", "preactivation", "cancellation"):
                continue
            conditions.append((f"node:{row.name}",
                               _dot(row.coefficients, [constants[i] for i in row.indices]),
                               _dot(row.coefficients, [slopes[i] for i in row.indices]),
                               row.lower, row.upper))
    interval = exact_affine_parameter_interval(conditions)
    evidence = {**interval, "polynomial_constructed": False,
                "polynomial_degree": None, "roots_total": 0,
                "roots_inside_interval": 0, "exact_replay_result": None}
    if interval["alpha_interval_empty"]:
        return None, evidence
    polynomial = (
        problem.d * t_slope ** 2 - _dot(c_slope, c_slope),
        2 * (problem.d * t_constant * t_slope - _dot(c_constant, c_slope)),
        problem.d * t_constant ** 2 - _dot(c_constant, c_constant)
        - problem.d * problem.epsilon)
    while polynomial and polynomial[0] == 0:
        polynomial = polynomial[1:]
    evidence.update(polynomial_constructed=True,
                    polynomial=[_fs(value) for value in polynomial],
                    polynomial_degree=len(polynomial) - 1)
    roots = []
    roots_total = 0
    if len(polynomial) == 2:
        roots = [("rational", -polynomial[1] / polynomial[0], None)]
    elif len(polynomial) == 3:
        a, b, c = polynomial
        discriminant = b * b - 4 * a * c
        if discriminant >= 0:
            exact_sqrt = _fraction_square_root(discriminant)
            if exact_sqrt is not None:
                roots = [("rational", value, None) for value in sorted(set(
                    (-b + sign * exact_sqrt) / (2 * a) for sign in (-1, 1)))]
            else:
                roots_total = 2
                normalized_a, normalized_b = (a, b) if a > 0 else (-a, -b)
                # Exact endpoint comparisons happen before any root isolation.
                for sign in (-1, 1):
                    if not _irrational_quadratic_branch_inside(
                            normalized_a, normalized_b, discriminant, sign, interval):
                        continue
                    square = math.sqrt(float(discriminant))
                    approximation = (-float(normalized_b) + sign * square) / (
                        2 * float(normalized_a))
                    roots.append(("algebraic", None,
                                  _isolate_quadratic_root(polynomial, approximation)))
    elif polynomial and polynomial[0]:
        pass  # Nonzero constant: this particular line has no roots.
    else:
        # Identically zero polynomial: an admissible rational point suffices.
        lo, hi = interval["alpha_interval_lower"], interval["alpha_interval_upper"]
        value = (_fr(lo) if lo is not None else
                 min(Fraction(0), _fr(hi)) if hi is not None else Fraction(0))
        roots = [("rational", value, None)]
    evidence["roots_total"] = roots_total or len(roots)
    failures = []
    for kind, root, isolation in roots:
        if not _root_inside_parameter_interval(kind, root, isolation,
                                               polynomial, interval):
            continue
        evidence["roots_inside_interval"] += 1
        try:
            if kind == "rational":
                witness = {"schema": WITNESS_SCHEMA,
                           "source_values": [_fs(a + b * root) for a, b in
                                             zip(source_constant, source_slope)],
                           "t": _fs(t_constant + t_slope * root),
                           "relu_active": list(pattern)}
                replay = replay_exact_perspective_witness(problem, witness)
            else:
                witness = {"schema": WITNESS_SCHEMA,
                           "algebraic_root": {
                               "polynomial": [_fs(value) for value in polynomial],
                               "isolating_interval": [_fs(value) for value in isolation]},
                           "source_affine": [
                               {"constant": _fs(a), "slope": _fs(b)} for a, b
                               in zip(source_constant, source_slope)],
                           "t_affine": {"constant": _fs(t_constant),
                                        "slope": _fs(t_slope)},
                           "relu_active": list(pattern)}
                replay = replay_algebraic_perspective_witness(problem, witness)
            evidence["exact_replay_result"] = replay
            return witness, evidence
        except ExactSolveFailure:
            raise
        except RuntimeError as error:
            failures.append(f"{type(error).__name__}: {error}")
    evidence["exact_replay_result"] = {"verified": False, "failures": failures}
    return None, evidence


def reconstruct_exact_fixed_pattern_witness(
        problem: ExactPerspectiveProblem, solution, pattern,
        source_count: int, timeout_seconds: float, node_lp=None):
    """Exact 127-correction-variable/one-root reconstruction."""
    started = time.perf_counter()
    deadline = started + timeout_seconds
    d = problem.d
    candidate = np.asarray(solution[:source_count], dtype=np.float64)
    t_candidate = _fr(solution[source_count + d])
    if len(problem.X) != source_count or len(pattern) != d:
        raise RuntimeError("fixed-pattern reconstruction topology differs")
    active = [index for index, value in enumerate(pattern) if value]
    gamma = list(problem.gamma)
    w1_beta = _matvec(problem.W1, problem.beta)
    h_matrix = []
    h_t = []
    for output in range(d):
        row = []
        for feature in range(d):
            value = gamma[feature] if output == feature else Fraction(0)
            value += sum((problem.W2[output][neuron]
                          * problem.W1[neuron][feature] * gamma[feature]
                          for neuron in active), Fraction(0))
            row.append(value)
        h_matrix.append(row)
        h_t.append(problem.beta[output] + problem.b2[output] + sum(
            (problem.W2[output][neuron]
             * (w1_beta[neuron] + problem.b1[neuron])
             for neuron in active), Fraction(0)))
    difference_matrix = [[h_matrix[row][feature] - h_matrix[0][feature]
                          for feature in range(d)]
                         for row in range(1, d)]
    difference_t = [h_t[row] - h_t[0] for row in range(1, d)]
    x_numeric = np.asarray([[float(value) for value in row]
                            for row in problem.X], dtype=np.float64)
    x_numeric -= x_numeric.mean(axis=1, keepdims=True)
    m_numeric = np.asarray([[float(value) for value in row]
                            for row in difference_matrix], dtype=np.float64)
    sensitivity = m_numeric @ x_numeric.T
    bases, slack = box_aware_correction_bases(
        sensitivity, candidate, problem.low, problem.high)
    xi_candidate = [_fr(value) for value in candidate]
    centered_x0 = [value - sum(problem.x0, Fraction(0)) / d
                   for value in problem.x0]
    c_candidate = list(centered_x0)
    for source, row in enumerate(problem.X):
        mean = sum(row, Fraction(0)) / d
        value = xi_candidate[source]
        for feature, coefficient in enumerate(row):
            c_candidate[feature] += (coefficient - mean) * value
    residual = [_dot(row, c_candidate) + coefficient * t_candidate
                for row, coefficient in zip(difference_matrix, difference_t)]
    # Same authenticated box upper bound as the LP, recomputed exactly.
    c_lower, c_upper = list(centered_x0), list(centered_x0)
    for row, lo, hi in zip(problem.X, problem.low, problem.high):
        mean = sum(row, Fraction(0)) / d
        for feature, coefficient in enumerate(row):
            a, b = (coefficient - mean) * lo, (coefficient - mean) * hi
            c_lower[feature] += min(a, b)
            c_upper[feature] += max(a, b)
    t_upper = _exact_sqrt_upper(sum(
        (max(abs(lo), abs(hi)) ** 2 for lo, hi in zip(c_lower, c_upper)),
        Fraction(0)) / d + problem.epsilon)
    t_lower, _lower_proof = exact_dyadic_sqrt_lower(problem.epsilon)
    families = []
    centered_cache = {}

    def centered_source(index):
        if index not in centered_cache:
            row = problem.X[index]
            mean = sum(row, Fraction(0)) / d
            centered_cache[index] = [value - mean for value in row]
        return centered_cache[index]

    # Each basis supplies t and, if available, a nonbasic source direction.
    # Fixed sources remain the exact dyadic LP proposal. All work is bounded by
    # the existing global deadline and at most eight 30-second family budgets.
    exact_matrix_cache, factor_cache = {}, {}
    linear_systems = []
    slack_diagnostics = bases[0].get("slack_diagnostics", {})

    def emit(row):
        families.append(row)
        print(json.dumps({"stage": "box_aware_reconstruction_family",
                          "family": len(families) - 1, **row}), flush=True)

    for basis in bases:
        if len(families) >= 8 or time.perf_counter() >= deadline:
            break
        columns = basis["columns"]
        remaining_sources = sorted(
            (index for index in range(source_count)
             if index not in columns and problem.low[index] < problem.high[index]),
            key=lambda index: (-slack[index], index))
        directions = [None] + remaining_sources[:1]
        basis_started = time.perf_counter()
        pending = []
        for free_source in directions:
            if len(families) + len(pending) >= 8:
                break
            row = {key: value for key, value in basis.items() if key != "columns"}
            row.update(
                free_parameter="t" if free_source is None else
                f"source_delta[{free_source}]",
                per_family_time_cap_seconds=30.0,
                alpha_interval=None, polynomial_degree=None,
                roots_total=0, roots_inside_interval=0,
                exact_replay_result=None, verified=False)
            fixed_conditions = [
                (f"fixed_source[{index}]", value, Fraction(0), lo, hi)
                for index, (value, lo, hi) in enumerate(zip(
                    xi_candidate, problem.low, problem.high))
                if index not in columns and index != free_source]
            if free_source is not None:
                fixed_conditions.append(("fixed_t", t_candidate,
                                         Fraction(0), t_lower, t_upper))
            early_interval = exact_affine_parameter_interval(fixed_conditions)
            if early_interval["alpha_interval_empty"]:
                row.update(early_interval, alpha_interval=early_interval,
                           polynomial_constructed=False,
                           failure="fixed family constraints violate authenticated box/scale",
                           runtime_seconds=time.perf_counter() - basis_started)
                emit(row)
            else:
                pending.append((free_source, row))
        if not pending:
            continue
        shared_deadline = min(deadline, basis_started + 30.0)
        if time.perf_counter() >= shared_deadline:
            for _source, row in pending:
                row.update(failure="reconstruction family deadline",
                           runtime_seconds=time.perf_counter() - basis_started)
                emit(row)
            continue
        restore_alarm = _start_reconstruction_alarm(shared_deadline - time.perf_counter())
        profile = {"basis_sha256": basis["basis_sha256"],
                   "selected_source_slacks": basis.get("selected_source_slacks"),
                   "selected_source_slacks_exact": basis.get("selected_source_slacks_exact"),
                   "minimum_selected_source_slack": basis["minimum_selected_source_slack"],
                   "shared_free_directions": [row["free_parameter"] for _, row in pending]}
        linear_systems.append(profile)
        recorded = set()
        try:
            def remaining():
                value = shared_deadline - time.perf_counter()
                if value <= 0:
                    raise ExactSolveFailure("reconstruction family deadline")
                return value
            assembly_started = time.perf_counter()
            if columns not in exact_matrix_cache:
                exact_matrix_cache[columns] = [
                    [_dot(m, centered_source(column)) for column in columns]
                    for m in difference_matrix]
            exact_matrix = exact_matrix_cache[columns]
            rhs_columns, direction_data = [], []
            for free_source, row in pending:
                t_constant = Fraction(0) if free_source is None else t_candidate
                t_slope = Fraction(1) if free_source is None else Fraction(0)
                free_c = ([Fraction(0)] * d if free_source is None
                          else centered_source(free_source))
                rhs_columns.append([-r + v * (t_candidate - t_constant)
                                    for r, v in zip(residual, difference_t)])
                rhs_columns.append([-v * t_slope - _dot(m, free_c)
                                    for m, v in zip(difference_matrix, difference_t)])
                direction_data.append((t_constant, t_slope, free_c))
            profile["rational_assembly_seconds"] = time.perf_counter() - assembly_started
            condition = float(np.linalg.cond(sensitivity[:, columns]))
            profile["numerical_condition_estimate"] = condition if math.isfinite(condition) else None
            profile["numerical_condition_estimate_nonfinite"] = not math.isfinite(condition)
            solved = _solve_exact_correction_rhs(
                exact_matrix, rhs_columns, remaining(), factor_cache, profile)
            for ordinal, ((free_source, row), (t_constant, t_slope, free_c)) in enumerate(
                    zip(pending, direction_data)):
                remaining()
                constant_delta, slope_delta = solved[2 * ordinal:2 * ordinal + 2]
                source_constant = list(xi_candidate)
                source_slope = [Fraction(0)] * source_count
                c_constant, c_slope = list(c_candidate), list(free_c)
                if free_source is not None:
                    source_slope[free_source] = Fraction(1)
                for j, column in enumerate(columns):
                    source_constant[column] += constant_delta[j]
                    source_slope[column] = slope_delta[j]
                    for feature, value in enumerate(centered_source(column)):
                        c_constant[feature] += value * constant_delta[j]
                        c_slope[feature] += value * slope_delta[j]
                if any(_dot(m, c_constant) + v * t_constant or
                       _dot(m, c_slope) + v * t_slope
                       for m, v in zip(difference_matrix, difference_t)):
                    raise ExactSolveFailure("affine family cancellation replay failed")
                remaining()
                found, evidence = _replay_affine_family(
                    problem, pattern, source_constant, source_slope,
                    c_constant, c_slope, t_constant, t_slope, t_upper, node_lp)
                row.update(evidence, linear_solve=profile)
                row["alpha_interval"] = {
                    key: evidence[key] for key in (
                        "alpha_interval_lower", "alpha_interval_upper",
                        "alpha_interval_empty", "active_constraint_at_lower",
                        "active_constraint_at_upper")}
                row.update(verified=found is not None,
                           runtime_seconds=time.perf_counter() - basis_started)
                emit(row)
                recorded.add(id(row))
                if found is not None:
                    return found, {
                        "attempted": True, "verified": True,
                        "method": "BAREISS_127_CORRECTIONS_PLUS_QUADRATIC_ROOT",
                        "families": families, "linear_systems": linear_systems,
                        "basis_slack_diagnostics": slack_diagnostics,
                        "selected_correction_variables": d - 1,
                        "runtime_seconds": time.perf_counter() - started}
        except (RuntimeError, ExactSolveFailure, OverflowError) as error:
            profile["failure"] = f"{type(error).__name__}: {error}"
            for _source, row in pending:
                if id(row) not in recorded:
                    row.update(failure=profile["failure"], linear_solve=profile,
                               runtime_seconds=time.perf_counter() - basis_started)
                    emit(row)
        finally:
            restore_alarm()
    return None, {
        "attempted": True, "verified": False,
        "method": "BAREISS_127_CORRECTIONS_PLUS_QUADRATIC_ROOT",
        "families": families, "linear_systems": linear_systems,
        "basis_slack_diagnostics": slack_diagnostics,
        "permits_infeasibility_claim": False,
        "portfolio_limit": 8, "per_family_time_cap_seconds": 30.0,
        "runtime_seconds": time.perf_counter() - started}


def execute(capture_root: Path, downstream_report: Path, output: Path,
            artifact_dir: Path, maximum_nodes: int,
            wall_seconds: float, maximum_patterns: int) -> dict:
    started = time.perf_counter()
    _ensure_production_modules()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    downstream = _authenticate_downstream_token(downstream_report)
    snapshot, authentication = _load_authenticated_capture(capture_root)
    token = downstream["analysis_token_index"]
    identity = {
        "pinned_deept_revision": authentication["pinned_deept_revision"],
        "scientific_manifest_sha256": authentication[
            "scientific_manifest_sha256"],
        "production_manifest_sha256": authentication[
            "production_manifest_sha256"],
    }
    gamma, beta, W1, b1, W2, b2, epsilon, parameter_identity = \
        _load_parameters(identity)
    _verify_cross_authentication(authentication, downstream, parameter_identity)
    weights = snapshot["weights"].numpy()
    low = snapshot["range_low"].numpy()
    high = snapshot["range_high"].numpy()
    bounds = _derive_exact_bounds(
        weights[0, token], weights[1:, token], low, high,
        gamma, beta, W1, b1, epsilon)
    backend = _backend_inventory()
    model = _root_model_summary(EXPECTED_SOURCES, bounds)
    lp_started = time.perf_counter()
    lp = build_exact_perspective_lp(
        low, high, gamma, beta, W1, b1, W2, b2, bounds)
    lp_build_seconds = time.perf_counter() - lp_started
    lp_artifact = persist_exact_lp(
        lp, artifact_dir / "root_canonical_exact_lp.jsonl.gz")
    canonical_inequality_count = len(canonicalize_all_inequalities(lp))
    solver = None
    raw_ray_record = None
    certificate = None
    certificate_replay = None
    reconstruction_attempts = []
    fallback = {"attempted": False}
    if backend["packages"]["highspy"]:
        solver = solve_highspy(
            lp, artifact_dir / "root_highspy.log",
            artifact_dir / "root_highspy_exact_row_scaling.json")
        if solver["raw_dual_ray"] is not None:
            raw_ray_record = _atomic_json(
                artifact_dir / "root_highspy_raw_dual_ray.json", {
                    "schema": "CORET_HIGHSPY_RAW_DUAL_RAY_V1",
                    "canonical_lp_sha256": lp.identity(),
                    "solver_scaled_lp_sha256": solver[
                        "solver_scaling"]["solver_scaled_lp_sha256"],
                    "row_scaling_sha256": solver[
                        "solver_scaling"]["row_scaling_sha256"],
                    "row_order_sha256": _sha_json(
                        [row.name for row in lp.rows]),
                    "solver_scaled_row_values": [
                        float(value) for value in solver["raw_dual_ray"]],
                    "original_row_values": [
                        float(value)
                        for value in solver["original_row_dual_ray"]],
                    "highs_version": solver["highs_version"],
                    "model_status": solver["model_status"],
                })
            certificate, reconstruction_attempts, _repair_status = \
                repair_direct_dual_ray(
                    lp, solver["original_row_dual_ray"],
                    min(600.0, max(1.0, wall_seconds / 4)))
        elif solver["infeasible"]:
            fallback["attempted"] = True
            certificate, fallback = phase1_exact_farkas_fallback(
                lp, min(600.0, max(1.0, wall_seconds / 4)))
    else:
        fallback["attempted"] = True
        certificate, fallback = phase1_exact_farkas_fallback(
            lp, min(600.0, max(1.0, wall_seconds / 4)))
    certificate_record = None
    if certificate is not None:
        certificate_replay = verify_exact_lp_farkas(lp, certificate)
        certificate_record = _atomic_json(
            artifact_dir / "root_exact_farkas_certificate.json", certificate)

    root = {
        "variable_count": lp.column_count,
        "row_count": len(lp.rows), "nnz": lp.nnz,
        "canonical_inequality_count": canonical_inequality_count,
        "cone_dimension": 0,
        "canonical_lp_sha256": lp.identity(),
        "canonical_lp_path": lp_artifact["path"],
        "canonical_lp_artifact_sha256": lp_artifact["sha256"],
        "solver_scaling": (None if solver is None else
                           solver["solver_scaling"]),
        "solver_scaling_path": (None if solver is None else str(
            artifact_dir / "root_highspy_exact_row_scaling.json")),
        "solver_scaling_artifact_sha256": (
            None if solver is None else cluster_common.sha256(
                artifact_dir / "root_highspy_exact_row_scaling.json")),
        "lp_build_seconds": lp_build_seconds,
        "solver_backend": backend["selected"],
        "highspy_version": None if solver is None else solver["highspy_version"],
        "highs_version": None if solver is None else solver["highs_version"],
        "options": None if solver is None else solver["options"],
        "solver_status": ("HIGHSPY_UNAVAILABLE" if solver is None
                          else solver["model_status"]),
        "runtime_seconds": (None if solver is None
                            else solver["runtime_seconds"]),
        "feasible": None if solver is None else solver["feasible"],
        "infeasible": None if solver is None else solver["infeasible"],
        "direct_dual_ray_available": (
            False if solver is None else solver["direct_dual_ray_available"]),
        "raw_dual_ray_path": (None if raw_ray_record is None else
                              str(artifact_dir /
                                  "root_highspy_raw_dual_ray.json")),
        "raw_dual_ray_sha256": (None if raw_ray_record is None else
                                cluster_common.sha256(
                                    artifact_dir /
                                    "root_highspy_raw_dual_ray.json")),
        "certificate_attempted": bool(reconstruction_attempts) or
                                 fallback.get("attempted", False),
        "certificate_reconstruction_attempts": reconstruction_attempts,
        "phase_i_fallback": fallback,
        "certificate_verified": certificate_replay is not None,
        "certificate_path": (None if certificate_record is None else
                             str(artifact_dir /
                                 "root_exact_farkas_certificate.json")),
        "certificate_sha256": (None if certificate_record is None else
                               cluster_common.sha256(
                                   artifact_dir /
                                   "root_exact_farkas_certificate.json")),
        "certificate_exact_replay": certificate_replay,
        "certificate_extraction_path": (
            "highspy.getDualRay -> exact support repair -> exact Farkas replay"
            if reconstruction_attempts else
            ("scipy HiGHS Phase-I/Farkas proposal -> exact replay"
             if fallback.get("attempted") else None)),
    }
    exact_witness = _empty_exact_witness_record()
    branch = {
        "attempted": False, "nodes_created": 0,
        "nodes_closed_by_certificate": 0, "nodes_open": 0,
        "maximum_depth": 0, "phase_patterns_attempted": 0,
        "proof_tree_path": None, "proof_tree_sha256": None,
        "limits": {"maximum_nodes": maximum_nodes,
                   "wall_clock_seconds": wall_seconds,
                   "maximum_exact_witness_pattern_attempts": maximum_patterns},
        "branch_rule": (
            "largest deterministic triangle-hull violation; ties by neuron index"),
        "continuous_source_branching": False,
    }
    complete_tree_replay = None
    if solver is not None and solver["feasible"]:
        if (solver["column_values"] is None
                or len(solver["column_values"]) != lp.column_count
                or solver["solver_scaling"][
                    "original_canonical_lp_sha256"] != lp.identity()
                or solver["solver_scaling"]["solver_scaled_lp_sha256"] !=
                    solver["construction_diagnostic"][
                        "solver_scaled_lp_sha256"]):
            raise RuntimeError("feasible root LP solution identity differs")
        exact_problem = _exact_problem_from_arrays(
            weights, token, low, high, gamma, beta, W1, b1, W2, b2,
            epsilon)
        node_lps_by_identity = {}

        def solve_phase_node(phases, node_id):
            node_lp = lp_with_relu_phases(lp, phases, EXPECTED_SOURCES)
            node_lps_by_identity[node_lp.identity()] = node_lp
            safe = node_id.replace(".", "_")
            node_solver = solve_highspy(
                node_lp, artifact_dir / f"bab_{safe}_highspy.log",
                artifact_dir / f"bab_{safe}_row_scaling.json")
            node_certificate = None
            if node_solver["infeasible"]:
                if node_solver["original_row_dual_ray"] is not None:
                    node_certificate, _attempts, _status = \
                        repair_direct_dual_ray(
                            node_lp, node_solver["original_row_dual_ray"],
                            min(300.0, max(1.0, wall_seconds / 8)))
                if node_certificate is None:
                    node_certificate, _fallback = \
                        phase1_exact_farkas_fallback(
                            node_lp,
                            min(300.0, max(1.0, wall_seconds / 8)))
                if node_certificate is not None:
                    verify_exact_lp_farkas(node_lp, node_certificate)
            return {
                "solver_status": node_solver["model_status"],
                "feasible": node_solver["feasible"],
                "infeasible": node_solver["infeasible"],
                "solution": node_solver["column_values"],
                "certificate": node_certificate,
            }

        def attempt_witness(pattern, solution, node_id):
            found, direct = attempt_exact_lp_candidate_witness(
                exact_problem, solution, pattern, EXPECTED_SOURCES)
            reconstruction = None
            if found is None:
                try:
                    found, reconstruction = \
                        reconstruct_exact_fixed_pattern_witness(
                            exact_problem, solution, pattern,
                            EXPECTED_SOURCES,
                            min(240.0, max(
                                0.1, started + wall_seconds
                                - time.perf_counter())),
                            node_lp=lp_with_relu_phases(
                                lp, dict(enumerate(pattern)), EXPECTED_SOURCES))
                except (RuntimeError, ExactSolveFailure) as error:
                    reconstruction = {
                        "attempted": True, "verified": False,
                        "method":
                            "BAREISS_127_CORRECTIONS_PLUS_QUADRATIC_ROOT",
                        "failure": f"{type(error).__name__}: {error}"}
            evidence = {
                "attempted": True, "verified": found is not None,
                "direct_dyadic_replay": direct,
                "correction_reconstruction": reconstruction}
            if found is None:
                return None, evidence
            record = _atomic_json(
                artifact_dir / "exact_layernorm_cancellation_witness.json",
                found)
            replay = (replay_algebraic_perspective_witness(exact_problem, record)
                      if "algebraic_root" in record else
                      replay_exact_perspective_witness(exact_problem, record))
            return record, {
                **evidence, "witness_path": str(
                    artifact_dir / "exact_layernorm_cancellation_witness.json"),
                "witness_sha256": cluster_common.sha256(
                    artifact_dir / "exact_layernorm_cancellation_witness.json"),
                "exact_replay": replay,
            }

        found, continuation = run_phase_continuation(
            solver["column_values"], bounds, EXPECTED_SOURCES,
            maximum_nodes, maximum_patterns, started + wall_seconds,
            solve_phase_node, attempt_witness)
        tree = continuation.pop("tree")
        if found is None and continuation["nodes_open"] == 0:
            complete_tree_replay = verify_branch_tree(
                tree, lambda cert: verify_exact_lp_farkas(
                    node_lps_by_identity[cert["canonical_lp_sha256"]], cert))
        tree_record = _atomic_json(
            artifact_dir / "relu_phase_proof_tree.json",
            tree)
        branch.update({
            **continuation,
            "proof_tree_path": str(artifact_dir /
                                   "relu_phase_proof_tree.json"),
            "proof_tree_sha256": cluster_common.sha256(
                artifact_dir / "relu_phase_proof_tree.json"),
            "complete_tree_exact_replay": complete_tree_replay})
        if found is not None:
            latest = branch["witness_attempts"][-1]
            exact_witness.update({
                "attempted": True, "verified": True,
                "phase_pattern_sha256": found.get(
                    "phase_pattern_sha256",
                    _sha_json(found["relu_active"])),
                "bases_attempted": branch["phase_patterns_attempted"],
                "source_box_check": True,
                "layernorm_equality_check": True,
                "relu_sign_check": True,
                "maximum_exact_residual": "0",
                "witness_path": latest["witness_path"],
                "witness_sha256": latest["witness_sha256"],
            })
    final_status = scientific_status_from_proof(
        exact_witness_verified=exact_witness["verified"],
        root_certificate_verified=certificate_replay is not None,
        complete_tree_verified=(complete_tree_replay or {}).get(
            "permits_excluded", False),
        open_nodes=branch["nodes_open"])
    if final_status == EXCLUDED:
        interpretation = "POST_ATTENTION_LAYERNORM_ABSTRACTION_CAUSAL"
    elif final_status == FEASIBLE:
        interpretation = "LAYERNORM_ABSTRACTION_EXONERATED_SEARCH_UPSTREAM"
    else:
        interpretation = "LP_RESULT_WITHOUT_DECISIVE_EXACT_PROOF"
    report = {
        "schema": SCHEMA,
        "authentication": {**authentication,
                           "downstream_cancellation": downstream,
                           "parameter_authentication": parameter_identity},
        "analysis_token_index": token,
        "perspective_formulation": {
            "dimension": DIMENSION, "source_count": EXPECTED_SOURCES,
            "epsilon": _fs(epsilon),
            "pre_ln_identity": EXPECTED_CANONICAL_IDENTITY,
            "parameter_hashes": {
                "gamma": parameter_identity["layernorm"]["gamma_sha256"],
                "beta": parameter_identity["layernorm"]["beta_sha256"],
                "W1": parameter_identity["ffn_first"]["weight_sha256"],
                "b1": parameter_identity["ffn_first"]["bias_sha256"],
                "W2": parameter_identity["ffn_second"]["weight_sha256"],
                "b2": parameter_identity["ffn_second"]["bias_sha256"],
            },
            **model,
        },
        "t_bounds": {
            "lower": _fs(bounds["t_lower"]), "upper": _fs(bounds["t_upper"]),
            **bounds["t_lower_proof"],
            "max_centered_norm_squared": _fs(bounds["max_norm_squared"]),
            "derivation": (
                "exact dyadic shared-source affine coordinate supports; "
                "sum of squared per-coordinate max magnitudes; outward "
                "binary64 rational square-root upper bound"),
            "exact_replay_verified": (
                bounds["t_upper"] * bounds["t_upper"] >=
                bounds["max_norm_squared"] / DIMENSION + epsilon),
        },
        "relu_bounds": {
            "stable_active": len(bounds["stable_active"]),
            "stable_inactive": len(bounds["stable_inactive"]),
            "unstable": len(bounds["unstable"]),
            "stable_active_indices": bounds["stable_active"],
            "stable_inactive_indices": bounds["stable_inactive"],
            "unstable_indices": bounds["unstable"],
            "bounds_sha256": bounds["bounds_sha256"],
            "derivation_runtime_seconds": bounds["runtime_seconds"],
            "all_coefficients_exact_rationals": True,
        },
        "proof_backend": backend, "root_lp": root,
        "exact_witness": exact_witness, "branch_and_bound": branch,
        "final_status": final_status,
        "causal_interpretation": interpretation,
        "runtime_seconds": time.perf_counter() - started,
        "scientific_queries": 0, "bound_calls": 0, "gpu_jobs": 0,
    }
    return _atomic_json(output, report)


def persist_highs_rejection(output: Path,
                            error: HighsCanonicalLPDiagnosticError):
    return _atomic_json(output, {
        "schema": "CORET_HIGHSPY_CANONICAL_LP_REJECTION_DIAGNOSTIC_V1",
        "final_status": "HIGHS_CANONICAL_LP_CONSTRUCTION_REJECTED",
        "diagnostic": error.diagnostic,
        "error": f"{type(error).__name__}: {error}",
        "scientific_result": None,
        "scientific_queries": 0, "bound_calls": 0, "gpu_jobs": 0,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--downstream-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--maximum-nodes", type=int, default=2048)
    parser.add_argument("--wall-clock-limit-seconds", type=float, default=7200)
    parser.add_argument("--maximum-exact-witness-patterns", type=int, default=32)
    args = parser.parse_args()
    if (args.maximum_nodes <= 0 or args.wall_clock_limit_seconds <= 0
            or args.maximum_exact_witness_patterns <= 0):
        raise RuntimeError("oracle limits must be positive")
    output = args.output.expanduser().resolve()
    try:
        report = execute(
            args.capture_root.expanduser().resolve(),
            args.downstream_report.expanduser().resolve(), output,
            args.artifact_dir.expanduser().resolve(),
            args.maximum_nodes, args.wall_clock_limit_seconds,
            args.maximum_exact_witness_patterns)
    except HighsCanonicalLPDiagnosticError as error:
        report = persist_highs_rejection(output, error)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
