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
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Iterable, Sequence

import numpy as np
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
    for row, lo_raw, hi_raw in zip(generators, low, high):
        values = [_fr(value) for value in row]
        mean = sum(values, Fraction(0)) / d
        lo, hi = _fr(lo_raw), _fr(hi_raw)
        for coordinate, value in enumerate(values):
            coefficient = value - mean
            first, second = coefficient * lo, coefficient * hi
            lower[coordinate] += min(first, second)
            upper[coordinate] += max(first, second)
    return centered_center, lower, upper


def _linear_interval(weights, lower, upper, bias=Fraction(0)):
    lo = hi = _fr(bias)
    for weight, left, right in zip(weights, lower, upper):
        weight = _fr(weight)
        first, second = weight * left, weight * right
        lo += min(first, second)
        hi += max(first, second)
    return lo, hi


def _derive_exact_bounds(center, generators, low, high, gamma, beta,
                         W1, b1, epsilon):
    started = time.perf_counter()
    c0, c_lower, c_upper = _centered_coordinate_bounds(
        center, generators, low, high)
    max_norm_squared = sum((max(abs(lo), abs(hi)) ** 2
                            for lo, hi in zip(c_lower, c_upper)), Fraction(0))
    upper = _exact_sqrt_upper(max_norm_squared / DIMENSION + epsilon)
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
        t_first, t_second = Fraction(0), upper * q
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
        "centered_lower": c_lower, "centered_upper": c_upper,
        "max_norm_squared": max_norm_squared, "t_upper": upper,
        "g_lower": g_lower, "g_upper": g_upper,
        "stable_active": active, "stable_inactive": inactive,
        "unstable": unstable, "bounds_sha256": _sha_json(bounds_payload),
        "runtime_seconds": time.perf_counter() - started,
    }


def _backend_inventory() -> dict:
    packages = {}
    for name in ("cvxpy", "clarabel", "scs"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    # No adapter in this program may call a backend unless it can export a ray
    # that the exact rational checker above can replay.  The current frozen
    # environment has no such installed backend.
    return {
        "packages": packages,
        "selected": "NONE_CERTIFICATE_CAPABLE_INSTALLED",
        "certificate_extraction_supported": False,
        "limitation": (
            "No installed conic backend exposes a usable infeasibility ray; "
            "model construction/checking remain available and the result "
            "must stay INCONCLUSIVE."),
    }


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
        "cone_dimension": DIMENSION + 2,
        "cone_encoding": (
            "||(2*c_0,...,2*c_127,(1-128)*t)||_2 <= (1+128)*t"),
        "cone_equivalent_inequality": "sum(c_j^2) <= 128*t^2",
        "epsilon_restored_only_in_exact_witness_replay": True,
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

    # With no ray-exporting conic backend, neither exclusion, a phase proposal,
    # exact witness reconstruction, nor sound phase branching may begin.  This
    # is deliberately a typed proof limitation rather than solver inference.
    root = {
        **{key: model[key] for key in (
            "variable_count", "equality_count", "inequality_count",
            "cone_dimension")},
        "solver_backend": backend["selected"],
        "solver_version": None, "solver_status": "BACKEND_UNAVAILABLE",
        "primal_residuals": None, "dual_residuals": None,
        "runtime_seconds": 0.0, "feasible": None,
        "certificate_attempted": False, "certificate_verified": False,
        "certificate_path": None, "certificate_sha256": None,
        "certificate_extraction_path": None,
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
    final_status = INCONCLUSIVE
    interpretation = "CONIC_BACKEND_LIMITATION_NO_CAUSAL_CONCLUSION"
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
            "lower": "0/1", "upper": _fs(bounds["t_upper"]),
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
        "conic_backend": backend, "root_conic": root,
        "exact_witness": exact_witness, "branch_and_bound": branch,
        "final_status": final_status,
        "causal_interpretation": interpretation,
        "runtime_seconds": time.perf_counter() - started,
        "scientific_queries": 0, "bound_calls": 0, "gpu_jobs": 0,
    }
    return _atomic_json(output, report)


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
    report = execute(
        args.capture_root.expanduser().resolve(),
        args.downstream_report.expanduser().resolve(),
        args.output.expanduser().resolve(),
        args.artifact_dir.expanduser().resolve(),
        args.maximum_nodes, args.wall_clock_limit_seconds,
        args.maximum_exact_witness_patterns)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
