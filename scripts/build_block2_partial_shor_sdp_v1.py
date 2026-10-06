#!/usr/bin/env python3
"""Model construction ONLY for a numerical partial-Shor go/no-go experiment.

No SDP solve, proof certificate, branching, or producer execution is implemented.
Exact rational coefficients are retained; CVXPY receives a floating proposal copy.
The lifted vector is ONLY q=[c,t,u]. Original xi variables are never lifted.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.sparse import coo_matrix


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "CORET_BLOCK2_PARTIAL_SHOR_NUMERICAL_MODEL_V1"
_spec = importlib.util.spec_from_file_location(
    "partial_shor_exact_perspective", ROOT / "scripts/analyze_block2_exact_layernorm_perspective_causal_v1.py")
oracle = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = oracle
_spec.loader.exec_module(oracle)


def solver_inventory():
    packages = {}
    for name in ("cvxpy", "scs", "clarabel", "cvxopt", "mosek", "highspy", "scipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    installed, psd = [], []
    if packages["cvxpy"]:
        import cvxpy as cp
        installed = cp.installed_solvers()
        from cvxpy.reductions.solvers.defines import SOLVER_MAP_CONIC
        from cvxpy.constraints.psd import PSD
        psd = [name for name in installed if name in SOLVER_MAP_CONIC
               and PSD in SOLVER_MAP_CONIC[name].SUPPORTED_CONSTRAINTS]
    return {"packages": packages, "cvxpy_installed_solvers": installed,
            "cvxpy_psd_solvers": psd, "highs_supports_psd": False,
            "minimal_prepared_handoff_dependencies": ["cvxpy", "scs"],
            "dependencies_installed_by_this_script": False}


@dataclass(frozen=True)
class MomentEquation:
    name: str
    # Upper-triangular Q coordinates. An off-diagonal entry appears ONCE,
    # not twice as it would in a full symmetric trace coefficient matrix.
    terms: tuple[tuple[int, int, Fraction], ...]
    rhs: Fraction


@dataclass(frozen=True)
class PartialShorModel:
    source_count: int
    dimension: int
    epsilon: Fraction
    linear_lp: object              # z=[xi,c,t,u], all exact coefficients
    moment_equalities: tuple[MomentEquation, ...]
    g_affine: tuple[tuple[tuple[int, Fraction], ...], ...]
    original_lp_sha256: str
    authentication: dict

    @property
    def q_dimension(self):
        return 2 * self.dimension + 1

    def summary(self):
        lp = self.linear_lp
        equalities = sum(row.lower is not None and row.lower == row.upper for row in lp.rows)
        inequalities = sum((row.lower is not None) + (row.upper is not None)
                           for row in lp.rows if row.lower is None or row.lower != row.upper)
        bound_equalities = sum(lo is not None and lo == hi
                               for lo, hi in zip(lp.column_lower, lp.column_upper))
        bound_inequalities = sum((lo is not None) + (hi is not None)
                                for lo, hi in zip(lp.column_lower, lp.column_upper)
                                if lo is None or lo != hi)
        source_rows = [row for row in lp.rows if row.name.startswith("centered[")]
        source_nnz = sum(sum(index < self.source_count for index in row.indices) for row in source_rows)
        q = self.q_dimension
        q_symmetric = q * (q + 1) // 2
        return {
            "schema": SCHEMA, "proof_authority": False,
            "rank_one_constraint_relaxed": True, "xi_lifted": False,
            "source_count": self.source_count, "hidden_dimension": self.dimension,
            "q_order": "c[0:d],t,u[0:d]", "q_dimension": q,
            "psd_block_dimension": q + 1,
            "scalar_variable_count": self.source_count + q + q_symmetric,
            "symmetric_Q_scalar_count": q_symmetric,
            "linear_row_equalities": equalities, "linear_row_inequalities": inequalities,
            "column_bound_equalities": bound_equalities,
            "column_bound_inequalities": bound_inequalities,
            "moment_equalities": len(self.moment_equalities),
            "equality_count": equalities + bound_equalities + len(self.moment_equalities),
            "inequality_count": inequalities + bound_inequalities,
            "psd_cone_count": 1, "source_matrix_nnz": source_nnz,
            "linear_matrix_nnz": lp.nnz,
            "cancellation_equation_count": sum(row.name.startswith("cancellation[") for row in lp.rows),
            "epsilon_exact": oracle._fs(self.epsilon),
            "original_canonical_lp_sha256": self.original_lp_sha256,
            "substituted_linear_lp_sha256": lp.identity(),
            "dense_moment_binary64_bytes": 8 * (q + 1) ** 2,
            "source_matrix_csr_bytes_estimate": 12 * source_nnz + 4 * (self.dimension + 1),
            "source_dense_normal_matrix_bytes_if_formed": 8 * self.source_count ** 2,
            "authentication": self.authentication,
        }

    def record(self):
        lp = self.linear_lp
        return {"summary": self.summary(),
                "variables": [{"name": name, "lower": None if lo is None else oracle._fs(lo),
                               "upper": None if hi is None else oracle._fs(hi)}
                              for name, lo, hi in zip(lp.variable_names, lp.column_lower, lp.column_upper)],
                "linear_rows": [{"name": row.name,
                                 "entries": [[index, oracle._fs(value)] for index, value in zip(row.indices, row.coefficients)],
                                 "lower": None if row.lower is None else oracle._fs(row.lower),
                                 "upper": None if row.upper is None else oracle._fs(row.upper)} for row in lp.rows],
                "moment_equalities": [{"name": row.name, "rhs": oracle._fs(row.rhs),
                                      "terms": [[i, j, oracle._fs(value)] for i, j, value in row.terms]}
                                     for row in self.moment_equalities],
                "g_affine": [[[i, oracle._fs(value)] for i, value in row] for row in self.g_affine]}


def build_partial_shor(root_lp, epsilon, source_count, dimension, *, authentication=None):
    """Exact pullback of the existing canonical rows under g=Bc+alpha*t.

    No McCormick/RLT constraints, hierarchy, new phase decisions, source lifting,
    or numerical constraint tightening are introduced.
    """
    n, d = source_count, dimension
    epsilon = oracle._fr(epsilon)
    names = tuple([f"xi[{i}]" for i in range(n)] + [f"c[{i}]" for i in range(d)] + ["t"]
                  + [f"g[{i}]" for i in range(d)] + [f"u[{i}]" for i in range(d)])
    if (root_lp.variable_names != names or epsilon <= 0 or
            len(root_lp.column_lower) != len(names) or len(root_lp.column_upper) != len(names)):
        raise RuntimeError("canonical perspective topology/epsilon differs")
    c0, t, g0, u0 = n, n + d, n + d + 1, n + 2 * d + 1
    if root_lp.column_lower[t] is None or root_lp.column_lower[t] <= 0:
        raise RuntimeError("certified positive t bound is missing")
    if sum(row.name.startswith("centered[") for row in root_lp.rows) != d or sum(
            row.name.startswith("cancellation[") for row in root_lp.rows) != d - 1:
        raise RuntimeError("source/cancellation equation population differs")
    definitions = []
    for i in range(d):
        rows = [row for row in root_lp.rows if row.name == f"preactivation[{i}]"]
        if len(rows) != 1:
            raise RuntimeError("unique canonical g definition is missing")
        row = rows[0]
        coefficients = dict(zip(row.indices, row.coefficients))
        if (row.lower != 0 or row.upper != 0 or coefficients.get(g0 + i) != 1 or
                any(index != g0 + i and not c0 <= index <= t for index in coefficients)):
            raise RuntimeError("canonical g definition is not the frozen homogeneous affine relation")
        definitions.append(tuple(sorted((index - c0, -value)
                                         for index, value in coefficients.items() if index != g0 + i)))

    def substitute(row):
        entries = []
        for index, value in zip(row.indices, row.coefficients):
            if g0 <= index < u0:
                entries.extend((n + coordinate, value * coefficient)
                               for coordinate, coefficient in definitions[index - g0])
            else:
                entries.append((index if index < g0 else index - d, value))
        indices, coefficients = oracle._coalesced(entries)
        return oracle.ExactLPRow(row.name, indices, coefficients, row.lower, row.upper)

    linear = []
    for row in root_lp.rows:
        converted = substitute(row)
        if row.name.startswith("preactivation["):
            if converted.indices or converted.lower != 0 or converted.upper != 0:
                raise RuntimeError("exact g substitution replay failed")
        else:
            linear.append(converted)
    # Preserve any original g bounds as linear constraints on its exact affine
    # definition, rather than accidentally dropping bounds while eliminating g.
    for i in range(d):
        lo, hi = root_lp.column_lower[g0 + i], root_lp.column_upper[g0 + i]
        if lo is not None or hi is not None:
            linear.append(substitute(oracle.ExactLPRow(f"g_bound[{i}]", (g0 + i,), (Fraction(1),), lo, hi)))
    constraint_keys = {(row.indices, row.coefficients, row.lower, row.upper) for row in linear}
    for i in range(d):
        for row in (oracle.ExactLPRow(f"shor_u_nonnegative[{i}]", (u0 + i,), (Fraction(-1),), None, Fraction(0)),
                    oracle.ExactLPRow(f"shor_u_above_g[{i}]", (g0 + i, u0 + i),
                                      (Fraction(1), Fraction(-1)), None, Fraction(0))):
            converted = substitute(row)
            key = (converted.indices, converted.coefficients, converted.lower, converted.upper)
            if key not in constraint_keys:
                linear.append(converted)
                constraint_keys.add(key)
    retained = tuple(range(g0)) + tuple(range(u0, u0 + d))
    lp = oracle.ExactCanonicalLP(tuple(names[i] for i in retained),
        tuple(root_lp.column_lower[i] for i in retained),
        tuple(root_lp.column_upper[i] for i in retained), tuple(linear))
    quadratic = [MomentEquation("layernorm_quadratic", tuple(
        [(i, i, Fraction(-1)) for i in range(d)] + [(d, d, Fraction(d))]), d * epsilon)]
    for i in range(d):
        u = d + 1 + i
        terms = {(u, u): Fraction(1)}
        for coordinate, coefficient in definitions[i]:
            key = (min(u, coordinate), max(u, coordinate))
            terms[key] = terms.get(key, Fraction(0)) - coefficient
        quadratic.append(MomentEquation(f"relu_complementarity[{i}]",
            tuple((a, b, value) for (a, b), value in sorted(terms.items()) if value), Fraction(0)))
    return PartialShorModel(n, d, epsilon, lp, tuple(quadratic), tuple(definitions),
                            root_lp.identity(), authentication or {"synthetic": True})


def prepare_production(capture_root, downstream_report):
    """CPU-only reuse of the SAME authentication and exact linear formulation."""
    started = time.perf_counter()
    oracle._ensure_production_modules()
    downstream = oracle._authenticate_downstream_token(Path(downstream_report))
    snapshot, authentication = oracle._load_authenticated_capture(Path(capture_root))
    identity = {key: authentication[key] for key in
                ("pinned_deept_revision", "scientific_manifest_sha256", "production_manifest_sha256")}
    gamma, beta, W1, b1, W2, b2, epsilon, parameter_identity = oracle._load_parameters(identity)
    oracle._verify_cross_authentication(authentication, downstream, parameter_identity)
    token = downstream["analysis_token_index"]
    weights = snapshot["weights"].numpy()
    low, high = snapshot["range_low"].numpy(), snapshot["range_high"].numpy()
    print(json.dumps({"stage": "shor_inputs_authenticated", "seconds": time.perf_counter() - started}), flush=True)
    bounds = oracle._derive_exact_bounds(weights[0, token], weights[1:, token], low, high,
                                        gamma, beta, W1, b1, epsilon)
    root_lp = oracle.build_exact_perspective_lp(low, high, gamma, beta, W1, b1, W2, b2, bounds)
    model = build_partial_shor(root_lp, epsilon, oracle.EXPECTED_SOURCES, oracle.DIMENSION,
        authentication={"capture": authentication, "downstream": downstream,
                        "parameters": parameter_identity, "bounds_sha256": bounds["bounds_sha256"],
                        "t_lower_exact": oracle._fs(bounds["t_lower"]),
                        "t_upper_exact": oracle._fs(bounds["t_upper"])})
    print(json.dumps({"stage": "shor_model_constructed", "seconds": time.perf_counter() - started,
                      **model.summary()}), flush=True)
    return model


def _proposal_float(value):
    converted = float(value)
    if not np.isfinite(converted) or (value != 0 and converted == 0):
        raise RuntimeError("nonfinite/underflowed floating SDP proposal coefficient")
    return converted


def as_cvxpy_problem(model):
    """Construct, but DO NOT solve, a floating SDP copy with zero proof authority."""
    import cvxpy as cp
    n, k = model.source_count, model.q_dimension
    xi, q, Q = cp.Variable(n, name="xi_unlifted"), cp.Variable(k, name="q"), cp.Variable((k, k), symmetric=True, name="Q")
    moment = cp.bmat([[np.ones((1, 1)), cp.reshape(q, (1, k), order="C")],
                      [cp.reshape(q, (k, 1), order="C"), Q]])
    z = cp.hstack([xi, q])
    lp = model.linear_lp
    row_indices, col_indices, values = [], [], []
    for i, row in enumerate(lp.rows):
        row_indices.extend([i] * len(row.indices))
        col_indices.extend(row.indices)
        values.extend(_proposal_float(value) for value in row.coefficients)
    A = coo_matrix((values, (row_indices, col_indices)), shape=(len(lp.rows), lp.column_count)).tocsr()
    constraints = [moment >> 0]
    expression = A @ z
    equal = [i for i, row in enumerate(lp.rows) if row.lower is not None and row.lower == row.upper]
    lower = [i for i, row in enumerate(lp.rows) if row.lower is not None and row.lower != row.upper]
    upper = [i for i, row in enumerate(lp.rows) if row.upper is not None and row.lower != row.upper]
    for indices, sense, bound in ((equal, "eq", "lower"), (lower, "lo", "lower"), (upper, "hi", "upper")):
        if indices:
            rhs = np.array([_proposal_float(getattr(lp.rows[i], bound)) for i in indices])
            constraints.append(expression[indices] == rhs if sense == "eq" else
                               expression[indices] >= rhs if sense == "lo" else expression[indices] <= rhs)
    for bounds, lower_bound in ((lp.column_lower, True), (lp.column_upper, False)):
        indices = [i for i, value in enumerate(bounds) if value is not None]
        if indices:
            rhs = np.array([_proposal_float(bounds[i]) for i in indices])
            constraints.append(z[indices] >= rhs if lower_bound else z[indices] <= rhs)
    for row in model.moment_equalities:
        expression = sum(_proposal_float(value) * Q[i, j] for i, j, value in row.terms)
        constraints.append(expression == _proposal_float(row.rhs))
    return cp.Problem(cp.Minimize(0), constraints), {"xi": xi, "q": q, "Q": Q, "M": moment}


def numerical_telemetry(model, xi, q, Q, *, solver=None, status=None, solver_stats=None):
    """Numerical diagnostics only; NEVER an exclusion or exact witness verdict."""
    result = {**model.summary(), "solver": solver, "solver_status": status,
              "primal_residual": None, "dual_residual": None, "objective": None,
              "infeasibility_metric": None, "moment_eigenvalues": None,
              "numerical_rank_estimate": None, "q_reconstruction_consistency": None,
              "final_status": "NUMERICAL_PROBE_ONLY_NO_SCIENTIFIC_AUTHORITY"}
    info = ((getattr(solver_stats, "extra_stats", None) or {}).get("info", {})
            if solver_stats is not None and isinstance(getattr(solver_stats, "extra_stats", None), dict) else {})
    def scalar(value):
        return float(value) if value is not None and np.isfinite(value) else None
    result.update(primal_residual=scalar(info.get("res_pri")), dual_residual=scalar(info.get("res_dual")),
                  objective=scalar(info.get("pobj")), infeasibility_metric=scalar(info.get("res_infeas")))
    if solver_stats is not None:
        result.update(solver_solve_seconds=scalar(solver_stats.solve_time),
                      solver_setup_seconds=scalar(solver_stats.setup_time), solver_iterations=solver_stats.num_iters)
    if xi is None or q is None or Q is None:
        result["primal_present"] = False
        return result
    xi, q, Q = np.asarray(xi), np.asarray(q), np.asarray(Q)
    k = model.q_dimension
    if xi.shape != (model.source_count,) or q.shape != (k,) or Q.shape != (k, k) or not all(
            np.isfinite(value).all() for value in (xi, q, Q)):
        result.update(primal_present=True, malformed_primal=True)
        return result
    M = np.block([[np.ones((1, 1)), q[None, :]], [q[:, None], Q]])
    eigenvalues = np.linalg.eigvalsh((M + M.T) * .5)
    tolerance = 1e-7 * max(1., float(np.max(np.abs(eigenvalues))))
    z = np.concatenate([xi, q])
    def row_violation(row):
        v = sum(float(a) * z[j] for j, a in zip(row.indices, row.coefficients))
        return max(0., 0. if row.lower is None else float(row.lower) - v,
                   0. if row.upper is None else v - float(row.upper))
    moment_residual = max((abs(sum(float(a) * Q[i, j] for i, j, a in row.terms) - float(row.rhs))
                           for row in model.moment_equalities), default=0.)
    bound_violation = max([0.] + [max(0., 0. if lo is None else float(lo) - z[i],
        0. if hi is None else z[i] - float(hi)) for i, (lo, hi) in enumerate(
            zip(model.linear_lp.column_lower, model.linear_lp.column_upper))])
    result.update(primal_present=True, malformed_primal=False,
        independently_measured_linear_violation=max(map(row_violation, model.linear_lp.rows), default=0.),
        independently_measured_bound_violation=bound_violation,
        independently_measured_moment_equality_residual=moment_residual,
        moment_eigenvalues=eigenvalues.tolist(), minimum_moment_eigenvalue=float(eigenvalues[0]),
        moment_symmetry_residual=float(np.max(np.abs(M - M.T))),
        numerical_rank_estimate=int(np.count_nonzero(eigenvalues > tolerance)),
        numerical_rank_tolerance=tolerance,
        q_reconstruction_consistency={"first_moment_row_max_error": float(np.max(np.abs(M[0, 1:] - q))),
            "source_relation_max_error": max((row_violation(row) for row in model.linear_lp.rows
                                              if row.name.startswith("centered[")), default=0.),
            "rank_one_gap_frobenius": float(np.linalg.norm(Q - np.outer(q, q)))})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--capture-root", type=Path)
    parser.add_argument("--downstream-report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.inventory:
        print(json.dumps(solver_inventory(), indent=2), flush=True)
        return 0
    if args.capture_root is None or args.downstream_report is None or args.output is None:
        parser.error("model construction requires --capture-root, --downstream-report, --output")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError("refusing to overwrite a model artifact")
    model = prepare_production(args.capture_root.expanduser().resolve(), args.downstream_report.expanduser().resolve())
    record = model.record()
    record["model_sha256"] = oracle._sha_json(record)
    record["solver_inventory"] = solver_inventory()
    oracle._atomic_json(output, record)
    print(json.dumps({"model_path": str(output), "model_sha256": record["model_sha256"],
                      "artifact_sha256": oracle.cluster_common.sha256(output),
                      "solver_execution": "NOT_RUN", **model.summary()}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
