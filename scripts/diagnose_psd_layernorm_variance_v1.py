#!/usr/bin/env python3
"""Certified PSD-aware LayerNorm variance diagnostic.

This file is deliberately separate from the production verifier.  It either
audits whether stopped-pilot coefficient states were persisted, or evaluates
an explicit hash-bound manifest of such states.  The optimizer is untrusted;
acceptance comes only from :func:`recheck_dual_witness`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import cluster_common


ORACLE_SCHEMA = "CORET_PSD_LAYERNORM_VARIANCE_ORACLE_V1"
INPUT_SCHEMA = "CORET_PSD_LAYERNORM_VARIANCE_INPUT_V1"
INVENTORY_SCHEMA = "CORET_PSD_LAYERNORM_VARIANCE_INVENTORY_V1"
PROPERTY_ID = "deept_table7_stdln3_s001_line1794_tok11"
PINNED_REVISION = "16ffe4075f1f8a7c87fa2a187d8c46cfd51e07bf"
SOURCE_SET_MODEL = "p100_linf_shared_generator_ids_cartesian_ranges"
MULTIPLIERS = ("0.95", "0.90", "0.75", "0.50", "0.25")
NUMERICAL_REASONS = {
    "fp64_roundoff_coordinate_box",
    "sound_fp64_coordinate_box_replacement_with_numerical",
}
PRECISION = 100
U = 2.0 ** -53


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    payload = dict(value)
    payload["record_sha256"] = cluster_common.canonical(payload)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _decimal(value: float) -> Decimal:
    return Decimal.from_float(float(value))


def _outward_float(value: Decimal, upward: bool) -> float:
    result = float(value)
    exact = Decimal.from_float(result)
    if (upward and exact < value) or (not upward and exact > value):
        result = math.nextafter(result, math.inf if upward else -math.inf)
    return result


def _gamma(operations: int) -> float:
    value = operations * U
    if value >= 1.0:
        raise RuntimeError("dot-product error model is outside its domain")
    return value / (1.0 - value)


def center_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    if vector.ndim != 1:
        raise RuntimeError("centering expects one vector")
    return vector - vector.mean()


def dual_objective(center: np.ndarray, generators: np.ndarray,
                   low: np.ndarray, high: np.ndarray,
                   witness: np.ndarray) -> dict:
    """Untrusted binary64 objective used only to find a useful witness."""
    center = np.asarray(center, dtype=np.float64)
    generators = np.asarray(generators, dtype=np.float64)
    low, high = np.asarray(low, dtype=np.float64), np.asarray(high,
                                                                    dtype=np.float64)
    witness = np.asarray(witness, dtype=np.float64)
    y_projected = center_vector(witness)
    b = center_vector(center)
    q = generators @ y_projected if len(generators) else np.empty(0)
    minimizing = np.where(q >= 0.0, low, high)
    center_term = 2.0 * float(np.dot(witness, b))
    quadratic_term = float(np.dot(witness, witness))
    minimum_linear = float(np.dot(q, minimizing)) if len(q) else 0.0
    value = (center_term - quadratic_term + 2.0 * minimum_linear) / len(center)
    return {
        "objective": value,
        "center_term": center_term,
        "quadratic_term": quadratic_term,
        "minimum_linear_term": minimum_linear,
        "support_term": -minimum_linear,
    }


def optimize_witness(center: np.ndarray, generators: np.ndarray,
                     low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, dict]:
    """Find a candidate dual vector; no soundness claim depends on this call."""
    center = np.asarray(center, dtype=np.float64)
    generators = np.asarray(generators, dtype=np.float64)
    low, high = np.asarray(low, dtype=np.float64), np.asarray(high,
                                                                    dtype=np.float64)
    b = center_vector(center)

    def fun_gradient(candidate):
        y = center_vector(np.asarray(candidate, dtype=np.float64))
        q = generators @ y if len(generators) else np.empty(0)
        minimizing = np.where(q >= 0.0, low, high)
        support_vector = (generators.T @ minimizing
                          if len(generators) else np.zeros_like(y))
        support_vector = center_vector(support_vector)
        objective = (2.0 * np.dot(y, b) - np.dot(y, y)
                     + 2.0 * (np.dot(q, minimizing) if len(q) else 0.0))
        gradient = 2.0 * (b - y + support_vector)
        dimension = len(center)
        return -float(objective / dimension), -gradient / dimension

    result = minimize(fun_gradient, b, method="L-BFGS-B", jac=True,
                      options={"maxiter": 1000, "ftol": 1e-14,
                               "gtol": 1e-10, "maxls": 50})
    witness = center_vector(np.asarray(result.x, dtype=np.float64))
    return witness, {
        "success": bool(result.success), "status": int(result.status),
        "message": str(result.message), "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "optimizer_objective": -float(result.fun),
    }


def _projected_decimal(vector: np.ndarray, rounding: str) -> list[Decimal]:
    with localcontext() as context:
        context.prec = PRECISION
        context.rounding = rounding
        values = [_decimal(value) for value in vector]
        mean = sum(values, Decimal(0)) / Decimal(len(values))
        return [value - mean for value in values]


def _projected_float_interval(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Compute P*vector with independent directed Decimal passes.
    lower_decimal = _projected_decimal(vector, ROUND_FLOOR)
    upper_decimal = _projected_decimal(vector, ROUND_CEILING)
    lower = np.array([_outward_float(value, False)
                      for value in lower_decimal], dtype=np.float64)
    upper = np.array([_outward_float(value, True)
                      for value in upper_decimal], dtype=np.float64)
    if np.any(lower > upper):
        raise RuntimeError("projected witness interval is malformed")
    return lower, upper


def _dot_intervals(generators: np.ndarray, witness: np.ndarray):
    if not len(generators):
        return np.empty(0), np.empty(0)
    py_low, py_high = _projected_float_interval(witness)
    midpoint = py_low + 0.5 * (py_high - py_low)
    radius = np.maximum(midpoint - py_low, py_high - midpoint)
    q_hat = generators @ midpoint
    gamma = _gamma(2 * len(witness) + 2)
    absolute = np.abs(generators)
    scale_hat = absolute @ np.abs(midpoint)
    input_hat = absolute @ radius
    # The positive BLAS reductions may round down.  Divide by (1-gamma) to
    # upper-bound their exact sums, then include dot realization error.
    scale_upper = np.nextafter(scale_hat / (1.0 - gamma), math.inf)
    input_upper = np.nextafter(input_hat / (1.0 - gamma), math.inf)
    error = np.nextafter(
        gamma * scale_upper + input_upper
        + len(witness) * np.nextafter(0.0, 1.0), math.inf)
    if not all(np.isfinite(item).all() for item in
               (q_hat, scale_upper, input_upper, error)):
        raise RuntimeError("dot-product certificate overflow")
    return (np.nextafter(q_hat - error, -math.inf),
            np.nextafter(q_hat + error, math.inf))


def recheck_dual_witness(center: np.ndarray, generators: np.ndarray,
                         low: np.ndarray, high: np.ndarray,
                         witness: np.ndarray) -> dict:
    """Independently recheck weak duality with directed/outward arithmetic.

    The state and witness entries are interpreted as exact IEEE-754 binary64
    values.  Projection/short sums use directed Decimal arithmetic.  Long
    generator dot products use the standard gamma bound plus interval input
    propagation; the adverse range support is then summed upward in Decimal.
    """
    center = np.asarray(center, dtype=np.float64)
    generators = np.asarray(generators, dtype=np.float64)
    low, high = np.asarray(low, dtype=np.float64), np.asarray(high,
                                                                    dtype=np.float64)
    witness = np.asarray(witness, dtype=np.float64)
    if (center.ndim != 1 or generators.ndim != 2
            or generators.shape[1:] != center.shape
            or low.shape != (len(generators),)
            or high.shape != low.shape or witness.shape != center.shape):
        raise RuntimeError("dual witness shape differs")
    if (center.size == 0 or not all(np.isfinite(item).all() for item in
            (center, generators, low, high, witness))
            or np.any(low > high)):
        raise RuntimeError("dual witness input is malformed")

    q_low, q_high = _dot_intervals(generators, witness)
    with localcontext() as down:
        down.prec = PRECISION
        down.rounding = ROUND_FLOOR
        c_values = [_decimal(value) for value in center]
        y_values = [_decimal(value) for value in witness]
        c_mean = sum(c_values, Decimal(0)) / Decimal(len(center))
        center_dot_lower = sum(
            (y * (c - c_mean) for y, c in zip(y_values, c_values)),
            Decimal(0))
        center_term_lower = Decimal(2) * center_dot_lower

    with localcontext() as up:
        up.prec = PRECISION
        up.rounding = ROUND_CEILING
        quadratic_upper = sum((value * value for value in y_values),
                              Decimal(0))
        adverse_terms = []
        for ql, qh, lower, upper in zip(q_low, q_high, low, high):
            candidates = [
                -_decimal(q) * _decimal(bound)
                for q in (ql, qh) for bound in (lower, upper)]
            adverse_terms.append(max(candidates))
        adverse_support_upper = sum(adverse_terms, Decimal(0))

    with localcontext() as down:
        down.prec = PRECISION
        down.rounding = ROUND_FLOOR
        numerator_lower = (center_term_lower - quadratic_upper
                           - Decimal(2) * adverse_support_upper)
        objective_lower = numerator_lower / Decimal(len(center))

    outward = _outward_float(objective_lower, False)
    if Decimal.from_float(outward) > objective_lower:
        raise RuntimeError("outward lower conversion failed")
    return {
        "certificate_recheck_objective_decimal": str(objective_lower),
        "certificate_recheck_objective": float(objective_lower),
        "psd_dual_candidate_lower_outward_safe": outward,
        "center_term_lower_decimal": str(center_term_lower),
        "quadratic_term_upper_decimal": str(quadratic_upper),
        "support_term_upper_decimal": str(adverse_support_upper),
        "witness_l2_norm": float(np.linalg.norm(witness)),
        "source_set_range_model": (
            "cartesian_product_of_authenticated_per_generator_[low,high]; "
            "adverse_support=h_K(-A^T y)"),
    }


def brute_force_minimum(center: np.ndarray, generators: np.ndarray,
                        low: np.ndarray, high: np.ndarray,
                        points: int = 101) -> float:
    """Small-test grid oracle.  Deliberately rejects production dimensions."""
    if len(generators) > 3:
        raise RuntimeError("grid oracle is restricted to at most three generators")
    axes = [np.linspace(lower, upper, points)
            for lower, upper in zip(low, high)]
    best = math.inf
    for index in np.ndindex(*(len(axis) for axis in axes)):
        coefficients = np.array([axes[i][item]
                                 for i, item in enumerate(index)])
        value = center + (coefficients @ generators
                          if len(generators) else 0.0)
        variance = float(np.dot(center_vector(value), center_vector(value))
                         / len(center))
        best = min(best, variance)
    if not axes:
        value = center_vector(center)
        best = float(np.dot(value, value) / len(center))
    return best


def _load_torch_state(path: Path, expected_sha: str, expected_schema: str,
                      state_key: str | None):
    if sha256(path) != expected_sha:
        raise RuntimeError(f"state artifact SHA differs: {path}")
    import torch  # Artifact decoding only; no CUDA operation is reachable.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != expected_schema:
        raise RuntimeError("state artifact schema differs")
    if payload.get("pinned_revision") != PINNED_REVISION:
        raise RuntimeError("state artifact pinned revision differs")
    state = payload
    if state_key is not None:
        states = payload.get("states")
        if not isinstance(states, dict) or state_key not in states:
            raise RuntimeError(f"state key is absent: {state_key}")
        state = states[state_key]
    required = {"weights", "range_low", "range_high", "proof"}
    if not isinstance(state, dict) or not required <= set(state):
        raise RuntimeError("serialized state fields differ")
    weights = state["weights"]
    low, high = state["range_low"], state["range_high"]
    proof = state["proof"]
    if (weights.device.type != "cpu" or weights.dtype != torch.float64
            or low.dtype != torch.float64 or high.dtype != torch.float64):
        raise RuntimeError("serialized state must be CPU float64")
    if weights.ndim != 3 or weights.shape[-1] != 128:
        raise RuntimeError("serialized state is not a hidden-state zonotope")
    generator_count = int(weights.shape[0] - 1)
    if (tuple(low.shape) != (generator_count,)
            or tuple(high.shape) != (generator_count,)
            or len(proof.get("ids", ())) != generator_count
            or len(proof.get("masks", ())) != generator_count
            or len(proof.get("reasons", ())) != generator_count
            or int(proof.get("num_tokens", -1)) != weights.shape[1]
            or len(set(proof["ids"])) != generator_count
            or not bool(torch.isfinite(weights).all()
                        and torch.isfinite(low).all()
                        and torch.isfinite(high).all())
            or bool((low > high).any())):
        raise RuntimeError("serialized state topology/ranges are unauthenticated")
    return {
        "weights": weights.numpy(), "low": low.numpy(), "high": high.numpy(),
        "ids": list(proof["ids"]), "reasons": list(proof["reasons"]),
        "num_tokens": int(proof["num_tokens"]), "artifact_sha256": expected_sha,
        "artifact_schema": payload.get("schema"),
    }


def _variant(state: dict, token: int, native_only: bool) -> dict:
    if token < 0 or token >= state["num_tokens"]:
        raise RuntimeError("diagnostic token is outside the state")
    selected = np.arange(len(state["ids"]), dtype=np.int64)
    if native_only:
        selected = np.array([
            index for index, reason in enumerate(state["reasons"])
            if reason not in NUMERICAL_REASONS], dtype=np.int64)
    weights = state["weights"]
    chosen_reasons = [state["reasons"][index] for index in selected]
    return {
        "center": weights[0, token].copy(),
        "generators": weights[1:, token][selected].copy(),
        "low": state["low"][selected].copy(),
        "high": state["high"][selected].copy(),
        "ids": [state["ids"][index] for index in selected],
        "reasons": chosen_reasons,
        "native_generator_count": sum(
            reason not in NUMERICAL_REASONS for reason in chosen_reasons),
        "numerical_generator_count": sum(
            reason in NUMERICAL_REASONS for reason in chosen_reasons),
    }


def _witness_hash(header: dict, witness: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(header, sort_keys=True,
                             separators=(",", ":")).encode())
    digest.update(np.ascontiguousarray(witness, dtype="<f8").tobytes())
    return digest.hexdigest()


def evaluate_variant(record: dict, name: str, state: dict,
                     native_only: bool) -> dict:
    token = int(record["token_index"])
    variant = _variant(state, token, native_only)
    witness, optimizer = optimize_witness(
        variant["center"], variant["generators"],
        variant["low"], variant["high"])
    candidate = dual_objective(
        variant["center"], variant["generators"], variant["low"],
        variant["high"], witness)
    checked = recheck_dual_witness(
        variant["center"], variant["generators"], variant["low"],
        variant["high"], witness)
    header = {
        "property_id": record["property_id"], "multiplier": record["multiplier"],
        "variant": name, "token_index": token,
        "state_artifact_sha256": state["artifact_sha256"],
        "generator_ids_sha256": hashlib.sha256(json.dumps(
            variant["ids"], separators=(",", ":")).encode()).hexdigest(),
    }
    return {
        **header,
        "tested_radius": float(record["tested_radius"]),
        "generator_count": len(variant["generators"]),
        "native_generator_count": variant["native_generator_count"],
        "numerical_generator_count": variant["numerical_generator_count"],
        "old_generic_variance_lower": (
            float(record["old_generic_variance_lower"])
            if name == "complete" else None),
        "old_sound_variance_lower": (
            float(record["old_sound_variance_lower"])
            if name == "complete" else None),
        "old_bound_scope": (
            "authenticated_stopped_pilot_complete_state"
            if name == "complete" else
            "not_persisted_for_diagnostic_projection"),
        "psd_dual_candidate_lower": candidate["objective"],
        **checked,
        "psd_positive": checked["psd_dual_candidate_lower_outward_safe"] > 0,
        "optimizer_objective": optimizer["optimizer_objective"],
        "optimizer": optimizer,
        "optimizer_vs_checker_gap": (
            optimizer["optimizer_objective"]
            - checked["psd_dual_candidate_lower_outward_safe"]),
        "center_term": candidate["center_term"],
        "quadratic_term": candidate["quadratic_term"],
        "support_term": candidate["support_term"],
        "every_generator_range_authenticated": True,
        "state_artifact_hash": state["artifact_sha256"],
        "state_artifact_schema": state["artifact_schema"],
        "witness_hash": _witness_hash(header, witness),
        "claim_scope": ("diagnostic_native_only_subset_not_complete_state"
                        if native_only else "complete_persisted_state"),
    }


def evaluate_manifest(path: Path) -> dict:
    manifest = cluster_common.verified_json(path)
    if manifest.get("schema") != INPUT_SCHEMA:
        raise RuntimeError("PSD oracle input schema differs")
    if manifest.get("property_id") != PROPERTY_ID:
        raise RuntimeError("PSD oracle property identity differs")
    if (manifest.get("pinned_revision") != PINNED_REVISION
            or manifest.get("source_set_model") != SOURCE_SET_MODEL):
        raise RuntimeError("PSD oracle source-domain identity differs")
    records = manifest.get("evaluations")
    if not isinstance(records, list) or [row.get("multiplier") for row in records] != list(MULTIPLIERS):
        raise RuntimeError("PSD oracle multiplier population differs")
    outputs = []
    for record in records:
        source_spec = record.get("source_result")
        if not isinstance(source_spec, dict):
            raise RuntimeError("source pilot result identity is absent")
        source_path = (path.parent / source_spec["path"]).resolve()
        if sha256(source_path) != source_spec["sha256"]:
            raise RuntimeError("source pilot result SHA differs")
        source = cluster_common.verified_json(source_path)
        multiplier = source.get("multiplier", {}).get("display")
        diagnostic = source.get("layernorm_domain_diagnostic")
        if (source.get("property_id") != PROPERTY_ID
                or multiplier != record["multiplier"]
                or not isinstance(diagnostic, dict)
                or diagnostic.get("domain_admissible") is not False
                or int(record["token_index"])
                != int(diagnostic["minimum_token_index"])
                or float(record["tested_radius"])
                != float(source["tested_radius"])):
            raise RuntimeError("source pilot result semantics differ")
        record = dict(record)
        record["old_generic_variance_lower"] = float(
            diagnostic["sound_variance_lower"])
        record["old_sound_variance_lower"] = float(
            diagnostic["sound_variance_lower"])
        complete_spec = record.get("complete_state")
        if not isinstance(complete_spec, dict):
            raise RuntimeError("complete-state artifact is absent")
        complete = _load_torch_state(
            (path.parent / complete_spec["path"]).resolve(),
            complete_spec["sha256"], complete_spec["schema"],
            complete_spec.get("state_key"))
        if (complete["num_tokens"] != int(diagnostic["token_count"])
                or len(complete["ids"]) != int(diagnostic["generator_count"])):
            raise RuntimeError("complete-state diagnostic topology differs")
        outputs.append(evaluate_variant(record, "complete", complete, False))
        outputs.append(evaluate_variant(record, "native_only", complete, True))
        pre = record.get("pre_reduction_state")
        if pre is not None:
            authenticated = _load_torch_state(
                (path.parent / pre["path"]).resolve(), pre["sha256"],
                pre["schema"],
                pre.get("state_key"))
            outputs.append(evaluate_variant(
                record, "authenticated_pre_reduction", authenticated, False))
    return {
        "schema": ORACLE_SCHEMA, "property_id": PROPERTY_ID,
        "input_manifest_sha256": sha256(path), "results": outputs,
        "optimizer_is_untrusted": True,
        "certificate_semantics": (
            "directed/outward weak-duality lower bound over authenticated ranges"),
        "scientific_queries": 0, "bound_calls": 0,
    }


def inventory(pilot_root: Path, property_id: str) -> dict:
    if property_id != PROPERTY_ID:
        raise RuntimeError("inventory property identity differs")
    evaluation_root = pilot_root / "properties" / property_id / "evaluations"
    records = sorted(evaluation_root.glob("*.json"))
    rows = [cluster_common.verified_json(path) for path in records]
    multipliers = [row.get("multiplier", {}).get("display") for row in rows]
    tensors = sorted(pilot_root.glob(f"**/{property_id}/**/*.pt"))
    tensor_rows = [{"path": str(path), "sha256": sha256(path),
                    "byte_count": path.stat().st_size} for path in tensors]
    return {
        "schema": INVENTORY_SCHEMA, "property_id": property_id,
        "pilot_root": str(pilot_root), "persisted_evaluation_count": len(rows),
        "persisted_multipliers": multipliers,
        "expected_multipliers": list(MULTIPLIERS),
        "coefficient_state_artifacts": tensor_rows,
        "coefficient_states_available": bool(tensor_rows),
        "primary_experiment_status": (
            "READY_FOR_HASH_BOUND_EVALUATION" if tensor_rows else
            "BLOCKED_MISSING_PERSISTED_COEFFICIENT_TENSORS"),
        "persistence_explanation": (
            "campaign execute_property deletes its workspace after each result; "
            "a LayerNorm domain failure returns before certificate.pt is saved"),
        "scientific_queries": 0, "bound_calls": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--manifest", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    inspect = subparsers.add_parser("inventory")
    inspect.add_argument("--pilot-root", required=True, type=Path)
    inspect.add_argument("--property-id", default=PROPERTY_ID)
    inspect.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "evaluate":
        result = evaluate_manifest(args.manifest.resolve())
    else:
        result = inventory(args.pilot_root.resolve(), args.property_id)
    _atomic_json(args.output.resolve(), result)
    print(json.dumps(cluster_common.verified_json(args.output.resolve()),
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
