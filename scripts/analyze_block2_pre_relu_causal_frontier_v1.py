#!/usr/bin/env python3
"""Locate the first authenticated pre-ReLU cancellation frontier.

All numerical solvers are proposal mechanisms.  FEASIBLE requires exact
shared-source replay; EXCLUDED requires an exactly replayed rational Farkas
certificate.  Every other outcome is UNRESOLVED.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from functools import lru_cache
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


relu = _module("authenticated_relu_oracle",
               REPO / "scripts/analyze_block2_relu_causal_oracle_v1.py")
frontier = relu.frontier
cluster_common = frontier.cluster_common

SCHEMA = "CORET_BLOCK2_PRE_RELU_CAUSAL_FRONTIER_V1"
FEASIBLE = "EXACT_RELU_CANCELLATION_FEASIBLE"
EXCLUDED = "EXACT_RELU_CANCELLATION_EXCLUDED"
UNRESOLVED = "EXACT_RELU_CANCELLATION_UNRESOLVED"
FFN_FIRST_PARAMETER = "bert.encoder.layer.2.intermediate.dense"

BOUNDARIES = (
    ("post_attention_ln_pre_reduction",
     "post-attention LayerNorm before generator reduction",
     "compose_w1", None),
    ("post_attention_ln_post_reduction",
     "post-attention LayerNorm after generator reduction",
     "compose_w1", "post_attention_layernorm_reduction"),
    # The capture has no first-affine-before-numerical-injection state.
    ("ffn_first_pre_reduction",
     "first FFN affine after numerical injection, before reduction",
     "direct", "first_affine_plus_numerical_injection_combined"),
    ("ffn_first_post_reduction",
     "first FFN affine after reduction; exact pre-ReLU input",
     "direct", "first_affine_generator_reduction"),
)


def _fraction(value) -> Fraction:
    return Fraction.from_float(float(value))


def _load_ffn_first_parameters(capture_identity: dict, *, source=None,
                               blob_loader=None):
    source = dict(frontier.DEFAULT_PARAMETER_SOURCE if source is None else source)
    blob_loader = blob_loader or frontier._git_blob_at_revision
    for capture_field, source_field in (
            ("pinned_deept_revision", "pinned_revision"),
            ("scientific_manifest_sha256", "scientific_manifest_sha256"),
            ("production_manifest_sha256", "production_manifest_sha256")):
        if capture_identity.get(capture_field) != source.get(source_field):
            raise RuntimeError(
                f"capture/parameter source identity differs: {capture_field}")
    checkpoint_raw = blob_loader(source["pinned_revision"],
                                 source["checkpoint_git_path"])
    config_raw = blob_loader(source["pinned_revision"],
                             source["config_git_path"])
    if hashlib.sha256(checkpoint_raw).hexdigest() != source["checkpoint_sha256"]:
        raise RuntimeError("authenticated checkpoint artifact SHA differs")
    if hashlib.sha256(config_raw).hexdigest() != source["config_sha256"]:
        raise RuntimeError("authenticated config artifact SHA differs")
    config = json.loads(config_raw)
    if (config.get("num_hidden_layers") != 3
            or config.get("hidden_size") != 128
            or config.get("intermediate_size") != 128):
        raise RuntimeError("authenticated Block-2 architecture differs")
    checkpoint = torch.load(io.BytesIO(checkpoint_raw), map_location="cpu",
                            weights_only=False)
    weight = checkpoint[f"{FFN_FIRST_PARAMETER}.weight"].detach().cpu().to(
        dtype=torch.float64).contiguous().numpy()
    bias = checkpoint[f"{FFN_FIRST_PARAMETER}.bias"].detach().cpu().to(
        dtype=torch.float64).contiguous().numpy()
    if (weight.shape != (128, 128) or bias.shape != (128,)
            or not np.isfinite(weight).all() or not np.isfinite(bias).all()):
        raise RuntimeError("authenticated Block-2 FFN first affine differs")
    return weight, bias, {
        "parameter_source": (
            f"git:{frontier.adapter.DEEPT_REPOSITORY}@"
            f"{source['pinned_revision']}:{source['checkpoint_git_path']}"),
        "parameter": FFN_FIRST_PARAMETER,
        "checkpoint_sha256": source["checkpoint_sha256"],
        "config_sha256": source["config_sha256"],
        "weight_shape": list(weight.shape), "bias_shape": list(bias.shape),
        "weight_sha256": frontier._array_sha(weight),
        "bias_sha256": frontier._array_sha(bias),
        "parameter_identity_authenticated": True,
    }


def _exact_state_coordinate(state: dict, source: int | None,
                            coordinate: int) -> Fraction:
    if source is None:
        return _fraction(state["center"][coordinate])
    return _fraction(state["generators"][source, coordinate])


def _compose_first_affine(state: dict, weight: np.ndarray, bias: np.ndarray):
    """Return numerical proposal state plus exact W1/b1 coefficient oracle."""
    numeric = {
        **state,
        "center": np.asarray(weight @ state["center"] + bias,
                             dtype=np.float64),
        "generators": np.asarray(state["generators"] @ weight.T,
                                 dtype=np.float64),
    }

    @lru_cache(maxsize=128)
    def center(output):
        return _fraction(bias[output]) + sum(
            (_fraction(weight[output, feature])
             * _exact_state_coordinate(state, None, feature)
             for feature in range(128)), Fraction(0))

    @lru_cache(maxsize=131072)
    def generator(source, output):
        return sum(
            (_fraction(weight[output, feature])
             * _exact_state_coordinate(state, source, feature)
             for feature in range(128)), Fraction(0))

    # Safe exact interval composition avoids eagerly forming 14k*128 exact
    # W1 coefficients.  Exact source correlation is still replayed by the
    # callbacks above for every accepted witness/certificate.
    x_lower, x_upper = relu._exact_affine_bounds(
        state["center"], state["generators"], state["low"], state["high"])
    lower, upper = [], []
    for output in range(128):
        lo = hi = _fraction(bias[output])
        for feature in range(128):
            coefficient = _fraction(weight[output, feature])
            first = coefficient * x_lower[feature]
            second = coefficient * x_upper[feature]
            lo += min(first, second); hi += max(first, second)
        lower.append(lo); upper.append(hi)
    identity = relu._sha_json({
        "operation": "exact_authenticated_w1_b1_composition",
        "input_center": frontier._array_sha(state["center"]),
        "input_generators": frontier._array_sha(state["generators"]),
        "weight": frontier._array_sha(weight), "bias": frontier._array_sha(bias),
    })
    exact = {"center": center, "generator": generator,
             "bounds": (lower, upper), "identity_sha256": identity}
    return numeric, exact


def _classify(problem, certificate_dir: Path, stem: str, lp_timeout: float,
              milp_timeout: float, exact_timeout: float,
              maximum_patterns: int, maximum_bases: int) -> dict:
    started = time.perf_counter()
    hull_result, hull = relu.solve_hull(problem, lp_timeout)
    farkas_verified = False
    witness_verified = False
    evidence_sha = None
    milp_status = None
    if not hull_result.success:
        certificate, diagnostics = relu.propose_farkas(problem, lp_timeout)
        hull["farkas_search"] = diagnostics
        if certificate is not None:
            path = certificate_dir / f"{stem}_farkas.json"
            persisted = relu._atomic_json(path, certificate)
            relu._verify_farkas(problem, relu._ratios(persisted["lambda"]),
                                relu._ratios(persisted["mu"]))
            farkas_verified = True
            evidence_sha = cluster_common.sha256(path)
            status = EXCLUDED
        else:
            status = UNRESOLVED
    else:
        status = UNRESOLVED
        excluded_patterns = []
        for ordinal in range(maximum_patterns):
            result, row = relu.solve_exact_relu_milp(
                problem, milp_timeout, excluded_patterns)
            milp_status = int(row["solver_status"])
            if not result.success:
                break
            pattern = [False] * problem.m
            for index in problem.active:
                pattern[index] = True
            offset = problem.n + len(problem.unstable)
            for local, index in enumerate(problem.unstable):
                pattern[index] = bool(round(result.x[offset + local]))
            candidate, _interior = relu.fixed_pattern_interior_candidate(
                problem, pattern)
            if candidate is not None:
                path = certificate_dir / f"{stem}_exact_witness.json"
                witness, _details = relu.exact_fixed_pattern_witness(
                    problem, candidate, pattern, path, exact_timeout,
                    maximum_bases=maximum_bases)
                if witness is not None:
                    relu.verify_exact_relu_witness(problem, witness)
                    witness_verified = True
                    evidence_sha = cluster_common.sha256(path)
                    status = FEASIBLE
                    break
            excluded_patterns.append(pattern)
    native = sum(reason not in frontier.NUMERICAL_REASONS
                 for reason in problem.source["reasons"])
    return {
        "source_count": problem.n, "native_count": native,
        "numerical_count": problem.n - native,
        "exact_relu_status": status,
        "hull_lp_status": int(hull.get("solver_status", hull_result.status)),
        "exact_farkas_verified": farkas_verified,
        "milp_status": milp_status,
        "exact_witness_verified": witness_verified,
        "witness_or_certificate_sha256": evidence_sha,
        "runtime_seconds": time.perf_counter() - started,
    }


def _causal_frontier(rows: list[dict]) -> dict:
    feasible = [row for row in rows if row["exact_relu_status"] == FEASIBLE]
    unresolved_predecessors = []
    if not feasible:
        return {"causal_frontier_found": False, "causal_transition": None,
                "transition_type": "PRE_RELU_CAUSAL_FRONTIER_UNRESOLVED",
                "repair_family_to_test_next": None,
                "earliest_feasible_state": None,
                "unresolved_predecessors": []}
    earliest = feasible[0]
    predecessors = [row for row in rows if row["order"] < earliest["order"]]
    unresolved_predecessors = [row["captured_state_name"] for row in predecessors
                               if row["exact_relu_status"] == UNRESOLVED]
    if earliest["order"] == 0:
        transition = "CAUSAL_FRONTIER_UPSTREAM_OF_CAPTURED_FFN_PATH"
        return {"causal_frontier_found": False, "causal_transition": None,
                "transition_type": transition,
                "repair_family_to_test_next": None,
                "earliest_feasible_state": earliest["captured_state_name"],
                "unresolved_predecessors": unresolved_predecessors}
    previous = rows[earliest["order"] - 1]
    if previous["exact_relu_status"] != EXCLUDED:
        return {"causal_frontier_found": False, "causal_transition": None,
                "transition_type": "PRE_RELU_CAUSAL_FRONTIER_UNRESOLVED",
                "repair_family_to_test_next": None,
                "earliest_feasible_state": earliest["captured_state_name"],
                "unresolved_predecessors": unresolved_predecessors}
    transition_type = earliest["incoming_transition_type"]
    repairs = {
        "post_attention_layernorm_reduction":
            "correlation-preserving reduction at post-attention LayerNorm",
        "first_affine_plus_numerical_injection_combined":
            ("combined-boundary correlation preservation; the capture cannot "
             "separate affine propagation from numerical injection"),
        "first_affine_generator_reduction":
            "correlation-preserving reduction at first-affine output",
    }
    return {
        "causal_frontier_found": True,
        "causal_transition": {
            "before": previous["captured_state_name"],
            "after": earliest["captured_state_name"]},
        "transition_type": transition_type,
        "repair_family_to_test_next": repairs[transition_type],
        "earliest_feasible_state": earliest["captured_state_name"],
        "unresolved_predecessors": unresolved_predecessors,
    }


def execute(manifest: Path, output: Path, certificate_dir: Path,
            lp_timeout: float, milp_timeout: float, exact_timeout: float,
            maximum_patterns: int, maximum_bases: int):
    started = time.perf_counter()
    payload, identity, token = frontier._load_capture(manifest)
    w1, b1, first_identity = _load_ffn_first_parameters(identity)
    w2, b2, second_identity = frontier._load_ffn_second_parameters(identity)
    available = list(payload["states"])
    missing = [{
        "requested_boundary": "first FFN affine before numerical injection",
        "captured": False,
        "reason": "job-2995 capture contains no separate pre-injection state",
    }]
    decoded = {name: frontier._decode_state(payload["states"][name], token)
               for name, *_rest in BOUNDARIES}
    residual = decoded["post_attention_ln_post_reduction"]
    rows = []
    certificate_dir.mkdir(parents=True, exist_ok=True)
    for order, (name, semantic, mode, incoming) in enumerate(BOUNDARIES):
        if mode == "compose_w1":
            h, exact = _compose_first_affine(decoded[name], w1, b1)
            current_residual = decoded[name]
        else:
            h, exact = decoded[name], None
            current_residual = residual
        problem = relu.ReluCancellationProblem(
            h, current_residual, w2, b2, label=name,
            exact_preactivation=exact)
        row = _classify(problem, certificate_dir, name, lp_timeout,
                        milp_timeout, exact_timeout, maximum_patterns,
                        maximum_bases)
        rows.append({"order": order, "captured_state_name": name,
                     "semantic_boundary": semantic,
                     "incoming_transition_type": incoming,
                     "preactivation_mode": mode, **row})
        print(f"{name}: {row['exact_relu_status']}", flush=True)
    result = {
        "schema": SCHEMA, "manifest_authentication": identity,
        "analysis_token_index": token,
        "actual_captured_state_names": available,
        "tested_upstream_states": rows, "missing_boundaries": missing,
        "ffn_first_parameter_authentication": first_identity,
        "ffn_second_parameter_authentication": second_identity,
        **_causal_frontier(rows),
        "runtime_seconds": time.perf_counter() - started,
        "scientific_queries": 0, "bound_calls": 0, "gpu_jobs": 0,
    }
    return relu._atomic_json(output, result)


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
    execute(args.manifest.resolve(), args.output.resolve(),
            args.certificate_dir.resolve(), args.lp_timeout_seconds,
            args.milp_timeout_seconds, args.exact_solve_timeout_seconds,
            args.maximum_patterns, args.maximum_bases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
