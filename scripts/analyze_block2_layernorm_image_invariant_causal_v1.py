#!/usr/bin/env python3
"""Exact LayerNorm-image-invariant causal oracle for the Block-2 frontier."""
from __future__ import annotations

import argparse
from fractions import Fraction
from functools import lru_cache
import hashlib
import importlib.util
import io
import json
import os
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


prefrontier = _module(
    "pre_relu_frontier_authenticated",
    REPO / "scripts/analyze_block2_pre_relu_causal_frontier_v1.py")
relu = prefrontier.relu
frontier = prefrontier.frontier
cluster_common = frontier.cluster_common

SCHEMA = "CORET_BLOCK2_LAYERNORM_IMAGE_INVARIANT_CAUSAL_V1"
STATUS_EXCLUDES = "LAYERNORM_IMAGE_INVARIANT_EXCLUDES_CANCELLATION"
STATUS_FEASIBLE = "LAYERNORM_IMAGE_INVARIANT_CANCELLATION_FEASIBLE"
STATUS_INCONCLUSIVE = "LAYERNORM_IMAGE_INVARIANT_ORACLE_INCONCLUSIVE"
LN_PARAMETER = "bert.encoder.layer.2.attention.output.LayerNorm"
CAPTURE_STATE = "post_attention_ln_pre_reduction"
EXPECTED_STAGE = "block2_post_attention"
EXPECTED_LAYERNORM_INDEX = 5
TESTED_RADIUS = 0.00060791015625


def _fraction(value) -> Fraction:
    return Fraction.from_float(float(value))


def _load_layernorm_parameters(capture_identity: dict, *, source=None,
                               blob_loader=None):
    source = dict(frontier.DEFAULT_PARAMETER_SOURCE if source is None else source)
    blob_loader = blob_loader or frontier._git_blob_at_revision
    for capture_field, source_field in (
            ("pinned_deept_revision", "pinned_revision"),
            ("scientific_manifest_sha256", "scientific_manifest_sha256"),
            ("production_manifest_sha256", "production_manifest_sha256")):
        if capture_identity.get(capture_field) != source.get(source_field):
            raise RuntimeError(
                f"capture/LayerNorm parameter identity differs: {capture_field}")
    checkpoint_raw = blob_loader(source["pinned_revision"],
                                 source["checkpoint_git_path"])
    config_raw = blob_loader(source["pinned_revision"],
                             source["config_git_path"])
    if hashlib.sha256(checkpoint_raw).hexdigest() != source["checkpoint_sha256"]:
        raise RuntimeError("authenticated checkpoint artifact SHA differs")
    if hashlib.sha256(config_raw).hexdigest() != source["config_sha256"]:
        raise RuntimeError("authenticated config artifact SHA differs")
    config = json.loads(config_raw)
    if config.get("num_hidden_layers") != 3 or config.get("hidden_size") != 128:
        raise RuntimeError("authenticated LayerNorm architecture differs")
    checkpoint = torch.load(io.BytesIO(checkpoint_raw), map_location="cpu",
                            weights_only=False)
    gamma = checkpoint[f"{LN_PARAMETER}.weight"].detach().cpu().to(
        dtype=torch.float64).contiguous().numpy()
    beta = checkpoint[f"{LN_PARAMETER}.bias"].detach().cpu().to(
        dtype=torch.float64).contiguous().numpy()
    if (gamma.shape != (128,) or beta.shape != (128,)
            or not np.isfinite(gamma).all() or not np.isfinite(beta).all()):
        raise RuntimeError("authenticated post-attention LayerNorm differs")
    return gamma, beta, {
        "parameter_source": (
            f"git:{frontier.adapter.DEEPT_REPOSITORY}@"
            f"{source['pinned_revision']}:{source['checkpoint_git_path']}"),
        "parameter": LN_PARAMETER,
        "gamma_shape": list(gamma.shape), "beta_shape": list(beta.shape),
        "gamma_sha256": frontier._array_sha(gamma),
        "beta_sha256": frontier._array_sha(beta),
        "zero_gamma_count": int(np.count_nonzero(gamma == 0.0)),
        "pinned_revision": source["pinned_revision"],
        "checkpoint_sha256": source["checkpoint_sha256"],
        "config_sha256": source["config_sha256"],
        "parameter_identity_authenticated": True,
    }


def _layernorm_invariant(state: dict, gamma: np.ndarray, beta: np.ndarray):
    if state["center"].shape != gamma.shape or beta.shape != gamma.shape:
        raise RuntimeError("LayerNorm invariant dimensions differ")
    nonzero = [int(index) for index in np.flatnonzero(gamma != 0.0)]
    zero = [int(index) for index in np.flatnonzero(gamma == 0.0)]
    rows = [("weighted_mean_zero", tuple(nonzero))]
    rows.extend((f"zero_gamma_{index}", (index,)) for index in zero)

    @lru_cache(maxsize=256)
    def exact_constant(row):
        label, coordinates = rows[row]
        if label == "weighted_mean_zero":
            return sum(((_fraction(state["center"][index])
                         - _fraction(beta[index])) / _fraction(gamma[index])
                        for index in coordinates), Fraction(0))
        index = coordinates[0]
        return _fraction(state["center"][index]) - _fraction(beta[index])

    @lru_cache(maxsize=131072)
    def exact_coefficient(row, source):
        label, coordinates = rows[row]
        if label == "weighted_mean_zero":
            return sum((_fraction(state["generators"][source, index])
                        / _fraction(gamma[index]) for index in coordinates),
                       Fraction(0))
        return _fraction(state["generators"][source, coordinates[0]])

    numeric_A = np.empty((len(rows), len(state["ids"])), dtype=np.float64)
    numeric_b = np.empty(len(rows), dtype=np.float64)
    for row in range(len(rows)):
        numeric_b[row] = float(exact_constant(row))
        for source in range(len(state["ids"])):
            numeric_A[row, source] = float(exact_coefficient(row, source))
    digest = hashlib.sha256()
    for row in range(len(rows)):
        values = [exact_constant(row)] + [
            exact_coefficient(row, source) for source in range(len(state["ids"]))]
        for value in values:
            digest.update(f"{value.numerator}/{value.denominator};".encode())
    identity = relu._sha_json({
        "semantic_operator": "exact_standard_layernorm_image",
        "state_center": frontier._array_sha(state["center"]),
        "state_generators": frontier._array_sha(state["generators"]),
        "gamma": frontier._array_sha(gamma), "beta": frontier._array_sha(beta),
        "rows": rows, "exact_coefficients_sha256": digest.hexdigest(),
    })
    return {
        "numeric_A": numeric_A, "numeric_b": numeric_b,
        "exact_constant": exact_constant,
        "exact_coefficient": exact_coefficient,
        "identity_sha256": identity,
        "exact_coefficients_sha256": digest.hexdigest(),
        "nonzero_gamma_count": len(nonzero),
        "zero_gamma_count": len(zero), "equality_count": len(rows),
    }


def _invariant_residual(invariant: dict, values: list[Fraction]):
    return [
        invariant["exact_constant"](row) + sum(
            (invariant["exact_coefficient"](row, source) * xi
             for source, xi in enumerate(values)), Fraction(0))
        for row in range(invariant["equality_count"])]


def _read_candidate_from_experiment(frontier_manifest: Path) -> Path | None:
    report_path = frontier_manifest.parent / "experiment_report.json"
    if not report_path.is_file():
        return None
    report = cluster_common.verified_json(report_path)
    if (report.get("schema") != "CORET_PSD_LAYERNORM_EXPERIMENT_JOB_V2"
            or report.get("property_id") != frontier.zero.PROPERTY_ID
            or float(report.get("tested_radius", float("nan"))) !=
            TESTED_RADIUS):
        raise RuntimeError("PSD experiment report identity differs")
    record = report.get("capture_manifest")
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        return None
    path = Path(record["path"]).expanduser()
    if not path.is_absolute():
        path = (report_path.parent / path).resolve()
    if record.get("sha256") != cluster_common.sha256(path):
        raise RuntimeError("PSD input capture manifest SHA differs")
    return path


def _compatible_source_subset(input_state: dict, output_state: dict) -> tuple[bool, str]:
    output_index = {item: index for index, item in enumerate(output_state["ids"])}
    if len(output_index) != len(output_state["ids"]):
        return False, "output source IDs are not unique"
    for index, source_id in enumerate(input_state["ids"]):
        target = output_index.get(source_id)
        if target is None:
            return False, f"input source ID is absent from output: {source_id}"
        left = (float(input_state["low"][index]).hex(),
                float(input_state["high"][index]).hex(),
                input_state["masks"][index], input_state["reasons"][index])
        right = (float(output_state["low"][target]).hex(),
                 float(output_state["high"][target]).hex(),
                 output_state["masks"][target], output_state["reasons"][target])
        if left != right:
            return False, f"shared source metadata differs: {source_id}"
    return True, "authenticated LayerNorm input sources are preserved in output"


def _inspect_reusable_input(frontier_manifest: Path, capture_identity: dict,
                            output_state: dict, explicit: Path | None = None):
    candidate = explicit or _read_candidate_from_experiment(frontier_manifest)
    empty = {"inspected": True, "found": False, "artifact_path": None,
             "artifact_sha256": None, "stage_identity_verified": False,
             "source_identity_compatible": False}
    if candidate is None or not candidate.is_file():
        return {**empty, "reason":
                "LAYERNORM_EXACT_GRAPH_NEEDS_ONE_PASSIVE_INPUT_CAPTURE"}
    capture = _module(
        "psd_layernorm_input_capture_authenticated",
        REPO / "scripts/run_sound_fp64_3l_psd_state_capture_v1.py")
    manifest = capture.verify_capture(candidate)
    if (manifest.get("property_id") != capture_identity["property_id"]
            or float(manifest.get("tested_radius")) !=
            float(capture_identity["tested_radius"])
            or manifest.get("stage_label") != EXPECTED_STAGE
            or manifest.get("pinned_deept_revision") !=
            capture_identity["pinned_deept_revision"]):
        return {**empty, "artifact_path": str(candidate),
                "artifact_sha256": cluster_common.sha256(candidate),
                "reason": "pre-LayerNorm capture stage/property identity differs"}
    artifact = (candidate.parent / manifest["tensor_artifact_path"]).resolve()
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    state = frontier._decode_state(
        payload["states"]["post_last_reduction"],
        int(capture_identity["minimum_token_index"]))
    compatible, reason = _compatible_source_subset(state, output_state)
    return {"inspected": True, "found": compatible,
            "artifact_path": str(artifact),
            "artifact_sha256": manifest["tensor_artifact_sha256"],
            "capture_manifest_path": str(candidate),
            "capture_manifest_sha256": cluster_common.sha256(candidate),
            "stage_identity_verified": True,
            "source_identity_compatible": compatible, "reason": reason}


def execute(manifest_path: Path, output: Path, certificate_dir: Path,
            pre_layernorm_manifest: Path | None, lp_timeout: float,
            milp_timeout: float, exact_timeout: float,
            maximum_patterns: int, maximum_bases: int):
    started = time.perf_counter()
    payload, identity, token = frontier._load_capture(manifest_path)
    gamma, beta, ln_identity = _load_layernorm_parameters(identity)
    w1, b1, w1_identity = prefrontier._load_ffn_first_parameters(identity)
    w2, b2, w2_identity = frontier._load_ffn_second_parameters(identity)
    y = frontier._decode_state(payload["states"][CAPTURE_STATE], token)
    invariant = _layernorm_invariant(y, gamma, beta)
    h, exact_h = prefrontier._compose_first_affine(y, w1, b1)
    problem = relu.ReluCancellationProblem(
        h, y, w2, b2, label="layernorm_image_invariant",
        exact_preactivation=exact_h, additional_equalities=invariant)
    certificate_dir.mkdir(parents=True, exist_ok=True)
    decision = prefrontier._classify(
        problem, certificate_dir, "layernorm_image_invariant",
        lp_timeout, milp_timeout, exact_timeout,
        maximum_patterns, maximum_bases)
    farkas_path = certificate_dir / "layernorm_image_invariant_farkas.json"
    witness_path = certificate_dir / "layernorm_image_invariant_exact_witness.json"
    witness = (cluster_common.verified_json(witness_path)
               if witness_path.is_file() else None)
    if decision["exact_relu_status"] == prefrontier.EXCLUDED:
        final_status = STATUS_EXCLUDES
        interpretation = "POST_ATTENTION_LAYERNORM_ABSTRACTION_CAUSALLY_RESPONSIBLE"
    elif decision["exact_relu_status"] == prefrontier.FEASIBLE:
        final_status = STATUS_FEASIBLE
        interpretation = "LAYERNORM_CAUSALITY_STILL_UNRESOLVED"
    else:
        final_status = STATUS_INCONCLUSIVE
        interpretation = "LAYERNORM_IMAGE_INVARIANT_ORACLE_INCONCLUSIVE"
    capture_for_reuse = dict(identity)
    capture_for_reuse["tested_radius"] = payload["identity"]["tested_radius"]
    reuse = _inspect_reusable_input(
        manifest_path, capture_for_reuse, y, pre_layernorm_manifest)
    report = {
        "schema": SCHEMA, "property_id": identity["property_id"],
        "tested_radius": payload["identity"]["tested_radius"],
        "capture_authentication": identity,
        "layernorm_parameter_authentication": ln_identity,
        "ffn_first_parameter_authentication": w1_identity,
        "ffn_second_parameter_authentication": w2_identity,
        "invariant": {key: invariant[key] for key in (
            "nonzero_gamma_count", "zero_gamma_count", "equality_count",
            "exact_coefficients_sha256", "identity_sha256")},
        "downstream_problem": {
            "source_count": problem.n,
            "equality_count": int(problem.E.shape[0]),
            "inequality_count": int(2 * problem.n + 3 * len(problem.unstable)),
            "relu_unstable_count": len(problem.unstable)},
        "hull_lp": {
            "status": decision["hull_lp_status"],
            "feasible": decision["hull_lp_status"] == 0,
            "runtime_seconds": decision["hull_lp_runtime_seconds"],
            "exact_farkas_attempted": decision["hull_lp_status"] != 0,
            "exact_farkas_verified": decision["exact_farkas_verified"],
            "certificate_path": str(farkas_path) if farkas_path.is_file() else None,
            "certificate_sha256": (cluster_common.sha256(farkas_path)
                                   if farkas_path.is_file() else None)},
        "exact_relu_witness": {
            "attempted": decision["milp_status"] is not None,
            "verified": decision["exact_witness_verified"],
            "activation_sign_check": (witness or {}).get("activation_sign_check"),
            "source_box_check": (witness or {}).get("source_box_check"),
            "invariant_check": (witness or {}).get("invariant_check"),
            "maximum_exact_residual": (witness or {}).get("maximum_exact_residual"),
            "witness_path": str(witness_path) if witness is not None else None,
            "witness_sha256": (cluster_common.sha256(witness_path)
                               if witness is not None else None)},
        "reusable_pre_layernorm_input": reuse,
        "final_status": final_status, "causal_interpretation": interpretation,
        "runtime_seconds": time.perf_counter() - started,
        "scientific_queries": 0, "bound_calls": 0, "gpu_jobs": 0,
    }
    return relu._atomic_json(output, report)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--certificate-dir", required=True, type=Path)
    parser.add_argument("--pre-layernorm-manifest", type=Path)
    parser.add_argument("--lp-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--milp-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--exact-solve-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--maximum-patterns", type=int, default=8)
    parser.add_argument("--maximum-bases", type=int, default=32)
    args = parser.parse_args()
    execute(args.manifest.resolve(), args.output.resolve(),
            args.certificate_dir.resolve(),
            (args.pre_layernorm_manifest.resolve()
             if args.pre_layernorm_manifest else None),
            args.lp_timeout_seconds, args.milp_timeout_seconds,
            args.exact_solve_timeout_seconds, args.maximum_patterns,
            args.maximum_bases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
