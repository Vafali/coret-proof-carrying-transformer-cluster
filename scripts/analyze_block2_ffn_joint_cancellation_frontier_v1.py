#!/usr/bin/env python3
"""Read-only CPU analyzer for the authenticated Block-2 FFN frontier.

The numerical solvers are candidate generators only.  Exact feasibility is
accepted only after dyadic/Bareiss reconstruction and replay.  Strict
exclusion is accepted only when an exact rational replay of a dual witness is
positive.  In particular, the residual and FFN branch always share one source
box; independently concretized branch intervals are never combined.
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
import io

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "research_hab"))

import cluster_common
import coret_deept_exact_standard_ln_adapter as adapter
import deept_stagea_model as stagea


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


zero = _load_module(
    "block2_zero_variance_exact_engine",
    REPO / "scripts/decide_block2_output_zero_variance_v1.py")

SCHEMA = "CORET_BLOCK2_FFN_JOINT_CANCELLATION_FRONTIER_V1"
STATUS_ZERO = "JOINT_CANCELLATION_EXACTLY_FEASIBLE"
STATUS_POSITIVE = "JOINT_CANCELLATION_STRICTLY_EXCLUDED"
STATUS_UNRESOLVED = "JOINT_CANCELLATION_UNRESOLVED"
EXPECTED_MANIFEST_SCHEMA = "CORET_BLOCK2_FFN_FRONTIER_MANIFEST_V1"
EXPECTED_ARTIFACT_SCHEMA = "CORET_BLOCK2_FFN_FRONTIER_CAPTURE_V1"
FFN_SECOND_PARAMETER = "bert.encoder.layer.2.output.dense"
NUMERICAL_REASONS = zero.oracle.NUMERICAL_REASONS


def _fraction(value) -> Fraction:
    return Fraction.from_float(float(value))


def _json_sha(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    header = json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)},
                        sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(header + value.tobytes()).hexdigest()


def _tensor_sha(value: torch.Tensor) -> str:
    return _array_sha(value.detach().cpu().contiguous().numpy())


def _atomic_json(path: Path, value: dict) -> dict:
    payload = dict(value)
    payload["record_sha256"] = cluster_common.canonical(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return cluster_common.verified_json(path)


def _load_capture(manifest_path: Path) -> tuple[dict, dict, int]:
    manifest = cluster_common.verified_json(manifest_path)
    if (manifest.get("schema") != EXPECTED_MANIFEST_SCHEMA
            or manifest.get("property_id") != zero.PROPERTY_ID
            or manifest.get("multiplier") != zero.MULTIPLIER
            or manifest.get("stage_label") !=
            "block2_ffn_joint_cancellation_frontier"
            or manifest.get("pinned_deept_revision") != zero.oracle.PINNED_REVISION
            or manifest.get("scientific_manifest_sha256") !=
            cluster_common.SCIENTIFIC_MANIFEST_SHA
            or manifest.get("production_manifest_sha256") !=
            cluster_common.PRODUCTION_MANIFEST_SHA
            or manifest.get("source_set_model") != zero.oracle.SOURCE_SET_MODEL):
        raise RuntimeError("FFN frontier manifest identity differs")
    artifact_path = (manifest_path.parent
                     / manifest["tensor_artifact_path"]).resolve()
    if cluster_common.sha256(artifact_path) != manifest[
            "tensor_artifact_sha256"]:
        raise RuntimeError("FFN frontier artifact SHA differs")
    payload = torch.load(
        artifact_path, map_location="cpu", weights_only=False)
    expected_names = (
        "post_attention_ln_pre_reduction",
        "post_attention_ln_post_reduction",
        "ffn_first_pre_reduction", "ffn_first_post_reduction",
        "relu_raw_post_relaxation", "relu_post_injection_pre_reduction",
        "relu_post_reduction", "ffn_second_pre_reduction",
        "ffn_second_post_reduction", "residual_pair_pre_ffn",
        "residual_pair_pre_skip", "residual_pair_post_ffn",
        "residual_pair_post_skip", "residual_sum_pre_numerical_injection",
        "residual_sum_post_numerical_pre_reduction",
        "residual_sum_post_reduction")
    if (payload.get("schema") != EXPECTED_ARTIFACT_SCHEMA
            or payload.get("pinned_revision") != zero.oracle.PINNED_REVISION
            or payload.get("identity") != manifest.get("artifact_identity")
            or tuple((payload.get("states") or {}).keys()) != expected_names):
        raise RuntimeError("FFN frontier artifact identity/inventory differs")
    variants = manifest.get("variants")
    if (not isinstance(variants, list)
            or [row.get("state_key") for row in variants] != list(expected_names)):
        raise RuntimeError("FFN frontier manifest state inventory differs")
    for row in variants:
        state = payload["states"][row["state_key"]]
        proof = state["proof"]
        weights = state["weights"].detach().cpu().contiguous()
        low = state["range_low"].detach().cpu().contiguous()
        high = state["range_high"].detach().cpu().contiguous()
        ids_sha = _json_sha(proof["ids"])
        ranges_sha = hashlib.sha256(
            low.numpy().tobytes() + high.numpy().tobytes()).hexdigest()
        provenance_sha = _json_sha({
            "masks": proof["masks"], "reasons": proof["reasons"],
            "num_tokens": proof["num_tokens"],
        })
        actual = {
            "center_sha256": _tensor_sha(weights[0]),
            "generator_sha256": _tensor_sha(weights[1:]),
            "generator_ids_sha256": ids_sha,
            "ranges_sha256": ranges_sha,
            "provenance_sha256": provenance_sha,
            "generator_id_range_provenance_sha256": _json_sha({
                "generator_ids_sha256": ids_sha,
                "ranges_sha256": ranges_sha,
                "provenance_sha256": provenance_sha,
            }),
            "generator_count": int(weights.shape[0] - 1),
            "native_generator_count": sum(
                reason not in NUMERICAL_REASONS for reason in proof["reasons"]),
            "numerical_generator_count": sum(
                reason in NUMERICAL_REASONS for reason in proof["reasons"]),
            "token_count": int(weights.shape[1]),
            "feature_dimension": int(weights.shape[2]),
            "dtype": str(weights.dtype).replace("torch.", ""),
        }
        if any(row.get(field) != value for field, value in actual.items()):
            raise RuntimeError(
                f"FFN frontier state hash differs: {row['state_key']}")
    if (_json_sha(payload.get("reductions")) !=
            manifest.get("reduction_records_sha256")
            or payload.get("aliases") != manifest.get("aliases")):
        raise RuntimeError("FFN frontier reduction/alias identity differs")
    for side in ("pre", "post"):
        left = payload["states"][f"residual_pair_{side}_ffn"]
        right = payload["states"][f"residual_pair_{side}_skip"]
        if (left["proof"] != right["proof"]
                or not torch.equal(left["range_low"], right["range_low"])
                or not torch.equal(left["range_high"], right["range_high"])):
            raise RuntimeError(f"FFN frontier {side} pair identity differs")
    result_path = (manifest_path.parent / manifest["result_path"]).resolve()
    if cluster_common.sha256(result_path) != manifest["result_sha256"]:
        raise RuntimeError("FFN frontier result SHA differs")
    result = cluster_common.verified_json(result_path)
    diagnostic = result.get("domain_failure_diagnostic")
    if (result.get("schema") != "CORET_SOUND_FP64_3L_PROPERTY_RESULT_V1"
            or result.get("property_id") != zero.PROPERTY_ID
            or result.get("terminal_status") != "UNCERTIFIED_DOMAIN_FAILURE"
            or result.get("scientific_evaluation_complete") is not True
            or result.get("certified_at_historical_radius") is not False
            or result.get("classification") != "FAILED_AT_HISTORICAL_RADIUS"
            or result.get("failure_category") !=
            "SOUND_LAYERNORM_DOMAIN_FAILURE"
            or result.get("failure_reason") !=
            "SOUND_FP64_LAYERNORM_VARIANCE_DOMAIN_FAILURE"
            or result.get("generic_fallback_count") != 0
            or not isinstance(diagnostic, dict)
            or diagnostic.get("reason_code") !=
            "SOUND_FP64_LAYERNORM_VARIANCE_DOMAIN_FAILURE"
            or diagnostic.get("label") != "block2_output"
            or diagnostic.get("domain_admissible") is not False):
        raise RuntimeError("captured failure result semantics differ")
    token = int(diagnostic["minimum_token_index"])
    for name, state in payload["states"].items():
        if not 0 <= token < int(state["weights"].shape[1]):
            raise RuntimeError(f"captured token is outside state: {name}")
    identity = {
        "schema": EXPECTED_MANIFEST_SCHEMA,
        "property_id": manifest["property_id"],
        "state_count": len(payload["states"]),
        "shared_pair_identity_verified": True,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": cluster_common.sha256(manifest_path),
        "artifact_path": str(artifact_path),
        "artifact_sha256": manifest["tensor_artifact_sha256"],
        "result_path": str(result_path),
        "result_sha256": manifest["result_sha256"],
        "minimum_token_index": token,
    }
    return payload, identity, token


def _load_ffn_second_parameters() -> tuple[np.ndarray, np.ndarray, dict]:
    raw = stagea._git_blob(adapter.DEEPT_REPOSITORY, stagea.CHECKPOINT_GIT_PATH)
    if hashlib.sha256(raw).hexdigest() != stagea.CHECKPOINT_SHA256:
        raise RuntimeError("frozen checkpoint identity differs")
    checkpoint = torch.load(io.BytesIO(raw), map_location="cpu",
                            weights_only=False)
    weight = checkpoint[f"{FFN_SECOND_PARAMETER}.weight"].detach().cpu().to(
        dtype=torch.float64).contiguous().numpy()
    bias = checkpoint[f"{FFN_SECOND_PARAMETER}.bias"].detach().cpu().to(
        dtype=torch.float64).contiguous().numpy()
    if (weight.shape != (128, 512) or bias.shape != (128,)
            or not np.isfinite(weight).all() or not np.isfinite(bias).all()):
        raise RuntimeError("frozen Block-2 FFN second affine differs")
    return weight, bias, {
        "parameter": FFN_SECOND_PARAMETER,
        "checkpoint_sha256": stagea.CHECKPOINT_SHA256,
        "weight_sha256": _array_sha(weight),
        "bias_sha256": _array_sha(bias),
    }


def _decode_state(state: dict, token: int) -> dict:
    weights = state["weights"].detach().cpu().contiguous().numpy()
    low = state["range_low"].detach().cpu().contiguous().numpy()
    high = state["range_high"].detach().cpu().contiguous().numpy()
    proof = state["proof"]
    return {
        "center": np.asarray(weights[0, token], dtype=np.float64),
        "generators": np.asarray(weights[1:, token], dtype=np.float64),
        "low": np.asarray(low, dtype=np.float64),
        "high": np.asarray(high, dtype=np.float64),
        "ids": list(proof["ids"]), "masks": list(proof["masks"]),
        "reasons": list(proof["reasons"]),
        "num_tokens": int(proof["num_tokens"]),
    }


def _align_sources(left: dict, right: dict) -> tuple[dict, dict, dict]:
    """Align two affine states without changing shared source semantics."""
    order = list(left["ids"])
    seen = set(order)
    order.extend(item for item in right["ids"] if item not in seen)
    if len(set(order)) != len(order):
        raise RuntimeError("source IDs are not unique")
    index = {item: position for position, item in enumerate(order)}
    width_left = left["generators"].shape[1]
    width_right = right["generators"].shape[1]
    aligned = []
    metadata = {}
    for label, state, width in (("left", left, width_left),
                                ("right", right, width_right)):
        generators = np.zeros((len(order), width), dtype=np.float64)
        low = np.full(len(order), np.nan, dtype=np.float64)
        high = np.full(len(order), np.nan, dtype=np.float64)
        reasons = [None] * len(order)
        masks = [None] * len(order)
        for source, lo, hi, reason, mask, row in zip(
                state["ids"], state["low"], state["high"],
                state["reasons"], state["masks"], state["generators"]):
            target = index[source]
            generators[target] = row
            low[target], high[target] = lo, hi
            reasons[target], masks[target] = reason, mask
            previous = metadata.get(source)
            current = (float(lo).hex(), float(hi).hex(), reason, int(mask))
            if previous is not None and previous != current:
                raise RuntimeError(
                    f"shared source range/provenance differs: {source}")
            metadata[source] = current
        aligned.append({"center": state["center"], "generators": generators})
    ranges_low = np.array([float.fromhex(metadata[item][0]) for item in order])
    ranges_high = np.array([float.fromhex(metadata[item][1]) for item in order])
    reasons = [metadata[item][2] for item in order]
    masks = [metadata[item][3] for item in order]
    return aligned[0], aligned[1], {
        "ids": order, "low": ranges_low, "high": ranges_high,
        "reasons": reasons, "masks": masks,
    }


class ExactJointAffine:
    """Exact-real r + branch or r + W2*branch+b2 over one source box."""

    def __init__(self, branch: dict, skip: dict, *, weight=None, bias=None,
                 label: str):
        branch, skip, source = _align_sources(branch, skip)
        self.branch, self.skip, self.source = branch, skip, source
        self.weight = None if weight is None else np.asarray(weight)
        self.bias = None if bias is None else np.asarray(bias)
        self.label = label
        if self.weight is None:
            if branch["center"].shape != skip["center"].shape:
                raise RuntimeError("direct joint branch dimensions differ")
            center = branch["center"] + skip["center"]
            generators = branch["generators"] + skip["generators"]
        else:
            if (self.weight.shape[1] != branch["center"].size
                    or self.weight.shape[0] != skip["center"].size
                    or self.bias.shape != skip["center"].shape):
                raise RuntimeError("remaining FFN affine dimensions differ")
            center = skip["center"] + self.weight @ branch["center"] + self.bias
            generators = skip["generators"] + branch["generators"] @ self.weight.T
        self.numeric_center = np.asarray(center, dtype=np.float64)
        self.numeric_generators = np.asarray(generators, dtype=np.float64)
        semantics = {
            "label": label, "ids_sha256": _json_sha(source["ids"]),
            "branch_center_sha256": _array_sha(branch["center"]),
            "branch_generators_sha256": _array_sha(branch["generators"]),
            "skip_center_sha256": _array_sha(skip["center"]),
            "skip_generators_sha256": _array_sha(skip["generators"]),
            "weight_sha256": None if self.weight is None else _array_sha(self.weight),
            "bias_sha256": None if self.bias is None else _array_sha(self.bias),
        }
        self.semantics_sha256 = _json_sha(semantics)

    def _coordinate(self, row: int, column: int | None) -> Fraction:
        if column is None:
            branch = self.branch["center"]
            skip = self.skip["center"]
        else:
            branch = self.branch["generators"][column]
            skip = self.skip["generators"][column]
        value = _fraction(skip[row])
        if self.weight is None:
            value += _fraction(branch[row])
        else:
            if column is None:
                value += _fraction(self.bias[row])
            value += sum((_fraction(value) * _fraction(coefficient)
                          for value, coefficient in
                          zip(branch, self.weight[row])), Fraction(0))
        return value

    def exact_center_difference(self, row: int) -> Fraction:
        return self._coordinate(row, None) - self._coordinate(-1, None)

    def exact_coefficient(self, row: int, column: int) -> Fraction:
        return (self._coordinate(row, column)
                - self._coordinate(-1, column))

    def exact_residual(self, values: list[Fraction]) -> list[Fraction]:
        if len(values) != len(self.source["ids"]):
            raise RuntimeError("exact joint candidate source count differs")
        branch_value = [
            _fraction(value) + sum(
                (_fraction(self.branch["generators"][index, coordinate]) * xi
                 for index, xi in enumerate(values)), Fraction(0))
            for coordinate, value in enumerate(self.branch["center"])]
        skip_value = [
            _fraction(value) + sum(
                (_fraction(self.skip["generators"][index, coordinate]) * xi
                 for index, xi in enumerate(values)), Fraction(0))
            for coordinate, value in enumerate(self.skip["center"])]
        if self.weight is None:
            output = [left + right for left, right in
                      zip(branch_value, skip_value)]
        else:
            output = [
                skip_value[row] + _fraction(self.bias[row]) + sum(
                    (_fraction(weight) * value for weight, value in
                     zip(self.weight[row], branch_value)), Fraction(0))
                for row in range(len(skip_value))]
        return [value - output[-1] for value in output[:-1]]

    def exact_replay(self, values: list[Fraction]) -> dict:
        residuals = self.exact_residual(values)
        if any(residual != 0 for residual in residuals):
            raise RuntimeError("exact joint cancellation replay is nonzero")
        return {
            "exact_equalities": len(residuals),
            "exact_box_constraints": len(values),
            "maximum_exact_residual": "0", "exact_variance": "0",
            "joint_residual_ffn_correlation_preserved": True,
        }

    def exact_dual_lower(self, witness: np.ndarray) -> dict:
        y = [_fraction(value) for value in np.asarray(witness, dtype=np.float64)]
        mean = sum(y, Fraction(0)) / len(y)
        y = [value - mean for value in y]
        if self.weight is None:
            branch_dual = y
        else:
            branch_dual = [sum(
                (_fraction(self.weight[row, feature]) * y[row]
                 for row in range(len(y))), Fraction(0))
                for feature in range(self.weight.shape[1])]
        center_term = sum(
            (_fraction(value) * coefficient for value, coefficient in
             zip(self.skip["center"], y)), Fraction(0))
        center_term += sum(
            (_fraction(value) * coefficient for value, coefficient in
             zip(self.branch["center"], branch_dual)), Fraction(0))
        if self.weight is not None:
            center_term += sum(
                (_fraction(value) * coefficient for value, coefficient in
                 zip(self.bias, y)), Fraction(0))
        support = Fraction(0)
        for index in range(len(self.source["ids"])):
            coefficient = sum(
                (_fraction(value) * dual for value, dual in
                 zip(self.skip["generators"][index], y)), Fraction(0))
            coefficient += sum(
                (_fraction(value) * dual for value, dual in
                 zip(self.branch["generators"][index], branch_dual)),
                Fraction(0))
            bound = (self.source["low"][index] if coefficient >= 0
                     else self.source["high"][index])
            support += coefficient * _fraction(bound)
        objective = (2 * (center_term + support)
                     - sum((value * value for value in y), Fraction(0)))
        lower = objective / len(y)
        outward = float(lower)
        if Fraction.from_float(outward) > lower:
            outward = math.nextafter(outward, -math.inf)
        return {
            "exact_numerator": str(lower.numerator),
            "exact_denominator": str(lower.denominator),
            "strictly_positive": lower > 0,
            "outward_safe_lower_binary64": outward,
            "witness_sha256": _array_sha(np.asarray(witness, dtype=np.float64)),
        }

    def problem(self) -> dict:
        problem = zero.centered_problem(
            self.numeric_center, self.numeric_generators,
            self.source["low"], self.source["high"], self.source["ids"])
        problem["problem_sha256"] = hashlib.sha256(
            (problem["problem_sha256"] + self.semantics_sha256).encode()
        ).hexdigest()
        problem.update({
            "exact_center_difference": self.exact_center_difference,
            "exact_coefficient": self.exact_coefficient,
            "exact_residual": self.exact_residual,
            "exact_replay": self.exact_replay,
        })
        return problem


def _counts(model: ExactJointAffine) -> dict:
    numerical = sum(reason in NUMERICAL_REASONS
                    for reason in model.source["reasons"])
    return {
        "generator_count": len(model.source["ids"]),
        "native_generator_count": len(model.source["ids"]) - numerical,
        "numerical_generator_count": numerical,
    }


def _decide(name: str, model: ExactJointAffine, certificate_dir: Path,
            exact_timeout: float) -> dict:
    started = time.perf_counter()
    problem = model.problem()
    candidate, primal = zero.numerical_primal_search(problem, name)
    gate = zero.near_zero_gate(problem, primal)
    untrusted_dual = zero.dual_scaling_search(problem, candidate, name)
    # Exact dual replay is potentially expensive for W2-mapped states.  A
    # nonpositive numerical candidate cannot establish strict exclusion, so
    # replay it only when it can possibly discharge that obligation.
    if untrusted_dual["best"]["outward_safe_lower"] > 0.0:
        exact_dual = model.exact_dual_lower(
            np.array([float.fromhex(item) for item in
                      untrusted_dual["best"]["witness_binary64_hex"]]))
    else:
        exact_dual = {
            "strictly_positive": False,
            "outward_safe_lower_binary64": 0.0,
            "witness_sha256": untrusted_dual["best"]["witness_sha256"],
            "replay_skipped": (
                "numerical candidate was nonpositive and cannot prove "
                "strict exclusion"),
        }
    certificate = None
    exact_status = {"attempted": False}
    if exact_dual["strictly_positive"]:
        status = STATUS_POSITIVE
        exact_status["reason"] = "exact rational dual replay is positive"
        evidence_hash = exact_dual["witness_sha256"]
    elif not gate["is_compelling_near_zero"]:
        status = STATUS_UNRESOLVED
        exact_status["reason"] = "numerical candidate is not compellingly zero"
        evidence_hash = exact_dual["witness_sha256"]
    else:
        certificate, exact_status = zero.construct_exact_zero_certificate(
            problem, candidate, 127, variant_name=name,
            solve_timeout_seconds=exact_timeout)
        if certificate is None:
            status = STATUS_UNRESOLVED
            evidence_hash = exact_dual["witness_sha256"]
        else:
            status = STATUS_ZERO
            certificate_path = certificate_dir / f"{name}_exact_zero.json"
            certificate = zero._atomic_json(certificate_path, certificate)
            zero.verify_exact_zero_certificate(problem, certificate)
            evidence_hash = cluster_common.sha256(certificate_path)
    return {
        "node": name, "status": status, **_counts(model),
        "joint_semantics_sha256": model.semantics_sha256,
        "problem_sha256": problem["problem_sha256"],
        "numerical_primal": primal, "near_zero_gate": gate,
        "exact_dual": exact_dual,
        "exact_zero_certificate_status": exact_status,
        "witness_or_dual_hash": evidence_hash,
        "runtime_seconds": time.perf_counter() - started,
    }


def _state_models(payload: dict, token: int, weight: np.ndarray,
                  bias: np.ndarray) -> dict[str, ExactJointAffine]:
    states = {name: _decode_state(state, token)
              for name, state in payload["states"].items()}
    skip = states["post_attention_ln_post_reduction"]
    models = {}
    for name in ("relu_raw_post_relaxation",
                 "relu_post_injection_pre_reduction",
                 "relu_post_reduction"):
        models[name] = ExactJointAffine(
            states[name], skip, weight=weight, bias=bias, label=name)
    for name in ("ffn_second_pre_reduction", "ffn_second_post_reduction"):
        models[name] = ExactJointAffine(states[name], skip, label=name)
    for side in ("pre", "post"):
        models[f"residual_pair_{side}"] = ExactJointAffine(
            states[f"residual_pair_{side}_ffn"],
            states[f"residual_pair_{side}_skip"],
            label=f"residual_pair_{side}")
    zero_state = lambda state: {
        **state, "center": np.zeros_like(state["center"]),
        "generators": np.zeros_like(state["generators"]),
    }
    for name in ("residual_sum_pre_numerical_injection",
                 "residual_sum_post_numerical_pre_reduction",
                 "residual_sum_post_reduction"):
        state = states[name]
        models[name] = ExactJointAffine(
            state, zero_state(state), label=name)
    return models


TRANSITIONS = (
    ("post_attention_layernorm_reduction", None, None),
    ("ffn_first_affine_and_injection", None, None),
    ("ffn_first_reduction", None, None),
    ("relu_relaxation", None, "relu_raw_post_relaxation"),
    ("relu_numerical_injection", "relu_raw_post_relaxation",
     "relu_post_injection_pre_reduction"),
    ("relu_reduction", "relu_post_injection_pre_reduction",
     "relu_post_reduction"),
    ("ffn_second_affine_and_injection", "relu_post_reduction",
     "ffn_second_pre_reduction"),
    ("ffn_second_reduction", "ffn_second_pre_reduction",
     "ffn_second_post_reduction"),
    ("residual_pair_alignment", "ffn_second_post_reduction",
     "residual_pair_pre"),
    ("residual_pair_reduction", "residual_pair_pre", "residual_pair_post"),
    ("affine_residual_addition", "residual_pair_post",
     "residual_sum_pre_numerical_injection"),
    ("residual_numerical_injection", "residual_sum_pre_numerical_injection",
     "residual_sum_post_numerical_pre_reduction"),
    ("final_residual_reduction",
     "residual_sum_post_numerical_pre_reduction",
     "residual_sum_post_reduction"),
)


def _repair_oracle(first: str | None) -> dict:
    if first == "relu_relaxation":
        return {
            "kind": "joint_multivariate_relu_relaxation",
            "falsification": ("preserve the authenticated residual correlation "
                              "through one stronger captured ReLU relaxation"),
        }
    if first in {"relu_reduction", "ffn_second_reduction",
                 "residual_pair_reduction", "final_residual_reduction"}:
        return {
            "kind": "correlation_preserving_boundary_reduction",
            "falsification": ("preserve exactly the residual-cancellation "
                              "correlation at this one reduction boundary"),
        }
    if first in {"relu_numerical_injection",
                 "ffn_second_affine_and_injection",
                 "residual_numerical_injection"}:
        return {
            "kind": "correlated_numerical_error_at_single_transition",
            "falsification": ("replace only this coordinate-box numerical "
                              "injection by a correlation-preserving oracle"),
        }
    return {
        "kind": None,
        "falsification": ("no repair oracle is justified until an excluded-to-"
                          "exactly-feasible adjacent transition is proved"),
    }


def execute(manifest_path: Path, output_path: Path, certificate_dir: Path,
            exact_timeout: float) -> dict:
    started = time.perf_counter()
    payload, identity, token = _load_capture(manifest_path)
    weight, bias, parameter_identity = _load_ffn_second_parameters()
    models = _state_models(payload, token, weight, bias)
    rows = {}
    partial_path = output_path.with_suffix(".partial.json")
    for name, model in models.items():
        rows[name] = _decide(name, model, certificate_dir, exact_timeout)
        _atomic_json(partial_path, {
            "schema": SCHEMA, "capture_identity": identity,
            "parameter_identity": parameter_identity,
            "completed_nodes": list(rows), "nodes": list(rows.values()),
        })
        print(json.dumps({
            "event": "FFN_FRONTIER_NODE_COMPLETE", "node": name,
            "status": rows[name]["status"],
            "runtime_seconds": rows[name]["runtime_seconds"],
        }, sort_keys=True), flush=True)
    unresolved = {
        "status": STATUS_UNRESOLVED,
        "reason": ("capture authenticates the preactivation but does not "
                   "serialize an exact piecewise-ReLU branch witness; no "
                   "claim is made across that nonlinear predecessor"),
    }
    table = []
    first = None
    for transition, before_name, after_name in TRANSITIONS:
        before = rows.get(before_name, unresolved)
        after = rows.get(after_name, unresolved)
        false_to_true = (before["status"] == STATUS_POSITIVE
                         and after["status"] == STATUS_ZERO)
        if false_to_true and first is None:
            first = transition
        table.append({
            "transition": transition,
            "before_node": before_name,
            "after_node": after_name,
            "before_status": before["status"],
            "after_status": after["status"],
            "first_false_to_true": transition == first,
            "generator_count_before": before.get("generator_count"),
            "generator_count_after": after.get("generator_count"),
            "native_count_before": before.get("native_generator_count"),
            "native_count_after": after.get("native_generator_count"),
            "numerical_count_before": before.get("numerical_generator_count"),
            "numerical_count_after": after.get("numerical_generator_count"),
            "witness_or_dual_hash_before": before.get("witness_or_dual_hash"),
            "witness_or_dual_hash_after": after.get("witness_or_dual_hash"),
            "runtime_seconds": (before.get("runtime_seconds", 0.0)
                                + after.get("runtime_seconds", 0.0)),
        })
    report = {
        "schema": SCHEMA, "capture_identity": identity,
        "parameter_identity": parameter_identity,
        "analysis_token_index": token,
        "joint_condition": "exists xi: P(r(xi)+f(xi))=0",
        "source_correlation_policy": (
            "ordered authenticated source-ID union with exact shared ranges"),
        "nodes": list(rows.values()), "transition_table": table,
        "first_false_to_true_transition": first,
        "frontier_identified": first is not None,
        "single_minimal_repair_oracle": _repair_oracle(first),
        "pre_relu_boundary": unresolved,
        "scientific_queries": 0, "bound_calls": 0,
        "gpu_jobs": 0, "production_files_modified": 0,
        "runtime_seconds": time.perf_counter() - started,
    }
    return _atomic_json(output_path, report)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--certificate-dir", type=Path, required=True)
    parser.add_argument("--exact-solve-timeout-seconds", type=float,
                        default=180.0)
    args = parser.parse_args()
    report = execute(args.manifest, args.output, args.certificate_dir,
                     args.exact_solve_timeout_seconds)
    for row in report["transition_table"]:
        print(json.dumps({"event": "FFN_FRONTIER_TRANSITION", **row},
                         sort_keys=True), flush=True)
    print(json.dumps({
        "status": "CORET_BLOCK2_FFN_FRONTIER_EXISTING_CAPTURE_READY",
        "first_false_to_true_transition":
            report["first_false_to_true_transition"],
        "output": str(args.output.resolve()),
        "record_sha256": report["record_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
