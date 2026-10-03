#!/usr/bin/env python3
"""One-property opt-in PSD-aware Block-2 LayerNorm experiment."""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "research_hab")]

import cluster_common
import coret_psd_layernorm_experiment_v1 as experiment
import inspect_block2_layernorm_capture_linkage_v1 as linkage_diagnostic
import run_sound_fp64_3l_campaign as campaign
import run_sound_fp64_3l_psd_state_capture_v1 as capture
import run_sound_fp64_finish_3l_v1 as finish3l


SCHEMA = "CORET_PSD_LAYERNORM_EXPERIMENT_JOB_V2"
PROPERTY_ID = "deept_table7_stdln3_s001_line1794_tok11"
MULTIPLIER = "0.75"
TESTED_RADIUS = 0.00060791015625
TESTED_RADIUS_HEX = "0x1.3eb851eb851ecp-11"
TRACE_SCHEMA = "CORET_PSD_LAYERNORM_INVOCATION_TRACE_V2"
NEXT_CAPTURE_SCHEMA = "CORET_PSD_NEXT_LAYERNORM_STATE_CAPTURE_V1"
NEXT_MANIFEST_SCHEMA = "CORET_PSD_NEXT_LAYERNORM_CAPTURE_MANIFEST_V1"
NEXT_STAGE = "block2_output"
NEXT_LAYERNORM_INDEX = 6
NEXT_REDUCTION_LABEL = "b2_ffn_residual"
FRONTIER_CAPTURE_SCHEMA = "CORET_BLOCK2_FFN_FRONTIER_CAPTURE_V1"
FRONTIER_MANIFEST_SCHEMA = "CORET_BLOCK2_FFN_FRONTIER_MANIFEST_V1"
FRONTIER_STAGE = "block2_ffn_joint_cancellation_frontier"
INPUT_CAPTURE_SCHEMA = "CORET_BLOCK2_POST_ATTENTION_LN_INPUT_CAPTURE_V1"
INPUT_MANIFEST_SCHEMA = (
    "CORET_BLOCK2_POST_ATTENTION_LN_INPUT_CAPTURE_MANIFEST_V1")
INPUT_OPERATOR = "bert.encoder.layer.2.attention.output.LayerNorm"
INPUT_BLOCK_INDEX = 2
INPUT_LAYERNORM_INDEX = 5
EXPECTED_INPUT_CANONICAL_STATE_IDENTITY_SHA256 = (
    "c2fbf1d175157dfecca9c3da95573b593921a4dc05ca1b6a3e7eacaf730ea507")

FRONTIER_REDUCTION_STATES = {
    "b2_post_attention_layernorm": (
        "post_attention_ln_pre_reduction",
        "post_attention_ln_post_reduction"),
    "b2_ffn_first": (
        "ffn_first_pre_reduction", "ffn_first_post_reduction"),
    "b2_relu": (
        "relu_post_injection_pre_reduction", "relu_post_reduction"),
    "b2_ffn_second": (
        "ffn_second_pre_reduction", "ffn_second_post_reduction"),
    "b2_ffn_residual": (
        "residual_sum_post_numerical_pre_reduction",
        "residual_sum_post_reduction"),
}
FRONTIER_PAIR_LABEL = "b2_pre_ffn_residual_pair"
FRONTIER_STATE_NAMES = (
    "post_attention_ln_pre_reduction",
    "post_attention_ln_post_reduction",
    "ffn_first_pre_reduction",
    "ffn_first_post_reduction",
    "relu_raw_post_relaxation",
    "relu_post_injection_pre_reduction",
    "relu_post_reduction",
    "ffn_second_pre_reduction",
    "ffn_second_post_reduction",
    "residual_pair_pre_ffn",
    "residual_pair_pre_skip",
    "residual_pair_post_ffn",
    "residual_pair_post_skip",
    "residual_sum_pre_numerical_injection",
    "residual_sum_post_numerical_pre_reduction",
    "residual_sum_post_reduction",
)


def _atomic_json(path: Path, value: dict) -> dict:
    payload = dict(value)
    payload["record_sha256"] = cluster_common.canonical(payload)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return cluster_common.verified_json(path)


def _runtime_provenance() -> dict:
    """Collect environment provenance before scientific execution starts."""
    torch = finish3l.torch

    def command(*args):
        completed = subprocess.run(
            args, cwd=REPO, check=True, capture_output=True, text=True,
            timeout=5)
        return completed.stdout.strip()

    cudnn = getattr(torch.backends, "cudnn", None)
    try:
        gpu_driver_rows = command(
            "nvidia-smi", "--query-gpu=index,name,driver_version",
            "--format=csv,noheader")
    except (FileNotFoundError, subprocess.SubprocessError):
        gpu_driver_rows = None
    return {
        "git_head": command("git", "rev-parse", "HEAD"),
        "git_status_porcelain": command("git", "status", "--porcelain").splitlines(),
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": cudnn.version() if cudnn is not None else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": cudnn.allow_tf32 if cudnn is not None else None,
        "gpu_model_and_driver": gpu_driver_rows,
    }


def _authenticate_oracle(path: Path) -> dict:
    report = cluster_common.verified_json(path)
    if (report.get("schema") != "CORET_PSD_LAYERNORM_VARIANCE_ORACLE_V1"
            or report.get("property_id") != PROPERTY_ID):
        raise RuntimeError("PSD oracle report identity differs")
    rows = report.get("results")
    matches = [row for row in rows if row.get("multiplier") == MULTIPLIER
               and row.get("variant") == "complete"] if isinstance(rows, list) else []
    if len(matches) != 1:
        raise RuntimeError("PSD oracle complete 0.75x record differs")
    row = matches[0]
    if (float(row.get("tested_radius")) != TESTED_RADIUS
            or row.get("psd_positive") is not True
            or float(row.get("psd_dual_candidate_lower_outward_safe")) <= 0):
        raise RuntimeError("PSD oracle positive certificate differs")
    return {"path": str(path), "sha256": cluster_common.sha256(path),
            "record": row}


def _captured_state_identity(manifest_path: Path, manifest: dict) -> dict:
    artifact = (manifest_path.parent
                / manifest["tensor_artifact_path"]).resolve()
    payload = capture.sound.torch.load(
        artifact, map_location="cpu", weights_only=False)
    snapshot = payload["states"]["post_last_reduction"]
    return _snapshot_state_identity(snapshot)


def _snapshot_state_identity(snapshot: dict) -> dict:
    """Reconstruct the exact PSD canonical identity from a CPU snapshot."""
    proof = snapshot["proof"]
    state = SimpleNamespace(
        zonotope_w=snapshot["weights"],
        error_term_range_low=snapshot["range_low"],
        error_term_range_high=snapshot["range_high"],
        num_error_terms=int(snapshot["weights"].shape[0] - 1),
        num_words=int(snapshot["weights"].shape[1]),
        word_embedding_size=int(snapshot["weights"].shape[2]))
    support = experiment.structural.SupportProof(
        tuple(proof["masks"]), tuple(proof["ids"]),
        tuple(proof["reasons"]), int(proof["num_tokens"]))
    return experiment.state_identity(state, support)


class LayerNormExperimentHarness:
    """Authenticate and trace the sole opt-in PSD LayerNorm application."""

    def __init__(self, property_id: str, radius: float,
                 expected_state_identity: dict, oracle_identity: dict,
                 input_capture=None):
        self.property_id = property_id
        self.radius = radius
        self.expected_state_identity = expected_state_identity
        self.oracle_identity = oracle_identity
        self.input_capture = input_capture
        self.trace = []
        self.certificates = []
        self.total_callback_invocations = 0
        self.non_target_invocations = 0
        self.target_stage_encounters = 0
        self.target_state_matches = 0
        self.psd_certificate_applications = 0
        self.psd_certificate_rejections = 0
        self.evaluator_exception = None

    def _bind_range_certificate(self, certificate: dict) -> None:
        certificate.pop("canonical_certificate_sha256", None)
        certificate.update({
            "property_id": self.property_id,
            "tested_radius": self.radius,
            "tested_radius_hex": self.radius.hex(),
            "oracle_report_sha256": self.oracle_identity["sha256"],
        })
        certificate["canonical_certificate_sha256"] = (
            experiment._json_hash(certificate))

    def __call__(self, *, residual, proof, normalizer, delegate,
                 diagnostics, original_layernorm=None):
        self.total_callback_invocations += 1
        stage = diagnostics.get("label")
        index = int(getattr(delegate, "_layer_norm_index", -1))
        entry = {
            "ordinal": self.total_callback_invocations - 1,
            "stage": stage,
            "layernorm_index": index,
            "state_identity": None,
            "matched_property": self.property_id == PROPERTY_ID,
            "matched_radius": (self.radius == TESTED_RADIUS
                               and self.radius.hex() == TESTED_RADIUS_HEX),
            "matched_stage": stage == experiment.TARGET_LABEL,
            "matched_index": index == experiment.TARGET_LAYER_NORM_INDEX,
            "matched_state": False,
            "target_state_match": False,
            "psd_applied": False,
            "reject_reason": None,
            "psd_variance_certificate_authenticated": False,
            "psd_variance_lower_consumed": False,
            "sqrt_semantic_lower_constructed": False,
            "reciprocal_semantic_range_consumed": False,
            "full_psd_layernorm_completed": False,
            "semantic_range_certificate": None,
        }
        if entry["matched_stage"]:
            self.target_stage_encounters += 1
        else:
            self.non_target_invocations += 1
            entry["reject_reason"] = "NON_TARGET_ORIGINAL_PATH"
            self.trace.append(entry)
            if original_layernorm is None:
                raise RuntimeError(
                    "PSD callback reached a non-target LayerNorm without its "
                    "original implementation")
            return original_layernorm(
                residual=residual, proof=proof, normalizer=normalizer,
                delegate=delegate, diagnostics=diagnostics)
        try:
            live_identity = experiment.state_identity(residual, proof)
            entry["state_identity"] = live_identity
            entry["matched_state"] = (
                live_identity == self.expected_state_identity)
            entry["target_state_match"] = all((
                entry["matched_property"], entry["matched_radius"],
                entry["matched_stage"], entry["matched_index"],
                entry["matched_state"], bool(self.oracle_identity)))
            if not entry["target_state_match"]:
                entry["reject_reason"] = "AUTHENTICATED_TARGET_MISMATCH"
                self.psd_certificate_rejections += 1
                raise RuntimeError("PSD LayerNorm authenticated target differs")
            self.target_state_matches += 1
            if self.target_state_matches != 1:
                entry["reject_reason"] = "DUPLICATE_AUTHENTICATED_TARGET"
                self.psd_certificate_rejections += 1
                raise RuntimeError("PSD LayerNorm target state visited twice")
            if self.input_capture is not None:
                self.input_capture.bind_invocation(
                    layernorm_index=index, invocation_ordinal=entry["ordinal"],
                    live_state_identity=live_identity)
            milestones = {}
            result = experiment.execute_experimental_layernorm(
                residual=residual, proof=proof, normalizer=normalizer,
                delegate=delegate, diagnostics=diagnostics,
                milestones=milestones)
            certificate = result.get("certificate")
            if not isinstance(certificate, dict):
                entry["reject_reason"] = "PSD_CERTIFICATE_ABSENT"
                self.psd_certificate_rejections += 1
                raise RuntimeError("experimental PSD certificate was not emitted")
            if certificate.get("state_identity") != live_identity:
                entry["reject_reason"] = "PSD_CERTIFICATE_STATE_MISMATCH"
                self.psd_certificate_rejections += 1
                raise RuntimeError("experimental PSD certificate state differs")
            range_certificate = certificate.get(
                "native_transition", {}).get("semantic_range_certificate")
            if not isinstance(range_certificate, dict):
                entry["reject_reason"] = "SEMANTIC_RANGE_CERTIFICATE_ABSENT"
                self.psd_certificate_rejections += 1
                raise RuntimeError("semantic range certificate was not emitted")
            self._bind_range_certificate(range_certificate)
            self.certificates.append(certificate)
            self.psd_certificate_applications += 1
            entry["psd_applied"] = True
            return result
        except Exception as error:
            if entry["reject_reason"] is None:
                entry["reject_reason"] = f"{type(error).__name__}: {error}"
                self.psd_certificate_rejections += 1
            raise
        finally:
            for name in (
                    "psd_variance_certificate_authenticated",
                    "psd_variance_lower_consumed",
                    "sqrt_semantic_lower_constructed",
                    "reciprocal_semantic_range_consumed",
                    "full_psd_layernorm_completed"):
                if "milestones" in locals():
                    entry[name] = bool(milestones.get(name, False))
            if ("milestones" in locals()
                    and isinstance(milestones.get(
                        "semantic_range_certificate"), dict)):
                range_certificate = milestones["semantic_range_certificate"]
                self._bind_range_certificate(range_certificate)
                entry["semantic_range_certificate"] = range_certificate
            self.trace.append(entry)

    def record_evaluator_exception(self, error: Exception) -> None:
        self.evaluator_exception = {
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback_tail": traceback.format_exc().splitlines()[-40:],
            "last_callback_trace_entry": (
                self.trace[-1] if self.trace else None),
        }

    def counters(self) -> dict:
        return {
            "total_callback_invocations": self.total_callback_invocations,
            "non_target_invocations": self.non_target_invocations,
            "target_stage_encounters": self.target_stage_encounters,
            "target_state_matches": self.target_state_matches,
            "psd_certificate_applications": self.psd_certificate_applications,
            "psd_certificate_rejections": self.psd_certificate_rejections,
        }

    def validate(self) -> None:
        counters = self.counters()
        if counters["target_state_matches"] != 1:
            raise RuntimeError("PSD experiment target-state match count differs")
        if counters["psd_certificate_applications"] != 1:
            raise RuntimeError("PSD experiment certificate application count differs")
        if counters["psd_certificate_rejections"] != 0:
            raise RuntimeError("PSD experiment certificate was rejected")
        if self.input_capture is not None:
            self.input_capture.validate()

    def record(self) -> dict:
        return {
            "schema": TRACE_SCHEMA,
            "property_id": self.property_id,
            "tested_radius": self.radius,
            "tested_radius_hex": self.radius.hex(),
            "expected_state_identity": self.expected_state_identity,
            "oracle_report_sha256": self.oracle_identity.get("sha256"),
            "counters": self.counters(),
            "invocations": self.trace,
            "evaluator_exception": self.evaluator_exception,
        }


class PostAttentionLayerNormInputCapture:
    """Schedule-passive reference capture for the index-5 LayerNorm input."""

    def __init__(self, expected_state_identity: dict,
                 required_canonical_identity: str =
                 EXPECTED_INPUT_CANONICAL_STATE_IDENTITY_SHA256):
        self.expected_state_identity = expected_state_identity
        self.required_canonical_identity = required_canonical_identity
        self.snapshot = None
        self.state_identity = None
        self.state_ref = None
        self.proof_ref = None
        self.layernorm_index = None
        self.invocation_ordinal = None
        self.live_state_identity = None
        self.encounters = 0
        self.input_unchanged = None
        self.original_returned = False
        self.materialized_after_execution = False
        self._original = None

    @contextlib.contextmanager
    def installed(self):
        if self._original is not None:
            raise RuntimeError("pre-LayerNorm input capture is already installed")
        self._original = finish3l._layernorm_variance_state

        def wrapper(state, proof, label):
            if label != experiment.TARGET_LABEL:
                return self._original(state, proof, label)
            self.encounters += 1
            if self.encounters != 1:
                raise RuntimeError("target pre-LayerNorm input encountered twice")
            self.state_ref = state
            self.proof_ref = proof
            result = self._original(state, proof, label)
            self.original_returned = True
            return result

        finish3l._layernorm_variance_state = wrapper
        try:
            yield self
        finally:
            finish3l._layernorm_variance_state = self._original
            self._original = None

    def bind_invocation(self, *, layernorm_index: int,
                        invocation_ordinal: int, live_state_identity: dict):
        if (self.state_ref is None or self.proof_ref is None
                or self.encounters != 1 or not self.original_returned):
            raise RuntimeError("pre-LayerNorm input reference was not captured")
        if layernorm_index != INPUT_LAYERNORM_INDEX:
            raise RuntimeError("pre-LayerNorm capture index differs")
        if live_state_identity != self.expected_state_identity:
            raise RuntimeError("pre-LayerNorm authenticated input differs")
        if self.layernorm_index is not None:
            raise RuntimeError("pre-LayerNorm invocation was bound twice")
        self.layernorm_index = int(layernorm_index)
        self.invocation_ordinal = int(invocation_ordinal)
        self.live_state_identity = live_state_identity

    def materialize_after_execution(self):
        if (not self.original_returned or self.state_ref is None
                or self.proof_ref is None or self.layernorm_index is None):
            raise RuntimeError("pre-LayerNorm reference is incomplete")
        if self.materialized_after_execution:
            raise RuntimeError("pre-LayerNorm input was materialized twice")
        self.snapshot = _frontier_snapshot(self.state_ref, self.proof_ref)
        self.state_identity = experiment.state_identity(
            self.state_ref, self.proof_ref)
        self.input_unchanged = (
            self.state_identity == self.expected_state_identity
            and self.live_state_identity == self.expected_state_identity)
        if not self.input_unchanged:
            raise RuntimeError("pre-LayerNorm capture input was mutated")
        if self.state_identity.get("canonical_state_identity_sha256") != \
                self.required_canonical_identity:
            raise RuntimeError("pre-LayerNorm canonical input identity differs")
        self.materialized_after_execution = True

    def validate(self):
        if (self.encounters != 1 or self.snapshot is None
                or self.layernorm_index != INPUT_LAYERNORM_INDEX
                or self.invocation_ordinal is None
                or self.live_state_identity != self.state_identity
                or self.input_unchanged is not True
                or not self.original_returned
                or not self.materialized_after_execution):
            raise RuntimeError("pre-LayerNorm input capture is incomplete")


class NextLayerNormFailureCapture:
    """Semantically passive reference capture for the next domain failure."""

    def __init__(self):
        self.encounters = 0
        self.pre_reduction = None
        self.complete = None
        self.diagnostics = None
        self.layernorm_index = None
        self.reduction_label = None

    def __call__(self, *, state, proof, label, layernorm_index, diagnostics,
                 pre_reduction_state, pre_reduction_proof, reduction_label):
        self.encounters += 1
        if self.encounters != 1:
            raise RuntimeError("next LayerNorm failure captured more than once")
        if label != NEXT_STAGE or int(layernorm_index) != NEXT_LAYERNORM_INDEX:
            raise RuntimeError("next LayerNorm failure stage/index differs")
        if (reduction_label != NEXT_REDUCTION_LABEL
                or diagnostics.get("label") != NEXT_STAGE
                or diagnostics.get("domain_admissible") is not False):
            raise RuntimeError("next LayerNorm failure semantics differ")
        self.pre_reduction = (pre_reduction_state, pre_reduction_proof)
        self.complete = (state, proof)
        self.diagnostics = dict(diagnostics)
        self.layernorm_index = int(layernorm_index)
        self.reduction_label = reduction_label

    def materialize(self) -> dict:
        if (self.encounters != 1 or self.pre_reduction is None
                or self.complete is None or self.diagnostics is None):
            raise RuntimeError("next LayerNorm failure was not captured")
        pre = capture._snapshot(*self.pre_reduction)
        post = capture._snapshot(*self.complete)
        if (self.diagnostics["generator_count"]
                != int(post["weights"].shape[0] - 1)):
            raise RuntimeError("next LayerNorm diagnostic topology differs")
        return {
            "pre_last_reduction": pre,
            "post_last_reduction": post,
            "complete_layernorm_input_alias": "post_last_reduction",
            "stage_label": NEXT_STAGE,
            "layernorm_index": NEXT_LAYERNORM_INDEX,
            "reduction_label": NEXT_REDUCTION_LABEL,
            "diagnostics": self.diagnostics,
        }


def _frontier_snapshot(state, proof) -> dict:
    low, high = capture.sound._ranges(state)
    snapshot = {
        "weights": state.zonotope_w.detach().cpu().clone(),
        "range_low": low.detach().cpu().clone(),
        "range_high": high.detach().cpu().clone(),
        "proof": {
            "masks": list(proof.masks), "ids": list(proof.ids),
            "reasons": list(proof.reasons), "num_tokens": proof.num_tokens,
        },
    }
    weights = snapshot["weights"]
    generators = int(weights.shape[0] - 1)
    if (weights.ndim != 3 or weights.dtype != capture.sound.torch.float64
            or weights.device.type != "cpu"
            or snapshot["range_low"].dtype != capture.sound.torch.float64
            or snapshot["range_high"].dtype != capture.sound.torch.float64
            or tuple(snapshot["range_low"].shape) != (generators,)
            or tuple(snapshot["range_high"].shape) != (generators,)
            or len(proof.ids) != generators
            or len(proof.masks) != generators
            or len(proof.reasons) != generators
            or int(proof.num_tokens) != int(weights.shape[1])
            or len(set(proof.ids)) != generators
            or not bool(capture.sound.torch.isfinite(weights).all())
            or not bool(capture.sound.torch.isfinite(
                snapshot["range_low"]).all())
            or not bool(capture.sound.torch.isfinite(
                snapshot["range_high"]).all())
            or bool((snapshot["range_low"] > snapshot["range_high"]).any())):
        raise RuntimeError("FFN frontier snapshot topology/ranges differ")
    return snapshot


def _frontier_state_hashes(state: dict) -> dict:
    proof = state["proof"]
    range_sha = hashlib.sha256(
        state["range_low"].contiguous().numpy().tobytes()
        + state["range_high"].contiguous().numpy().tobytes()).hexdigest()
    ids_sha = capture._json_sha(proof["ids"])
    provenance_sha = capture._json_sha({
        "masks": proof["masks"], "reasons": proof["reasons"],
        "num_tokens": proof["num_tokens"],
    })
    reasons = proof["reasons"]
    return {
        "center_sha256": capture._tensor_sha(state["weights"][0]),
        "generator_sha256": capture._tensor_sha(state["weights"][1:]),
        "generator_ids_sha256": ids_sha,
        "ranges_sha256": range_sha,
        "provenance_sha256": provenance_sha,
        "generator_id_range_provenance_sha256": capture._json_sha({
            "generator_ids_sha256": ids_sha,
            "ranges_sha256": range_sha,
            "provenance_sha256": provenance_sha,
        }),
        "generator_count": len(reasons),
        "native_generator_count": sum(
            reason not in capture.NUMERICAL_REASONS for reason in reasons),
        "numerical_generator_count": sum(
            reason in capture.NUMERICAL_REASONS for reason in reasons),
        "token_count": int(state["weights"].shape[1]),
        "feature_dimension": int(state["weights"].shape[2]),
        "dtype": str(state["weights"].dtype).replace("torch.", ""),
    }


class FFNFrontierCapture:
    """Semantically passive Block-2 FFN boundary capture interposer."""

    def __init__(self):
        self.states = {}
        self.reductions = {}
        self._original_inject = None
        self._original_reduce = None
        self._original_reduce_pair = None

    def _put(self, name, state, proof):
        if name in self.states:
            raise RuntimeError(f"FFN frontier state captured twice: {name}")
        self.states[name] = _frontier_snapshot(state, proof)

    @contextlib.contextmanager
    def installed(self):
        sound = capture.sound
        if self._original_inject is not None:
            raise RuntimeError("FFN frontier capture is already installed")
        self._original_inject = sound._inject
        self._original_reduce = sound._maybe_reduce
        self._original_reduce_pair = sound._maybe_reduce_pair

        def inject_wrapper(output, proof, inputs, label, operations,
                           measurements, condition=1.0, reserve=None):
            result = self._original_inject(
                output, proof, inputs, label, operations, measurements,
                condition=condition, reserve=reserve)
            if label == "b2_relu":
                self._put("relu_raw_post_relaxation", output, proof)
            elif label == "b2_ffn_residual":
                self._put(
                    "residual_sum_pre_numerical_injection", output, proof)
            return result

        def reduce_wrapper(state, proof, label, reductions):
            names = FRONTIER_REDUCTION_STATES.get(label)
            before_records = len(reductions)
            if names is not None:
                self._put(names[0], state, proof)
            result = self._original_reduce(state, proof, label, reductions)
            if names is not None:
                self._put(names[1], result[0], result[1])
                self.reductions[label] = copy.deepcopy(
                    reductions[before_records:])
            return result

        def pair_wrapper(left, right, proof, label, reductions):
            before_records = len(reductions)
            if label == FRONTIER_PAIR_LABEL:
                self._put("residual_pair_pre_ffn", left, proof)
                self._put("residual_pair_pre_skip", right, proof)
            result = self._original_reduce_pair(
                left, right, proof, label, reductions)
            if label == FRONTIER_PAIR_LABEL:
                self._put("residual_pair_post_ffn", result[0], result[2])
                self._put("residual_pair_post_skip", result[1], result[2])
                self.reductions[label] = copy.deepcopy(
                    reductions[before_records:])
            return result

        sound._inject = inject_wrapper
        sound._maybe_reduce = reduce_wrapper
        sound._maybe_reduce_pair = pair_wrapper
        try:
            yield self
        finally:
            sound._inject = self._original_inject
            sound._maybe_reduce = self._original_reduce
            sound._maybe_reduce_pair = self._original_reduce_pair
            self._original_inject = None
            self._original_reduce = None
            self._original_reduce_pair = None

    def materialize(self) -> dict:
        if tuple(self.states) != FRONTIER_STATE_NAMES:
            raise RuntimeError(
                "FFN frontier state inventory differs: "
                f"{tuple(self.states)}")
        expected_reductions = (
            "b2_post_attention_layernorm", "b2_ffn_first", "b2_relu",
            "b2_ffn_second", FRONTIER_PAIR_LABEL, "b2_ffn_residual")
        if tuple(self.reductions) != expected_reductions:
            raise RuntimeError("FFN frontier reduction inventory differs")
        for side in ("pre", "post"):
            left = self.states[f"residual_pair_{side}_ffn"]
            right = self.states[f"residual_pair_{side}_skip"]
            if (left["proof"] != right["proof"]
                    or not capture.sound.torch.equal(
                        left["range_low"], right["range_low"])
                    or not capture.sound.torch.equal(
                        left["range_high"], right["range_high"])):
                raise RuntimeError(
                    f"FFN frontier {side} pair does not share IDs/ranges")
        return {
            "states": self.states,
            "reductions": self.reductions,
            "aliases": {
                "relu_pre_activation": "ffn_first_post_reduction",
                "known_exact_zero_pre_final_reduction":
                    "residual_sum_post_numerical_pre_reduction",
                "known_exact_zero_post_final_reduction":
                    "residual_sum_post_reduction",
            },
        }


def _relative(path: Path, root: Path) -> str:
    return os.path.relpath(path.resolve(), root.resolve())


def _persist_next_capture(output_root: Path,
                          captured: NextLayerNormFailureCapture,
                          source_capture: dict, result_path: Path) -> dict:
    materialized = captured.materialize()
    states = {
        "pre_last_reduction": materialized["pre_last_reduction"],
        "post_last_reduction": materialized["post_last_reduction"],
    }
    identity = {
        "property_id": PROPERTY_ID,
        "multiplier": MULTIPLIER,
        "tested_radius": TESTED_RADIUS,
        "tested_radius_hex": TESTED_RADIUS_HEX,
        "stage_label": NEXT_STAGE,
        "layernorm_index": NEXT_LAYERNORM_INDEX,
        "pinned_deept_revision": source_capture["pinned_deept_revision"],
        "scientific_manifest_sha256": source_capture[
            "scientific_manifest_sha256"],
        "production_manifest_sha256": source_capture[
            "production_manifest_sha256"],
        "source_set_model": source_capture["source_set_model"],
    }
    artifact_path = output_root / "next_layernorm_states.pt"
    capture._write_torch_atomic(artifact_path, {
        "schema": NEXT_CAPTURE_SCHEMA,
        "pinned_revision": source_capture["pinned_deept_revision"],
        "identity": identity,
        "states": states,
        "complete_layernorm_input_alias": "post_last_reduction",
        "reduction_label": NEXT_REDUCTION_LABEL,
        "diagnostics": materialized["diagnostics"],
    })
    variants = []
    for name, key in (
            ("pre_last_reduction", "pre_last_reduction"),
            ("post_last_reduction", "post_last_reduction"),
            ("complete_layernorm_input", "post_last_reduction")):
        variants.append({
            "capture_variant": name, "state_key": key,
            **capture._state_hashes(states[key]),
        })
    manifest_path = output_root / "next_layernorm_capture_manifest.json"
    manifest = _atomic_json(manifest_path, {
        "schema": NEXT_MANIFEST_SCHEMA,
        **identity,
        "tensor_artifact_path": _relative(artifact_path, output_root),
        "tensor_artifact_sha256": cluster_common.sha256(artifact_path),
        "artifact_identity": identity,
        "result_path": _relative(result_path, output_root),
        "result_sha256": cluster_common.sha256(result_path),
        "reduction_label": NEXT_REDUCTION_LABEL,
        "reduction_applied": (
            int(states["pre_last_reduction"]["weights"].shape[0])
            != int(states["post_last_reduction"]["weights"].shape[0])),
        "diagnostics": materialized["diagnostics"],
        "variants": variants,
    })
    oracle_path = output_root / "next_layernorm_psd_oracle_input.json"
    oracle_input = _atomic_json(oracle_path, {
        "schema": "CORET_PSD_LAYERNORM_VARIANCE_INPUT_V1",
        "property_id": PROPERTY_ID,
        "pinned_revision": source_capture["pinned_deept_revision"],
        "source_set_model": source_capture["source_set_model"],
        "capture_manifest": {
            "path": _relative(manifest_path, output_root),
            "sha256": cluster_common.sha256(manifest_path),
        },
        "evaluations": [{
            "property_id": PROPERTY_ID,
            "multiplier": MULTIPLIER,
            "tested_radius": TESTED_RADIUS,
            "stage_label": NEXT_STAGE,
            "layernorm_index": NEXT_LAYERNORM_INDEX,
            "token_index": int(materialized["diagnostics"][
                "minimum_token_index"]),
            "source_result": {
                "path": _relative(result_path, output_root),
                "sha256": cluster_common.sha256(result_path),
            },
            "complete_state": {
                "path": _relative(artifact_path, output_root),
                "sha256": cluster_common.sha256(artifact_path),
                "schema": NEXT_CAPTURE_SCHEMA,
                "state_key": "post_last_reduction",
            },
            "pre_reduction_state": {
                "path": _relative(artifact_path, output_root),
                "sha256": cluster_common.sha256(artifact_path),
                "schema": NEXT_CAPTURE_SCHEMA,
                "state_key": "pre_last_reduction",
            },
        }],
    })
    verified = _verify_next_capture(manifest_path)
    return {
        "artifact_path": str(artifact_path),
        "artifact_sha256": cluster_common.sha256(artifact_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": cluster_common.sha256(manifest_path),
        "manifest_record_sha256": manifest["record_sha256"],
        "oracle_input_path": str(oracle_path),
        "oracle_input_sha256": cluster_common.sha256(oracle_path),
        "oracle_input_record_sha256": oracle_input["record_sha256"],
        "diagnostics": materialized["diagnostics"],
        "pre_generator_count": int(
            states["pre_last_reduction"]["weights"].shape[0] - 1),
        "post_generator_count": int(
            states["post_last_reduction"]["weights"].shape[0] - 1),
        "verified_identity": verified,
    }


def _verify_ffn_frontier_capture(manifest_path: Path) -> dict:
    manifest = cluster_common.verified_json(manifest_path)
    if (manifest.get("schema") != FRONTIER_MANIFEST_SCHEMA
            or manifest.get("property_id") != PROPERTY_ID
            or manifest.get("multiplier") != MULTIPLIER
            or manifest.get("tested_radius") != TESTED_RADIUS
            or manifest.get("tested_radius_hex") != TESTED_RADIUS_HEX
            or manifest.get("stage_label") != FRONTIER_STAGE
            or manifest.get("pinned_deept_revision") !=
            capture.prefix.PINNED_REVISION
            or manifest.get("scientific_manifest_sha256") !=
            cluster_common.SCIENTIFIC_MANIFEST_SHA
            or manifest.get("production_manifest_sha256") !=
            cluster_common.PRODUCTION_MANIFEST_SHA
            or manifest.get("source_set_model") != capture.SOURCE_SET_MODEL):
        raise RuntimeError("FFN frontier manifest identity differs")
    artifact_path = (manifest_path.parent
                     / manifest["tensor_artifact_path"]).resolve()
    if cluster_common.sha256(artifact_path) != manifest[
            "tensor_artifact_sha256"]:
        raise RuntimeError("FFN frontier artifact SHA differs")
    payload = capture.sound.torch.load(
        artifact_path, map_location="cpu", weights_only=False)
    if (payload.get("schema") != FRONTIER_CAPTURE_SCHEMA
            or payload.get("pinned_revision") !=
            capture.prefix.PINNED_REVISION
            or payload.get("identity") != manifest.get("artifact_identity")
            or tuple((payload.get("states") or {}).keys()) !=
            FRONTIER_STATE_NAMES):
        raise RuntimeError("FFN frontier artifact identity/inventory differs")
    states = payload["states"]
    variants = manifest.get("variants")
    if (not isinstance(variants, list)
            or [row.get("state_key") for row in variants]
            != list(FRONTIER_STATE_NAMES)):
        raise RuntimeError("FFN frontier manifest state inventory differs")
    for row in variants:
        name = row["state_key"]
        actual = _frontier_state_hashes(states[name])
        for field, value in actual.items():
            if row.get(field) != value:
                raise RuntimeError(
                    f"FFN frontier state hash differs: {name}/{field}")
    reductions = payload.get("reductions")
    if (not isinstance(reductions, dict)
            or capture._json_sha(reductions) !=
            manifest.get("reduction_records_sha256")):
        raise RuntimeError("FFN frontier reduction records differ")
    aliases = payload.get("aliases")
    if aliases != manifest.get("aliases"):
        raise RuntimeError("FFN frontier aliases differ")
    for side in ("pre", "post"):
        left = states[f"residual_pair_{side}_ffn"]
        right = states[f"residual_pair_{side}_skip"]
        if (left["proof"] != right["proof"]
                or not capture.sound.torch.equal(
                    left["range_low"], right["range_low"])
                or not capture.sound.torch.equal(
                    left["range_high"], right["range_high"])):
            raise RuntimeError(
                f"FFN frontier persisted {side} pair identity differs")
    result_path = (manifest_path.parent / manifest["result_path"]).resolve()
    if cluster_common.sha256(result_path) != manifest["result_sha256"]:
        raise RuntimeError("FFN frontier result SHA differs")
    _validate_next_failure_result(campaign._verified_result(result_path))
    return {
        "schema": FRONTIER_MANIFEST_SCHEMA,
        "property_id": PROPERTY_ID,
        "state_count": len(states),
        "tensor_artifact_sha256": manifest["tensor_artifact_sha256"],
        "result_sha256": manifest["result_sha256"],
        "shared_pair_identity_verified": True,
    }


def _persist_ffn_frontier_capture(
        output_root: Path, captured: FFNFrontierCapture,
        source_capture: dict, result_path: Path) -> dict:
    materialized = captured.materialize()
    identity = {
        "property_id": PROPERTY_ID,
        "multiplier": MULTIPLIER,
        "tested_radius": TESTED_RADIUS,
        "tested_radius_hex": TESTED_RADIUS_HEX,
        "stage_label": FRONTIER_STAGE,
        "pinned_deept_revision": source_capture["pinned_deept_revision"],
        "scientific_manifest_sha256": source_capture[
            "scientific_manifest_sha256"],
        "production_manifest_sha256": source_capture[
            "production_manifest_sha256"],
        "source_set_model": source_capture["source_set_model"],
    }
    artifact_path = output_root / "block2_ffn_frontier_states.pt"
    capture._write_torch_atomic(artifact_path, {
        "schema": FRONTIER_CAPTURE_SCHEMA,
        "pinned_revision": source_capture["pinned_deept_revision"],
        "identity": identity,
        **materialized,
    })
    variants = [{
        "state_key": name,
        **_frontier_state_hashes(materialized["states"][name]),
    } for name in FRONTIER_STATE_NAMES]
    manifest_path = output_root / "block2_ffn_frontier_manifest.json"
    manifest = _atomic_json(manifest_path, {
        "schema": FRONTIER_MANIFEST_SCHEMA,
        **identity,
        "tensor_artifact_path": _relative(artifact_path, output_root),
        "tensor_artifact_sha256": cluster_common.sha256(artifact_path),
        "artifact_identity": identity,
        "result_path": _relative(result_path, output_root),
        "result_sha256": cluster_common.sha256(result_path),
        "aliases": materialized["aliases"],
        "reduction_records_sha256": capture._json_sha(
            materialized["reductions"]),
        "variants": variants,
    })
    verified = _verify_ffn_frontier_capture(manifest_path)
    return {
        "artifact_path": str(artifact_path),
        "artifact_sha256": cluster_common.sha256(artifact_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": cluster_common.sha256(manifest_path),
        "manifest_record_sha256": manifest["record_sha256"],
        "state_count": len(materialized["states"]),
        "verified_identity": verified,
    }


def _input_linkage_identity(*, state_hashes: dict, invocation_ordinal: int,
                            frontier_manifest_sha256: str,
                            output_state_hashes: dict,
                            source_capture: dict,
                            canonical_state_identity: dict | None = None) -> dict:
    revision = source_capture.get(
        "pinned_deept_revision", source_capture.get("pinned_revision"))
    fields = {
        "property_id": PROPERTY_ID,
        "tested_radius_hex": TESTED_RADIUS_HEX,
        "block_index": INPUT_BLOCK_INDEX,
        "layernorm_index": INPUT_LAYERNORM_INDEX,
        "stage": experiment.TARGET_LABEL,
        "parameter_name": INPUT_OPERATOR,
        "invocation_ordinal": int(invocation_ordinal),
        "input_state_identity_sha256": capture._json_sha(state_hashes),
        "existing_output_state_name": "post_attention_ln_pre_reduction",
        "existing_output_identity_sha256": capture._json_sha(
            output_state_hashes),
        "existing_frontier_manifest_sha256": frontier_manifest_sha256,
        "pinned_revision": revision,
        "checkpoint_sha256": capture.prefix.CHECKPOINT_SHA256,
    }
    if canonical_state_identity is not None:
        fields.update({
            "input_state_identity_schema":
                "CORET_FRONTIER_MANIFEST_STATE_ROW_V1",
            "canonical_state_identity_sha256": canonical_state_identity[
                "canonical_state_identity_sha256"],
        })
    fields["linkage_identity_sha256"] = capture._json_sha(fields)
    return fields


def _verify_pre_layernorm_input_capture(
        manifest_path: Path, *, allow_legacy_missing_canonical: bool = False,
        expected_canonical_identity: dict | None = None) -> dict:
    manifest = cluster_common.verified_json(manifest_path)
    operator = manifest.get("operator") or {}
    model = manifest.get("model_authentication") or {}
    state_record = manifest.get("state") or {}
    linkage = manifest.get("invocation_linkage") or {}
    reproduction = manifest.get("passive_reproduction") or {}
    if (manifest.get("schema") != INPUT_MANIFEST_SCHEMA
            or manifest.get("property_id") != PROPERTY_ID
            or manifest.get("tested_radius") != TESTED_RADIUS
            or manifest.get("tested_radius_hex") != TESTED_RADIUS_HEX
            or operator != {
                "block_index": INPUT_BLOCK_INDEX,
                "layernorm_index": INPUT_LAYERNORM_INDEX,
                "stage": experiment.TARGET_LABEL,
                "parameter_name": INPUT_OPERATOR,
                "invocation_ordinal": linkage.get("invocation_ordinal")}
            or model.get("pinned_revision") != capture.prefix.PINNED_REVISION
            or model.get("checkpoint_sha256") !=
            capture.prefix.CHECKPOINT_SHA256
            or model.get("scientific_manifest_sha256") !=
            cluster_common.SCIENTIFIC_MANIFEST_SHA
            or model.get("production_manifest_sha256") !=
            cluster_common.PRODUCTION_MANIFEST_SHA
            or linkage.get("verified") is not True
            or linkage.get("existing_output_state_name") !=
            "post_attention_ln_pre_reduction"
            or reproduction.get("psd_application_count") != 1
            or reproduction.get("psd_rejection_count") != 0
            or reproduction.get("next_failure_stage") != NEXT_STAGE
            or reproduction.get("next_failure_reason") !=
            finish3l.LAYERNORM_DOMAIN_REASON
            or reproduction.get("scientific_queries") != 1):
        raise RuntimeError("pre-LayerNorm input manifest identity differs")
    artifact_path = (manifest_path.parent
                     / manifest["tensor_artifact_path"]).resolve()
    if cluster_common.sha256(artifact_path) != state_record.get(
            "tensor_artifact_sha256"):
        raise RuntimeError("pre-LayerNorm input artifact SHA differs")
    payload = capture.sound.torch.load(
        artifact_path, map_location="cpu", weights_only=False)
    if (payload.get("schema") != INPUT_CAPTURE_SCHEMA
            or payload.get("identity") != manifest.get("artifact_identity")
            or set(payload.get("states") or {}) != {"pre_layernorm_input"}):
        raise RuntimeError("pre-LayerNorm input artifact identity differs")
    actual = _frontier_state_hashes(
        payload["states"]["pre_layernorm_input"])
    actual_canonical = _snapshot_state_identity(
        payload["states"]["pre_layernorm_input"])
    recorded_canonical = manifest.get("canonical_state_identity")
    if recorded_canonical is None:
        if (not allow_legacy_missing_canonical
                or expected_canonical_identity is None):
            raise RuntimeError(
                "pre-LayerNorm canonical state identity is absent")
        recorded_canonical = expected_canonical_identity
    if (not isinstance(recorded_canonical, dict)
            or actual_canonical != recorded_canonical
            or actual_canonical.get("canonical_state_identity_sha256") !=
            EXPECTED_INPUT_CANONICAL_STATE_IDENTITY_SHA256):
        raise RuntimeError("pre-LayerNorm canonical state identity differs")
    actual = {**actual,
              "source_ids_sha256": actual["generator_ids_sha256"],
              "hidden_dimension": actual["feature_dimension"]}
    for field, value in actual.items():
        if state_record.get(field) != value:
            raise RuntimeError(f"pre-LayerNorm input state differs: {field}")
    frontier_path = Path(linkage["existing_frontier_manifest_path"])
    if (not frontier_path.is_file()
            or cluster_common.sha256(frontier_path) !=
            linkage["existing_frontier_manifest_sha256"]):
        raise RuntimeError("linked frontier manifest SHA differs")
    frontier_manifest = cluster_common.verified_json(frontier_path)
    matches = [row for row in frontier_manifest.get("variants", [])
               if row.get("state_key") ==
               "post_attention_ln_pre_reduction"]
    if len(matches) != 1:
        raise RuntimeError("linked LayerNorm output state is absent")
    expected_linkage = _input_linkage_identity(
        state_hashes=actual,
        invocation_ordinal=linkage["invocation_ordinal"],
        frontier_manifest_sha256=linkage[
            "existing_frontier_manifest_sha256"],
        output_state_hashes={key: value for key, value in matches[0].items()
                             if key != "state_key"},
        source_capture=model,
        canonical_state_identity=(
            actual_canonical
            if linkage.get("input_state_identity_schema") is not None
            else None))
    if (linkage.get("linkage_identity_sha256") !=
            expected_linkage["linkage_identity_sha256"]
            or state_record.get("token_count") != matches[0].get("token_count")
            or state_record.get("feature_dimension") !=
            matches[0].get("feature_dimension")):
        raise RuntimeError("pre/output LayerNorm invocation linkage differs")
    return {"schema": INPUT_MANIFEST_SCHEMA, "verified": True,
            "artifact_sha256": state_record["tensor_artifact_sha256"],
            "linkage_identity_sha256": linkage["linkage_identity_sha256"],
            "canonical_state_identity": actual_canonical,
            "legacy_manifest_without_canonical": (
                manifest.get("canonical_state_identity") is None)}


def _persist_pre_layernorm_input_capture(
        output_root: Path, captured: PostAttentionLayerNormInputCapture,
        source_capture: dict, result: dict, frontier_record: dict,
        harness: LayerNormExperimentHarness) -> dict:
    captured.validate()
    snapshot = captured.snapshot
    canonical_state_identity = _snapshot_state_identity(snapshot)
    if canonical_state_identity != captured.state_identity:
        raise RuntimeError("persisted/PSD canonical state identity differs")
    state_hashes = _frontier_state_hashes(snapshot)
    state_hashes = {
        **state_hashes,
        "source_ids_sha256": state_hashes["generator_ids_sha256"],
        "hidden_dimension": state_hashes["feature_dimension"],
    }
    frontier_path = Path(frontier_record["manifest_path"])
    frontier_manifest = cluster_common.verified_json(frontier_path)
    matches = [row for row in frontier_manifest["variants"]
               if row["state_key"] == "post_attention_ln_pre_reduction"]
    if len(matches) != 1:
        raise RuntimeError("post-attention LayerNorm output linkage is absent")
    output_hashes = {key: value for key, value in matches[0].items()
                     if key != "state_key"}
    linkage_fields = _input_linkage_identity(
        state_hashes=state_hashes,
        invocation_ordinal=captured.invocation_ordinal,
        frontier_manifest_sha256=frontier_record["manifest_sha256"],
        output_state_hashes=output_hashes, source_capture=source_capture,
        canonical_state_identity=canonical_state_identity)
    identity = {
        "property_id": PROPERTY_ID, "tested_radius": TESTED_RADIUS,
        "tested_radius_hex": TESTED_RADIUS_HEX,
        "operator": {
            "block_index": INPUT_BLOCK_INDEX,
            "layernorm_index": INPUT_LAYERNORM_INDEX,
            "stage": experiment.TARGET_LABEL,
            "parameter_name": INPUT_OPERATOR,
            "invocation_ordinal": captured.invocation_ordinal},
        "pinned_deept_revision": source_capture["pinned_deept_revision"],
        "checkpoint_sha256": capture.prefix.CHECKPOINT_SHA256,
    }
    artifact_path = output_root / "pre_layernorm_input_state.pt"
    capture._write_torch_atomic(artifact_path, {
        "schema": INPUT_CAPTURE_SCHEMA, "identity": identity,
        "states": {"pre_layernorm_input": snapshot},
        "input_unchanged": captured.input_unchanged,
    })
    model_authentication = {
        "pinned_revision": source_capture["pinned_deept_revision"],
        "checkpoint_sha256": capture.prefix.CHECKPOINT_SHA256,
        "scientific_manifest_sha256": source_capture[
            "scientific_manifest_sha256"],
        "production_manifest_sha256": source_capture[
            "production_manifest_sha256"],
    }
    linkage = {
        "verified": True,
        "existing_output_state_name": "post_attention_ln_pre_reduction",
        "existing_frontier_manifest_path": str(frontier_path),
        **linkage_fields,
    }
    manifest_path = output_root / "pre_layernorm_input_manifest.json"
    manifest = _atomic_json(manifest_path, {
        "schema": INPUT_MANIFEST_SCHEMA,
        "property_id": PROPERTY_ID, "tested_radius": TESTED_RADIUS,
        "tested_radius_hex": TESTED_RADIUS_HEX,
        "operator": identity["operator"],
        "tensor_artifact_path": _relative(artifact_path, output_root),
        "artifact_identity": identity,
        "canonical_state_identity": canonical_state_identity,
        "state": {**state_hashes,
                  "tensor_artifact_sha256": cluster_common.sha256(
                      artifact_path)},
        "invocation_linkage": linkage,
        "model_authentication": model_authentication,
        "passive_reproduction": {
            "psd_application_count": harness.psd_certificate_applications,
            "psd_rejection_count": harness.psd_certificate_rejections,
            "next_failure_stage": result.get("failure_stage"),
            "next_failure_reason": result.get("failure_reason"),
            "scientific_queries": 1,
            "input_unchanged": captured.input_unchanged,
        },
    })
    verified = _verify_pre_layernorm_input_capture(manifest_path)
    return {"artifact_path": str(artifact_path),
            "artifact_sha256": cluster_common.sha256(artifact_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": cluster_common.sha256(manifest_path),
            "manifest_record_sha256": manifest["record_sha256"],
            "verified_identity": verified}


def _frontier_output_state_record(manifest_path: Path) -> dict:
    _verify_ffn_frontier_capture(manifest_path)
    manifest = cluster_common.verified_json(manifest_path)
    matches = [row for row in manifest["variants"]
               if row["state_key"] == "post_attention_ln_pre_reduction"]
    if len(matches) != 1:
        raise RuntimeError("authenticated frontier LayerNorm output is absent")
    return matches[0]


def _enforce_existing_frontier_linkage(
        output_root: Path, expected_manifest: Path,
        reproduced_manifest: Path) -> dict:
    """Persist a content-level diagnostic before any linkage hard failure."""
    comparison = linkage_diagnostic.compare_frontier_manifests(
        expected_manifest, reproduced_manifest)
    diagnostic_path = output_root / "layernorm_output_linkage_diagnostic.json"
    persisted = _atomic_json(diagnostic_path, comparison)
    if persisted.get("semantic_equal") is not True:
        first = persisted.get("first_semantic_difference")
        raise RuntimeError(
            "reproduced/job-2995 LayerNorm output identity differs"
            + (f" at {first}" if first else ""))
    return {
        "path": str(diagnostic_path),
        "sha256": cluster_common.sha256(diagnostic_path),
        "record_sha256": persisted["record_sha256"],
        "classification": persisted["classification"],
        "semantic_equal": True,
    }


def _authenticate_prior_experiment(report_path: Path) -> dict:
    report = cluster_common.verified_json(report_path)
    if (report.get("schema") != SCHEMA
            or report.get("verdict") !=
            "CORET_PSD_LAYERNORM_EXPERIMENT_COMPLETE"
            or report.get("property_id") != PROPERTY_ID
            or report.get("tested_radius") != TESTED_RADIUS
            or report.get("tested_radius_hex") != TESTED_RADIUS_HEX
            or report.get("terminal_status") != "UNCERTIFIED_DOMAIN_FAILURE"
            or report.get("generic_fallback_count") != 0):
        raise RuntimeError("prior PSD experiment identity differs")
    capture_record = report.get("capture_manifest") or {}
    oracle_record = report.get("oracle_report") or {}
    frontier_record = report.get("block2_ffn_frontier_capture") or {}
    paths = {}
    for name, record, path_field, sha_field in (
            ("capture_manifest", capture_record, "path", "sha256"),
            ("oracle_report", oracle_record, "path", "sha256"),
            ("frontier_manifest", frontier_record,
             "manifest_path", "manifest_sha256")):
        path = Path(record.get(path_field, "")).expanduser()
        if (not path.is_absolute() or not path.is_file()
                or cluster_common.sha256(path) != record.get(sha_field)):
            raise RuntimeError(f"prior PSD experiment {name} differs")
        paths[name] = path
    _frontier_output_state_record(paths["frontier_manifest"])
    return paths


def _validate_next_failure_result(result: dict) -> dict:
    diagnostic = result.get("domain_failure_diagnostic")
    if (result.get("terminal_status") != "UNCERTIFIED_DOMAIN_FAILURE"
            or result.get("scientific_evaluation_complete") is not True
            or result.get("certified_at_historical_radius") is not False
            or result.get("classification") != "FAILED_AT_HISTORICAL_RADIUS"
            or result.get("failure_category") !=
            "SOUND_LAYERNORM_DOMAIN_FAILURE"
            or result.get("failure_stage") != "block2_to_margin"
            or result.get("failure_reason") != finish3l.LAYERNORM_DOMAIN_REASON
            or result.get("generic_fallback_count") != 0
            or not isinstance(diagnostic, dict)
            or diagnostic.get("reason_code") != finish3l.LAYERNORM_DOMAIN_REASON
            or diagnostic.get("label") != NEXT_STAGE
            or diagnostic.get("domain_admissible") is not False):
        raise RuntimeError("next LayerNorm campaign result semantics differ")
    return diagnostic


def _verify_next_capture(manifest_path: Path) -> dict:
    manifest = cluster_common.verified_json(manifest_path)
    if (manifest.get("schema") != NEXT_MANIFEST_SCHEMA
            or manifest.get("property_id") != PROPERTY_ID
            or manifest.get("multiplier") != MULTIPLIER
            or manifest.get("tested_radius") != TESTED_RADIUS
            or manifest.get("tested_radius_hex") != TESTED_RADIUS_HEX
            or manifest.get("stage_label") != NEXT_STAGE
            or manifest.get("layernorm_index") != NEXT_LAYERNORM_INDEX
            or manifest.get("reduction_label") != NEXT_REDUCTION_LABEL
            or manifest.get("pinned_deept_revision") !=
            capture.prefix.PINNED_REVISION
            or manifest.get("scientific_manifest_sha256") !=
            cluster_common.SCIENTIFIC_MANIFEST_SHA
            or manifest.get("production_manifest_sha256") !=
            cluster_common.PRODUCTION_MANIFEST_SHA
            or manifest.get("source_set_model") != capture.SOURCE_SET_MODEL):
        raise RuntimeError("next LayerNorm capture manifest identity differs")
    artifact_path = (manifest_path.parent
                     / manifest["tensor_artifact_path"]).resolve()
    if cluster_common.sha256(artifact_path) != manifest["tensor_artifact_sha256"]:
        raise RuntimeError("next LayerNorm capture artifact SHA differs")
    payload = capture.sound.torch.load(
        artifact_path, map_location="cpu", weights_only=False)
    if (payload.get("schema") != NEXT_CAPTURE_SCHEMA
            or payload.get("pinned_revision") !=
            manifest.get("pinned_deept_revision")
            or payload.get("identity") != manifest.get("artifact_identity")
            or payload.get("complete_layernorm_input_alias") !=
            "post_last_reduction"
            or payload.get("reduction_label") != NEXT_REDUCTION_LABEL):
        raise RuntimeError("next LayerNorm capture artifact identity differs")
    states = payload.get("states")
    if set(states or {}) != {"pre_last_reduction", "post_last_reduction"}:
        raise RuntimeError("next LayerNorm capture state inventory differs")
    variants = manifest.get("variants")
    expected = (
        ("pre_last_reduction", "pre_last_reduction"),
        ("post_last_reduction", "post_last_reduction"),
        ("complete_layernorm_input", "post_last_reduction"),
    )
    if (not isinstance(variants, list) or len(variants) != len(expected)
            or [(row.get("capture_variant"), row.get("state_key"))
                for row in variants] != list(expected)):
        raise RuntimeError("next LayerNorm capture variant inventory differs")
    for variant, (_name, key) in zip(variants, expected):
        actual = capture._state_hashes(states[key])
        for field, value in actual.items():
            if variant.get(field) != value:
                raise RuntimeError(
                    f"next LayerNorm capture hash differs: {key}/{field}")
    pre_count = int(states["pre_last_reduction"]["weights"].shape[0] - 1)
    post_count = int(states["post_last_reduction"]["weights"].shape[0] - 1)
    if manifest.get("reduction_applied") is not (pre_count != post_count):
        raise RuntimeError("next LayerNorm reduction predicate differs")
    result_path = (manifest_path.parent / manifest["result_path"]).resolve()
    if cluster_common.sha256(result_path) != manifest["result_sha256"]:
        raise RuntimeError("next LayerNorm result SHA differs")
    result = campaign._verified_result(result_path)
    diagnostic = _validate_next_failure_result(result)
    if (manifest.get("diagnostics") != diagnostic
            or payload.get("diagnostics") != diagnostic
            or int(diagnostic.get("generator_count", -1)) != post_count
            or int(diagnostic.get("native_generator_count", -1))
            + int(diagnostic.get("numerical_generator_count", -1))
            != post_count):
        raise RuntimeError("next LayerNorm diagnostic/state identity differs")
    return {
        "schema": NEXT_MANIFEST_SCHEMA,
        "property_id": PROPERTY_ID,
        "stage_label": NEXT_STAGE,
        "layernorm_index": NEXT_LAYERNORM_INDEX,
        "reduction_label": NEXT_REDUCTION_LABEL,
        "pre_generator_count": pre_count,
        "post_generator_count": post_count,
        "tensor_artifact_sha256": manifest["tensor_artifact_sha256"],
        "result_sha256": manifest["result_sha256"],
    }


@contextlib.contextmanager
def _installed_finish_hook(harness: LayerNormExperimentHarness,
                           next_capture=None):
    original = finish3l.execute

    def wrapped(*args, **kwargs):
        kwargs["experimental_post_attention_layernorm"] = harness
        kwargs["experimental_layernorm_failure_capture"] = next_capture
        try:
            return original(*args, **kwargs)
        except Exception as error:
            harness.record_evaluator_exception(error)
            raise

    finish3l.execute = wrapped
    try:
        yield
    finally:
        finish3l.execute = original


def execute(campaign_root: Path, artifact_root: Path, capture_manifest: Path,
            oracle_report: Path, output_root: Path, device_index: int,
            existing_frontier_manifest: Path | None = None) -> dict:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("refusing to overwrite PSD experiment root")
    output_root.mkdir(parents=True, exist_ok=True)
    runtime_provenance_path = output_root / "runtime_provenance.json"
    runtime_provenance = _atomic_json(
        runtime_provenance_path, _runtime_provenance())
    runtime_provenance_ref = {
        "path": str(runtime_provenance_path),
        "sha256": cluster_common.sha256(runtime_provenance_path),
        "record_sha256": runtime_provenance["record_sha256"],
    }
    try:
        capture_identity = capture.verify_capture(capture_manifest)
        oracle_identity = _authenticate_oracle(oracle_report)
        if (oracle_identity["record"].get("state_artifact_hash")
                != capture_identity["tensor_artifact_sha256"]):
            raise RuntimeError("PSD oracle/capture state identity differs")
        expected_state_identity = _captured_state_identity(
            capture_manifest, capture_identity)
        row, _source, _identity = capture._prepare_row(
            campaign_root, artifact_root)
        if (row.get("property_id") != PROPERTY_ID
                or campaign._candidate_radius(row) != TESTED_RADIUS):
            raise RuntimeError("PSD experiment property/radius identity differs")
    except Exception as error:
        failure = {
            "schema": TRACE_SCHEMA,
            "property_id": PROPERTY_ID,
            "tested_radius": TESTED_RADIUS,
            "tested_radius_hex": TESTED_RADIUS_HEX,
            "expected_state_identity": None,
            "oracle_report_sha256": None,
            "counters": {
                "total_callback_invocations": 0,
                "non_target_invocations": 0,
                "target_stage_encounters": 0,
                "target_state_matches": 0,
                "psd_certificate_applications": 0,
                "psd_certificate_rejections": 0,
            },
            "invocations": [],
            "evaluator_exception": {
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback_tail": traceback.format_exc().splitlines()[-40:],
                "last_callback_trace_entry": None,
            },
            "runtime_provenance": runtime_provenance_ref,
        }
        trace_path = output_root / "psd_layernorm_invocation_trace.json"
        trace = _atomic_json(trace_path, failure)
        _atomic_json(output_root / "experiment_report.json", {
            "schema": SCHEMA,
            "verdict": "CORET_PSD_LAYERNORM_EXPERIMENT_FAIL_CLOSED",
            "property_id": PROPERTY_ID,
            "tested_radius": TESTED_RADIUS,
            "authentication_failure": f"{type(error).__name__}: {error}",
            "invocation_trace_path": str(trace_path),
            "invocation_trace_sha256": cluster_common.sha256(trace_path),
            "invocation_trace_record_sha256": trace["record_sha256"],
            "runtime_provenance": runtime_provenance_ref,
        })
        raise
    input_capture = PostAttentionLayerNormInputCapture(expected_state_identity)
    harness = LayerNormExperimentHarness(
        PROPERTY_ID, TESTED_RADIUS, expected_state_identity, oracle_identity,
        input_capture=input_capture)
    next_capture = NextLayerNormFailureCapture()
    frontier_capture = FFNFrontierCapture()
    execution_root = output_root / "scientific_execution"
    device = f"cuda:{device_index}"
    campaign._property_boundary_cleanup(device)
    campaign_error = None
    result = None
    try:
        with input_capture.installed(), frontier_capture.installed(), \
                _installed_finish_hook(harness, next_capture):
            result = campaign.execute_property(row, execution_root, device)
        input_capture.materialize_after_execution()
    except Exception as error:
        campaign_error = error
        if harness.evaluator_exception is None:
            harness.record_evaluator_exception(error)
    finally:
        campaign._property_boundary_cleanup(device)
    trace_path = output_root / "psd_layernorm_invocation_trace.json"
    trace = _atomic_json(trace_path, harness.record())
    if campaign_error is not None:
        _atomic_json(output_root / "experiment_report.json", {
            "schema": SCHEMA,
            "verdict": "CORET_PSD_LAYERNORM_EXPERIMENT_FAIL_CLOSED",
            "property_id": PROPERTY_ID,
            "tested_radius": TESTED_RADIUS,
            "campaign_exception": (
                f"{type(campaign_error).__name__}: {campaign_error}"),
            "invocation_trace_path": str(trace_path),
            "invocation_trace_sha256": cluster_common.sha256(trace_path),
            "invocation_trace_record_sha256": trace["record_sha256"],
            "counters": harness.counters(),
            "evaluator_exception": harness.evaluator_exception,
            "runtime_provenance": runtime_provenance_ref,
        })
        raise campaign_error
    try:
        harness.validate()
    except Exception as error:
        _atomic_json(output_root / "experiment_report.json", {
            "schema": SCHEMA,
            "verdict": "CORET_PSD_LAYERNORM_EXPERIMENT_FAIL_CLOSED",
            "property_id": PROPERTY_ID,
            "tested_radius": TESTED_RADIUS,
            "harness_failure": f"{type(error).__name__}: {error}",
            "invocation_trace_path": str(trace_path),
            "invocation_trace_sha256": cluster_common.sha256(trace_path),
            "invocation_trace_record_sha256": trace["record_sha256"],
            "counters": harness.counters(),
            "runtime_provenance": runtime_provenance_ref,
        })
        raise
    certificate_path = output_root / "psd_layernorm_certificate.json"
    certificate = _atomic_json(certificate_path, harness.certificates[0])
    result_path = execution_root / "properties" / PROPERTY_ID / "result.json"
    if not result_path.is_file():
        raise RuntimeError("PSD experiment property result is absent")
    result = cluster_common.verified_json(result_path)
    _validate_next_failure_result(result)
    next_capture_record = _persist_next_capture(
        output_root, next_capture, capture_identity, result_path)
    frontier_capture_record = _persist_ffn_frontier_capture(
        output_root, frontier_capture, capture_identity, result_path)
    linkage_frontier_record = frontier_capture_record
    if existing_frontier_manifest is not None:
        linkage_frontier_record = {
            "manifest_path": str(existing_frontier_manifest),
            "manifest_sha256": cluster_common.sha256(
                existing_frontier_manifest),
        }
    input_capture_record = _persist_pre_layernorm_input_capture(
        output_root, input_capture, capture_identity, result,
        linkage_frontier_record, harness)
    output_linkage = None
    if existing_frontier_manifest is not None:
        output_linkage = _enforce_existing_frontier_linkage(
            output_root, existing_frontier_manifest,
            Path(frontier_capture_record["manifest_path"]))
    return _atomic_json(output_root / "experiment_report.json", {
        "schema": SCHEMA,
        "verdict": "CORET_PSD_LAYERNORM_EXPERIMENT_COMPLETE",
        "property_id": PROPERTY_ID,
        "multiplier": MULTIPLIER,
        "tested_radius": TESTED_RADIUS,
        "tested_radius_hex": TESTED_RADIUS_HEX,
        "capture_manifest": {
            "path": str(capture_manifest),
            "sha256": cluster_common.sha256(capture_manifest),
            "identity": capture_identity,
        },
        "oracle_report": oracle_identity,
        "certificate_path": str(certificate_path),
        "certificate_sha256": cluster_common.sha256(certificate_path),
        "certificate_record_sha256": certificate["record_sha256"],
        "invocation_trace_path": str(trace_path),
        "invocation_trace_sha256": cluster_common.sha256(trace_path),
        "invocation_trace_record_sha256": trace["record_sha256"],
        "invocation_counters": harness.counters(),
        "evaluator_exception": harness.evaluator_exception,
        "property_result_path": str(result_path),
        "property_result_sha256": cluster_common.sha256(result_path),
        "terminal_status": result.get("terminal_status"),
        "certified": result.get("certified_at_historical_radius"),
        "final_sound_lower_margin": result.get("final_sound_lower_margin"),
        "next_failure_stage": result.get("failure_stage"),
        "next_failure_reason": result.get("failure_reason"),
        "generic_fallback_count": result.get("generic_fallback_count"),
        "next_layernorm_capture": next_capture_record,
        "block2_ffn_frontier_capture": frontier_capture_record,
        "post_attention_layernorm_input_capture": input_capture_record,
        "input_capture_linked_to_existing_job2995": (
            existing_frontier_manifest is not None),
        "job2995_output_linkage": output_linkage,
        "runtime_provenance": runtime_provenance_ref,
        "scientific_queries": 1,
        "bound_calls": 1,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--capture-manifest", type=Path)
    parser.add_argument("--oracle-report", type=Path)
    parser.add_argument("--prior-experiment-report", type=Path)
    parser.add_argument("--existing-frontier-manifest", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()
    if args.prior_experiment_report is not None:
        if (args.capture_manifest is not None or args.oracle_report is not None
                or args.existing_frontier_manifest is not None):
            parser.error("prior experiment report cannot be mixed with explicit inputs")
        inputs = _authenticate_prior_experiment(
            args.prior_experiment_report.resolve())
        capture_manifest = inputs["capture_manifest"]
        oracle_report = inputs["oracle_report"]
        existing_frontier = inputs["frontier_manifest"]
    else:
        if args.capture_manifest is None or args.oracle_report is None:
            parser.error("capture manifest and oracle report are required")
        capture_manifest = args.capture_manifest.resolve()
        oracle_report = args.oracle_report.resolve()
        existing_frontier = (args.existing_frontier_manifest.resolve()
                             if args.existing_frontier_manifest else None)
    report = execute(
        args.campaign_root.resolve(), args.artifact_root.resolve(),
        capture_manifest, oracle_report, args.output_root.resolve(),
        args.device_index, existing_frontier)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
