#!/usr/bin/env python3
"""One-property opt-in PSD-aware Block-2 LayerNorm experiment."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "research_hab")]

import cluster_common
import coret_psd_layernorm_experiment_v1 as experiment
import run_sound_fp64_3l_campaign as campaign
import run_sound_fp64_3l_psd_state_capture_v1 as capture
import run_sound_fp64_finish_3l_v1 as finish3l


SCHEMA = "CORET_PSD_LAYERNORM_EXPERIMENT_JOB_V2"
PROPERTY_ID = "deept_table7_stdln3_s001_line1794_tok11"
MULTIPLIER = "0.75"
TESTED_RADIUS = 0.00060791015625
TESTED_RADIUS_HEX = "0x1.3eb851eb851ecp-11"
TRACE_SCHEMA = "CORET_PSD_LAYERNORM_INVOCATION_TRACE_V2"


def _atomic_json(path: Path, value: dict) -> dict:
    payload = dict(value)
    payload["record_sha256"] = cluster_common.canonical(payload)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return cluster_common.verified_json(path)


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
                 expected_state_identity: dict, oracle_identity: dict):
        self.property_id = property_id
        self.radius = radius
        self.expected_state_identity = expected_state_identity
        self.oracle_identity = oracle_identity
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


@contextlib.contextmanager
def _installed_finish_hook(harness: LayerNormExperimentHarness):
    original = finish3l.execute

    def wrapped(*args, **kwargs):
        kwargs["experimental_post_attention_layernorm"] = harness
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
            oracle_report: Path, output_root: Path, device_index: int) -> dict:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("refusing to overwrite PSD experiment root")
    output_root.mkdir(parents=True, exist_ok=True)
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
        })
        raise
    harness = LayerNormExperimentHarness(
        PROPERTY_ID, TESTED_RADIUS, expected_state_identity, oracle_identity)
    execution_root = output_root / "scientific_execution"
    device = f"cuda:{device_index}"
    campaign._property_boundary_cleanup(device)
    campaign_error = None
    result = None
    try:
        with _installed_finish_hook(harness):
            result = campaign.execute_property(row, execution_root, device)
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
        })
        raise
    certificate_path = output_root / "psd_layernorm_certificate.json"
    certificate = _atomic_json(certificate_path, harness.certificates[0])
    result_path = execution_root / "properties" / PROPERTY_ID / "result.json"
    if not result_path.is_file():
        raise RuntimeError("PSD experiment property result is absent")
    result = cluster_common.verified_json(result_path)
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
        "scientific_queries": 1,
        "bound_calls": 1,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--capture-manifest", required=True, type=Path)
    parser.add_argument("--oracle-report", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()
    report = execute(
        args.campaign_root.resolve(), args.artifact_root.resolve(),
        args.capture_manifest.resolve(), args.oracle_report.resolve(),
        args.output_root.resolve(), args.device_index)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
