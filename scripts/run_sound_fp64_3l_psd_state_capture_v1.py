#!/usr/bin/env python3
"""One-property passive state capture for the PSD LayerNorm oracle.

The scientific execution remains the existing campaign path.  During this
standalone job only, an interposer records references around the final
``b2_attention_residual`` reduction and at the subsequent LayerNorm boundary.
The references are copied to CPU only after the expected domain-failure result
has returned.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "research_hab")]

import cluster_common
import coret_production_prefix_trace_v1 as prefix
import coret_sound_fp64_block0_feasibility_v1 as sound
import run_sound_fp64_3l_campaign as campaign
import run_sound_fp64_3l_radius_recovery_pilot as pilot
import run_sound_fp64_finish_3l_v1 as finish3l


SCHEMA = "CORET_PSD_LAYERNORM_STATE_CAPTURE_JOB_V1"
ARTIFACT_SCHEMA = "CORET_PSD_LAYERNORM_STATE_CAPTURE_V1"
MANIFEST_SCHEMA = "CORET_PSD_LAYERNORM_STATE_CAPTURE_MANIFEST_V1"
ORACLE_INPUT_SCHEMA = "CORET_PSD_LAYERNORM_VARIANCE_INPUT_V1"
PROPERTY_ID = "deept_table7_stdln3_s001_line1794_tok11"
HISTORICAL_RADIUS = 0.0008105468750000001
MULTIPLIER = "0.75"
TESTED_RADIUS = 0.00060791015625
TESTED_RADIUS_HEX = "0x1.3eb851eb851ecp-11"
REDUCTION_LABEL = "b2_attention_residual"
LAYERNORM_LABEL = "block2_post_attention"
SOURCE_SET_MODEL = "p100_linf_shared_generator_ids_cartesian_ranges"
EXPECTED_NOMINAL_VARIANCE = 1.0327479929312602
EXPECTED_PLAIN_LOWER = 0.0
EXPECTED_SOUND_LOWER = -22.155688844277535
NUMERICAL_REASONS = {
    "fp64_roundoff_coordinate_box",
    "sound_fp64_coordinate_box_replacement_with_numerical",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> dict:
    payload = dict(value)
    payload["record_sha256"] = cluster_common.canonical(payload)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return cluster_common.verified_json(path)


def _tensor_sha(tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    header = json.dumps({"dtype": str(array.dtype), "shape": list(array.shape)},
                        sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def _json_sha(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _snapshot(state, proof) -> dict:
    low, high = sound._ranges(state)
    payload = {
        "weights": state.zonotope_w.detach().cpu().clone(),
        "range_low": low.detach().cpu().clone(),
        "range_high": high.detach().cpu().clone(),
        "proof": {
            "masks": list(proof.masks), "ids": list(proof.ids),
            "reasons": list(proof.reasons), "num_tokens": proof.num_tokens,
        },
    }
    validate_snapshot(payload)
    return payload


def validate_snapshot(state: dict) -> None:
    torch = sound.torch
    if set(state) != {"weights", "range_low", "range_high", "proof"}:
        raise RuntimeError("capture state fields differ")
    weights, low, high = state["weights"], state["range_low"], state["range_high"]
    proof = state["proof"]
    if (not all(isinstance(value, torch.Tensor) for value in
                (weights, low, high))
            or weights.device.type != "cpu" or low.device.type != "cpu"
            or high.device.type != "cpu"
            or any(value.dtype != torch.float64 for value in
                   (weights, low, high))
            or weights.ndim != 3 or weights.shape[-1] != 128):
        raise RuntimeError("capture tensor shape/dtype differs")
    generators = int(weights.shape[0] - 1)
    if (tuple(low.shape) != (generators,)
            or tuple(high.shape) != (generators,)
            or len(proof.get("ids", ())) != generators
            or len(proof.get("masks", ())) != generators
            or len(proof.get("reasons", ())) != generators
            or int(proof.get("num_tokens", -1)) != weights.shape[1]
            or len(set(proof["ids"])) != generators
            or not bool(torch.isfinite(weights).all()
                        and torch.isfinite(low).all()
                        and torch.isfinite(high).all())
            or bool((low > high).any())):
        raise RuntimeError("capture state topology/ranges differ")


def _state_hashes(state: dict) -> dict:
    validate_snapshot(state)
    proof = state["proof"]
    range_sha = hashlib.sha256(
        state["range_low"].contiguous().numpy().tobytes()
        + state["range_high"].contiguous().numpy().tobytes()).hexdigest()
    ids_sha = _json_sha(proof["ids"])
    provenance_sha = _json_sha({
        "masks": proof["masks"], "reasons": proof["reasons"],
        "num_tokens": proof["num_tokens"],
    })
    combined = _json_sha({
        "generator_ids_sha256": ids_sha, "ranges_sha256": range_sha,
        "provenance_sha256": provenance_sha,
    })
    reasons = proof["reasons"]
    return {
        "center_sha256": _tensor_sha(state["weights"][0]),
        "generator_sha256": _tensor_sha(state["weights"][1:]),
        "generator_ids_sha256": ids_sha,
        "ranges_sha256": range_sha,
        "provenance_sha256": provenance_sha,
        "generator_id_range_provenance_sha256": combined,
        "generator_count": len(reasons),
        "native_generator_count": sum(
            reason not in NUMERICAL_REASONS for reason in reasons),
        "numerical_generator_count": sum(
            reason in NUMERICAL_REASONS for reason in reasons),
        "token_count": int(state["weights"].shape[1]),
        "hidden_dimension": int(state["weights"].shape[2]),
        "dtype": str(state["weights"].dtype).replace("torch.", ""),
    }


class PassiveCapture:
    def __init__(self):
        self.pre = None
        self.post = None
        self.complete = None
        self.reduction_witness = None
        self._post_object = None
        self._original_reduce = None
        self._original_layernorm = None

    @contextlib.contextmanager
    def installed(self, enabled: bool = True):
        if not enabled:
            yield self
            return
        if self._original_reduce is not None:
            raise RuntimeError("capture interposer is already installed")
        self._original_reduce = sound.sound_reduce
        self._original_layernorm = finish3l._layernorm_variance_state

        def reduce_wrapper(state, proof, cap, label):
            result = self._original_reduce(state, proof, cap, label)
            if label == REDUCTION_LABEL:
                if self.pre is not None:
                    raise RuntimeError("last reduction captured more than once")
                self.pre = (state, proof)
                self.post = (result[0], result[1])
                self._post_object = result[0]
                self.reduction_witness = copy.deepcopy(result[2])
            return result

        def layernorm_wrapper(state, proof, label):
            if label == LAYERNORM_LABEL:
                if self.complete is not None:
                    raise RuntimeError("LayerNorm input captured more than once")
                if self._post_object is None or state is not self._post_object:
                    raise RuntimeError("LayerNorm input is not last-reduction output")
                self.complete = (state, proof)
            return self._original_layernorm(state, proof, label)

        sound.sound_reduce = reduce_wrapper
        finish3l._layernorm_variance_state = layernorm_wrapper
        try:
            yield self
        finally:
            sound.sound_reduce = self._original_reduce
            finish3l._layernorm_variance_state = self._original_layernorm
            self._original_reduce = None
            self._original_layernorm = None

    def materialize(self) -> dict:
        if (self.pre is None or self.post is None or self.complete is None
                or self.reduction_witness is None):
            raise RuntimeError("required capture boundary was not reached")
        if self.complete[0] is not self.post[0] or self.complete[1] != self.post[1]:
            raise RuntimeError("post-reduction/LayerNorm state identity differs")
        pre = _snapshot(*self.pre)
        post = _snapshot(*self.post)
        return {"pre_last_reduction": pre, "post_last_reduction": post}


def _write_torch_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sound.torch.save(payload, temporary)
    os.replace(temporary, path)


def verify_capture(manifest_path: Path) -> dict:
    manifest = cluster_common.verified_json(manifest_path)
    if (manifest.get("schema") != MANIFEST_SCHEMA
            or manifest.get("property_id") != PROPERTY_ID
            or manifest.get("multiplier") != MULTIPLIER
            or manifest.get("tested_radius") != TESTED_RADIUS
            or manifest.get("tested_radius_hex") != TESTED_RADIUS_HEX
            or manifest.get("pinned_deept_revision") != prefix.PINNED_REVISION
            or manifest.get("source_set_model") != SOURCE_SET_MODEL):
        raise RuntimeError("capture manifest identity differs")
    artifact_path = (manifest_path.parent / manifest["tensor_artifact_path"]).resolve()
    if sha256(artifact_path) != manifest["tensor_artifact_sha256"]:
        raise RuntimeError("capture tensor artifact SHA differs")
    payload = sound.torch.load(
        artifact_path, map_location="cpu", weights_only=False)
    if (payload.get("schema") != ARTIFACT_SCHEMA
            or payload.get("pinned_revision") != prefix.PINNED_REVISION
            or payload.get("identity") != manifest["artifact_identity"]):
        raise RuntimeError("capture tensor artifact identity differs")
    states = payload.get("states")
    if set(states or {}) != {"pre_last_reduction", "post_last_reduction"}:
        raise RuntimeError("capture state inventory differs")
    for variant in manifest["variants"]:
        name, key = variant["capture_variant"], variant["state_key"]
        if name == "complete_layernorm_input":
            if key != "post_last_reduction":
                raise RuntimeError("complete LayerNorm state alias differs")
        elif name != key:
            raise RuntimeError("pre/post capture labels differ")
        actual = _state_hashes(states[key])
        for field, value in actual.items():
            if variant.get(field) != value:
                raise RuntimeError(f"capture variant hash differs: {name}/{field}")
    pre, post = states["pre_last_reduction"], states["post_last_reduction"]
    witness_path = (manifest_path.parent
                    / manifest["reduction_witness_path"]).resolve()
    if sha256(witness_path) != manifest["reduction_witness_sha256"]:
        raise RuntimeError("capture reduction witness SHA differs")
    witness = cluster_common.verified_json(witness_path)
    if (witness.get("label") != REDUCTION_LABEL
            or int(witness.get("input_generator_count", -1))
            != int(pre["weights"].shape[0] - 1)
            or int(witness.get("output_generator_count", -1))
            != int(post["weights"].shape[0] - 1)
            or witness.get("retained_ids")
            != post["proof"]["ids"][:len(witness.get("retained_ids", []))]):
        raise RuntimeError("capture reduction transition differs")
    result_path = (manifest_path.parent / manifest["result_path"]).resolve()
    if sha256(result_path) != manifest["result_sha256"]:
        raise RuntimeError("capture scientific result SHA differs")
    result = campaign._verified_result(result_path)
    if (result.get("property_id") != PROPERTY_ID
            or result.get("failure_category") !=
            "SOUND_LAYERNORM_DOMAIN_FAILURE"):
        raise RuntimeError("capture scientific result semantics differ")
    return manifest


def _relative(path: Path, root: Path) -> str:
    return os.path.relpath(path.resolve(), root.resolve())


def _prepare_row(campaign_root: Path, artifact_root: Path):
    prepared, identity = pilot._prepare(campaign_root, artifact_root)
    matches = [(row, source) for row, source in prepared
               if row["property_id"] == PROPERTY_ID]
    if len(matches) != 1:
        raise RuntimeError("capture property is absent or duplicated")
    row, source = copy.deepcopy(matches[0][0]), matches[0][1]
    historical = campaign._candidate_radius(row)
    tested = pilot._tested_radius(historical, 75, 100)
    if (historical != HISTORICAL_RADIUS or tested != TESTED_RADIUS
            or tested.hex() != TESTED_RADIUS_HEX):
        raise RuntimeError("capture radius identity differs")
    reference = row["cached_DeepT_reference"]
    reference[campaign.CANDIDATE_RADIUS_FIELD] = tested
    reference[campaign.CANDIDATE_RADIUS_HEX_FIELD] = tested.hex()
    return row, source, identity


def execute(campaign_root: Path, artifact_root: Path, output_root: Path,
            device_index: int) -> dict:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("refusing to overwrite PSD capture root")
    output_root.mkdir(parents=True, exist_ok=True)
    row, source, identity = _prepare_row(campaign_root, artifact_root)
    capture = PassiveCapture()
    execution_root = output_root / "scientific_execution"
    device = f"cuda:{device_index}"
    campaign._property_boundary_cleanup(device)
    try:
        with capture.installed():
            result = campaign.execute_property(row, execution_root, device)
        if (result.get("terminal_status") != "UNCERTIFIED_DOMAIN_FAILURE"
                or result.get("scientific_evaluation_complete") is not True
                or result.get("certified_at_historical_radius") is not False
                or result.get("failure_category") !=
                "SOUND_LAYERNORM_DOMAIN_FAILURE"
                or result.get("failure_stage") != "block2_to_margin"
                or result.get("generic_fallback_count") != 0):
            raise RuntimeError("capture rerun scientific outcome differs")
        diagnostic = result["domain_failure_diagnostic"]
        checks = (
            math.isclose(diagnostic["nominal_centered_second_moment"],
                         EXPECTED_NOMINAL_VARIANCE, rel_tol=1e-12, abs_tol=1e-12),
            diagnostic["plain_relational_variance_lower_bound"]
            == EXPECTED_PLAIN_LOWER,
            math.isclose(diagnostic["sound_variance_lower"],
                         EXPECTED_SOUND_LOWER, rel_tol=1e-12, abs_tol=1e-12),
        )
        if not all(checks):
            raise RuntimeError("capture rerun LayerNorm diagnostic differs")
        states = capture.materialize()
        witness = capture.reduction_witness
        if (witness["label"] != REDUCTION_LABEL
                or witness["output_generator_count"] != 14000):
            raise RuntimeError("captured last reduction differs")

        result_path = (execution_root / "properties" / PROPERTY_ID
                       / "result.json")
        result_sha = sha256(result_path)
        witness_path = output_root / "reduction_witness.json"
        witness_record = _atomic_json(witness_path, witness)
        campaign_identity = identity["campaign_identity"]
        artifact_identity = {
            "property_id": PROPERTY_ID, "multiplier": MULTIPLIER,
            "tested_radius": TESTED_RADIUS,
            "tested_radius_hex": TESTED_RADIUS_HEX,
            "stage_label": LAYERNORM_LABEL,
            "pinned_deept_revision": prefix.PINNED_REVISION,
            "scientific_manifest_sha256": campaign_identity[
                "scientific_manifest_sha256"],
            "production_manifest_sha256": campaign_identity[
                "production_manifest_sha256"],
            "source_set_model": SOURCE_SET_MODEL,
        }
        artifact_path = output_root / "psd_layernorm_states.pt"
        _write_torch_atomic(artifact_path, {
            "schema": ARTIFACT_SCHEMA,
            "pinned_revision": prefix.PINNED_REVISION,
            "identity": artifact_identity,
            "states": states,
            "complete_layernorm_input_alias": "post_last_reduction",
            "reduction_witness_sha256": sha256(witness_path),
        })
        artifact_sha = sha256(artifact_path)
        variants = []
        for name, key in (
                ("pre_last_reduction", "pre_last_reduction"),
                ("post_last_reduction", "post_last_reduction"),
                ("complete_layernorm_input", "post_last_reduction")):
            variants.append({
                "capture_variant": name, "state_key": key,
                **_state_hashes(states[key]),
            })
        manifest_path = output_root / "capture_manifest.json"
        manifest = _atomic_json(manifest_path, {
            "schema": MANIFEST_SCHEMA, **artifact_identity,
            "tensor_shape": list(states["post_last_reduction"][
                "weights"].shape),
            "tensor_artifact_path": _relative(artifact_path, output_root),
            "tensor_artifact_sha256": artifact_sha,
            "artifact_identity": artifact_identity,
            "result_path": _relative(result_path, output_root),
            "result_sha256": result_sha,
            "source_exact_radius_record_sha256": source["record_sha256"],
            "reduction_label": REDUCTION_LABEL,
            "reduction_witness_path": _relative(witness_path, output_root),
            "reduction_witness_sha256": sha256(witness_path),
            "reduction_witness_record_sha256": witness_record[
                "record_sha256"],
            "variants": variants,
        })
        oracle_path = output_root / "psd_oracle_input.json"
        _atomic_json(oracle_path, {
            "schema": ORACLE_INPUT_SCHEMA, "property_id": PROPERTY_ID,
            "pinned_revision": prefix.PINNED_REVISION,
            "source_set_model": SOURCE_SET_MODEL,
            "capture_manifest": {
                "path": _relative(manifest_path, output_root),
                "sha256": sha256(manifest_path),
            },
            "evaluations": [{
                "property_id": PROPERTY_ID, "multiplier": MULTIPLIER,
                "tested_radius": TESTED_RADIUS, "token_index": int(
                    diagnostic["minimum_token_index"]),
                "source_result": {
                    "path": _relative(result_path, output_root),
                    "sha256": result_sha,
                },
                "complete_state": {
                    "path": _relative(artifact_path, output_root),
                    "sha256": artifact_sha, "schema": ARTIFACT_SCHEMA,
                    "state_key": "post_last_reduction",
                },
                "pre_reduction_state": {
                    "path": _relative(artifact_path, output_root),
                    "sha256": artifact_sha, "schema": ARTIFACT_SCHEMA,
                    "state_key": "pre_last_reduction",
                },
            }],
        })
        verify_capture(manifest_path)
        return _atomic_json(output_root / "capture_report.json", {
            "schema": SCHEMA,
            "verdict": "CORET_PSD_STATE_CAPTURE_COMPLETE",
            "property_id": PROPERTY_ID, "multiplier": MULTIPLIER,
            "tested_radius": TESTED_RADIUS,
            "tested_radius_hex": TESTED_RADIUS_HEX,
            "scientific_outcome_reproduced": True,
            "capture_manifest_path": str(manifest_path),
            "capture_manifest_sha256": sha256(manifest_path),
            "tensor_artifact_path": str(artifact_path),
            "tensor_artifact_sha256": artifact_sha,
            "oracle_input_path": str(oracle_path),
            "oracle_input_sha256": sha256(oracle_path),
            "scientific_queries": 1, "bound_calls": 1,
        })
    finally:
        campaign._property_boundary_cleanup(device)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--campaign-root", required=True, type=Path)
    run.add_argument("--artifact-root", required=True, type=Path)
    run.add_argument("--output-root", required=True, type=Path)
    run.add_argument("--device-index", type=int, default=0)
    verify = sub.add_parser("verify")
    verify.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        result = verify_capture(args.manifest.resolve())
    else:
        result = execute(
            args.campaign_root.resolve(), args.artifact_root.resolve(),
            args.output_root.resolve(), args.device_index)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
