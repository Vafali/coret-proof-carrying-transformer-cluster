#!/usr/bin/env python3
"""Passive one-property capture. CPU authentication never executes a verifier."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path

import cluster_common as C
import run_transformer_benchmark24_v1 as B

PROPERTY_ID = "deept_table7_stdln3_s004_line216_tok04"
STAGE = "block2_output"
TOKEN_INDEX = 4  # Native tensor index, not a conversion from the perturbed token.
SEPARATOR_PROPERTY_ID = "deept_table7_stdln3_s003_line2031_tok08"
TARGET_TOKENS = {PROPERTY_ID: TOKEN_INDEX, SEPARATOR_PROPERTY_ID: 8}
EXPECTED_GENERATORS = 14000
ARTIFACT_SCHEMA = "CORET_BENCHMARK24_BLOCK2_OUTPUT_LN_STATE_V1"
MANIFEST_SCHEMA = "CORET_BENCHMARK24_BLOCK2_OUTPUT_LN_CAPTURE_MANIFEST_V1"
SOURCE_SET_MODEL = "p100_linf_shared_generator_ids_cartesian_ranges"


def identity(manifest, property_id=PROPERTY_ID):
    if property_id not in TARGET_TOKENS:
        raise RuntimeError("capture frozen property/radius/tokens/source identity differs")
    rows = [r for r in manifest["properties"] if r["property_id"] == property_id]
    if len(rows) != 1:
        raise RuntimeError("capture requires one exact frozen benchmark member")
    row = rows[0]
    expected = {"property_id": property_id, "stage_label": STAGE,
            "layernorm_index": 6, "reduction_label": "b2_ffn_residual",
            "token_index": TARGET_TOKENS[property_id], "token_index_convention": "zero_based_native_tensor",
            "tested_radius": row["tested_radius"], "tested_radius_hex": row["tested_radius_hex"],
            "token_ids": row["token_ids"], "sequence_length": row["sequence_length"],
            "clean_label": row["clean_label"], "perturbed_token_position": row["token_position"],
            "benchmark_manifest_sha256": manifest["manifest_sha256"],
            "source_population_sha256": manifest["source_population_sha256"],
            "source_artifacts": manifest["source_artifacts"],
            "frozen_execution_source_hashes": manifest["frozen_execution_source_hashes"],
            "source_set_model": SOURCE_SET_MODEL}
    if property_id == SEPARATOR_PROPERTY_ID:
        if row["tested_radius"] != 0.00130126953125 or row["tested_radius_hex"] != (0.00130126953125).hex():
            raise RuntimeError("s003 capture historical radius differs")
        # Authenticate the already accepted producer revision, not the obsolete
        # baseline producer. The frozen benchmark/model/data identities stay intact.
        import run_sound_fp64_separator_property_v1 as revision
        expected["producer_source_audit"] = revision.source_audit(manifest)
    return expected


def validate_state(state, expected):
    import torch  # CPU tensors only; no producer import or operator call.
    if set(state) != {"weights", "range_low", "range_high", "proof"}:
        raise RuntimeError("capture state fields differ")
    weights, low, high = (state[k] for k in ("weights", "range_low", "range_high"))
    shape = (EXPECTED_GENERATORS + 1, expected["sequence_length"], 128)
    if (any(not isinstance(x, torch.Tensor) or x.dtype != torch.float64 or x.device.type != "cpu"
            for x in (weights, low, high)) or tuple(weights.shape) != shape or
            tuple(low.shape) != (EXPECTED_GENERATORS,) or tuple(high.shape) != tuple(low.shape) or
            not all(bool(torch.isfinite(x).all()) for x in (weights, low, high)) or bool((low > high).any())):
        raise RuntimeError("capture state shape/dtype/ranges/finite values differ")
    proof = state["proof"]
    if (set(proof) != {"ids", "masks", "reasons", "num_tokens"} or
            any(len(proof[k]) != EXPECTED_GENERATORS for k in ("ids", "masks", "reasons")) or
            proof["num_tokens"] != expected["sequence_length"] or
            any(not isinstance(x, str) or not x for x in proof["ids"] + proof["reasons"]) or
            len(set(proof["ids"])) != EXPECTED_GENERATORS or
            any(type(x) is not int or not 0 <= x < (1 << expected["sequence_length"]) for x in proof["masks"])):
        raise RuntimeError("capture ordered IDs/provenance/topology differ")
    for token in range(expected["sequence_length"]):
        absent = [i for i, mask in enumerate(proof["masks"]) if not mask & (1 << token)]
        if absent and bool((weights[1:, token][absent] != 0).any()):
            raise RuntimeError("capture support mask omits nonzero coefficients")


def component_hashes(state):
    # Same hash schema already used by the existing zero-variance oracle.
    from decide_block2_output_zero_variance_v1 import _state_hashes
    return _state_hashes(state)


def _diagnostic(diagnostic, expected):
    token = expected["token_index"]
    if expected["property_id"] == SEPARATOR_PROPERTY_ID:
        failed = diagnostic.get("separator_attempt", {})
        if failed.get("admissible") is not False or failed.get("failed_token") != token:
            raise RuntimeError("capture unresolved separator token differs")
    if (diagnostic.get("label") != STAGE or diagnostic.get("domain_admissible") is not False or
            diagnostic.get("reason_code") != "SOUND_FP64_LAYERNORM_VARIANCE_DOMAIN_FAILURE" or
            type(diagnostic.get("minimum_token_index")) is not int or
            not 0 <= diagnostic["minimum_token_index"] < expected["sequence_length"] or
            (expected["property_id"] == PROPERTY_ID and diagnostic["minimum_token_index"] != token) or
            diagnostic.get("generator_count") != EXPECTED_GENERATORS or
            diagnostic.get("input_shape") != [EXPECTED_GENERATORS + 1, expected["sequence_length"], 128] or
            diagnostic.get("token_count") != expected["sequence_length"] or
            diagnostic.get("hidden_dimension") != 128 or
            type(diagnostic.get("minimum_coordinate_index")) is not int or
            not 0 <= diagnostic["minimum_coordinate_index"] < 128 or
            not B._finite(diagnostic.get("sound_variance_lower")) or diagnostic["sound_variance_lower"] > 0):
        raise RuntimeError("capture failure diagnostic/stage/token differs")


def _result(result, expected, diagnostic):
    if (result.get("schema") != "CORET_SOUND_FP64_3L_PROPERTY_RESULT_V1" or
            result.get("property_id") != expected["property_id"] or
            result.get("historical_candidate_radius_hex") != expected["tested_radius_hex"] or
            result.get("historical_candidate_radius") != expected["tested_radius"] or
            result.get("candidate_source") != "cached_DeepT_reference.certified_lower_endpoint_binary64" or
            result.get("clean_label") != expected["clean_label"] or
            result.get("terminal_status") != "UNCERTIFIED_DOMAIN_FAILURE" or
            result.get("scientific_evaluation_complete") is not True or
            result.get("certified_at_historical_radius") is not False or
            result.get("binary_search_performed") is not False or
            result.get("generic_fallback_count") != 0 or
            result.get("domain_failure_diagnostic") != diagnostic):
        raise RuntimeError("capture source-result identity/domain evidence differs")


def verify_capture(path):
    """Authenticate just the captured affine state, not the producer theorem."""
    import torch
    path = Path(path)
    manifest = C.verified_json(path)
    expected = identity(B.read_protocol(), manifest.get("identity", {}).get("property_id"))
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("identity") != expected:
        raise RuntimeError("capture frozen property/radius/tokens/source identity differs")
    artifact = (path.parent / manifest["tensor_artifact_path"]).resolve()
    if C.sha256(artifact) != manifest["tensor_artifact_sha256"]:
        raise RuntimeError("capture artifact SHA differs")
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    if (payload.get("schema") != ARTIFACT_SCHEMA or payload.get("identity") != expected or
            payload.get("pinned_revision") != expected["source_artifacts"]["pinned_DeepT_revision"] or
            set(payload.get("states", {})) != {"post_last_reduction"}):
        raise RuntimeError("capture artifact identity/state inventory differs")
    state = payload["states"]["post_last_reduction"]
    validate_state(state, expected)
    if component_hashes(state) != manifest.get("state_hashes"):
        raise RuntimeError("capture component hash differs")
    diagnostic = payload["diagnostics"]
    _diagnostic(diagnostic, expected)
    if diagnostic != manifest.get("diagnostics"):
        raise RuntimeError("capture diagnostic linkage differs")
    result_path = (path.parent / manifest["result_path"]).resolve()
    if C.sha256(result_path) != manifest["result_sha256"]:
        raise RuntimeError("capture source-result SHA differs")
    _result(C.verified_json(result_path), expected, diagnostic)
    return state, {"manifest_path": str(path.resolve()), "manifest_sha256": C.sha256(path),
                   "artifact_path": str(artifact), "artifact_sha256": manifest["tensor_artifact_sha256"],
                   "result_path": str(result_path), "result_sha256": manifest["result_sha256"],
                   "identity": expected, "token_index": expected["token_index"]}


class PassiveFailureCapture:
    def __init__(self):
        self.state = self.proof = self.diagnostic = None

    def __call__(self, *, state, proof, label, layernorm_index, diagnostics,
                 pre_reduction_state, pre_reduction_proof, reduction_label):
        if (self.state is not None or label != STAGE or layernorm_index != 6 or
                reduction_label != "b2_ffn_residual"):
            raise RuntimeError("capture boundary differs or was encountered twice")
        # No copies, numerical operations, parameter substitutions, or state writes.
        self.state, self.proof, self.diagnostic = state, proof, dict(diagnostics)


@contextmanager
def installed_callback(finish, capture):
    original = finish.execute
    def execute(*args, **kwargs):
        if kwargs.get("experimental_layernorm_failure_capture") is not None:
            raise RuntimeError("capture callback already installed")
        kwargs["experimental_layernorm_failure_capture"] = capture
        return original(*args, **kwargs)
    finish.execute = execute
    try:
        yield
    finally:
        finish.execute = original


def execute(property_id, artifact_root, output_root, device):
    if property_id not in TARGET_TOKENS or device != "cuda:0":
        raise RuntimeError("one frozen property / one visible GPU only")
    manifest = B.read_protocol()
    expected = identity(manifest, property_id)
    errors = B.artifact_errors(manifest, artifact_root)
    if property_id == PROPERTY_ID:
        errors += B.execution_source_errors(manifest)
    if errors:
        raise RuntimeError(f"capture preflight failed: {errors}")
    output_root = output_root.resolve()
    if (output_root.exists() or output_root == Path.home() or output_root == C.REPO or
            output_root.is_relative_to(artifact_root.resolve()) or output_root.is_relative_to(C.REPO / "frozen")):
        raise RuntimeError("capture requires a NEW isolated output root; never overwrite prior records")
    # Heavy producer imports are reachable only in this explicitly manual path.
    import run_sound_fp64_finish_3l_v1 as finish
    import run_sound_fp64_3l_psd_state_capture_v1 as existing_capture
    captured = PassiveFailureCapture()
    with installed_callback(finish, captured):
        if property_id == SEPARATOR_PROPERTY_ID:
            import run_sound_fp64_separator_property_v1 as revision
            revision.execute(property_id, artifact_root, output_root / "execution", device)
        else:
            B.run_one(manifest, property_id, artifact_root, output_root / "execution", device)
    if captured.state is None:
        raise RuntimeError("expected block2_output failure callback was not reached")
    _diagnostic(captured.diagnostic, expected)
    # Materialize only after the unchanged property execution has returned.
    state = existing_capture._snapshot(captured.state, captured.proof)
    validate_state(state, expected)
    result_path = output_root / "execution/producer/properties" / property_id / "result.json"
    _result(C.verified_json(result_path), expected, captured.diagnostic)
    output_root.mkdir(parents=True, exist_ok=True)
    artifact = output_root / "pre_block2_output_layernorm_state.pt"
    existing_capture._write_torch_atomic(artifact, {
        "schema": ARTIFACT_SCHEMA, "identity": expected,
        "pinned_revision": expected["source_artifacts"]["pinned_DeepT_revision"],
        "states": {"post_last_reduction": state}, "diagnostics": captured.diagnostic})
    B.atomic_record(output_root / "capture_manifest.json", {
        "schema": MANIFEST_SCHEMA, "identity": expected,
        "tensor_artifact_path": artifact.name, "tensor_artifact_sha256": C.sha256(artifact),
        "state_hashes": component_hashes(state), "diagnostics": captured.diagnostic,
        "result_path": str(result_path.relative_to(output_root)), "result_sha256": C.sha256(result_path)})
    _, authenticated = verify_capture(output_root / "capture_manifest.json")
    return authenticated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--property-id", required=True, choices=sorted(TARGET_TOKENS))
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0", choices=["cuda:0"])
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        parser.error("capture requires CUDA_VISIBLE_DEVICES=0 and one allocated GPU")
    print(json.dumps(execute(args.property_id, args.artifact_root, args.output_root, args.device), indent=2), flush=True)


if __name__ == "__main__":
    main()
