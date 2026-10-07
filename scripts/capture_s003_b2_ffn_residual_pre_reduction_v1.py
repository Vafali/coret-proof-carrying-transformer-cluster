#!/usr/bin/env python3
"""Passive pre-b2_ffn_residual capture, linked to the existing s003 post-state."""
from contextlib import contextmanager
import argparse
import json
import os
from pathlib import Path

import cluster_common as C
import capture_benchmark24_block2_output_zero_variance_v1 as CAP
import run_transformer_benchmark24_v1 as B
import run_sound_fp64_separator_property_v1 as REV

PROPERTY_ID = CAP.SEPARATOR_PROPERTY_ID
TOKEN_INDEX = 8
BOUNDARY = "immediately_before_b2_ffn_residual_reduction"
STATE_NAME = "pre_b2_ffn_residual"
ARTIFACT_SCHEMA = "CORET_S003_B2_FFN_RESIDUAL_PRE_REDUCTION_STATE_V1"
MANIFEST_SCHEMA = "CORET_S003_B2_FFN_RESIDUAL_PRE_REDUCTION_CAPTURE_V1"


def identity():
    return {**CAP.identity(B.read_protocol(), PROPERTY_ID),
            "capture_boundary": BOUNDARY,
            "capture_adapter_sha256": C.sha256(Path(__file__)),
            "capture_validation_sha256": C.sha256(Path(CAP.__file__))}


def validate_transition(pre, post, record, expected):
    """Authenticate the reported boundary/membership; no reduction is rerun.

    This is linkage validation, not a new independent reduction theorem. Native
    production already checks reduction soundness before reaching the callback.
    The separate variance certificate independently checks the captured box.
    """
    import torch
    before, after = record.get("count_before"), record.get("count_after")
    if (record.get("operator") != "b2_ffn_residual" or type(before) is not int or type(after) is not int
            or before <= CAP.EXPECTED_GENERATORS or after != CAP.EXPECTED_GENERATORS):
        raise RuntimeError("pre/post reduction boundary/count differs")
    CAP.validate_state(pre, expected, generator_count=before)
    CAP.validate_state(post, expected)
    kept, removed, boxes = (record.get(k) for k in ("retained_ids", "absorbed_ids", "replacement_ids"))
    if (any(not isinstance(x, list) or any(not isinstance(i, str) for i in x)
            for x in (kept, removed, boxes)) or
            record.get("retained") != len(kept) or record.get("absorbed") != len(removed) or
            record.get("added_box_generators") != len(boxes) or
            len(kept) + len(removed) != before or len(kept) + len(boxes) != after or
            len(set(kept + removed)) != before or set(kept + removed) != set(pre["proof"]["ids"]) or
            kept + boxes != post["proof"]["ids"] or
            not B._finite(record.get("support_inflation")) or record["support_inflation"] < 0):
        raise RuntimeError("pre/post reduction ordered membership differs")
    positions = {i: r for r, i in enumerate(pre["proof"]["ids"])}
    # Frozen reduction protects all constrained ranges. Removed rows therefore
    # have zero midpoint: the real center is unchanged, though adding zero may
    # legitimately change a signed-zero bit. Post-versus-prior comparison below
    # remains bitwise strict; no assumption about pre/post signed zero is needed.
    if any(pre["range_low"][positions[i]].item() != -1. or pre["range_high"][positions[i]].item() != 1.
           for i in removed):
        raise RuntimeError("frozen reduction removed a protected ranged generator")
    if not torch.equal(pre["weights"][0], post["weights"][0]):
        raise RuntimeError("pre/post reduction zero-midpoint center differs")
    for output_row, identifier in enumerate(kept):
        input_row = positions[identifier]
        if (not torch.equal(pre["weights"][input_row + 1].view(torch.int64),
                            post["weights"][output_row + 1].view(torch.int64)) or
                any(pre[k][input_row].view(torch.int64).item() != post[k][output_row].view(torch.int64).item()
                    for k in ("range_low", "range_high")) or
                any(pre["proof"][k][input_row] != post["proof"][k][output_row] for k in ("masks", "reasons"))):
            raise RuntimeError("pre/post retained coefficient/range/provenance differs")


class PassiveCapture:
    def __init__(self):
        self.pre = self.pre_proof = self.post = self.post_proof = self.diagnostic = None
        self.reduction = None

    def __call__(self, *, state, proof, label, layernorm_index, diagnostics,
                 pre_reduction_state, pre_reduction_proof, reduction_label):
        if (self.pre is not None or label != CAP.STAGE or layernorm_index != 6 or
                reduction_label != "b2_ffn_residual" or pre_reduction_state is None or
                pre_reduction_proof is None):
            raise RuntimeError("pre-reduction capture boundary missing/different/repeated")
        # References only; no clone, arithmetic, solver, or state mutation here.
        self.pre, self.pre_proof = pre_reduction_state, pre_reduction_proof
        self.post, self.post_proof, self.diagnostic = state, proof, dict(diagnostics)


@contextmanager
def installed_capture(finish, capture):
    original = finish.execute
    def execute(*args, **kwargs):
        if kwargs.get("experimental_layernorm_failure_capture") is not None:
            raise RuntimeError("failure capture callback already installed")
        kwargs["experimental_layernorm_failure_capture"] = capture
        result = original(*args, **kwargs)
        records = [r for r in result.get("reductions", []) if r.get("operator") == "b2_ffn_residual"]
        if len(records) != 1:
            raise RuntimeError("one actual b2_ffn_residual reduction record required")
        capture.reduction = dict(records[0])
        return result
    finish.execute = execute
    try:
        yield
    finally:
        finish.execute = original


def verify_capture(path):
    import torch
    path = Path(path).resolve()
    manifest = C.verified_json(path)
    expected = identity()
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("identity") != expected:
        raise RuntimeError("pre-reduction frozen/source/property identity differs")
    prior_path = Path(manifest["post_capture"]["manifest_path"])
    if C.sha256(prior_path) != manifest["post_capture"]["manifest_sha256"]:
        raise RuntimeError("linked post-capture manifest SHA differs")
    post, post_auth = CAP.verify_capture(prior_path)
    if post_auth != manifest["post_capture"] or post_auth["identity"] != CAP.identity(B.read_protocol(), PROPERTY_ID):
        raise RuntimeError("linked post-capture identity differs")
    if manifest.get("observed_post_state_hashes") != CAP.component_hashes(post):
        raise RuntimeError("rerun post-state differs from existing authenticated capture")
    artifact = (path.parent / manifest["tensor_artifact_path"]).resolve()
    if C.sha256(artifact) != manifest["tensor_artifact_sha256"]:
        raise RuntimeError("pre-reduction artifact SHA differs")
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    if (payload.get("schema") != ARTIFACT_SCHEMA or payload.get("identity") != expected or
            payload.get("pinned_revision") != expected["source_artifacts"]["pinned_DeepT_revision"] or
            set(payload.get("states", {})) != {STATE_NAME} or
            payload.get("native_reduction_record") != manifest.get("native_reduction_record")):
        raise RuntimeError("pre-reduction artifact identity/inventory/record differs")
    pre = payload["states"][STATE_NAME]
    validate_transition(pre, post, payload["native_reduction_record"], expected)
    if CAP.component_hashes(pre) != manifest.get("state_hashes"):
        raise RuntimeError("pre-reduction component hash differs")
    diagnostic = payload["diagnostics"]
    CAP._diagnostic(diagnostic, expected)
    if diagnostic != manifest.get("diagnostics"):
        raise RuntimeError("pre-reduction diagnostic linkage differs")
    result_path = (path.parent / manifest["result_path"]).resolve()
    if C.sha256(result_path) != manifest["result_sha256"]:
        raise RuntimeError("pre-reduction source-result SHA differs")
    CAP._result(C.verified_json(result_path), expected, diagnostic)
    post_diagnostic = C.verified_json(prior_path)["diagnostics"]
    post_attempt = post_diagnostic.get("separator_attempt", {})
    return pre, {"manifest_path": str(path), "manifest_sha256": C.sha256(path),
                 "artifact_path": str(artifact), "artifact_sha256": manifest["tensor_artifact_sha256"],
                 "result_path": str(result_path), "result_sha256": manifest["result_sha256"],
                 "identity": expected, "token_index": TOKEN_INDEX,
                 "state_hashes": manifest["state_hashes"], "post_capture": post_auth,
                 "post_no_separator_evidence": (post_attempt.get("admissible") is False and
                                                post_attempt.get("failed_token") == TOKEN_INDEX),
                 "post_diagnostic": post_diagnostic,
                 "native_reduction_record": manifest["native_reduction_record"]}


def execute(post_manifest, artifact_root, output_root, device):
    if device != "cuda:0":
        raise RuntimeError("one visible GPU only")
    expected = identity()
    prior_post, prior_auth = CAP.verify_capture(post_manifest)
    if prior_auth["identity"] != CAP.identity(B.read_protocol(), PROPERTY_ID):
        raise RuntimeError("existing post-state must be the exact s003/token8 capture")
    output_root, artifact_root = Path(output_root).resolve(), Path(artifact_root).resolve()
    if (output_root.exists() or output_root in (Path.home(), C.REPO) or
            output_root.is_relative_to(artifact_root) or output_root.is_relative_to(C.REPO / "frozen")):
        raise RuntimeError("requires a NEW isolated pre-reduction capture root")
    prior_hashes = CAP.component_hashes(prior_post)
    del prior_post
    import run_sound_fp64_finish_3l_v1 as finish
    import run_sound_fp64_3l_psd_state_capture_v1 as snapshot
    captured = PassiveCapture()
    with installed_capture(finish, captured):
        REV.execute(PROPERTY_ID, artifact_root, output_root / "execution", device)
    if captured.pre is None:
        raise RuntimeError("expected pre-b2_ffn_residual failure callback not reached")
    CAP._diagnostic(captured.diagnostic, expected)
    # Snapshot only after unchanged execution returns. Persist only the pre-state.
    post = snapshot._snapshot(captured.post, captured.post_proof)
    if CAP.component_hashes(post) != prior_hashes:
        raise RuntimeError("rerun post-state differs from existing authenticated capture; comparison rejected")
    pre = snapshot._snapshot(captured.pre, captured.pre_proof)
    validate_transition(pre, post, captured.reduction, expected)
    del post
    result_path = output_root / "execution/producer/properties" / PROPERTY_ID / "result.json"
    CAP._result(C.verified_json(result_path), expected, captured.diagnostic)
    output_root.mkdir(parents=True, exist_ok=True)
    artifact = output_root / "pre_b2_ffn_residual_state.pt"
    snapshot._write_torch_atomic(artifact, {
        "schema": ARTIFACT_SCHEMA, "identity": expected,
        "pinned_revision": expected["source_artifacts"]["pinned_DeepT_revision"],
        "states": {STATE_NAME: pre}, "native_reduction_record": captured.reduction,
        "diagnostics": captured.diagnostic})
    B.atomic_record(output_root / "capture_manifest.json", {
        "schema": MANIFEST_SCHEMA, "identity": expected,
        "tensor_artifact_path": artifact.name, "tensor_artifact_sha256": C.sha256(artifact),
        "state_hashes": CAP.component_hashes(pre), "native_reduction_record": captured.reduction,
        "diagnostics": captured.diagnostic, "post_capture": prior_auth,
        "observed_post_state_hashes": prior_hashes,
        "result_path": str(result_path.relative_to(output_root)), "result_sha256": C.sha256(result_path)})
    _, authenticated = verify_capture(output_root / "capture_manifest.json")
    return authenticated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--post-capture-manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda:0"], default="cuda:0")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        parser.error("requires CUDA_VISIBLE_DEVICES=0 and one allocated GPU")
    print(json.dumps(execute(args.post_capture_manifest, args.artifact_root, args.output_root, args.device),
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
