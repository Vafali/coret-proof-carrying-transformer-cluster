"""CPU-only tiny capture/linkage and exact-separator fixtures; no property runs."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import cluster_common as C
import capture_benchmark24_block2_output_zero_variance_v1 as CAP
import capture_s003_b2_ffn_residual_pre_reduction_v1 as P
import certify_s003_pre_reduction_separating_variance_v1 as S
import separating_variance_checker_v1 as K
from test_benchmark24_block2_output_zero_variance_v1 import (
    capture_fixture, s003_capture_fixture, rewrite_artifact, write_json)


@pytest.fixture
def comparison_fixture(s003_capture_fixture):
    post_manifest, post_artifact, payload = s003_capture_fixture
    post = payload["states"]["post_last_reduction"]
    # Correlated pre generators: all four move x0 and x127 together. Native
    # coordinate boxes lose that correlation. Only TWO nonzero coordinates
    # are needed to exercise the diagnostic; no 14k state is instantiated.
    post["weights"][1:] = 0.
    post["weights"][1, 8, 0] = 2.
    post["weights"][2, 8, -1] = 2.
    post["range_low"][:] = -1.
    post["range_high"][:] = 1.
    post["proof"].update(ids=["reduction_box::b2_ffn_residual::000000",
                              "reduction_box::b2_ffn_residual::000001"],
                         masks=[1 << 8] * 2,
                         reasons=["sound_fp64_coordinate_box_replacement_with_numerical"] * 2)
    rewrite_artifact(post_manifest, post_artifact, payload, component_hashes=True)
    _, post_auth = CAP.verify_capture(post_manifest)
    pre_weights = torch.zeros(5, 17, 128, dtype=torch.float64)
    pre_weights[0] = post["weights"][0]
    pre_weights[1:, 8, 0] = .5
    pre_weights[1:, 8, -1] = .5
    pre = {"weights": pre_weights, "range_low": torch.full((4,), -1., dtype=torch.float64),
           "range_high": torch.ones(4, dtype=torch.float64),
           "proof": {"ids": ["source0", "source1", "source2", "numerical0"],
                     "masks": [1 << 8] * 4,
                     "reasons": ["native_semantic"] * 3 + ["fp64_roundoff_coordinate_box"],
                     "num_tokens": 17}}
    record = {"operator": "b2_ffn_residual", "count_before": 4, "count_after": 2,
              "retained": 0, "absorbed": 4, "added_box_generators": 2, "support_inflation": 0.,
              "retained_ids": [], "absorbed_ids": list(pre["proof"]["ids"]),
              "replacement_ids": list(post["proof"]["ids"]),
              "selection_rule": "tiny fixture", "ranking_sha256": "tiny fixture"}
    root = post_manifest.parent / "pre"
    root.mkdir()
    artifact = root / "pre_b2_ffn_residual_state.pt"
    pre_payload = {"schema": P.ARTIFACT_SCHEMA, "identity": P.identity(),
                   "pinned_revision": payload["pinned_revision"], "states": {P.STATE_NAME: pre},
                   "native_reduction_record": record, "diagnostics": payload["diagnostics"]}
    torch.save(pre_payload, artifact)
    manifest = root / "capture_manifest.json"
    write_json(manifest, {"schema": P.MANIFEST_SCHEMA, "identity": P.identity(),
                         "tensor_artifact_path": artifact.name, "tensor_artifact_sha256": C.sha256(artifact),
                         "state_hashes": CAP.component_hashes(pre), "native_reduction_record": record,
                         "diagnostics": payload["diagnostics"], "post_capture": post_auth,
                         "observed_post_state_hashes": CAP.component_hashes(post),
                         "result_path": "../result.json", "result_sha256": C.sha256(root.parent / "result.json")})
    return manifest, artifact, pre_payload, post_manifest, post


def test_authenticates_all_pre_generators_against_existing_post(comparison_fixture):
    manifest, _, payload, _, _ = comparison_fixture
    pre, auth = P.verify_capture(manifest)
    assert len(pre["proof"]["ids"]) == 4 > CAP.EXPECTED_GENERATORS
    assert torch.equal(pre["weights"].view(torch.int64), payload["states"][P.STATE_NAME]["weights"].view(torch.int64))
    assert auth["identity"]["tested_radius"] == 0.00130126953125
    assert auth["token_index"] == 8 and auth["post_no_separator_evidence"]
    operands, _ = S.load_operands(manifest)
    assert len(operands[1]) == len(operands[2]) == len(operands[3]) == len(operands[4]) == 4
    assert len(operands[0]) == 128


@pytest.mark.parametrize("change", ["coefficient", "range", "id", "mask", "provenance", "omitted", "order"])
def test_pre_capture_tampering_rejected(comparison_fixture, change):
    manifest, artifact, payload, _, _ = comparison_fixture
    state = payload["states"][P.STATE_NAME]
    if change == "coefficient": state["weights"][1, 8, 0] += .125
    elif change == "range": state["range_low"][0] = -.5
    elif change == "id": state["proof"]["ids"][0] = "impostor"
    elif change == "mask": state["proof"]["masks"][0] = 0
    elif change == "provenance": state["proof"]["reasons"][0] = "different"
    elif change == "omitted": state["weights"] = state["weights"][:-1]
    elif change == "order": state["proof"]["ids"].reverse()
    torch.save(payload, artifact)
    value = C.verified_json(manifest)
    value["tensor_artifact_sha256"] = C.sha256(artifact)
    write_json(manifest, value)
    with pytest.raises(RuntimeError): P.verify_capture(manifest)


@pytest.mark.parametrize("field", ["property_id", "tested_radius", "token_ids", "source_artifacts",
                                  "producer_source_audit", "capture_boundary"])
def test_pre_capture_identity_tampering_rejected(comparison_fixture, field):
    manifest, _, _, _, _ = comparison_fixture
    value = C.verified_json(manifest)
    value["identity"][field] = "wrong"
    write_json(manifest, value)
    with pytest.raises(RuntimeError, match="frozen/source/property"): P.verify_capture(manifest)


def test_post_state_substitution_rejected(comparison_fixture):
    manifest, _, _, _, _ = comparison_fixture
    value = C.verified_json(manifest)
    value["observed_post_state_hashes"]["generator_sha256"] = "impostor"
    write_json(manifest, value)
    with pytest.raises(RuntimeError, match="rerun post-state"): P.verify_capture(manifest)


def test_reduction_metadata_mutation_rejected(comparison_fixture):
    manifest, artifact, payload, _, _ = comparison_fixture
    payload["native_reduction_record"]["count_before"] = 3
    torch.save(payload, artifact)
    value = C.verified_json(manifest)
    value.update(tensor_artifact_sha256=C.sha256(artifact),
                 native_reduction_record=payload["native_reduction_record"])
    write_json(manifest, value)
    with pytest.raises(RuntimeError): P.verify_capture(manifest)


def test_positive_pre_bound_requires_persisted_full_exact_replay(comparison_fixture, monkeypatch):
    manifest, _, _, _, post = comparison_fixture
    monkeypatch.setattr(S.S, "bounded_proposal", lambda *_: {
        "y": [1.] + [0.] * 126, "solver_status": 4, "objective_has_no_proof_authority": -999.})
    report = S.execute(manifest, manifest.parent / "pre_separator.json")
    assert report["final_status"] == S.S.CERTIFIED
    assert report["causal_decision"] == S.CAUSAL
    assert report["exact_check"]["all_generators_replayed"] == 4
    assert K.read_rational(report["exact_check"]["variance_lower"]) == Fraction(1, 64)
    operands, auth = S.load_operands(manifest)
    cert = C.verified_json(Path(report["certificate_path"]))["certificate"]
    assert K.verify_certificate(*operands, auth, cert)["verified"]
    # The post coordinate-box state actually admits the zero vector in this
    # synthetic fixture. No zero-variance solver is needed or invoked.
    zero = post["weights"][0, 8] - .5 * post["weights"][1, 8] + .5 * post["weights"][2, 8]
    assert torch.equal(zero, torch.zeros(128, dtype=torch.float64))
    # Exact coefficients: x0=1+2*(-1/2), x127=-1+2*(1/2).
    assert Fraction(1) + 2 * Fraction(-1, 2) == 0
    assert Fraction(-1) + 2 * Fraction(1, 2) == 0
    monkeypatch.setattr(S.S, "bounded_proposal", lambda *_: pytest.fail("replay only must not solve"))
    reread = S.execute(manifest, manifest.parent / "pre_replay.json", certificate_path=Path(report["certificate_path"]))
    assert reread["causal_decision"] == S.CAUSAL


def test_existing_bounded_lp_on_tiny_pre_fixture_then_exact_replay(comparison_fixture):
    manifest, _, _, _, _ = comparison_fixture
    report = S.execute(manifest, manifest.parent / "actual_tiny_lp.json", proposal_timeout=5.)
    assert report["final_status"] == S.S.CERTIFIED
    assert report["causal_decision"] == S.CAUSAL
    assert report["exact_check"]["all_generators_replayed"] == 4
    assert K.read_rational(report["exact_check"]["variance_lower"]) > 0


@pytest.mark.parametrize("kind", ["zero", "unseparated", "timeout"])
def test_numerical_success_cannot_authorize_positive_bound(comparison_fixture, monkeypatch, kind):
    manifest, _, _, _, _ = comparison_fixture
    # x1==x127 is NOT separated over this source box, unlike x0-x127.
    y = [0.] * 127
    if kind == "unseparated": y[1] = 1.
    result = {"y": None, "reason": "NUMERICAL_PROPOSAL_TIMEOUT"} if kind == "timeout" else {
        "y": y, "solver_status": 0, "objective_has_no_proof_authority": -100.}
    monkeypatch.setattr(S.S, "bounded_proposal", lambda *_: result)
    report = S.execute(manifest, manifest.parent / f"none_{kind}.json")
    assert report["final_status"] == S.S.NO_SEPARATOR
    assert report["causal_decision"] == S.NOT_ISOLATED
    assert report["exact_check"] is None and report["certificate_path"] is None


@pytest.mark.parametrize("field", ["delta", "variance_lower", "included_generator_count", "y_rationals"])
def test_optimistic_or_changed_certificate_rejected(comparison_fixture, field):
    manifest, _, _, _, _ = comparison_fixture
    operands, auth = S.load_operands(manifest)
    cert = K.construct_certificate(*operands, [Fraction(1)] + [Fraction(0)] * 126, auth)
    if field == "included_generator_count": cert[field] -= 1
    elif field == "y_rationals":
        cert[field][0] = K.rational(Fraction(2))
        cert["candidate_sha256"] = K.digest(cert[field])
    else: cert[field] = K.rational(K.read_rational(cert[field]) + Fraction(1, 2**52))
    cert["certificate_sha256"] = K.digest({k: v for k, v in cert.items() if k != "certificate_sha256"})
    with pytest.raises(RuntimeError, match="replay"):
        K.verify_certificate(*operands, auth, cert)


def test_passive_capture_and_rerun_linkage_with_mock_backend_only(comparison_fixture, monkeypatch):
    manifest, _, payload, post_manifest, post = comparison_fixture
    pre = payload["states"][P.STATE_NAME]
    pre_bits, post_bits = pre["weights"].view(torch.int64).clone(), post["weights"].view(torch.int64).clone()
    raw = {k: v for k, v in C.verified_json(post_manifest.parent / "result.json").items()
           if k != "record_sha256"}
    raw.update(runtime_seconds=0., final_generator_count=CAP.EXPECTED_GENERATORS)
    def mock_finish(*args, **kwargs):
        kwargs["experimental_layernorm_failure_capture"](
            state=post, proof=post["proof"], label=CAP.STAGE, layernorm_index=6,
            diagnostics=payload["diagnostics"], pre_reduction_state=pre,
            pre_reduction_proof=pre["proof"], reduction_label="b2_ffn_residual")
        return {"reductions": [payload["native_reduction_record"]]}
    finish = SimpleNamespace(execute=mock_finish)
    def mock_backend(row, workspace, artifact_root, device):
        finish.execute()
        P.B.atomic_record(workspace / "properties" / P.PROPERTY_ID / "result.json", raw)
        return raw, None
    monkeypatch.setitem(sys.modules, "run_sound_fp64_finish_3l_v1", finish)
    monkeypatch.setitem(sys.modules, "run_sound_fp64_3l_psd_state_capture_v1", SimpleNamespace(
        _snapshot=lambda state, proof: deepcopy(state),
        _write_torch_atomic=lambda path, value: torch.save(value, path)))
    monkeypatch.setattr(P.B, "_existing_backend", mock_backend)
    monkeypatch.setattr(P.B, "artifact_errors", lambda *_: [])
    output = manifest.parent / "mocked-new-capture"
    authenticated = P.execute(post_manifest, manifest.parent / "mock-inputs", output, "cuda:0")
    assert finish.execute is mock_finish
    assert authenticated["identity"]["capture_boundary"] == P.BOUNDARY
    assert torch.equal(pre["weights"].view(torch.int64), pre_bits)
    assert torch.equal(post["weights"].view(torch.int64), post_bits)
    saved = torch.load(output / "pre_b2_ffn_residual_state.pt", weights_only=False)
    assert set(saved["states"]) == {P.STATE_NAME}  # No second post-state dump.
    with pytest.raises(RuntimeError, match="NEW isolated"):
        P.execute(post_manifest, manifest.parent / "mock-inputs", output, "cuda:0")
    post["weights"][1, 8, 0] += .125
    with pytest.raises(RuntimeError, match="rerun post-state"):
        P.execute(post_manifest, manifest.parent / "mock-inputs", manifest.parent / "mismatch", "cuda:0")
    assert not (manifest.parent / "mismatch/pre_b2_ffn_residual_state.pt").exists()


def test_missing_pre_boundary_and_duplicate_callback_fail_loudly():
    capture = P.PassiveCapture()
    args = dict(state=object(), proof=object(), label=CAP.STAGE, layernorm_index=6, diagnostics={},
                pre_reduction_state=None, pre_reduction_proof=None, reduction_label="b2_ffn_residual")
    with pytest.raises(RuntimeError, match="missing/different/repeated"): capture(**args)
    args.update(pre_reduction_state=object(), pre_reduction_proof=object())
    capture(**args)
    with pytest.raises(RuntimeError, match="missing/different/repeated"): capture(**args)


def test_cannot_overwrite_authenticated_inputs(comparison_fixture):
    manifest, artifact, _, post_manifest, _ = comparison_fixture
    for path in (manifest, artifact, post_manifest):
        with pytest.raises(RuntimeError, match="overwrite"):
            S.execute(manifest, path)
