from __future__ import annotations

import copy
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace
import sys
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if "gmpy2" not in sys.modules:
    sys.modules["gmpy2"] = types.SimpleNamespace()
SPEC = importlib.util.spec_from_file_location(
    "ln_input_capture_runner",
    ROOT / "scripts/run_sound_fp64_3l_psd_layernorm_experiment_v1.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class State:
    def __init__(self, generators=2):
        self.zonotope_w = torch.zeros(
            generators + 1, 2, 128, dtype=torch.float64)
        self.zonotope_w[0, 0, 0] = 1.25
        self.error_term_range_low = torch.full(
            (generators,), -1.0, dtype=torch.float64)
        self.error_term_range_high = torch.ones(
            generators, dtype=torch.float64)
        self.num_error_terms = generators
        self.num_words = 2
        self.word_embedding_size = 128
        self.device = torch.device("cpu")


def proof(generators=2):
    return RUNNER.experiment.structural.SupportProof(
        tuple([3] * generators),
        tuple(f"g{index}" for index in range(generators)),
        tuple(["native_semantic"] * generators), 2)


def _capture(monkeypatch, mutate=False, finalize=True):
    state, support = State(), proof()
    expected = RUNNER.experiment.state_identity(state, support)
    recorder = RUNNER.PostAttentionLayerNormInputCapture(
        expected, expected["canonical_state_identity_sha256"])
    observed = {}

    def original(value, proof_value, label):
        observed["snapshot_exists_before_semantics"] = recorder.snapshot is not None
        observed["state_is_identical"] = value is state
        observed["proof_is_identical"] = proof_value is support
        observed["label"] = label
        if mutate:
            value.zonotope_w[0, 0, 0] += 1.0
        return "centered", "variance", {"label": label}

    monkeypatch.setattr(RUNNER.finish3l, "_layernorm_variance_state", original)
    with recorder.installed():
        result = RUNNER.finish3l._layernorm_variance_state(
            state, support, RUNNER.experiment.TARGET_LABEL)
    if finalize:
        recorder.bind_invocation(
            layernorm_index=5, invocation_ordinal=5,
            live_state_identity=RUNNER.experiment.state_identity(state, support))
        recorder.materialize_after_execution()
    return recorder, state, support, observed, result


def test_correct_invocation_selected_before_semantic_operations(monkeypatch):
    recorder, state, _support, observed, result = _capture(monkeypatch)
    recorder.validate()
    assert observed == {
        "snapshot_exists_before_semantics": False,
        "state_is_identical": True,
        "proof_is_identical": True,
        "label": RUNNER.experiment.TARGET_LABEL,
    }
    assert result[0] == "centered"
    assert torch.equal(recorder.snapshot["weights"], state.zonotope_w)
    assert recorder.materialized_after_execution is True


def test_wrapper_entry_is_strictly_reference_only(monkeypatch):
    class HostileTensorState:
        def __getattribute__(self, _name):
            raise AssertionError("capture touched input state before original")

    state = HostileTensorState()
    support = object()
    expected = {"canonical_state_identity_sha256": "test"}
    recorder = RUNNER.PostAttentionLayerNormInputCapture(expected, "test")
    observed = []

    def forbidden(*_args, **_kwargs):
        raise AssertionError("pre-call tensor inspection occurred")

    def original(value, proof_value, label):
        observed.append((value is state, proof_value is support, label))
        return "unchanged"

    monkeypatch.setattr(RUNNER.experiment, "state_identity", forbidden)
    monkeypatch.setattr(RUNNER, "_frontier_snapshot", forbidden)
    monkeypatch.setattr(RUNNER.finish3l, "_layernorm_variance_state", original)
    with recorder.installed():
        assert RUNNER.finish3l._layernorm_variance_state(
            state, support, RUNNER.experiment.TARGET_LABEL) == "unchanged"
    assert observed == [(True, True, RUNNER.experiment.TARGET_LABEL)]
    assert recorder.snapshot is None
    assert recorder.original_returned is True


def test_snapshot_and_hash_are_deferred_until_explicit_materialization(
        monkeypatch):
    state, support = State(), proof()
    expected = RUNNER.experiment.state_identity(state, support)
    recorder = RUNNER.PostAttentionLayerNormInputCapture(
        expected, expected["canonical_state_identity_sha256"])
    events = []
    real_snapshot = RUNNER._frontier_snapshot
    real_identity = RUNNER.experiment.state_identity

    monkeypatch.setattr(
        RUNNER.finish3l, "_layernorm_variance_state",
        lambda *_args: events.append("original") or "result")
    monkeypatch.setattr(
        RUNNER, "_frontier_snapshot",
        lambda *args: events.append("snapshot") or real_snapshot(*args))
    monkeypatch.setattr(
        RUNNER.experiment, "state_identity",
        lambda *args: events.append("identity") or real_identity(*args))
    with recorder.installed():
        RUNNER.finish3l._layernorm_variance_state(
            state, support, RUNNER.experiment.TARGET_LABEL)
    assert events == ["original"]
    recorder.bind_invocation(
        layernorm_index=5, invocation_ordinal=5,
        live_state_identity=expected)
    assert events == ["original"]
    recorder.materialize_after_execution()
    assert events[0] == "original"
    assert events[1:] == ["snapshot", "identity"]


def test_production_canonical_identity_is_frozen():
    assert RUNNER.EXPECTED_INPUT_CANONICAL_STATE_IDENTITY_SHA256 == (
        "c2fbf1d175157dfecca9c3da95573b593921a4dc05ca1b6a3e7eacaf730ea507")


def test_provenance_collection_is_not_in_target_wrapper():
    source = inspect.getsource(RUNNER.PostAttentionLayerNormInputCapture.installed)
    assert "_runtime_provenance" not in source
    assert "state_identity" not in source
    assert "_frontier_snapshot" not in source
    assert ".cpu(" not in source
    assert "synchronize" not in source


def test_runtime_provenance_records_required_fields_before_execution(
        monkeypatch):
    def completed(args, **_kwargs):
        output = "head\n" if args[0] == "git" else "0, A40, driver\n"
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(RUNNER.subprocess, "run", completed)
    record = RUNNER._runtime_provenance()
    assert set((
        "git_head", "git_status_porcelain", "python_version",
        "pytorch_version", "torch_cuda_version", "cudnn_version",
        "cuda_visible_devices", "cublas_workspace_config",
        "deterministic_algorithms", "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32", "gpu_model_and_driver")) <= set(record)
    assert record["gpu_model_and_driver"] == "0, A40, driver"


def test_wrong_layernorm_index_rejected(monkeypatch):
    recorder, state, support, *_ = _capture(monkeypatch, finalize=False)
    with pytest.raises(RuntimeError, match="index differs"):
        recorder.bind_invocation(
            layernorm_index=4, invocation_ordinal=4,
            live_state_identity=RUNNER.experiment.state_identity(state, support))


def test_wrong_stage_is_not_captured_and_validation_fails(monkeypatch):
    state, support = State(), proof()
    expected = RUNNER.experiment.state_identity(state, support)
    recorder = RUNNER.PostAttentionLayerNormInputCapture(
        expected, expected["canonical_state_identity_sha256"])
    monkeypatch.setattr(
        RUNNER.finish3l, "_layernorm_variance_state",
        lambda *_args: (None, None, {}))
    with recorder.installed():
        RUNNER.finish3l._layernorm_variance_state(
            state, support, "block2_output")
    with pytest.raises(RuntimeError, match="incomplete"):
        recorder.validate()


def test_capture_hook_detects_input_mutation(monkeypatch):
    with pytest.raises(RuntimeError, match="authenticated input differs"):
        _capture(monkeypatch, mutate=True)


def test_wrong_canonical_input_identity_hard_fails(monkeypatch):
    state, support = State(), proof()
    expected = RUNNER.experiment.state_identity(state, support)
    recorder = RUNNER.PostAttentionLayerNormInputCapture(expected, "0" * 64)
    monkeypatch.setattr(
        RUNNER.finish3l, "_layernorm_variance_state",
        lambda *_args: "result")
    with recorder.installed():
        RUNNER.finish3l._layernorm_variance_state(
            state, support, RUNNER.experiment.TARGET_LABEL)
    recorder.bind_invocation(
        layernorm_index=5, invocation_ordinal=5,
        live_state_identity=expected)
    with pytest.raises(RuntimeError, match="canonical input identity differs"):
        recorder.materialize_after_execution()


def test_source_metadata_hashes_are_deterministic(monkeypatch):
    recorder, *_ = _capture(monkeypatch)
    first = RUNNER._frontier_state_hashes(recorder.snapshot)
    second = RUNNER._frontier_state_hashes(copy.deepcopy(recorder.snapshot))
    assert first == second
    assert first["generator_count"] == 2


def test_same_snapshot_reproduces_psd_canonical_identity(monkeypatch):
    recorder, *_ = _capture(monkeypatch)
    assert RUNNER._snapshot_state_identity(recorder.snapshot) == \
        RUNNER._snapshot_state_identity(copy.deepcopy(recorder.snapshot))


@pytest.mark.parametrize("mutation", (
    "coefficient", "source_id", "range", "mask", "provenance"))
def test_canonical_identity_rejects_semantic_component_mutation(
        monkeypatch, mutation):
    recorder, *_ = _capture(monkeypatch)
    snapshot = copy.deepcopy(recorder.snapshot)
    original = RUNNER._snapshot_state_identity(snapshot)
    if mutation == "coefficient":
        snapshot["weights"][0, 0, 0] += 1.0
    elif mutation == "source_id":
        snapshot["proof"]["ids"][0] = "mutated-id"
    elif mutation == "range":
        snapshot["range_high"][0] += 0.25
    elif mutation == "mask":
        snapshot["proof"]["masks"][0] ^= 1
    else:
        snapshot["proof"]["reasons"][0] = "mutated-provenance"
    changed = RUNNER._snapshot_state_identity(snapshot)
    assert changed["canonical_state_identity_sha256"] != \
        original["canonical_state_identity_sha256"]


def test_missing_target_invocation_hard_fails():
    recorder = RUNNER.PostAttentionLayerNormInputCapture({})
    with pytest.raises(RuntimeError, match="incomplete"):
        recorder.validate()


def test_existing_output_root_is_refused_before_execution(tmp_path):
    root = tmp_path / "occupied"; root.mkdir(); (root / "x").write_text("x")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        RUNNER.execute(Path("campaign"), Path("artifacts"), Path("capture"),
                       Path("oracle"), root, 0)


def _persist_fixture(tmp_path, monkeypatch):
    recorder, *_ = _capture(monkeypatch)
    monkeypatch.setattr(
        RUNNER, "EXPECTED_INPUT_CANONICAL_STATE_IDENTITY_SHA256",
        recorder.state_identity["canonical_state_identity_sha256"])
    state_hashes = RUNNER._frontier_state_hashes(recorder.snapshot)
    frontier_path = tmp_path / "block2_ffn_frontier_manifest.json"
    RUNNER._atomic_json(frontier_path, {
        "schema": RUNNER.FRONTIER_MANIFEST_SCHEMA,
        "variants": [{"state_key": "post_attention_ln_pre_reduction",
                      **state_hashes}],
    })
    source = {
        "pinned_deept_revision": RUNNER.capture.prefix.PINNED_REVISION,
        "scientific_manifest_sha256":
            RUNNER.cluster_common.SCIENTIFIC_MANIFEST_SHA,
        "production_manifest_sha256":
            RUNNER.cluster_common.PRODUCTION_MANIFEST_SHA,
    }
    result = {"failure_stage": RUNNER.NEXT_STAGE,
              "failure_reason": RUNNER.finish3l.LAYERNORM_DOMAIN_REASON}
    harness = SimpleNamespace(psd_certificate_applications=1,
                              psd_certificate_rejections=0)
    record = RUNNER._persist_pre_layernorm_input_capture(
        tmp_path, recorder, source, result,
        {"manifest_path": str(frontier_path),
         "manifest_sha256": RUNNER.cluster_common.sha256(frontier_path)},
        harness)
    return record


def test_linkage_to_frontier_output_is_verified(tmp_path, monkeypatch):
    record = _persist_fixture(tmp_path, monkeypatch)
    checked = RUNNER._verify_pre_layernorm_input_capture(
        Path(record["manifest_path"]))
    assert checked["verified"] is True


def test_manifest_row_hash_is_distinct_secondary_identity(
        tmp_path, monkeypatch):
    record = _persist_fixture(tmp_path, monkeypatch)
    manifest = RUNNER.cluster_common.verified_json(Path(record["manifest_path"]))
    assert manifest["invocation_linkage"]["input_state_identity_sha256"] != \
        manifest["canonical_state_identity"][
            "canonical_state_identity_sha256"]
    assert RUNNER._verify_pre_layernorm_input_capture(
        Path(record["manifest_path"]))["verified"] is True


def test_persisted_reload_reproduces_canonical_identity(tmp_path, monkeypatch):
    record = _persist_fixture(tmp_path, monkeypatch)
    path = Path(record["manifest_path"])
    manifest = RUNNER.cluster_common.verified_json(path)
    payload = RUNNER.capture.sound.torch.load(
        Path(record["artifact_path"]), map_location="cpu", weights_only=False)
    recomputed = RUNNER._snapshot_state_identity(
        payload["states"]["pre_layernorm_input"])
    assert recomputed == manifest["canonical_state_identity"]


def test_verifier_rejects_recorded_canonical_mismatch(tmp_path, monkeypatch):
    record = _persist_fixture(tmp_path, monkeypatch)
    path = Path(record["manifest_path"])
    manifest = RUNNER.cluster_common.verified_json(path)
    manifest.pop("record_sha256")
    manifest["canonical_state_identity"]["weights_sha256"] = "0" * 64
    RUNNER._atomic_json(path, manifest)
    with pytest.raises(RuntimeError, match="canonical state identity differs"):
        RUNNER._verify_pre_layernorm_input_capture(path)


def test_job2997_legacy_manifest_authenticates_only_against_live_canonical(
        tmp_path, monkeypatch):
    record = _persist_fixture(tmp_path, monkeypatch)
    path = Path(record["manifest_path"])
    manifest = RUNNER.cluster_common.verified_json(path)
    expected = manifest.pop("canonical_state_identity")
    manifest.pop("record_sha256")
    linkage = manifest["invocation_linkage"]
    linkage.pop("input_state_identity_schema")
    linkage.pop("canonical_state_identity_sha256")
    linkage.pop("linkage_identity_sha256")
    linkage["linkage_identity_sha256"] = RUNNER.capture._json_sha(
        {key: value for key, value in linkage.items()
         if key not in {"verified", "existing_frontier_manifest_path"}})
    RUNNER._atomic_json(path, manifest)
    checked = RUNNER._verify_pre_layernorm_input_capture(
        path, allow_legacy_missing_canonical=True,
        expected_canonical_identity=expected)
    assert checked["legacy_manifest_without_canonical"] is True
    with pytest.raises(RuntimeError, match="canonical state identity is absent"):
        RUNNER._verify_pre_layernorm_input_capture(path)


def test_property_or_radius_mismatch_is_rejected(tmp_path, monkeypatch):
    record = _persist_fixture(tmp_path, monkeypatch)
    manifest = RUNNER.cluster_common.verified_json(Path(record["manifest_path"]))
    manifest.pop("record_sha256"); manifest["tested_radius"] = 0.0
    RUNNER._atomic_json(Path(record["manifest_path"]), manifest)
    with pytest.raises(RuntimeError, match="manifest identity differs"):
        RUNNER._verify_pre_layernorm_input_capture(Path(record["manifest_path"]))


def test_checkpoint_or_revision_mismatch_is_rejected(tmp_path, monkeypatch):
    record = _persist_fixture(tmp_path, monkeypatch)
    path = Path(record["manifest_path"])
    manifest = RUNNER.cluster_common.verified_json(path)
    manifest.pop("record_sha256")
    manifest["model_authentication"]["checkpoint_sha256"] = "0" * 64
    RUNNER._atomic_json(path, manifest)
    with pytest.raises(RuntimeError, match="manifest identity differs"):
        RUNNER._verify_pre_layernorm_input_capture(path)


def test_frontier_linkage_hash_mutation_is_rejected(tmp_path, monkeypatch):
    record = _persist_fixture(tmp_path, monkeypatch)
    path = Path(record["manifest_path"])
    manifest = RUNNER.cluster_common.verified_json(path)
    manifest.pop("record_sha256")
    manifest["invocation_linkage"]["linkage_identity_sha256"] = "0" * 64
    RUNNER._atomic_json(path, manifest)
    with pytest.raises(RuntimeError, match="invocation linkage differs"):
        RUNNER._verify_pre_layernorm_input_capture(path)


def test_prior_job2995_inputs_are_hash_authenticated(tmp_path, monkeypatch):
    capture = tmp_path / "capture.json"; capture.write_text("capture")
    oracle = tmp_path / "oracle.json"; oracle.write_text("oracle")
    frontier = tmp_path / "frontier.json"; frontier.write_text("frontier")
    monkeypatch.setattr(RUNNER, "_frontier_output_state_record",
                        lambda path: {"state_key": path.name})
    report_path = tmp_path / "experiment_report.json"
    RUNNER._atomic_json(report_path, {
        "schema": RUNNER.SCHEMA,
        "verdict": "CORET_PSD_LAYERNORM_EXPERIMENT_COMPLETE",
        "property_id": RUNNER.PROPERTY_ID,
        "tested_radius": RUNNER.TESTED_RADIUS,
        "tested_radius_hex": RUNNER.TESTED_RADIUS_HEX,
        "terminal_status": "UNCERTIFIED_DOMAIN_FAILURE",
        "generic_fallback_count": 0,
        "capture_manifest": {"path": str(capture),
                             "sha256": RUNNER.cluster_common.sha256(capture)},
        "oracle_report": {"path": str(oracle),
                          "sha256": RUNNER.cluster_common.sha256(oracle)},
        "block2_ffn_frontier_capture": {
            "manifest_path": str(frontier),
            "manifest_sha256": RUNNER.cluster_common.sha256(frontier)},
    })
    paths = RUNNER._authenticate_prior_experiment(report_path)
    assert paths["frontier_manifest"] == frontier


def test_prior_job2995_hash_mismatch_is_rejected(tmp_path, monkeypatch):
    capture = tmp_path / "capture.json"; capture.write_text("capture")
    oracle = tmp_path / "oracle.json"; oracle.write_text("oracle")
    frontier = tmp_path / "frontier.json"; frontier.write_text("frontier")
    monkeypatch.setattr(RUNNER, "_frontier_output_state_record",
                        lambda _path: {})
    report_path = tmp_path / "experiment_report.json"
    RUNNER._atomic_json(report_path, {
        "schema": RUNNER.SCHEMA,
        "verdict": "CORET_PSD_LAYERNORM_EXPERIMENT_COMPLETE",
        "property_id": RUNNER.PROPERTY_ID,
        "tested_radius": RUNNER.TESTED_RADIUS,
        "tested_radius_hex": RUNNER.TESTED_RADIUS_HEX,
        "terminal_status": "UNCERTIFIED_DOMAIN_FAILURE",
        "generic_fallback_count": 0,
        "capture_manifest": {"path": str(capture), "sha256": "bad"},
        "oracle_report": {"path": str(oracle),
                          "sha256": RUNNER.cluster_common.sha256(oracle)},
        "block2_ffn_frontier_capture": {
            "manifest_path": str(frontier),
            "manifest_sha256": RUNNER.cluster_common.sha256(frontier)},
    })
    with pytest.raises(RuntimeError, match="capture_manifest differs"):
        RUNNER._authenticate_prior_experiment(report_path)


def test_optional_capture_does_not_change_default_harness_signature():
    assert RUNNER.LayerNormExperimentHarness.__init__.__defaults__ == (None,)
