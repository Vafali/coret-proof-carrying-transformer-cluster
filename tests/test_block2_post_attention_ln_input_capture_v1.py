from __future__ import annotations

import copy
import importlib.util
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


def _capture(monkeypatch, mutate=False):
    state, support = State(), proof()
    recorder = RUNNER.PostAttentionLayerNormInputCapture()
    observed = {}

    def original(value, proof_value, label):
        observed["snapshot_exists_before_semantics"] = recorder.snapshot is not None
        if mutate:
            value.zonotope_w[0, 0, 0] += 1.0
        return "centered", "variance", {"label": label}

    monkeypatch.setattr(RUNNER.finish3l, "_layernorm_variance_state", original)
    with recorder.installed():
        result = RUNNER.finish3l._layernorm_variance_state(
            state, support, RUNNER.experiment.TARGET_LABEL)
    return recorder, state, support, observed, result


def test_correct_invocation_selected_before_semantic_operations(monkeypatch):
    recorder, state, _support, observed, result = _capture(monkeypatch)
    recorder.bind_invocation(
        layernorm_index=5, invocation_ordinal=5,
        live_state_identity=recorder.state_identity)
    recorder.validate()
    assert observed["snapshot_exists_before_semantics"] is True
    assert result[0] == "centered"
    assert torch.equal(recorder.snapshot["weights"], state.zonotope_w)


def test_wrong_layernorm_index_rejected(monkeypatch):
    recorder, *_ = _capture(monkeypatch)
    with pytest.raises(RuntimeError, match="index differs"):
        recorder.bind_invocation(
            layernorm_index=4, invocation_ordinal=4,
            live_state_identity=recorder.state_identity)


def test_wrong_stage_is_not_captured_and_validation_fails(monkeypatch):
    state, support = State(), proof()
    recorder = RUNNER.PostAttentionLayerNormInputCapture()
    monkeypatch.setattr(
        RUNNER.finish3l, "_layernorm_variance_state",
        lambda *_args: (None, None, {}))
    with recorder.installed():
        RUNNER.finish3l._layernorm_variance_state(
            state, support, "block2_output")
    with pytest.raises(RuntimeError, match="incomplete"):
        recorder.validate()


def test_capture_hook_detects_input_mutation(monkeypatch):
    with pytest.raises(RuntimeError, match="input was mutated"):
        _capture(monkeypatch, mutate=True)


def test_source_metadata_hashes_are_deterministic(monkeypatch):
    recorder, *_ = _capture(monkeypatch)
    first = RUNNER._frontier_state_hashes(recorder.snapshot)
    second = RUNNER._frontier_state_hashes(copy.deepcopy(recorder.snapshot))
    assert first == second
    assert first["generator_count"] == 2


def test_missing_target_invocation_hard_fails():
    recorder = RUNNER.PostAttentionLayerNormInputCapture()
    with pytest.raises(RuntimeError, match="incomplete"):
        recorder.validate()


def test_existing_output_root_is_refused_before_execution(tmp_path):
    root = tmp_path / "occupied"; root.mkdir(); (root / "x").write_text("x")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        RUNNER.execute(Path("campaign"), Path("artifacts"), Path("capture"),
                       Path("oracle"), root, 0)


def _persist_fixture(tmp_path, monkeypatch):
    recorder, *_ = _capture(monkeypatch)
    recorder.bind_invocation(
        layernorm_index=5, invocation_ordinal=5,
        live_state_identity=recorder.state_identity)
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
