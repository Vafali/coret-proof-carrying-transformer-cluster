from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "linkage_diagnostic",
    ROOT / "scripts/inspect_block2_layernorm_capture_linkage_v1.py")
DIAGNOSTIC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTIC)


def _write_json(path: Path, value: dict) -> None:
    payload = dict(value)
    payload["record_sha256"] = DIAGNOSTIC._canonical(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _state() -> dict:
    weights = torch.zeros(3, 2, 4, dtype=torch.float64)
    weights[0, 0, 0] = 1.25
    weights[1, 0, 1] = 0.5
    weights[2, 1, 2] = 0.25
    return {
        "weights": weights,
        "range_low": torch.tensor([-1.0, -0.5], dtype=torch.float64),
        "range_high": torch.tensor([1.0, 0.75], dtype=torch.float64),
        "proof": {
            "masks": [1, 2],
            "ids": ["semantic-source", "fp64_numerical::test::000000"],
            "reasons": ["native_semantic", "fp64_roundoff_coordinate_box"],
            "num_tokens": 2,
        },
    }


def _manifest(root: Path, name: str, state: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        "property_id": DIAGNOSTIC.PROPERTY_ID,
        "multiplier": DIAGNOSTIC.MULTIPLIER,
        "tested_radius": DIAGNOSTIC.TESTED_RADIUS,
        "tested_radius_hex": DIAGNOSTIC.TESTED_RADIUS_HEX,
        "stage_label": DIAGNOSTIC.FRONTIER_STAGE,
        "pinned_deept_revision": DIAGNOSTIC.PINNED_REVISION,
        "scientific_manifest_sha256": DIAGNOSTIC.SCIENTIFIC_MANIFEST_SHA256,
        "production_manifest_sha256": DIAGNOSTIC.PRODUCTION_MANIFEST_SHA256,
        "source_set_model": DIAGNOSTIC.SOURCE_SET_MODEL,
    }
    reductions = {"synthetic": []}
    aliases = {"fixture": DIAGNOSTIC.STATE_KEY}
    artifact = root / f"{name}.pt"
    torch.save({
        "schema": DIAGNOSTIC.FRONTIER_CAPTURE_SCHEMA,
        "pinned_revision": DIAGNOSTIC.PINNED_REVISION,
        "identity": identity,
        "states": {DIAGNOSTIC.STATE_KEY: state},
        "reductions": reductions,
        "aliases": aliases,
    }, artifact)
    result = root / f"{name}_result.json"
    result.write_text("{}\n", encoding="utf-8")
    manifest = root / f"{name}.json"
    _write_json(manifest, {
        "schema": DIAGNOSTIC.FRONTIER_MANIFEST_SCHEMA,
        **identity,
        "tensor_artifact_path": artifact.name,
        "tensor_artifact_sha256": DIAGNOSTIC._sha256(artifact),
        "artifact_identity": identity,
        "result_path": result.name,
        "result_sha256": DIAGNOSTIC._sha256(result),
        "reduction_records_sha256": DIAGNOSTIC._canonical(reductions),
        "aliases": aliases,
        "variants": [DIAGNOSTIC._legacy_state_record(state)],
    })
    return manifest


def test_artifact_container_identity_is_not_semantic_identity(tmp_path):
    expected = _manifest(tmp_path / "expected", "state", _state())
    reproduced = _manifest(tmp_path / "reproduced", "other-name", _state())
    report = DIAGNOSTIC.compare_frontier_manifests(expected, reproduced)
    assert report["semantic_equal"] is True
    assert report["classification"] == "SEMANTIC_STATE_MATCH"
    assert (report["expected_identity"]["artifact_path"]
            != report["reproduced_identity"]["artifact_path"])


def test_tensor_mismatch_reports_first_field_and_maximum_difference(tmp_path):
    left, right = _state(), _state()
    right["weights"][0, 1, 3] += 0.125
    expected = _manifest(tmp_path / "expected", "state", left)
    reproduced = _manifest(tmp_path / "reproduced", "state", right)
    report = DIAGNOSTIC.compare_frontier_manifests(expected, reproduced)
    assert report["semantic_equal"] is False
    assert report["first_semantic_difference"] == "hashes.center_sha256"
    row = report["tensor_comparison"]["center"]
    assert row["exact_equal"] is False
    assert row["maximum_absolute_difference"] == 0.125


def test_ordered_source_identity_mismatch_is_not_normalized(tmp_path):
    left, right = _state(), _state()
    right["proof"]["ids"] = list(reversed(right["proof"]["ids"]))
    expected = _manifest(tmp_path / "expected", "state", left)
    reproduced = _manifest(tmp_path / "reproduced", "state", right)
    report = DIAGNOSTIC.compare_frontier_manifests(expected, reproduced)
    assert report["semantic_equal"] is False
    comparison = report["ordered_metadata_comparison"]["source_ids"]
    assert comparison["exact_equal"] is False
    assert comparison["first_difference"]["index"] == 0


def test_manifest_hash_mismatch_fails_closed(tmp_path):
    path = _manifest(tmp_path, "state", _state())
    value = json.loads(path.read_text(encoding="utf-8"))
    value["tested_radius"] = 0.5
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="canonical hash mismatch"):
        DIAGNOSTIC.compare_frontier_manifests(path, path)


def test_runner_persists_diagnostic_before_hard_failure(
        tmp_path, monkeypatch):
    if "gmpy2" not in sys.modules:
        monkeypatch.setitem(sys.modules, "gmpy2", types.SimpleNamespace())
    runner_spec = importlib.util.spec_from_file_location(
        "linkage_runner_for_test",
        ROOT / "scripts/run_sound_fp64_3l_psd_layernorm_experiment_v1.py")
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)
    report = {
        "schema": DIAGNOSTIC.SCHEMA,
        "classification": "TRUE_SCIENTIFIC_REPRODUCTION_MISMATCH",
        "semantic_equal": False,
        "first_semantic_difference": "hashes.center_sha256",
    }
    monkeypatch.setattr(
        runner.linkage_diagnostic, "compare_frontier_manifests",
        lambda *_args: copy.deepcopy(report))
    with pytest.raises(RuntimeError, match="hashes.center_sha256"):
        runner._enforce_existing_frontier_linkage(
            tmp_path, Path("expected"), Path("reproduced"))
    persisted = runner.cluster_common.verified_json(
        tmp_path / "layernorm_output_linkage_diagnostic.json")
    assert persisted["semantic_equal"] is False
    assert persisted["first_semantic_difference"] == "hashes.center_sha256"
