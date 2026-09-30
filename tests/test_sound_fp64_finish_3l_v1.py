import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch


# The local audit environment omits the cluster's MPFR binding.  Authentication
# tests never enter numerical execution; a sentinel module keeps that boundary
# testable without pretending to validate MPFR arithmetic.
if "gmpy2" not in sys.modules:
    sys.modules["gmpy2"] = types.SimpleNamespace()


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "finish3l", ROOT / "scripts/run_sound_fp64_finish_3l_v1.py")
finish3l = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(finish3l)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path, num_tokens=4):
    generators = 2
    artifact = tmp_path / "block1.pt"
    state = {
        "weights": torch.zeros(
            generators + 1, num_tokens, 128, dtype=torch.float64),
        "range_low": -torch.ones(generators, dtype=torch.float64),
        "range_high": torch.ones(generators, dtype=torch.float64),
        "proof": {
            "masks": [2, 2], "ids": ["g0", "g1"],
            "reasons": ["fixture", "fixture"],
            "num_tokens": num_tokens,
        },
    }
    embedded = {
        "generic_fallback_count": 0,
        "dispatch_counts": {"LayerNorm": 1},
        "mpfr_spots": [{"one_ulp_inward_rejected": True}],
    }
    torch.save({
        "schema": finish3l.INPUT_SCHEMA,
        "pinned_revision": finish3l.sound.PINNED_REVISION,
        "states": {"pre_block2": state}, "report": embedded,
    }, artifact)
    artifact_sha = _sha(artifact)
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "schema": finish3l.INPUT_REPORT_SCHEMA,
        "verdict": "CORET_SOUND_FP64_BLOCK1_READY",
        "final_artifact_schema": finish3l.INPUT_SCHEMA,
        "final_artifact_sha256": artifact_sha,
        "final_generator_count": generators,
        "block2_feasible": True, "qk_recomputed": False,
        "generic_fallback_count": 0,
        "scientific_properties": 0, "bound_calls": 0,
        "stages": [{
            "output_schema": finish3l.INPUT_SCHEMA,
            "output_sha256": artifact_sha,
            "generic_fallback_count": 0,
        }],
    }))
    return artifact, report


def test_preflight_authenticates_without_operator_execution(tmp_path, monkeypatch):
    artifact, report = _fixture(tmp_path)
    monkeypatch.setattr(
        finish3l.sound, "pinned_zonotope",
        lambda: pytest.fail("preflight entered verifier execution"))
    result = finish3l.authenticate(
        artifact, report, _sha(artifact), _sha(report))
    assert result["state_identity"]["generator_count"] == 2
    assert result["operator_calls"] == 0
    assert result["scientific_properties"] == result["bound_calls"] == 0


def test_campaign_sequence_length_reaches_pre_block2_boundary(tmp_path):
    artifact, report = _fixture(tmp_path, num_tokens=20)
    with pytest.raises(RuntimeError, match="hidden-state shape differs"):
        finish3l.authenticate(
            artifact, report, _sha(artifact), _sha(report))
    result = finish3l.authenticate(
        artifact, report, _sha(artifact), _sha(report),
        expected_num_tokens=20)
    assert result["state_identity"]["shape"] == [3, 20, 128]
    assert result["state_identity"]["generator_count"] == 2


@pytest.mark.parametrize("field,value", [
    ("block2_feasible", False),
    ("generic_fallback_count", 1),
    ("final_generator_count", 3),
])
def test_preflight_rejects_report_identity_mutation(
        tmp_path, field, value):
    artifact, report = _fixture(tmp_path)
    payload = json.loads(report.read_text())
    payload[field] = value
    report.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="report identity"):
        finish3l.authenticate(artifact, report, _sha(artifact), _sha(report))


def test_state_validation_rejects_generator_policy_violation(tmp_path):
    artifact, _report = _fixture(tmp_path)
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    state = payload["states"]["pre_block2"]
    state["proof"]["ids"][1] = state["proof"]["ids"][0]
    with pytest.raises(RuntimeError, match="not unique"):
        finish3l._validate_state(state)


def test_cluster_execution_is_explicitly_cuda_only(tmp_path, monkeypatch):
    artifact, report = _fixture(tmp_path)
    monkeypatch.setattr(finish3l, "authenticate", lambda *_args: {})
    monkeypatch.setattr(finish3l.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        finish3l.execute(
            artifact, report, tmp_path / "out.pt", tmp_path / "out.json", 0)
