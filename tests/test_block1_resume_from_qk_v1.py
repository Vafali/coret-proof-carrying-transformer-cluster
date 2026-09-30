from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/run_block1_resume_from_qk_v1.py"
SPEC = importlib.util.spec_from_file_location("block1_resume", SCRIPT)
resume = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(resume)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _proof(ids):
    return {
        "masks": [1 << index for index in range(len(ids))],
        "ids": list(ids), "reasons": ["unit" for _ in ids],
        "num_tokens": 4,
    }


def _state(shape, ids):
    count = len(ids)
    return {
        "weights": torch.zeros(shape, dtype=torch.float64),
        "range_low": -torch.ones(count, dtype=torch.float64),
        "range_high": torch.ones(count, dtype=torch.float64),
        "proof": _proof(ids),
    }


def _fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(resume, "EXPECTED_GENERATORS", 2)
    monkeypatch.setattr(resume, "EXPECTED_PRE_REDUCTION_GENERATORS", 4)
    monkeypatch.setattr(resume, "EXPECTED_NATIVE_FRESH", 1)
    monkeypatch.setattr(resume, "EXPECTED_NUMERICAL_FRESH", 1)
    monkeypatch.setattr(resume, "EXPECTED_RETAINED", 1)
    monkeypatch.setattr(resume, "EXPECTED_ABSORBED", 3)
    monkeypatch.setattr(resume, "EXPECTED_REPLACEMENTS", 1)
    monkeypatch.setattr(resume, "EXPECTED_QK_PREDECESSOR_SHA256", "p" * 64)
    ids = ["retained", "replacement"]
    reduction = {
        "operator": "b1_qk", "count_before": 4, "count_after": 2,
        "retained": 1, "absorbed": 3, "added_box_generators": 1,
        "support_inflation": 3.0819791163594346e-13,
        "retained_ids": ids[:1], "replacement_ids": ids[1:],
    }
    spot = {
        "one_ulp_inward_rejected": True,
        "state_reserve_contains_machine_error": True,
    }
    embedded = {
        "schema": resume.INPUT_REPORT_SCHEMA,
        "input_sha256": "p" * 64, "input_generator_count": 2,
        "pre_reduction_generator_count": 4,
        "output_generator_count": 2,
        "native_fresh_generator_count": 1,
        "fp64_numerical_fresh_count": 1,
        "generic_fallback_count": 0, "all_outputs_finite": True,
        "post_qk_reduction": reduction, "mpfr_spot": spot,
    }
    payload = {
        "schema": resume.INPUT_SCHEMA,
        "pinned_revision": resume.sound.PINNED_REVISION,
        "states": {
            "hidden": _state((3, 4, 128), ["hidden0", "hidden1"]),
            "qk": _state((4, 3, 4, 4), ids),
        },
        "report": embedded,
    }
    artifact = tmp_path / "qk.pt"
    torch.save(payload, artifact)
    external = dict(embedded)
    external["output_sha256"] = _sha(artifact)
    report = tmp_path / "qk.json"
    report.write_text(json.dumps(external, sort_keys=True) + "\n")
    return artifact, report


def test_authenticated_qk_is_accepted_without_qk_execution(tmp_path, monkeypatch):
    artifact, report = _fixture(tmp_path, monkeypatch)
    result = resume.authenticate(
        artifact, report, _sha(artifact), _sha(report))
    assert result["qk_invocations"] == 0
    assert result["scientific_properties"] == 0
    assert result["bound_calls"] == 0
    assert result["qk"]["generator_axis"] == 1
    assert result["qk"]["generator_count"] == 2


def test_authentication_rejects_hash_schema_and_reduction_order(
        tmp_path, monkeypatch):
    artifact, report = _fixture(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="artifact SHA256 mismatch"):
        resume.authenticate(artifact, report, "0" * 64, _sha(report))
    with pytest.raises(RuntimeError, match="report SHA256 mismatch"):
        resume.authenticate(artifact, report, _sha(artifact), "0" * 64)

    payload = torch.load(artifact, weights_only=False)
    payload["states"]["qk"]["proof"]["ids"].reverse()
    changed = tmp_path / "changed.pt"
    torch.save(payload, changed)
    external = json.loads(report.read_text())
    external["output_sha256"] = _sha(changed)
    changed_report = tmp_path / "changed.json"
    changed_report.write_text(json.dumps(external, sort_keys=True) + "\n")
    with pytest.raises(RuntimeError, match="reduction output order differs"):
        resume.authenticate(
            changed, changed_report, _sha(changed), _sha(changed_report))


def test_state_validation_rejects_narrow_invalid_range_and_duplicate_ids():
    state = _state((4, 3, 2, 2), ["a", "b"])
    accepted = resume._validate_state("qk", state, 2)
    assert accepted["generator_axis"] == 1
    state["range_low"][0] = 2.0
    with pytest.raises(RuntimeError, match="range ordering"):
        resume._validate_state("qk", state, 2)
    state = _state((4, 3, 2, 2), ["a", "a"])
    with pytest.raises(RuntimeError, match="identity/order"):
        resume._validate_state("qk", state, 2)


def test_continuation_stage_inventory_starts_after_qk(tmp_path):
    specs = resume._stage_specs(tmp_path / "qk.pt", tmp_path / "out")
    assert [item[0] for item in specs] == [
        "softmax", "av_prepare", "attention_value",
        "attention_projection_residual", "post_attention_layernorm",
        "ffn_affine_relu", "ffn_output", "ffn_residual_prepare",
        "ffn_residual", "final_layernorm_recenter_reduction",
    ]
    source = SCRIPT.read_text()
    assert "dispatch.qk" not in source
    assert "run_block1_qk" not in source
    assert '"qk_recomputed": False' in source


def test_mpfr_and_generic_fallback_gates_are_fail_closed():
    source = SCRIPT.read_text()
    implementation = (
        REPO / "research_hab/coret_sound_fp64_block0_feasibility_v1.py"
    ).read_text()
    assert "one_ulp_inward_rejected" in source
    assert "state_reserve_contains_machine_error" in source
    assert "generic fallback invoked" in source
    assert "final Block-1 state exceeds generator policy" in source
    assert "block1_actual_av_retained_coefficient" in implementation
    assert "block1_post_attention_layernorm_variance_nominal" in implementation
    assert "block1_post_attention_layernorm_sqrt_lower" in implementation
