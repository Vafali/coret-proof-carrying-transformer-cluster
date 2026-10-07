"""Result/persistence regressions; production operators are never executed."""
from contextlib import nullcontext
from pathlib import Path
import json
import math

import pytest
import torch

from test_sound_fp64_3l_campaign import campaign, _manifest_and_plan
from test_sound_fp64_epsilon_floor_v1 import (
    prepared_zero, execute, native_context, fp64_default)
import run_sound_fp64_epsilon_floor_property_v1 as revision
import semantic_epsilon_floor_checker_v1 as checker


@pytest.mark.parametrize("lower,verdict", [
    (5.292863350147482, "CORET_SOUND_FP64_3L_PROPERTY_READY"),
    (0., "CORET_SOUND_FP64_3L_UNCERTIFIED_MARGIN"),
    (-7.429728960608593, "CORET_SOUND_FP64_3L_UNCERTIFIED_MARGIN"),
])
def test_final_margin_verdict_finite_values(lower, verdict):
    assert campaign.finish3l._final_margin_verdict(lower, max(1., lower+1.)) == verdict


@pytest.mark.parametrize("lower,upper", [
    (math.nan, 1.), (math.inf, math.inf), (-math.inf, 1.),
    (0., math.nan), (0., math.inf), (0., -math.inf), (2., 1.),
])
def test_final_nonfinite_or_malformed_margin_remains_exception(lower, upper):
    with pytest.raises(RuntimeError, match="nonfinite or malformed"):
        campaign.finish3l._final_margin_verdict(lower, upper)


def fake_property_path(monkeypatch, lower, upper, witnesses=()):
    """Exercise the real campaign adapter, without any verifier/GPU operator."""
    manifest, _, historical, identity = _manifest_and_plan()
    row = campaign._resolve_campaign_input(manifest["properties"][0], historical, identity)
    monkeypatch.setattr(campaign, "_property_source", lambda *_: nullcontext())
    monkeypatch.setattr(campaign, "_cuda_memory_snapshot", lambda *_: {
        "allocated": 0, "reserved": 0, "max_allocated": 0, "max_reserved": 0})

    def export(path, **kwargs):
        path.write_bytes(b"synthetic-block0-not-a-scientific-state")
        return {"generic_fallback_count": 0}

    def stage(name, function, inputs, output, schema, device):
        output.write_bytes(b"synthetic-stage-no-operator-executed")
        return {"name": name, "output_schema": schema, "generic_fallback_count": 0}

    def block1_report(path, stages):
        result = path.with_suffix(".json")
        result.write_text("{}")
        return result

    def finish(input_path, input_report, output, output_report, *args, **kwargs):
        report = {
            "verdict": "CORET_SOUND_FP64_3L_PROPERTY_READY",  # Adapter must still validate finite margins.
            "clean_label": row["clean_label"], "fixture_token_ids": row["token_ids"],
            "fixture_rho_hex": campaign._candidate_radius(row).hex(),
            "final_sound_margin": lower, "final_sound_margin_upper": upper,
            "numerical_widening": .001, "max_numerical_native_ratio": .02,
            "final_generator_count": 14000, "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0, "reductions": [], "generic_fallback_count": 0,
        }
        torch.save({"schema": campaign.finish3l.OUTPUT_SCHEMA,
                    "separating_variance_witnesses": list(witnesses)}, output)
        output_report.write_text(json.dumps(report))
        return report

    monkeypatch.setattr(campaign.sound, "export_block0_state", export)
    monkeypatch.setattr(campaign, "_run_stage", stage)
    monkeypatch.setattr(campaign, "_block1_report", block1_report)
    monkeypatch.setattr(campaign.finish3l, "execute", finish)
    # The synthetic final artifact is intentionally not a complete model state;
    # its bytes are authenticated by the actual campaign integrity layer.
    monkeypatch.setattr(campaign.sound, "_load_artifact", lambda path, schema:
        torch.load(path, map_location="cpu", weights_only=False)
        if path.name == "certificate.pt" else {})
    return row


@pytest.mark.parametrize("lower", [5.292863350147482, 0., -7.429728960608593])
def test_benchmark_normalization_keeps_nonpositive_result_scientific(monkeypatch, tmp_path, lower):
    row = fake_property_path(monkeypatch, lower, max(1., lower+1.))
    raw = campaign.execute_property(row, tmp_path, "cuda:0")
    manifest = revision.B.read_protocol()
    frozen_row = manifest["properties"][0]
    raw.update(property_id=frozen_row["property_id"],
               historical_candidate_radius_hex=frozen_row["tested_radius_hex"],
               clean_label=frozen_row["clean_label"])
    normalized = revision.B.normalize(manifest, frozen_row, raw)
    assert normalized["producer_proof_status"] == ("CERTIFIED_MARGIN" if lower > 0 else "UNCERTIFIED")
    assert normalized["final_sound_lower_margin"] == lower
    assert normalized["certificate_integrity_status"] == "PASS"
    assert normalized["independently_checked_certificate_status"] == "NOT_AVAILABLE"
    assert normalized["final_proof_status"] == "INCONCLUSIVE"
    revision.B.validate_result(manifest, normalized)


@pytest.mark.parametrize("lower", [5.292863350147482, 0., -7.429728960608593])
def test_campaign_completed_margin_status_and_resume(monkeypatch, tmp_path, lower):
    row = fake_property_path(monkeypatch, lower, max(1., lower+1.))
    result = campaign.execute_property(row, tmp_path, "cuda:0")
    assert result["terminal_status"] == "COMPLETE"  # Existing schema convention.
    assert result["scientific_evaluation_complete"] is True
    assert result["certified_at_historical_radius"] is (lower > 0)
    assert result["classification"] == ("CERTIFIED_AT_HISTORICAL_RADIUS" if lower > 0
                                        else "FAILED_AT_HISTORICAL_RADIUS")
    assert result["final_sound_lower_margin"] == lower
    assert result["producer_revision"] == revision.REVISION
    assert result["final_generator_count"] == 14000
    assert result["numerical_widening"] == .001
    assert result["max_numerical_native_ratio"] == .02
    assert result["failure_stage"] == (None if lower > 0 else "final_margin")
    assert result["failure_reason"] == (None if lower > 0 else "NONPOSITIVE_SOUND_MARGIN")
    path = tmp_path/"properties"/row["property_id"]/"result.json"
    before = path.read_bytes()
    monkeypatch.setattr(campaign.sound, "export_block0_state", lambda *_args, **_kwargs:
                        pytest.fail("completed property rerun"))
    assert campaign.execute_property(row, tmp_path, "cuda:0") == result
    assert path.read_bytes() == before


@pytest.mark.parametrize("lower,upper", [(math.nan, 1.), (math.inf, math.inf),
    (-math.inf, 1.), (0., math.nan), (0., math.inf), (0., -math.inf)])
def test_nonfinite_margin_is_infrastructure_failure(monkeypatch, tmp_path, lower, upper):
    row = fake_property_path(monkeypatch, lower, upper)
    result = campaign.execute_property(row, tmp_path, "cuda:0")
    assert result["terminal_status"] == "FAIL_CLOSED"
    assert result["classification"] == "INFRASTRUCTURE_FAILURE"
    assert result["scientific_evaluation_complete"] is False
    assert result["certified_at_historical_radius"] is None
    assert result["final_sound_lower_margin"] is None
    assert result["failure_stage"] == "block2_to_margin"
    assert "nonfinite or malformed" in result["failure_reason"]
    assert not (tmp_path/"properties"/row["property_id"]/"certificate.pt").exists()


def test_negative_margin_persists_exact_replayed_epsilon_floor_witness(
        native_context, monkeypatch, tmp_path):
    state, proof, prepared = prepared_zero(monkeypatch)
    execute(native_context, state, proof, prepared)  # Tiny CPU-only local transition.
    payload = prepared["payload"]
    row = fake_property_path(monkeypatch, -7.429728960608593, 1., [payload])
    result = campaign.execute_property(row, tmp_path, "cuda:0")
    assert result["classification"] == "FAILED_AT_HISTORICAL_RADIUS"
    path = Path(result["layernorm_domain_witnesses_path"])
    assert path.is_file()
    assert campaign.cluster_common.sha256(path) == result["layernorm_domain_witnesses_sha256"]
    saved = torch.load(path, map_location="cpu", weights_only=False)["witnesses"][0]
    assert checker.replay_persisted(saved, payload["source_state_identity"], payload["parameter_hashes"],
                                    payload["source_domain"], payload["output_state_identity"])["verified"]
    assert result["epsilon_floor_labels_tokens"] == [{"label": payload["label"], "tokens": [0,1]}]
    intervals = result["epsilon_floor_semantic_intervals"][0]["tokens"]
    assert intervals[0]["regularized_range"] == payload["tokens"][0]["regularized_range"]
    assert intervals[0]["sqrt_range"] == payload["tokens"][0]["sqrt_range"]
    assert intervals[0]["reciprocal_range"] == payload["tokens"][0]["reciprocal_range"]
