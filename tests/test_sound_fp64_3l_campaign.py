import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


if "gmpy2" not in sys.modules:
    sys.modules["gmpy2"] = types.SimpleNamespace()

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "campaign3l", ROOT / "scripts/run_sound_fp64_3l_campaign.py")
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def _manifest_and_plan():
    properties = []
    for ordinal in range(127):
        radius = 0.0005 + ordinal * 1e-9
        properties.append({
            "property_id": f"property_{ordinal:03d}",
            "benchmark_ordinal": ordinal,
            "sentence_ordinal": ordinal // 20,
            "token_position": 1,
            "sequence_length": 2,
            "token_ids": [101, 102],
            "clean_label": 1, "nominal_prediction": 1,
            "cached_DeepT_reference": {
                "certified_lower_endpoint_binary64": radius,
                "certified_lower_endpoint_binary64_hex": radius.hex(),
            },
        })
    def assignment(row):
        return {key: row[key] for key in (
            "property_id", "benchmark_ordinal", "sentence_ordinal",
            "token_position", "sequence_length")}
    plan = {
        "canonical_manifest_sha256": "p" * 64,
        "workers": [
            {"chunks": [{"properties": [assignment(row) for row in
                          properties[:63]]}]},
            {"chunks": [{"properties": [assignment(row) for row in
                          properties[63:]]}]},
        ],
    }
    return {"properties": properties}, plan


def test_preflight_finds_exact_frozen_population_and_split(monkeypatch, tmp_path):
    manifest, plan = _manifest_and_plan()
    monkeypatch.setattr(
        campaign.cluster_common, "verify_artifact_manifest", lambda _root: {})
    monkeypatch.setattr(
        campaign.cluster_common, "load_production_manifest",
        lambda _root: manifest)
    monkeypatch.setattr(campaign.a40_fresh_common, "load_plan", lambda: plan)
    result = campaign.preflight(tmp_path)
    assert result["property_count"] == 127
    assert result["worker_property_counts"] == [63, 64]
    assert len(set(sum(result["worker_property_ids"], []))) == 127
    assert result["scientific_queries"] == 0


def test_property_source_is_scoped_and_restored():
    original = (campaign.prefix.FIXTURE_TOKEN_IDS,
                campaign.prefix.FIXTURE_PERTURBED_TOKEN,
                campaign.prefix.FIXTURE_RHO)
    row = {"token_ids": [101, 77, 102], "token_position": 1}
    with campaign._property_source(row, 0.125):
        assert campaign.prefix.FIXTURE_TOKEN_IDS == (101, 77, 102)
        assert campaign.prefix.FIXTURE_PERTURBED_TOKEN == 1
        assert campaign.prefix.FIXTURE_RHO == 0.125
    assert (campaign.prefix.FIXTURE_TOKEN_IDS,
            campaign.prefix.FIXTURE_PERTURBED_TOKEN,
            campaign.prefix.FIXTURE_RHO) == original


def test_property_failure_is_atomic_fail_closed_and_not_retried(
        monkeypatch, tmp_path):
    row = _manifest_and_plan()[0]["properties"][0]
    monkeypatch.setattr(
        campaign.sound, "export_block0_state",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic domain failure")))
    result = campaign.execute_property(row, tmp_path, "cuda:0")
    assert result["terminal_status"] == "FAIL_CLOSED"
    assert result["certified_at_historical_radius"] is False
    assert result["verifier_evaluations"] == 1
    assert result["binary_search_performed"] is False
    assert result["failure_stage"] == "block0"
    path = tmp_path / "properties" / row["property_id"] / "result.json"
    before = path.read_bytes()
    again = campaign.execute_property(row, tmp_path, "cuda:0")
    assert again == result and path.read_bytes() == before


def test_preflight_rejects_cross_worker_overlap(monkeypatch, tmp_path):
    manifest, plan = _manifest_and_plan()
    plan["workers"][1]["chunks"][0]["properties"][0] = \
        plan["workers"][0]["chunks"][0]["properties"][0]
    monkeypatch.setattr(
        campaign.cluster_common, "verify_artifact_manifest", lambda _root: {})
    monkeypatch.setattr(
        campaign.cluster_common, "load_production_manifest",
        lambda _root: manifest)
    monkeypatch.setattr(campaign.a40_fresh_common, "load_plan", lambda: plan)
    with pytest.raises(RuntimeError, match="does not cover"):
        campaign.preflight(tmp_path)

