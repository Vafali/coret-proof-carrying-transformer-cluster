import importlib.util
import json
import sys
import types
from contextlib import contextmanager
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
            "canonical_binary_test_index": ordinal,
            "sentence_ordinal": ordinal,
            "sentence_id": f"sentence_{ordinal:03d}",
            "source_test_line": ordinal,
            "token_position": 1,
            "sequence_length": 2,
            "source_dimension": 128,
            "raw_sentence_sha256": f"{ordinal:064x}",
            "token": "x", "token_id": 102,
            "clean_label": 1, "nominal_prediction": 1,
            "cached_DeepT_reference": {
                "certified_lower_endpoint_binary64": radius,
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
    historical = {
        "properties": [
            {field: row[field] for field in campaign.TOKEN_IDENTITY_FIELDS}
            for row in properties
        ],
        "examples": [{
            "canonical_binary_test_index": row["canonical_binary_test_index"],
            "clean_label": row["clean_label"],
            "nominal_prediction": row["nominal_prediction"],
            "raw_sentence_sha256": row["raw_sentence_sha256"],
            "sentence_id": row["sentence_id"],
            "sentence_ordinal": row["sentence_ordinal"],
            "sequence_length_including_special_tokens": row["sequence_length"],
            "source_test_line": row["source_test_line"],
            "token_ids": [101, 102],
            "tokens": ["[CLS]", "x"],
            "eligible_token_positions": [1],
        } for row in properties],
    }
    identity = {
        "path": "/frozen/historical.json",
        "file_sha256": "f" * 64,
        "canonical_manifest_sha256": "s" * 64,
        "source": "frozen historical scientific manifest examples[].token_ids",
    }
    return {"properties": properties}, plan, historical, identity


def test_preflight_finds_exact_frozen_population_and_split(monkeypatch, tmp_path):
    manifest, plan, historical, identity = _manifest_and_plan()
    monkeypatch.setattr(
        campaign.cluster_common, "verify_artifact_manifest", lambda _root: {})
    monkeypatch.setattr(
        campaign.cluster_common, "load_production_manifest",
        lambda _root: manifest)
    monkeypatch.setattr(campaign.a40_fresh_common, "load_plan", lambda: plan)
    monkeypatch.setattr(
        campaign, "_load_authoritative_token_source",
        lambda _root: (historical, identity))
    result = campaign.preflight(tmp_path)
    assert result["property_count"] == 127
    assert result["worker_property_counts"] == [63, 64]
    assert len(set(sum(result["worker_property_ids"], []))) == 127
    assert result["candidate_radius_source"] == (
        "cached_DeepT_reference.certified_lower_endpoint_binary64")
    assert result["resolved_token_property_count"] == 127
    assert result["authoritative_token_source"] == identity
    assert result["scientific_queries"] == 0


def _single_input():
    manifest, _, historical, identity = _manifest_and_plan()
    return manifest["properties"][0], historical, identity


def test_missing_embedded_token_ids_resolve_from_authoritative_source():
    row, historical, identity = _single_input()
    assert "token_ids" not in row
    resolved = campaign._resolve_campaign_input(row, historical, identity)
    assert resolved["property_id"] == row["property_id"]
    assert resolved["token_ids"] == [101, 102]
    assert len(resolved["token_ids"]) == resolved["sequence_length"]
    assert resolved["token_input_source"] == identity


def test_conflicting_authoritative_token_sources_reject():
    row, historical, identity = _single_input()
    conflicting = dict(historical["examples"][0])
    conflicting["token_ids"] = [101, 999]
    historical["examples"].append(conflicting)
    with pytest.raises(RuntimeError, match="absent or ambiguous"):
        campaign._resolve_campaign_input(row, historical, identity)


def test_missing_authoritative_token_source_rejects():
    row, historical, identity = _single_input()
    historical["examples"] = historical["examples"][1:]
    with pytest.raises(RuntimeError, match="absent or ambiguous"):
        campaign._resolve_campaign_input(row, historical, identity)


def test_property_id_mismatch_rejects():
    row, historical, identity = _single_input()
    row = {**row, "property_id": "different_property"}
    with pytest.raises(RuntimeError, match="property source is absent"):
        campaign._resolve_campaign_input(row, historical, identity)


def test_authoritative_token_count_mismatch_rejects():
    row, historical, identity = _single_input()
    historical["examples"][0]["token_ids"] = [101]
    with pytest.raises(RuntimeError, match="token count differs"):
        campaign._resolve_campaign_input(row, historical, identity)


def test_embedded_token_ids_must_agree_with_authoritative_source():
    row, historical, identity = _single_input()
    matching = campaign._resolve_campaign_input(
        {**row, "token_ids": [101, 102]}, historical, identity)
    assert matching["token_ids"] == [101, 102]
    with pytest.raises(RuntimeError, match="embedded and authoritative"):
        campaign._resolve_campaign_input(
            {**row, "token_ids": [101, 999]}, historical, identity)


def test_clean_label_nominal_prediction_mismatch_still_rejects():
    row, historical, identity = _single_input()
    row = {**row, "clean_label": 0, "nominal_prediction": 1}
    historical["properties"][0]["clean_label"] = 0
    historical["examples"][0]["clean_label"] = 0
    with pytest.raises(RuntimeError, match="clean/nominal label differs"):
        campaign._resolve_campaign_input(row, historical, identity)


@pytest.mark.parametrize("candidate", [0.125, 0.0])
def test_candidate_radius_accepts_authoritative_decimal_including_zero(candidate):
    row = {"cached_DeepT_reference": {
        "certified_lower_endpoint_binary64": candidate,
    }}
    parsed = campaign._candidate_radius(row)
    assert parsed == candidate
    if candidate == 0.0:
        assert parsed.hex() == "0x0.0p+0"


def test_candidate_radius_rejects_missing_authoritative_decimal():
    row = {"cached_DeepT_reference": {
        "certified_lower_endpoint_binary64_hex": "0x1.0p-10",
    }}
    with pytest.raises(RuntimeError, match="missing authoritative"):
        campaign._candidate_radius(row)


@pytest.mark.parametrize("candidate", [float("nan"), float("inf"),
                                         float("-inf"), -0.125])
def test_candidate_radius_rejects_nonfinite_or_negative(candidate):
    row = {"cached_DeepT_reference": {
        "certified_lower_endpoint_binary64": candidate,
    }}
    with pytest.raises(RuntimeError, match="finite and nonnegative"):
        campaign._candidate_radius(row)


def test_candidate_radius_accepts_matching_optional_hex():
    candidate = 0.000625
    row = {"cached_DeepT_reference": {
        "certified_lower_endpoint_binary64": candidate,
        "certified_lower_endpoint_binary64_hex": candidate.hex(),
    }}
    assert campaign._candidate_radius(row) == candidate


def test_candidate_radius_rejects_mismatching_optional_hex():
    row = {"cached_DeepT_reference": {
        "certified_lower_endpoint_binary64": 0.000625,
        "certified_lower_endpoint_binary64_hex": (0.00125).hex(),
    }}
    with pytest.raises(RuntimeError, match="radii differ"):
        campaign._candidate_radius(row)


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


def test_campaign_block0_initializes_reduction_ledger_before_first_reduction(
        monkeypatch, tmp_path):
    class ReachedReductionLedger(RuntimeError):
        pass

    class FakeZonotope:
        def __init__(self, **_kwargs):
            self.num_error_terms = campaign.sound.MAXIMUM_GENERATORS + 1

    class FakeDispatch:
        counts = {}

        def layer_norm(self, state, _parameters, _mode):
            return state

        def reduce(self, *_args):
            raise AssertionError("oversized state must use sound reduction")

    @contextmanager
    def pinned_zonotope():
        yield FakeZonotope

    proof = types.SimpleNamespace()
    checkpoint = {
        "bert.embeddings.word_embeddings.weight":
            campaign.sound.torch.zeros((3, 1), dtype=campaign.sound.torch.float64),
        "bert.embeddings.position_embeddings.weight":
            campaign.sound.torch.zeros((2, 1), dtype=campaign.sound.torch.float64),
        "bert.embeddings.token_type_embeddings.weight":
            campaign.sound.torch.zeros((1, 1), dtype=campaign.sound.torch.float64),
    }

    monkeypatch.setattr(campaign.sound, "pinned_zonotope", pinned_zonotope)
    monkeypatch.setattr(campaign.sound.prefix, "_load_checkpoint",
                        lambda: checkpoint)
    monkeypatch.setattr(campaign.sound.prefix, "FIXTURE_TOKEN_IDS", (1, 2))
    monkeypatch.setattr(campaign.sound, "_args", lambda _device: object())
    monkeypatch.setattr(campaign.sound.structural, "local_mask", lambda _p: 1)
    monkeypatch.setattr(campaign.sound.structural, "proof_from_masks",
                        lambda *_args: proof)
    monkeypatch.setattr(campaign.sound.structural, "attach_support",
                        lambda *_args: None)
    monkeypatch.setattr(campaign.sound.structural, "get_support",
                        lambda _state: proof)
    monkeypatch.setattr(
        campaign.sound.structural, "StructuralNativeSemanticOperators",
        lambda: types.SimpleNamespace())
    monkeypatch.setattr(
        campaign.sound.production, "NativeProductionDispatch",
        lambda delegate: FakeDispatch())
    monkeypatch.setattr(campaign.sound.production, "_recenter_native_ranges",
                        lambda state: state)
    monkeypatch.setattr(campaign.sound, "_inject",
                        lambda state, state_proof, *_args, **_kwargs:
                        (state, state_proof))
    monkeypatch.setattr(campaign.sound, "_metrics",
                        lambda *_args: {"bounded": True})
    monkeypatch.setattr(campaign.sound, "_parameter",
                        lambda *_args: object())
    monkeypatch.setattr(campaign.sound, "_layernorm_majorant",
                        lambda *_args: 0.0)
    monkeypatch.setattr(campaign.sound, "_reserve_from_majorant",
                        lambda *_args: 0.0)

    def inspect_reduction_ledger(_state, _proof, label, reductions):
        assert label == "block0_pre_qk_reduction"
        assert reductions == []
        reductions.append({"label": label, "support_inflation": 0.0})
        assert reductions[0]["label"] == label
        raise ReachedReductionLedger

    monkeypatch.setattr(
        campaign.sound, "_maybe_reduce", inspect_reduction_ledger)
    with pytest.raises(ReachedReductionLedger):
        campaign.sound.export_block0_state(
            tmp_path / "unused.pt", device="cpu",
            run_representative_mpfr=False)


def test_property_failure_is_atomic_fail_closed_and_not_retried(
        monkeypatch, tmp_path):
    manifest, _, historical, identity = _manifest_and_plan()
    row = campaign._resolve_campaign_input(
        manifest["properties"][0], historical, identity)
    monkeypatch.setattr(
        campaign.sound, "export_block0_state",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic domain failure")))
    result = campaign.execute_property(row, tmp_path, "cuda:0")
    assert result["terminal_status"] == "FAIL_CLOSED"
    assert result["certified_at_historical_radius"] is None
    assert result["scientific_evaluation_complete"] is False
    assert result["classification"] == "INFRASTRUCTURE_FAILURE"
    assert result["failure_category"] == "RUNTIME_EXCEPTION"
    assert result["verifier_evaluations"] == 1
    assert result["binary_search_performed"] is False
    assert result["failure_stage"] == "block0"
    path = tmp_path / "properties" / row["property_id"] / "result.json"
    before = path.read_bytes()
    again = campaign.execute_property(row, tmp_path, "cuda:0")
    assert again == result and path.read_bytes() == before


def test_layernorm_domain_failure_is_scientific_noncertification(
        monkeypatch, tmp_path):
    manifest, _, historical, identity = _manifest_and_plan()
    row = campaign._resolve_campaign_input(
        manifest["properties"][0], historical, identity)
    diagnostic = {
        "reason_code": campaign.finish3l.LAYERNORM_DOMAIN_REASON,
        "label": "block2_post_attention",
        "minimum_token_index": 1,
        "minimum_coordinate_index": 7,
        "sound_variance_lower": -0.125,
        "domain_admissible": False,
    }
    path = tmp_path / "properties" / row["property_id"] / "result.json"
    campaign._atomic_json(path, {
        "schema": campaign.RESULT_SCHEMA,
        "terminal_status": "UNCERTIFIED_DOMAIN_FAILURE",
        "scientific_evaluation_complete": True,
        "certified_at_historical_radius": False,
        "classification": "FAILED_AT_HISTORICAL_RADIUS",
        "failure_category": "SOUND_LAYERNORM_DOMAIN_FAILURE",
        "domain_failure_diagnostic": diagnostic,
        "generic_fallback_count": 0,
    })
    result = campaign._verified_result(path)
    assert result["terminal_status"] == "UNCERTIFIED_DOMAIN_FAILURE"
    assert result["scientific_evaluation_complete"] is True
    assert result["certified_at_historical_radius"] is False
    assert result["classification"] == "FAILED_AT_HISTORICAL_RADIUS"
    assert result["failure_category"] == "SOUND_LAYERNORM_DOMAIN_FAILURE"
    assert result["domain_failure_diagnostic"] == diagnostic
    assert result["generic_fallback_count"] == 0


def test_campaign_status_counts_exclude_infrastructure_from_scientific_failure():
    rows = [
        {"scientific_evaluation_complete": True,
         "certified_at_historical_radius": True},
        {"scientific_evaluation_complete": True,
         "certified_at_historical_radius": False},
        {"scientific_evaluation_complete": False,
         "certified_at_historical_radius": None},
    ]
    assert campaign._status_counts(rows) == {
        "completed_sound_evaluations": 2,
        "certified_at_historical_radius": 1,
        "failed_at_historical_radius": 1,
        "infrastructure_failures": 1,
    }


def test_failure_categories_distinguish_oom_and_adapter_errors():
    assert campaign._failure_category(
        RuntimeError("CUDA out of memory")) == "CUDA_OUT_OF_MEMORY"
    assert campaign._failure_category(RuntimeError(
        "pre-Block-2 hidden-state shape differs")) == \
        "ADAPTER_STATE_TRANSFER"
    assert campaign._failure_category(ValueError("other")) == \
        "RUNTIME_EXCEPTION"


def test_cpu_property_boundary_cleanup_never_enters_cuda(monkeypatch):
    monkeypatch.setattr(
        campaign.sound.torch.cuda, "is_available", lambda: False)
    result = campaign._property_boundary_cleanup("cuda:0")
    assert result["cuda_cache_trimmed"] is False


def test_preflight_rejects_cross_worker_overlap(monkeypatch, tmp_path):
    manifest, plan, historical, identity = _manifest_and_plan()
    plan["workers"][1]["chunks"][0]["properties"][0] = \
        plan["workers"][0]["chunks"][0]["properties"][0]
    monkeypatch.setattr(
        campaign.cluster_common, "verify_artifact_manifest", lambda _root: {})
    monkeypatch.setattr(
        campaign.cluster_common, "load_production_manifest",
        lambda _root: manifest)
    monkeypatch.setattr(campaign.a40_fresh_common, "load_plan", lambda: plan)
    monkeypatch.setattr(
        campaign, "_load_authoritative_token_source",
        lambda _root: (historical, identity))
    with pytest.raises(RuntimeError, match="does not cover"):
        campaign.preflight(tmp_path)
