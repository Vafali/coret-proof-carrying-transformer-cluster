import importlib.util
import json
import sys
import types
from fractions import Fraction
from pathlib import Path

import pytest


if "gmpy2" not in sys.modules:
    sys.modules["gmpy2"] = types.SimpleNamespace()

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "radius_pilot",
    ROOT / "scripts/run_sound_fp64_3l_radius_recovery_pilot.py")
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


def _eligible(property_id, radius):
    return {
        "property_id": property_id,
        "historical_candidate_radius": radius,
        "scientific_evaluation_complete": True,
        "classification": "FAILED_AT_HISTORICAL_RADIUS",
        "failure_category": "SOUND_LAYERNORM_DOMAIN_FAILURE",
        "record_sha256": (property_id.encode().hex() + "0" * 64)[:64],
    }


def _population():
    anchors = {index: value for index, value in zip(
        pilot.SELECTION_INDICES, pilot.EXPECTED_SELECTION)}
    rows = []
    anchor_indices = list(pilot.SELECTION_INDICES)
    for index in range(37):
        if index in anchors:
            property_id, radius = anchors[index]
        else:
            lower = max(item for item in anchor_indices if item < index)
            upper = min(item for item in anchor_indices if item > index)
            low_radius = anchors[lower][1]
            high_radius = anchors[upper][1]
            radius = low_radius + (high_radius - low_radius) * (
                (index - lower) / (upper - lower))
            property_id = f"synthetic_{index:02d}"
        rows.append(_eligible(property_id, radius))
    return rows


def test_exactly_37_eligible_records_are_required():
    assert len(pilot._eligible_population(_population())) == 37
    with pytest.raises(RuntimeError, match="eligible failure count"):
        pilot._eligible_population(_population()[:-1])


@pytest.mark.parametrize("mutation", [
    {"classification": "CERTIFIED_AT_HISTORICAL_RADIUS"},
    {"scientific_evaluation_complete": False,
     "classification": "INFRASTRUCTURE_FAILURE"},
    {"failure_category": "OTHER"},
])
def test_filter_excludes_noneligible_records(mutation):
    rows = _population()
    rows[0] = {**rows[0], **mutation}
    with pytest.raises(RuntimeError, match="eligible failure count"):
        pilot._eligible_population(rows)


def test_deterministic_sorting_indices_and_exact_selection():
    rows = list(reversed(_population()))
    eligible = pilot._eligible_population(rows)
    selected = pilot._selected_population(eligible)
    assert pilot.SELECTION_INDICES == (0, 7, 14, 22, 29, 36)
    assert [(row["property_id"], row["historical_candidate_radius"])
            for row in selected] == list(pilot.EXPECTED_SELECTION)


def test_altered_historical_radius_rejects():
    rows = _population()
    rows[14]["historical_candidate_radius"] += 1e-9
    with pytest.raises(RuntimeError, match="selection differs"):
        pilot._selected_population(pilot._eligible_population(rows))


def test_missing_selected_property_rejects():
    rows = _population()
    rows[22]["property_id"] = "wrong_property"
    with pytest.raises(RuntimeError, match="selection differs"):
        pilot._selected_population(pilot._eligible_population(rows))


def test_additional_eligible_property_rejects():
    rows = _population() + [_eligible("extra", 0.002)]
    with pytest.raises(RuntimeError, match="eligible failure count"):
        pilot._eligible_population(rows)


@pytest.mark.parametrize("numerator,denominator", pilot.MULTIPLIERS)
def test_rational_radius_is_single_deterministic_binary64_rounding(
        numerator, denominator):
    historical = pilot.EXPECTED_SELECTION[0][1]
    expected = float(Fraction.from_float(historical)
                     * Fraction(numerator, denominator))
    actual = pilot._tested_radius(historical, numerator, denominator)
    assert actual == expected
    assert float.fromhex(actual.hex()) == actual


def _pilot_result(historical=0.001, numerator=95, denominator=100,
                  classification="FAILED_AT_TESTED_RADIUS",
                  property_id="property"):
    tested = pilot._tested_radius(historical, numerator, denominator)
    scientific = classification != "INFRASTRUCTURE_FAILURE"
    certified = (classification == "CERTIFIED_AT_TESTED_RADIUS"
                 if scientific else None)
    return {
        "schema": pilot.RESULT_SCHEMA,
        "property_id": property_id,
        "historical_candidate_radius": historical,
        "historical_candidate_radius_hex": historical.hex(),
        "multiplier": {
            "numerator": numerator, "denominator": denominator,
            "display": f"{numerator / denominator:.2f}",
            "binary64": float(numerator / denominator),
            "binary64_hex": float(numerator / denominator).hex(),
        },
        "tested_radius": tested,
        "tested_radius_hex": tested.hex(),
        "scientific_evaluation_complete": scientific,
        "certified_at_tested_radius": certified,
        "classification": classification,
        "runtime_seconds": 1.0,
        "hardware": {}, "source_hashes": {},
    }


def test_resume_accepts_valid_completed_record(tmp_path):
    path = tmp_path / "result.json"
    pilot._atomic_json(path, _pilot_result())
    assert pilot._verified_pilot_result(path)["tested_radius_hex"].startswith(
        "0x")


def test_corrupt_or_incomplete_record_is_not_silently_accepted(tmp_path):
    path = tmp_path / "result.json"
    pilot._atomic_json(path, _pilot_result())
    payload = json.loads(path.read_text())
    payload["tested_radius"] *= 0.5
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError):
        pilot._verified_pilot_result(path)


def test_worker_resume_skips_valid_certified_result(monkeypatch, tmp_path):
    row, source = _prepared_row()
    historical = pilot.campaign._candidate_radius(row)
    output_root = tmp_path / "out"
    path = pilot._result_path(output_root, row["property_id"], 0, 95, 100)
    pilot._atomic_json(path, _pilot_result(
        historical, 95, 100, "CERTIFIED_AT_TESTED_RADIUS",
        row["property_id"]))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(pilot, "_prepare", lambda *_args: ([(row, source)], {
        "eligible_population_sha256": "e" * 64}))
    monkeypatch.setattr(pilot, "_hardware", lambda _device: {"gpu": "fake"})
    monkeypatch.setattr(
        pilot.campaign, "execute_property",
        lambda *_args, **_kwargs: pytest.fail("valid result was recomputed"))
    result = pilot.run_worker(tmp_path, tmp_path, output_root, 0)
    assert result["persisted_evaluation_count"] == 1
    assert result["new_scientific_evaluations"] == 0
    assert result["scientific_queries"] == 0


def _prepared_row():
    property_id, historical = pilot.EXPECTED_SELECTION[0]
    return {
        "property_id": property_id,
        "sentence_ordinal": 1, "token_position": 11,
        "sequence_length": 2, "token_ids": [101, 102],
        "clean_label": 1, "nominal_prediction": 1,
        "token_input_source": {"sha256": "a" * 64},
        "cached_DeepT_reference": {
            pilot.campaign.CANDIDATE_RADIUS_FIELD: historical,
        },
    }, _eligible(property_id, historical)


def test_worker_continues_after_domain_failure_and_stops_at_first_certification(
        monkeypatch, tmp_path):
    row, source = _prepared_row()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        pilot, "_prepare", lambda *_args: ([(row, source)], {
            "eligible_population_sha256": "e" * 64}))
    monkeypatch.setattr(pilot, "_hardware", lambda _device: {"gpu": "fake"})
    monkeypatch.setattr(
        pilot.campaign, "_property_boundary_cleanup", lambda _device: {})
    calls = []

    def execute(evaluation_row, _root, _device):
        radius = pilot.campaign._candidate_radius(evaluation_row)
        calls.append(radius)
        certified = len(calls) == 2
        return {
            "scientific_evaluation_complete": True,
            "certified_at_historical_radius": certified,
            "failure_category": (None if certified else
                                 "SOUND_LAYERNORM_DOMAIN_FAILURE"),
            "failure_stage": (None if certified else "block2_to_margin"),
            "domain_failure_diagnostic": ({"reason_code": "domain"}
                                          if not certified else None),
            "final_sound_lower_margin": 0.1 if certified else None,
            "runtime_seconds": 1.0,
            "peak_gpu_allocated_bytes": 1,
            "peak_gpu_reserved_bytes": 2,
            "final_generator_count": 14000,
            "max_numerical_native_ratio": 0.01,
        }

    monkeypatch.setattr(pilot.campaign, "execute_property", execute)
    report = pilot.run_worker(tmp_path, tmp_path, tmp_path / "out", 0)
    assert len(calls) == 2
    assert calls[0] == pilot._tested_radius(
        pilot.EXPECTED_SELECTION[0][1], 95, 100)
    assert calls[1] == pilot._tested_radius(
        pilot.EXPECTED_SELECTION[0][1], 90, 100)
    assert report["persisted_evaluation_count"] == 2
    assert report["binary_searches"] == 0


def test_infrastructure_is_not_scientific_noncertification():
    row, source = _prepared_row()
    record = pilot._evaluation_record(
        row, source, {
            "scientific_evaluation_complete": False,
            "certified_at_historical_radius": None,
            "failure_category": "CUDA_OUT_OF_MEMORY",
            "failure_stage": "block0", "runtime_seconds": 1.0,
        }, pilot.EXPECTED_SELECTION[0][1], 95, 100,
        pilot._tested_radius(pilot.EXPECTED_SELECTION[0][1], 95, 100),
        1.0, {})
    assert record["classification"] == "INFRASTRUCTURE_FAILURE"
    assert record["scientific_evaluation_complete"] is False
    assert record["certified_at_tested_radius"] is None


def test_layernorm_domain_failure_maps_to_failed_tested_radius():
    row, source = _prepared_row()
    historical = pilot.campaign._candidate_radius(row)
    tested = pilot._tested_radius(historical, 95, 100)
    record = pilot._evaluation_record(
        row, source, {
            "scientific_evaluation_complete": True,
            "certified_at_historical_radius": False,
            "failure_category": "SOUND_LAYERNORM_DOMAIN_FAILURE",
            "failure_stage": "block2_to_margin",
            "domain_failure_diagnostic": {"sound_variance_lower": -0.1},
            "runtime_seconds": 1.0,
        }, historical, 95, 100, tested, 1.0, {})
    assert record["classification"] == "FAILED_AT_TESTED_RADIUS"
    assert record["certified_at_tested_radius"] is False
    assert record["layernorm_domain_diagnostic"][
        "sound_variance_lower"] == -0.1


def test_completed_negative_margin_is_scientific_failure():
    row, source = _prepared_row()
    historical = pilot.campaign._candidate_radius(row)
    tested = pilot._tested_radius(historical, 95, 100)
    record = pilot._evaluation_record(
        row, source, {
            "scientific_evaluation_complete": True,
            "certified_at_historical_radius": False,
            "failure_category": None, "failure_stage": None,
            "final_sound_lower_margin": -0.01,
            "runtime_seconds": 1.0,
        }, historical, 95, 100, tested, 1.0, {})
    assert record["classification"] == "FAILED_AT_TESTED_RADIUS"
    assert record["failure_category"] == "NONPOSITIVE_FINAL_SOUND_MARGIN"


def test_summary_does_not_count_infrastructure_as_scientific_failure(
        monkeypatch, tmp_path):
    row, source = _prepared_row()
    output_root = tmp_path / "pilot"
    historical = pilot.campaign._candidate_radius(row)
    path = pilot._result_path(output_root, row["property_id"], 0, 95, 100)
    pilot._atomic_json(path, _pilot_result(
        historical, 95, 100, "INFRASTRUCTURE_FAILURE",
        row["property_id"]))
    monkeypatch.setattr(pilot, "_prepare", lambda *_args: ([(row, source)], {
        "eligible_population_sha256": "e" * 64}))
    summary = pilot.summarize(
        tmp_path, tmp_path, output_root, tmp_path / "summary.json")
    assert summary["infrastructure_failures"] == 1
    assert summary["no_recovery_observed_at_or_above_0.25"] == 0
    assert summary["properties_recovering_at_0.95"] == 0


def test_summary_uses_authenticated_1x_failure_without_reexecution(
        monkeypatch, tmp_path):
    row, source = _prepared_row()
    output_root = tmp_path / "pilot"
    historical = pilot.campaign._candidate_radius(row)
    path = pilot._result_path(output_root, row["property_id"], 0, 95, 100)
    pilot._atomic_json(path, _pilot_result(
        historical, 95, 100, "CERTIFIED_AT_TESTED_RADIUS",
        row["property_id"]))
    monkeypatch.setattr(pilot, "_prepare", lambda *_args: ([(row, source)], {
        "eligible_population_sha256": "e" * 64}))
    summary = pilot.summarize(
        tmp_path, tmp_path, output_root, tmp_path / "summary.json")
    result = summary["properties"][0]
    assert result["historical_1.00x_outcome"] == "FAIL"
    assert result["historical_1.00x_reexecuted"] is False
    assert result["immediately_preceding_tested_failed_multiplier"] == "1.00"


def test_ladder_excludes_historical_radius_and_has_maximum_30_evaluations():
    assert all(n < d for n, d in pilot.MULTIPLIERS)
    assert len(pilot.MULTIPLIERS) == 5
    assert len(pilot.EXPECTED_SELECTION) * len(pilot.MULTIPLIERS) == 30


def test_preflight_is_metadata_only_and_reports_zero_queries(monkeypatch):
    row, source = _prepared_row()
    prepared = [(row, source)] * 6
    monkeypatch.setattr(pilot, "_prepare", lambda *_args: (prepared, {
        "campaign_identity": {
            "scientific_manifest_sha256": "s" * 64,
            "production_manifest_sha256": "p" * 64,
        },
        "token_source": {"sha256": "t" * 64},
        "eligible_population_sha256": "e" * 64,
    }))
    result = pilot.preflight(Path("campaign"), Path("artifacts"))
    assert result["scientific_queries"] == 0
    assert result["bound_calls"] == 0
    assert result["maximum_new_evaluations"] == 30
