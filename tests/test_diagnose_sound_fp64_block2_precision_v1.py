import importlib.util
import sys
import types
from pathlib import Path

import pytest


if "gmpy2" not in sys.modules:
    sys.modules["gmpy2"] = types.SimpleNamespace()

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "precision_diagnosis",
    ROOT / "scripts/diagnose_sound_fp64_block2_precision_v1.py")
diagnosis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnosis)


def _row(multiplier="0.95", radius=0.00095, sound=-0.2,
         numerical=0.05, generators=14000):
    diagnostic = {
        "input_shape": [generators + 1, 20, 128],
        "token_count": 20, "hidden_dimension": 128,
        "generator_count": generators,
        "minimum_token_index": 4, "minimum_coordinate_index": 0,
        "nominal_centered_second_moment": 1.0,
        "variance_affine_center": 0.8,
        "variance_lower_support": 1.0,
        "variance_upper_support": 1.0,
        "plain_relational_variance_lower_bound": -0.15,
        "numerical_variance_widening_upper_bound": numerical,
        "sound_variance_lower": sound,
        "sound_variance_upper_at_minimum": 1.8,
        "layernorm_epsilon": 1e-12,
        "sqrt_input_lower": sound + 1e-12,
        "sqrt_safety_margin": sound,
        "native_generator_count": 12000,
        "numerical_generator_count": 2000,
    }
    return {
        "classification": "FAILED_AT_TESTED_RADIUS",
        "failure_category": "SOUND_LAYERNORM_DOMAIN_FAILURE",
        "multiplier": {"display": multiplier},
        "tested_radius": radius, "tested_radius_hex": radius.hex(),
        "runtime_seconds": 10.0,
        "layernorm_domain_diagnostic": diagnostic,
    }


def test_domain_extractor_reports_decisive_components():
    result = diagnosis._domain_row(_row())
    assert result["at_generator_cap"] is True
    assert result["plain_relational_already_nonpositive"] is True
    assert result["positivity_deficit"] == 0.2
    assert result["numerical_widening_to_deficit_ratio"] == 0.25
    assert result["minimum_token_index"] == 4


def test_non_domain_record_rejects():
    row = _row()
    row["failure_category"] = "CUDA_OUT_OF_MEMORY"
    with pytest.raises(RuntimeError, match="not a LayerNorm-domain"):
        diagnosis._domain_row(row)


def test_incomplete_domain_diagnostic_rejects():
    row = _row()
    del row["layernorm_domain_diagnostic"]["variance_affine_center"]
    with pytest.raises(RuntimeError, match="incomplete"):
        diagnosis._domain_row(row)


def test_radius_trend_distinguishes_shrinking_source_from_constant_floor():
    first = diagnosis._domain_row(_row("0.95", 0.00095, -0.2, 0.05))
    last = diagnosis._domain_row(_row("0.25", 0.00025, -0.19, 0.049))
    trend = diagnosis._trend([first, last])
    assert trend["tested_radius_ratio"] == pytest.approx(0.25 / 0.95)
    assert trend["numerical_widening_ratio"] == pytest.approx(0.049 / 0.05)
    assert trend["generator_count_constant"] is True
    assert trend["failing_token_constant"] is True


def test_diagnose_is_read_only_and_never_calls_evaluator(monkeypatch, tmp_path):
    property_id = diagnosis.DEFAULT_PROPERTY
    monkeypatch.setattr(diagnosis.pilot, "preflight", lambda *_args: {
        "selected_properties": [{"property_id": property_id}]})
    monkeypatch.setattr(
        diagnosis, "_source_paths", lambda _root: {
            property_id: tmp_path / "result.json",
            "certified": tmp_path / "certified.json",
        })

    source = {
        "property_id": property_id,
        "failure_category": "SOUND_LAYERNORM_DOMAIN_FAILURE",
        "scientific_evaluation_complete": True,
        "historical_candidate_radius": 0.001,
        "historical_candidate_radius_hex": (0.001).hex(),
    }
    monkeypatch.setattr(
        diagnosis.campaign, "_verified_result", lambda _path: source)
    monkeypatch.setattr(
        diagnosis, "_pilot_rows", lambda *_args: [_row()])
    monkeypatch.setattr(
        diagnosis, "_certified_comparator", lambda _paths: {
            "property_id": "certified"})
    monkeypatch.setattr(
        diagnosis.campaign, "execute_property",
        lambda *_args, **_kwargs: pytest.fail("diagnostic entered evaluator"))
    report = diagnosis.diagnose(tmp_path, tmp_path, tmp_path, property_id)
    assert report["scientific_queries"] == 0
    assert report["bound_calls"] == 0
    assert report["evaluations"][0]["sound_variance_lower"] == -0.2
