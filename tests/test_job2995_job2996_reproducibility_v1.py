from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "reproducibility_inspector",
    ROOT / "scripts/inspect_job2995_job2996_reproducibility_v1.py")
INSPECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSPECTOR)
LINKAGE = INSPECTOR.linkage


def _write_json(path: Path, value: dict) -> None:
    payload = dict(value)
    payload["record_sha256"] = LINKAGE._canonical(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _state(offset=0.0) -> dict:
    weights = torch.zeros(3, 2, 4, dtype=torch.float64)
    weights[0, 0, 0] = 1.0 + offset
    weights[1, 0, 1] = 0.5
    weights[2, 1, 2] = 0.25
    return {
        "weights": weights,
        "range_low": torch.tensor([-1.0, -0.5], dtype=torch.float64),
        "range_high": torch.tensor([1.0, 0.75], dtype=torch.float64),
        "proof": {
            "masks": [1, 2], "ids": ["native", "numerical"],
            "reasons": ["native_semantic", "fp64_roundoff_coordinate_box"],
            "num_tokens": 2,
        },
    }


def _frontier(root: Path, states: dict) -> Path:
    root.mkdir(parents=True)
    identity = {
        "property_id": LINKAGE.PROPERTY_ID,
        "multiplier": LINKAGE.MULTIPLIER,
        "tested_radius": LINKAGE.TESTED_RADIUS,
        "tested_radius_hex": LINKAGE.TESTED_RADIUS_HEX,
        "stage_label": LINKAGE.FRONTIER_STAGE,
        "pinned_deept_revision": LINKAGE.PINNED_REVISION,
        "scientific_manifest_sha256": LINKAGE.SCIENTIFIC_MANIFEST_SHA256,
        "production_manifest_sha256": LINKAGE.PRODUCTION_MANIFEST_SHA256,
        "source_set_model": LINKAGE.SOURCE_SET_MODEL,
    }
    reductions, aliases = {"fixture": []}, {"fixture": list(states)[0]}
    artifact = root / "states.pt"
    torch.save({
        "schema": LINKAGE.FRONTIER_CAPTURE_SCHEMA,
        "pinned_revision": LINKAGE.PINNED_REVISION,
        "identity": identity, "states": states,
        "reductions": reductions, "aliases": aliases,
    }, artifact)
    result = root / "result.json"; result.write_text("{}\n")
    manifest = root / "manifest.json"
    variants = []
    for name, state in states.items():
        row = LINKAGE._legacy_state_record(state); row["state_key"] = name
        variants.append(row)
    _write_json(manifest, {
        "schema": LINKAGE.FRONTIER_MANIFEST_SCHEMA, **identity,
        "tensor_artifact_path": artifact.name,
        "tensor_artifact_sha256": LINKAGE._sha256(artifact),
        "artifact_identity": identity,
        "result_path": result.name, "result_sha256": LINKAGE._sha256(result),
        "reduction_records_sha256": LINKAGE._canonical(reductions),
        "aliases": aliases, "variants": variants,
    })
    return manifest


def _trace(path: Path, state_identity: dict, reserve: float) -> Path:
    _write_json(path, {
        "schema": "CORET_PSD_LAYERNORM_INVOCATION_TRACE_V2",
        "property_id": LINKAGE.PROPERTY_ID,
        "tested_radius": LINKAGE.TESTED_RADIUS,
        "tested_radius_hex": LINKAGE.TESTED_RADIUS_HEX,
        "invocations": [{
            "ordinal": 5, "stage": LINKAGE.TARGET_STAGE,
            "layernorm_index": 5, "state_identity": state_identity,
            "psd_applied": True,
            "semantic_range_certificate": {"reserve": reserve},
        }],
    })
    return path


def test_all_state_comparison_localizes_first_common_difference(tmp_path):
    names = [LINKAGE.STATE_KEY, "ffn_first_post_reduction", "last"]
    left = {name: _state() for name in names}
    right = copy.deepcopy(left)
    right["ffn_first_post_reduction"]["weights"][0, 0, 0] += 0.125
    expected = _frontier(tmp_path / "expected", left)
    reproduced = _frontier(tmp_path / "reproduced", right)
    report = INSPECTOR.compare_all_states(expected, reproduced)
    assert report["first_common_state_that_differs"] == (
        "ffn_first_post_reduction")
    assert report["states"][0]["semantic_equal"] is True
    assert report["states"][1]["center"][
        "maximum_absolute_difference"] == 0.125


def test_trace_comparison_authenticates_equal_input_and_reports_other_diff(
        tmp_path):
    state_identity = {"weights_sha256": "same", "generator_count": 7}
    left = _trace(tmp_path / "left.json", state_identity, 1.0)
    right = _trace(tmp_path / "right.json", state_identity, 2.0)
    report = INSPECTOR.compare_traces(left, right)
    assert report["target_input_state_identity_available_in_both"] is True
    assert report["target_input_state_identity_equal"] is True
    assert report["field_difference_count"] == 1


def test_runtime_difference_has_precedence_over_unresolved_fields(tmp_path):
    left = tmp_path / "2995.log"; right = tmp_path / "2996.log"
    left.write_text("CUDA_VISIBLE_DEVICES=1\n")
    right.write_text("CUDA_VISIBLE_DEVICES=0\n")
    runtime = INSPECTOR.runtime_configuration([left], [right])
    result = INSPECTOR.classify(
        traces={"target_input_state_identity_equal": True},
        runtime=runtime,
        provenance={"only_changed_executed_scientific_runner": True})
    assert result["root_cause"] == "ROOT_CAUSE_RUNTIME_CONFIGURATION_DRIFT"


def test_missing_job_provenance_remains_unresolved():
    runtime = INSPECTOR.runtime_configuration([], [])
    result = INSPECTOR.classify(
        traces={"target_input_state_identity_equal": True},
        runtime=runtime,
        provenance={"only_changed_executed_scientific_runner": True})
    assert result["root_cause"] == "ROOT_CAUSE_UNRESOLVED"


def test_capture_hook_audit_records_schedule_but_not_value_mutation():
    report = INSPECTOR.capture_hook_audit(ROOT)
    assert report["in_place_tensor_operations"] is False
    assert report["calls_original_once_with_same_objects"] is True
    assert report["synchronization_or_order_change"] is True
