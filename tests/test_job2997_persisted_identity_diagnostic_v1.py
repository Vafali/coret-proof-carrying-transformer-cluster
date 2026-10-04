from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "research_hab")]
if "gmpy2" not in sys.modules:
    sys.modules["gmpy2"] = types.SimpleNamespace()
SPEC = importlib.util.spec_from_file_location(
    "job2997_diagnostic",
    ROOT / "scripts/verify_job2997_pre_layernorm_capture_v1.py")
DIAG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAG)
RUNNER = DIAG.runner


def snapshot(tokens=2):
    weights = torch.zeros(3, tokens, 128, dtype=torch.float64)
    weights[0, 0, 0] = 1.25
    return {
        "weights": weights,
        "range_low": torch.tensor([-1.0, -0.5], dtype=torch.float64),
        "range_high": torch.tensor([1.0, 0.75], dtype=torch.float64),
        "proof": {
            "ids": ["g0", "g1"],
            "masks": [1, 3],
            "reasons": ["native_semantic", "fp64_numerical"],
            "num_tokens": tokens,
        },
    }


def state_record(value):
    hashes = RUNNER._frontier_state_hashes(value)
    return {**hashes,
            "source_ids_sha256": hashes["generator_ids_sha256"],
            "hidden_dimension": hashes["feature_dimension"]}


def compare(value, expected=None, record=None):
    identity = RUNNER._snapshot_state_identity(value)
    expected = identity if expected is None else expected
    return DIAG.compare_snapshot(
        value, expected, expected,
        state_record(value) if record is None else record)


def test_identical_state_is_canonically_equal():
    result = compare(snapshot())
    assert result["classification"] == "PERSISTED_STATE_CANONICALLY_IDENTICAL"
    assert result["first_differing_component"] is None


@pytest.mark.parametrize(("mutation", "classification"), (
    ("coefficient", "PERSISTED_WEIGHTS_DIFFER"),
    ("source_id", "PERSISTED_SOURCE_IDS_DIFFER"),
    ("range_low", "PERSISTED_RANGE_LOW_DIFFER"),
    ("range_high", "PERSISTED_RANGE_HIGH_DIFFER"),
    ("mask", "PERSISTED_MASKS_DIFFER"),
    ("provenance", "PERSISTED_PROVENANCE_DIFFER"),
    ("topology", "PERSISTED_TOPOLOGY_DIFFER"),
))
def test_semantic_mutation_classified(mutation, classification):
    base = snapshot()
    expected = RUNNER._snapshot_state_identity(base)
    changed = copy.deepcopy(base)
    if mutation == "coefficient":
        changed["weights"][0, 0, 0] += 1.0
    elif mutation == "source_id":
        changed["proof"]["ids"][0] = "changed"
    elif mutation == "range_low":
        changed["range_low"][0] -= 0.25
    elif mutation == "range_high":
        changed["range_high"][0] += 0.25
    elif mutation == "mask":
        changed["proof"]["masks"][0] ^= 1
    elif mutation == "provenance":
        changed["proof"]["reasons"][0] = "changed"
    else:
        changed = snapshot(tokens=3)
    result = compare(changed, expected)
    assert result["classification"] == classification


def test_canonical_digest_construction_bug_classified():
    value = snapshot()
    expected = RUNNER._snapshot_state_identity(value)
    expected = {**expected, "canonical_state_identity_sha256": "0" * 64}
    result = compare(value, expected)
    assert result["classification"] == \
        "PERSISTED_CANONICAL_DIGEST_CONSTRUCTION_BUG"


def test_raw_artifact_manifest_mismatch_classified_independently():
    value = snapshot()
    record = state_record(value)
    record["center_sha256"] = "0" * 64
    result = compare(value, record=record)
    assert result["classification"] == "PERSISTED_ARTIFACT_MANIFEST_MISMATCH"
    assert result["raw_artifact_manifest_comparison"][
        "all_manifest_hash_fields_equal"] is False


def test_diagnostic_json_written_before_nonzero_exit(tmp_path, monkeypatch):
    output = tmp_path / "diagnostic.json"
    monkeypatch.setattr(sys, "argv", [
        "verify", "--capture-root", str(tmp_path / "absent"),
        "--job2995-trace", str(tmp_path / "absent-trace.json"),
        "--output", str(output)])
    assert DIAG.main() == 1
    assert output.is_file()
    report = __import__("json").loads(output.read_text())
    assert report["final_status"] == \
        "EXISTING_JOB2997_PRE_LAYERNORM_CAPTURE_REJECTED"


def test_strict_semantic_verifier_remains_fail_closed(tmp_path):
    with pytest.raises(Exception):
        RUNNER._verify_pre_layernorm_input_capture(
            tmp_path / "missing-manifest.json")
