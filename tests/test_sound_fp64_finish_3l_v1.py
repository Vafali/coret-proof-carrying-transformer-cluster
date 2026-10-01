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


def _fake_layernorm_state(minimum, *, malformed=False):
    weights = torch.zeros((2, 2, 128), dtype=torch.float64)
    if malformed:
        weights[0, 0, 0] = float("nan")

    class State:
        num_error_terms = 1
        device = torch.device("cpu")

        def __init__(self, tensor, low=None, high=None):
            self.zonotope_w = tensor
            self._low = low
            self._high = high

        def matmul(self, _matrix):
            return self

        def multiply(self, _value):
            return self

        def add(self, _other):
            return self

        def square_and_sum_and_repeat(self):
            low = torch.full((2, 128), 0.25, dtype=torch.float64)
            low[1, 7] = minimum
            high = torch.ones((2, 128), dtype=torch.float64)
            return State(self.zonotope_w.clone(), low, high)

        def concretize(self):
            return self._low, self._high

    proof = types.SimpleNamespace(
        num_tokens=2, ids=("g0",), masks=(3,), reasons=("fixture",))
    return State(weights), proof


def test_valid_nonpositive_layernorm_variance_is_controlled_domain_result(
        monkeypatch):
    state, proof = _fake_layernorm_state(-0.125)
    monkeypatch.setattr(finish3l.structural, "validate_support",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        finish3l.sound, "_ranges",
        lambda _state: (torch.tensor([-1.0], dtype=torch.float64),
                        torch.tensor([1.0], dtype=torch.float64)))
    _centered, _variance, diagnostic = finish3l._layernorm_variance_state(
        state, proof, "block2_post_attention")
    assert diagnostic["reason_code"] == finish3l.LAYERNORM_DOMAIN_REASON
    assert diagnostic["domain_admissible"] is False
    assert diagnostic["minimum_token_index"] == 1
    assert diagnostic["minimum_coordinate_index"] == 7
    assert diagnostic["sound_variance_lower"] == -0.125
    assert diagnostic["sqrt_safety_margin"] == -0.125


def test_malformed_layernorm_state_remains_infrastructure_error(monkeypatch):
    state, proof = _fake_layernorm_state(-0.125, malformed=True)
    monkeypatch.setattr(finish3l.structural, "validate_support",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        finish3l.sound, "_ranges",
        lambda _state: (torch.tensor([-1.0], dtype=torch.float64),
                        torch.tensor([1.0], dtype=torch.float64)))
    with pytest.raises(RuntimeError, match="malformed nonfinite") as caught:
        finish3l._layernorm_variance_state(
            state, proof, "block2_post_attention")


def test_positive_layernorm_variance_remains_admissible(monkeypatch):
    state, proof = _fake_layernorm_state(0.125)
    monkeypatch.setattr(finish3l.structural, "validate_support",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        finish3l.sound, "_ranges",
        lambda _state: (torch.tensor([-1.0], dtype=torch.float64),
                        torch.tensor([1.0], dtype=torch.float64)))
    _centered, _variance, diagnostic = finish3l._layernorm_variance_state(
        state, proof, "block2_post_attention")
    assert diagnostic["sound_variance_lower"] == 0.125
    assert diagnostic["domain_admissible"] is True
    assert diagnostic["sqrt_input_lower"] > finish3l.LAYER_NORM_EPSILON
