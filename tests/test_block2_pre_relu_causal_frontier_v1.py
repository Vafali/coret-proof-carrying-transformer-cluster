from __future__ import annotations

from fractions import Fraction
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "pre_relu_frontier",
    ROOT / "scripts/analyze_block2_pre_relu_causal_frontier_v1.py")
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def _row(order, status, incoming=None, name=None):
    return {"order": order, "exact_relu_status": status,
            "incoming_transition_type": incoming,
            "captured_state_name": name or f"s{order}"}


def test_excluded_to_feasible_across_reduction():
    rows = [_row(0, ANALYZER.EXCLUDED),
            _row(1, ANALYZER.FEASIBLE,
                 "post_attention_layernorm_reduction")]
    result = ANALYZER._causal_frontier(rows)
    assert result["causal_frontier_found"] is True
    assert result["transition_type"] == "post_attention_layernorm_reduction"


def test_excluded_to_feasible_across_combined_numerical_boundary():
    rows = [_row(0, ANALYZER.EXCLUDED),
            _row(1, ANALYZER.FEASIBLE,
                 "first_affine_plus_numerical_injection_combined")]
    result = ANALYZER._causal_frontier(rows)
    assert "cannot separate" in result["repair_family_to_test_next"]


def test_feasible_at_earliest_is_upstream():
    result = ANALYZER._causal_frontier([_row(0, ANALYZER.FEASIBLE)])
    assert result["transition_type"] == (
        "CAUSAL_FRONTIER_UPSTREAM_OF_CAPTURED_FFN_PATH")
    assert result["causal_frontier_found"] is False


def test_unresolved_to_feasible_is_not_guessed():
    rows = [_row(0, ANALYZER.UNRESOLVED), _row(1, ANALYZER.FEASIBLE)]
    result = ANALYZER._causal_frontier(rows)
    assert result["transition_type"] == "PRE_RELU_CAUSAL_FRONTIER_UNRESOLVED"
    assert result["unresolved_predecessors"] == ["s0"]


def _parameter_blobs():
    checkpoint = {
        f"{ANALYZER.FFN_FIRST_PARAMETER}.weight": torch.eye(128),
        f"{ANALYZER.FFN_FIRST_PARAMETER}.bias": torch.zeros(128),
        f"{ANALYZER.frontier.FFN_SECOND_PARAMETER}.weight": torch.eye(128),
        f"{ANALYZER.frontier.FFN_SECOND_PARAMETER}.bias": torch.zeros(128),
    }
    stream = io.BytesIO(); torch.save(checkpoint, stream)
    checkpoint_raw = stream.getvalue()
    config_raw = json.dumps({"num_hidden_layers": 3, "hidden_size": 128,
                             "intermediate_size": 128}).encode()
    source = {
        "pinned_revision": "rev", "scientific_manifest_sha256": "science",
        "production_manifest_sha256": "production",
        "checkpoint_git_path": "checkpoint", "config_git_path": "config",
        "checkpoint_sha256": hashlib.sha256(checkpoint_raw).hexdigest(),
        "config_sha256": hashlib.sha256(config_raw).hexdigest(),
    }
    identity = {"pinned_deept_revision": "rev",
                "scientific_manifest_sha256": "science",
                "production_manifest_sha256": "production"}
    blobs = {"checkpoint": checkpoint_raw, "config": config_raw}
    return identity, source, lambda _revision, path: blobs[path]


def test_exact_w1_b1_authentication():
    identity, source, loader = _parameter_blobs()
    weight, bias, record = ANALYZER._load_ffn_first_parameters(
        identity, source=source, blob_loader=loader)
    assert np.array_equal(weight, np.eye(128))
    assert np.array_equal(bias, np.zeros(128))
    assert record["parameter_identity_authenticated"] is True


def test_exact_w2_b2_authentication():
    identity, source, loader = _parameter_blobs()
    weight, bias, record = ANALYZER.frontier._load_ffn_second_parameters(
        identity, source=source, blob_loader=loader)
    assert weight.shape == (128, 128) and bias.shape == (128,)
    assert record["parameter_identity_authenticated"] is True


def _state():
    center = np.zeros(128); center[0] = 0.25
    generators = np.zeros((1, 128)); generators[0, 0] = 0.5
    return {"center": center, "generators": generators,
            "low": np.array([-1.0]), "high": np.array([1.0]),
            "ids": ["shared"], "masks": [1],
            "reasons": ["native_semantic"], "num_tokens": 1}


def test_exact_w1_composition_replays_not_rounded_product():
    state = _state()
    state["center"][0] = 0.3
    weight = np.zeros((128, 128)); weight[0, 0] = 0.1
    bias = np.zeros(128)
    numeric, exact = ANALYZER._compose_first_affine(state, weight, bias)
    expected = (Fraction.from_float(0.1) * Fraction.from_float(0.3))
    assert exact["center"](0) == expected
    assert exact["generator"](0, 0) == (
        Fraction.from_float(0.1) * Fraction.from_float(0.5))
    assert Fraction.from_float(float(numeric["center"][0])) != expected


def test_composed_w1_exact_witness_replays_through_relu_and_w2(tmp_path):
    source = _state(); source["center"][:] = -1.0
    source["generators"][:] = 0.0
    h, exact = ANALYZER._compose_first_affine(
        source, np.eye(128), np.zeros(128))
    residual = _state(); residual["center"][:] = 0.0
    residual["generators"][:] = 0.0
    problem = ANALYZER.relu.ReluCancellationProblem(
        h, residual, np.eye(128), np.zeros(128),
        exact_preactivation=exact)
    witness, status = ANALYZER.relu.exact_fixed_pattern_witness(
        problem, np.array([0.0]), [False] * 128,
        tmp_path / "composed.json", 5.0)
    assert status["verified"] is True
    replay = ANALYZER.relu.verify_exact_relu_witness(problem, witness)
    assert replay["exact_preactivation_composition_replayed"] is True


def test_shared_residual_and_ffn_identity_is_preserved():
    state = _state()
    numeric, exact = ANALYZER._compose_first_affine(
        state, np.eye(128), np.zeros(128))
    problem = ANALYZER.relu.ReluCancellationProblem(
        numeric, state, np.zeros((128, 128)), np.zeros(128),
        exact_preactivation=exact)
    assert problem.n == 1 and problem.source["ids"] == ["shared"]


def test_exact_witness_replay_remains_authoritative(tmp_path):
    h = _state(); h["center"][:] = -1.0; h["generators"][:] = 0.0
    residual = _state(); residual["center"][:] = 0.0
    residual["generators"][:] = 0.0
    problem = ANALYZER.relu.ReluCancellationProblem(
        h, residual, np.zeros((128, 128)), np.zeros(128))
    witness, status = ANALYZER.relu.exact_fixed_pattern_witness(
        problem, np.array([0.0]), [False] * 128,
        tmp_path / "witness.json", 5.0)
    assert status["verified"] and witness["maximum_exact_residual"] == "0"


def test_exact_farkas_replay_remains_authoritative():
    h = _state(); h["center"][:] = -1.0; h["generators"][:] = 0.0
    residual = _state(); residual["center"][:] = 0.0
    residual["center"][1] = 1.0; residual["generators"][:] = 0.0
    problem = ANALYZER.relu.ReluCancellationProblem(
        h, residual, np.zeros((128, 128)), np.zeros(128))
    result, _ = ANALYZER.relu.solve_hull(problem)
    assert result.status == 2
    certificate, _ = ANALYZER.relu.propose_farkas(problem)
    checked = ANALYZER.relu._verify_farkas(
        problem, ANALYZER.relu._ratios(certificate["lambda"]),
        ANALYZER.relu._ratios(certificate["mu"]))
    assert checked["exact_stationarity"] is True


def test_corrupted_parameter_identity_rejected():
    identity, source, loader = _parameter_blobs()
    identity["pinned_deept_revision"] = "wrong"
    with pytest.raises(RuntimeError, match="source identity differs"):
        ANALYZER._load_ffn_first_parameters(
            identity, source=source, blob_loader=loader)


def test_corrupted_source_identity_rejected():
    left = _state(); right = _state(); right["low"][0] = -0.5
    with pytest.raises(RuntimeError, match="shared source range/provenance"):
        ANALYZER.frontier._align_sources(left, right)


def test_missing_pre_injection_boundary_is_explicit_not_fabricated():
    captured = [name for name, *_ in ANALYZER.BOUNDARIES]
    assert "ffn_first_before_numerical_injection" not in captured
    assert captured == [
        "post_attention_ln_pre_reduction",
        "post_attention_ln_post_reduction",
        "ffn_first_pre_reduction", "ffn_first_post_reduction"]
