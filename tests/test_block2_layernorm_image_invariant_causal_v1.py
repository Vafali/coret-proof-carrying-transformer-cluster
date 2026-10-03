from __future__ import annotations

import copy
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
    "layernorm_image_oracle",
    ROOT / "scripts/analyze_block2_layernorm_image_invariant_causal_v1.py")
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


def state(center=None, generators=None, low=None, high=None, ids=None):
    center = np.zeros(128) if center is None else np.asarray(center, dtype=np.float64)
    generators = (np.empty((0, 128), dtype=np.float64) if generators is None
                  else np.asarray(generators, dtype=np.float64))
    count = len(generators)
    return {"center": center, "generators": generators,
            "low": np.asarray([] if low is None else low, dtype=np.float64),
            "high": np.asarray([] if high is None else high, dtype=np.float64),
            "ids": list(ids or [f"g{i}" for i in range(count)]),
            "masks": [1] * count, "reasons": ["native_semantic"] * count,
            "num_tokens": 1}


def inactive_h(source):
    return {**source, "center": np.full(128, -1.0),
            "generators": np.zeros_like(source["generators"])}


def problem(y, invariant, residual=None):
    residual = y if residual is None else residual
    return ORACLE.relu.ReluCancellationProblem(
        inactive_h(y), residual, np.zeros((128, 128)), np.zeros(128),
        additional_equalities=invariant)


def test_exact_layernorm_output_satisfies_weighted_mean_zero():
    center = np.arange(128, dtype=np.float64)
    center -= center.mean()
    y = state(center)
    inv = ORACLE._layernorm_invariant(y, np.ones(128), np.zeros(128))
    assert ORACLE._invariant_residual(inv, []) == [Fraction(0)]


def test_zero_gamma_coordinate_is_exact_equality():
    gamma = np.ones(128); gamma[7] = 0.0
    beta = np.zeros(128); beta[7] = 2.0
    center = np.zeros(128); center[7] = 2.0
    inv = ORACLE._layernorm_invariant(state(center), gamma, beta)
    assert inv["zero_gamma_count"] == 1
    assert inv["equality_count"] == 2
    assert ORACLE._invariant_residual(inv, []) == [Fraction(0), Fraction(0)]


def _excluded_problem():
    center = np.zeros(128); center[1] = 1.0
    generators = np.zeros((1, 128)); generators[0, 0] = 1.0
    y = state(center, generators, [-1.0], [1.0], ["shared"])
    inv = ORACLE._layernorm_invariant(y, np.ones(128), np.zeros(128))
    return problem(y, inv), inv


def test_invariant_constrained_hull_infeasible_has_exact_farkas():
    p, _inv = _excluded_problem()
    result, _ = ORACLE.relu.solve_hull(p)
    assert result.status == 2
    certificate, _ = ORACLE.relu.propose_farkas(p)
    assert certificate is not None
    checked = ORACLE.relu._verify_farkas(
        p, ORACLE.relu._ratios(certificate["lambda"]),
        ORACLE.relu._ratios(certificate["mu"]))
    assert checked["exact_stationarity"] is True


def test_corrupted_farkas_certificate_rejected():
    p, _inv = _excluded_problem()
    certificate, _ = ORACLE.relu.propose_farkas(p)
    mus = [Fraction(0) for _ in certificate["mu"]]
    with pytest.raises(RuntimeError):
        ORACLE.relu._verify_farkas(
            p, ORACLE.relu._ratios(certificate["lambda"]), mus)


def test_large_source_farkas_uses_exact_bound_elimination():
    count = 513
    h = state(np.full(128, -1.0), np.zeros((count, 128)),
              np.full(count, -1.0), np.ones(count),
              [f"g{i}" for i in range(count)])
    residual = copy.deepcopy(h)
    residual["center"] = np.zeros(128); residual["center"][1] = 1.0
    p = ORACLE.relu.ReluCancellationProblem(
        h, residual, np.zeros((128, 128)), np.zeros(128))
    result, _ = ORACLE.relu.solve_hull(p, 5.0)
    assert result.status == 2
    certificate, diagnostics = ORACLE.relu.propose_farkas(p, 5.0)
    assert certificate is not None
    assert diagnostics["exact_repair"]["repair_backend"] == (
        "source_bound_elimination_plus_bareiss")


def test_unstable_relu_farkas_row_order_replays_exactly():
    h = state(np.zeros(128), np.ones((1, 128)), [-1.0], [1.0], ["g"])
    residual = state(np.r_[0.0, 1.0, np.zeros(126)],
                     np.zeros((1, 128)), [-1.0], [1.0], ["g"])
    p = ORACLE.relu.ReluCancellationProblem(
        h, residual, np.zeros((128, 128)), np.zeros(128))
    result, _ = ORACLE.relu.solve_hull(p, 5.0)
    assert result.status == 2
    certificate, _ = ORACLE.relu.propose_farkas(p, 5.0)
    assert certificate is not None


def _feasible_problem():
    generators = np.zeros((1, 128))
    generators[0, 0], generators[0, 1] = 1.0, -1.0
    y = state(np.zeros(128), generators, [-1.0], [1.0], ["shared"])
    inv = ORACLE._layernorm_invariant(y, np.ones(128), np.zeros(128))
    return problem(y, inv), inv


def test_invariant_constrained_exact_witness_replay(tmp_path):
    p, _inv = _feasible_problem()
    witness, status = ORACLE.relu.exact_fixed_pattern_witness(
        p, np.array([0.0]), [False] * 128,
        tmp_path / "witness.json", 5.0)
    assert status["verified"] is True
    assert witness["invariant_check"] is True


def test_witness_violating_invariant_rejected(tmp_path):
    generators = np.zeros((1, 128)); generators[0, 0] = 1.0
    y = state(np.zeros(128), generators, [-1.0], [1.0], ["shared"])
    inv = ORACLE._layernorm_invariant(y, np.ones(128), np.zeros(128))
    residual = state(np.zeros(128), np.zeros((1, 128)), [-1.0], [1.0], ["shared"])
    p = problem(y, inv, residual)
    witness, _ = ORACLE.relu.exact_fixed_pattern_witness(
        p, np.array([0.0]), [False] * 128, tmp_path / "valid.json", 5.0)
    corrupted = copy.deepcopy(witness)
    corrupted["xi_rationals"][0] = {"numerator": "1", "denominator": "2"}
    with pytest.raises(RuntimeError, match="additional equality replay"):
        ORACLE.relu.verify_exact_relu_witness(p, corrupted)


def test_shared_residual_ffn_source_identity_preserved():
    p, _inv = _feasible_problem()
    assert p.source["ids"] == ["shared"] and p.n == 1


def _parameter_blobs(gamma=None):
    gamma = torch.ones(128) if gamma is None else gamma
    checkpoint = {
        f"{ORACLE.LN_PARAMETER}.weight": gamma,
        f"{ORACLE.LN_PARAMETER}.bias": torch.zeros(128),
    }
    stream = io.BytesIO(); torch.save(checkpoint, stream)
    checkpoint_raw = stream.getvalue()
    config_raw = json.dumps({"num_hidden_layers": 3,
                             "hidden_size": 128}).encode()
    source = {"pinned_revision": "rev",
              "scientific_manifest_sha256": "science",
              "production_manifest_sha256": "production",
              "checkpoint_git_path": "checkpoint", "config_git_path": "config",
              "checkpoint_sha256": hashlib.sha256(checkpoint_raw).hexdigest(),
              "config_sha256": hashlib.sha256(config_raw).hexdigest()}
    identity = {"pinned_deept_revision": "rev",
                "scientific_manifest_sha256": "science",
                "production_manifest_sha256": "production"}
    blobs = {"checkpoint": checkpoint_raw, "config": config_raw}
    return identity, source, lambda _revision, path: blobs[path]


def test_gamma_beta_authentication_and_zero_count():
    gamma = torch.ones(128); gamma[3] = 0.0
    identity, source, loader = _parameter_blobs(gamma)
    _gamma, _beta, record = ORACLE._load_layernorm_parameters(
        identity, source=source, blob_loader=loader)
    assert record["zero_gamma_count"] == 1
    assert record["parameter_identity_authenticated"] is True


def test_gamma_beta_authentication_mismatch_hard_failure():
    identity, source, loader = _parameter_blobs()
    identity["production_manifest_sha256"] = "wrong"
    with pytest.raises(RuntimeError, match="parameter identity differs"):
        ORACLE._load_layernorm_parameters(
            identity, source=source, blob_loader=loader)


def test_capture_identity_mismatch_hard_failure():
    identity, source, loader = _parameter_blobs()
    source["checkpoint_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="checkpoint artifact SHA"):
        ORACLE._load_layernorm_parameters(
            identity, source=source, blob_loader=loader)


def test_reusable_pre_layernorm_sources_are_recognized():
    before = state(np.zeros(128), np.zeros((1, 128)), [-1.0], [1.0], ["g"])
    after = state(np.zeros(128), np.zeros((2, 128)), [-1.0, -1.0],
                  [1.0, 1.0], ["g", "fresh"])
    compatible, reason = ORACLE._compatible_source_subset(before, after)
    assert compatible is True and "preserved" in reason


def test_incompatible_pre_layernorm_source_is_rejected():
    before = state(np.zeros(128), np.zeros((1, 128)), [-1.0], [1.0], ["g"])
    after = state(np.zeros(128), np.zeros((1, 128)), [0.0], [1.0], ["g"])
    compatible, reason = ORACLE._compatible_source_subset(before, after)
    assert compatible is False and "metadata differs" in reason


def test_absent_reusable_capture_requests_passive_capture_only(tmp_path):
    output = state()
    identity = {"property_id": ORACLE.frontier.zero.PROPERTY_ID,
                "tested_radius": ORACLE.TESTED_RADIUS,
                "minimum_token_index": 0,
                "pinned_deept_revision": "rev"}
    result = ORACLE._inspect_reusable_input(
        tmp_path / "missing_frontier.json", identity, output)
    assert result["found"] is False
    assert result["reason"] == (
        "LAYERNORM_EXACT_GRAPH_NEEDS_ONE_PASSIVE_INPUT_CAPTURE")
