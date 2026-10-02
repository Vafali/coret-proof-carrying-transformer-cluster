from __future__ import annotations

from fractions import Fraction
import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ffn_joint_frontier",
    ROOT / "scripts/analyze_block2_ffn_joint_cancellation_frontier_v1.py")
FRONTIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FRONTIER)


def state(center, generators, low, high, ids=None, reasons=None):
    generators = np.asarray(generators, dtype=np.float64)
    count = len(generators)
    return {
        "center": np.asarray(center, dtype=np.float64),
        "generators": generators,
        "low": np.asarray(low, dtype=np.float64),
        "high": np.asarray(high, dtype=np.float64),
        "ids": list(ids or [f"g{index}" for index in range(count)]),
        "masks": [1] * count,
        "reasons": list(reasons or ["native_semantic"] * count),
        "num_tokens": 1,
    }


def test_joint_source_alignment_preserves_shared_id_and_range():
    left = state([0.0, 0.0], [[1.0, 0.0]], [-1.0], [1.0], ["shared"])
    right = state([0.0, 0.0], [[0.0, 1.0], [1.0, 1.0]],
                  [-1.0, 0.0], [1.0, 2.0], ["shared", "right_only"])
    aligned_left, aligned_right, source = FRONTIER._align_sources(left, right)
    assert source["ids"] == ["shared", "right_only"]
    assert aligned_left["generators"].tolist() == [[1.0, 0.0], [0.0, 0.0]]
    assert aligned_right["generators"].tolist() == [[0.0, 1.0], [1.0, 1.0]]


def test_conflicting_shared_source_range_is_rejected():
    left = state([0.0, 0.0], [[1.0, 0.0]], [-1.0], [1.0], ["shared"])
    right = state([0.0, 0.0], [[0.0, 1.0]], [-1.0], [0.5], ["shared"])
    with pytest.raises(RuntimeError, match="range/provenance differs"):
        FRONTIER._align_sources(left, right)


def test_exact_remaining_affine_uses_dyadic_products_not_rounded_matmul():
    branch = state([0.5, -0.25], [[0.25, 0.5]], [-1.0], [1.0])
    skip = state([1.0, -1.0], [[-0.5, 0.25]], [-1.0], [1.0])
    weight = np.array([[0.5, 0.25], [-0.75, 0.5]], dtype=np.float64)
    bias = np.array([0.125, -0.25], dtype=np.float64)
    model = FRONTIER.ExactJointAffine(
        branch, skip, weight=weight, bias=bias, label="mapped")
    expected = ((Fraction(1) + Fraction(1, 8)
                 + Fraction(1, 2) * Fraction(1, 2)
                 + Fraction(1, 4) * Fraction(-1, 4))
                - (Fraction(-1) - Fraction(1, 4)
                   + Fraction(-3, 4) * Fraction(1, 2)
                   + Fraction(1, 2) * Fraction(-1, 4)))
    assert model.exact_center_difference(0) == expected


def test_exact_joint_zero_certificate_replays_shared_correlation():
    # Sum is [1, -1] + xi * [-1, 1], hence xi=1 is exactly constant.
    branch = state([0.0, 0.0], [[-1.0, 1.0]], [0.0], [2.0])
    skip = state([1.0, -1.0], [[0.0, 0.0]], [0.0], [2.0])
    model = FRONTIER.ExactJointAffine(branch, skip, label="feasible")
    problem = model.problem()
    certificate, status = FRONTIER.zero.construct_exact_zero_certificate(
        problem, np.array([1.0]), 2, solve_timeout_seconds=5.0)
    assert status["verified"] is True
    assert status["status_code"] == FRONTIER.zero.EXACT_ZERO_VERIFIED
    checked = FRONTIER.zero.verify_exact_zero_certificate(problem, certificate)
    assert checked["joint_residual_ffn_correlation_preserved"] is True


def test_exact_dual_replay_strictly_excludes_constant_difference():
    empty = np.empty((0, 2), dtype=np.float64)
    branch = state([0.0, 0.0], empty, [], [])
    skip = state([1.0, -1.0], empty, [], [])
    model = FRONTIER.ExactJointAffine(branch, skip, label="excluded")
    checked = model.exact_dual_lower(np.array([1.0, -1.0]))
    assert checked["strictly_positive"] is True
    assert Fraction(int(checked["exact_numerator"]),
                    int(checked["exact_denominator"])) > 0


def test_exact_callback_detects_corrupted_candidate():
    branch = state([0.0, 0.0], [[-1.0, 1.0]], [0.0], [2.0])
    skip = state([1.0, -1.0], [[0.0, 0.0]], [0.0], [2.0])
    model = FRONTIER.ExactJointAffine(branch, skip, label="mutation")
    with pytest.raises(RuntimeError, match="replay is nonzero"):
        model.exact_replay([Fraction(1, 2)])


def test_transition_inventory_stops_at_final_reduction():
    assert FRONTIER.TRANSITIONS[-1] == (
        "final_residual_reduction",
        "residual_sum_post_numerical_pre_reduction",
        "residual_sum_post_reduction")
    assert all("layernorm" not in transition or
               transition == "post_attention_layernorm_reduction"
               for transition, _before, _after in FRONTIER.TRANSITIONS)
