from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/run_block1_qk_cluster_v1.py"
SPEC = importlib.util.spec_from_file_location("block1_qk_handoff", SCRIPT)
handoff = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(handoff)


def test_frozen_input_identity_and_counts():
    assert handoff.EXPECTED_INPUT_SHA256 == (
        "a0953d1a9b65a1a8dd810567d161a2dd4a4b5d763c6694f5fbb1cc45f7b0503a")
    assert handoff.EXPECTED_INPUT_GENERATORS == 14_000
    assert handoff.EXPECTED_QK_FRESH == 64


def test_hash_mismatch_fails_before_loading(tmp_path):
    candidate = tmp_path / "wrong.pt"
    candidate.write_bytes(b"not the authenticated QK artifact")
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        handoff.preflight(candidate, handoff.EXPECTED_INPUT_SHA256)


def test_entrypoint_stops_at_qk_and_has_no_property_or_search_calls():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "dispatch.qk(q, k)" in source
    assert "dispatch.softmax" not in source
    assert "attention_value" not in source
    assert "binary_search" not in source
    assert "verify_safety" not in source
    assert "bound(" not in source


def test_rigorous_float32_checker_is_not_misreported_for_fp64():
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"executed": False' in source
    assert "restricted to frozen float32 witnesses" in source
    assert "analytic_fp64_reserve_checked" in source


def test_qk_4d_fresh_rows_use_generator_axis_one():
    rows = torch.arange(
        4 * (handoff.EXPECTED_INPUT_GENERATORS + handoff.EXPECTED_QK_FRESH),
        dtype=torch.float64).reshape(
            4, handoff.EXPECTED_INPUT_GENERATORS + handoff.EXPECTED_QK_FRESH,
            1, 1)
    # This is the previous bug: dimension zero is the four-head axis.
    erroneous = rows[
        handoff.EXPECTED_INPUT_GENERATORS:
        handoff.EXPECTED_INPUT_GENERATORS + handoff.EXPECTED_QK_FRESH]
    assert erroneous.numel() == 0

    selected = handoff._select_native_fresh_rows(
        rows, handoff.EXPECTED_INPUT_GENERATORS + handoff.EXPECTED_QK_FRESH)
    assert selected.shape == (4, handoff.EXPECTED_QK_FRESH, 1, 1)
    assert torch.equal(
        selected, rows[:, handoff.EXPECTED_INPUT_GENERATORS:, :, :])


def test_qk_fresh_row_selector_3d_fallback_and_fail_closed_checks():
    total = handoff.EXPECTED_INPUT_GENERATORS + handoff.EXPECTED_QK_FRESH
    rows = torch.arange(total, dtype=torch.float64).reshape(total, 1, 1)
    selected = handoff._select_native_fresh_rows(rows, total)
    assert selected.shape == (handoff.EXPECTED_QK_FRESH, 1, 1)
    assert torch.equal(selected, rows[handoff.EXPECTED_INPUT_GENERATORS:])
    with pytest.raises(RuntimeError, match="generator count differs"):
        handoff._select_native_fresh_rows(rows, total - 1)
    with pytest.raises(RuntimeError, match="unsupported QK generator-row rank"):
        handoff._select_native_fresh_rows(torch.zeros(total, 1), total)
