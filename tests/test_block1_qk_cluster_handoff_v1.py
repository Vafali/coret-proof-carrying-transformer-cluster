from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


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
