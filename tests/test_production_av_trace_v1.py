from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research_hab"))

import coret_production_av_checker_v1 as checker
from coret_trace_witness_v1 import canonical_bytes, seal


@pytest.fixture(scope="module")
def av_trace_root():
    configured = os.environ.get("CORET_AV_TRACE_ROOT")
    if not configured:
        pytest.skip("set CORET_AV_TRACE_ROOT to a bounded produced trace")
    root = Path(configured).resolve()
    assert (root / "av_trace.json").is_file()
    return root


def _copy(source, target):
    destination = target / "trace"
    shutil.copytree(source, destination)
    graph = json.loads((destination / "av_trace.json").read_text())
    return destination, graph


def _write(root, graph):
    seal(graph)
    (root / "av_trace.json").write_bytes(canonical_bytes(graph) + b"\n")


def _replace_blob(root, descriptor, transform):
    path = root / descriptor["relative_path"]
    raw = transform(bytearray(path.read_bytes()))
    digest = hashlib.sha256(raw).hexdigest()
    replacement = path.parent / f"{digest}.float32.le.bin"
    replacement.write_bytes(raw)
    descriptor["sha256"] = digest
    descriptor["relative_path"] = replacement.relative_to(root).as_posix()
    descriptor["byte_count"] = len(raw)
    seal(descriptor)


def _mock_rigorous():
    coordinate = {
        "center_upper": "0.0", "retained_upper": "0.0"}
    return {
        "accepted": True, "family": "A.V",
        "numerical_backend": "checker_only_rigorous_fp64",
        "backend": {
            "coordinate_results": [dict(coordinate) for _ in range(512)],
            "wall_seconds": 0.0,
        },
    }


def _parent_accepts(monkeypatch):
    monkeypatch.setattr(
        checker.softmax, "check_production_softmax",
        lambda _root: {"accepted": True})


def _fast_check(monkeypatch, root):
    _parent_accepts(monkeypatch)
    monkeypatch.setattr(checker, "_check_precise_dot",
                        lambda _witness, _root: _mock_rigorous())
    monkeypatch.setattr(
        checker, "_check_av_sidecar",
        lambda *_args: {"center": 0.0, "retained": 0.0, "fresh": 0.0,
                       "coordinate": 0.0, "local": 0.0})
    return checker.check_production_av(root)


def _sidecar_check(monkeypatch, root):
    _parent_accepts(monkeypatch)
    monkeypatch.setattr(checker, "_check_precise_dot",
                        lambda _witness, _root: _mock_rigorous())
    return checker.check_production_av(root)


def _seal_state_and_graph(root, graph, state):
    raw = root / state["producer_tensor_content_ids"][
        "numerical_radius"]["relative_path"]
    values = (item[0] for item in struct.iter_unpack("<f", raw.read_bytes()))
    state["numerical_sidecar_linkage"]["max_radius_hex"] = max(values).hex()
    seal(state)
    _write(root, graph)


def test_complete_av_trace_accepts(av_trace_root, monkeypatch):
    _parent_accepts(monkeypatch)
    result = checker.check_production_av(av_trace_root)
    assert result["accepted"] is True
    assert all(result[name] is True for name in (
        "PRODUCTION_V_PROJECTION_PASS",
        "PRODUCTION_V_HEAD_MAPPING_PASS",
        "PRODUCTION_AV_NUMERICAL_CHECK_PASS",
        "PRODUCTION_AV_RELATIONAL_STATE_PASS",
        "PRODUCTION_AV_STATE_CONTINUITY_PASS",
        "PRODUCTION_AV_BITWISE_EQUIVALENCE_PASS",
    ))
    assert result["probability_generator_count"] == 1092
    assert result["value_generator_count"] == 900
    assert result["fresh_generator_count"] == 512
    assert result["numerical_native_ratio"] < 1


@pytest.mark.parametrize("mutation", [
    "v_parameter", "v_output", "v_head_map", "softmax_predecessor",
    "dropped_softmax_relation", "generator_reorder", "support",
    "fresh_id_order", "native_radius_one_ulp", "output_blob",
])
def test_av_trace_mutations_reject(
        av_trace_root, tmp_path, monkeypatch, mutation):
    root, graph = _copy(av_trace_root, tmp_path)
    projection, heads, av = graph["transition_records"]
    output = graph["state_records"][2]
    if mutation == "v_parameter":
        projection["operator_witness"]["weight"]["sha256"] = "0" * 64
        seal(projection)
    elif mutation == "v_output":
        descriptor = graph["state_records"][0][
            "producer_tensor_content_ids"]["weights"]
        path = root / descriptor["relative_path"]
        raw = bytearray(path.read_bytes()); raw[0] ^= 1; path.write_bytes(raw)
    elif mutation == "v_head_map":
        heads["tau_k"]["permutation"] = [2, 0, 3, 1]; seal(heads)
    elif mutation == "softmax_predecessor":
        av["predecessor_state_ids"][0] = "substituted_softmax_state"
        seal(av)
    elif mutation == "dropped_softmax_relation":
        av["operator_witness"]["softmax_range_low_sha256"] = "0" * 64
        seal(av)
    elif mutation == "generator_reorder":
        output["generator_ids"][-2:] = reversed(output["generator_ids"][-2:])
        seal(output)
    elif mutation == "support":
        output["generator_support_masks"][0] ^= 1; seal(output)
    elif mutation == "fresh_id_order":
        ids = av["tau_k"]["fresh_generator_ids"]
        ids[-2:] = reversed(ids[-2:]); seal(av)
    elif mutation == "native_radius_one_ulp":
        descriptor = output["producer_tensor_content_ids"]["weights"]
        row = 1093
        offset = ((0 * 1605 + row) * 4 + 0) * 32 + 0
        def narrow(raw):
            bits = struct.unpack_from("<I", raw, offset * 4)[0]
            assert 0 < bits < 0x7f800000
            struct.pack_into("<I", raw, offset * 4, bits - 1)
            return bytes(raw)
        _replace_blob(root, descriptor, narrow); seal(output)
    else:
        descriptor = output["producer_tensor_content_ids"]["weights"]
        path = root / descriptor["relative_path"]
        raw = bytearray(path.read_bytes()); raw[-1] ^= 1; path.write_bytes(raw)
    _write(root, graph)
    with pytest.raises((AssertionError, KeyError)):
        _fast_check(monkeypatch, root)


@pytest.mark.parametrize("mutation,row", [
    ("dropped_bilinear", 1093),
    ("narrowed_retained", 1),
    ("narrowed_fresh", 1093),
])
def test_av_sidecar_narrowing_rejects(
        av_trace_root, tmp_path, monkeypatch, mutation, row):
    root, graph = _copy(av_trace_root, tmp_path)
    state = graph["state_records"][2]
    descriptor = state["producer_tensor_content_ids"]["numerical_radius"]
    offset = ((0 * 1605 + row) * 4 + 0) * 32 + 0
    def narrow(raw):
        value = struct.unpack_from("<f", raw, offset * 4)[0]
        assert value > 0
        replacement = 0.0 if mutation == "dropped_bilinear" else value * 0.5
        struct.pack_into("<f", raw, offset * 4, replacement)
        return bytes(raw)
    _replace_blob(root, descriptor, narrow)
    _seal_state_and_graph(root, graph, state)
    with pytest.raises(AssertionError, match="sidecar too narrow"):
        _sidecar_check(monkeypatch, root)
