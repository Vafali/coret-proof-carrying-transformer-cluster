from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import struct
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research_hab"))

import coret_production_softmax_checker_v1 as checker
from coret_trace_witness_v1 import canonical_bytes, seal


@pytest.fixture(scope="module")
def softmax_trace_root():
    configured = os.environ.get("CORET_SOFTMAX_TRACE_ROOT")
    if not configured:
        pytest.skip("set CORET_SOFTMAX_TRACE_ROOT to a bounded produced trace")
    root = Path(configured).resolve()
    assert (root / "softmax_trace.json").is_file()
    return root


def _copy(source, target):
    destination = target / "trace"
    shutil.copytree(source, destination)
    graph = json.loads((destination / "softmax_trace.json").read_text())
    return destination, graph


def _write(root, graph):
    seal(graph)
    (root / "softmax_trace.json").write_bytes(canonical_bytes(graph) + b"\n")


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


def _fast_check(monkeypatch, root):
    monkeypatch.setattr(
        checker.prefix, "check_production_prefix",
        lambda _root: {"PRODUCTION_QK_NUMERICAL_CHECK_PASS": True})
    return checker.check_production_softmax(root)


def test_complete_softmax_trace_accepts(softmax_trace_root, monkeypatch):
    result = _fast_check(monkeypatch, softmax_trace_root)
    assert result["accepted"] is True
    assert all(result[name] is True for name in (
        "PRODUCTION_SCORE_SCALING_PASS",
        "PRODUCTION_SOFTMAX_PIVOT_PASS",
        "PRODUCTION_SOFTMAX_EXP_PASS",
        "PRODUCTION_SOFTMAX_SUM_RECIPROCAL_PASS",
        "PRODUCTION_SOFTMAX_EQUALITY_CONSTRAINTS_PASS",
        "PRODUCTION_SOFTMAX_RANGED_SYMBOL_PASS",
        "PRODUCTION_SOFTMAX_STATE_CONTINUITY_PASS",
        "PRODUCTION_SOFTMAX_BITWISE_EQUIVALENCE_PASS",
    ))
    assert result["input_generator_count"] == 964
    assert result["output_generator_count"] == 1092
    assert result["numerical_native_ratio"] < 1


@pytest.mark.parametrize("mutation", [
    "score_scale", "pivot", "equality_substitution", "exp_domain",
    "exp_relaxation",
    "exp_fresh_order", "exp_fresh_id_order", "denominator_sum",
    "reciprocal_domain", "reciprocal_allocation",
    "dropped_numerical", "dropped_range", "generator_reorder",
    "narrowed_sidecar", "output_blob",
])
def test_softmax_mutations_reject(
        softmax_trace_root, tmp_path, monkeypatch, mutation):
    root, graph = _copy(softmax_trace_root, tmp_path)
    transition = graph["transition_records"][1]
    witness = transition["operator_witness"]
    if mutation == "score_scale":
        graph["transition_records"][0]["tau_k"][
            "scale_float32_hex"] = float(0.25).hex()
        seal(graph["transition_records"][0])
    elif mutation == "pivot":
        witness["equality"]["initial_pivot_indices"][0] -= 1
        seal(transition)
    elif mutation == "equality_substitution":
        descriptor = witness["equality"]["optimal_values"]
        def change(raw):
            bits = struct.unpack_from("<I", raw, 0)[0]
            struct.pack_into("<I", raw, 0, bits + 1)
            return bytes(raw)
        _replace_blob(root, descriptor, change); seal(transition)
    elif mutation == "exp_domain":
        descriptor = witness["exp_heads"][0]["lower"]
        def narrow(raw):
            index = witness["exp_heads"][0]["active_flat_indices"][0]
            value = struct.unpack_from("<f", raw, index * 4)[0]
            struct.pack_into("<f", raw, index * 4, value + 0.1)
            return bytes(raw)
        _replace_blob(root, descriptor, narrow); seal(transition)
    elif mutation == "exp_relaxation":
        descriptor = witness["exp_heads"][0]["slope"]
        def alter_exp_slope(raw):
            index = witness["exp_heads"][0]["active_flat_indices"][0]
            value = struct.unpack_from("<f", raw, index * 4)[0]
            struct.pack_into("<f", raw, index * 4, value * 1.01)
            return bytes(raw)
        _replace_blob(root, descriptor, alter_exp_slope); seal(transition)
    elif mutation == "exp_fresh_order":
        active = witness["exp_heads"][0]["active_flat_indices"]
        active[:2] = reversed(active[:2]); seal(transition)
    elif mutation == "exp_fresh_id_order":
        ids = transition["tau_k"]["exp_generator_ids"]
        ids[:2] = reversed(ids[:2]); seal(transition)
    elif mutation == "denominator_sum":
        descriptor = graph["state_records"][1][
            "producer_tensor_content_ids"]["weights"]
        path = root / descriptor["relative_path"]
        raw = bytearray(path.read_bytes()); raw[0] ^= 1; path.write_bytes(raw)
    elif mutation == "reciprocal_domain":
        descriptor = witness["reciprocal"]["lower"]
        def change_reciprocal_domain(raw):
            struct.pack_into("<f", raw, 0, -1.0)
            return bytes(raw)
        _replace_blob(root, descriptor, change_reciprocal_domain)
        seal(transition)
    elif mutation == "reciprocal_allocation":
        witness["reciprocal"]["active_flat_indices"].pop(); seal(transition)
    elif mutation == "dropped_numerical":
        del graph["state_records"][2]["numerical_sidecar_linkage"]
        seal(graph["state_records"][2])
    elif mutation == "dropped_range":
        state = graph["state_records"][3]
        state["native_range_metadata"] = {
            "kind": "absent", "low": None, "high": None}
        seal(state)
    elif mutation == "generator_reorder":
        state = graph["state_records"][3]
        state["generator_ids"][-2:] = reversed(state["generator_ids"][-2:])
        mapping = state["ghost_state_linkage"]["ordered_native_to_ghost"]
        mapping[-2]["ghost_id"], mapping[-1]["ghost_id"] = (
            mapping[-1]["ghost_id"], mapping[-2]["ghost_id"])
        seal(state)
    elif mutation == "narrowed_sidecar":
        state = graph["state_records"][3]
        descriptor = state["producer_tensor_content_ids"]["numerical_radius"]
        def narrow_eta(raw):
            values = [item[0] for item in struct.iter_unpack("<f", raw)]
            index = max(range(len(values)), key=values.__getitem__)
            values[index] *= 0.25
            return b"".join(struct.pack("<f", value) for value in values)
        _replace_blob(root, descriptor, narrow_eta)
        raw = (root / descriptor["relative_path"]).read_bytes()
        state["numerical_sidecar_linkage"]["max_radius_hex"] = max(
            item[0] for item in struct.iter_unpack("<f", raw)).hex()
        seal(state)
    else:
        descriptor = graph["state_records"][3][
            "producer_tensor_content_ids"]["weights"]
        path = root / descriptor["relative_path"]
        raw = bytearray(path.read_bytes()); raw[-1] ^= 1; path.write_bytes(raw)
    _write(root, graph)
    with pytest.raises((AssertionError, KeyError)):
        _fast_check(monkeypatch, root)
