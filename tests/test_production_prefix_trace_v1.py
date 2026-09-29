from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research_hab"))

import coret_production_prefix_checker_v1 as checker
import coret_production_prefix_trace_v1 as producer
from coret_trace_witness_v1 import canonical_bytes, seal


@pytest.fixture(scope="module")
def trace_fixture(tmp_path_factory):
    repository = REPO / "research_hab/public_benchmarks/DeepT"
    code = tempfile.TemporaryDirectory(prefix="coret_prefix_native_")
    packed = subprocess.check_output([
        "git", "-C", str(repository), "archive", "--format=tar",
        producer.PINNED_REVISION,
        "Robustness-Verification-for-Transformers/Verifiers",
    ])
    with tarfile.open(fileobj=io.BytesIO(packed), mode="r:") as archive:
        archive.extractall(code.name, filter="data")
    verifier_root = (Path(code.name)
                     / "Robustness-Verification-for-Transformers/Verifiers")
    prior = sys.modules.get("Verifiers")
    package = types.ModuleType("Verifiers")
    package.__path__ = [str(verifier_root)]
    sys.modules["Verifiers"] = package
    from Verifiers.Zonotope import Zonotope

    args = SimpleNamespace(
        perturbed_words=1, attack_type="lp", device=torch.device("cpu"),
        cpu=True, all_words=False, num_input_error_terms=128,
        use_dot_product_variant3=False, use_other_dot_product_ordering=False,
        concretize_special_norm_error_together=False)
    root = tmp_path_factory.mktemp("production_prefix")
    states, graph = producer.build_production_prefix_trace(
        root, Zonotope, args)
    yield root, states, graph
    for name in tuple(sys.modules):
        if name == "Verifiers" or name.startswith("Verifiers."):
            del sys.modules[name]
    if prior is not None:
        sys.modules["Verifiers"] = prior
    code.cleanup()


def _copy(trace_fixture, tmp_path):
    root, _states, graph = trace_fixture
    destination = tmp_path / "trace"
    shutil.copytree(root, destination)
    return destination, copy.deepcopy(graph)


def _write(root, graph):
    seal(graph)
    (root / "trace.json").write_bytes(canonical_bytes(graph) + b"\n")


def _replace_blob(root, descriptor, transform):
    path = root / descriptor["relative_path"]
    raw = transform(bytearray(path.read_bytes()))
    digest = hashlib.sha256(raw).hexdigest()
    replacement = root / "blobs" / f"{digest}.float32.le.bin"
    replacement.write_bytes(raw)
    descriptor["sha256"] = digest
    descriptor["relative_path"] = replacement.relative_to(root).as_posix()
    descriptor["byte_count"] = len(raw)
    seal(descriptor)


def test_real_prefix_valid_and_bitwise_transparent(trace_fixture):
    root, states, graph = trace_fixture
    result = checker.check_production_prefix(root)
    assert all(result[name] is True for name in (
        "PRODUCTION_SOURCE_DOMAIN_PASS",
        "PRODUCTION_AFFINE_RESIDUAL_PREFIX_PASS",
        "FIRST_PRODUCTION_LAYERNORM_TRACE_PASS",
        "PRODUCTION_PREFIX_STATE_CONTINUITY_PASS",
        "PRODUCTION_PREFIX_BITWISE_EQUIVALENCE_PASS"))
    assert states[0].zonotope_w.shape == (129, 4, 128)
    assert states[1].zonotope_w.shape == (901, 4, 128)
    assert graph["transition_records"][0]["tau_k"][
        "sqrt_active_flat_indices"] == list(range(128, 256))
    assert graph["transition_records"][0]["tau_k"][
        "reciprocal_active_flat_indices"] == list(range(128, 256))


@pytest.mark.parametrize(
    "mutation", ["radius", "mask", "source_permutation", "range_narrowing"])
def test_source_domain_mutations_reject(trace_fixture, tmp_path, mutation):
    root, graph = _copy(trace_fixture, tmp_path)
    domain = graph["source_domain"]
    if mutation == "radius":
        domain["epsilon_hex"] = float(1.0 / 1599.0).hex()
    elif mutation == "mask":
        domain["input_source_mask"][0] = 0
    elif mutation == "source_permutation":
        domain["source_symbol_ids"][:2] = reversed(
            domain["source_symbol_ids"][:2])
    else:
        graph["state_records"][0]["explicit_ranges"][0][0] = (-0.5).hex()
        seal(graph["state_records"][0])
    seal(domain); _write(root, graph)
    with pytest.raises(AssertionError):
        checker.check_production_prefix(root)


def test_embedding_affine_and_predecessor_mutations_reject(
        trace_fixture, tmp_path):
    root, graph = _copy(trace_fixture, tmp_path / "affine")
    descriptor = graph["source_domain"]["embedding_components"]["word"]
    def change(raw):
        bits = struct.unpack_from("<I", raw, 0)[0]
        struct.pack_into("<I", raw, 0, bits + 1)
        return bytes(raw)
    _replace_blob(root, descriptor, change)
    seal(graph["source_domain"]); _write(root, graph)
    with pytest.raises(AssertionError):
        checker.check_production_prefix(root)

    root, graph = _copy(trace_fixture, tmp_path / "predecessor")
    graph["transition_records"][0]["predecessor_state_id"] = "substituted"
    seal(graph["transition_records"][0]); _write(root, graph)
    with pytest.raises(AssertionError):
        checker.check_production_prefix(root)


def test_layernorm_mean_variance_and_sqrt_allocation_mutations_reject(
        trace_fixture, tmp_path):
    root, graph = _copy(trace_fixture, tmp_path / "mean")
    graph["transition_records"][0]["tau_k"]["mean_divisor"] = 127
    seal(graph["transition_records"][0]); _write(root, graph)
    with pytest.raises(AssertionError):
        checker.check_production_prefix(root)

    root, graph = _copy(trace_fixture, tmp_path / "variance")
    graph["transition_records"][0]["tau_k"]["variance_divisor"] = 127
    seal(graph["transition_records"][0]); _write(root, graph)
    with pytest.raises(AssertionError):
        checker.check_production_prefix(root)

    root, graph = _copy(trace_fixture, tmp_path / "sqrt")
    graph["transition_records"][0]["tau_k"][
        "sqrt_active_flat_indices"].pop()
    seal(graph["transition_records"][0]); _write(root, graph)
    with pytest.raises(AssertionError, match="incorrect sqrt allocation"):
        checker.check_production_prefix(root)


def test_reciprocal_allocation_and_generator_order_mutations_reject(
        trace_fixture, tmp_path):
    root, graph = _copy(trace_fixture, tmp_path / "reciprocal")
    graph["transition_records"][0]["tau_k"][
        "reciprocal_active_flat_indices"].pop()
    seal(graph["transition_records"][0]); _write(root, graph)
    with pytest.raises(AssertionError, match="incorrect reciprocal allocation"):
        checker.check_production_prefix(root)

    root, graph = _copy(trace_fixture, tmp_path / "order")
    output = graph["state_records"][1]
    output["generator_ids"][-2:] = reversed(output["generator_ids"][-2:])
    mapping = output["ghost_state_linkage"]["ordered_native_to_ghost"]
    mapping[-2]["ghost_id"], mapping[-1]["ghost_id"] = (
        mapping[-1]["ghost_id"], mapping[-2]["ghost_id"])
    seal(output); _write(root, graph)
    with pytest.raises(AssertionError, match="generator identity/order"):
        checker.check_production_prefix(root)


def test_narrowed_sidecar_rejects(trace_fixture, tmp_path):
    root, graph = _copy(trace_fixture, tmp_path)
    output = graph["state_records"][1]
    descriptor = output["producer_tensor_content_ids"]["numerical_radius"]
    _replace_blob(root, descriptor, lambda raw: bytes(len(raw)))
    seal(output); _write(root, graph)
    with pytest.raises(AssertionError, match="numerical sidecar too narrow"):
        checker.check_production_prefix(root)


def test_dropped_persistent_state_and_output_bit_mutations_reject(
        trace_fixture, tmp_path):
    root, graph = _copy(trace_fixture, tmp_path / "ghost")
    graph["state_records"][1]["ghost_state_linkage"][
        "ordered_native_to_ghost"] = []
    seal(graph["state_records"][1]); _write(root, graph)
    with pytest.raises(AssertionError, match="mapping/order"):
        checker.check_production_prefix(root)

    root, graph = _copy(trace_fixture, tmp_path / "numerical")
    del graph["state_records"][1]["numerical_sidecar_linkage"]
    seal(graph["state_records"][1]); _write(root, graph)
    with pytest.raises((AssertionError, KeyError)):
        checker.check_production_prefix(root)

    root, graph = _copy(trace_fixture, tmp_path / "blob")
    descriptor = graph["state_records"][1][
        "producer_tensor_content_ids"]["weights"]
    path = root / descriptor["relative_path"]
    raw = bytearray(path.read_bytes()); raw[0] ^= 1; path.write_bytes(raw)
    with pytest.raises(AssertionError, match="blob identity"):
        checker.check_production_prefix(root)
