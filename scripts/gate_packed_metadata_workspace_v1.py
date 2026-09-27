#!/usr/bin/env python3
"""Bounded real-state parity/timing gate for packed integer metadata."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import types

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "research_hab"), str(REPO / "tests")]
import coret_structural_support_precise_dot_v1 as structural
import coret_bounded_native_execution_v1 as bounded
import coret_native_semantics_checker_v1 as checker
import test_packed_metadata_workspace as reference

QK_ROOT = Path(
    "/mnt/c/users/david-despacho/documents/vafali projects/lookahead-branching/"
    "research_hab/results/coret_multisource_native_b0_capture_v1_20260920")
AV_PATH = Path(
    "/mnt/c/users/david-despacho/documents/vafali projects/lookahead-branching/"
    "research_hab/results/coret_deept_av_removable_slack_oracle_v1_20260920/"
    "coret_deept_B0_AV_operands_v1.pt")


class Proxy:
    def __init__(self, weights, source=None):
        self.zonotope_w = weights
        self.error_term_range_low = None if source is None else source.error_term_range_low
        self.error_term_range_high = None if source is None else source.error_term_range_high
        self.num_input_error_terms_special_norm = 128
        self.p = 100
        self.eps = 1.0 / 1600.0
        self.perturbed_word_index = 10
    @property
    def num_error_terms(self): return int(self.zonotope_w.shape[1] - 1)
    @property
    def num_words(self): return int(self.zonotope_w.shape[2])
    @property
    def word_embedding_size(self): return int(self.zonotope_w.shape[3])


def install_factory():
    package = types.ModuleType("Verifiers")
    module = types.ModuleType("Verifiers.Zonotope")
    module.make_zonotope_new_weights_same_args = (
        lambda new_weights, source_zonotope, clone=False:
        Proxy(new_weights.clone() if clone else new_weights, source_zonotope))
    package.Zonotope = module
    sys.modules["Verifiers"] = package
    sys.modules["Verifiers.Zonotope"] = module


def hidden_masks(tokens=20, perturbed=10):
    local = structural.local_mask
    return tuple(
        [local(perturbed)] * 128
        + [local(t) if t == perturbed else 0 for t in range(tokens)]
        + [local(perturbed)] * 128
        + [local(perturbed)] * 128
        + [local(t) if t == perturbed else 0
           for t in range(tokens) for _ in range(128)])


def probability_masks(hidden, tokens=20):
    local, dense = structural.local_mask, structural.dense_mask
    return tuple(
        [dense(tokens) if mask else 0 for mask in hidden]
        + [local(q) for _h in range(4) for q in range(tokens) for _k in range(tokens)]
        + [local(q) for _h in range(4) for q in range(tokens) for _k in range(tokens)]
        + [local(q) for q in range(tokens) for _h in range(4) for _k in range(tokens)])


def comparable(diagnostics):
    return {key: value for key, value in diagnostics.items()
            if key != "stage_timing_seconds"}


def state_hash(tensor):
    digest = hashlib.sha256()
    for chunk in bounded._tensor_chunks(tensor):
        digest.update(memoryview(chunk.numpy()).cast("B"))
    return digest.hexdigest()


def timed_full(left, right, lp, rp, mode, legacy):
    name = "_ordered_qk_radius" if mode == "QK" else "_av_radius"
    original = getattr(structural, name)
    inventory = []
    if mode == "QK":
        def replacement(a, b, x, y, g):
            result = legacy(a, b, x, y, g)
            inventory.extend(result[3])
            return result[:3]
        kwargs = {"mode": mode}
    else:
        def replacement(a, b, x, y, ga, gb, tile):
            result = legacy(a, b, x, y, ga, gb, tile)
            inventory.extend(result[4])
            return result[:4]
        kwargs = {"mode": mode, "generator_tile": 112}
    diagnostics = {}
    setattr(structural, name, replacement)
    torch.cuda.synchronize(); started = time.perf_counter()
    try:
        output = structural.precise_dot_structural(
            left, right, lp, rp, diagnostics=diagnostics, **kwargs)
        torch.cuda.synchronize()
    finally:
        setattr(structural, name, original)
    return output, diagnostics, time.perf_counter() - started, inventory


def timed_packed(left, right, lp, rp, mode):
    diagnostics = {}
    kwargs = {"mode": mode}
    if mode == "A.V": kwargs["generator_tile"] = 112
    recorded = []
    original_builder = structural._IndexArenaBuilder
    class RecordingBuilder(original_builder):
        def add(self, values):
            values = tuple(values); recorded.append(values)
            return super().add(values)
    structural._IndexArenaBuilder = RecordingBuilder
    torch.cuda.synchronize(); started = time.perf_counter()
    try:
        output = structural.precise_dot_structural(
            left, right, lp, rp, diagnostics=diagnostics, **kwargs)
        torch.cuda.synchronize()
    finally:
        structural._IndexArenaBuilder = original_builder
    return output, diagnostics, time.perf_counter() - started, recorded


def check_family(family, old, new, old_diag, new_diag, old_seconds, new_seconds,
                 gmax, old_inventory, packed_sequences):
    equal = torch.equal(old.zonotope_w, new.zonotope_w)
    center_equal = torch.equal(old.zonotope_w[:, :1], new.zonotope_w[:, :1])
    retained_equal = torch.equal(
        old.zonotope_w[:, 1:1+gmax], new.zonotope_w[:, 1:1+gmax])
    fresh_equal = torch.equal(
        old.zonotope_w[:, 1+gmax:], new.zonotope_w[:, 1+gmax:])
    proof_equal = structural.get_support(old) == structural.get_support(new)
    diagnostics_equal = comparable(old_diag) == comparable(new_diag)
    if family == "QK":
        expected_sequences = [task[2] for task in old_inventory]
    else:
        expected_sequences = []
        for task in old_inventory:
            expected_sequences.extend(task[:4])
    index_sequences_equal = packed_sequences == expected_sequences
    task_order_equal = len(old_inventory) == new_diag["grouped_launches"]
    certificate = {
        "revision": checker.PINNED, "family": family,
        "native_result_unchanged": equal,
        "generic_semantic_remainder_used": False,
        "output": bounded.state_record_blockwise(new),
    }
    checker_pass = checker.check_common(certificate, family)
    if not all((equal, center_equal, retained_equal, fresh_equal,
                proof_equal, diagnostics_equal, index_sequences_equal,
                task_order_equal, checker_pass)):
        raise RuntimeError(f"{family} packed metadata parity failed")
    return {
        "old_seconds": old_seconds, "packed_seconds": new_seconds,
        "speedup": old_seconds / new_seconds,
        "whole_output_bitwise_equal": equal,
        "center_bitwise_equal": center_equal,
        "retained_bitwise_equal": retained_equal,
        "fresh_radius_bitwise_equal": fresh_equal,
        "support_provenance_equal": proof_equal,
        "certificate_diagnostics_equal_excluding_timing": diagnostics_equal,
        "every_index_sequence_equal": index_sequences_equal,
        "task_order_and_count_equal": task_order_equal,
        "checker_pass": checker_pass,
        "old_sha256": state_hash(old.zonotope_w),
        "packed_sha256": state_hash(new.zonotope_w),
        "grouped_launches": new_diag["grouped_launches"],
        "executed_quadratic_MACs": new_diag["executed_quadratic_MACs"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    torch.use_deterministic_algorithms(True)
    install_factory(); device = torch.device("cuda:0")
    qrec = torch.load(QK_ROOT / "coret_native_b0_q_operand_v1.pt",
                      map_location="cpu", weights_only=False)
    krec = torch.load(QK_ROOT / "coret_native_b0_k_operand_v1.pt",
                      map_location="cpu", weights_only=False)
    avrec = torch.load(AV_PATH, map_location="cpu", weights_only=False)
    q = Proxy(qrec["coefficient_tensor"].to(device))
    k = Proxy(krec["coefficient_tensor"].to(device))
    a = Proxy(avrec["A"]["weights"].to(device))
    v = Proxy(avrec["V_transposed"]["weights"].to(device))
    hidden = hidden_masks(); probability = probability_masks(hidden)
    hp = structural.proof_from_masks(hidden, 20, "depth6_b0_hidden")
    pp = structural.proof_from_masks(probability, 20, "depth6_b0_probability")
    with torch.no_grad():
        # Warm kernels and allocator using the packed implementation.
        warmq, _, _, _ = timed_packed(q, k, hp, hp, "QK")
        warma, _, _, _ = timed_packed(a, v, pp, hp, "A.V")
        del warmq, warma; gc.collect(); torch.cuda.synchronize()
        oldq, oldqd, oldqt, oldqi = timed_full(
            q, k, hp, hp, "QK", reference.legacy_qk)
        newq, newqd, newqt, newqi = timed_packed(q, k, hp, hp, "QK")
        olda, oldad, oldat, oldai = timed_full(
            a, v, pp, hp, "A.V", reference.legacy_av)
        newa, newad, newat, newai = timed_packed(a, v, pp, hp, "A.V")
        qresult = check_family("QK", oldq, newq, oldqd, newqd,
                               oldqt, newqt, q.num_error_terms, oldqi, newqi)
        avresult = check_family("A.V", olda, newa, oldad, newad,
                                oldat, newat,
                                max(a.num_error_terms, v.num_error_terms),
                                oldai, newai)
    result = {
        "schema": "CORET_PACKED_METADATA_WORKSPACE_GATE_V1",
        "representative_property": "deept_table7_stdln6_s000_line504_tok10",
        "qk": qresult, "av": avresult,
        "complete_properties_executed": 0,
        "scientific_queries_executed": 0,
        "bound_calls_executed": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"qk_bitwise": qresult["whole_output_bitwise_equal"],
                      "av_bitwise": avresult["whole_output_bitwise_equal"],
                      "qk_speedup": qresult["speedup"],
                      "av_speedup": avresult["speedup"],
                      "complete_properties_executed": 0,
                      "scientific_queries_executed": 0,
                      "bound_calls_executed": 0}, sort_keys=True))


if __name__ == "__main__": main()
