#!/usr/bin/env python3
"""Bounded allocation/workspace audit for structural B0 QK and A.V.

No verifier, radius-search, or bound entrypoint is imported or invoked.  The
immutable real B0 operands have the same 4-head/20-token topology as the frozen
depth-6 s000 property; only the two precise-dot operators are exercised.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import types
import statistics

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research_hab"))
import coret_structural_support_precise_dot_v1 as structural

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


def install_proxy_factory():
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


def memory_state(device):
    stats = torch.cuda.memory_stats(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    return {
        "allocated": allocated,
        "reserved": reserved,
        "reserved_minus_allocated": reserved - allocated,
        "inactive_split": stats.get("inactive_split_bytes.all.current", 0),
        "allocation_current": stats.get("allocation.all.current", 0),
        "segment_current": stats.get("segment.all.current", 0),
        "num_alloc_retries": stats.get("num_alloc_retries", 0),
        "num_ooms": stats.get("num_ooms", 0),
    }


ALLOC_NAMES = {
    "aten::empty", "aten::empty_like", "aten::zeros", "aten::zeros_like",
    "aten::clone", "aten::new_empty", "aten::new_zeros",
}
INIT_NAMES = ALLOC_NAMES | {"aten::zero_", "aten::fill_", "aten::copy_"}
H2D_NAMES = {"aten::to", "aten::_to_copy", "aten::copy_", "cudaMemcpyAsync"}


def profile_one(label, call, device):
    torch.cuda.synchronize(device)
    before = memory_state(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        profile_memory=True, record_shapes=True,
    ) as prof:
        with torch.profiler.record_function(label):
            output = call()
        torch.cuda.synchronize(device)
    wall = time.perf_counter() - started
    after = memory_state(device)
    events = list(prof.events())
    positive = [e for e in events if getattr(e, "cuda_memory_usage", 0) > 0]
    sizes = Counter(int(e.cuda_memory_usage) for e in positive)
    ops = []
    for row in prof.key_averages():
        if (row.key in INIT_NAMES or row.key in H2D_NAMES
                or "cudaMalloc" in row.key or "cudaFree" in row.key
                or "Memset" in row.key):
            ops.append({
                "name": row.key, "calls": int(row.count),
                "self_cpu_seconds": row.self_cpu_time_total / 1e6,
                "self_cuda_seconds": getattr(
                    row, "self_device_time_total",
                    getattr(row, "self_cuda_time_total", 0.0)) / 1e6,
                "cuda_memory_usage": int(row.cuda_memory_usage),
                "self_cuda_memory_usage": int(row.self_cuda_memory_usage),
            })
    allocation_cpu = sum(x["self_cpu_seconds"] for x in ops if x["name"] in ALLOC_NAMES)
    initialization_cuda = sum(x["self_cuda_seconds"] for x in ops if x["name"] in INIT_NAMES)
    h2d_cpu = sum(x["self_cpu_seconds"] for x in ops if x["name"] in H2D_NAMES)
    return output, {
        "label": label, "wall_seconds_including_profiler": wall,
        "before": before, "after": after,
        "peak_allocated": torch.cuda.max_memory_allocated(device),
        "peak_reserved": torch.cuda.max_memory_reserved(device),
        "positive_cuda_memory_event_count": len(positive),
        "positive_cuda_memory_event_bytes": sum(int(e.cuda_memory_usage) for e in positive),
        "repeated_positive_allocation_sizes_top10": [
            {"bytes": size, "count": count} for size, count in sizes.most_common(10)],
        "selected_profiler_ops": ops,
        "allocation_dispatch_cpu_seconds": allocation_cpu,
        "initialization_cuda_seconds": initialization_cuda,
        "h2d_dispatch_cpu_seconds": h2d_cpu,
        "allocation_plus_initialization_fraction_of_wall": (
            (allocation_cpu + initialization_cuda) / wall if wall else 0.0),
        "allocator_retry_delta": after["num_alloc_retries"] - before["num_alloc_retries"],
        "oom_delta": after["num_ooms"] - before["num_ooms"],
    }


def metadata_workspace_microbenchmark(device, tasks=17476):
    """No arithmetic: current versus reusable/packed index storage only."""
    cpu = [torch.arange(112, dtype=torch.long),
           torch.arange(112, dtype=torch.long),
           torch.tensor([3], dtype=torch.long),
           torch.tensor([7], dtype=torch.long)]
    torch.cuda.synchronize(device); started = time.perf_counter()
    for _ in range(tasks):
        made = [torch.tensor(x.tolist(), dtype=torch.long, device=device)
                for x in cpu]
        del made
    torch.cuda.synchronize(device)
    current = time.perf_counter() - started
    buffers = [torch.empty_like(x, device=device) for x in cpu]
    torch.cuda.synchronize(device); started = time.perf_counter()
    for _ in range(tasks):
        for buffer, source in zip(buffers, cpu):
            buffer.copy_(source, non_blocking=False)
    torch.cuda.synchronize(device)
    persistent = time.perf_counter() - started
    packed_samples = []
    packed_bytes = 0
    for _ in range(3):
        torch.cuda.synchronize(device); started = time.perf_counter()
        flat = torch.cat(cpu * tasks)
        gpu_flat = flat.to(device)
        torch.cuda.synchronize(device)
        packed_samples.append(time.perf_counter() - started)
        packed_bytes = flat.numel() * flat.element_size()
        del flat, gpu_flat
    del buffers
    return {
        "task_count": tasks, "metadata_tensor_count": tasks * 4,
        "current_individual_tensor_seconds": current,
        "persistent_four_buffers_seconds": persistent,
        "packed_build_and_single_H2D_seconds_samples": packed_samples,
        "packed_build_and_single_H2D_seconds_median": statistics.median(packed_samples),
        "packed_bytes": packed_bytes,
    }


def cap_allocation_microbenchmark(device, reps=50):
    elements = 128450560 // 4
    warm = torch.empty(elements, device=device); del warm
    torch.cuda.synchronize(device)
    values = {}
    for name, factory in (
            ("empty", lambda: torch.empty(elements, device=device)),
            ("zeros", lambda: torch.zeros(elements, device=device))):
        started = time.perf_counter()
        for _ in range(reps):
            tensor = factory(); del tensor
        torch.cuda.synchronize(device)
        values[name + "_seconds_per_call"] = (
            time.perf_counter() - started) / reps
    tensor = torch.empty(elements, device=device)
    torch.cuda.synchronize(device); started = time.perf_counter()
    for _ in range(reps):
        tensor.zero_()
    torch.cuda.synchronize(device)
    values["persistent_zero_seconds_per_call"] = (
        time.perf_counter() - started) / reps
    del tensor
    values.update({"bytes": 128450560, "repetitions": reps})
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    install_proxy_factory()
    qrec = torch.load(QK_ROOT / "coret_native_b0_q_operand_v1.pt",
                      map_location="cpu", weights_only=False)
    krec = torch.load(QK_ROOT / "coret_native_b0_k_operand_v1.pt",
                      map_location="cpu", weights_only=False)
    avrec = torch.load(AV_PATH, map_location="cpu", weights_only=False)
    q = Proxy(qrec["coefficient_tensor"].to(device))
    k = Proxy(krec["coefficient_tensor"].to(device))
    a = Proxy(avrec["A"]["weights"].to(device))
    v = Proxy(avrec["V_transposed"]["weights"].to(device))
    hidden = hidden_masks()
    probability = probability_masks(hidden)
    hp = structural.proof_from_masks(hidden, 20, "depth6_b0_hidden")
    pp = structural.proof_from_masks(probability, 20, "depth6_b0_probability")
    diagnostics = {}
    with torch.no_grad():
        # Warm allocator and kernels, then release outputs while retaining cache.
        oq = structural.precise_dot_structural(q, k, hp, hp, mode="QK")
        oa = structural.precise_dot_structural(
            a, v, pp, hp, mode="A.V", generator_tile=112)
        torch.cuda.synchronize(device)
        del oq, oa
        gc.collect(); torch.cuda.synchronize(device)
        steady_before = memory_state(device)
        qout, qstats = profile_one(
            "bounded_depth6_B0_QK",
            lambda: structural.precise_dot_structural(
                q, k, hp, hp, mode="QK", diagnostics=diagnostics), device)
        qdiag = dict(diagnostics); diagnostics.clear()
        del qout; gc.collect(); torch.cuda.synchronize(device)
        avout, avstats = profile_one(
            "bounded_depth6_B0_AV",
            lambda: structural.precise_dot_structural(
                a, v, pp, hp, mode="A.V", generator_tile=112,
                diagnostics=diagnostics), device)
        avdiag = dict(diagnostics)
        del avout; gc.collect(); torch.cuda.synchronize(device)
        metadata_microbenchmark = metadata_workspace_microbenchmark(device)
        cap_microbenchmark = cap_allocation_microbenchmark(device)
        final = memory_state(device)
    result = {
        "schema": "CORET_DEPTH6_WORKSPACE_AUDIT_V1",
        "representative_property": "deept_table7_stdln6_s000_line504_tok10",
        "boundary": "matched-topology immutable-real B0 QK and A.V only",
        "qk_shape": list(q.zonotope_w.shape),
        "av_shapes": [list(a.zonotope_w.shape), list(v.zonotope_w.shape)],
        "steady_before": steady_before, "qk": qstats, "av": avstats,
        "qk_diagnostics": qdiag, "av_diagnostics": avdiag,
        "metadata_workspace_microbenchmark": metadata_microbenchmark,
        "cap_allocation_microbenchmark": cap_microbenchmark,
        "final": final,
        "complete_properties_executed": 0,
        "scientific_queries_executed": 0,
        "bound_calls_executed": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "qk_wall_seconds": qstats["wall_seconds_including_profiler"],
        "av_wall_seconds": avstats["wall_seconds_including_profiler"],
        "scientific_queries_executed": 0,
        "bound_calls_executed": 0,
        "complete_properties_executed": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
