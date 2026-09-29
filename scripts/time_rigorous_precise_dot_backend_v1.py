#!/usr/bin/env python3
"""Manual cached-state correctness/timing gate for rigorous precise dot.

This is not a verifier query.  It loads immutable cached operand states,
constructs the current producer output only as the object being checked, and
passes immutable IEEE bytes to the independent driver/NVRTC backend.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research_hab"
sys.path.insert(0, str(RESEARCH))

import coret_deept_exact_standard_ln_adapter as adapter
import coret_rigorous_precise_dot_backend_v1 as rigorous
import coret_structural_support_precise_dot_v1 as structural


SCHEMA = "CORET_RIGOROUS_PRECISE_DOT_CACHED_TIMING_V1"
LABELS = ("B0_QK", "B1_QK", "B2_QK", "B0_AV", "B1_AV", "B2_AV")
FILENAMES = {
    label: f"coret_structural_support_{label}_operands_v1.pt"
    for label in LABELS
}
EXPECTED_ARTIFACT_SHA256 = {
    "B0_QK": "4e2f80a4b4bf95b85a5169b229f0adb4a2cdbbb57c66fbf9943fc69ce75face2",
    "B1_QK": "29e814725ab75765e4234daab4a90ff7cdd70d4fc3edfb94b82e922406f04a39",
    "B2_QK": "279e410c10164af51b262f3cbbcccda4deb8adc1855579635945dc09e3359337",
    "B0_AV": "084f6adf8a074790e075559395258136115e28f6c3898a8e8c1dca73b02067a1",
    "B1_AV": "5b9c6b23d022365dca308c1d88e28301bee19ce8092c6f988c4428e19fb1b07b",
    "B2_AV": "9b7704fdd9c7c0f02a215cbbbf49c0313cc97bfdc6bb1af9e1c3876a3a3f92dd",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def zono(Z, args, state):
    low, high = state["range_low"], state["range_high"]
    return Z(
        args=args, p=state["p"], eps=state["eps"],
        perturbed_word_index=state["perturbed_word_index"],
        zonotope_w=state["coefficient_tensor"].to(args.device), clone=False,
        error_term_range_low=None if low is None else low.to(args.device),
        error_term_range_high=None if high is None else high.to(args.device))


def proof(state):
    return structural.SupportProof(
        tuple(state["support_masks"]), tuple(state["support_ids"]),
        tuple(state["support_reasons"]), int(state["num_tokens"]))


def raw(tensor):
    return tensor.detach().cpu().contiguous().numpy().tobytes(order="C")


def run_one(label, path, Z, args):
    artifact_hash = sha256(path)
    if artifact_hash != EXPECTED_ARTIFACT_SHA256[label]:
        raise RuntimeError(
            f"{label} cached operand hash mismatch: {artifact_hash} != "
            f"{EXPECTED_ARTIFACT_SHA256[label]}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if label.endswith("QK"):
        left_name, right_name, mode = "Q", "K", "QK"
    else:
        left_name, right_name, mode = "probability", "V_transposed", "A.V"
    left_state, right_state = payload[left_name], payload[right_name]
    left = zono(Z, args, left_state)
    right = zono(Z, args, right_state)
    lp, rp = proof(left_state), proof(right_state)
    before_left = raw(left.zonotope_w)
    before_right = raw(right.zonotope_w)
    diagnostics = {}
    torch.cuda.synchronize()
    producer_started = time.perf_counter()
    output = structural.precise_dot_structural(
        left, right, lp, rp, mode=mode,
        generator_tile=structural.AV_GENERATOR_TILE,
        av_temporary_cap_bytes=structural.AV_TEMPORARY_CAP_BYTES,
        diagnostics=diagnostics)
    torch.cuda.synchronize()
    producer_seconds = time.perf_counter() - producer_started
    after_left, after_right = raw(left.zonotope_w), raw(right.zonotope_w)
    if before_left != after_left or before_right != after_right:
        raise RuntimeError("producer mutated cached precise-dot operands")
    output_raw = raw(output.zonotope_w)
    left_shape = tuple(left.zonotope_w.shape)
    right_shape = tuple(right.zonotope_w.shape)
    output_shape = tuple(output.zonotope_w.shape)
    del output, left, right, payload
    gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
    torch.cuda.synchronize()
    checker = rigorous.check_f32(
        left_raw=before_left, right_raw=before_right, output_raw=output_raw,
        left_shape=left_shape, right_shape=right_shape,
        output_shape=output_shape, device_index=0)
    if not checker["accepted"]:
        raise RuntimeError(
            f"{label} rigorous checker rejected producer output: "
            f"minimum_slack={checker['minimum_soundness_slack']} "
            f"minimum_index={checker['minimum_soundness_slack_index']} "
            f"coordinate={checker['coordinate_results'][checker['minimum_soundness_slack_index']]} "
            f"maximum_required={checker['maximum_required_upper']} "
            f"kernel_seconds={checker['kernel_seconds']}")
    return {
        "label": label, "mode": mode,
        "artifact_path": str(path.resolve()),
        "artifact_sha256": artifact_hash,
        "operand_shapes": {"left": left_shape, "right": right_shape},
        "output_shape": output_shape,
        "producer_seconds": producer_seconds,
        "producer_diagnostics": diagnostics,
        "producer_operands_bitwise_unchanged": True,
        "checker": checker,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--labels", nargs="+", choices=LABELS,
                        default=list(LABELS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorized-cached-state-timing", action="store_true")
    args_ns = parser.parse_args()
    if not args_ns.authorized_cached_state_timing:
        raise RuntimeError("explicit cached-state timing authorization required")
    artifact_dir = Path(args_ns.artifact_dir).resolve()
    paths = {label: artifact_dir / FILENAMES[label] for label in args_ns.labels}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"cached operands unavailable: {missing}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the cached-state timing gate")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    modules = adapter.FrozenDeepTModules()
    from Verifiers.Zonotope import Zonotope
    deept_args = adapter.build_deept_args(modules, torch.device("cuda:0"))
    records = []
    started = time.perf_counter()
    for label in args_ns.labels:
        records.append(run_one(label, paths[label], Zonotope, deept_args))
    result = {
        "schema": SCHEMA,
        "records": records,
        "all_accepted": all(item["checker"]["accepted"] for item in records),
        "worst_checker_seconds": max(
            item["checker"]["kernel_seconds"] for item in records),
        "total_wall_seconds": time.perf_counter() - started,
        "scientific_queries": 0,
        "bound_calls": 0,
    }
    body = json.dumps(result, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()
    result["canonical_sha256"] = hashlib.sha256(body).hexdigest()
    destination = Path(args_ns.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(
        result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "schema": SCHEMA, "all_accepted": result["all_accepted"],
        "worst_checker_seconds": result["worst_checker_seconds"],
        "canonical_sha256": result["canonical_sha256"],
        "output": str(destination.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
