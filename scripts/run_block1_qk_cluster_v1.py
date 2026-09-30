#!/usr/bin/env python3
"""Standalone authenticated A40 handoff for sound-FP64 Block-1 QK.

This entry point performs no property verification or radius search.  It loads
the already-reduced Block-1 Q/K operands, executes the unchanged native precise
QK transformer, embeds the existing analytic binary64 roundoff allowance, and
applies the accepted 14,000-generator sound reduction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research_hab"))

import coret_native_semantics_production_graph_v1 as production
import coret_sound_fp64_block0_feasibility_v1 as sound
import coret_structural_support_precise_dot_v1 as structural


SCHEMA = "CORET_SOUND_FP64_BLOCK1_QK_CLUSTER_JOB_V1"
INPUT_SCHEMA = "CORET_SOUND_FP64_BLOCK1_QK_INPUT_V1"
OUTPUT_SCHEMA = "CORET_SOUND_FP64_BLOCK1_QK_V1"
EXPECTED_INPUT_SHA256 = (
    "a0953d1a9b65a1a8dd810567d161a2dd4a4b5d763c6694f5fbb1cc45f7b0503a"
)
EXPECTED_INPUT_GENERATORS = 14_000
EXPECTED_QK_FRESH = 64


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_authenticated(path: Path, expected_sha256: str):
    actual = _sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"aligned Q/K artifact SHA256 mismatch: {actual} != {expected_sha256}")
    payload = sound._load_artifact(path, INPUT_SCHEMA)
    states = payload.get("states", {})
    if set(states) != {"hidden", "q", "k"}:
        raise RuntimeError("aligned Q/K artifact state inventory differs")
    reduction = payload.get("report", {}).get("reductions", [])
    if len(reduction) != 1:
        raise RuntimeError("aligned Q/K reduction witness missing")
    record = reduction[0]
    expected = {
        "operator": "b1_pre_qk_pair", "count_before": 14362,
        "count_after": 14000, "retained": 12976, "absorbed": 1386,
        "added_box_generators": 1024,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError("aligned Q/K reduction identity differs")
    if record.get("support_inflation") != 3.8968828164342995e-13:
        raise RuntimeError("aligned Q/K reduction inflation differs")
    return payload, actual


def preflight(input_path: Path, expected_sha256: str) -> dict:
    payload, actual = _load_authenticated(input_path, expected_sha256)
    q = payload["states"]["q"]
    k = payload["states"]["k"]
    q_shape, k_shape = list(q["weights"].shape), list(k["weights"].shape)
    if q_shape != [4, 14001, 4, 32] or k_shape != q_shape:
        raise RuntimeError(f"aligned Q/K tensor shape differs: {q_shape}, {k_shape}")
    if q["proof"] != k["proof"]:
        raise RuntimeError("aligned Q/K proof universes differ")
    if len(q["proof"]["ids"]) != EXPECTED_INPUT_GENERATORS:
        raise RuntimeError("aligned Q/K generator identity count differs")
    return {
        "schema": SCHEMA,
        "preflight_only": True,
        "input_artifact": str(input_path),
        "input_sha256": actual,
        "q_shape": q_shape, "k_shape": k_shape,
        "input_generator_count": EXPECTED_INPUT_GENERATORS,
        "native_qk_invocations": 0,
    }


def _mpfr_spot(q, k, raw_qk, reserve) -> dict:
    """One actual retained coefficient checked with the accepted MPFR helper."""
    with sound._mp_context(sound.gmpy2.RoundToNearest):
        exact = sound._mp(0)
        for feature in range(q.zonotope_w.shape[-1]):
            exact += (sound._mp(q.zonotope_w[0, 0, 0, feature])
                      * sound._mp(k.zonotope_w[0, 1, 0, feature]))
            exact += (sound._mp(q.zonotope_w[0, 1, 0, feature])
                      * sound._mp(k.zonotope_w[0, 0, 0, feature]))
    return sound._oracle_containment(
        "block1_qk_actual_retained_coefficient", exact,
        raw_qk.zonotope_w[0, 1, 0, 0], float(reserve.max()))


def _select_native_fresh_rows(rows: torch.Tensor,
                              native_generator_count: int) -> torch.Tensor:
    expected_total = EXPECTED_INPUT_GENERATORS + EXPECTED_QK_FRESH
    if native_generator_count != expected_total:
        raise RuntimeError(
            f"native QK generator count differs: {native_generator_count} "
            f"!= {expected_total}")
    start, stop = EXPECTED_INPUT_GENERATORS, expected_total
    if rows.ndim == 3:
        generator_axis = 0
        selected = rows[start:stop, :, :]
    elif rows.ndim == 4:
        generator_axis = 1
        selected = rows[:, start:stop, :, :]
    else:
        raise RuntimeError(f"unsupported QK generator-row rank: {rows.ndim}")
    if selected.numel() == 0:
        raise RuntimeError("native QK fresh-row selection is empty")
    if selected.shape[generator_axis] != EXPECTED_QK_FRESH:
        raise RuntimeError(
            "native QK fresh-row selection count differs: "
            f"{selected.shape[generator_axis]} != {EXPECTED_QK_FRESH}")
    return selected


def execute(input_path: Path, output_path: Path, report_path: Path,
            expected_sha256: str, device_index: int) -> dict:
    if output_path.exists() or report_path.exists():
        raise RuntimeError("QK output/report already exists; refusing overwrite")
    payload, actual_sha = _load_authenticated(input_path, expected_sha256)
    if not torch.cuda.is_available():
        raise RuntimeError("standalone QK cluster job requires CUDA")
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    prior_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    started_total = time.perf_counter()
    try:
        with sound.pinned_zonotope() as Zonotope:
            args = sound._args(device)
            hidden, hidden_proof = sound._state_from_payload(
                payload["states"]["hidden"], Zonotope, args, device)
            q, proof = sound._state_from_payload(
                payload["states"]["q"], Zonotope, args, device)
            k, k_proof = sound._state_from_payload(
                payload["states"]["k"], Zonotope, args, device)
            if proof != k_proof:
                raise RuntimeError("aligned Q/K proof universes differ after load")
            if q.num_error_terms != EXPECTED_INPUT_GENERATORS:
                raise RuntimeError("aligned Q/K generator count differs after load")

            delegate = structural.StructuralNativeSemanticOperators()
            delegate._qk_index = 1
            delegate._hidden = proof
            dispatch = production.NativeProductionDispatch(delegate=delegate)
            measurements, reductions = [], []
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started_qk = time.perf_counter()

            raw_qk = dispatch.qk(q, k)
            raw_proof = structural.get_support(raw_qk)
            if raw_qk.num_error_terms != (
                    EXPECTED_INPUT_GENERATORS + EXPECTED_QK_FRESH):
                raise RuntimeError("native QK fresh-generator count differs")
            operations = 32 * (q.num_error_terms + 1) ** 2 * 16 + 4096
            reserve = sound._reserve_from_majorant(
                sound._bilinear_majorant(q, k), operations)
            mpfr_spot = _mpfr_spot(q, k, raw_qk, reserve)
            qk, qk_proof = sound._inject(
                raw_qk, raw_proof, [q, k], "b1_qk", operations,
                measurements, reserve=reserve)
            pre_reduction_count = qk.num_error_terms
            qk, qk_proof = sound._maybe_reduce(
                qk, qk_proof, "b1_qk", reductions)
            torch.cuda.synchronize()
            qk_seconds = time.perf_counter() - started_qk

            if len(reductions) != 1 or qk.num_error_terms != 14_000:
                raise RuntimeError("post-QK normal reduction policy differs")
            measurement = sound._state_measurement(
                "block1_qk", qk, qk_proof, qk_seconds)
            qk_low, qk_high = qk.concretize()
            scale = 1.0 / math.sqrt(32)
            scale_reserve = sound._outward_positive(
                sound._absolute_hull(qk) * abs(scale)
                * (2 * sound._gamma(1) + sound.FP64_U))
            score_low = qk_low * scale - scale_reserve
            score_high = qk_high * scale + scale_reserve
            native_fresh_rows = _select_native_fresh_rows(
                sound._generator_rows(raw_qk), raw_qk.num_error_terms)
            minimum_native_fresh_radius = float(native_fresh_rows.min())
            all_finite = bool(
                torch.isfinite(qk.zonotope_w).all()
                and torch.isfinite(score_low).all()
                and torch.isfinite(score_high).all())
            if not all_finite or minimum_native_fresh_radius < 0:
                raise RuntimeError("QK finite/nonnegative-radius domain gate failed")

            semantic_report = {
                "schema": SCHEMA,
                "pinned_revision": sound.PINNED_REVISION,
                "input_sha256": actual_sha,
                "input_generator_count": EXPECTED_INPUT_GENERATORS,
                "native_retained_generator_count": EXPECTED_INPUT_GENERATORS,
                "native_fresh_generator_count": EXPECTED_QK_FRESH,
                "fp64_numerical_fresh_count": EXPECTED_QK_FRESH,
                "pre_reduction_generator_count": pre_reduction_count,
                "output_generator_count": qk.num_error_terms,
                "post_qk_reduction": reductions[0],
                "measurement": measurement,
                "sound_score_lower_min": float(score_low.min()),
                "sound_score_upper_max": float(score_high.max()),
                "minimum_native_fresh_radius": minimum_native_fresh_radius,
                "scale_positive_margin": scale,
                "all_outputs_finite": all_finite,
                "mpfr_spot": mpfr_spot,
                "rigorous_qk_checker": {
                    "executed": False,
                    "reason": (
                        "accepted rigorous precise-dot backend is explicitly "
                        "restricted to frozen float32 witnesses; this state is float64"),
                    "analytic_fp64_reserve_checked": True,
                },
                "generic_fallback_count": dispatch.generic_family_invocations,
            }
            if dispatch.generic_family_invocations != 0:
                raise RuntimeError("generic QK fallback was invoked")

            artifact_payload = {
                "schema": OUTPUT_SCHEMA,
                "pinned_revision": sound.PINNED_REVISION,
                "states": {
                    "hidden": sound._state_payload(hidden, hidden_proof),
                    "qk": sound._state_payload(qk, qk_proof),
                },
                # Runtime and allocator telemetry are intentionally excluded
                # from the resumable scientific artifact.
                "report": semantic_report,
            }
            torch.save(artifact_payload, output_path)
            output_sha = _sha256(output_path)
            torch.cuda.synchronize()
            total_seconds = time.perf_counter() - started_total
            report = dict(semantic_report)
            report.update({
                "input_artifact": str(input_path),
                "output_artifact": str(output_path),
                "output_sha256": output_sha,
                "qk_seconds": qk_seconds,
                "total_seconds": total_seconds,
                "peak_gpu_allocated_bytes": int(
                    torch.cuda.max_memory_allocated()),
                "peak_gpu_reserved_bytes": int(
                    torch.cuda.max_memory_reserved()),
                "gpu_name": torch.cuda.get_device_name(device),
                "compute_capability": list(
                    torch.cuda.get_device_capability(device)),
            })
            report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n")
            return report
    finally:
        torch.set_default_dtype(prior_dtype)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--expected-sha256", default=EXPECTED_INPUT_SHA256)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.expected_sha256 != EXPECTED_INPUT_SHA256:
        raise RuntimeError("expected input SHA is frozen and may not be overridden")
    if args.preflight_only:
        result = preflight(args.input, args.expected_sha256)
    else:
        if args.output is None or args.report is None:
            parser.error("--output and --report are required for execution")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        result = execute(
            args.input, args.output, args.report, args.expected_sha256,
            args.device_index)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
