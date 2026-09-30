#!/usr/bin/env python3
"""Authenticated sound-FP64 Block-1 continuation starting after QK.

The input is the immutable output of the standalone Block-1 QK job.  This
driver never evaluates Q/K or QK: it verifies the QK artifact and report, then
uses the existing semantic-stage entry points to execute score scaling through
the pre-Block-2 state.  Intermediate artifacts make every semantic boundary
explicit and auditable.
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

import coret_sound_fp64_block0_feasibility_v1 as sound


SCHEMA = "CORET_SOUND_FP64_BLOCK1_CONTINUATION_JOB_V1"
INPUT_SCHEMA = "CORET_SOUND_FP64_BLOCK1_QK_V1"
INPUT_REPORT_SCHEMA = "CORET_SOUND_FP64_BLOCK1_QK_CLUSTER_JOB_V1"
FINAL_SCHEMA = "CORET_SOUND_FP64_BLOCK1_FINAL_V1"
EXPECTED_INPUT_SHA256 = (
    "5b8a8c588e5eed84f15f4433561232f21f980c1998232890da17a81d02307192")
EXPECTED_REPORT_SHA256 = (
    "938c71b27a4f761b2e31bd61ae8878e962d4bfb6e236539d1dec845f289f8393")
EXPECTED_QK_PREDECESSOR_SHA256 = (
    "a0953d1a9b65a1a8dd810567d161a2dd4a4b5d763c6694f5fbb1cc45f7b0503a")
EXPECTED_GENERATORS = 14_000
EXPECTED_PRE_REDUCTION_GENERATORS = 14_128
EXPECTED_NATIVE_FRESH = 64
EXPECTED_NUMERICAL_FRESH = 64
EXPECTED_RETAINED = 13_936
EXPECTED_ABSORBED = 192
EXPECTED_REPLACEMENTS = 64
LAYER_NORM_EPSILON = 1e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _generator_axis(weights: torch.Tensor) -> int:
    if weights.ndim == 3:
        return 0
    if weights.ndim == 4:
        return 1
    raise RuntimeError(f"unsupported relational-state rank: {weights.ndim}")


def _validate_state(name: str, state: dict, expected_generators: int) -> dict:
    if set(state) != {"weights", "range_low", "range_high", "proof"}:
        raise RuntimeError(f"{name}: state field inventory differs")
    weights, low, high = (
        state["weights"], state["range_low"], state["range_high"])
    if not all(isinstance(item, torch.Tensor) for item in (weights, low, high)):
        raise RuntimeError(f"{name}: state tensors missing")
    axis = _generator_axis(weights)
    generator_count = int(weights.shape[axis]) - 1
    if generator_count != expected_generators:
        raise RuntimeError(
            f"{name}: generator count differs: {generator_count} "
            f"!= {expected_generators}")
    if weights.dtype != torch.float64 or low.dtype != torch.float64 \
            or high.dtype != torch.float64:
        raise RuntimeError(f"{name}: state dtype is not binary64")
    if tuple(low.shape) != (generator_count,) \
            or tuple(high.shape) != (generator_count,):
        raise RuntimeError(f"{name}: explicit range shape differs")
    if not bool(torch.isfinite(weights).all() and torch.isfinite(low).all()
                and torch.isfinite(high).all()):
        raise RuntimeError(f"{name}: nonfinite state or range")
    if bool((low > high).any()):
        raise RuntimeError(f"{name}: invalid explicit range ordering")
    proof = state["proof"]
    if set(proof) != {"masks", "ids", "reasons", "num_tokens"}:
        raise RuntimeError(f"{name}: provenance field inventory differs")
    masks, ids, reasons = proof["masks"], proof["ids"], proof["reasons"]
    if not (len(masks) == len(ids) == len(reasons) == generator_count):
        raise RuntimeError(f"{name}: provenance length differs")
    if len(set(ids)) != generator_count \
            or not all(isinstance(item, str) and item for item in ids):
        raise RuntimeError(f"{name}: generator identity/order is invalid")
    num_tokens = int(proof["num_tokens"])
    if num_tokens <= 0 or not all(
            isinstance(mask, int) and 0 <= mask < (1 << num_tokens)
            for mask in masks):
        raise RuntimeError(f"{name}: support provenance is invalid")
    if not all(isinstance(reason, str) and reason for reason in reasons):
        raise RuntimeError(f"{name}: provenance reason is invalid")
    return {
        "shape": list(weights.shape), "generator_axis": axis,
        "generator_count": generator_count, "num_tokens": num_tokens,
        "ordered_ids_sha256": hashlib.sha256(json.dumps(
            ids, separators=(",", ":")).encode()).hexdigest(),
        "ranges_sha256": hashlib.sha256(
            low.contiguous().numpy().tobytes()
            + high.contiguous().numpy().tobytes()).hexdigest(),
    }


def _validate_qk_reduction(reduction: dict, qk_ids: list[str]) -> None:
    expected = {
        "operator": "b1_qk", "count_before": EXPECTED_PRE_REDUCTION_GENERATORS,
        "count_after": EXPECTED_GENERATORS, "retained": EXPECTED_RETAINED,
        "absorbed": EXPECTED_ABSORBED,
        "added_box_generators": EXPECTED_REPLACEMENTS,
    }
    if any(reduction.get(key) != value for key, value in expected.items()):
        raise RuntimeError("embedded QK reduction identity differs")
    if reduction.get("support_inflation") != 3.0819791163594346e-13:
        raise RuntimeError("embedded QK reduction support inflation differs")
    retained = reduction.get("retained_ids")
    replacements = reduction.get("replacement_ids")
    if not isinstance(retained, list) or not isinstance(replacements, list):
        raise RuntimeError("embedded QK reduction ID mapping missing")
    if retained + replacements != qk_ids:
        raise RuntimeError("embedded QK reduction output order differs")


def authenticate(qk_path: Path, qk_report_path: Path,
                 expected_sha256: str = EXPECTED_INPUT_SHA256,
                 expected_report_sha256: str = EXPECTED_REPORT_SHA256) -> dict:
    actual_sha = _sha256(qk_path)
    report_sha = _sha256(qk_report_path)
    if actual_sha != expected_sha256:
        raise RuntimeError(
            f"QK artifact SHA256 mismatch: {actual_sha} != {expected_sha256}")
    if report_sha != expected_report_sha256:
        raise RuntimeError(
            f"QK report SHA256 mismatch: {report_sha} "
            f"!= {expected_report_sha256}")
    payload = sound._load_artifact(qk_path, INPUT_SCHEMA)
    if set(payload.get("states", {})) != {"hidden", "qk"}:
        raise RuntimeError("QK state inventory differs")
    hidden = _validate_state(
        "hidden", payload["states"]["hidden"], EXPECTED_GENERATORS)
    qk = _validate_state("qk", payload["states"]["qk"], EXPECTED_GENERATORS)
    if hidden["num_tokens"] != qk["num_tokens"]:
        raise RuntimeError("QK/hidden provenance token universes differ")
    if hidden["shape"] != [EXPECTED_GENERATORS + 1, 4, 128]:
        raise RuntimeError(f"hidden state shape differs: {hidden['shape']}")
    if qk["shape"] != [4, EXPECTED_GENERATORS + 1, 4, 4]:
        raise RuntimeError(f"QK state shape differs: {qk['shape']}")

    embedded = payload.get("report", {})
    if (embedded.get("schema") != INPUT_REPORT_SCHEMA
            or embedded.get("input_sha256")
            != EXPECTED_QK_PREDECESSOR_SHA256
            or embedded.get("input_generator_count") != EXPECTED_GENERATORS
            or embedded.get("pre_reduction_generator_count")
            != EXPECTED_PRE_REDUCTION_GENERATORS
            or embedded.get("output_generator_count") != EXPECTED_GENERATORS
            or embedded.get("native_fresh_generator_count")
            != EXPECTED_NATIVE_FRESH
            or embedded.get("fp64_numerical_fresh_count")
            != EXPECTED_NUMERICAL_FRESH
            or embedded.get("generic_fallback_count") != 0
            or embedded.get("all_outputs_finite") is not True):
        raise RuntimeError("embedded QK predecessor/topology metadata differs")
    _validate_qk_reduction(
        embedded.get("post_qk_reduction", {}),
        payload["states"]["qk"]["proof"]["ids"])
    qk_mpfr_spot = embedded.get("mpfr_spot")
    if (not isinstance(qk_mpfr_spot, dict)
            or qk_mpfr_spot.get("one_ulp_inward_rejected") is not True
            or qk_mpfr_spot.get(
                "state_reserve_contains_machine_error") is not True):
        raise RuntimeError("embedded QK MPFR/one-ULP evidence differs")

    external = json.loads(qk_report_path.read_text())
    if (external.get("schema") != INPUT_REPORT_SCHEMA
            or external.get("output_sha256") != actual_sha):
        raise RuntimeError("external QK report identity differs")
    for key, value in embedded.items():
        if external.get(key) != value:
            raise RuntimeError(f"external/embedded QK report differs at {key}")
    return {
        "schema": SCHEMA, "preflight_only": True,
        "input_path": str(qk_path), "input_sha256": actual_sha,
        "input_report_path": str(qk_report_path),
        "input_report_sha256": report_sha,
        "hidden": hidden, "qk": qk, "qk_invocations": 0,
        "qk_mpfr_spot": qk_mpfr_spot,
        "scientific_properties": 0, "bound_calls": 0,
    }


def _stage_specs(qk_path: Path, root: Path):
    stage1 = root / "01_softmax.pt"
    av_input = root / "02_av_input.pt"
    av = root / "03_av.pt"
    post_residual = root / "04_attention_residual.pt"
    post_ln = root / "05_post_attention_ln.pt"
    ffn1 = root / "06_ffn_relu.pt"
    ffn2 = root / "07_ffn_output.pt"
    ffn_residual_input = root / "08_ffn_residual_input.pt"
    ffn_residual = root / "09_ffn_residual.pt"
    final = root / "10_block1_final.pt"
    return [
        ("softmax", sound.run_block1_softmax_only, (qk_path,), stage1,
         "CORET_SOUND_FP64_BLOCK1_STAGE1_V1"),
        ("av_prepare", sound.run_block1_stage2_prepare, (stage1,), av_input,
         "CORET_SOUND_FP64_BLOCK1_AV_INPUT_V1"),
        ("attention_value", sound.run_block1_stage2_av_only, (av_input,), av,
         "CORET_SOUND_FP64_BLOCK1_STAGE2_AV_V1"),
        ("attention_projection_residual", sound.run_block1_stage2_post_prepare,
         (av,), post_residual, "CORET_SOUND_FP64_BLOCK1_POST_RESIDUAL_V1"),
        ("post_attention_layernorm", sound.run_block1_stage2_post_ln,
         (post_residual,), post_ln, "CORET_SOUND_FP64_BLOCK1_STAGE2_POST_V1"),
        ("ffn_affine_relu", sound.run_block1_stage3_ffn1, (post_ln,), ffn1,
         "CORET_SOUND_FP64_BLOCK1_FFN1_V1"),
        ("ffn_output", sound.run_block1_stage3_ffn2, (ffn1,), ffn2,
         "CORET_SOUND_FP64_BLOCK1_FFN2_V1"),
        ("ffn_residual_prepare", sound.run_block1_stage3_residual_prepare,
         (post_ln, ffn2), ffn_residual_input,
         "CORET_SOUND_FP64_BLOCK1_FFN_RESIDUAL_INPUT_V1"),
        ("ffn_residual", sound.run_block1_stage3_residual_add,
         (ffn_residual_input,), ffn_residual,
         "CORET_SOUND_FP64_BLOCK1_OUTPUT_RESIDUAL_V1"),
        ("final_layernorm_recenter_reduction", sound.run_block1_stage3_final_ln,
         (ffn_residual,), final, FINAL_SCHEMA),
    ]


def _measurement(report: dict) -> dict | None:
    rows = report.get("measurements", [])
    return rows[-1] if rows else None


def _run_stage(name, function, inputs, output, expected_schema, device):
    if output.exists():
        raise RuntimeError(f"refusing to overwrite stage output: {output}")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    result = function(*inputs, output, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    payload = sound._load_artifact(output, expected_schema)
    for state_name, state in payload.get("states", {}).items():
        _validate_state(
            f"{name}.{state_name}", state,
            int(state["weights"].shape[_generator_axis(state["weights"])]) - 1)
    report = result["report"]
    return {
        "name": name, "output_path": str(output),
        "output_schema": expected_schema, "output_sha256": _sha256(output),
        "wall_seconds": elapsed,
        "peak_allocated_bytes": (int(torch.cuda.max_memory_allocated())
                                 if device.type == "cuda" else 0),
        "peak_reserved_bytes": (int(torch.cuda.max_memory_reserved())
                                if device.type == "cuda" else 0),
        "measurement": _measurement(report),
        "reductions": report.get("reductions", []),
        "variance_lower": report.get("variance_lower"),
        "variance_upper_min": report.get("variance_upper_min"),
        "mpfr_spots": report.get("mpfr_spots", []),
        "dispatch_counts": report.get("dispatch_counts", {}),
        "generic_fallback_count": int(report.get("generic_fallback_count", 0)),
    }


def execute(qk_path: Path, qk_report_path: Path, output_root: Path,
            final_report_path: Path, device_index: int) -> dict:
    authenticated = authenticate(qk_path, qk_report_path)
    if final_report_path.exists():
        raise RuntimeError("refusing to overwrite final continuation report")
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{device_index}")
    if not torch.cuda.is_available():
        raise RuntimeError("Block-1 continuation requires CUDA")
    torch.cuda.set_device(device)
    stages = []
    started = time.perf_counter()
    for spec in _stage_specs(qk_path, output_root):
        stages.append(_run_stage(*spec, device))

    final_path = Path(stages[-1]["output_path"])
    final_payload = sound._load_artifact(final_path, FINAL_SCHEMA)
    final_state = final_payload["states"].get("pre_block2")
    if final_state is None:
        raise RuntimeError("final pre-Block-2 state missing")
    final_identity = _validate_state(
        "pre_block2", final_state,
        int(final_state["weights"].shape[
            _generator_axis(final_state["weights"])]) - 1)
    if final_identity["generator_count"] > sound.MAXIMUM_GENERATORS:
        raise RuntimeError("final Block-1 state exceeds generator policy")
    final_measurement = stages[-1]["measurement"]
    if final_measurement is None \
            or not all(math.isfinite(float(final_measurement[key])) for key in (
                "sound_support_max", "plain_fp64_relational_support_max",
                "explicit_numerical_support_max", "lower_min", "upper_max")):
        raise RuntimeError("final Block-1 state is not finite/measurable")

    generic_fallbacks = sum(stage["generic_fallback_count"] for stage in stages)
    # NativeProductionDispatch exposes generic fallbacks separately and never
    # increments a named family for them.  Every recorded stage must therefore
    # have only the expected native-family dispatch inventory.
    if generic_fallbacks:
        raise RuntimeError("generic fallback invoked during Block-1 continuation")
    expected_dispatch = {
        "softmax": {"softmax": 1},
        "attention_value": {"A.V": 1},
        "post_attention_layernorm": {"LayerNorm": 1},
        "ffn_affine_relu": {"ReLU": 1},
        "final_layernorm_recenter_reduction": {"LayerNorm": 1},
    }
    for stage in stages:
        expected = expected_dispatch.get(stage["name"], {})
        if stage["dispatch_counts"] != expected:
            raise RuntimeError(
                f"{stage['name']}: native dispatch inventory differs: "
                f"{stage['dispatch_counts']} != {expected}")
        for reduction in stage["reductions"]:
            if (reduction.get("count_after") != sound.MAXIMUM_GENERATORS
                    or not math.isfinite(float(
                        reduction.get("support_inflation", math.nan)))
                    or float(reduction["support_inflation"]) < 0
                    or not reduction.get("selection_rule")
                    or not reduction.get("ranking_sha256")):
                raise RuntimeError(
                    f"{stage['name']}: reduction witness summary differs")
    mpfr_spots = ([authenticated["qk_mpfr_spot"]]
                  + [spot for stage in stages for spot in stage["mpfr_spots"]])
    if not mpfr_spots or not all(
            spot.get("one_ulp_inward_rejected") is True
            and spot.get("state_reserve_contains_machine_error") is True
            for spot in mpfr_spots):
        raise RuntimeError("Block-1 MPFR/one-ULP validation incomplete")

    layernorms = {}
    for stage in stages:
        if stage["name"] in {
                "post_attention_layernorm",
                "final_layernorm_recenter_reduction"}:
            lower = stage["variance_lower"]
            if lower is None or lower <= 0:
                raise RuntimeError(f"{stage['name']}: nonpositive variance")
            layernorms[stage["name"]] = {
                "sound_variance_lower": lower,
                "sqrt_domain_margin_after_epsilon": lower + LAYER_NORM_EPSILON,
                "generator_count": stage["measurement"]["generator_count"],
                "numerical_native_ratio": stage["measurement"][
                    "numerical_native_ratio"],
            }

    report = {
        "schema": SCHEMA,
        "verdict": "CORET_SOUND_FP64_BLOCK1_READY",
        "authenticated_input": authenticated,
        "source_hashes": {
            "continuation_runner_sha256": _sha256(Path(__file__)),
            "sound_fp64_implementation_sha256": _sha256(Path(sound.__file__)),
        },
        "stages": stages, "layernorms": layernorms,
        "reductions": [dict(stage=stage["name"], **reduction)
                       for stage in stages for reduction in stage["reductions"]],
        "mpfr_spots": mpfr_spots,
        "generic_fallback_count": generic_fallbacks,
        "final_artifact_path": str(final_path),
        "final_artifact_schema": FINAL_SCHEMA,
        "final_artifact_sha256": _sha256(final_path),
        "final_generator_count": final_identity["generator_count"],
        "final_range": [final_measurement["lower_min"],
                        final_measurement["upper_max"]],
        "final_sound_support_max": final_measurement["sound_support_max"],
        "final_plain_fp64_support_max": final_measurement[
            "plain_fp64_relational_support_max"],
        "final_fp64_widening_max": final_measurement[
            "explicit_numerical_support_max"],
        "final_widening_native_ratio": final_measurement[
            "numerical_native_ratio"],
        "block2_feasible": True,
        "total_seconds": time.perf_counter() - started,
        "qk_recomputed": False,
        "scientific_properties": 0, "bound_calls": 0,
    }
    final_report_path.parent.mkdir(parents=True, exist_ok=True)
    final_report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qk-artifact", type=Path, required=True)
    parser.add_argument("--qk-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--final-report", type=Path)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        result = authenticate(args.qk_artifact, args.qk_report)
    else:
        if args.output_root is None or args.final_report is None:
            parser.error("--output-root and --final-report are required")
        result = execute(
            args.qk_artifact, args.qk_report, args.output_root,
            args.final_report, args.device_index)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
