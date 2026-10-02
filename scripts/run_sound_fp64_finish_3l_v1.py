#!/usr/bin/env python3
"""Authenticated one-job sound-FP64 completion of the frozen 3-layer fixture.

This driver starts at the accepted Block-1 final artifact and executes Block 2,
the final representation boundary, the frozen pooler, and the direct binary
classification margin without intermediate semantic substitutions.  It is a
cluster handoff: importing or running ``--preflight-only`` never enters a
verifier operator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import gmpy2
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research_hab"))

import coret_native_semantics_production_graph_v1 as production
import coret_production_prefix_trace_v1 as prefix
import coret_sound_fp64_block0_feasibility_v1 as sound
import coret_structural_support_precise_dot_v1 as structural
from deept_stagea_model import deept_functional_reference


SCHEMA = "CORET_SOUND_FP64_3L_FINISH_JOB_V1"
INPUT_SCHEMA = "CORET_SOUND_FP64_BLOCK1_FINAL_V1"
INPUT_REPORT_SCHEMA = "CORET_SOUND_FP64_BLOCK1_CONTINUATION_JOB_V1"
OUTPUT_SCHEMA = "CORET_SOUND_FP64_3L_FINAL_V1"
EXPECTED_INPUT_SHA256 = (
    "3eb164bca1858a1f0fa653edc3bacab17dc6ccbb116e5ac7ede4ee5b5e71df10")
EXPECTED_REPORT_SHA256 = (
    "cc6d644a551dfc0beee8284a5c1ee5b94f12d9453d9a700e813735fa35565f74")
LAYER_NORM_EPSILON = 1e-12
LAYERNORM_DOMAIN_REASON = "SOUND_FP64_LAYERNORM_VARIANCE_DOMAIN_FAILURE"


def _layernorm_variance_state(residual, proof, label: str):
    """Build the exact native pre-sqrt variance and audit its domain.

    Diagnostics never replace DeepT's relational square transformer and never
    participate in the domain decision.
    """
    weights = residual.zonotope_w
    if weights.ndim != 3 or weights.shape[-1] != 128:
        raise RuntimeError(f"{label}: malformed LayerNorm input shape")
    token_count, width = int(weights.shape[1]), int(weights.shape[2])
    if (int(proof.num_tokens) != token_count
            or len(proof.ids) != residual.num_error_terms
            or len(proof.masks) != residual.num_error_terms
            or len(proof.reasons) != residual.num_error_terms):
        raise RuntimeError(f"{label}: malformed LayerNorm provenance")
    if not bool(torch.isfinite(weights).all()):
        raise RuntimeError(f"{label}: malformed nonfinite LayerNorm input")
    low_ranges, high_ranges = sound._ranges(residual)
    if (tuple(low_ranges.shape) != (residual.num_error_terms,)
            or tuple(high_ranges.shape) != (residual.num_error_terms,)
            or not bool(torch.isfinite(low_ranges).all()
                        and torch.isfinite(high_ranges).all())
            or bool((low_ranges > high_ranges).any())):
        raise RuntimeError(f"{label}: malformed LayerNorm ranges")
    structural.validate_support(residual, proof, token_axis=1)

    average = torch.ones((width, width), dtype=torch.float64,
                         device=residual.device) / width
    centered = residual.add(residual.matmul(average).multiply(-1.0))
    variance = centered.square_and_sum_and_repeat().multiply(1.0 / width)
    variance_low, variance_high = variance.concretize()
    variance_center = variance.zonotope_w[0]
    expected_shape = (token_count, width)
    if (tuple(variance_low.shape) != expected_shape
            or tuple(variance_high.shape) != expected_shape
            or tuple(variance_center.shape) != expected_shape
            or not bool(torch.isfinite(variance_low).all()
                        and torch.isfinite(variance_high).all()
                        and torch.isfinite(variance_center).all())
            or bool((variance_low > variance_high).any())):
        raise RuntimeError(f"{label}: malformed LayerNorm variance enclosure")

    flat_index = int(torch.argmin(variance_low).item())
    token_index, coordinate_index = divmod(flat_index, width)
    variance_min = float(variance_low[token_index, coordinate_index])
    center_at_min = float(variance_center[token_index, coordinate_index])

    diagnostics = {
        "reason_code": LAYERNORM_DOMAIN_REASON,
        "label": label,
        "input_shape": list(weights.shape),
        "token_count": token_count,
        "hidden_dimension": width,
        "generator_count": int(residual.num_error_terms),
        "minimum_token_index": token_index,
        "minimum_coordinate_index": coordinate_index,
        "nominal_centered_second_moment": float(
            centered.zonotope_w[0, token_index].square().mean()),
        "variance_affine_center": center_at_min,
        "variance_lower_support": center_at_min - variance_min,
        "variance_upper_support": (
            float(variance_high[token_index, coordinate_index])
            - center_at_min),
        "sound_variance_lower": variance_min,
        "sound_variance_upper_at_minimum": float(
            variance_high[token_index, coordinate_index]),
        "layernorm_epsilon": LAYER_NORM_EPSILON,
        "sqrt_input_lower": variance_min + LAYER_NORM_EPSILON,
        "native_sqrt_threshold": LAYER_NORM_EPSILON,
        "sqrt_safety_margin": variance_min,
        "domain_admissible": variance_min > 0,
    }
    if variance_min <= 0:
        # This independent interval decomposition is diagnostic only. It
        # bounds the exact centered variance attributable to explicit FP64
        # generators; the authoritative domain remains the relational lower.
        numerical_reasons = {
            "fp64_roundoff_coordinate_box",
            "sound_fp64_coordinate_box_replacement_with_numerical",
        }
        rows = centered.zonotope_w[1:, token_index, :]
        native_indices = [
            index for index, reason in enumerate(proof.reasons)
            if reason not in numerical_reasons]
        numerical_indices = [
            index for index, reason in enumerate(proof.reasons)
            if reason in numerical_reasons]

        def contribution(indices):
            if not indices:
                zero = torch.zeros_like(centered.zonotope_w[0, token_index])
                return zero, zero
            index = torch.tensor(
                indices, dtype=torch.long, device=residual.device)
            selected = rows.index_select(0, index)
            selected_low = low_ranges.index_select(
                0, index).reshape(-1, 1)
            selected_high = high_ranges.index_select(
                0, index).reshape(-1, 1)
            return (torch.minimum(selected * selected_low,
                                  selected * selected_high).sum(dim=0),
                    torch.maximum(selected * selected_low,
                                  selected * selected_high).sum(dim=0))

        native_delta_low, native_delta_high = contribution(native_indices)
        numerical_delta_low, numerical_delta_high = contribution(
            numerical_indices)
        centered_center = centered.zonotope_w[0, token_index]
        plain_low = centered_center + native_delta_low
        plain_high = centered_center + native_delta_high
        plain_square_lower = torch.where(
            (plain_low <= 0) & (plain_high >= 0),
            torch.zeros_like(plain_low),
            torch.minimum(plain_low.square(), plain_high.square()))
        plain_abs = torch.maximum(plain_low.abs(), plain_high.abs())
        numerical_abs = torch.maximum(
            numerical_delta_low.abs(), numerical_delta_high.abs())
        diagnostics.update({
            "plain_relational_variance_lower_bound": float(
                plain_square_lower.mean()),
            "plain_relational_bound_method": (
                "coordinate_interval_lower_bound_from_non_numerical_generators"),
            "numerical_variance_widening_upper_bound": float((
                2.0 * plain_abs * numerical_abs
                + numerical_abs.square()).mean()),
            "native_generator_count": len(native_indices),
            "numerical_generator_count": len(numerical_indices),
        })
    return centered, variance, diagnostics


def _domain_failure_report(authenticated, clean_label, nominal_logits,
                           diagnostics, rows, reductions, dispatch,
                           total_started, experimental_layernorm=None):
    """Return a controlled, non-certificate outcome for a valid domain miss."""
    return {
        "schema": SCHEMA,
        "verdict": "CORET_SOUND_FP64_3L_UNCERTIFIED_DOMAIN_FAILURE",
        "authenticated_input": authenticated,
        "fixture_token_ids": list(prefix.FIXTURE_TOKEN_IDS),
        "fixture_rho_hex": float(prefix.FIXTURE_RHO).hex(),
        "clean_label": clean_label,
        "nominal_logits": nominal_logits,
        "domain_failure": diagnostics,
        "stages": rows,
        "reductions": reductions,
        "dispatch_counts_before_domain_failure": dict(dispatch.counts),
        "generic_fallback_count": 0,
        "final_sound_margin": None,
        "final_generator_count": diagnostics["generator_count"],
        "total_seconds": time.perf_counter() - total_started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "scientific_properties": 0,
        "bound_calls": 0,
        "experimental_psd_layernorm": experimental_layernorm,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_state(state: dict, expected_num_tokens: int = 4) -> dict:
    if not isinstance(expected_num_tokens, int) or expected_num_tokens <= 0:
        raise RuntimeError("expected pre-Block-2 token count is invalid")
    if set(state) != {"weights", "range_low", "range_high", "proof"}:
        raise RuntimeError("pre-Block-2 state field inventory differs")
    weights, low, high = state["weights"], state["range_low"], state["range_high"]
    if not all(isinstance(item, torch.Tensor) for item in (weights, low, high)):
        raise RuntimeError("pre-Block-2 state tensors missing")
    if (weights.ndim != 3
            or tuple(weights.shape[1:]) != (expected_num_tokens, 128)):
        raise RuntimeError("pre-Block-2 hidden-state shape differs")
    generators = int(weights.shape[0]) - 1
    if not 0 < generators <= sound.MAXIMUM_GENERATORS:
        raise RuntimeError("pre-Block-2 generator count violates frozen policy")
    if any(item.dtype != torch.float64 for item in (weights, low, high)):
        raise RuntimeError("pre-Block-2 state is not binary64")
    if tuple(low.shape) != (generators,) or tuple(high.shape) != (generators,):
        raise RuntimeError("pre-Block-2 range shape differs")
    if not bool(torch.isfinite(weights).all() and torch.isfinite(low).all()
                and torch.isfinite(high).all()):
        raise RuntimeError("pre-Block-2 state is nonfinite")
    if bool((low > high).any()):
        raise RuntimeError("pre-Block-2 ranges are invalid")
    proof = state["proof"]
    if set(proof) != {"masks", "ids", "reasons", "num_tokens"}:
        raise RuntimeError("pre-Block-2 provenance inventory differs")
    if int(proof["num_tokens"]) != expected_num_tokens:
        raise RuntimeError("pre-Block-2 token universe differs")
    if not (len(proof["masks"]) == len(proof["ids"])
            == len(proof["reasons"]) == generators):
        raise RuntimeError("pre-Block-2 provenance length differs")
    if len(set(proof["ids"])) != generators:
        raise RuntimeError("pre-Block-2 ordered generator IDs are not unique")
    if not all(isinstance(mask, int)
               and 0 <= mask < (1 << expected_num_tokens)
               for mask in proof["masks"]):
        raise RuntimeError("pre-Block-2 support masks are invalid")
    return {
        "shape": list(weights.shape), "generator_count": generators,
        "ordered_ids_sha256": hashlib.sha256(json.dumps(
            proof["ids"], separators=(",", ":")).encode()).hexdigest(),
        "ranges_sha256": hashlib.sha256(
            low.contiguous().numpy().tobytes()
            + high.contiguous().numpy().tobytes()).hexdigest(),
    }


def authenticate(input_path: Path, report_path: Path,
                 expected_input_sha256: str = EXPECTED_INPUT_SHA256,
                 expected_report_sha256: str = EXPECTED_REPORT_SHA256,
                 expected_num_tokens: int = 4) -> dict:
    input_sha, report_sha = _sha256(input_path), _sha256(report_path)
    if input_sha != expected_input_sha256:
        raise RuntimeError(
            f"Block-1 artifact SHA256 mismatch: {input_sha} != "
            f"{expected_input_sha256}")
    if report_sha != expected_report_sha256:
        raise RuntimeError(
            f"Block-1 report SHA256 mismatch: {report_sha} != "
            f"{expected_report_sha256}")
    payload = sound._load_artifact(input_path, INPUT_SCHEMA)
    if set(payload.get("states", {})) != {"pre_block2"}:
        raise RuntimeError("Block-1 final state inventory differs")
    identity = _validate_state(
        payload["states"]["pre_block2"], expected_num_tokens)
    report = json.loads(report_path.read_text())
    if (report.get("schema") != INPUT_REPORT_SCHEMA
            or report.get("verdict") != "CORET_SOUND_FP64_BLOCK1_READY"
            or report.get("final_artifact_schema") != INPUT_SCHEMA
            or report.get("final_artifact_sha256") != input_sha
            or report.get("final_generator_count") != identity["generator_count"]
            or report.get("block2_feasible") is not True
            or report.get("qk_recomputed") is not False
            or report.get("generic_fallback_count") != 0
            or report.get("scientific_properties") != 0
            or report.get("bound_calls") != 0):
        raise RuntimeError("Block-1 continuation report identity differs")
    stages = report.get("stages")
    if (not isinstance(stages, list) or not stages
            or stages[-1].get("output_schema") != INPUT_SCHEMA
            or stages[-1].get("output_sha256") != input_sha
            or any(stage.get("generic_fallback_count", 0) != 0
                   for stage in stages)):
        raise RuntimeError("Block-1 stage chain does not terminate at input")
    embedded = payload.get("report", {})
    if (embedded.get("generic_fallback_count") != 0
            or embedded.get("dispatch_counts") != {"LayerNorm": 1}
            or not embedded.get("mpfr_spots")
            or not all(spot.get("one_ulp_inward_rejected") is True
                       for spot in embedded["mpfr_spots"])):
        raise RuntimeError("embedded Block-1 final evidence differs")
    return {
        "input_sha256": input_sha, "report_sha256": report_sha,
        "state_identity": identity, "operator_calls": 0,
        "scientific_properties": 0, "bound_calls": 0,
    }


def _clean_label(checkpoint: dict) -> tuple[int, list[float]]:
    ids = torch.tensor([prefix.FIXTURE_TOKEN_IDS], dtype=torch.long)
    mask = torch.ones_like(ids)
    logits, _ = deept_functional_reference(checkpoint, {
        "num_attention_heads": 4, "hidden_size": 128,
        "num_hidden_layers": 3,
    }, ids, mask)
    if tuple(logits.shape) != (1, 2) or not bool(torch.isfinite(logits).all()):
        raise RuntimeError("frozen nominal classifier result is invalid")
    return int(logits.argmax(dim=-1).item()), logits[0].tolist()


def _spot(label: str, exact, machine, reserve: float) -> dict:
    return sound._oracle_containment(label, exact, machine, reserve)


def _stage_measure(label, state, proof, started, rows):
    if state.device.type == "cuda":
        torch.cuda.synchronize()
    row = sound._state_measurement(
        label, state, proof, time.perf_counter() - started)
    rows.append(row)
    return row


def execute(input_path: Path, report_path: Path, output_path: Path,
            output_report: Path, device_index: int,
            expected_input_sha256: str = EXPECTED_INPUT_SHA256,
            expected_report_sha256: str = EXPECTED_REPORT_SHA256,
            run_representative_mpfr: bool = True,
            expected_num_tokens: int | None = None,
            experimental_post_attention_layernorm=None,
            experimental_layernorm_failure_capture=None) -> dict:
    if expected_num_tokens is None:
        expected_num_tokens = len(prefix.FIXTURE_TOKEN_IDS)
    authenticated = authenticate(
        input_path, report_path, expected_input_sha256,
        expected_report_sha256, expected_num_tokens)
    if output_path.exists() or output_report.exists():
        raise RuntimeError("refusing to overwrite 3-layer output")
    if not torch.cuda.is_available():
        raise RuntimeError("sound-FP64 3-layer completion requires CUDA")
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    total_started = time.perf_counter()
    prior_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        payload = sound._load_artifact(input_path, INPUT_SCHEMA)
        with sound.pinned_zonotope() as Zonotope:
            args = sound._args(device)
            hidden, hidden_proof = sound._state_from_payload(
                payload["states"]["pre_block2"], Zonotope, args, device)
            structural.validate_support(hidden, hidden_proof, token_axis=1)
            checkpoint = prefix._load_checkpoint()
            clean_label, nominal_logits = _clean_label(checkpoint)
            base = "bert.encoder.layer.2"
            parameters = {
                name: sound._parameter(checkpoint, path, device)
                for name, path in {
                    "query": base + ".attention.self.query",
                    "key": base + ".attention.self.key",
                    "value": base + ".attention.self.value",
                    "attention_output": base + ".attention.output.dense",
                    "attention_ln": base + ".attention.output.LayerNorm",
                    "ffn_first": base + ".intermediate.dense",
                    "ffn_second": base + ".output.dense",
                    "output_ln": base + ".output.LayerNorm",
                    "pooler": "bert.pooler.dense",
                }.items()
            }
            delegate = structural.StructuralNativeSemanticOperators()
            delegate._qk_index = 2
            delegate._softmax_index = 2
            delegate._av_index = 2
            delegate._layer_norm_index = 5
            delegate._relu_index = 2
            delegate._hidden = hidden_proof
            dispatch = production.NativeProductionDispatch(delegate=delegate)
            measurements, reductions, rows, spots = [], [], [], []
            layernorms = {}
            experimental_layernorm = None
            block_input, block_input_proof = hidden, hidden_proof

            # Q/K and precise attention scores.
            started = time.perf_counter()
            q, qp = sound._dense_sound(
                hidden, hidden_proof, parameters["query"], "b2_q_affine",
                measurements)
            k, kp = sound._dense_sound(
                hidden, hidden_proof, parameters["key"], "b2_k_affine",
                measurements)
            q, k, pair_proof = sound._align_states(q, qp, k, kp,
                                                    "b2_qk_branches")
            q = q.add_attention_heads_dim(4)
            k = k.add_attention_heads_dim(4)
            structural.attach_support(q, pair_proof)
            structural.attach_support(k, pair_proof)
            q, k, pair_proof = sound._maybe_reduce_pair(
                q, k, pair_proof, "b2_pre_qk_pair", reductions)
            delegate._hidden = pair_proof
            raw_qk = dispatch.qk(q, k)
            raw_qk_proof = structural.get_support(raw_qk)
            qk_ops = 32 * (q.num_error_terms + 1) ** 2 * 16 + 4096
            qk_reserve = sound._reserve_from_majorant(
                sound._bilinear_majorant(q, k), qk_ops)
            if run_representative_mpfr:
                with sound._mp_context(gmpy2.RoundToNearest):
                    exact = gmpy2.mpfr(0)
                    for feature in range(q.zonotope_w.shape[-1]):
                        exact += (sound._mp(q.zonotope_w[0, 0, 0, feature])
                                  * sound._mp(k.zonotope_w[0, 1, 0, feature]))
                        exact += (sound._mp(q.zonotope_w[0, 1, 0, feature])
                                  * sound._mp(k.zonotope_w[0, 0, 0, feature]))
                spots.append(_spot(
                    "block2_qk_retained_coefficient", exact,
                    raw_qk.zonotope_w[0, 1, 0, 0], float(qk_reserve.max())))
            qk, qk_proof = sound._inject(
                raw_qk, raw_qk_proof, [q, k], "b2_qk", qk_ops,
                measurements, reserve=qk_reserve)
            qk, qk_proof = sound._maybe_reduce(
                qk, qk_proof, "b2_qk", reductions)
            _stage_measure("block2_qk", qk, qk_proof, started, rows)

            # Scale and native relational softmax.
            started = time.perf_counter()
            scale = 1.0 / math.sqrt(32)
            raw_scores = qk.multiply(scale)
            structural.attach_support(raw_scores, qk_proof)
            score_reserve = sound._outward_positive(
                sound._absolute_hull(qk) * abs(scale)
                * (2 * sound._gamma(1) + sound.FP64_U))
            scores, score_proof = sound._inject(
                raw_scores, qk_proof, [qk], "b2_score_scaling", 1,
                measurements, reserve=score_reserve)
            scores, score_proof = sound._maybe_reduce(
                scores, score_proof, "b2_score_scaling", reductions)
            score_low, score_high = scores.concretize()
            delegate._score = score_proof
            raw_probability = dispatch.softmax(scores, no_constraints=False)
            raw_probability_proof = structural.get_support(raw_probability)
            softmax_ops = 64 * (scores.num_error_terms + 1) + 8192
            probability, probability_proof = sound._inject(
                raw_probability, raw_probability_proof, [scores],
                "b2_softmax", softmax_ops, measurements,
                reserve=sound._reserve_from_majorant(
                    sound._softmax_majorant(scores), softmax_ops))
            probability, probability_proof = sound._maybe_reduce(
                probability, probability_proof, "b2_softmax", reductions)
            _stage_measure(
                "block2_softmax", probability, probability_proof, started, rows)

            # V projection and precise A.V.
            started = time.perf_counter()
            value, value_proof = sound._dense_sound(
                hidden, hidden_proof, parameters["value"], "b2_v_affine",
                measurements)
            value = value.add_attention_heads_dim(4)
            structural.attach_support(value, value_proof)
            probability, value, av_proof = sound._align_states(
                probability, probability_proof, value, value_proof,
                "b2_av_branches")
            probability, value, av_proof = sound._maybe_reduce_pair(
                probability, value, av_proof, "b2_pre_av_pair", reductions)
            delegate._probability = av_proof
            delegate._value = av_proof
            raw_context = dispatch.attention_value(probability, value)
            raw_context_proof = structural.get_support(raw_context)
            av_ops = 32 * (probability.num_error_terms + 1) ** 2 * 16 + 4096
            av_reserve = sound._reserve_from_majorant(
                sound._bilinear_majorant(probability, value.t()), av_ops)
            if run_representative_mpfr:
                transposed = value.t()
                with sound._mp_context(gmpy2.RoundToNearest):
                    exact = gmpy2.mpfr(0)
                    for key_index in range(value.num_words):
                        exact += (sound._mp(
                            probability.zonotope_w[0, 0, 0, key_index])
                                  * sound._mp(
                            transposed.zonotope_w[0, 1, 0, key_index]))
                        exact += (sound._mp(
                            probability.zonotope_w[0, 1, 0, key_index])
                                  * sound._mp(
                            transposed.zonotope_w[0, 0, 0, key_index]))
                spots.append(_spot(
                    "block2_av_retained_coefficient", exact,
                    raw_context.zonotope_w[0, 1, 0, 0],
                    float(av_reserve.max())))
            context, context_proof = sound._inject(
                raw_context, raw_context_proof, [probability, value],
                "b2_attention_value", av_ops, measurements,
                reserve=av_reserve)
            context, context_proof = sound._maybe_reduce(
                context, context_proof, "b2_attention_value", reductions)
            _stage_measure(
                "block2_attention_value", context, context_proof, started, rows)

            # Attention output, residual, and first LayerNorm.
            started = time.perf_counter()
            context = context.remove_attention_heads_dim()
            structural.attach_support(context, context_proof)
            attention, attention_proof = sound._dense_sound(
                context, context_proof, parameters["attention_output"],
                "b2_attention_output_affine", measurements)
            attention, attention_proof = sound._maybe_reduce(
                attention, attention_proof, "b2_attention_output_affine",
                reductions)
            attention, residual_input, residual_proof = sound._align_states(
                attention, attention_proof, block_input, block_input_proof,
                "b2_attention_residual")
            attention, residual_input, residual_proof = sound._maybe_reduce_pair(
                attention, residual_input, residual_proof,
                "b2_pre_attention_residual_pair", reductions)
            raw_residual = attention.add(residual_input)
            residual, residual_proof = sound._inject(
                raw_residual, residual_proof, [attention, residual_input],
                "b2_attention_residual", 1, measurements,
                reserve=sound._add_reserve(attention, residual_input))
            residual, residual_proof = sound._maybe_reduce(
                residual, residual_proof, "b2_attention_residual", reductions)
            width = residual.word_embedding_size
            if dispatch.generic_family_invocations != 0:
                raise RuntimeError(
                    "generic fallback reached before Block-2 LayerNorm")
            centered, variance, variance_diagnostics = \
                _layernorm_variance_state(
                    residual, residual_proof, "block2_post_attention")
            variance_low, variance_high = variance.concretize()
            variance_min = variance_diagnostics["sound_variance_lower"]
            if not variance_diagnostics["domain_admissible"]:
                if experimental_post_attention_layernorm is None:
                    if experimental_layernorm_failure_capture is not None:
                        experimental_layernorm_failure_capture(
                            state=residual, proof=residual_proof,
                            label="block2_post_attention",
                            layernorm_index=delegate._layer_norm_index,
                            diagnostics=variance_diagnostics,
                            pre_reduction_state=raw_residual,
                            pre_reduction_proof=residual_proof,
                            reduction_label="b2_attention_residual")
                    return _domain_failure_report(
                        authenticated, clean_label, nominal_logits,
                        variance_diagnostics, rows, reductions, dispatch,
                        total_started)
                experiment = experimental_post_attention_layernorm(
                    residual=residual, proof=residual_proof,
                    normalizer=parameters["attention_ln"], delegate=delegate,
                    diagnostics=variance_diagnostics)
                required = {"output", "proof", "reserve", "certificate"}
                if not isinstance(experiment, dict) or not required <= set(experiment):
                    raise RuntimeError(
                        "experimental LayerNorm result fields differ")
                raw_post = experiment["output"]
                raw_post_proof = experiment["proof"]
                ln_reserve = experiment["reserve"]
                experimental_layernorm = experiment["certificate"]
                dispatch.counts["LayerNorm"] += 1
                dispatch.certificates.append(experimental_layernorm)
            else:
                delegate._hidden = residual_proof
                delegate._attention_output = residual_proof
                raw_post = dispatch.layer_norm(
                    residual, parameters["attention_ln"], "standard")
                raw_post_proof = structural.get_support(raw_post)
                ln_reserve = None
            ln_ops = 16 * 128 * (residual.num_error_terms + 1) ** 2 + 4096
            if ln_reserve is None:
                ln_reserve = sound._reserve_from_majorant(
                    sound._layernorm_majorant(
                        residual, parameters["attention_ln"]), ln_ops)
            post, post_proof = sound._inject(
                raw_post, raw_post_proof, [residual],
                "b2_post_attention_layernorm", ln_ops, measurements,
                reserve=ln_reserve)
            post, post_proof = sound._maybe_reduce(
                post, post_proof, "b2_post_attention_layernorm", reductions)
            if run_representative_mpfr:
                values = centered.zonotope_w[0, 0].detach().cpu().tolist()
                with sound._mp_context(gmpy2.RoundToNearest):
                    exact_variance = sum((sound._mp(item) * sound._mp(item)
                                          for item in values),
                                         gmpy2.mpfr(0)) / len(values)
                    exact_root = gmpy2.sqrt(sound._mp(
                        variance_low[0, 0] + LAYER_NORM_EPSILON))
                spots.extend([
                    _spot("block2_post_attention_variance_nominal",
                          exact_variance,
                          centered.zonotope_w[0, 0].square().sum()
                          / len(values), float(ln_reserve.max())),
                    _spot("block2_post_attention_sqrt_lower", exact_root,
                          torch.sqrt(
                              variance_low[0, 0] + LAYER_NORM_EPSILON),
                          float(ln_reserve.max())),
                ])
            row = _stage_measure(
                "block2_post_attention_layernorm", post, post_proof, started,
                rows)
            layernorms["post_attention"] = {
                "sound_variance_lower": (
                    experimental_layernorm["minimum_psd_lower"]
                    if experimental_layernorm is not None else variance_min),
                "generic_sound_variance_lower": variance_min,
                "sqrt_domain_margin": (
                    experimental_layernorm["minimum_psd_lower"]
                    + LAYER_NORM_EPSILON
                    if experimental_layernorm is not None
                    else variance_min + LAYER_NORM_EPSILON),
                "generator_count": row["generator_count"],
                "numerical_native_ratio": row["numerical_native_ratio"],
                "variance_diagnostics": variance_diagnostics,
                "experimental_psd_certificate": experimental_layernorm,
            }

            # FFN, ReLU, residual, and final LayerNorm.
            started = time.perf_counter()
            affine, affine_proof = sound._dense_sound(
                post, post_proof, parameters["ffn_first"], "b2_ffn_first",
                measurements)
            affine, affine_proof = sound._maybe_reduce(
                affine, affine_proof, "b2_ffn_first", reductions)
            delegate._post_attention = affine_proof
            raw_relu = dispatch.relu(affine)
            raw_relu_proof = structural.get_support(raw_relu)
            relu_ops = 8 * (affine.num_error_terms + 1) + 128
            relu, relu_proof = sound._inject(
                raw_relu, raw_relu_proof, [affine], "b2_relu", relu_ops,
                measurements, condition=2.0)
            relu, relu_proof = sound._maybe_reduce(
                relu, relu_proof, "b2_relu", reductions)
            ffn, ffn_proof = sound._dense_sound(
                relu, relu_proof, parameters["ffn_second"], "b2_ffn_second",
                measurements)
            ffn, ffn_proof = sound._maybe_reduce(
                ffn, ffn_proof, "b2_ffn_second", reductions)
            _stage_measure("block2_ffn_output", ffn, ffn_proof, started, rows)

            started = time.perf_counter()
            ffn, post_aligned, output_proof = sound._align_states(
                ffn, ffn_proof, post, post_proof, "b2_ffn_residual")
            ffn, post_aligned, output_proof = sound._maybe_reduce_pair(
                ffn, post_aligned, output_proof,
                "b2_pre_ffn_residual_pair", reductions)
            raw_output_input = ffn.add(post_aligned)
            output_input, output_input_proof = sound._inject(
                raw_output_input, output_proof, [ffn, post_aligned],
                "b2_ffn_residual", 1, measurements,
                reserve=sound._add_reserve(ffn, post_aligned))
            pre_output_reduction = output_input
            pre_output_reduction_proof = output_input_proof
            output_input, output_input_proof = sound._maybe_reduce(
                output_input, output_input_proof, "b2_ffn_residual", reductions)
            if dispatch.generic_family_invocations != 0:
                raise RuntimeError(
                    "generic fallback reached before Block-2 LayerNorm")
            centered, variance, variance_diagnostics = \
                _layernorm_variance_state(
                    output_input, output_input_proof, "block2_output")
            variance_low, variance_high = variance.concretize()
            variance_min = variance_diagnostics["sound_variance_lower"]
            if not variance_diagnostics["domain_admissible"]:
                if experimental_layernorm_failure_capture is not None:
                    experimental_layernorm_failure_capture(
                        state=output_input, proof=output_input_proof,
                        label="block2_output",
                        layernorm_index=delegate._layer_norm_index,
                        diagnostics=variance_diagnostics,
                        pre_reduction_state=pre_output_reduction,
                        pre_reduction_proof=pre_output_reduction_proof,
                        reduction_label="b2_ffn_residual")
                return _domain_failure_report(
                    authenticated, clean_label, nominal_logits,
                    variance_diagnostics, rows, reductions, dispatch,
                    total_started, experimental_layernorm)
            delegate._layer_norm_index = 6
            delegate._post_attention = output_input_proof
            delegate._relu = output_input_proof
            raw_output = dispatch.layer_norm(
                output_input, parameters["output_ln"], "standard")
            raw_output_proof = structural.get_support(raw_output)
            ln_ops = 16 * 128 * (output_input.num_error_terms + 1) ** 2 + 4096
            ln_reserve = sound._reserve_from_majorant(
                sound._layernorm_majorant(output_input, parameters["output_ln"]),
                ln_ops)
            output, output_proof = sound._inject(
                raw_output, raw_output_proof, [output_input],
                "b2_output_layernorm", ln_ops, measurements,
                reserve=ln_reserve)
            output, output_proof = sound._maybe_reduce(
                output, output_proof, "b2_output_layernorm", reductions)
            if run_representative_mpfr:
                values = centered.zonotope_w[0, 0].detach().cpu().tolist()
                with sound._mp_context(gmpy2.RoundToNearest):
                    exact_variance = sum((sound._mp(item) * sound._mp(item)
                                          for item in values),
                                         gmpy2.mpfr(0)) / len(values)
                    exact_root = gmpy2.sqrt(sound._mp(
                        variance_low[0, 0] + LAYER_NORM_EPSILON))
                spots.extend([
                    _spot("block2_output_variance_nominal", exact_variance,
                          centered.zonotope_w[0, 0].square().sum()
                          / len(values), float(ln_reserve.max())),
                    _spot("block2_output_sqrt_lower", exact_root,
                          torch.sqrt(
                              variance_low[0, 0] + LAYER_NORM_EPSILON),
                          float(ln_reserve.max())),
                ])
            output, output_proof, recenter = sound._recenter_sound(
                output, output_proof, "final_encoder_recenter", measurements,
                reductions)
            output, output_proof = sound._maybe_reduce(
                output, output_proof, "final_encoder_reduction", reductions)
            row = _stage_measure(
                "block2_final_layernorm", output, output_proof, started, rows)
            layernorms["output"] = {
                "sound_variance_lower": variance_min,
                "sqrt_domain_margin": variance_min + LAYER_NORM_EPSILON,
                "generator_count": row["generator_count"],
                "numerical_native_ratio": row["numerical_native_ratio"],
                "variance_diagnostics": variance_diagnostics,
            }

            # Exact frozen pooler and direct classifier margin.
            started = time.perf_counter()
            pooled = sound._make_like(
                output, output.zonotope_w[:, :1, :],
                *sound._ranges(output))
            pooled_proof = structural.SupportProof(
                tuple(1 if (mask & 1) else 0 for mask in output_proof.masks),
                output_proof.ids, output_proof.reasons, 1)
            structural.attach_support(pooled, pooled_proof)
            structural.validate_support(pooled, pooled_proof, token_axis=1)
            pooled_affine, pooled_affine_proof = sound._dense_sound(
                pooled, pooled_proof, parameters["pooler"], "pooler_affine",
                measurements)
            pooled_affine, pooled_affine_proof = sound._maybe_reduce(
                pooled_affine, pooled_affine_proof, "pooler_affine", reductions)
            delegate._hidden = pooled_affine_proof
            raw_pooled = dispatch.tanh(pooled_affine)
            raw_pooled_proof = structural.get_support(raw_pooled)
            tanh_ops = 16 * (pooled_affine.num_error_terms + 1) + 1024
            pooled, pooled_proof = sound._inject(
                raw_pooled, raw_pooled_proof, [pooled_affine], "pooler_tanh",
                tanh_ops, measurements, condition=2.0)
            pooled, pooled_proof = sound._maybe_reduce(
                pooled, pooled_proof, "pooler_tanh", reductions)

            other = 1 - clean_label
            classifier = SimpleNamespace(
                weight=(checkpoint["classifier.weight"][clean_label:clean_label + 1]
                        - checkpoint["classifier.weight"][other:other + 1])
                       .to(device=device, dtype=torch.float64),
                bias=(checkpoint["classifier.bias"][clean_label]
                      - checkpoint["classifier.bias"][other]).reshape(1)
                     .to(device=device, dtype=torch.float64))
            margin, margin_proof = sound._dense_sound(
                pooled, pooled_proof, classifier, "direct_classifier_margin",
                measurements)
            margin, margin_proof = sound._maybe_reduce(
                margin, margin_proof, "direct_classifier_margin", reductions)
            margin_low, margin_high = margin.concretize()
            final_lower = float(margin_low.min())
            final_upper = float(margin_high.max())
            margin_row = _stage_measure(
                "final_direct_margin", margin, margin_proof, started, rows)
            classifier_reserve = max(
                item["maximum_local_widening"] for item in measurements
                if item["label"] == "direct_classifier_margin")
            if run_representative_mpfr:
                with sound._mp_context(gmpy2.RoundToNearest):
                    exact = sound._mp(classifier.bias[0])
                    for feature in range(128):
                        exact += (sound._mp(pooled.zonotope_w[0, 0, feature])
                                  * sound._mp(classifier.weight[0, feature]))
                spots.append(_spot(
                    "direct_classifier_margin_center", exact,
                    margin.zonotope_w[0, 0, 0], classifier_reserve))

            torch.cuda.synchronize()
            if dispatch.generic_family_invocations != 0:
                raise RuntimeError("generic fallback reached in Block-2 completion")
            expected_dispatch = {
                "QK": 1, "softmax": 1, "A.V": 1, "LayerNorm": 2,
                "ReLU": 1, "tanh": 1,
            }
            if dict(dispatch.counts) != expected_dispatch:
                raise RuntimeError(
                    f"native dispatch inventory differs: {dict(dispatch.counts)}")
            if (run_representative_mpfr and (not spots or not all(
                    spot.get("one_ulp_inward_rejected") is True
                    and spot.get("state_reserve_contains_machine_error") is True
                    for spot in spots))):
                raise RuntimeError("representative MPFR/one-ULP evidence incomplete")
            if final_lower <= 0:
                raise RuntimeError(
                    f"final sound margin is not positive: {final_lower}")
            maximum_ratio = max(row["numerical_native_ratio"] for row in rows)
            maximum_reduction_inflation = max(
                (float(item["support_inflation"]) for item in reductions),
                default=0.0)
            plain_margin_lower = (
                float(sound._center_row(margin).min())
                - float(sound._native_relational_support(margin, margin_proof)))
            numerical_widening = float(
                sound._explicit_numerical_support(margin, margin_proof))
            total_seconds = time.perf_counter() - total_started
            artifact_report = {
                "schema": SCHEMA,
                "verdict": "CORET_SOUND_FP64_3L_PROPERTY_READY",
                "authenticated_input": authenticated,
                "fixture_token_ids": list(prefix.FIXTURE_TOKEN_IDS),
                "fixture_rho_hex": float(prefix.FIXTURE_RHO).hex(),
                "clean_label": clean_label, "nominal_logits": nominal_logits,
                "final_sound_margin": final_lower,
                "final_sound_margin_upper": final_upper,
                "plain_fp64_margin": plain_margin_lower,
                "numerical_widening": numerical_widening,
                "block2_layernorms": layernorms,
                "max_numerical_native_ratio": maximum_ratio,
                "final_generator_count": margin.num_error_terms,
                "final_range": [final_lower, final_upper],
                "reduction_inflation_max": maximum_reduction_inflation,
                "recenter": recenter,
                "stages": rows, "reductions": reductions,
                "mpfr_spots": spots,
                "representative_mpfr_checks_performed":
                    run_representative_mpfr,
                "dispatch_counts": dict(dispatch.counts),
                "generic_fallback_count": 0,
                "experimental_psd_layernorm": experimental_layernorm,
                "total_seconds": total_seconds,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "scientific_properties": 0, "bound_calls": 0,
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            saved = sound._save_artifact(
                output_path, OUTPUT_SCHEMA,
                {"margin": (margin, margin_proof)}, artifact_report)
            report = dict(artifact_report)
            report.update({
                "output_artifact_path": str(output_path),
                "output_artifact_schema": OUTPUT_SCHEMA,
                "output_artifact_sha256": saved["sha256"],
                "source_hashes": {
                    "runner_sha256": _sha256(Path(__file__)),
                    "sound_fp64_implementation_sha256": _sha256(
                        Path(sound.__file__)),
                },
            })
            output_report.parent.mkdir(parents=True, exist_ok=True)
            output_report.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n")
            returned = dict(report)
            returned["output_report_sha256"] = _sha256(output_report)
            return returned
    finally:
        torch.set_default_dtype(prior_dtype)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block1-artifact", type=Path, required=True)
    parser.add_argument("--block1-report", type=Path, required=True)
    parser.add_argument("--output-artifact", type=Path)
    parser.add_argument("--output-report", type=Path)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        result = authenticate(args.block1_artifact, args.block1_report)
    else:
        if args.output_artifact is None or args.output_report is None:
            parser.error("--output-artifact and --output-report are required")
        result = execute(
            args.block1_artifact, args.block1_report, args.output_artifact,
            args.output_report, args.device_index)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
