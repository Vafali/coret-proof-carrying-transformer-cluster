#!/usr/bin/env python3
"""Bounded trace from accepted A.V through the Block-0 pre-QK boundary."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch

import coret_native_semantics_production_graph_v1 as production
import coret_production_av_trace_v1 as av
import coret_production_prefix_trace_v1 as prefix
import coret_production_softmax_trace_v1 as softmax
import coret_structural_support_precise_dot_v1 as structural
from coret_trace_witness_v1 import BlobStore, canonical_bytes, seal


PURPOSE = "bounded_real_production_block0_trace"
MAXIMUM_GENERATORS = 14000
FP32_U = 2.0 ** -24
ADD_GAMMA = FP32_U / (1.0 - FP32_U)
LAYERNORM_LOCAL_RESERVE = 2.0 ** -14
RELU_LOCAL_RESERVE = 2.0 ** -20


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _parameter(state, base: str, device):
    return SimpleNamespace(
        weight=state[f"{base}.weight"].float().to(device),
        bias=state[f"{base}.bias"].float().to(device))


def _load_state(root, record, Zonotope, args):
    device = torch.device(args.device)
    weights = av._load_f32(
        root, record["producer_tensor_content_ids"]["weights"], device)
    numerical = av._load_f32(
        root, record["producer_tensor_content_ids"]["numerical_radius"], device)
    native = record["native_range_metadata"]
    if native["kind"] == "absent":
        low = high = None
    elif native["kind"] == "explicit":
        low = av._load_f32(root, native["low"], device)
        high = av._load_f32(root, native["high"], device)
    else:
        raise RuntimeError("unknown parent range representation")
    state = Zonotope(
        args=args, p=100, eps=prefix.FIXTURE_RHO,
        perturbed_word_index=prefix.FIXTURE_PERTURBED_TOKEN,
        zonotope_w=weights, error_term_range_low=low,
        error_term_range_high=high, clone=False)
    proof = av._proof_from_record(record, len(prefix.FIXTURE_TOKEN_IDS))
    structural.attach_support(state, proof)
    structural.validate_support(state, proof)
    return state, proof, numerical


def _outward(value):
    value = torch.clamp(value.double(), min=0)
    value = value * (1.0 + 2.0 ** -40) + 2.0 ** -48
    result = value.float()
    return torch.nextafter(result, torch.full_like(result, math.inf))


def _head_merge_sidecar(value):
    rows = value.shape[1]
    return value.permute(1, 2, 0, 3).reshape(rows, value.shape[2], -1)


def _pad_rows(value, rows):
    if value.shape[0] > rows:
        raise RuntimeError("cannot shrink a numerical predecessor")
    if value.shape[0] == rows:
        return value
    return torch.cat([value, torch.zeros(
        (rows - value.shape[0],) + tuple(value.shape[1:]),
        dtype=value.dtype, device=value.device)], dim=0)


def _add_sidecar(left, left_eta, right, right_eta, output):
    rows = output.zonotope_w.shape[0]
    le = _pad_rows(left_eta, rows).double()
    re = _pad_rows(right_eta, rows).double()
    lw = _pad_rows(left.zonotope_w, rows).double()
    rw = _pad_rows(right.zonotope_w, rows).double()
    machine = output.zonotope_w.double()
    exact_machine_operands = lw + rw
    local = (machine - exact_machine_operands).abs()
    local += ADD_GAMMA * (lw.abs() + rw.abs())
    return _outward(le + re + local)


def _range_abs(z):
    if z.error_term_range_low is None:
        return torch.ones(z.num_error_terms, dtype=torch.float64,
                          device=z.zonotope_w.device)
    return torch.maximum(z.error_term_range_low.double().abs(),
                         z.error_term_range_high.double().abs())


def _coordinate_numerical(z, eta):
    result = eta[0].double().clone()
    if z.num_error_terms:
        multiplier = _range_abs(z).reshape(-1, *([1] * (eta.ndim - 1)))
        result += (eta[1:].double() * multiplier).sum(dim=0)
    return result


def _center_sidecar(source, eta, centered):
    x, e, y = (source.zonotope_w.double(), eta.double(),
               centered.zonotope_w.double())
    mean_x, mean_e = x.mean(dim=-1, keepdim=True), e.mean(dim=-1, keepdim=True)
    expected = x - mean_x
    return _outward(e + mean_e + (y - expected).abs())


def _square_repeat_sidecar(centered, eta, variance):
    """Dependency-local sidecar for native self dot-product-and-repeat."""
    x, e = centered.zonotope_w.double(), eta.double()
    rows, tokens, width = x.shape
    generators = rows - 1
    result = torch.zeros_like(variance.zonotope_w, dtype=torch.float64)
    center = (2.0 * x[0].abs() * e[0] + e[0].square()).sum(
        dim=-1, keepdim=True)
    if generators:
        diagonal = (2.0 * x[1:].abs() * e[1:]
                    + e[1:].square()).sum(dim=-1)
        center += 0.5 * diagonal.sum(dim=0).unsqueeze(-1)
        retained = 2.0 * (
            x[0].abs().unsqueeze(0) * e[1:]
            + x[1:].abs() * e[0].unsqueeze(0)
            + e[0].unsqueeze(0) * e[1:]).sum(dim=-1)
        result[1:rows] = retained.unsqueeze(-1).expand(-1, -1, width)
        sum_abs = x[1:].abs().sum(dim=0)
        sum_eta = e[1:].sum(dim=0)
        fresh = (sum_abs * sum_eta + 0.5 * sum_eta.square()).sum(dim=-1)
        for token in range(tokens):
            result[rows + token, token] = fresh[token]
    result[0] = center.expand(-1, width)
    # The exact native realization discrepancy is covered locally rather than
    # charged to unrelated rows.
    result += variance.zonotope_w.double().abs() * (129.0 * FP32_U)
    return _outward(result)


def _unary_sidecar(source, eta, output, kind):
    low, high = source.concretize()
    low, high = low.double(), high.double()
    active = low != high
    if kind == "sqrt":
        sqrt_low, sqrt_high = low.sqrt(), high.sqrt()
        slope = (sqrt_high - sqrt_low) / (high - low)
        slope = torch.where(active, slope, torch.zeros_like(slope))
        critical = ((high - low) / (2.0 * (sqrt_high - sqrt_low))).square()
        intercept = sqrt_low - slope * low
        constant = 0.5 * (critical.sqrt() - slope * critical + intercept)
        fresh = 0.5 * (slope * critical - critical.sqrt() + intercept)
        point = low.sqrt()
    else:
        slope = -1.0 / high.square()
        bottom = 1.0 / high - slope * high
        top = 1.0 / low - slope * low
        constant, fresh = 0.5 * (top + bottom), 0.5 * (top - bottom)
        point = 1.0 / low
    rows = source.zonotope_w.shape[0]
    result = torch.zeros_like(output.zonotope_w, dtype=torch.float64)
    expected = source.zonotope_w.double() * slope.unsqueeze(0)
    expected[0] += constant
    expected[:, ~active] = 0
    expected[0, ~active] = point[~active]
    result[:rows] = slope.abs().unsqueeze(0) * eta.double()
    result[:rows] += (output.zonotope_w[:rows].double() - expected).abs()

    numerical = _coordinate_numerical(source, eta)
    expanded_low, expanded_high = low - numerical, high + numerical
    if kind == "sqrt":
        if bool((expanded_low <= 0).any()):
            raise RuntimeError(
                "sqrt numerical sidecar crosses zero: "
                f"native_min={float(low.min())} "
                f"numerical_max={float(numerical.max())} "
                f"expanded_min={float(expanded_low.min())}")
        critical_expanded = 1.0 / (4.0 * slope.square())
        critical_expanded = torch.maximum(
            expanded_low, torch.minimum(expanded_high, critical_expanded))
        values = torch.stack([
            expanded_low.sqrt() - slope * expanded_low,
            expanded_high.sqrt() - slope * expanded_high,
            critical_expanded.sqrt() - slope * critical_expanded])
    else:
        if bool((expanded_low <= 0).any()):
            raise RuntimeError("reciprocal numerical sidecar crosses zero")
        critical_expanded = (-1.0 / slope).sqrt()
        critical_expanded = torch.maximum(
            expanded_low, torch.minimum(expanded_high, critical_expanded))
        values = torch.stack([
            expanded_low.reciprocal() - slope * expanded_low,
            expanded_high.reciprocal() - slope * expanded_high,
            critical_expanded.reciprocal() - slope * critical_expanded])
    residual_low, residual_high = values.amin(dim=0), values.amax(dim=0)
    deficit = torch.maximum(
        torch.clamp((constant - fresh) - residual_low, min=0),
        torch.clamp(residual_high - (constant + fresh), min=0))
    active_flat = active.flatten().nonzero().flatten().tolist()
    for owner, flat in enumerate(active_flat):
        token, feature = divmod(flat, active.shape[-1])
        row = rows + owner
        result[row, token, feature] = (
            deficit[token, feature]
            + abs(float(output.zonotope_w[row, token, feature])
                  - float(fresh[token, feature])))
    result[0, ~active] += torch.maximum(
        (expanded_low.sqrt() if kind == "sqrt"
         else expanded_high.reciprocal())[~active] - point[~active],
        point[~active] - (expanded_high.sqrt() if kind == "sqrt"
                          else expanded_low.reciprocal())[~active])
    return _outward(result)


def _multiply_sidecar(left, left_eta, right, right_eta, output):
    a, b = left.zonotope_w.double(), right.zonotope_w.double()
    ea, eb = left_eta.double(), right_eta.double()
    rows = max(a.shape[0], b.shape[0])
    a, b = _pad_rows(a, rows), _pad_rows(b, rows)
    ea, eb = _pad_rows(ea, rows), _pad_rows(eb, rows)
    result = torch.zeros_like(output.zonotope_w, dtype=torch.float64)
    sensitivity = a.abs() * eb + b.abs() * ea + ea * eb
    diagonal = sensitivity[1:]
    result[0] = sensitivity[0] + 0.5 * diagonal.sum(dim=0)
    result[1:rows] = (
        a[0].abs().unsqueeze(0) * eb[1:]
        + b[1:].abs() * ea[0].unsqueeze(0)
        + ea[0].unsqueeze(0) * eb[1:]
        + b[0].abs().unsqueeze(0) * ea[1:]
        + a[1:].abs() * eb[0].unsqueeze(0)
        + eb[0].unsqueeze(0) * ea[1:])
    sum_a, sum_b = a[1:].abs().sum(dim=0), b[1:].abs().sum(dim=0)
    sum_ea, sum_eb = ea[1:].sum(dim=0), eb[1:].sum(dim=0)
    fresh = (sum_a * sum_eb + sum_b * sum_ea
             + sum_ea * sum_eb - 0.5 * diagonal.sum(dim=0))
    tokens, width = fresh.shape
    for token in range(tokens):
        for feature in range(width):
            result[rows + token * width + feature, token, feature] = fresh[
                token, feature]
    result += output.zonotope_w.double().abs() * (3.0 * FP32_U)
    return _outward(result)


def _layernorm_sidecar(source, source_eta, output, normalizer):
    width = source.word_embedding_size
    average = torch.ones((width, width), device=source.device) / width
    centered = source.add(source.matmul(average).multiply(-1.0))
    centered_eta = _center_sidecar(source, source_eta, centered)
    variance_raw = centered.square_and_sum_and_repeat()
    variance_eta = _square_repeat_sidecar(centered, centered_eta, variance_raw)
    variance = variance_raw.multiply(1.0 / width).add(1e-12)
    variance_eta = _outward(variance_eta.double() / width)
    sqrt_state = variance.sqrt()
    sqrt_eta = _unary_sidecar(variance, variance_eta, sqrt_state, "sqrt")
    reciprocal = sqrt_state.reciprocal(
        original_implementation=True, y_positive_constraint=False)
    reciprocal_eta = _unary_sidecar(
        sqrt_state, sqrt_eta, reciprocal, "reciprocal")
    expanded = centered.expand_error_terms_to_match_zonotope(reciprocal)
    expanded_eta = _pad_rows(centered_eta, reciprocal.zonotope_w.shape[0])
    product = expanded.multiply(reciprocal)
    product_eta = _multiply_sidecar(
        expanded, expanded_eta, reciprocal, reciprocal_eta, product)
    scale = normalizer.weight.detach().double()
    expected = product.zonotope_w.double() * scale
    expected[0] += normalizer.bias.detach().double()
    required = product_eta.double() * scale.abs()
    required += (output.zonotope_w.double() - expected).abs()
    required += output.zonotope_w.double().abs() * (2.0 * FP32_U)
    required[0] += LAYERNORM_LOCAL_RESERVE
    result = _outward(required)
    native_low, _ = variance.concretize()
    return result, {
        "minimum_native_variance_hex": float(native_low.min()).hex(),
        "maximum_input_numerical_hex": float(
            _coordinate_numerical(source, source_eta).max()).hex(),
        "maximum_required_hex": float(
            _coordinate_numerical(output, result).max()).hex(),
        "local_reserve_hex": float(LAYERNORM_LOCAL_RESERVE).hex(),
        "placement": "dependency_local_center_retained_fresh",
    }


def _relu_sidecar(source, source_eta, output):
    required = _coordinate_numerical(source, source_eta)
    required += RELU_LOCAL_RESERVE
    result = torch.zeros_like(output.zonotope_w, dtype=torch.float64)
    result[0] = required
    return _outward(result), {
        "maximum_input_numerical_hex": float(required.max()).hex(),
        "local_reserve_hex": float(RELU_LOCAL_RESERVE).hex(),
        "placement": "coordinatewise_center_numerical_source",
        "lipschitz": 1,
    }


def _recenter_sidecar(source, source_eta, output):
    low = source.error_term_range_low.double()
    high = source.error_term_range_high.double()
    offset, scale = (low + high) / 2.0, (high - low) / 2.0
    x, y, eta = (source.zonotope_w.double(), output.zonotope_w.double(),
                 source_eta.double())
    result = torch.zeros_like(y)
    view = (-1,) + (1,) * (x.ndim - 1)
    exact_center = x[0] + (x[1:] * offset.reshape(view)).sum(dim=0)
    result[0] = (eta[0]
                 + (eta[1:] * offset.abs().reshape(view)).sum(dim=0)
                 + (y[0] - exact_center).abs())
    exact_rows = x[1:] * scale.reshape(view)
    result[1:] = (eta[1:] * scale.abs().reshape(view)
                  + (y[1:] - exact_rows).abs())
    return _outward(result)


def _layernorm_tau(source_proof, output_proof, trace, label, width):
    inherited = len(source_proof.ids)
    return {
        "mode": "standard",
        "branch": "positive_variance_standard_layernorm",
        "epsilon_hex": float(1e-12).hex(),
        "mean_divisor": width,
        "variance_divisor": width,
        "variance_fresh_token_order": list(range(source_proof.num_tokens)),
        "sqrt_active_flat_indices": list(trace.sqrt_flat_indices),
        "reciprocal_active_flat_indices": list(trace.reciprocal_flat_indices),
        "product_active_flat_indices": list(range(source_proof.num_tokens * width)),
        "fresh_generator_ids": list(output_proof.ids[inherited:]),
        "fresh_support_masks": list(output_proof.masks[inherited:]),
        "input_support_masks": list(source_proof.masks),
        "native_boolean_order": "row_major",
        "ranged_symbol_action": "preserve_until_block_boundary",
        "label": label,
    }


def build_production_block0_trace(root, parent_av_root, Zonotope, args):
    root, parent_av_root = Path(root), Path(parent_av_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    local_parent = root / "parent"
    if parent_av_root != local_parent.resolve():
        if local_parent.exists():
            raise RuntimeError("Block-0 trace parent destination already exists")
        shutil.copytree(parent_av_root, local_parent)
        parent_av_root = local_parent.resolve()
    parent_graph_path = parent_av_root / "av_trace.json"
    parent_raw = parent_graph_path.read_bytes()
    parent_graph = json.loads(parent_raw)
    av_record = parent_graph["state_records"][-1]
    av_state, av_proof, av_eta = _load_state(
        parent_av_root, av_record, Zonotope, args)

    softmax_path = parent_av_root / parent_graph["parent_softmax"]["relative_path"]
    softmax_graph = json.loads(softmax_path.read_text())
    prefix_path = softmax_path.parent / softmax_graph["parent_prefix"]["relative_path"]
    prefix_graph = json.loads(prefix_path.read_text())
    reduced_record = prefix_graph["state_records"][3]
    residual, residual_proof, residual_eta = _load_state(
        prefix_path.parent, reduced_record, Zonotope, args)

    device = av_state.zonotope_w.device
    checkpoint = prefix._load_checkpoint()
    names = {
        "attention_output": "bert.encoder.layer.0.attention.output.dense",
        "attention_layernorm": "bert.encoder.layer.0.attention.output.LayerNorm",
        "ffn_first": "bert.encoder.layer.0.intermediate.dense",
        "ffn_second": "bert.encoder.layer.0.output.dense",
        "ffn_layernorm": "bert.encoder.layer.0.output.LayerNorm",
    }
    parameters = {key: _parameter(checkpoint, value, device)
                  for key, value in names.items()}

    delegate = structural.StructuralNativeSemanticOperators()
    delegate._hidden = residual_proof
    delegate._attention_output = av_proof
    delegate._layer_norm_index = 1
    delegate._reduction_index = 1
    dispatch = production.NativeProductionDispatch(delegate=delegate)

    states, etas, proofs, choices = [], [], [], []
    merged = av_state.remove_attention_heads_dim()
    merged_eta = _head_merge_sidecar(av_eta)
    states.append(merged); etas.append(merged_eta); proofs.append(av_proof)

    attention = merged.dense(parameters["attention_output"])
    attention_eta = prefix._affine_sidecar(
        merged, merged_eta, parameters["attention_output"])
    states.append(attention); etas.append(attention_eta); proofs.append(av_proof)

    expanded_residual = residual.expand_error_terms_to_match_zonotope(attention)
    expanded_residual_eta = _pad_rows(residual_eta, attention.zonotope_w.shape[0])
    attention_residual = attention.add(expanded_residual)
    attention_residual_eta = _add_sidecar(
        attention, attention_eta, expanded_residual, expanded_residual_eta,
        attention_residual)
    attention_residual_proof = structural._aligned_union_proof(
        residual_proof, av_proof, attention_residual.num_error_terms,
        "residual_attention_1")
    states.append(attention_residual); etas.append(attention_residual_eta)
    proofs.append(attention_residual_proof)

    post_attention = dispatch.layer_norm(
        attention_residual, parameters["attention_layernorm"], "standard")
    post_attention_proof = structural.get_support(post_attention)
    replay, ln1_membership = structural._replay_layer_norm_with_membership(
        attention_residual, parameters["attention_layernorm"], "standard",
        attention_residual_proof, "layernorm_1")
    structural._assert_layer_norm_replay_parity(
        post_attention, replay, "block0.post_attention")
    post_attention_eta, ln1_choice = _layernorm_sidecar(
        attention_residual, attention_residual_eta, post_attention,
        parameters["attention_layernorm"])
    choices.append((ln1_membership, ln1_choice))
    states.append(post_attention); etas.append(post_attention_eta)
    proofs.append(post_attention_proof)

    ffn_first = post_attention.dense(parameters["ffn_first"])
    ffn_first_eta = prefix._affine_sidecar(
        post_attention, post_attention_eta, parameters["ffn_first"])
    states.append(ffn_first); etas.append(ffn_first_eta)
    proofs.append(post_attention_proof)

    relu_lower, relu_upper = ffn_first.concretize()
    relu_active = (relu_lower * relu_upper < 0)
    relu = dispatch.relu(ffn_first)
    relu_proof = structural.get_support(relu)
    relu_eta, relu_choice = _relu_sidecar(ffn_first, ffn_first_eta, relu)
    states.append(relu); etas.append(relu_eta); proofs.append(relu_proof)

    ffn_second = relu.dense(parameters["ffn_second"])
    ffn_second_eta = prefix._affine_sidecar(
        relu, relu_eta, parameters["ffn_second"])
    states.append(ffn_second); etas.append(ffn_second_eta); proofs.append(relu_proof)

    expanded_post = post_attention.expand_error_terms_to_match_zonotope(
        ffn_second)
    expanded_post_eta = _pad_rows(post_attention_eta,
                                  ffn_second.zonotope_w.shape[0])
    ffn_residual = ffn_second.add(expanded_post)
    ffn_residual_eta = _add_sidecar(
        ffn_second, ffn_second_eta, expanded_post, expanded_post_eta,
        ffn_residual)
    ffn_residual_proof = structural._aligned_union_proof(
        post_attention_proof, relu_proof, ffn_residual.num_error_terms,
        "residual_ffn_2")
    states.append(ffn_residual); etas.append(ffn_residual_eta)
    proofs.append(ffn_residual_proof)

    block_output = dispatch.layer_norm(
        ffn_residual, parameters["ffn_layernorm"], "standard")
    block_output_proof = structural.get_support(block_output)
    replay, ln2_membership = structural._replay_layer_norm_with_membership(
        ffn_residual, parameters["ffn_layernorm"], "standard",
        ffn_residual_proof, "layernorm_2")
    structural._assert_layer_norm_replay_parity(
        block_output, replay, "block0.output")
    block_output_eta, ln2_choice = _layernorm_sidecar(
        ffn_residual, ffn_residual_eta, block_output,
        parameters["ffn_layernorm"])
    choices.append((ln2_membership, ln2_choice))
    states.append(block_output); etas.append(block_output_eta)
    proofs.append(block_output_proof)

    recentered = production._recenter_native_ranges(block_output)
    if recentered is block_output:
        raise RuntimeError("Block-0 output unexpectedly has no ranged symbols")
    structural.attach_support(recentered, block_output_proof)
    recentered_eta = _recenter_sidecar(
        block_output, block_output_eta, recentered)
    states.append(recentered); etas.append(recentered_eta)
    proofs.append(block_output_proof)

    reduced = dispatch.reduce(recentered, MAXIMUM_GENERATORS)
    reduced_proof = structural.get_support(reduced)
    if reduced.num_error_terms != recentered.num_error_terms:
        raise RuntimeError("bounded Block-0 fixture unexpectedly reduces generators")
    reduced_eta = recentered_eta.clone()
    states.append(reduced); etas.append(reduced_eta); proofs.append(reduced_proof)

    # An independent native rerun proves that instrumentation and sidecars do
    # not alter any producer tensor or topology.
    plain = structural.StructuralNativeSemanticOperators()
    plain._hidden = residual_proof; plain._attention_output = av_proof
    plain._layer_norm_index = 1; plain._reduction_index = 1
    pd = production.NativeProductionDispatch(delegate=plain)
    pc = av_state.remove_attention_heads_dim()
    pa = pc.dense(parameters["attention_output"])
    pr = residual.expand_error_terms_to_match_zonotope(pa)
    pl1 = pd.layer_norm(pa.add(pr), parameters["attention_layernorm"], "standard")
    pf1 = pl1.dense(parameters["ffn_first"])
    prelu = pd.relu(pf1)
    pr2 = pl1.expand_error_terms_to_match_zonotope(prelu)
    pl2 = pd.layer_norm(prelu.dense(parameters["ffn_second"]).add(pr2),
                         parameters["ffn_layernorm"], "standard")
    prc = production._recenter_native_ranges(pl2)
    pout = pd.reduce(prc, MAXIMUM_GENERATORS)
    plain_states = [pc, pa, pa.add(pr), pl1, pf1, prelu,
                    prelu.dense(parameters["ffn_second"]),
                    prelu.dense(parameters["ffn_second"]).add(pr2),
                    pl2, prc, pout]
    for index, (instrumented, uninstrumented) in enumerate(
            zip(states, plain_states)):
        structural._assert_layer_norm_replay_parity(
            instrumented, uninstrumented, f"block0.state.{index}")

    store = BlobStore(root)
    state_ids = [
        "b0_00_av_heads_merged", "b0_01_attention_output_projection",
        "b0_02_attention_residual", "b0_03_post_attention_layernorm",
        "b0_04_ffn_first_projection", "b0_05_relu",
        "b0_06_ffn_second_projection", "b0_07_ffn_residual",
        "b0_08_output_layernorm", "b0_09_ranges_recentered",
        "b0_10_pre_block1_qk_reduced"]
    records = [softmax._state_record(store, z, proof, identifier, eta)
               for z, eta, proof, identifier in zip(
                   states, etas, proofs, state_ids)]

    parameter_records = {}
    for key, parameter in parameters.items():
        parameter_records[key] = {
            "weight": store.f32_tensor(parameter.weight, f"{key}.weight"),
            "bias": store.f32_tensor(parameter.bias, f"{key}.bias")}

    def transition(identifier, family, inputs, output_index, tau, witness):
        return seal({
            "transition_id": identifier, "operator_family": family,
            "input_state_ids": inputs,
            "predecessor_state_ids": inputs,
            "output_state_ids": [state_ids[output_index]],
            "tau_k": tau, "operator_witness": witness})

    transitions = []
    transitions.append(transition(
        "b0_t00_attention_head_merge", "attention_head_merge",
        [av_record["state_id"]], 0,
        {"heads": 4, "head_width": 32, "permutation": [1, 2, 0, 3],
         "reshape": [1605, 4, 128], "generator_transition": "identity"},
        {"input_weights_sha256": av_record["producer_tensor_content_ids"]
         ["weights"]["sha256"]}))

    affine_specs = [
        (1, 0, "attention_output", "b0_t01_attention_output"),
        (4, 3, "ffn_first", "b0_t04_ffn_first"),
        (6, 5, "ffn_second", "b0_t06_ffn_second")]
    for out_index, source_index, key, identifier in affine_specs:
        transitions.append(transition(
            identifier, "affine_projection",
            [state_ids[source_index]], out_index,
            {"projection": key, "input_width": 128, "output_width": 128,
             "generator_transition": "ordered_identity",
             "numerical_policy": "coefficient_row_local_abs_weight_plus_gamma129",
             "gamma_129_hex": float(prefix.AFFINE_GAMMA_129).hex()},
            {**parameter_records[key],
             "input_weights_sha256": records[source_index]
             ["producer_tensor_content_ids"]["weights"]["sha256"]}))

    transitions.insert(2, transition(
        "b0_t02_attention_residual", "residual_add",
        [state_ids[1], reduced_record["state_id"]], 2,
        {"alignment": "right_zero_pad_to_left_generator_count",
         "generator_transition": "aligned_union",
         "rounding_gamma_hex": float(ADD_GAMMA).hex()},
        {"right_parent_weights_sha256": reduced_record[
            "producer_tensor_content_ids"]["weights"]["sha256"]}))

    ln1_tau = _layernorm_tau(
        attention_residual_proof, post_attention_proof, choices[0][0],
        "layernorm_1", 128)
    ln1_tau["numerical_envelope"] = choices[0][1]
    transitions.insert(3, transition(
        "b0_t03_post_attention_layernorm", "LayerNorm", [state_ids[2]], 3,
        ln1_tau, parameter_records["attention_layernorm"]))

    relu_active_flat = torch.arange(relu_active.numel(), device=device).reshape(
        relu_active.shape)[relu_active].detach().cpu().tolist()
    transitions.insert(5, transition(
        "b0_t05_relu", "ReLU", [state_ids[4]], 5,
        {"coordinate_cases_sha256": _sha(relu_active.cpu().numpy().tobytes()),
         "active_flat_indices": relu_active_flat,
         "fresh_generator_ids": list(
             relu_proof.ids[len(post_attention_proof.ids):]),
         "native_boolean_order": "row_major",
         "numerical_envelope": relu_choice}, {}))

    transitions.insert(7, transition(
        "b0_t07_ffn_residual", "residual_add",
        [state_ids[6], state_ids[3]], 7,
        {"alignment": "right_zero_pad_to_left_generator_count",
         "generator_transition": "aligned_union",
         "rounding_gamma_hex": float(ADD_GAMMA).hex()}, {}))

    ln2_tau = _layernorm_tau(
        ffn_residual_proof, block_output_proof, choices[1][0],
        "layernorm_2", 128)
    ln2_tau["numerical_envelope"] = choices[1][1]
    transitions.insert(8, transition(
        "b0_t08_output_layernorm", "LayerNorm", [state_ids[7]], 8,
        ln2_tau, parameter_records["ffn_layernorm"]))

    transitions.append(transition(
        "b0_t09_ranged_symbol_recenter", "ranged_symbol_recenter",
        [state_ids[8]], 9,
        {"action": "native_affine_recenter_to_minus1_plus1",
         "generator_transition": "ordered_identity", "deleted_indices": []},
        {"input_range_low": records[8]["native_range_metadata"]["low"],
         "input_range_high": records[8]["native_range_metadata"]["high"]}))
    transitions.append(transition(
        "b0_t10_native_generator_reduction", "generator_reduction",
        [state_ids[9]], 10,
        {"maximum": MAXIMUM_GENERATORS, "action": "identity_below_threshold",
         "input_generator_count": recentered.num_error_terms,
         "retained_indices": list(range(recentered.num_error_terms)),
         "removed_indices": [], "replacement_ids": []}, {}))

    # Repair transition order after the explicit insertions above.
    order = {f"b0_t{index:02d}": index for index in range(11)}
    transitions.sort(key=lambda item: order[item["transition_id"][:6]])
    for record, state in zip(records, states):
        record["producer_tensor_content_ids"]["weights"]["sha256"]

    transparency = {
        state_id: records[index]["producer_tensor_content_ids"]["weights"]["sha256"]
        for index, state_id in enumerate(state_ids)}
    graph = {
        "schema": prefix.SCHEMA,
        "run_manifest": seal({
            "pinned_deept_revision": prefix.PINNED_REVISION,
            "purpose": PURPOSE, "scientific_query": False,
            "bound_entrypoint_called": False,
            "prefix_stop": "block0_output_recentered_reduced_before_block1_qk",
            "parent_av_trace_sha256": _sha(parent_raw),
            "producer_transparency": transparency}),
        "parent_av": {"relative_path": "parent/av_trace.json",
                      "sha256": _sha(parent_raw)},
        "parent_reduced_state": {
            "state_id": reduced_record["state_id"],
            "weights_sha256": reduced_record[
                "producer_tensor_content_ids"]["weights"]["sha256"]},
        "graph_nodes": state_ids,
        "content_store": {"schema": prefix.BLOB_SCHEMA, "root": "."},
        "state_records": records, "transition_records": transitions,
        "final_property_record": None,
    }
    seal(graph)
    (root / "block0_trace.json").write_bytes(canonical_bytes(graph) + b"\n")
    return states, graph
