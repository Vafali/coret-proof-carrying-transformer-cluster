#!/usr/bin/env python3
"""Bounded production trace from accepted QK through native softmax.

This module is producer-side instrumentation only.  It executes the pinned
DeepT operations unchanged, then records the discrete softmax trace and a
coefficient-indexed numerical sidecar.  The independent checker lives in a
separate, Torch-free module.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

import coret_native_semantics_production_graph_v1 as production
import coret_production_prefix_trace_v1 as prefix
import coret_structural_support_precise_dot_v1 as structural
from coret_trace_witness_v1 import BlobStore, canonical_bytes, seal


PURPOSE = "bounded_real_production_softmax_trace"
SCORE_SCALE = 1.0 / math.sqrt(32.0)
FP32_U = 2.0 ** -24


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _proof_with_softmax_fresh(source, heads: int, tokens: int):
    masks = tuple(structural.local_mask(query)
                  for _head in range(heads)
                  for query in range(tokens) for _key in range(tokens))
    masks += tuple(structural.local_mask(query)
                   for query in range(tokens)
                   for _head in range(heads) for _key in range(tokens))
    ids = tuple(f"softmax_0_fresh_{index:06d}"
                for index in range(len(masks)))
    reasons = tuple("native_row_local_softmax_topology" for _ in masks)
    return structural.SupportProof(source.masks + masks, source.ids + ids,
                                   source.reasons + reasons,
                                   source.num_tokens)


def _state_record(store, z, proof, state_id: str, numerical):
    """State serializer supporting the explicit ranges created by softmax."""
    numerical = numerical.to(device=z.zonotope_w.device,
                             dtype=z.zonotope_w.dtype)
    if numerical.shape != z.zonotope_w.shape:
        raise RuntimeError("softmax numerical sidecar shape mismatch")
    if not bool(torch.isfinite(numerical).all()) or bool((numerical < 0).any()):
        raise RuntimeError("softmax numerical sidecar is invalid")
    if len(proof.ids) != z.num_error_terms:
        raise RuntimeError("softmax proof/generator count mismatch")
    if z.error_term_range_low is None:
        if z.error_term_range_high is not None:
            raise RuntimeError("incomplete range metadata")
        ranges = [[(-1.0).hex(), (1.0).hex()]
                  for _ in range(z.num_error_terms)]
        native_ranges = {"kind": "absent", "low": None, "high": None}
    else:
        if z.error_term_range_high is None:
            raise RuntimeError("incomplete range metadata")
        low = z.error_term_range_low.detach().cpu().tolist()
        high = z.error_term_range_high.detach().cpu().tolist()
        if len(low) != z.num_error_terms or len(high) != z.num_error_terms:
            raise RuntimeError("native range count mismatch")
        ranges = [[float(a).hex(), float(b).hex()]
                  for a, b in zip(low, high)]
        native_ranges = {
            "kind": "explicit",
            "low": store.f32_tensor(z.error_term_range_low,
                                     f"{state_id}.range_low"),
            "high": store.f32_tensor(z.error_term_range_high,
                                      f"{state_id}.range_high"),
        }
    mapping = [{"native_row": index + 1, "ghost_id": identifier,
                "relation": "identity"}
               for index, identifier in enumerate(proof.ids)]
    return seal({
        "state_id": state_id,
        "producer_tensor_content_ids": {
            "weights": store.f32_tensor(z.zonotope_w,
                                         f"{state_id}.weights"),
            "numerical_radius": store.f32_tensor(
                numerical, f"{state_id}.numerical_radius"),
        },
        "generator_ids": list(proof.ids),
        "generator_support_masks": [int(value) for value in proof.masks],
        "generator_support_reasons": list(proof.reasons),
        "explicit_ranges": ranges,
        "native_range_metadata": native_ranges,
        "dtype": "float32",
        "shape": [int(value) for value in z.zonotope_w.shape],
        "numerical_sidecar_linkage": {
            "kind": "coefficient_symmetric_radius",
            "content": "numerical_radius",
            "max_radius_hex": float(numerical.max()).hex(),
        },
        "ghost_state_linkage": {"ordered_native_to_ghost": mapping},
    })


def _outward(value):
    value = torch.where(value > 0,
                        value * (1.0 + 2.0 ** -10) + 2.0 ** -40,
                        value)
    result = value.float()
    return torch.where(result > 0,
                       torch.nextafter(result, torch.full_like(result, math.inf)),
                       result)


def _score_sidecar(qk, eta):
    x = qk.zonotope_w.detach().double()
    scale = torch.tensor(SCORE_SCALE, dtype=torch.float32).double()
    machine = qk.multiply(SCORE_SCALE).zonotope_w.detach().double()
    exact = x * scale
    return _outward(eta.detach().double() * scale.abs()
                    + (machine - exact).abs())


def _residual_extrema(kind, low, high, slope):
    """Exact-real residual extrema for fixed witnessed affine slope."""
    if kind == "exp":
        candidates = [low, high]
        critical = slope.log()
        inside = (critical >= low) & (critical <= high)
        critical = torch.where(inside, critical, low)
        values = [low.exp() - slope * low,
                  high.exp() - slope * high,
                  critical.exp() - slope * critical]
    else:
        if bool((low <= 0).any()):
            raise RuntimeError("checker-expanded reciprocal domain is not positive")
        candidates = [low, high]
        critical = (-1.0 / slope).sqrt()
        inside = (critical >= low) & (critical <= high)
        critical = torch.where(inside, critical, low)
        values = [low.reciprocal() - slope * low,
                  high.reciprocal() - slope * high,
                  critical.reciprocal() - slope * critical]
    del candidates
    stacked = torch.stack(values)
    return stacked.amin(dim=0), stacked.amax(dim=0)


def _fixed_relaxation_sidecar(source_w, source_eta, output_w, slope,
                              constant, fresh, active, kind):
    """Propagate N through a fixed admissible native unary relaxation."""
    x = source_w.detach().double()
    eta = source_eta.detach().double()
    y = output_w.detach().double()
    slope = slope.detach().double()
    constant = constant.detach().double()
    fresh = fresh.detach().double()
    active = active.detach().bool()
    rows = x.shape[0]
    result = torch.zeros_like(y)
    central = slope * x
    central[0] += constant
    result[:rows] = slope.abs().unsqueeze(0) * eta
    result[:rows] += (y[:rows] - central).abs()

    # Concretized input uncertainty at each scalar coordinate.  The trace
    # fixture has implicit [-1,1] ranges before equality substitution.
    radius = x[1:].abs().sum(dim=0)
    numerical = eta.sum(dim=0)
    low = x[0] - radius - numerical
    high = x[0] + radius + numerical
    residual_low, residual_high = _residual_extrema(
        kind, low, high, slope)
    deficit = torch.maximum(
        torch.clamp((constant - fresh) - residual_low, min=0),
        torch.clamp(residual_high - (constant + fresh), min=0))
    active_flat = active.flatten().nonzero().flatten()
    for owner, flat in enumerate(active_flat.tolist()):
        coordinate = torch.unravel_index(
            torch.tensor(flat), active.shape)
        row = rows + owner
        coordinate = tuple(int(v) for v in coordinate)
        result[(row,) + coordinate] = (
            deficit[coordinate]
            + abs(float(y[(row,) + coordinate])
                  - float(fresh[coordinate])))
    return _outward(result)


def _native_softmax_replay(scores, preconstraint, Verifiers_Zonotope):
    """Replay exact pinned primitives, exposing otherwise internal states."""
    process_values = Verifiers_Zonotope.process_values
    make = Verifiers_Zonotope.make_zonotope_new_weights_same_args
    heads, rows, tokens, values = scores.zonotope_w.shape
    sums, collapsed = [], []
    exp_trace = []
    for head in range(heads):
        source = scores.zonotope_w[head:head + 1]
        vals = source.unsqueeze(-1)
        diffs = vals.repeat(1, 1, 1, 1, values).transpose(3, 4) - vals.repeat(
            1, 1, 1, 1, values)
        flat = diffs.permute(1, 0, 2, 3, 4).reshape(
            rows, tokens, values * values)
        zflat = make(flat.clone(), scores, clone=False)
        low, high = zflat.concretize()
        active = low != high
        tcrit = ((high.exp() - low.exp()) / (high - low)).log()
        tcrit[high == low] = math.inf
        neginf = tcrit == -math.inf
        tcrit[neginf] = (0.5 * low + 0.5 * high)[neginf]
        optimal = torch.min(torch.min(tcrit, low + 0.95), high)
        slope = optimal.exp()
        constant = 0.5 * (slope * (1 - optimal - high))
        constant += 0.5 * high.exp()
        fresh = 0.5 * (slope * (optimal - high - 1))
        fresh += 0.5 * high.exp()
        sum_w, collapsed_fresh = process_values(
            source, scores, 1, tokens, values,
            keep_intermediate_zonotopes=False)
        sums.append(sum_w); collapsed.append(collapsed_fresh)
        exp_trace.append((flat, low, high, active, optimal, slope,
                          constant, fresh))
    sum_w = torch.cat(sums, dim=0)
    collapsed_fresh = torch.cat(collapsed, dim=0)
    boxes = torch.zeros(heads * tokens * values, heads, tokens, values,
                        device=scores.device)
    indices = torch.arange(heads * tokens * values, device=scores.device)
    mask = torch.ones_like(collapsed_fresh, dtype=torch.bool)
    boxes[indices, mask] = collapsed_fresh[mask]
    boxes = boxes.permute(1, 0, 2, 3)
    denominator = make(torch.cat([sum_w, boxes], dim=1), scores, clone=False)
    replay_pre = denominator.reciprocal(
        original_implementation=False, y_positive_constraint=False)
    if not torch.equal(replay_pre.zonotope_w, preconstraint.zonotope_w):
        raise RuntimeError("native softmax replay differs before equality")
    return denominator, exp_trace


def _denominator_sidecar(scores, score_eta, denominator, exp_trace):
    heads, rows, tokens, values = scores.zonotope_w.shape
    old_rows = rows
    result = torch.zeros_like(denominator.zonotope_w, dtype=torch.float64)
    for head, trace in enumerate(exp_trace):
        flat, _low, _high, active, _optimal, slope, constant, fresh = trace
        eta = score_eta[head].detach().double()
        # d(q,i,j)=s(q,j)-s(q,i); the diagonal is the exact zero identity.
        eta_diff = eta.unsqueeze(-2) + eta.unsqueeze(-1)
        diagonal = torch.eye(values, dtype=torch.bool,
                             device=eta.device).view(1, 1, values, values)
        eta_diff = eta_diff.masked_fill(diagonal, 0)
        eta_flat = eta_diff.reshape(rows, tokens, values * values)
        # Reconstruct native raw exp weights for the sidecar obligation.
        exp_old = flat.detach().double() * slope.detach().double().unsqueeze(0)
        exp_old[0] += constant.detach().double()
        equal = ~active
        exp_old[0, equal] = trace[1][equal].double().exp()
        exp_old[1:, equal] = 0
        full_shape = (rows + int(active.sum()), tokens, values * values)
        exp_full = torch.zeros(full_shape, dtype=torch.float64,
                               device=flat.device)
        exp_full[:rows] = exp_old
        active_flat = active.flatten().nonzero().flatten()
        for owner, index in enumerate(active_flat.tolist()):
            coordinate = torch.unravel_index(torch.tensor(index), active.shape)
            coordinate = tuple(int(v) for v in coordinate)
            exp_full[(rows + owner,) + coordinate] = fresh[coordinate]
        exp_eta = _fixed_relaxation_sidecar(
            flat, eta_flat, exp_full, slope, constant, fresh, active, "exp").double()
        # Old rows and their sidecars are summed over j in native order.
        old = exp_full[:rows].reshape(rows, tokens, values, values)
        old_eta = exp_eta[:rows].reshape(rows, tokens, values, values)
        central_sum = old.sum(dim=-1)
        stored = denominator.zonotope_w[head, :rows].double()
        result[head, :rows] = (old_eta.sum(dim=-1)
                               + (stored - central_sum).abs())
        # Native collapses all exp fresh terms for a fixed (head,q,i) into
        # one coordinate-fresh row, preserving head/query/key order.
        fresh_values = fresh.reshape(tokens, values, values)
        fresh_eta = exp_eta[rows:]
        for query in range(tokens):
            for key in range(values):
                owner = head * tokens * values + query * values + key
                row = rows + owner
                indices = [query * values * values + key * values + j
                           for j in range(values) if active[query, key * values + j]]
                eta_sum = sum(float(fresh_eta[local, query,
                                                key * values + j])
                              for local, j in enumerate([])) if False else 0.0
                # Map active flat positions to their compact native row.
                compact = {flat_index: pos for pos, flat_index in
                           enumerate(active.flatten().nonzero().flatten().tolist())}
                eta_sum = sum(float(exp_eta[rows + compact[index], query,
                                            key * values + (index % values)])
                              for index in indices)
                central = fresh_values[query, key].sum().double()
                stored_value = denominator.zonotope_w[
                    head, row, query, key].double()
                result[head, row, query, key] = (
                    eta_sum + (stored_value - central).abs())
    return _outward(result)


def _reciprocal_parameters(denominator):
    low, high = denominator.concretize()
    active = low != high
    mean_slope = (high.reciprocal() - low.reciprocal()) / (high - low)
    critical = (-mean_slope.reciprocal()).sqrt()
    optimal = torch.max(critical, high / 2.0 + 0.01)
    slope = -optimal.reciprocal().square()
    intercept = low.reciprocal() - slope * low
    constant = 0.5 * (optimal.reciprocal()
                      - slope * optimal + intercept)
    fresh = 0.5 * (slope * optimal
                   - optimal.reciprocal() + intercept)
    return low, high, active, optimal, slope, constant, fresh


def _equality_trace_and_sidecar(pre, pre_eta, output, Verifiers_Zonotope):
    weights = pre.zonotope_w.permute(1, 0, 2, 3).reshape(
        pre.zonotope_w.shape[1], -1, pre.zonotope_w.shape[3])
    eta = pre_eta.permute(1, 0, 2, 3).reshape_as(weights)
    left = weights[:, :, 0]
    right = -weights[:, :, 1:].sum(dim=-1); right[0] += 1
    differences = left - right
    initial = Verifiers_Zonotope.get_last_nonzero_index_per_row(
        differences.t()).squeeze(0)
    fix = initial != 0
    constraints = differences[:, fix]
    denominator_pivot = initial[fix]
    denominator = constraints.gather(
        0, denominator_pivot.unsqueeze(0)).squeeze(0)
    left_fix = left[:, fix]
    left_pivot = left_fix.gather(
        0, denominator_pivot.unsqueeze(0)).squeeze(0)
    constants = left_fix - left_pivot * constraints / denominator
    linear = constraints / denominator
    optimal, removed = Verifiers_Zonotope.find_values_that_minimizes_width(
        constants[1:], linear[1:], pre.num_input_error_terms_special_norm)
    optimal = optimal.reshape(-1)
    removed = (removed + 1).reshape(-1)
    # Capture the exact native three alpha2 iterations.
    iterations = []
    low = high = None
    for _ in range(3):
        low, high = Verifiers_Zonotope.alpha2(
            constraints, low, high,
            pre.num_input_error_terms_special_norm)
        iterations.append((low.clone(), high.clone()))
    if not torch.equal(low, output.error_term_range_low):
        raise RuntimeError("replayed equality lower ranges differ")
    if not torch.equal(high, output.error_term_range_high):
        raise RuntimeError("replayed equality upper ranges differ")

    # Coefficientwise interval sensitivity through the witnessed substitution.
    # The proof-only loop is scalar-heavy; move its tiny metadata/state view to
    # CPU once rather than introducing tens of thousands of CUDA synchronizes.
    weights_cpu = weights.detach().cpu().double()
    eta_cpu = eta.detach().cpu().double()
    output_eta_flat = torch.zeros_like(weights_cpu)
    output_weights = output.zonotope_w.permute(1, 0, 2, 3).reshape_as(
        weights).detach().cpu().double()
    d = differences.detach().cpu().double()
    left_cpu = left.detach().cpu().double()
    optimal_cpu = optimal.detach().cpu().double()
    for equation_index, equation in enumerate(fix.nonzero().flatten().tolist()):
        pivot = int(removed[equation_index])
        dk = float(d[pivot, equation])
        if dk == 0:
            raise RuntimeError("equality pivot is exactly zero")
        for row in range(weights.shape[0]):
            dr = float(d[row, equation])
            p = -dr / dk
            for key in range(weights.shape[-1]):
                x = float(weights_cpu[row, equation, key])
                xe = float(eta_cpu[row, equation, key])
                if key == 0:
                    c = float(left_cpu[pivot, equation]
                              - optimal_cpu[equation_index])
                    ce = float(eta_cpu[pivot, equation, 0])
                else:
                    c = float(weights_cpu[pivot, equation, key])
                    ce = float(eta_cpu[pivot, equation, key])
                central = x + p * c
                # The native equality substitution is a witnessed fixed
                # linear map on the machine affine state.  Checker-only N
                # sources are propagated through that map; they do not
                # perturb or reselect the native pivot.
                required = xe + abs(p) * ce
                actual = float(output_weights[row, equation, key])
                output_eta_flat[row, equation, key] = required + abs(actual - central)
    # Equations already exact are unchanged.
    fix_cpu = fix.detach().cpu()
    output_eta_flat[:, ~fix_cpu] = eta_cpu[:, ~fix_cpu]
    result = output_eta_flat.reshape(
        output.zonotope_w.shape[1], output.zonotope_w.shape[0],
        output.zonotope_w.shape[2], output.zonotope_w.shape[3]).permute(
            1, 0, 2, 3).to(output.device)
    return {
        "initial_pivot_indices": initial,
        "equations_to_fix": fix,
        "denominator_pivot_indices": denominator_pivot,
        "removed_generator_indices": removed,
        "optimal_values": optimal,
        "range_iterations": iterations,
        "constraints": constraints,
    }, _outward(result)


def build_production_softmax_trace(root, Zonotope, args):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    args.batch_softmax_computation = True
    args.keep_intermediate_zonotopes = False
    prefix_root = root / "prefix"
    states, prefix_graph = prefix.build_production_prefix_trace(
        prefix_root, Zonotope, args, instrument=True)
    qk = states[-1]
    qk_proof = structural.get_support(qk)
    qk_eta_descriptor = prefix_graph["state_records"][-1][
        "producer_tensor_content_ids"]["numerical_radius"]
    qk_eta_raw = (prefix_root / qk_eta_descriptor["relative_path"]).read_bytes()
    qk_eta = torch.frombuffer(bytearray(qk_eta_raw), dtype=torch.float32).reshape(
        qk.zonotope_w.shape).clone().to(qk.device)

    scores = qk.multiply(SCORE_SCALE)
    structural.attach_support(scores, qk_proof)
    score_eta = _score_sidecar(qk, qk_eta)

    # Authoritative constrained and unconstrained native executions.
    delegate = structural.StructuralNativeSemanticOperators()
    delegate._score = qk_proof
    dispatch = production.NativeProductionDispatch(delegate=delegate)
    output = dispatch.softmax(scores, no_constraints=False)
    output_proof = structural.get_support(output)
    structural.validate_support(output, output_proof, token_axis=-2)
    plain_delegate = structural.StructuralNativeSemanticOperators()
    plain_delegate._score = qk_proof
    plain_dispatch = production.NativeProductionDispatch(delegate=plain_delegate)
    preconstraint = plain_dispatch.softmax(scores, no_constraints=True)
    pre_proof = structural.get_support(preconstraint)

    import Verifiers.Zonotope as native
    denominator, exp_trace = _native_softmax_replay(
        scores, preconstraint, native)
    exp_count = 4 * 4 * 4
    exp_masks = tuple(structural.local_mask(query) for _head in range(4)
                      for query in range(4) for _key in range(4))
    denominator_proof = structural.SupportProof(
        qk_proof.masks + exp_masks,
        qk_proof.ids + tuple(f"softmax_0_fresh_{i:06d}"
                             for i in range(exp_count)),
        qk_proof.reasons + tuple("native_row_local_softmax_topology"
                                 for _ in range(exp_count)), 4)
    if tuple(pre_proof.ids) != tuple(output_proof.ids):
        raise RuntimeError("softmax constraint changed generator identity")
    expected_proof = _proof_with_softmax_fresh(qk_proof, 4, 4)
    if pre_proof != expected_proof or output_proof != expected_proof:
        raise RuntimeError("softmax support topology differs")

    denominator_eta = _denominator_sidecar(
        scores, score_eta, denominator, exp_trace)
    denominator_without_heads = denominator.remove_attention_heads_dim(
        clone=True)
    low, high, active, optimal, slope, constant, fresh = (
        _reciprocal_parameters(denominator_without_heads))
    pre_eta = _fixed_relaxation_sidecar(
        denominator.zonotope_w.permute(1, 2, 0, 3).reshape(
            denominator.zonotope_w.shape[1], 4, 16),
        denominator_eta.permute(1, 2, 0, 3).reshape(
            denominator_eta.shape[1], 4, 16),
        preconstraint.zonotope_w.permute(1, 2, 0, 3).reshape(
            preconstraint.zonotope_w.shape[1], 4, 16),
        slope, constant, fresh, active, "reciprocal")
    pre_eta = pre_eta.reshape(
        preconstraint.zonotope_w.shape[1], 4, 4, 4).permute(2, 0, 1, 3)
    equality, output_eta = _equality_trace_and_sidecar(
        preconstraint, pre_eta, output, native)

    # A second authoritative execution without witness I/O proves transparency.
    repeat_delegate = structural.StructuralNativeSemanticOperators()
    repeat_delegate._score = qk_proof
    repeat = production.NativeProductionDispatch(delegate=repeat_delegate).softmax(
        scores, no_constraints=False)
    if not torch.equal(repeat.zonotope_w, output.zonotope_w):
        raise RuntimeError("softmax instrumentation changed coefficients")
    if not torch.equal(repeat.error_term_range_low,
                       output.error_term_range_low):
        raise RuntimeError("softmax instrumentation changed lower ranges")
    if not torch.equal(repeat.error_term_range_high,
                       output.error_term_range_high):
        raise RuntimeError("softmax instrumentation changed upper ranges")

    store = BlobStore(root)
    score_record = _state_record(
        store, scores, qk_proof, "s0_block0_scaled_scores", score_eta)
    denom_record = _state_record(
        store, denominator, denominator_proof,
        "s1_block0_softmax_denominator", denominator_eta)
    pre_record = _state_record(
        store, preconstraint, pre_proof,
        "s2_block0_softmax_preconstraint", pre_eta)
    output_record = _state_record(
        store, output, output_proof,
        "s3_block0_softmax_output_before_av", output_eta)

    tensor = lambda value, label: store.f32_tensor(value, label)
    exp_records = []
    for head, (flat, low_e, high_e, active_e, optimal_e, slope_e,
               constant_e, fresh_e) in enumerate(exp_trace):
        exp_records.append({
            "head": head,
            "input_differences": tensor(flat, f"softmax.exp{head}.diffs"),
            "lower": tensor(low_e, f"softmax.exp{head}.lower"),
            "upper": tensor(high_e, f"softmax.exp{head}.upper"),
            "active_flat_indices": active_e.flatten().nonzero().flatten().tolist(),
            "t_opt": tensor(optimal_e, f"softmax.exp{head}.t_opt"),
            "slope": tensor(slope_e, f"softmax.exp{head}.slope"),
            "constant": tensor(constant_e, f"softmax.exp{head}.constant"),
            "fresh": tensor(fresh_e, f"softmax.exp{head}.fresh"),
        })
    equality_record = {
        "initial_pivot_indices": equality["initial_pivot_indices"].tolist(),
        "equations_to_fix": equality["equations_to_fix"].nonzero().flatten().tolist(),
        "denominator_pivot_indices": equality[
            "denominator_pivot_indices"].tolist(),
        "removed_generator_indices": equality[
            "removed_generator_indices"].tolist(),
        "optimal_values": tensor(equality["optimal_values"],
                                   "softmax.equality.optimal"),
        "constraints": tensor(equality["constraints"],
                               "softmax.equality.constraints"),
        "range_iterations": [{
            "low": tensor(lo, f"softmax.equality.range{index}.low"),
            "high": tensor(hi, f"softmax.equality.range{index}.high"),
        } for index, (lo, hi) in enumerate(equality["range_iterations"])],
    }

    qk_sha = prefix_graph["state_records"][-1][
        "producer_tensor_content_ids"]["weights"]["sha256"]
    transitions = [
        seal({
            "transition_id": "softmax_t0_score_scaling",
            "operator_family": "attention_score_scaling",
            "input_state_ids": [prefix_graph["graph_nodes"][-1]],
            "output_state_ids": [score_record["state_id"]],
            "predecessor_state_id": prefix_graph["graph_nodes"][-1],
            "tau_k": {
                "scale_binary64_hex": float(SCORE_SCALE).hex(),
                "scale_float32_hex": float(torch.tensor(SCORE_SCALE)).hex(),
                "head_width": 32,
                "generator_transition": "ordered_identity",
                "numerical_policy": "coefficient_local_fixed_scalar",
            },
            "operator_witness": {
                "input_weights_sha256": qk_sha,
                "output_weights_sha256": score_record[
                    "producer_tensor_content_ids"]["weights"]["sha256"],
            },
        }),
        seal({
            "transition_id": "softmax_t1_native_relational_softmax",
            "operator_family": "native_relational_softmax",
            "input_state_ids": [score_record["state_id"]],
            "output_state_ids": [denom_record["state_id"],
                                 pre_record["state_id"],
                                 output_record["state_id"]],
            "predecessor_state_id": score_record["state_id"],
            "tau_k": {
                "equation": "1/sum_j(exp(score_j-score_i))",
                "batch_softmax_computation": True,
                "heads": 4, "queries": 4, "keys": 4,
                "exp_relaxation": "pinned_minimal_area_v2",
                "exp_boolean_order": "head_query_i_j_row_major",
                "exp_collapsed_fresh_count": exp_count,
                "reciprocal_relaxation": "pinned_new_reciprocal",
                "reciprocal_boolean_order": "query_head_key_row_major",
                "reciprocal_fresh_count": exp_count,
                "numerator_policy": "exact_constant_one_new_softmax_no_multiply",
                "equality_branch": "native_sum_constraint_applied",
                "equality_initial_pivot_policy": "last_nonzero_generator_row",
                "equality_removal_policy": "native_width_minimizing_binary_search",
                "equality_tie_breaking": "left_candidate_on_equal_width",
                "range_iterations": 3,
                "recenter_action": "none_until_downstream_boundary",
                "numerical_policy": "coefficient_dependency_aware_fixed_relaxation",
                "exp_generator_ids": list(expected_proof.ids[
                    len(qk_proof.ids):len(qk_proof.ids) + exp_count]),
                "reciprocal_generator_ids": list(expected_proof.ids[
                    len(qk_proof.ids) + exp_count:]),
            },
            "operator_witness": {
                "input_weights_sha256": score_record[
                    "producer_tensor_content_ids"]["weights"]["sha256"],
                "denominator_weights_sha256": denom_record[
                    "producer_tensor_content_ids"]["weights"]["sha256"],
                "preconstraint_weights_sha256": pre_record[
                    "producer_tensor_content_ids"]["weights"]["sha256"],
                "output_weights_sha256": output_record[
                    "producer_tensor_content_ids"]["weights"]["sha256"],
                "exp_heads": exp_records,
                "reciprocal": {
                    "lower": tensor(low, "softmax.reciprocal.lower"),
                    "upper": tensor(high, "softmax.reciprocal.upper"),
                    "active_flat_indices": active.flatten().nonzero().flatten().tolist(),
                    "t_opt": tensor(optimal, "softmax.reciprocal.t_opt"),
                    "slope": tensor(slope, "softmax.reciprocal.slope"),
                    "constant": tensor(constant, "softmax.reciprocal.constant"),
                    "fresh": tensor(fresh, "softmax.reciprocal.fresh"),
                },
                "equality": equality_record,
                "native_certificate_sha256": _sha(canonical_bytes(
                    dispatch.certificates[-1])),
            },
        }),
    ]
    prefix_trace_raw = (prefix_root / "trace.json").read_bytes()
    graph = {
        "schema": prefix.SCHEMA,
        "run_manifest": seal({
            "pinned_deept_revision": prefix.PINNED_REVISION,
            "purpose": PURPOSE,
            "scientific_query": False,
            "bound_entrypoint_called": False,
            "prefix_stop": "block0_native_softmax_output_before_av",
            "parent_prefix_trace_sha256": _sha(prefix_trace_raw),
            "producer_transparency": {
                "instrumented_softmax_sha256": output_record[
                    "producer_tensor_content_ids"]["weights"]["sha256"],
                "uninstrumented_softmax_sha256": output_record[
                    "producer_tensor_content_ids"]["weights"]["sha256"],
                "instrumented_range_low_sha256": output_record[
                    "native_range_metadata"]["low"]["sha256"],
                "uninstrumented_range_low_sha256": output_record[
                    "native_range_metadata"]["low"]["sha256"],
                "instrumented_range_high_sha256": output_record[
                    "native_range_metadata"]["high"]["sha256"],
                "uninstrumented_range_high_sha256": output_record[
                    "native_range_metadata"]["high"]["sha256"],
            },
        }),
        "parent_prefix": {"relative_path": "prefix/trace.json",
                           "sha256": _sha(prefix_trace_raw)},
        "graph_nodes": [record["state_id"] for record in
                        (score_record, denom_record, pre_record, output_record)],
        "content_store": {"schema": prefix.BLOB_SCHEMA, "root": "."},
        "state_records": [score_record, denom_record, pre_record, output_record],
        "transition_records": transitions,
        "final_property_record": None,
    }
    seal(graph)
    (root / "softmax_trace.json").write_bytes(canonical_bytes(graph) + b"\n")
    return (scores, denominator, preconstraint, output), graph
