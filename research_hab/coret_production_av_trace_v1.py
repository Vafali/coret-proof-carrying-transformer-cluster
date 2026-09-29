#!/usr/bin/env python3
"""Bounded production trace from accepted softmax through native precise A.V."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

import coret_native_semantics_production_graph_v1 as production
import coret_production_prefix_trace_v1 as prefix
import coret_production_softmax_trace_v1 as softmax
import coret_structural_support_precise_dot_v1 as structural
from coret_native_numerical_witness_v1 import ContentAddressedWitnessStore
from coret_trace_witness_v1 import BlobStore, canonical_bytes, seal


PURPOSE = "bounded_real_production_av_trace"
ATTENTION_HEADS = 4
HEAD_WIDTH = 32
AV_LOCAL_ERROR_RESERVE = 2.0 ** -18


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load_f32(root: Path, descriptor, device):
    raw = (root / descriptor["relative_path"]).read_bytes()
    if len(raw) != descriptor["byte_count"] or _sha(raw) != descriptor["sha256"]:
        raise RuntimeError("parent content-addressed tensor differs")
    return torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(
        descriptor["shape"]).clone().to(device)


def _proof_from_record(record, tokens: int):
    proof = structural.SupportProof(
        tuple(int(value) for value in record["generator_support_masks"]),
        tuple(record["generator_ids"]),
        tuple(record["generator_support_reasons"]), tokens)
    if len(proof.ids) + 1 != record["shape"][1 if len(record["shape"]) == 4 else 0]:
        raise RuntimeError("parent proof/state generator count differs")
    return proof


def _value_parameter(device):
    state = prefix._load_checkpoint()
    parameter = SimpleNamespace(
        weight=state[
            "bert.encoder.layer.0.attention.self.value.weight"].float().to(device),
        bias=state[
            "bert.encoder.layer.0.attention.self.value.bias"].float().to(device))
    if parameter.weight.shape != (128, 128) or parameter.bias.shape != (128,):
        raise RuntimeError("frozen V projection architecture differs")
    return parameter


def _outward(value):
    value = torch.where(value > 0,
                        value * (1.0 + 2.0 ** -40) + 2.0 ** -50,
                        value)
    result = value.float()
    return torch.where(
        result > 0,
        torch.nextafter(result, torch.full_like(result, math.inf)), result)


def _av_sidecar(probability, value_transposed, probability_eta, value_eta,
                output):
    """Dependency-local upstream uncertainty for the native A.V equation."""
    device = output.zonotope_w.device
    a = probability.zonotope_w.detach().cpu().double()
    b = value_transposed.zonotope_w.detach().cpu().double()
    ea = probability_eta.detach().cpu().double()
    eb = value_eta.detach().cpu().double()
    heads, ga, rows, inner = a.shape
    gb, columns = b.shape[1], b.shape[2]
    ga -= 1
    gb -= 1
    gmin, gmax = min(ga, gb), max(ga, gb)
    result = torch.zeros(output.zonotope_w.shape, dtype=torch.float64)

    for head in range(heads):
        for query in range(rows):
            av, ae = a[head, :, query], ea[head, :, query]
            for column in range(columns):
                bv, be = b[head, :, column], eb[head, :, column]

                diagonal = (
                    av[1:1 + gmin].abs() * be[1:1 + gmin]
                    + bv[1:1 + gmin].abs() * ae[1:1 + gmin]
                    + ae[1:1 + gmin] * be[1:1 + gmin]).sum(dim=1)
                center = (
                    (av[0].abs() * be[0] + bv[0].abs() * ae[0]
                     + ae[0] * be[0]).sum()
                    + 0.5 * diagonal.sum() + AV_LOCAL_ERROR_RESERVE)
                result[head, 0, query, column] = center

                retained = torch.zeros(gmax, dtype=torch.float64)
                if gb:
                    retained[:gb] += (
                        av[0].abs().unsqueeze(0) * be[1:1 + gb]
                        + bv[1:1 + gb].abs() * ae[0].unsqueeze(0)
                        + ae[0].unsqueeze(0) * be[1:1 + gb]).sum(dim=1)
                if ga:
                    retained[:ga] += (
                        av[1:1 + ga].abs() * be[0].unsqueeze(0)
                        + bv[0].abs().unsqueeze(0) * ae[1:1 + ga]
                        + ae[1:1 + ga] * be[0].unsqueeze(0)).sum(dim=1)
                result[head, 1:1 + gmax, query, column] = retained

                sum_a = av[1:1 + ga].abs().sum(dim=0)
                sum_b = bv[1:1 + gb].abs().sum(dim=0)
                sum_ea = ae[1:1 + ga].sum(dim=0)
                sum_eb = be[1:1 + gb].sum(dim=0)
                all_pairs = (sum_a * sum_eb + sum_b * sum_ea
                             + sum_ea * sum_eb).sum()
                fresh = all_pairs - 0.5 * diagonal.sum()
                owner = (1 + gmax + head * rows * columns
                         + query * columns + column)
                result[head, owner, query, column] = fresh
    return _outward(result).to(device=device)


def build_production_av_trace(root, Zonotope, args):
    """Execute the frozen block-0 V/A.V path and stop before output projection."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    softmax_root = root / "softmax"
    softmax_states, softmax_graph = softmax.build_production_softmax_trace(
        softmax_root, Zonotope, args)
    probability = softmax_states[-1]
    probability_proof = structural.get_support(probability)
    probability_record = softmax_graph["state_records"][-1]
    probability_eta = _load_f32(
        softmax_root,
        probability_record["producer_tensor_content_ids"]["numerical_radius"],
        probability.device)

    prefix_path = softmax_root / softmax_graph["parent_prefix"]["relative_path"]
    prefix_root = prefix_path.parent
    prefix_graph = json.loads(prefix_path.read_text())
    reduced_record = prefix_graph["state_records"][3]
    reduced_weights = _load_f32(
        prefix_root, reduced_record["producer_tensor_content_ids"]["weights"],
        probability.device)
    reduced_eta = _load_f32(
        prefix_root,
        reduced_record["producer_tensor_content_ids"]["numerical_radius"],
        probability.device)
    reduced = Zonotope(
        args=args, p=100, eps=prefix.FIXTURE_RHO,
        perturbed_word_index=prefix.FIXTURE_PERTURBED_TOKEN,
        zonotope_w=reduced_weights, error_term_range_low=None,
        error_term_range_high=None, clone=False)
    reduced_proof = _proof_from_record(reduced_record, len(prefix.FIXTURE_TOKEN_IDS))
    structural.attach_support(reduced, reduced_proof)
    structural.validate_support(reduced, reduced_proof)

    parameter = _value_parameter(probability.device)
    value_projection = reduced.dense(parameter)
    value_projection_eta = prefix._affine_sidecar(
        reduced, reduced_eta, parameter)
    value_heads = value_projection.add_attention_heads_dim(ATTENTION_HEADS)
    value_heads_eta = value_projection_eta.reshape(
        value_projection_eta.shape[0], value_projection_eta.shape[1],
        ATTENTION_HEADS, HEAD_WIDTH).permute(2, 0, 1, 3).contiguous()

    numerical_store = ContentAddressedWitnessStore(root)
    delegate = structural.StructuralNativeSemanticOperators(
        numerical_witness_store=numerical_store)
    delegate._probability = probability_proof
    delegate._value = reduced_proof
    dispatch = production.NativeProductionDispatch(delegate=delegate)
    output = dispatch.attention_value(probability, value_heads)
    output_proof = structural.get_support(output)
    structural.validate_support(output, output_proof, token_axis=-2)
    value_transposed = value_heads.t()
    value_transposed_eta = value_heads_eta.transpose(-1, -2).contiguous()
    output_eta = _av_sidecar(
        probability, value_transposed, probability_eta,
        value_transposed_eta, output)

    # Repeat without witness I/O.  The authoritative output remains the first
    # native execution; this branch proves instrumentation transparency only.
    plain_value_projection = reduced.dense(parameter)
    plain_value_heads = plain_value_projection.add_attention_heads_dim(
        ATTENTION_HEADS)
    plain_delegate = structural.StructuralNativeSemanticOperators()
    plain_delegate._probability = probability_proof
    plain_delegate._value = reduced_proof
    plain_output = production.NativeProductionDispatch(
        delegate=plain_delegate).attention_value(probability, plain_value_heads)
    if not torch.equal(value_projection.zonotope_w,
                       plain_value_projection.zonotope_w):
        raise RuntimeError("V instrumentation changed projected coefficients")
    if not torch.equal(value_heads.zonotope_w, plain_value_heads.zonotope_w):
        raise RuntimeError("V instrumentation changed head mapping")
    if not torch.equal(output.zonotope_w, plain_output.zonotope_w):
        raise RuntimeError("A.V instrumentation changed native coefficients")
    if not torch.equal(output.error_term_range_low,
                       plain_output.error_term_range_low):
        raise RuntimeError("A.V instrumentation changed lower ranges")
    if not torch.equal(output.error_term_range_high,
                       plain_output.error_term_range_high):
        raise RuntimeError("A.V instrumentation changed upper ranges")
    if structural.get_support(plain_output) != output_proof:
        raise RuntimeError("A.V instrumentation changed support provenance")

    store = BlobStore(root)
    value_record = softmax._state_record(
        store, value_projection, reduced_proof,
        "a0_block0_v_projection", value_projection_eta)
    heads_record = softmax._state_record(
        store, value_heads, reduced_proof,
        "a1_block0_v_heads", value_heads_eta)
    output_record = softmax._state_record(
        store, output, output_proof,
        "a2_block0_av_output_before_projection", output_eta)
    value_weight = store.f32_tensor(
        parameter.weight, "block0.attention.value.weight")
    value_bias = store.f32_tensor(
        parameter.bias, "block0.attention.value.bias")

    certificate = dispatch.certificates[-1]
    av_witness = certificate["independent_numerical_witness"]
    softmax_output_sha = probability_record[
        "producer_tensor_content_ids"]["weights"]["sha256"]
    reduced_sha = reduced_record[
        "producer_tensor_content_ids"]["weights"]["sha256"]
    value_sha = value_record["producer_tensor_content_ids"]["weights"]["sha256"]
    heads_sha = heads_record["producer_tensor_content_ids"]["weights"]["sha256"]
    output_sha = output_record["producer_tensor_content_ids"]["weights"]["sha256"]
    fresh_count = output.num_error_terms - max(
        probability.num_error_terms, value_heads.num_error_terms)

    projection_transition = seal({
        "transition_id": "av_t0_v_affine_projection",
        "operator_family": "affine_projection",
        "input_state_ids": [reduced_record["state_id"]],
        "output_state_ids": [value_record["state_id"]],
        "predecessor_state_id": reduced_record["state_id"],
        "tau_k": {
            "projection": "V", "input_width": 128, "output_width": 128,
            "native_equation": "matmul(weight_transpose)_then_add_bias",
            "generator_transition": "ordered_identity",
            "support_transition": "token_mask_identity",
            "numerical_policy": "coefficient_row_local_abs_weight_plus_gamma129",
            "gamma_129_hex": float(prefix.AFFINE_GAMMA_129).hex(),
        },
        "operator_witness": {
            "weight": value_weight, "bias": value_bias,
            "input_weights_sha256": reduced_sha,
            "output_weights_sha256": value_sha,
        },
    })
    head_transition = seal({
        "transition_id": "av_t1_v_head_mapping",
        "operator_family": "attention_head_mapping",
        "input_state_ids": [value_record["state_id"]],
        "output_state_ids": [heads_record["state_id"]],
        "predecessor_state_id": value_record["state_id"],
        "tau_k": {
            "projection": "V", "heads": ATTENTION_HEADS,
            "head_width": HEAD_WIDTH, "reshape": [901, 4, 4, 32],
            "permutation": [2, 0, 1, 3],
            "generator_transition": "ordered_identity",
        },
        "operator_witness": {
            "input_weights_sha256": value_sha,
            "output_weights_sha256": heads_sha,
        },
    })
    av_transition = seal({
        "transition_id": "av_t2_native_precise_av",
        "operator_family": "A.V",
        "input_state_ids": [probability_record["state_id"],
                            heads_record["state_id"]],
        "output_state_ids": [output_record["state_id"]],
        "predecessor_state_ids": [probability_record["state_id"],
                                  heads_record["state_id"]],
        "tau_k": {
            "native_equation": certificate["equation"],
            "support_policy": certificate["range_logic"],
            "generator_order_policy": certificate["symbol_policy"],
            "heads": 4, "query_tokens": 4, "value_width": 32,
            "key_width": 4,
            "probability_generator_count": probability.num_error_terms,
            "value_generator_count": value_heads.num_error_terms,
            "retained_generator_count": max(
                probability.num_error_terms, value_heads.num_error_terms),
            "fresh_generator_count": fresh_count,
            "fresh_generator_ids": list(output_proof.ids[-fresh_count:]),
            "numerical_backend": "checker_only_rigorous_fp64",
            "upstream_sensitivity": "dependency_aware_factorized_O_gD",
            "sensitivity_terms": [
                "abs_probability_times_eta_value",
                "abs_value_times_eta_probability",
                "eta_probability_times_eta_value"],
            "local_error_reserve_hex": float(AV_LOCAL_ERROR_RESERVE).hex(),
            "local_error_placement": "once_in_center_coordinate",
            "range_policy": "inherit_probability_then_append_minus1_plus1",
        },
        "operator_witness": {
            "input_probability_weights_sha256": softmax_output_sha,
            "input_value_weights_sha256": heads_sha,
            "transposed_value_weights_sha256": av_witness["right"]["sha256"],
            "output_weights_sha256": output_sha,
            "softmax_range_low_sha256": probability_record[
                "native_range_metadata"]["low"]["sha256"],
            "softmax_range_high_sha256": probability_record[
                "native_range_metadata"]["high"]["sha256"],
            "structural_diagnostics": certificate["support_diagnostics"],
            "independent_numerical_witness": av_witness,
            "native_certificate_sha256": _sha(canonical_bytes(certificate)),
        },
    })

    parent_raw = (softmax_root / "softmax_trace.json").read_bytes()
    graph = {
        "schema": prefix.SCHEMA,
        "run_manifest": seal({
            "pinned_deept_revision": prefix.PINNED_REVISION,
            "purpose": PURPOSE, "scientific_query": False,
            "bound_entrypoint_called": False,
            "prefix_stop": "block0_native_precise_av_output_before_projection",
            "parent_softmax_trace_sha256": _sha(parent_raw),
            "producer_transparency": {
                "instrumented_v_projection_sha256": value_sha,
                "uninstrumented_v_projection_sha256": value_sha,
                "instrumented_v_heads_sha256": heads_sha,
                "uninstrumented_v_heads_sha256": heads_sha,
                "instrumented_av_sha256": output_sha,
                "uninstrumented_av_sha256": output_sha,
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
        "parent_softmax": {
            "relative_path": "softmax/softmax_trace.json",
            "sha256": _sha(parent_raw),
        },
        "parent_reduced_state": {
            "state_id": reduced_record["state_id"],
            "weights_sha256": reduced_sha,
        },
        "graph_nodes": [value_record["state_id"], heads_record["state_id"],
                        output_record["state_id"]],
        "content_store": {"schema": prefix.BLOB_SCHEMA, "root": "."},
        "state_records": [value_record, heads_record, output_record],
        "transition_records": [projection_transition, head_transition,
                               av_transition],
        "final_property_record": None,
    }
    seal(graph)
    (root / "av_trace.json").write_bytes(canonical_bytes(graph) + b"\n")
    return (value_projection, value_heads, output), graph
