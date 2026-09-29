#!/usr/bin/env python3
"""Producer-only trace integration for the real frozen graph prefix.

The bounded fixture uses the frozen 3-layer checkpoint, real embedding
parameters, the pinned Zonotope source constructor, and the production native
LayerNorm dispatch.  It stops immediately after embedding LayerNorm.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import torch

import coret_deept_exact_standard_ln_adapter as adapter
import coret_native_semantics_production_graph_v1 as production
import coret_structural_support_precise_dot_v1 as structural
from coret_trace_witness_v1 import (BLOB_SCHEMA, PINNED_REVISION, SCHEMA,
                                    BlobStore, canonical_bytes, seal)
from deept_stagea_model import (CHECKPOINT_GIT_PATH, CHECKPOINT_SHA256,
                                _git_blob)


FIXTURE_TOKEN_IDS = (101, 2023, 2003, 102)
FIXTURE_PERTURBED_TOKEN = 1
FIXTURE_RHO = 1.0 / 1600.0
# Uniform checker-only coefficient envelope.  The independent checker derives
# the required interval from exact IEEE inputs; the fixture's measured maximum
# requirement is 3.812e-7, below this frozen binary radius.
NUMERICAL_RADIUS = 2.0 ** -20
PURPOSE = "bounded_real_production_prefix_trace"
PRODUCTION_MAX_ERROR_TERMS = 14000


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _ranges(z):
    if z.error_term_range_low is not None or z.error_term_range_high is not None:
        raise RuntimeError("embedding prefix unexpectedly has explicit ranges")
    return [[(-1.0).hex(), (1.0).hex()] for _ in range(z.num_error_terms)]


def _state_record(store, z, proof, state_id: str, numerical_radius: float):
    if len(proof.ids) != z.num_error_terms:
        raise RuntimeError("trace support/generator count mismatch")
    mapping = [{"native_row": index + 1, "ghost_id": identifier,
                "relation": "identity"}
               for index, identifier in enumerate(proof.ids)]
    if z.error_term_range_low is None:
        if z.error_term_range_high is not None:
            raise RuntimeError("incomplete native ranged-symbol metadata")
        native_range_metadata = {
            "kind": "absent", "low": None, "high": None}
    else:
        if z.error_term_range_high is None:
            raise RuntimeError("incomplete native ranged-symbol metadata")
        native_range_metadata = {
            "kind": "explicit",
            "low": store.f32_tensor(
                z.error_term_range_low, f"{state_id}.range_low"),
            "high": store.f32_tensor(
                z.error_term_range_high, f"{state_id}.range_high"),
        }
    value = {
        "state_id": state_id,
        "producer_tensor_content_ids": {
            "weights": store.f32_tensor(z.zonotope_w, f"{state_id}.weights"),
            "numerical_radius": store.f32_tensor(
                torch.full_like(z.zonotope_w, float(numerical_radius)),
                f"{state_id}.numerical_radius")},
        "centers": {"row": 0},
        "native_generator_coefficients": {"rows_start": 1},
        "generator_ids": list(proof.ids),
        "generator_support_masks": [int(mask) for mask in proof.masks],
        "generator_support_reasons": list(proof.reasons),
        "explicit_ranges": _ranges(z),
        "native_range_metadata": native_range_metadata,
        "dtype": "float32",
        "shape": [int(item) for item in z.zonotope_w.shape],
        "numerical_sidecar_linkage": {
            "kind": "coefficient_symmetric_radius",
            "content": "numerical_radius"},
        "ghost_state_linkage": {"ordered_native_to_ghost": mapping},
    }
    return seal(value)


def _load_checkpoint():
    raw = _git_blob(adapter.DEEPT_REPOSITORY, CHECKPOINT_GIT_PATH)
    if _sha(raw) != CHECKPOINT_SHA256:
        raise RuntimeError("frozen checkpoint identity differs")
    state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    return state


def _fixture_parameters():
    state = _load_checkpoint()
    token_ids = torch.tensor(FIXTURE_TOKEN_IDS, dtype=torch.long)
    positions = torch.arange(len(FIXTURE_TOKEN_IDS), dtype=torch.long)
    token_types = torch.zeros_like(token_ids)
    word = state["bert.embeddings.word_embeddings.weight"][token_ids].float()
    position = state["bert.embeddings.position_embeddings.weight"][positions].float()
    token_type = state["bert.embeddings.token_type_embeddings.weight"][token_types].float()
    pre = word + position + token_type
    normalizer = SimpleNamespace(
        weight=state["bert.embeddings.LayerNorm.weight"].float(),
        bias=state["bert.embeddings.LayerNorm.bias"].float())
    if pre.shape != (4, 128) or normalizer.weight.shape != (128,):
        raise RuntimeError("frozen prefix fixture architecture differs")
    return word, position, token_type, pre, normalizer


def build_production_prefix_trace(root, Zonotope, args, *, instrument=True):
    """Run the real source/LN/recenter/reduction prefix and stop pre-QK."""
    root = Path(root)
    store = BlobStore(root)
    word, position, token_type, pre, normalizer = _fixture_parameters()

    def source():
        z = Zonotope(args=args, p=100, eps=FIXTURE_RHO,
                     perturbed_word_index=FIXTURE_PERTURBED_TOKEN, value=pre)
        proof = structural.proof_from_masks(
            [structural.local_mask(FIXTURE_PERTURBED_TOKEN)] * 128,
            len(FIXTURE_TOKEN_IDS), "input_source")
        structural.attach_support(z, proof)
        return z, proof

    uninstrumented_input, _ = source()
    uninstrumented_output = uninstrumented_input.layer_norm(
        normalizer, "standard")
    uninstrumented_recentered = production._recenter_native_ranges(
        uninstrumented_output)
    uninstrumented_reduced = (
        uninstrumented_recentered.reduce_num_error_terms_box(
            PRODUCTION_MAX_ERROR_TERMS))
    traced_input, input_proof = source()
    delegate = structural.StructuralNativeSemanticOperators()
    dispatch = production.NativeProductionDispatch(delegate=delegate)
    traced_output = dispatch.layer_norm(traced_input, normalizer, "standard")
    output_proof = structural.get_support(traced_output)
    structural.validate_support(traced_output, output_proof)
    if (traced_output.error_term_range_low is not None
            or traced_output.error_term_range_high is not None):
        raise RuntimeError("bounded production prefix unexpectedly has ranges")
    traced_recentered = production._recenter_native_ranges(traced_output)
    if traced_recentered is not traced_output:
        raise RuntimeError("range-free native recenter was not identity")
    traced_reduced = dispatch.reduce(
        traced_recentered, PRODUCTION_MAX_ERROR_TERMS)
    reduced_proof = structural.get_support(traced_reduced)
    structural.validate_support(traced_reduced, reduced_proof)
    if not torch.equal(uninstrumented_input.zonotope_w,
                       traced_input.zonotope_w):
        raise RuntimeError("source instrumentation changed native coefficients")
    structural._assert_layer_norm_replay_parity(
        uninstrumented_output, traced_output, "embedding.LayerNorm")
    replay, fresh = structural._replay_layer_norm_with_membership(
        traced_input, normalizer, "standard", input_proof,
        "embedding.LayerNorm")
    structural._assert_layer_norm_replay_parity(
        traced_output, replay, "embedding.LayerNorm")
    structural._assert_layer_norm_replay_parity(
        uninstrumented_recentered, traced_recentered,
        "block0.pre_attention.recenter")
    structural._assert_layer_norm_replay_parity(
        uninstrumented_reduced, traced_reduced,
        "block0.pre_attention.reduction")
    if traced_reduced.num_error_terms != traced_recentered.num_error_terms:
        raise RuntimeError("native no-reduction branch changed generator count")
    if reduced_proof != output_proof:
        raise RuntimeError("native no-reduction branch changed support proof")
    expected_fresh_ids = list(output_proof.ids[len(input_proof.ids):])
    if len(expected_fresh_ids) != len(fresh.masks):
        raise RuntimeError("native LayerNorm fresh identity count differs")

    if not instrument:
        return (traced_input, traced_output, traced_recentered,
                traced_reduced), None

    source_record = _state_record(
        store, traced_input, input_proof, "p0_embedding_source", 0.0)
    output_record = _state_record(
        store, traced_output, output_proof, "p1_embedding_layernorm",
        NUMERICAL_RADIUS)
    recentered_record = _state_record(
        store, traced_recentered, output_proof,
        "p2_block0_ranges_recentered", NUMERICAL_RADIUS)
    reduced_record = _state_record(
        store, traced_reduced, reduced_proof,
        "p3_block0_pre_qk_reduced", NUMERICAL_RADIUS)
    source_sha = source_record["producer_tensor_content_ids"]["weights"]["sha256"]
    output_sha = output_record["producer_tensor_content_ids"]["weights"]["sha256"]
    recentered_sha = recentered_record[
        "producer_tensor_content_ids"]["weights"]["sha256"]
    reduced_sha = reduced_record[
        "producer_tensor_content_ids"]["weights"]["sha256"]
    component_records = {
        "word": store.f32_tensor(word, "embedding.word"),
        "position": store.f32_tensor(position, "embedding.position"),
        "token_type": store.f32_tensor(token_type, "embedding.token_type"),
        "gamma": store.f32_tensor(normalizer.weight, "embedding.LayerNorm.gamma"),
        "beta": store.f32_tensor(normalizer.bias, "embedding.LayerNorm.beta"),
    }
    source_domain = seal({
        "benchmark": "p100_single_token_embedding_linf",
        "p_cli": 100,
        "interpreted_domain": "Linf",
        "epsilon_hex": float(FIXTURE_RHO).hex(),
        "perturbed_token": FIXTURE_PERTURBED_TOKEN,
        "input_token_ids": list(FIXTURE_TOKEN_IDS),
        "source_symbol_ids": list(input_proof.ids),
        "source_ranges": [[(-1.0).hex(), (1.0).hex()]] * 128,
        "input_source_mask": [FIXTURE_PERTURBED_TOKEN] * 128,
        "source_coordinate_order": list(range(128)),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "model_identity": "deept_table7_stdln3_ckpt5",
        "property_identity": "bounded_prefix_tokens_101_2023_2003_102_tok1",
        "embedding_components": component_records,
        "embedding_sum_order": ["word", "position", "token_type"],
    })
    tau = {
        "mode": "standard",
        "epsilon_hex": float(1e-12).hex(),
        "mean_divisor": 128,
        "variance_divisor": 128,
        "variance_fresh_token_order": list(range(4)),
        "sqrt_active_flat_indices": list(fresh.sqrt_flat_indices),
        "reciprocal_active_flat_indices": list(
            fresh.reciprocal_flat_indices),
        "product_active_flat_indices": list(range(4 * 128)),
        "fresh_generator_ids": expected_fresh_ids,
        "fresh_support_masks": [int(mask) for mask in fresh.masks],
        "input_support_masks": [int(mask) for mask in input_proof.masks],
        "branch": "positive_variance_standard_layernorm",
        "ranged_symbol_action": "none",
        "native_boolean_order": "row_major",
    }
    layernorm_transition = seal({
        "transition_id": "prefix_t0_embedding_layernorm",
        "operator_family": "LayerNorm",
        "input_state_ids": [source_record["state_id"]],
        "output_state_ids": [output_record["state_id"]],
        "predecessor_state_id": source_record["state_id"],
        "tau_k": tau,
        "operator_witness": {
            "gamma": component_records["gamma"],
            "beta": component_records["beta"],
            "native_certificate_sha256": _sha(canonical_bytes(
                dispatch.certificates[0])),
        },
    })
    recenter_transition = seal({
        "transition_id": "prefix_t1_conditional_range_recenter",
        "operator_family": "ranged_symbol_recenter",
        "input_state_ids": [output_record["state_id"]],
        "output_state_ids": [recentered_record["state_id"]],
        "predecessor_state_id": output_record["state_id"],
        "tau_k": {
            "branch": "skip_no_explicit_ranges",
            "input_range_low_present": False,
            "input_range_high_present": False,
            "native_skip_predicate": "error_term_range_low_is_None",
            "native_return_identity": True,
            "generator_transition": "ordered_identity",
        },
        "operator_witness": {
            "input_weights_sha256": output_sha,
            "output_weights_sha256": recentered_sha,
            "recenter_map": "identity_no_explicit_ranges",
            "retained_generator_indices": list(range(
                traced_output.num_error_terms)),
            "deleted_generator_indices": [],
        },
    })
    reduction_transition = seal({
        "transition_id": "prefix_t2_native_generator_reduction",
        "operator_family": "generator_reduction",
        "input_state_ids": [recentered_record["state_id"]],
        "output_state_ids": [reduced_record["state_id"]],
        "predecessor_state_id": recentered_record["state_id"],
        "tau_k": {
            "branch": "no_reduction_input_count_le_maximum",
            "maximum_error_terms": PRODUCTION_MAX_ERROR_TERMS,
            "input_generator_count": traced_recentered.num_error_terms,
            "input_special_prefix_count": int(
                traced_recentered.num_input_error_terms_special_norm),
            "native_return_identity": True,
            "metric_policy": "not_evaluated_on_identity_branch",
            "ranking_quantities_hex": [],
            "tie_breaking": "not_applicable",
            "retained_generator_indices": list(range(
                traced_recentered.num_error_terms)),
            "removed_generator_indices": [],
            "replacement_coordinate_flat_indices": [],
            "replacement_generator_ids": [],
            "replacement_support_masks": [],
            "output_generator_count": traced_reduced.num_error_terms,
        },
        "operator_witness": {
            "metric_input_weights_sha256": recentered_sha,
            "output_weights_sha256": reduced_sha,
            "native_certificate_sha256": _sha(canonical_bytes(
                dispatch.certificates[1])),
            "replacement_construction": "none_identity_branch",
        },
    })
    graph = {
        "schema": SCHEMA,
        "run_manifest": seal({
            "pinned_deept_revision": PINNED_REVISION,
            "purpose": PURPOSE,
            "scientific_query": False,
            "bound_entrypoint_called": False,
            "prefix_stop": "block0_pre_qk_reduced_state",
            "abstract_affine_residual_transition_count": 0,
            "producer_transparency": {
                "uninstrumented_source_sha256": source_sha,
                "instrumented_source_sha256": source_sha,
                "uninstrumented_layernorm_sha256": output_sha,
                "instrumented_layernorm_sha256": output_sha,
                "uninstrumented_recenter_sha256": recentered_sha,
                "instrumented_recenter_sha256": recentered_sha,
                "uninstrumented_reduction_sha256": reduced_sha,
                "instrumented_reduction_sha256": reduced_sha,
                "membership_replay_bitwise_identical": True,
            },
        }),
        "source_domain": source_domain,
        "graph_nodes": [
            source_record["state_id"], output_record["state_id"],
            recentered_record["state_id"], reduced_record["state_id"]],
        "content_store": {"schema": BLOB_SCHEMA, "root": "."},
        "state_records": [source_record, output_record, recentered_record,
                          reduced_record],
        "transition_records": [layernorm_transition, recenter_transition,
                               reduction_transition],
        "final_property_record": None,
    }
    seal(graph)
    (root / "trace.json").write_bytes(canonical_bytes(graph) + b"\n")
    return (traced_input, traced_output, traced_recentered, traced_reduced), graph
