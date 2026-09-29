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
    """Run source construction and first production LayerNorm only."""
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
    traced_input, input_proof = source()
    delegate = structural.StructuralNativeSemanticOperators()
    dispatch = production.NativeProductionDispatch(delegate=delegate)
    traced_output = dispatch.layer_norm(traced_input, normalizer, "standard")
    output_proof = structural.get_support(traced_output)
    structural.validate_support(traced_output, output_proof)
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
    expected_fresh_ids = list(output_proof.ids[len(input_proof.ids):])
    if len(expected_fresh_ids) != len(fresh.masks):
        raise RuntimeError("native LayerNorm fresh identity count differs")

    if not instrument:
        return (traced_input, traced_output), None

    source_record = _state_record(
        store, traced_input, input_proof, "p0_embedding_source", 0.0)
    output_record = _state_record(
        store, traced_output, output_proof, "p1_embedding_layernorm",
        NUMERICAL_RADIUS)
    source_sha = source_record["producer_tensor_content_ids"]["weights"]["sha256"]
    output_sha = output_record["producer_tensor_content_ids"]["weights"]["sha256"]
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
    transition = seal({
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
    graph = {
        "schema": SCHEMA,
        "run_manifest": seal({
            "pinned_deept_revision": PINNED_REVISION,
            "purpose": PURPOSE,
            "scientific_query": False,
            "bound_entrypoint_called": False,
            "prefix_stop": "first_embedding_layernorm_output",
            "abstract_affine_residual_transition_count": 0,
            "producer_transparency": {
                "uninstrumented_source_sha256": source_sha,
                "instrumented_source_sha256": source_sha,
                "uninstrumented_layernorm_sha256": output_sha,
                "instrumented_layernorm_sha256": output_sha,
                "membership_replay_bitwise_identical": True,
            },
        }),
        "source_domain": source_domain,
        "graph_nodes": [source_record["state_id"], output_record["state_id"]],
        "content_store": {"schema": BLOB_SCHEMA, "root": "."},
        "state_records": [source_record, output_record],
        "transition_records": [transition],
        "final_property_record": None,
    }
    seal(graph)
    (root / "trace.json").write_bytes(canonical_bytes(graph) + b"\n")
    return (traced_input, traced_output), graph
