from __future__ import annotations

import io
import subprocess
import sys
import tarfile
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research_hab"))

import coret_native_semantics_checker_v1 as native_checker
import coret_structural_support_precise_dot_v1 as structural


@pytest.fixture(scope="module")
def native_context():
    repository = REPO / "research_hab/public_benchmarks/DeepT"
    revision = structural.native_proof.PINNED_REVISION
    with tempfile.TemporaryDirectory(prefix="coret_ln_native_test_") as temporary:
        packed = subprocess.check_output([
            "git", "-C", str(repository), "archive", "--format=tar", revision,
            "Robustness-Verification-for-Transformers/Verifiers",
        ])
        with tarfile.open(fileobj=io.BytesIO(packed), mode="r:") as archive:
            archive.extractall(temporary, filter="data")
        verifier_root = (Path(temporary)
                         / "Robustness-Verification-for-Transformers/Verifiers")
        prior = sys.modules.get("Verifiers")
        package = types.ModuleType("Verifiers")
        package.__path__ = [str(verifier_root)]
        sys.modules["Verifiers"] = package
        try:
            from Verifiers.Zonotope import Zonotope

            args = SimpleNamespace(
                perturbed_words=1, attack_type="lp",
                device=torch.device("cpu"), cpu=True, all_words=False,
                num_input_error_terms=1,
                use_dot_product_variant3=False,
                use_other_dot_product_ordering=False,
                concretize_special_norm_error_together=False,
            )
            yield Zonotope, args
        finally:
            for name in tuple(sys.modules):
                if name == "Verifiers" or name.startswith("Verifiers."):
                    del sys.modules[name]
            if prior is not None:
                sys.modules["Verifiers"] = prior


def make_state(native_context, support_mask: int):
    Zonotope, args = native_context
    center = torch.tensor([
        [-.8, -.15, .35, 1.1],
        [.55, -.45, .85, -.05],
        [-.25, .7, -.6, .42],
    ], dtype=torch.float32)
    weights = torch.zeros((2, 3, 4), dtype=torch.float32)
    weights[0] = center
    weights[1, 0] = torch.tensor([.01, -.01, .005, -.005])
    state = Zonotope(
        args=args, p=100, eps=.01, perturbed_word_index=0,
        zonotope_w=weights)
    proof = structural.SupportProof(
        (support_mask,), ("source_000000",),
        ("fixture_provenance_may_conservatively_cover_zero_coefficients",), 3)
    structural.attach_support(state, proof)
    return state, proof


def normalizer():
    return torch.nn.LayerNorm(4, eps=1e-12)


def test_legacy_all_active_topology_stays_on_original_path(native_context):
    state, _ = make_state(native_context, structural.local_mask(0))
    authoritative = state.layer_norm(normalizer(), "standard")
    result = structural.StructuralNativeSemanticOperators().layer_norm(
        state, normalizer(), "standard")
    assert torch.equal(result.value.zonotope_w, authoritative.zonotope_w)
    assert "support_transition_trace" not in result.certificate
    assert result.certificate["native_result_unchanged"] is True
    assert result.certificate["generic_semantic_remainder_used"] is False
    assert native_checker.check_common(result.certificate, "LayerNorm")


def test_singleton_sqrt_and_reciprocal_membership_is_exact(native_context):
    state, _ = make_state(native_context, structural.dense_mask(3))
    authoritative = state.layer_norm(normalizer(), "standard")
    result = structural.StructuralNativeSemanticOperators().layer_norm(
        state, normalizer(), "standard")
    trace = result.certificate["support_transition_trace"]
    assert result.value is not authoritative
    assert torch.equal(result.value.zonotope_w, authoritative.zonotope_w)
    assert trace["legacy_fresh_count"] == 39
    assert trace["native_fresh_count"] == 23
    assert trace["variance_fresh_count"] == 3
    assert trace["sqrt_flat_indices"] == [0, 1, 2, 3]
    assert trace["reciprocal_flat_indices"] == [0, 1, 2, 3]
    assert trace["final_product_fresh_count"] == 12
    assert trace["authoritative_result_retained"] is True
    assert trace["replay_bitwise_identical"] is True
    assert result.certificate["support_proof"]["validated"] is True


def test_mixed_boolean_membership_preserves_native_flat_order():
    predicate = torch.tensor([
        [False, True, False],
        [True, False, True],
    ])
    masks, flat = structural._native_boolean_membership(
        predicate, 2, 3, structural.dense_mask(2), "fixture")
    assert flat == (1, 3, 5)
    assert masks == (
        structural.local_mask(0),
        structural.local_mask(1),
        structural.local_mask(1),
    )


def test_sqrt_and_reciprocal_predicates_are_independent():
    sqrt = torch.tensor([[True, False], [False, False]])
    reciprocal = torch.tensor([[False, False], [False, True]])
    sqrt_masks, sqrt_flat = structural._native_boolean_membership(
        sqrt, 2, 2, structural.dense_mask(2), "sqrt")
    reciprocal_masks, reciprocal_flat = structural._native_boolean_membership(
        reciprocal, 2, 2, structural.dense_mask(2), "reciprocal")
    assert sqrt_flat == (0,)
    assert reciprocal_flat == (3,)
    assert sqrt_masks == (structural.local_mask(0),)
    assert reciprocal_masks == (structural.local_mask(1),)


def test_authoritative_replay_parity_rejects_any_bit_change(native_context):
    state, _ = make_state(native_context, structural.dense_mask(3))
    authoritative = state.layer_norm(normalizer(), "standard")
    replay, _ = structural._replay_layer_norm_with_membership(
        state, normalizer(), "standard", structural.get_support(state),
        "fixture")
    structural._assert_layer_norm_replay_parity(
        authoritative, replay, "fixture")
    replay.zonotope_w[0, 0, 0] = torch.nextafter(
        replay.zonotope_w[0, 0, 0], torch.tensor(float("inf")))
    with pytest.raises(RuntimeError, match="replay coefficients differ"):
        structural._assert_layer_norm_replay_parity(
            authoritative, replay, "fixture")


def test_input_proof_count_mismatch_fails_before_native(native_context):
    state, _ = make_state(native_context, structural.local_mask(0))
    state._coret_token_support = structural.proof_from_masks(
        [1, 1], 3, "bad")
    with pytest.raises(RuntimeError, match="input support/generator count mismatch"):
        structural.StructuralNativeSemanticOperators().layer_norm(
            state, normalizer(), "standard")


def test_mutated_traced_membership_is_rejected(native_context):
    state, _ = make_state(native_context, structural.dense_mask(3))
    result = structural.StructuralNativeSemanticOperators().layer_norm(
        state, normalizer(), "standard")
    proof = structural.get_support(result.value)
    present = result.value.zonotope_w[1:].ne(0).any(dim=2)
    generator, token = next(
        (int(g), int(t)) for g, t in present.nonzero().tolist())
    wrong_token = (token + 1) % proof.num_tokens
    masks = list(proof.masks)
    masks[generator] = structural.local_mask(wrong_token)
    corrupted = structural.SupportProof(
        tuple(masks), proof.ids, proof.reasons, proof.num_tokens)
    with pytest.raises(RuntimeError, match="outside proven support"):
        structural.validate_support(result.value, corrupted)


def test_reduction_ids_remain_prefix_of_layernorm_ids():
    source = structural.SupportProof(
        (1, 2), ("retained_source", "reduction_0_box_000000"),
        ("retained", "native_coordinate_box_generator"), 2)
    fresh_masks = (1, 2)
    fresh_reasons = ("sqrt", "product")
    output = SimpleNamespace(
        num_error_terms=4, num_words=2, word_embedding_size=2)
    proof = structural.StructuralNativeSemanticOperators()._layer_norm_output(
        source, output, "layernorm_fixture", fresh_masks, fresh_reasons)
    assert proof.ids[:2] == source.ids
    assert proof.reasons[:2] == source.reasons
    assert proof.ids[2:] == (
        "layernorm_fixture_fresh_000000",
        "layernorm_fixture_fresh_000001",
    )


def test_repeated_conditional_execution_is_deterministic(native_context):
    outputs = []
    for _ in range(2):
        state, _ = make_state(native_context, structural.dense_mask(3))
        result = structural.StructuralNativeSemanticOperators().layer_norm(
            state, normalizer(), "standard")
        proof = structural.get_support(result.value)
        outputs.append((result.value.zonotope_w.clone(), proof,
                        result.certificate["support_transition_trace"]))
    assert torch.equal(outputs[0][0], outputs[1][0])
    assert outputs[0][1] == outputs[1][1]
    assert outputs[0][2] == outputs[1][2]
