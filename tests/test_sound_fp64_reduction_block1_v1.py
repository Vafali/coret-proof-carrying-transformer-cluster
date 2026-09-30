from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research_hab"))

import coret_sound_fp64_block0_feasibility_v1 as experiment
import coret_structural_support_precise_dot_v1 as structural


def _state(generator_count=12, offset=0.0):
    device = torch.device("cpu")
    args = experiment._args(device)
    weights = torch.zeros(1 + generator_count, 2, 2, dtype=torch.float64)
    weights[0] = torch.tensor(
        [[1.0 + offset, -2.0], [0.5, 3.0]], dtype=torch.float64)
    for index in range(generator_count):
        weights[1 + index, index % 2, (index // 2) % 2] = (
            (index + 1) * 2.0 ** -10)
    low = -torch.ones(generator_count, dtype=torch.float64)
    high = torch.ones(generator_count, dtype=torch.float64)
    # A non-unit ranged symbol is protected by the reduction theorem.
    low[1], high[1] = -0.25, 0.75
    with experiment.pinned_zonotope() as Zonotope:
        z = Zonotope(
            args=args, p=100, eps=experiment.prefix.FIXTURE_RHO,
            perturbed_word_index=0, zonotope_w=weights,
            error_term_range_low=low, error_term_range_high=high,
            clone=False)
    proof = structural.SupportProof(
        tuple(structural.local_mask(index % 2)
              for index in range(generator_count)),
        tuple("input_source_000" if index == 0 else f"fresh_{index:03d}"
              for index in range(generator_count)),
        tuple("input_source" if index == 0 else "native_fresh"
              for index in range(generator_count)),
        2)
    structural.attach_support(z, proof)
    return z, proof


def test_sound_reduction_replays_and_rejects_all_required_mutations():
    source, proof = _state()
    reduced, reduced_proof, witness = experiment.sound_reduce(
        source, proof, 8, "unit")
    report = experiment.check_reduction_witness(
        source, proof, reduced, reduced_proof, witness)
    assert reduced.num_error_terms == 8
    assert report["accepted"]
    assert report["support_inflation"] >= 0
    assert proof.ids[0] in reduced_proof.ids
    assert proof.ids[1] in reduced_proof.ids
    mutations = experiment._reduction_mutations(
        source, proof, reduced, reduced_proof, witness)
    assert len(mutations) == 9
    assert all(mutations.values())


def test_pair_reduction_preserves_both_operand_hulls_and_order():
    left, proof = _state(offset=0.0)
    right, right_proof = _state(offset=0.25)
    assert proof == right_proof
    a, b, output_proof, witness = experiment.sound_reduce_pair(
        left, right, proof, 10, "pair")
    assert a.num_error_terms == b.num_error_terms == 10
    assert len(output_proof.ids) == 10
    assert witness["left_containment"]["accepted"]
    assert witness["right_containment"]["accepted"]
    assert witness["left_containment"]["support_inflation"] >= 0
    assert witness["right_containment"]["support_inflation"] >= 0
    assert output_proof.ids[:witness["retained"]] == tuple(
        proof.ids[index]
        for index in sorted(set(range(left.num_error_terms)) - set(
            witness["left_witness"]["dropped_indices"])))


def test_reduction_is_deterministic_and_provenance_explicit():
    source, proof = _state()
    first = experiment.sound_reduce(source, proof, 8, "repeat")
    second = experiment.sound_reduce(source, proof, 8, "repeat")
    assert torch.equal(first[0].zonotope_w, second[0].zonotope_w)
    assert first[1] == second[1]
    assert first[2] == second[2]
    assert all(identifier.startswith("reduction_box::repeat::")
               for identifier in first[1].ids[first[2]["retained_indices"].__len__():])


def test_invalid_cap_fails_closed():
    source, proof = _state(generator_count=4)
    with pytest.raises(RuntimeError, match="invalid sound reduction cap"):
        experiment.sound_reduce(source, proof, 3, "too_small")
