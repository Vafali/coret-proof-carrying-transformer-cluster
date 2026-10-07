"""Exact liveness projection at the frozen final encoder/head boundary only."""
from copy import deepcopy

import torch

import final_token_projection_checker_v1 as CHECK
from sound_fp64_layernorm_epsilon_floor_v1 import source_block


def project(state, proof, execution_identity, region):
    import coret_sound_fp64_block0_feasibility_v1 as sound
    structural = sound.structural
    if region != CHECK.REGION:
        raise RuntimeError("projection before final no-cross-token boundary is forbidden")
    if state.zonotope_w.ndim != 3 or state.zonotope_w.dtype != torch.float64:
        raise RuntimeError("final token projection requires FP64 hidden state")
    if state.num_error_terms:
        structural.validate_support(state, proof, token_axis=1)
    source = source_block(state, proof)
    input_identity = CHECK.state_identity(source)
    # Coefficients, not support-mask membership, authorize deletion.
    selected = state.zonotope_w[:, :1, :]
    active = (selected[1:] != 0).any(dim=(1, 2))
    kept = active.nonzero().flatten()
    indices = kept.detach().cpu().tolist()
    discarded = (~active).nonzero().flatten().detach().cpu().tolist()
    low, high = sound._ranges(state)
    weights = torch.cat((selected[:1], selected[1:].index_select(0, kept)), dim=0)
    output = sound._make_like(state, weights,
                              low.index_select(0, kept), high.index_select(0, kept))
    output_proof = structural.SupportProof(
        tuple(proof.masks[i] & 1 for i in indices),
        tuple(proof.ids[i] for i in indices),
        tuple(proof.reasons[i] for i in indices), 1)
    structural.attach_support(output, output_proof)
    # The legacy tensor validator cannot shape an empty claimed-mask array.
    # The independent byte checker below also validates the zero-row case,
    # including exact empty IDs/ranges/provenance; no dummy symbol is added.
    if output.num_error_terms:
        structural.validate_support(output, output_proof, token_axis=1)
    output_block = source_block(output, output_proof)
    output_identity = CHECK.state_identity(output_block)
    payload = {
        "schema": CHECK.SCHEMA, "region_proof": deepcopy(region),
        "execution_identity": deepcopy(execution_identity),
        "input": source, "output": output_block,
        "original_token_count": proof.num_tokens, "retained_token_indices": [0],
        "input_state_identity": input_identity, "output_state_identity": output_identity,
        "pre_projection_generator_ids": list(proof.ids),
        "post_projection_generator_ids": list(output_proof.ids),
        "retained_generator_indices": indices,
        "zero_after_projection_discarded_generator_ids": [proof.ids[i] for i in discarded],
        "generator_count_before": state.num_error_terms,
        "generator_count_after": output.num_error_terms,
    }
    CHECK.verify(payload, input_identity, output_identity, execution_identity, region)
    return output, output_proof, payload


def project_then_reduce(state, proof, execution_identity, region, reductions):
    import coret_sound_fp64_block0_feasibility_v1 as sound
    projected, projected_proof, payload = project(state, proof, execution_identity, region)
    first = len(reductions)
    output, output_proof = sound._maybe_reduce(
        projected, projected_proof, "b2_output_layernorm", reductions)
    payload["subsequent_reduction_records"] = deepcopy(reductions[first:])
    payload["generator_count_after_reduction"] = output.num_error_terms
    return output, output_proof, payload
