"""Third LayerNorm domain path: semantic PSD floor, never affine clamping."""
import sys

import semantic_epsilon_floor_checker_v1 as CHECK

SCHEMA = CHECK.SCHEMA


def block(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    if str(array.dtype) != "float64" or sys.byteorder != "little":
        raise RuntimeError("epsilon-floor requires little-endian binary64 encoding")
    return {"dtype": "float64", "byteorder": "little", "shape": list(array.shape),
            "data": array.tobytes()}


def source_block(state, proof):
    import coret_sound_fp64_block0_feasibility_v1 as sound
    low, high = sound._ranges(state)
    return {"weights": block(state.zonotope_w), "low": block(low), "high": block(high),
            "proof": {"ids": list(proof.ids), "masks": list(proof.masks),
                      "reasons": list(proof.reasons), "num_tokens": proof.num_tokens}}


def prepare(state, proof, generic_low, label, separator_attempt):
    import torch
    import coret_psd_layernorm_experiment_v1 as native
    generic = generic_low.detach().cpu().amin(dim=1).tolist()
    if (tuple(generic_low.shape) != tuple(state.zonotope_w.shape[1:]) or
            not bool(torch.isfinite(generic_low).all())):
        raise RuntimeError("epsilon-floor malformed generic domain")
    identity = native.state_identity(state, proof)
    source = source_block(state, proof)
    CHECK.authenticate_source(source, identity)
    failed = [t for t, value in enumerate(generic) if value <= 0]
    if not failed or separator_attempt.get("admissible") is not False:
        raise RuntimeError("epsilon-floor requires failed generic and separator paths")
    semantic = [0. if t in failed else value for t, value in enumerate(generic)]
    payload = {"schema": SCHEMA, "path": "semantic_epsilon_floor", "label": label,
               "source_state_identity": identity, "source": source,
               "failed_tokens": failed, "generic_lower_by_token": generic,
               "semantic_lower_by_token": semantic, "separator_attempt": separator_attempt,
               "source_domain": {"p": state.p, "rho_hex": float(state.eps).hex(),
                    "token_ids": list(native.sound.prefix.FIXTURE_TOKEN_IDS),
                    "perturbed_token": native.sound.prefix.FIXTURE_PERTURBED_TOKEN,
                    "checkpoint_sha256": native.sound.prefix.CHECKPOINT_SHA256,
                    "pinned_revision": native.sound.PINNED_REVISION}}
    return {"admissible": True, "payload": payload, "semantic_lower_by_token": semantic}


def execute_prepared(dispatch, state, proof, normalizer, prepared):
    import torch
    import coret_psd_layernorm_experiment_v1 as native
    import coret_sound_fp64_block0_feasibility_v1 as sound
    structural = sound.structural
    payload = prepared["payload"]
    identity = native.state_identity(state, proof)
    if (not prepared.get("admissible") or identity != payload["source_state_identity"] or
            prepared["semantic_lower_by_token"] != payload["semantic_lower_by_token"]):
        raise RuntimeError("epsilon-floor predecessor/prepared domain differs")
    failed = payload["failed_tokens"]
    if (failed != [t for t,x in enumerate(payload["generic_lower_by_token"]) if x <= 0] or
            payload["semantic_lower_by_token"] != [0. if t in failed else x
                         for t,x in enumerate(payload["generic_lower_by_token"])]):
        raise RuntimeError("epsilon-floor semantic token inventory differs")
    gamma, beta = block(normalizer.weight.double()), block(normalizer.bias.double())
    parameters = {"weight": CHECK.tensor_hash(gamma), "bias": CHECK.tensor_hash(beta)}
    payload.update(CHECK.construct(payload["source"], identity, gamma, beta, parameters,
                                   failed, payload["label"]))
    # Reuse the existing PSD semantic-range primitive in the producer; its
    # result is still independently checked by exact squares in CHECK.
    for claim in payload["tokens"]:
        hlow, hhigh = map(float.fromhex, claim["regularized_range"])
        root_low, _ = native._directed_sqrt_binary64(hlow, False)
        root_high, _ = native._directed_sqrt_binary64(hhigh, True)
        if [root_low.hex(), root_high.hex()] != claim["sqrt_range"]:
            raise RuntimeError("epsilon-floor PSD directed sqrt primitive differs")
    CHECK.verify(payload, identity, parameters)
    n, tokens, d = state.num_error_terms, state.num_words, state.word_embedding_size
    low, high = sound._ranges(state)
    good = [t for t in range(tokens) if t not in failed]
    index = dispatch.delegate._layer_norm_index
    weights = torch.zeros_like(state.zonotope_w)
    reserve = torch.zeros((tokens, d), dtype=torch.float64, device=state.device)
    keep_mask = sum(1 << t for t in good)
    masks = [m & keep_mask for m in proof.masks]
    ids, reasons = list(proof.ids), list(proof.reasons)
    out_low, out_high = low.clone(), high.clone()
    # No invalid token is ever passed to native sqrt/reciprocal. Independent
    # token arithmetic on the still-valid tokens retains native coefficients.
    if good:
        local_masks = tuple(sum(1 << j for j,t in enumerate(good) if m & (1 << t))
                            for m in proof.masks)
        local_proof = structural.SupportProof(local_masks, proof.ids, proof.reasons, len(good))
        local = state.__class__(args=state.args, p=state.p, eps=state.eps,
              perturbed_word_index=0, zonotope_w=state.zonotope_w[:, good].clone(),
              error_term_range_low=low.clone(), error_term_range_high=high.clone(), clone=False)
        structural.attach_support(local, local_proof)
        actual = local.layer_norm(normalizer, "standard")
        replay, trace = structural._replay_layer_norm_with_membership(
            local, normalizer, "standard", local_proof, f"layernorm_{index}")
        structural._assert_layer_norm_replay_parity(actual, replay, "epsilon_floor_valid_tokens")
        actual_proof = dispatch.delegate._layer_norm_output(
            local_proof, actual, f"layernorm_{index}", trace.masks, trace.reasons)
        weights = torch.zeros((actual.num_error_terms+1, tokens, d), dtype=torch.float64, device=state.device)
        weights[:, good] = actual.zonotope_w
        masks.extend(sum(1 << good[j] for j in range(len(good)) if m & (1 << j))
                     for m in actual_proof.masks[n:])
        ids.extend(actual_proof.ids[n:]); reasons.extend(actual_proof.reasons[n:])
        out_low, out_high = sound._ranges(actual)
        # Original successful-token majorant and reserve formula unchanged.
        reserve[good] = sound._reserve_from_majorant(sound._layernorm_majorant(local, normalizer),
                         16*128*(n+1)**2+4096)
    start = len(ids)
    count = len(failed)*d
    fresh = torch.zeros((count, tokens, d), dtype=torch.float64, device=state.device)
    offset = 0
    for claim in payload["tokens"]:
        t = claim["token_index"]
        for j, box in enumerate(claim["output_boxes"]):
            weights[0, t, j] = float.fromhex(box["center"])
            fresh[offset, t, j] = float.fromhex(box["radius"])
            reserve[t, j] = float.fromhex(box["reserve"])
            ids.append(f"semantic_epsilon_floor::{payload['label']}::{t:06d}::{j:06d}")
            masks.append(1 << t)
            reasons.append("semantic_layernorm_epsilon_floor_coordinate_box")
            offset += 1
    allocation = {"start_generator": start, "ids": ids[start:], "masks": masks[start:],
                  "reasons": reasons[start:], "low": [-1.]*count, "high": [1.]*count}
    payload["allocation"] = allocation
    output = sound._make_like(state, torch.cat((weights, fresh)),
                torch.cat((out_low, -torch.ones(count, device=state.device, dtype=torch.float64))),
                torch.cat((out_high, torch.ones(count, device=state.device, dtype=torch.float64))))
    result_proof = structural.SupportProof(tuple(masks), tuple(ids), tuple(reasons), tokens)
    structural.attach_support(output, result_proof)
    structural.validate_support(output, result_proof, token_axis=1)
    payload["output"] = source_block(output, result_proof)
    CHECK.verify_output(payload, payload["output"], block(reserve), identity, parameters)
    payload["output_state_identity"] = native.state_identity(output, result_proof)
    payload["reserve"] = block(reserve)
    payload["valid_tokens_native_fresh_count"] = start-n
    CHECK.replay_persisted(payload, identity, parameters, payload["source_domain"],
                           payload["output_state_identity"])
    dispatch.delegate._layer_norm_index += 1
    if index == 0 or index % 2 == 0:
        dispatch.delegate._hidden = result_proof
    else:
        dispatch.delegate._post_attention = result_proof
    dispatch.counts["LayerNorm"] += 1
    dispatch.certificates.append({"schema": SCHEMA, "path": "semantic_epsilon_floor",
         "label": payload["label"], "failed_tokens": failed,
         "source_state_identity": identity, "output_state_identity": payload["output_state_identity"],
         "domain_repaired_by_exact_separator": False, "independent_domain_checker_accepts": True})
    # Preserve the existing stage/campaign artifact transport. This historical
    # container now holds tagged separator OR epsilon-floor local witnesses.
    if not hasattr(dispatch, "separating_variance_witnesses"):
        dispatch.separating_variance_witnesses = []
    dispatch.separating_variance_witnesses.append(payload)
    return output, result_proof, reserve
