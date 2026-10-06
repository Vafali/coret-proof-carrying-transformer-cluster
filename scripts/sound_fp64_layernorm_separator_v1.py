"""Exact-replayed tokenwise LayerNorm domain fallback for sound FP64.

No state coefficient is clamped. Only failed variance domains use an additional
semantic range constraint. Successful native paths remain unchanged.
"""
from fractions import Fraction
import math
import time

import separating_variance_checker_v1 as CHECK

SCHEMA = "CORET_SOUND_FP64_SEPARATING_LAYERNORM_V1"
PROPOSAL_SECONDS = 60.0


def _operands(block):
    return (block["center"].tolist(), block["generators"].tolist(),
            block["low"].tolist(), block["high"].tolist(), block["ids"])


def _sparse_candidates(operands):
    import numpy as np
    c, g, low, high, _ = operands
    c, g, low, high = map(np.asarray, (c, g, low, high))
    differences = g[:, :-1]-g[:, -1:]
    lo = c[:-1]-c[-1] + np.minimum(differences*low[:, None], differences*high[:, None]).sum(axis=0)
    hi = c[:-1]-c[-1] + np.maximum(differences*low[:, None], differences*high[:, None]).sum(axis=0)
    gap = np.maximum(lo, -hi)
    order = sorted(range(len(gap)), key=lambda i: (-float(gap[i]), i))[:8]
    for coordinate in order:
        y = [Fraction(0)]*len(gap)
        y[coordinate] = Fraction(1)
        yield "COORDINATE_DIFFERENCE", y
    # Four fixed sparse combinations; order and signs are proposal-only.
    for a, b in zip(order[:4], order[1:5]):
        y = [Fraction(0)]*len(gap)
        y[a] = Fraction(1 if c[a] >= c[-1] else -1)
        y[b] = Fraction(1 if c[b] >= c[-1] else -1)
        yield "SPARSE_TWO_DIFFERENCES", y


def propose_checked_separator(operands, binding, budget=PROPOSAL_SECONDS):
    started = time.monotonic()
    attempts = 0
    for method, y in _sparse_candidates(operands):
        if time.monotonic()-started >= budget:
            return None, {"reason": "PROPOSAL_BUDGET_EXHAUSTED", "sparse_attempts": attempts}
        attempts += 1
        certificate = CHECK.construct_certificate(*operands, y, binding)
        if certificate is not None:
            CHECK.verify_certificate(*operands, binding, certificate)
            return certificate, {"method": method, "sparse_attempts": attempts,
                                 "seconds": time.monotonic()-started}
    remaining = budget-(time.monotonic()-started)
    if remaining <= 0:
        return None, {"reason": "PROPOSAL_BUDGET_EXHAUSTED", "sparse_attempts": attempts}
    from certify_block2_output_separating_variance_v1 import bounded_proposal
    proposal = bounded_proposal(operands[:4], min(60., remaining))
    y = proposal.get("y")
    certificate = None
    if (y is not None and len(y) == len(operands[0])-1
            and all(math.isfinite(value) for value in y) and any(y)):
        certificate = CHECK.construct_certificate(
            *operands, [Fraction.from_float(value) for value in y], binding)
        if certificate is not None:
            CHECK.verify_certificate(*operands, binding, certificate)
    return certificate, {"method": "BOUNDED_LP", "sparse_attempts": attempts,
                         "numerical_proposal": {k:v for k,v in proposal.items() if k != "y"},
                         "seconds": time.monotonic()-started}


def replay_payload(payload):
    """CPU decoding adapter; all bound arithmetic lives in stdlib CHECK.

    This checks the separator obligation, NOT a complete Transformer graph.
    The graph's authenticated predecessor must match source_state_identity.
    """
    if payload.get("schema") != SCHEMA:
        raise RuntimeError("separating LayerNorm schema differs")
    proof = payload["proof"]
    lower = list(payload["generic_lower_by_token"])
    tokens = payload["failed_tokens"]
    identity = payload["source_state_identity"]
    if (len(lower) != proof["num_tokens"] or proof["num_tokens"] != identity["token_count"]
            or len(proof["ids"]) != identity["generator_count"]
            or not all(math.isfinite(x) for x in lower)):
        raise RuntimeError("separating LayerNorm source topology/domain differs")
    if tokens != [i for i,v in enumerate(lower) if v <= 0] or len(set(tokens)) != len(tokens):
        raise RuntimeError("separating LayerNorm failed-token inventory differs")
    if [row["token_index"] for row in payload["tokens"]] != tokens:
        raise RuntimeError("separating LayerNorm missing/reordered token witness")
    for row in payload["tokens"]:
        block = row["operands"]
        if (block["ids"] != proof["ids"] or len(proof["ids"]) != len(block["generators"])
                or len(block["center"]) != identity["hidden_dimension"]
                or any(len(proof[k]) != len(proof["ids"]) for k in ("masks", "reasons"))):
            raise RuntimeError("separating LayerNorm support/ID inventory differs")
        for index, mask in enumerate(proof["masks"]):
            if type(mask) is not int or not 0 <= mask < (1 << proof["num_tokens"]):
                raise RuntimeError("separating LayerNorm support mask malformed")
            if not mask & (1 << row["token_index"]) and bool((block["generators"][index] != 0).any()):
                raise RuntimeError("separating LayerNorm support mask omits a coefficient")
        binding = {"source_state_identity": payload["source_state_identity"],
                   "source_domain": payload["source_domain"], "label": payload["label"],
                   "token_index": row["token_index"], "proof_sha256": CHECK.digest(proof)}
        checked = CHECK.verify_certificate(*_operands(block), binding, row["certificate"])
        bound = CHECK.downward_binary64(CHECK.read_rational(checked["variance_lower"]))
        if bound <= 0 or row["variance_lower_binary64"] != bound:
            raise RuntimeError("separating LayerNorm bound is not outward/positive")
        lower[row["token_index"]] = bound
    if lower != payload["semantic_lower_by_token"] or not all(math.isfinite(x) and x > 0 for x in lower):
        raise RuntimeError("separating LayerNorm semantic bounds differ")
    if "native_transition" in payload:
        ranges = payload["native_transition"]["semantic_range_certificate"]
        if ranges["range_override_tokens"] != tokens:
            raise RuntimeError("separating LayerNorm range override token inventory differs")
        CHECK.verify_semantic_sqrt_ranges(lower, ranges)
    return lower


def prepare(state, proof, generic_low, label):
    import torch
    # Fast predicate first: no CPU operands, LP, exact support or new certificate.
    if not bool((generic_low <= 0).any()):
        return None
    import coret_sound_fp64_block0_feasibility_v1 as sound
    import coret_psd_layernorm_experiment_v1 as native
    sound.structural.validate_support(state, proof, token_axis=1)
    if (state.zonotope_w.ndim != 3 or state.zonotope_w.dtype != torch.float64
            or tuple(generic_low.shape) != tuple(state.zonotope_w.shape[1:])
            or not bool(torch.isfinite(state.zonotope_w).all() and torch.isfinite(generic_low).all())):
        raise RuntimeError("separating LayerNorm malformed state/domain")
    low, high = sound._ranges(state)
    if not bool(torch.isfinite(low).all() and torch.isfinite(high).all()) or bool((low > high).any()):
        raise RuntimeError("separating LayerNorm malformed source box")
    token_lowers = generic_low.detach().cpu().amin(dim=1).tolist()
    failed = [i for i, value in enumerate(token_lowers) if value <= 0]
    raw_proof = {"ids": list(proof.ids), "masks": list(proof.masks),
                 "reasons": list(proof.reasons), "num_tokens": proof.num_tokens}
    identity = native.state_identity(state, proof)
    domain = {"p": state.p, "rho_hex": float(state.eps).hex(),
              "token_ids": list(sound.prefix.FIXTURE_TOKEN_IDS),
              "perturbed_token": sound.prefix.FIXTURE_PERTURBED_TOKEN,
              "pinned_revision": sound.PINNED_REVISION,
              "checkpoint_sha256": sound.prefix.CHECKPOINT_SHA256,
              "source_set": "shared_generator_ids_cartesian_ranges"}
    payload = {"schema": SCHEMA, "label": label, "source_state_identity": identity,
               "source_domain": domain, "proof": raw_proof,
               "generic_lower_by_token": token_lowers, "failed_tokens": failed, "tokens": []}
    semantic = list(token_lowers)
    for token in failed:
        rows = state.zonotope_w[:, token].detach().cpu().clone()
        block = {"center": rows[0], "generators": rows[1:],
                 "low": low.detach().cpu().clone(), "high": high.detach().cpu().clone(),
                 "ids": list(proof.ids)}
        binding = {"source_state_identity": identity, "source_domain": domain,
                   "label": label, "token_index": token, "proof_sha256": CHECK.digest(raw_proof)}
        certificate, telemetry = propose_checked_separator(_operands(block), binding)
        if certificate is None:
            return {"admissible": False, "failed_token": token, "proposal": telemetry}
        checked = CHECK.verify_certificate(*_operands(block), binding, certificate)
        bound = CHECK.downward_binary64(CHECK.read_rational(checked["variance_lower"]))
        if bound <= 0:
            return {"admissible": False, "failed_token": token, "reason": "POSITIVE_BOUND_UNREPRESENTABLE"}
        semantic[token] = bound
        payload["tokens"].append({"token_index": token, "operands": block,
                                  "certificate": certificate, "variance_lower_binary64": bound,
                                  "proposal": telemetry})
    payload["semantic_lower_by_token"] = semantic
    replay_payload(payload)
    return {"admissible": True, "payload": payload, "semantic_lower_by_token": semantic}


def execute_prepared(dispatch, state, proof, normalizer, prepared):
    import coret_psd_layernorm_experiment_v1 as native
    import coret_sound_fp64_block0_feasibility_v1 as sound
    if not prepared["admissible"]:
        raise RuntimeError("sound FP64 LayerNorm separating domain remains unresolved")
    payload = prepared["payload"]
    if native.state_identity(state, proof) != payload["source_state_identity"]:
        raise RuntimeError("separating LayerNorm predecessor substituted")
    lower = replay_payload(payload)
    summary = {"schema": SCHEMA, "label": payload["label"],
               "source_state_identity": payload["source_state_identity"],
               "failed_tokens": payload["failed_tokens"],
               "semantic_lower_by_token": lower,
               "token_certificate_sha256": [r["certificate"]["certificate_sha256"] for r in payload["tokens"]],
               "generic_semantic_remainder_used": False, "native_result_unchanged": False,
               "domain_repaired_by_exact_separator": True}
    output, output_proof, transition = native._native_layernorm_with_semantic_lower(
        state, proof, normalizer, dispatch.delegate, lower, {}, {},
        separator_context=summary)
    ops = 16*128*(state.num_error_terms+1)**2+4096
    reserve = sound._reserve_from_majorant(native._majorant_with_semantic_lower(state, normalizer, lower), ops)
    payload["native_transition"] = transition
    replay_payload(payload)
    summary["native_transition"] = transition
    dispatch.counts["LayerNorm"] += 1
    dispatch.certificates.append(summary)
    if not hasattr(dispatch, "separating_variance_witnesses"):
        dispatch.separating_variance_witnesses = []
    dispatch.separating_variance_witnesses.append(payload)
    return output, output_proof, reserve
