"""Tiny CPU-only semantic epsilon-floor images and hostile witness mutations."""
from copy import deepcopy
from fractions import Fraction
import math
from types import SimpleNamespace

import pytest
import torch

from test_sound_fp64_layernorm_separator_v1 import (
    sound, structural, finish, S, toy, fp64_default, native_context)
import sound_fp64_layernorm_epsilon_floor_v1 as E
import semantic_epsilon_floor_checker_v1 as C


def prepared_zero(monkeypatch, constant=0.):
    state, proof = toy()
    state.zonotope_w[0].fill_(constant)
    state.zonotope_w[1].fill_(.125)  # Perfectly correlated constant vectors.
    monkeypatch.setattr(S, "prepare", lambda *args: {"admissible": False, "reason": "NO_SEPARATOR"})
    prepared = sound._prepare_layernorm_separator(state, proof, -torch.ones(2,4), "test_floor")
    return state, proof, prepared


def execute(native_context, state, proof, prepared):
    Z, args = native_context
    z = Z(args=args, p=100, eps=.001, perturbed_word_index=0,
          zonotope_w=state.zonotope_w.clone(),
          error_term_range_low=state.error_term_range_low.clone(),
          error_term_range_high=state.error_term_range_high.clone(), clone=False)
    structural.attach_support(z, proof)
    dispatch = sound.production.NativeProductionDispatch(delegate=structural.StructuralNativeSemanticOperators())
    dispatch.delegate._layer_norm_index = 6
    parameter = SimpleNamespace(weight=torch.ones(4), bias=torch.zeros(4))
    out, op, reserve = sound._layernorm_sound_raw(dispatch, z, proof, parameter, "test_floor", prepared=prepared)
    return out, op, reserve, dispatch


@pytest.mark.parametrize("constant", [0., 7.])
def test_zero_variance_and_constant_vector_semantic_image(native_context, monkeypatch, constant):
    state, proof, prepared = prepared_zero(monkeypatch, constant)
    before = state.zonotope_w.view(torch.int64).clone()
    out, op, reserve, dispatch = execute(native_context, state, proof, prepared)
    assert prepared["payload"]["path"] == "semantic_epsilon_floor"
    assert out.num_error_terms == 1+2*4
    assert structural.validate_support(out, op)["validated"]
    assert torch.equal(state.zonotope_w.view(torch.int64), before)
    assert bool((out.zonotope_w[0] == 0).all())
    assert bool(torch.isfinite(out.zonotope_w).all())
    assert dispatch.delegate._layer_norm_index == 7
    assert dispatch.generic_family_invocations == 0
    payload = prepared["payload"]
    assert C.replay_persisted(payload, payload["source_state_identity"], payload["parameter_hashes"],
                             payload["source_domain"], payload["output_state_identity"])["verified"]


def test_epsilon_outward_sqrt_reciprocal_exact_containment(native_context, monkeypatch):
    state, proof, p = prepared_zero(monkeypatch)
    execute(native_context, state, proof, p)
    claim = p["payload"]["tokens"][0]
    hlow, hhigh = map(float.fromhex, claim["regularized_range"])
    sl, su = map(float.fromhex, claim["sqrt_range"])
    rl, ru = map(float.fromhex, claim["reciprocal_range"])
    f = Fraction.from_float
    assert hlow == 9.999999999999998e-13 and 0 < f(hlow) < f(1e-12)
    assert f(sl)**2 <= f(hlow) and f(su)**2 >= f(hhigh)
    assert f(rl)*f(su) <= 1 and f(ru)*f(sl) >= 1


def test_no_failed_token_is_passed_to_native_sqrt_or_reciprocal(native_context, monkeypatch):
    state, proof, p = prepared_zero(monkeypatch)
    Z, _ = native_context
    monkeypatch.setattr(Z, "sqrt", lambda *_: pytest.fail("invalid affine variance passed to sqrt"))
    monkeypatch.setattr(Z, "reciprocal", lambda *_: pytest.fail("invalid affine root passed to reciprocal"))
    execute(native_context, state, proof, p)


def test_mixed_only_failed_token_uses_floor(native_context, monkeypatch):
    state, proof = toy()
    state.zonotope_w[:,1] = state.zonotope_w[:,1,:1].expand(-1,4)
    monkeypatch.setattr(S, "prepare", lambda *_: {"admissible": False})
    generic = torch.tensor([[.7]*4, [-1.]*4])
    p = sound._prepare_layernorm_separator(state, proof, generic, "mixed")
    out, op, reserve, _ = execute(native_context, state, proof, p)
    assert p["payload"]["failed_tokens"] == [1]
    assert p["semantic_lower_by_token"] == [.7, 0.]
    # Independent native evaluation of the unchanged successful token.
    Z, args = native_context
    z = Z(args=args, p=100, eps=.001, perturbed_word_index=0,
          zonotope_w=state.zonotope_w[:,:1].clone(), clone=False)
    native = z.layer_norm(SimpleNamespace(weight=torch.ones(4), bias=torch.zeros(4)), "standard")
    assert torch.equal(out.zonotope_w[:native.num_error_terms+1,0].view(torch.int64),
                       native.zonotope_w[:,0].view(torch.int64))
    assert bool((out.zonotope_w[native.num_error_terms+1:,0] == 0).all())
    assert bool((reserve[0] > 0).all())


@pytest.mark.parametrize("mutation", ["epsilon", "h_lower", "h_upper", "sqrt", "reciprocal", "reserve",
    "omit_outward", "negative_domain", "allocation", "provenance", "source", "range", "id", "mask", "parameter"])
def test_tampered_epsilon_floor_witness_rejected(native_context, monkeypatch, mutation):
    state, proof, p = prepared_zero(monkeypatch)
    execute(native_context, state, proof, p)
    original = p["payload"]
    bad = deepcopy(original)
    row = bad["tokens"][0]
    if mutation == "epsilon": row["epsilon"] = (2e-12).hex()
    if mutation == "h_lower": row["regularized_range"][0] = (2e-12).hex()
    if mutation == "h_upper": row["regularized_range"][1] = (0.).hex()
    if mutation == "sqrt": row["sqrt_range"][0] = (1.).hex()
    if mutation == "reciprocal": row["reciprocal_range"][1] = (1.).hex()
    if mutation == "reserve": row["output_boxes"][0]["reserve"] = (0.).hex()
    if mutation == "omit_outward": row["regularized_range"][0] = (1e-12).hex()
    if mutation == "negative_domain": row["structural_rule"] = "UNCONSTRAINED_VARIANCE_ZONOTOPE"
    if mutation == "allocation": bad["allocation"]["ids"].reverse()
    if mutation == "provenance": bad["allocation"]["reasons"][0] = "forged"
    if mutation == "source": bad["source"]["weights"]["data"] = b"\0"*len(bad["source"]["weights"]["data"])
    if mutation == "range": bad["source"]["high"]["data"] = b"\0"*len(bad["source"]["high"]["data"])
    if mutation == "id": bad["source"]["proof"]["ids"][0] = "forged"
    if mutation == "mask": bad["source"]["proof"]["masks"][0] = 0
    if mutation == "parameter": bad["gamma"]["data"] = b"\0"*len(bad["gamma"]["data"])
    with pytest.raises(RuntimeError):
        C.replay_persisted(bad, original["source_state_identity"], original["parameter_hashes"],
                           original["source_domain"], original["output_state_identity"])


def test_malformed_semantic_state_fails_closed(monkeypatch):
    state, proof, _ = prepared_zero(monkeypatch)
    state.error_term_range_low[0] = 2.
    with pytest.raises(RuntimeError, match="box"):
        sound._prepare_layernorm_separator(state, proof, -torch.ones(2,4), "bad")


def test_floor_domain_uses_regularized_range_not_positive_variance(monkeypatch):
    state, proof, _ = prepared_zero(monkeypatch)
    low = -torch.ones(2,4)
    diagnostic = {"label": "floor", "sound_variance_lower": -1., "domain_admissible": False}
    effective, diag, p = finish._repair_layernorm_domain(state, proof, low, diagnostic)
    assert bool((effective == 0).all()) and diag["domain_admissible"]
    assert diag["sqrt_input_lower"] == 9.999999999999998e-13
    assert diag["unconstrained_variance_not_used_as_semantic_domain"]


def test_persistence_replay(native_context, monkeypatch, tmp_path):
    state, proof, p = prepared_zero(monkeypatch)
    execute(native_context, state, proof, p)
    path = tmp_path/"floor.pt"
    sound._save_artifact(path, "tiny", {}, {"_separating_variance_witnesses": [p["payload"]]})
    saved = torch.load(path, weights_only=False)["separating_variance_witnesses"][0]
    assert C.replay_persisted(saved, saved["source_state_identity"], saved["parameter_hashes"],
                             saved["source_domain"], saved["output_state_identity"])["verified"]


def test_checker_has_no_numerical_producer_imports():
    import ast
    from pathlib import Path
    imports = [n for n in ast.walk(ast.parse(Path(C.__file__).read_text()))
               if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = {a.name for n in imports if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module for n in imports if isinstance(n, ast.ImportFrom)}
    assert names == {"fractions", "hashlib", "json", "math", "struct", "separating_variance_checker_v1"}


def test_certified_separator_dispatch_is_bitwise_unchanged(native_context):
    state, proof = toy()
    Z, args = native_context
    z = Z(args=args, p=100, eps=.001, perturbed_word_index=0,
          zonotope_w=state.zonotope_w, error_term_range_low=state.error_term_range_low,
          error_term_range_high=state.error_term_range_high, clone=False)
    structural.attach_support(z, proof)
    p = S.prepare(z, proof, -torch.ones(2,4), "separator_success")
    normalizer = SimpleNamespace(weight=torch.ones(4), bias=torch.zeros(4))
    def dispatch():
        result = sound.production.NativeProductionDispatch(delegate=structural.StructuralNativeSemanticOperators())
        result.delegate._layer_norm_index = 6
        return result
    old = S.execute_prepared(dispatch(), z, proof, normalizer, deepcopy(p))
    actual = sound._layernorm_sound_raw(dispatch(), z, proof, normalizer, "separator_success", prepared=deepcopy(p))
    structural._assert_layer_norm_replay_parity(old[0], actual[0], "unchanged_separator")
    assert old[1] == actual[1]
    assert torch.equal(old[2].view(torch.int64), actual[2].view(torch.int64))


def test_forged_endpoint_does_not_authorize_execution(native_context, monkeypatch):
    state, proof, p = prepared_zero(monkeypatch)
    p["semantic_lower_by_token"] = [999., 0.]
    with pytest.raises(RuntimeError, match="domain differs"):
        execute(native_context, state, proof, p)


def test_nonconstant_box_contains_exact_centered_normalization(native_context, monkeypatch):
    state, proof, p = prepared_zero(monkeypatch)
    # An interval of actual centered vectors containing the zero vector.
    state.zonotope_w.zero_()
    state.zonotope_w[1,:,0] = 1.
    p = sound._prepare_layernorm_separator(state, proof, -torch.ones(2,4), "nonconstant")
    out, _, _, _ = execute(native_context, state, proof, p)
    # Use exact rational squares, avoiding reliance on a libm root as proof.
    claim = p["payload"]["tokens"][0]
    for xi in (Fraction(-1), Fraction(0), Fraction(1,2), Fraction(1)):
        centered = [3*xi/4, -xi/4, -xi/4, -xi/4]
        h = sum(a*a for a in centered)/4 + Fraction.from_float(1e-12)
        for j, a in enumerate(centered):
            b = claim["output_boxes"][j]
            lower, upper = C.support.read_rational(b["exact_lower"]), C.support.read_rational(b["exact_upper"])
            assert lower <= 0 <= upper
            assert max(-lower, upper)**2*h >= a*a


@pytest.mark.parametrize("field", ["weights", "low", "ids", "masks", "reasons"])
def test_persisted_output_mutation_rejected(native_context, monkeypatch, field):
    state, proof, p = prepared_zero(monkeypatch)
    execute(native_context, state, proof, p)
    original = p["payload"]
    bad = deepcopy(original)
    if field in ("weights", "low"):
        b = bad["output"][field]
        b["data"] = b"\0"*len(b["data"])
    else:
        bad["output"]["proof"][field][-1] = 0 if field == "masks" else "forged"
    with pytest.raises(RuntimeError):
        C.replay_persisted(bad, original["source_state_identity"], original["parameter_hashes"],
                           original["source_domain"], original["output_state_identity"])
