"""CPU-only tiny LayerNorm domains, exact replay and native range integration."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import sys
import importlib.util
from types import SimpleNamespace

import pytest
import torch

# These tests never invoke MPFR helpers; match the existing offline test shim
# on workstations without gmpy2. The ISIS production environment requires it.
if "gmpy2" not in sys.modules and importlib.util.find_spec("gmpy2") is None:
    sys.modules["gmpy2"] = SimpleNamespace()

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/"scripts"),
               str(Path(__file__).resolve().parents[1]/"research_hab")]
import sound_fp64_layernorm_separator_v1 as S
import coret_sound_fp64_block0_feasibility_v1 as sound
import coret_structural_support_precise_dot_v1 as structural
import run_sound_fp64_finish_3l_v1 as finish
from test_layernorm_support_transition_v1 import native_context


@pytest.fixture(autouse=True)
def fp64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def toy():
    weights = torch.tensor([[[3., 0., 0., 0.], [2., 0., 0., 0.]],
                            [[.1, 0., 0., 0.], [.1, 0., 0., 0.]]], dtype=torch.float64)
    state = SimpleNamespace(zonotope_w=weights, num_error_terms=1, num_words=2,
        word_embedding_size=4, device=torch.device("cpu"), p=100, eps=.001,
        error_term_range_low=torch.tensor([-1.], dtype=torch.float64),
        error_term_range_high=torch.tensor([1.], dtype=torch.float64))
    proof = structural.SupportProof((3,), ("g",), ("native",), 2)
    return state, proof


def test_positive_fast_path_does_not_propose_or_create_proof(monkeypatch):
    state, proof = toy()
    monkeypatch.setattr(S, "propose_checked_separator", lambda *_: pytest.fail("proposal called"))
    before = state.zonotope_w.view(torch.int64).clone()
    low = torch.ones(2, 4, dtype=torch.float64)
    assert S.prepare(state, proof, low, "any_layernorm") is None
    marker = state
    structural.attach_support(marker, proof)
    called = []
    dispatch = SimpleNamespace(layer_norm=lambda *args: called.append(args) or marker)
    raw, actual_proof, reserve = sound._layernorm_sound_raw(dispatch, state, proof, object(), "any", low)
    assert raw is marker and actual_proof == proof and reserve is None and len(called) == 1
    assert torch.equal(state.zonotope_w.view(torch.int64), before)


def test_sparse_fallback_certifies_each_failed_token_without_lp(monkeypatch):
    import certify_block2_output_separating_variance_v1 as LP
    monkeypatch.setattr(LP, "bounded_proposal", lambda *_: pytest.fail("LP should not be necessary"))
    state, proof = toy()
    generic = torch.full((2, 4), -1., dtype=torch.float64)
    before = state.zonotope_w.view(torch.int64).clone()
    prepared = S.prepare(state, proof, generic, "arbitrary_output")
    assert prepared["admissible"]
    assert prepared["payload"]["failed_tokens"] == [0, 1]
    assert all(row["proposal"]["method"] == "COORDINATE_DIFFERENCE" for row in prepared["payload"]["tokens"])
    assert all(x > 0 for x in S.replay_payload(prepared["payload"]))
    effective = sound._checked_layernorm_low(generic, prepared)
    assert bool((effective > 0).all()) and bool((generic == -1).all())
    assert torch.equal(state.zonotope_w.view(torch.int64), before)


@pytest.mark.parametrize("mutation", ["coefficient", "range", "id", "order", "mask", "y", "bound", "omitted_token"])
def test_saved_separator_tamper_rejected(mutation):
    state, proof = toy()
    payload = deepcopy(S.prepare(state, proof, torch.full((2,4), -1., dtype=torch.float64), "ln")["payload"])
    row = payload["tokens"][0]
    if mutation == "coefficient": row["operands"]["generators"][0,0] += .5
    if mutation == "range": row["operands"]["high"][0] = 2.
    if mutation == "id": row["operands"]["ids"][0] = "forged"
    if mutation == "order": row["certificate"]["v_rationals"].reverse()
    if mutation == "mask": payload["proof"]["masks"][0] = 0
    if mutation == "y": row["certificate"]["y_rationals"][0] = S.CHECK.rational(Fraction(0))
    if mutation == "bound": row["variance_lower_binary64"] = float("inf")
    if mutation == "omitted_token": payload["tokens"].pop()
    with pytest.raises(RuntimeError): S.replay_payload(payload)


def test_unverified_numerical_proposal_cannot_repair(monkeypatch):
    import certify_block2_output_separating_variance_v1 as LP
    state, proof = toy()
    state.zonotope_w.zero_()  # No strict separator can exist.
    monkeypatch.setattr(LP, "bounded_proposal", lambda *_: {
        "y": [1., 0., 0.], "solver_status": 0, "objective_has_no_proof_authority": -99999.})
    prepared = S.prepare(state, proof, torch.full((2,4), -1., dtype=torch.float64), "ln")
    assert prepared["admissible"] is False
    with pytest.raises(RuntimeError, match="unresolved"):
        S.execute_prepared(object(), state, proof, object(), prepared)


def test_successful_tokens_keep_exact_existing_lower(monkeypatch):
    state, proof = toy()
    generic = torch.tensor([[.7]*4, [-1.]*4], dtype=torch.float64)
    prepared = S.prepare(state, proof, generic, "ln")
    assert prepared["payload"]["failed_tokens"] == [1]
    effective = sound._checked_layernorm_low(generic, prepared)
    assert torch.equal(effective[0].view(torch.int64), generic[0].view(torch.int64))


def test_native_sqrt_reciprocal_and_provenance_integration(native_context):
    Zonotope, args = native_context
    state, proof = toy()
    z = Zonotope(args=args, p=100, eps=.001, perturbed_word_index=0,
        zonotope_w=state.zonotope_w, error_term_range_low=state.error_term_range_low,
        error_term_range_high=state.error_term_range_high, clone=False)
    structural.attach_support(z, proof)
    before = z.zonotope_w.view(torch.int64).clone()
    prepared = S.prepare(z, proof, torch.full((2,4), -1., dtype=torch.float64), "block_any_output")
    delegate = structural.StructuralNativeSemanticOperators()
    delegate._layer_norm_index = 6
    dispatch = sound.production.NativeProductionDispatch(delegate=delegate)
    parameter = SimpleNamespace(weight=torch.ones(4, dtype=torch.float64), bias=torch.zeros(4, dtype=torch.float64))
    out, out_proof, reserve = S.execute_prepared(dispatch, z, proof, parameter, prepared)
    assert bool(torch.isfinite(out.zonotope_w).all() and (reserve > 0).all())
    assert structural.validate_support(out, out_proof)["validated"]
    assert dispatch.counts["LayerNorm"] == 1 and dispatch.generic_family_invocations == 0
    assert delegate._layer_norm_index == 7 and delegate._hidden == out_proof
    assert torch.equal(z.zonotope_w.view(torch.int64), before)
    transition = prepared["payload"]["native_transition"]
    assert transition["sqrt_interval_lower_min"] > 1e-12
    assert transition["reciprocal_semantic_lower_min"] > 0
    # No transient concretize override survives its one intended primitive.
    assert "concretize" not in z.__dict__
    assert S.replay_payload(prepared["payload"])
    mutated = deepcopy(prepared["payload"])
    evidence = mutated["native_transition"]["semantic_range_certificate"]["directed_range_evidence"]
    evidence[0]["sqrt_output_lower_binary64_hex"] = (99.).hex()
    with pytest.raises(RuntimeError, match="sqrt range relation"):
        S.replay_payload(mutated)


def test_native_positive_path_is_bitwise_identical(native_context, monkeypatch):
    Zonotope, args = native_context
    state, proof = toy()
    z = Zonotope(args=args, p=100, eps=.001, perturbed_word_index=0,
        zonotope_w=state.zonotope_w, error_term_range_low=state.error_term_range_low,
        error_term_range_high=state.error_term_range_high, clone=False)
    structural.attach_support(z, proof)
    parameter = SimpleNamespace(weight=torch.ones(4), bias=torch.zeros(4))
    def dispatch():
        return sound.production.NativeProductionDispatch(delegate=structural.StructuralNativeSemanticOperators())
    baseline = dispatch().layer_norm(z, parameter, "standard")
    monkeypatch.setattr(S, "prepare", lambda *_: pytest.fail("must not enter failed-domain preparation"))
    actual, actual_proof, reserve = sound._layernorm_sound_raw(dispatch(), z, proof, parameter, "embedding_layernorm")
    structural._assert_layer_norm_replay_parity(baseline, actual, "bitwise_fast_path")
    assert actual_proof == structural.get_support(baseline) and reserve is None


def test_mixed_transition_preserves_native_coefficients_of_successful_token(native_context):
    Zonotope, args = native_context
    state, proof = toy()
    z = Zonotope(args=args, p=100, eps=.001, perturbed_word_index=0,
        zonotope_w=state.zonotope_w, error_term_range_low=state.error_term_range_low,
        error_term_range_high=state.error_term_range_high, clone=False)
    structural.attach_support(z, proof)
    parameter = SimpleNamespace(weight=torch.ones(4), bias=torch.zeros(4))
    baseline = z.layer_norm(parameter, "standard")
    generic = torch.tensor([[.7]*4, [-1.]*4])
    prepared = S.prepare(z, proof, generic, "output")
    delegate = structural.StructuralNativeSemanticOperators()
    delegate._layer_norm_index = 6
    dispatch = sound.production.NativeProductionDispatch(delegate=delegate)
    repaired, _, _ = S.execute_prepared(dispatch, z, proof, parameter, prepared)
    assert repaired.num_error_terms == baseline.num_error_terms
    assert torch.equal(repaired.zonotope_w[:,0].view(torch.int64), baseline.zonotope_w[:,0].view(torch.int64))
    assert prepared["payload"]["native_transition"]["semantic_range_certificate"]["range_override_tokens"] == [1]


def test_only_declared_producer_revision_can_bypass_old_source_pins():
    import run_sound_fp64_separator_property_v1 as runner
    import run_transformer_benchmark24_v1 as benchmark
    audit = runner.source_audit(benchmark.read_protocol())
    assert set(audit["changed_sources"]) == runner.CHANGED_EXECUTION_FILES
    assert audit["benchmark_manifest_unchanged"]


def test_finish_domain_repair_retains_fast_diagnostics_identity():
    state, proof = toy()
    low = torch.ones(2,4, dtype=torch.float64)
    diagnostic = {"label": "test", "sound_variance_lower": 1., "domain_admissible": True}
    actual, same, prepared = finish._repair_layernorm_domain(state, proof, low, diagnostic)
    assert actual is low and same is diagnostic and prepared is None
    diagnostic = {**diagnostic, "sound_variance_lower": -1., "domain_admissible": False}
    actual, updated, prepared = finish._repair_layernorm_domain(state, proof, -low, diagnostic)
    assert updated["domain_admissible"] and updated["generic_sound_variance_lower"] == -1.
    assert diagnostic["domain_admissible"] is False and prepared["admissible"]


def test_witness_payload_survives_artifact_serialization(tmp_path):
    state, proof = toy()
    payload = S.prepare(state, proof, torch.full((2,4), -1., dtype=torch.float64), "ln")["payload"]
    path = tmp_path/"saved.pt"
    report = {"metric": 1., "_separating_variance_witnesses": [payload]}
    sound._save_artifact(path, "tiny", {}, report)
    saved = torch.load(path, weights_only=False)
    assert report == {"metric": 1.} and saved["report"] == report
    assert S.replay_payload(saved["separating_variance_witnesses"][0])


def test_prepared_global_lower_metadata_must_agree_with_replayed_bounds():
    state, proof = toy()
    generic = torch.full((2,4), -1., dtype=torch.float64)
    prepared = S.prepare(state, proof, generic, "ln")
    prepared["semantic_lower_by_token"] = [99., 99.]
    with pytest.raises(RuntimeError, match="metadata differs"):
        sound._checked_layernorm_low(generic, prepared)


def test_stage_adapter_preserves_full_witness_before_campaign_cleanup(tmp_path):
    import run_sound_fp64_3l_campaign as campaign
    state, proof = toy()
    witness = S.prepare(state, proof, torch.full((2,4), -1., dtype=torch.float64), "ln")["payload"]
    def stage(path, device):
        sound._save_artifact(path, "toy", {}, {"_separating_variance_witnesses": [witness]})
        return {"generic_fallback_count": 0}
    result = campaign._run_stage("ln", stage, (), tmp_path/"toy.pt", "toy", "cpu")
    preserved = result.pop("_separating_variance_witnesses")
    assert S.replay_payload(preserved[0])
