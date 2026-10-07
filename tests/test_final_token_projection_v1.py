"""Bounded CPU-only exact projection and final-head liveness regressions."""
from copy import deepcopy
from fractions import Fraction
import ast
import inspect
from pathlib import Path
import subprocess

import pytest
import torch

from test_sound_fp64_layernorm_separator_v1 import (
    sound, structural, finish, native_context, fp64_default)
import sound_fp64_final_token_projection_v1 as P
import final_token_projection_checker_v1 as C


IDENTITY = {"property_id": "synthetic", "rho_hex": (0.001).hex(),
            "checkpoint_sha256": "synthetic-only-not-a-frozen-model"}


def state_fixture(native_context, n=10, dead_amplitude=1.):
    Z, args = native_context
    w = torch.zeros((n+1, 9, 4), dtype=torch.float64)
    w[0] = torch.tensor([1., 2., 3., 4.])
    w[1, 8] = dead_amplitude
    w[2, 0, 0] = .25
    w[2, 8] = dead_amplitude  # Shared generator MUST retain its token-0 row.
    for r in range(3, n+1):
        w[r, 0, (r-3) % 4] = 1. / (2**r)
    # Token 8-only row represents the semantic-floor box case.
    ids = ("semantic_epsilon_floor::block2_output::000008::000000", "shared") + tuple(
        f"native_{i}" for i in range(n-2))
    reasons = ("semantic_layernorm_epsilon_floor_coordinate_box", "native") + tuple(
        "fp64_roundoff_coordinate_box" if i % 2 else "native" for i in range(n-2))
    proof = structural.SupportProof((256, 257) + (1,)*(n-2), ids, reasons, 9)
    z = Z(args=args, p=100, eps=.001, perturbed_word_index=8, zonotope_w=w,
          error_term_range_low=-torch.ones(n), error_term_range_high=torch.ones(n), clone=False)
    structural.attach_support(z, proof)
    return z, proof


def project(native_context, n=10, dead_amplitude=1.):
    z, proof = state_fixture(native_context, n, dead_amplitude)
    output, op, witness = P.project(z, proof, IDENTITY, deepcopy(C.REGION))
    return z, proof, output, op, witness


def check(witness, original):
    return C.verify(witness, original["input_state_identity"], original["output_state_identity"],
                    IDENTITY, deepcopy(C.REGION))


def test_exact_projection_ids_ranges_masks_and_numerical_classification(native_context):
    z, proof, out, op, w = project(native_context)
    assert z.num_error_terms == 10 and out.num_error_terms == 9
    assert out.num_words == op.num_tokens == 1
    assert w["retained_generator_indices"] == list(range(1, 10))
    assert w["zero_after_projection_discarded_generator_ids"] == [proof.ids[0]]
    assert op.ids == proof.ids[1:] and op.reasons == proof.reasons[1:]
    assert op.masks == (1,)*9
    assert torch.equal(out.zonotope_w.view(torch.int64),
                       z.zonotope_w[[0]+list(range(2,11)), :1].view(torch.int64))
    assert torch.equal(out.error_term_range_low.view(torch.int64),
                       z.error_term_range_low[1:].view(torch.int64))
    assert check(w, w)["verified"]
    assert structural.validate_support(out, op)["validated"]


def test_exact_affine_semantic_equality_without_reduction(native_context):
    z, _, out, _, _ = project(native_context, n=3)
    f = Fraction.from_float
    for xi in ((-1., .5, 1.), (1., -1., -.5), (0., 0., 0.)):
        for j in range(4):
            original = f(float(z.zonotope_w[0,0,j])) + sum(
                f(float(z.zonotope_w[i+1,0,j]))*f(x) for i,x in enumerate(xi))
            projected = f(float(out.zonotope_w[0,0,j])) + sum(
                f(float(out.zonotope_w[i+1,0,j]))*f(x) for i,x in enumerate(xi[1:]))
            assert original == projected


def test_native_head_without_reduction_is_bitwise_equivalent(native_context):
    z, _, out, _, _ = project(native_context, n=3)
    old = sound._make_like(z, z.zonotope_w[:, :1].clone(), *sound._ranges(z))
    from types import SimpleNamespace
    param = SimpleNamespace(weight=torch.eye(4), bias=torch.zeros(4))
    old_affine, new_affine = old.dense(param), out.dense(param)
    old_tanh, new_tanh = old_affine.tanh(), new_affine.tanh()
    assert torch.equal(old_tanh.zonotope_w[0].view(torch.int64), new_tanh.zonotope_w[0].view(torch.int64))
    assert torch.equal(old_tanh.zonotope_w[2:4].view(torch.int64), new_tanh.zonotope_w[1:3].view(torch.int64))
    assert torch.equal(old_tanh.zonotope_w[4:].view(torch.int64), new_tanh.zonotope_w[3:].view(torch.int64))
    for before, after in zip(old_tanh.concretize(), new_tanh.concretize()):
        assert torch.equal(before.view(torch.int64), after.view(torch.int64))


def test_dead_coefficients_cannot_influence_final_reduction(native_context, monkeypatch):
    monkeypatch.setattr(sound, "MAXIMUM_GENERATORS", 6)
    results = []
    for amplitude in (1., 2.**40):
        z, proof = state_fixture(native_context, dead_amplitude=amplitude)
        before = z.zonotope_w.view(torch.int64).clone()
        records = []
        out, op, witness = P.project_then_reduce(z, proof, IDENTITY, C.REGION, records)
        assert (witness["generator_count_before"], witness["generator_count_after"],
                witness["generator_count_after_reduction"]) == (10,9,6)
        assert torch.equal(z.zonotope_w.view(torch.int64), before)
        assert witness["subsequent_reduction_records"] == records
        results.append((out, op, records))
    assert torch.equal(results[0][0].zonotope_w.view(torch.int64), results[1][0].zonotope_w.view(torch.int64))
    assert results[0][1:] == results[1][1:]


def test_unnecessary_reduction_is_skipped(native_context, monkeypatch):
    monkeypatch.setattr(sound, "sound_reduce", lambda *_: pytest.fail("unexpected reduction"))
    z, proof = state_fixture(native_context, n=3)
    out, _, witness = P.project_then_reduce(z, proof, IDENTITY, C.REGION, [])
    assert (witness["generator_count_before"], witness["generator_count_after"],
            witness["generator_count_after_reduction"]) == (3,2,2)
    assert not witness["subsequent_reduction_records"]


def test_all_zero_rows_can_be_removed_and_signed_zero_is_preserved(native_context):
    z, proof = state_fixture(native_context, n=3)
    z.zonotope_w[1:, 0].zero_()
    z.zonotope_w[0, 0, 0] = -0.
    z.zonotope_w[1, 0, 0] = -0.
    out, op, witness = P.project(z, proof, IDENTITY, C.REGION)
    assert out.num_error_terms == 0 and not op.ids
    assert torch.equal(out.zonotope_w[0].view(torch.int64), z.zonotope_w[0,:1].view(torch.int64))
    assert len(witness["zero_after_projection_discarded_generator_ids"]) == 3
    assert check(witness, witness)["verified"]


@pytest.mark.parametrize("mutation", ["coefficient", "range", "id", "order", "reason", "mask",
    "drop_active", "token_set", "premature", "execution", "discarded_ids", "nonfinite"])
def test_independent_checker_rejects_tampering(native_context, mutation):
    _, _, _, _, original = project(native_context, n=3)
    bad = deepcopy(original)
    out = bad["output"]
    if mutation == "coefficient": out["weights"]["data"] = b"\0"*len(out["weights"]["data"])
    if mutation == "range": out["low"]["data"] = b"\0"*len(out["low"]["data"])
    if mutation == "id": out["proof"]["ids"][0] = "forged"
    if mutation == "order": out["proof"]["ids"].reverse()
    if mutation == "reason": out["proof"]["reasons"][0] = "forged"
    if mutation == "mask": out["proof"]["masks"][0] = 0
    if mutation == "drop_active": bad["retained_generator_indices"].pop(0)
    if mutation == "token_set": bad["retained_token_indices"] = [8]
    if mutation == "premature": bad["region_proof"]["remaining_semantic_operators"].insert(0,"attention_QK")
    if mutation == "execution": bad["execution_identity"]["rho_hex"] = (1.).hex()
    if mutation == "discarded_ids": bad["zero_after_projection_discarded_generator_ids"].append("shared")
    if mutation == "nonfinite":
        import struct
        out["weights"]["data"] = struct.pack("<d", float("nan")) + out["weights"]["data"][8:]
    with pytest.raises(RuntimeError): check(bad, original)


def test_drop_active_rejected_even_with_self_consistent_forged_output_hash(native_context):
    _, _, _, _, original = project(native_context, n=3)
    bad = deepcopy(original)
    out = bad["output"]
    out["weights"]["shape"][0] -= 1
    out["weights"]["data"] = out["weights"]["data"][:32] + out["weights"]["data"][64:]
    for key in ("low", "high"):
        out[key]["shape"] = [1]
        out[key]["data"] = out[key]["data"][8:]
    for key in ("ids", "masks", "reasons"): out["proof"][key].pop(0)
    forged_identity = C.state_identity(out)
    bad["output_state_identity"] = forged_identity
    with pytest.raises(RuntimeError, match="coefficients/ranges/ordered"):
        C.verify(bad, original["input_state_identity"], forged_identity, IDENTITY, C.REGION)


@pytest.mark.parametrize("field", ["weights", "low", "high"])
def test_self_consistent_output_hash_cannot_authorize_changed_coefficients_or_ranges(native_context, field):
    _, _, _, _, original = project(native_context, n=3)
    bad = deepcopy(original)
    out = bad["output"]
    # Source identity remains authenticated, even if a producer recomputes all
    # output hashes after modifying its claimed projection.
    out[field]["data"] = b"\0"*len(out[field]["data"])
    identity = C.state_identity(out)
    bad["output_state_identity"] = identity
    with pytest.raises(RuntimeError, match="coefficients/ranges/ordered"):
        C.verify(bad, original["input_state_identity"], identity, IDENTITY, C.REGION)


def test_source_mutation_always_fails_external_authentication(native_context):
    _, _, _, _, original = project(native_context, n=3)
    bad = deepcopy(original)
    bad["input"]["weights"]["data"] = b"\0"*len(bad["input"]["weights"]["data"])
    bad["input_state_identity"] = C.state_identity(bad["input"])
    with pytest.raises(RuntimeError, match="authenticated state"):
        check(bad, original)


def test_projection_forbidden_before_remaining_cross_token_operator(native_context):
    z, proof = state_fixture(native_context)
    region = deepcopy(C.REGION)
    region["completed_transformer_blocks"] = [0,1]
    with pytest.raises(RuntimeError, match="forbidden"):
        P.project(z, proof, IDENTITY, region)


def test_projection_inserted_only_after_final_ln_without_upstream_edits():
    root = Path(__file__).resolve().parents[1]
    name = "scripts/run_sound_fp64_finish_3l_v1.py"
    previous = subprocess.check_output(["git", "show", "HEAD:"+name], cwd=root,
                                       timeout=10, text=True)
    current = (root/name).read_text()
    # Exact unchanged execution prefix: all Block-2 operators and LN reserves.
    old_prefix = previous.split("def execute(",1)[1].split(
        '            output, output_proof = sound._maybe_reduce(\n'
        '                output, output_proof, "b2_output_layernorm", reductions)',1)[0]
    new_prefix = current.split("def execute(",1)[1].split(
        '            # The frozen head selects token 0;',1)[0]
    assert new_prefix == old_prefix
    remainder = current.split("final_projection.project_then_reduce(",1)[1]
    assert '"QK"' not in remainder.split("expected_dispatch",1)[0]
    assert 'output.zonotope_w[:, :1, :]' in remainder


def test_checker_does_not_import_producer_or_tensor_libraries():
    tree = ast.parse(inspect.getsource(C))
    modules = {n.module if isinstance(n, ast.ImportFrom) else a.name
               for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
               for a in (n.names if isinstance(n, ast.Import) else [None])}
    assert modules == {"hashlib", "json", "math", "struct"}


def test_projection_witness_roundtrip_and_layernorm_witness_unchanged(
        native_context, monkeypatch, tmp_path):
    from test_sound_fp64_epsilon_floor_v1 import prepared_zero, execute
    import semantic_epsilon_floor_checker_v1 as floor_check
    state, proof, prepared = prepared_zero(monkeypatch)
    out, op, _, _ = execute(native_context, state, proof, prepared)
    original = deepcopy(prepared["payload"])
    projected, projected_proof, witness = P.project(out, op, IDENTITY, C.REGION)
    assert prepared["payload"] == original
    archive = tmp_path/"projection.pt"
    torch.save(witness, archive)
    assert check(torch.load(archive, weights_only=False), witness)["verified"]
    saved = tmp_path/"certificate.pt"
    sound._save_artifact(saved, "tiny", {"margin": (projected, projected_proof)},
                         {"_separating_variance_witnesses": [prepared["payload"]]})
    loaded = torch.load(saved, weights_only=False)["separating_variance_witnesses"][0]
    assert floor_check.replay_persisted(loaded, original["source_state_identity"],
        original["parameter_hashes"], original["source_domain"], original["output_state_identity"])["verified"]


def test_certified_separator_archive_still_replays_after_projection(native_context, tmp_path):
    from types import SimpleNamespace
    from test_sound_fp64_layernorm_separator_v1 import S, toy
    state, proof = toy()
    Z, args = native_context
    z = Z(args=args, p=100, eps=.001, perturbed_word_index=0,
          zonotope_w=state.zonotope_w, error_term_range_low=state.error_term_range_low,
          error_term_range_high=state.error_term_range_high, clone=False)
    structural.attach_support(z, proof)
    prepared = S.prepare(z, proof, -torch.ones(2,4), "separator_then_projection")
    dispatch = sound.production.NativeProductionDispatch(delegate=structural.StructuralNativeSemanticOperators())
    dispatch.delegate._layer_norm_index = 6
    parameter = SimpleNamespace(weight=torch.ones(4), bias=torch.zeros(4))
    out, op, _ = S.execute_prepared(dispatch, z, proof, parameter, prepared)
    before = out.zonotope_w.view(torch.int64).clone()
    projected, projected_proof, _ = P.project(out, op, IDENTITY, C.REGION)
    assert torch.equal(out.zonotope_w.view(torch.int64), before)
    path = tmp_path/"separator.pt"
    sound._save_artifact(path, "tiny", {"margin": (projected, projected_proof)},
                         {"_separating_variance_witnesses": [prepared["payload"]]})
    loaded = torch.load(path, weights_only=False)["separating_variance_witnesses"][0]
    assert S.replay_payload(loaded)


def test_campaign_result_keeps_projection_reference_on_negative_margin(monkeypatch, tmp_path):
    from test_sound_fp64_margin_results_v1 import fake_property_path, campaign
    row = fake_property_path(monkeypatch, -1., 1.)
    stub = campaign.finish3l.execute
    def finish_with_projection(*args, **kwargs):
        result = stub(*args, **kwargs)
        result.update(final_token_projection={"retained_token_indices": [0]},
                      final_token_projection_witness_path="synthetic/projection.pt",
                      final_token_projection_witness_sha256="synthetic-sha")
        return result
    monkeypatch.setattr(campaign.finish3l, "execute", finish_with_projection)
    result = campaign.execute_property(row, tmp_path, "cuda:0")
    assert result["classification"] == "FAILED_AT_HISTORICAL_RADIUS"
    assert result["final_token_projection"]["retained_token_indices"] == [0]
    assert result["final_token_projection_witness_path"] == "synthetic/projection.pt"
    assert result["final_token_projection_witness_sha256"] == "synthetic-sha"
