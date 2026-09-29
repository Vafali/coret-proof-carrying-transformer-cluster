from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research_hab"))

import coret_native_numerical_checker_v1 as checker
import coret_native_numerical_witness_v1 as witness_io
import coret_structural_support_precise_dot_v1 as structural


def canonical(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def reseal(value):
    value.pop("canonical_sha256", None)
    value["canonical_sha256"] = canonical(value)


class Proxy:
    def __init__(self, weights):
        self.zonotope_w = weights
        self.error_term_range_low = None
        self.error_term_range_high = None
        self.num_input_error_terms_special_norm = 0
        self.p = 100
        self.eps = 1.0 / 1600.0
        self.perturbed_word_index = 0

    @property
    def num_error_terms(self):
        return self.zonotope_w.shape[1] - 1

    @property
    def num_words(self):
        return self.zonotope_w.shape[2]

    @property
    def word_embedding_size(self):
        return self.zonotope_w.shape[3]


class AffineProxy:
    def __init__(self, weights):
        self.zonotope_w = weights
        self.error_term_range_low = None
        self.error_term_range_high = None
        self.num_input_error_terms_special_norm = 0

    @property
    def num_error_terms(self):
        return self.zonotope_w.shape[0] - 1


@pytest.fixture(autouse=True)
def proxy_factory(monkeypatch):
    package = types.ModuleType("Verifiers")
    module = types.ModuleType("Verifiers.Zonotope")
    module.make_zonotope_new_weights_same_args = (
        lambda new_weights, source_zonotope, clone=False:
        Proxy(new_weights.clone() if clone else new_weights))
    package.Zonotope = module
    monkeypatch.setitem(sys.modules, "Verifiers", package)
    monkeypatch.setitem(sys.modules, "Verifiers.Zonotope", module)


def _weights(masks, heads, rows, columns, token_axis, seed):
    torch.manual_seed(seed)
    value = torch.zeros(heads, 1 + len(masks), rows, columns,
                        dtype=torch.float32)
    value[:, 0] = torch.randn(heads, rows, columns)
    for generator, mask in enumerate(masks):
        for token in range(rows if token_axis == 2 else columns):
            if not mask & (1 << token):
                continue
            if token_axis == 2:
                value[:, 1 + generator, token, :] = torch.randn(
                    heads, columns)
            else:
                value[:, 1 + generator, :, token] = torch.randn(heads, rows)
    return value


def _output_proof(mode, temporary, left, right, call=0):
    gmax = max(len(left.masks), len(right.masks))
    ids = []
    for index in range(gmax):
        li = left.ids[index] if index < len(left.ids) else None
        ri = right.ids[index] if index < len(right.ids) else None
        assert li is None or ri is None or li == ri
        ids.append(li if li is not None else ri)
    fresh = len(temporary.masks) - gmax
    prefix = "qk" if mode == "QK" else "av"
    return structural.SupportProof(
        temporary.masks,
        tuple(ids) + tuple(
            f"{prefix}_{call}_fresh_{index:06d}" for index in range(fresh)),
        temporary.reasons,
        temporary.num_tokens)


def make_case(tmp_path, mode):
    store = witness_io.ContentAddressedWitnessStore(tmp_path)
    if mode == "QK":
        masks = (1, 2, 3)
        left_proof = structural.SupportProof(
            masks, ("shared_0", "shared_1", "shared_2"),
            ("fixture",) * 3, 2)
        right_proof = left_proof
        left = Proxy(_weights(masks, 2, 2, 3, 2, 1))
        right = Proxy(_weights(masks, 2, 2, 3, 2, 2))
        kwargs = {"mode": mode}
    else:
        left_masks, right_masks = (3, 1, 2), (1, 2)
        left_proof = structural.SupportProof(
            left_masks, ("shared_0", "shared_1", "probability_2"),
            ("fixture",) * 3, 2)
        right_proof = structural.SupportProof(
            right_masks, ("shared_0", "shared_1"),
            ("fixture",) * 2, 2)
        left = Proxy(_weights(left_masks, 2, 2, 2, 2, 3))
        # The precise-dot right operand is already V^T: [H,G,F,K].
        right = Proxy(_weights(right_masks, 2, 3, 2, 3, 4))
        kwargs = {"mode": mode, "generator_tile": 2}
    before_left = left.zonotope_w.clone()
    before_right = right.zonotope_w.clone()
    output = structural.precise_dot_structural(
        left, right, left_proof, right_proof, **kwargs)
    frozen_output = output.zonotope_w.clone()
    output_proof = _output_proof(
        mode, structural.get_support(output), left_proof, right_proof)
    structural.attach_support(output, output_proof)
    record = store.precise_dot(
        mode=mode, left=left, right=right, output=output,
        left_support=left_proof, right_support=right_proof,
        output_support=output_proof, call_index=0)
    assert torch.equal(left.zonotope_w, before_left)
    assert torch.equal(right.zonotope_w, before_right)
    assert torch.equal(output.zonotope_w, frozen_output)
    return store, record, left, right, output, left_proof, right_proof, output_proof


@pytest.mark.parametrize("mode", ["QK", "A.V"])
def test_valid_precise_dot_witness_and_producer_bits_unchanged(tmp_path, mode):
    _store, record, *_ = make_case(tmp_path, mode)
    report = checker.check_precise_dot_witness(record, tmp_path)
    assert report["accepted"] is True
    assert report["precision_bits"] == 256
    assert report["strict_no_empirical_epsilon"] is True


@pytest.mark.parametrize("mode", ["QK", "A.V"])
def test_rigorous_fp64_backend_valid_and_producer_bits_unchanged(
        tmp_path, mode, monkeypatch):
    monkeypatch.setenv("CORET_PRECISE_DOT_NUMERICAL_BACKEND", "rigorous_fp64")
    monkeypatch.setenv("CORET_CUDA_DRIVER", "/usr/lib/wsl/lib/libcuda.so.1")
    _store, record, _left, _right, output, *_ = make_case(tmp_path, mode)
    frozen = output.zonotope_w.clone()
    report = checker.check_precise_dot_witness(record, tmp_path)
    assert report["accepted"] is True
    assert report["numerical_backend"] == "checker_only_rigorous_fp64"
    assert report["backend"]["accepted"] is True
    assert torch.equal(output.zonotope_w, frozen)


def test_rigorous_fp64_numeric_narrowing_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("CORET_PRECISE_DOT_NUMERICAL_BACKEND", "rigorous_fp64")
    monkeypatch.setenv("CORET_CUDA_DRIVER", "/usr/lib/wsl/lib/libcuda.so.1")
    store, _record, left, right, output, lp, rp, op = make_case(tmp_path, "QK")
    mutated = output.zonotope_w.clone()
    gmax = max(left.num_error_terms, right.num_error_terms)
    mutated[0, 1 + gmax, 0, 0] = 0.0
    changed = Proxy(mutated)
    candidate = store.precise_dot(
        mode="QK", left=left, right=right, output=changed,
        left_support=lp, right_support=rp, output_support=op, call_index=0)
    # This mutation models a newly generated, internally self-consistent
    # certificate.  The checker must reject it numerically, not by commitment.
    candidate["frozen_output_sha256"] = candidate["output"]["sha256"]
    reseal(candidate)
    with pytest.raises(AssertionError, match="rigorous FP64 transition"):
        checker.check_precise_dot_witness(candidate, tmp_path)


def test_rigorous_fp64_stale_zero_claim_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("CORET_PRECISE_DOT_NUMERICAL_BACKEND", "rigorous_fp64")
    monkeypatch.setenv("CORET_CUDA_DRIVER", "/usr/lib/wsl/lib/libcuda.so.1")
    store, _record, left, right, output, lp, rp, op = make_case(tmp_path, "QK")
    mutated = left.zonotope_w.clone()
    # Generator zero has support only on token zero.  Insert the smallest
    # positive binary32 coefficient at token one while leaving stale metadata.
    mutated[0, 1, 1, 0] = torch.nextafter(
        torch.tensor(0.0), torch.tensor(float("inf")))
    changed = Proxy(mutated)
    candidate = store.precise_dot(
        mode="QK", left=changed, right=right, output=output,
        left_support=lp, right_support=rp, output_support=op, call_index=0)
    candidate["frozen_output_sha256"] = candidate["output"]["sha256"]
    reseal(candidate)
    with pytest.raises(AssertionError, match="outside claimed support"):
        checker.check_precise_dot_witness(candidate, tmp_path)


@pytest.mark.parametrize("mode", ["QK", "A.V"])
def test_one_ulp_narrow_precise_dot_output_rejected(tmp_path, mode):
    store, record, left, right, output, lp, rp, op = make_case(tmp_path, mode)
    mutated = output.zonotope_w.clone()
    gmax = max(left.num_error_terms, right.num_error_terms)
    index = 1 + gmax
    mutated[0, index, 0, 0] = torch.nextafter(
        mutated[0, index, 0, 0], torch.tensor(float("-inf")))
    changed = Proxy(mutated)
    changed.error_term_range_low = output.error_term_range_low
    changed.error_term_range_high = output.error_term_range_high
    candidate = store.precise_dot(
        mode=mode, left=left, right=right, output=changed,
        left_support=lp, right_support=rp, output_support=op, call_index=0)
    # Preserve the immutable producer-output commitment while substituting an
    # otherwise self-consistent one-ULP-narrow block.
    candidate["frozen_output_sha256"] = record["frozen_output_sha256"]
    reseal(candidate)
    with pytest.raises(AssertionError, match="output commitment"):
        checker.check_precise_dot_witness(candidate, tmp_path)


def test_support_provenance_order_and_range_mutations_rejected(tmp_path):
    _store, record, *_ = make_case(tmp_path, "QK")
    support = copy.deepcopy(record)
    support["output_support"]["masks"][0] = 0
    reseal(support["output_support"]); reseal(support)
    with pytest.raises(AssertionError, match="support transition"):
        checker.check_precise_dot_witness(support, tmp_path)

    provenance = copy.deepcopy(record)
    provenance["output_support"]["ids"][0] = "wrong_id"
    reseal(provenance["output_support"]); reseal(provenance)
    with pytest.raises(AssertionError, match="provenance order"):
        checker.check_precise_dot_witness(provenance, tmp_path)

    order = copy.deepcopy(record)
    order["left_support"]["ids"][:2] = reversed(
        order["left_support"]["ids"][:2])
    reseal(order["left_support"]); reseal(order)
    with pytest.raises(AssertionError, match="provenance IDs differ"):
        checker.check_precise_dot_witness(order, tmp_path)

    ranged = copy.deepcopy(record)
    ranged["left_ranges"] = {
        "kind": "explicit", "count": 3,
        "low_hex": [(-0.5).hex(), (-1.0).hex(), (-1.0).hex()],
        "high_hex": [(1.0).hex(), (1.0).hex(), (1.0).hex()],
    }
    reseal(ranged["left_ranges"]); reseal(ranged)
    with pytest.raises(AssertionError, match="range policy"):
        checker.check_precise_dot_witness(ranged, tmp_path)


@pytest.mark.parametrize("family,input_interval", [
    ("sqrt", (2.0, 3.0)),
    ("reciprocal", (2.0, 3.0)),
    ("exp", (-1.0, 0.5)),
    ("tanh", (-1.0, 0.5)),
])
def test_valid_directed_scalar_certificate_passes(family, input_interval):
    lo, hi = input_interval
    if family == "sqrt":
        f = math.sqrt
    elif family == "reciprocal":
        f = lambda x: 1.0 / x
        lo, hi = input_interval
        output_lo, output_hi = f(hi), f(lo)
        record = witness_io.scalar_interval_witness(
            family, lo, hi, math.nextafter(output_lo, -math.inf),
            math.nextafter(output_hi, math.inf))
        assert checker.check_scalar_interval_witness(record)["accepted"]
        return
    elif family == "exp":
        f = math.exp
    else:
        f = math.tanh
    record = witness_io.scalar_interval_witness(
        family, lo, hi, math.nextafter(f(lo), -math.inf),
        math.nextafter(f(hi), math.inf))
    assert checker.check_scalar_interval_witness(record)["accepted"]


def test_one_ulp_inward_scalar_endpoint_rejected():
    outward = math.nextafter(math.sqrt(2.0), -math.inf)
    record = witness_io.scalar_interval_witness(
        "sqrt", 2.0, 2.0, outward, math.nextafter(math.sqrt(2.0), math.inf))
    assert checker.check_scalar_interval_witness(record)["accepted"]
    mutated = copy.deepcopy(record)
    mutated["output_lower_hex"] = math.nextafter(
        outward, math.inf).hex()
    reseal(mutated)
    with pytest.raises(AssertionError, match="does not contain"):
        checker.check_scalar_interval_witness(mutated)


def test_optional_production_qk_hook_changes_only_certificate(tmp_path):
    masks = (1, 2, 3)
    proof = structural.SupportProof(
        masks, ("shared_0", "shared_1", "shared_2"),
        ("fixture",) * 3, 2)
    left = Proxy(_weights(masks, 1, 2, 3, 2, 11))
    right = Proxy(_weights(masks, 1, 2, 3, 2, 12))
    plain = structural.StructuralNativeSemanticOperators()
    plain._hidden = proof
    plain_result = plain.qk(left, right)

    store = witness_io.ContentAddressedWitnessStore(tmp_path)
    carrying = structural.StructuralNativeSemanticOperators(
        numerical_witness_store=store)
    carrying._hidden = proof
    carrying_result = carrying.qk(left, right)
    assert torch.equal(
        plain_result.value.zonotope_w, carrying_result.value.zonotope_w)
    assert structural.get_support(plain_result.value) == structural.get_support(
        carrying_result.value)
    assert "independent_numerical_witness" not in plain_result.certificate
    numerical = carrying_result.certificate["independent_numerical_witness"]
    assert checker.check_precise_dot_witness(numerical, tmp_path)["accepted"]


def _native_sqrt_float32(source):
    center, errors = source[0], source[1:]
    lower = center - errors.abs().sum(dim=0)
    upper = center + errors.abs().sum(dim=0)
    different = lower != upper
    tcrit = ((upper - lower) /
             (2 * (upper.sqrt() - lower.sqrt()))).square()
    slope = (upper.sqrt() - lower.sqrt()) / (upper - lower)
    intercept = lower.sqrt() - slope * lower
    constant = 0.5 * (tcrit.sqrt() - slope * tcrit + intercept)
    fresh = 0.5 * (slope * tcrit - tcrit.sqrt() + intercept)
    output = torch.zeros(
        1 + errors.shape[0] + int(different.sum()), *center.shape,
        dtype=source.dtype)
    output[0, different] = slope[different] * center[different] + constant[different]
    output[0, ~different] = lower[~different].sqrt()
    output[1:1 + errors.shape[0], different] = (
        errors[:, different] * slope[different])
    rows = torch.arange(1 + errors.shape[0], output.shape[0])
    output[rows, different] = fresh[different]
    indices = torch.arange(different.numel()).reshape(different.shape)[different]
    return output, indices.tolist()


def test_native_affine_sqrt_outward_fixture_passes_and_mutation_rejects(tmp_path):
    source = AffineProxy(torch.tensor([
        [[0.625]],
        [[0.125]],
    ], dtype=torch.float32))
    raw, active = _native_sqrt_float32(source.zonotope_w)
    # This test fixture deliberately supplies a generous coordinate-local
    # outward radius.  The separate regression below proves raw native float32
    # does not generally have this property.
    raw[-1, 0, 0] = raw[-1, 0, 0].abs() + 1e-5
    output = AffineProxy(raw)
    store = witness_io.ContentAddressedWitnessStore(tmp_path)
    record = store.native_affine_sqrt(
        source=source, output=output, fresh_flat_indices=active)
    assert checker.check_native_affine_sqrt_witness(
        record, tmp_path)["accepted"]

    changed = output.zonotope_w.clone()
    changed[-1, 0, 0] = 0.0
    mutated = store.native_affine_sqrt(
        source=source, output=AffineProxy(changed), fresh_flat_indices=active)
    with pytest.raises(AssertionError, match="does not enclose"):
        checker.check_native_affine_sqrt_witness(mutated, tmp_path)


def test_unchanged_pinned_float32_sqrt_exposes_exact_real_blocker(tmp_path):
    # Deterministic counterexample found by bounded equation-level search.
    # It uses the exact pinned formula but no verifier/model/bound entrypoint.
    source = AffineProxy(torch.tensor([
        [[0.6240205764770508]],
        [[0.4759345054626465]],
    ], dtype=torch.float32))
    raw, active = _native_sqrt_float32(source.zonotope_w)
    output = AffineProxy(raw)
    store = witness_io.ContentAddressedWitnessStore(tmp_path)
    record = store.native_affine_sqrt(
        source=source, output=output, fresh_flat_indices=active)
    with pytest.raises(AssertionError, match="does not enclose"):
        checker.check_native_affine_sqrt_witness(record, tmp_path)


def test_known_sqrt_counterexample_minimal_fresh_widening_passes(tmp_path):
    source = AffineProxy(torch.tensor([
        [[0.6240205764770508]],
        [[0.4759345054626465]],
    ], dtype=torch.float32))
    raw, active = _native_sqrt_float32(source.zonotope_w)
    # Smallest binary32 magnitude at or above the independently reconstructed
    # MPFR requirement for this frozen counterexample.  This is a fixture
    # oracle only; production must derive its own conservative bound.
    required = 0.038439254755346358331428922332805167389327843932867935
    widened = torch.tensor(required, dtype=torch.float32)
    if float(widened) < required:
        widened = torch.nextafter(
            widened, torch.tensor(math.inf, dtype=torch.float32))
    raw[-1, 0, 0] = torch.copysign(widened, raw[-1, 0, 0])
    output = AffineProxy(raw)
    store = witness_io.ContentAddressedWitnessStore(tmp_path)
    record = store.native_affine_sqrt(
        source=source, output=output, fresh_flat_indices=active)
    assert checker.check_native_affine_sqrt_witness(record, tmp_path)["accepted"]

    inward = raw.clone()
    smaller = torch.nextafter(
        widened, torch.tensor(-math.inf, dtype=torch.float32))
    inward[-1, 0, 0] = torch.copysign(smaller, inward[-1, 0, 0])
    candidate = store.native_affine_sqrt(
        source=source, output=AffineProxy(inward), fresh_flat_indices=active)
    with pytest.raises(AssertionError, match="does not enclose"):
        checker.check_native_affine_sqrt_witness(candidate, tmp_path)


def test_irrational_singleton_proves_membership_preservation_blocker(tmp_path):
    # Native membership is empty when l == u.  sqrt(2) is not representable as
    # binary32, so the unchanged one-point output has no symbol into which its
    # representation error can be charged.
    source = AffineProxy(torch.tensor([[[2.0]]], dtype=torch.float32))
    raw, active = _native_sqrt_float32(source.zonotope_w)
    assert active == []
    output = AffineProxy(raw)
    store = witness_io.ContentAddressedWitnessStore(tmp_path)
    record = store.native_affine_sqrt(
        source=source, output=output, fresh_flat_indices=active)
    with pytest.raises(AssertionError, match="does not enclose"):
        checker.check_native_affine_sqrt_witness(record, tmp_path)


def test_native_affine_sqrt_membership_and_range_mutations_reject(tmp_path):
    source = AffineProxy(torch.tensor([
        [[1.0, 4.0]],
        [[0.0, 0.5]],
    ], dtype=torch.float32))
    raw, active = _native_sqrt_float32(source.zonotope_w)
    raw[-1, 0, 1] = raw[-1, 0, 1].abs() + 1e-5
    output = AffineProxy(raw)
    store = witness_io.ContentAddressedWitnessStore(tmp_path)
    record = store.native_affine_sqrt(
        source=source, output=output, fresh_flat_indices=active)
    assert checker.check_native_affine_sqrt_witness(record, tmp_path)["accepted"]

    membership = copy.deepcopy(record)
    membership["fresh_flat_indices"] = [0]
    reseal(membership)
    with pytest.raises(AssertionError, match="membership/order"):
        checker.check_native_affine_sqrt_witness(membership, tmp_path)

    narrowed = copy.deepcopy(record)
    narrowed["output_ranges"] = {
        "kind": "explicit", "count": 2,
        "low_hex": [(-1.0).hex(), (-0.5).hex()],
        "high_hex": [(1.0).hex(), (1.0).hex()],
    }
    reseal(narrowed["output_ranges"]); reseal(narrowed)
    with pytest.raises(AssertionError, match="range policy"):
        checker.check_native_affine_sqrt_witness(narrowed, tmp_path)


def _ceil_binary64(value):
    candidate = float(value)
    if checker._mp(candidate) < value:
        candidate = math.nextafter(candidate, math.inf)
    return candidate


def _required_symmetric_radius(stored, interval):
    stored_mp = checker._mp(stored)
    lower_gap = checker._up(lambda: stored_mp - interval.lo)
    upper_gap = checker._up(lambda: interval.hi - stored_mp)
    return _ceil_binary64(max(lower_gap, upper_gap))


def _shadow_chain_record(input_value):
    source = checker.Interval(checker._mp(input_value))
    exact_sqrt = source.sqrt()
    stored_sqrt = float(torch.sqrt(torch.tensor(
        input_value, dtype=torch.float32)))
    sqrt_radius = _required_symmetric_radius(stored_sqrt, exact_sqrt)
    represented_sqrt = checker._shadow_interval(
        checker.Interval(checker._mp(stored_sqrt)),
        checker.Interval(checker._mp(sqrt_radius)), "fixture sqrt")

    reciprocal_image = represented_sqrt.reciprocal()
    stored_reciprocal = float(torch.reciprocal(torch.tensor(
        stored_sqrt, dtype=torch.float32)))
    reciprocal_radius = _required_symmetric_radius(
        stored_reciprocal, reciprocal_image)
    represented_reciprocal = checker._shadow_interval(
        checker.Interval(checker._mp(stored_reciprocal)),
        checker.Interval(checker._mp(reciprocal_radius)),
        "fixture reciprocal")

    scale, shift = 1.25, -0.125
    affine_image = (checker.Interval(checker._mp(scale))
                    * represented_reciprocal
                    + checker.Interval(checker._mp(shift)))
    stored_output = float(
        torch.tensor(scale, dtype=torch.float32)
        * torch.tensor(stored_reciprocal, dtype=torch.float32)
        + torch.tensor(shift, dtype=torch.float32))
    output_radius = _required_symmetric_radius(stored_output, affine_image)
    return witness_io.numerical_shadow_scalar_chain_witness(
        input_value=input_value,
        stored_sqrt=stored_sqrt,
        sqrt_error_radius=sqrt_radius,
        stored_reciprocal=stored_reciprocal,
        reciprocal_error_radius=reciprocal_radius,
        scale=scale,
        shift=shift,
        stored_output=stored_output,
        output_error_radius=output_radius)


@pytest.mark.parametrize("value", [2.0, 3.0, 5.0, 7.0, 10.0])
def test_numerical_shadow_represents_singleton_sqrt_chain(value):
    record = _shadow_chain_record(value)
    result = checker.check_numerical_shadow_scalar_chain_witness(record)
    assert result["accepted"]
    assert result["native_fresh_counts"] == [0, 0, 0]

    narrowed = copy.deepcopy(record)
    radius = float.fromhex(narrowed["sqrt_error_radius_hex"])
    narrowed["sqrt_error_radius_hex"] = math.nextafter(
        radius, -math.inf).hex()
    reseal(narrowed)
    with pytest.raises(AssertionError, match="sqrt numerical sidecar is too narrow"):
        checker.check_numerical_shadow_scalar_chain_witness(narrowed)


def test_numerical_shadow_composition_cannot_drop_predecessor_error():
    record = _shadow_chain_record(2.0)
    narrowed = copy.deepcopy(record)
    radius = float.fromhex(narrowed["reciprocal_error_radius_hex"])
    narrowed["reciprocal_error_radius_hex"] = math.nextafter(
        radius, -math.inf).hex()
    reseal(narrowed)
    with pytest.raises(AssertionError, match="reciprocal numerical sidecar is too narrow"):
        checker.check_numerical_shadow_scalar_chain_witness(narrowed)

    narrowed = copy.deepcopy(record)
    radius = float.fromhex(narrowed["output_error_radius_hex"])
    narrowed["output_error_radius_hex"] = math.nextafter(
        radius, -math.inf).hex()
    reseal(narrowed)
    with pytest.raises(
            AssertionError,
            match="LayerNorm affine output numerical sidecar is too narrow"):
        checker.check_numerical_shadow_scalar_chain_witness(narrowed)


@pytest.mark.parametrize("center,width", [
    (0.6240205764770508, 0.4759345054626465),
    (2.0, 0.00048828125),
    (2.0, 0.5),
])
def test_native_affine_sqrt_shadow_preserves_topology_and_covers_fp_gap(
        tmp_path, center, width):
    source = AffineProxy(torch.tensor([
        [[center]],
        [[width]],
    ], dtype=torch.float32))
    raw, active = _native_sqrt_float32(source.zonotope_w)
    frozen_bits = raw.view(torch.int32).clone()
    output = AffineProxy(raw)
    store = witness_io.ContentAddressedWitnessStore(tmp_path)

    probe = store.native_affine_sqrt_shadow(
        source=source, output=output, fresh_flat_indices=active,
        coordinate_numerical_radii=[1.0])
    probed = checker.check_native_affine_sqrt_shadow_witness(probe, tmp_path)
    deficit = checker._mp(probed["maximum_deficit"])
    assert deficit > 0
    radius = _ceil_binary64(deficit)
    record = store.native_affine_sqrt_shadow(
        source=source, output=output, fresh_flat_indices=active,
        coordinate_numerical_radii=[radius])
    result = checker.check_native_affine_sqrt_shadow_witness(record, tmp_path)
    assert result["accepted"]
    assert result["representation"] == (
        "native_plus_coordinate_numerical_sidecar")
    assert torch.equal(raw.view(torch.int32), frozen_bits)

    narrowed = store.native_affine_sqrt_shadow(
        source=source, output=output, fresh_flat_indices=active,
        coordinate_numerical_radii=[math.nextafter(radius, -math.inf)])
    with pytest.raises(AssertionError, match="does not enclose"):
        checker.check_native_affine_sqrt_shadow_witness(narrowed, tmp_path)


def test_native_affine_singleton_shadow_uses_no_native_fresh_symbol(tmp_path):
    source = AffineProxy(torch.tensor([[[2.0]]], dtype=torch.float32))
    raw, active = _native_sqrt_float32(source.zonotope_w)
    assert active == [] and raw.shape[0] == 1
    output = AffineProxy(raw)
    store = witness_io.ContentAddressedWitnessStore(tmp_path)
    probe = store.native_affine_sqrt_shadow(
        source=source, output=output, fresh_flat_indices=active,
        coordinate_numerical_radii=[1.0])
    result = checker.check_native_affine_sqrt_shadow_witness(probe, tmp_path)
    radius = _ceil_binary64(checker._mp(result["maximum_deficit"]))
    record = store.native_affine_sqrt_shadow(
        source=source, output=output, fresh_flat_indices=active,
        coordinate_numerical_radii=[radius])
    accepted = checker.check_native_affine_sqrt_shadow_witness(record, tmp_path)
    assert accepted["accepted"] and accepted["fresh_count"] == 0


def test_ghost_representation_distinguishes_native_square_decomposition():
    one = witness_io.ghost_precise_square_witness(
        center=0.0, coefficients=[("epsilon", 1.0)],
        stored_center=0.5, stored_retained=[("epsilon", 0.0)],
        stored_fresh=0.5)
    two = witness_io.ghost_precise_square_witness(
        center=0.0,
        coefficients=[("epsilon_1", 0.5), ("epsilon_2", 0.5)],
        stored_center=0.25,
        stored_retained=[("epsilon_1", 0.0), ("epsilon_2", 0.0)],
        stored_fresh=0.75)
    one_result = checker.check_ghost_precise_square_witness(one)
    two_result = checker.check_ghost_precise_square_witness(two)
    assert checker._mp(one_result["abstract_lower"]) == 0
    assert checker._mp(one_result["abstract_upper"]) == 1
    assert checker._mp(two_result["abstract_lower"]) == checker._mp("-0.5")
    assert checker._mp(two_result["abstract_upper"]) == 1
    assert one_result["ghost_ids"] == ["epsilon"]
    assert two_result["ghost_ids"] == ["epsilon_1", "epsilon_2"]


def test_ghost_representation_mutations_reject():
    record = witness_io.ghost_precise_square_witness(
        center=0.0,
        coefficients=[("epsilon_1", 0.5), ("epsilon_2", 0.5)],
        stored_center=0.25,
        stored_retained=[("epsilon_1", 0.0), ("epsilon_2", 0.0)],
        stored_fresh=0.75)

    reordered = copy.deepcopy(record)
    reordered["stored_retained"].reverse()
    reseal(reordered)
    with pytest.raises(AssertionError, match="identity/order"):
        checker.check_ghost_precise_square_witness(reordered)

    substituted = copy.deepcopy(record)
    substituted["coefficients"][1]["ghost_id"] = "epsilon_1"
    substituted["stored_retained"][1]["ghost_id"] = "epsilon_1"
    reseal(substituted)
    with pytest.raises(AssertionError, match="duplicate ghost"):
        checker.check_ghost_precise_square_witness(substituted)

    narrowed = copy.deepcopy(record)
    narrowed["stored_fresh_hex"] = math.nextafter(0.75, -math.inf).hex()
    reseal(narrowed)
    with pytest.raises(AssertionError, match="fresh radius is too narrow"):
        checker.check_ghost_precise_square_witness(narrowed)
