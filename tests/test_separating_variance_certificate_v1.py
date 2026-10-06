"""Small CPU fixtures only; never load the real captured state or run a property."""
import ast
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import separating_variance_checker_v1 as K
import certify_block2_output_separating_variance_v1 as S
from test_benchmark24_block2_output_zero_variance_v1 import (
    capture_fixture, rewrite_artifact)


def tiny():
    return ([2., 0., 0.], [[.5, 0., 0.], [0., 0., 0.]],
            [-1., -.5], [1., 2.], ["native", "numerical"])


def certificate():
    return K.construct_certificate(*tiny(), [Fraction(1), Fraction(0)], {"fixture": "tiny"})


def seal(cert):
    cert["certificate_sha256"] = K.digest({k:v for k,v in cert.items() if k != "certificate_sha256"})


def test_exact_separator_interval_norm_and_variance():
    cert = certificate()
    assert K.read_rational(cert["interval_lower"]) == Fraction(3, 2)
    assert K.read_rational(cert["interval_upper"]) == Fraction(5, 2)
    assert K.read_rational(cert["norm_v_squared"]) == 2
    assert K.read_rational(cert["variance_lower"]) == Fraction(3, 8)
    assert cert["v_rationals"] == [K.rational(Fraction(1)), K.rational(Fraction(0)), K.rational(Fraction(-1))]
    assert K.verify_certificate(*tiny(), {"fixture": "tiny"}, cert)["all_generators_replayed"] == 2
    # Direct exact variance at both endpoints must contain the certified bound.
    for xi in (Fraction(-1), Fraction(1)):
        x = [Fraction(2) + xi/2, Fraction(0), Fraction(0)]
        mean = sum(x)/3
        assert sum((a-mean)**2 for a in x)/3 >= Fraction(3, 8)


def test_negative_interval_asymmetric_box_and_rational_y():
    args = ([-2., 0.], [[.25, 0.]], [-2.], [.5], ["g"])
    cert = K.construct_certificate(*args, [Fraction(1, 3)], {})
    assert K.read_rational(cert["interval_lower"]) == Fraction(-5, 6)
    assert K.read_rational(cert["interval_upper"]) == Fraction(-5, 8)
    assert K.verify_certificate(*args, {}, cert)["verified"]


def test_no_separator_when_zero_inside_or_on_interval_boundary():
    for center in ([0., 0.], [1., 0.]):
        assert K.construct_certificate(center, [[1., 0.]], [-1.], [1.], ["g"], [Fraction(1)], {}) is None
    with pytest.raises(RuntimeError, match="zero"):
        K.construct_certificate(*tiny(), [Fraction(0), Fraction(0)], {})


def test_exact_original_binary64_not_rounded_differences():
    cert = K.construct_certificate([1., 2.**-54], [[0., 0.]], [-1.], [1.], ["g"], [Fraction(1)], {})
    assert K.read_rational(cert["center_functional"]) == 1-Fraction(1, 2**54)
    assert K.downward_binary64(K.read_rational(cert["variance_lower"])) <= K.read_rational(cert["variance_lower"])


def test_integer_aligned_support_matches_independent_fraction_sum():
    # Include subnormals, cancellation, different dyadic denominators and a
    # non-dyadic rational direction. This is only 4x4 arithmetic, not a state run.
    v = [Fraction(1, 3), Fraction(-7, 5), Fraction(2, 9), Fraction(38, 45)]
    assert sum(v) == 0
    denominator = 45
    integers = [int(x*denominator) for x in v]
    for values in ([2.**-1074, -2.**-500, .25, 2.**52],
                   [1., 1., 1., 1.], [0., -0., 1.25, -3.5]):
        assert K._binary64_dot(integers, denominator, values) == sum(
            (a*Fraction.from_float(b) for a,b in zip(v, values)), Fraction(0))


@pytest.mark.parametrize("change", ["coefficient", "center", "low", "high", "id", "order", "omitted"])
def test_state_tamper_rejected(change):
    args = deepcopy(tiny())
    if change == "coefficient": args[1][0][0] += .125
    if change == "center": args[0][0] += .125
    if change == "low": args[2][0] = -.5
    if change == "high": args[3][0] = 2.
    if change == "id": args[4][0] = "impostor"
    if change == "order":
        for i in (1, 2, 3, 4): args[i].reverse()
    if change == "omitted":
        for i in (1, 2, 3, 4): args[i].pop()
    with pytest.raises(RuntimeError):
        K.verify_certificate(*args, {"fixture": "tiny"}, certificate())


@pytest.mark.parametrize("field", ["delta", "variance_lower", "norm_v_squared", "interval_lower", "v_rationals",
                                  "included_generator_count", "y_rationals"])
def test_rehashed_semantic_claim_tamper_rejected(field):
    cert = certificate()
    if field == "v_rationals": cert[field].reverse()
    elif field == "included_generator_count": cert[field] -= 1
    elif field == "y_rationals":
        cert[field][0] = K.rational(Fraction(2))
        cert["candidate_sha256"] = K.digest(cert[field])
    else: cert[field] = K.rational(K.read_rational(cert[field]) + Fraction(1, 2**52))
    seal(cert)  # A self-consistent hash must NOT substitute for numerical replay.
    with pytest.raises(RuntimeError, match="replay"):
        K.verify_certificate(*tiny(), {"fixture": "tiny"}, cert)


def test_capture_binding_mutation_rejected():
    with pytest.raises(RuntimeError, match="identity"):
        K.verify_certificate(*tiny(), {"fixture": "different"}, certificate())


def test_every_generator_including_numerical_rows_enters_support():
    args = list(tiny())
    args[1] = [[.5, 0., 0.], [2., 0., 0.]]
    args[2], args[3] = [-1., -1.], [1., 1.]
    assert K.construct_certificate(*args, [Fraction(1), Fraction(0)], {}) is None


def test_small_numerical_lp_then_independent_exact_replay():
    result = S.bounded_proposal(tiny()[:4], 5.)
    assert result["y"] is not None and result["column_count"] == 6
    assert result["row_count"] == 9
    cert = K.construct_certificate(*tiny(), list(map(Fraction.from_float, result["y"])), {})
    assert cert is not None and K.verify_certificate(*tiny(), {}, cert)["verified"]


def test_hard_proposal_wall_clock_guard(monkeypatch):
    def slow(*args):
        time.sleep(2)
        raise AssertionError("must never reach here")
    monkeypatch.setattr(S, "_solve_proposal", slow)
    started = time.monotonic()
    result = S.bounded_proposal(tiny()[:4], .05)
    assert time.monotonic()-started < .5
    assert result["y"] is None and result["reason"] == "NUMERICAL_PROPOSAL_TIMEOUT"
    with pytest.raises(ValueError): S.bounded_proposal(tiny()[:4], 61.)


def test_proposal_programming_error_not_hidden(monkeypatch):
    def broken(*args): raise AssertionError("bug")
    monkeypatch.setattr(S, "_solve_proposal", broken)
    with pytest.raises(AssertionError, match="bug"): S.bounded_proposal(tiny()[:4], 1.)


def test_persisted_certificate_and_replay_only_path(capture_fixture, monkeypatch):
    manifest, artifact, payload = capture_fixture
    payload["states"]["post_last_reduction"]["range_high"][0] = .5
    rewrite_artifact(manifest, artifact, payload, component_hashes=True)
    monkeypatch.setattr(S, "EXPECTED_ARTIFACT_SHA", S.C.sha256(artifact))
    # Even a solver claiming FAILURE is only a proposal: exact support decides.
    monkeypatch.setattr(S, "bounded_proposal", lambda *_: {"y": [1.] + [0.]*126, "solver_status": 4})
    report = S.execute(manifest, manifest.parent/"separator.json")
    assert report["final_status"] == S.CERTIFIED and report["exact_check"]["verified"]
    assert K.read_rational(report["exact_check"]["variance_lower"]) == Fraction(1, 256)
    assert report["production_repair_enabled"] is False
    monkeypatch.setattr(S, "bounded_proposal", lambda *_: pytest.fail("replay must not solve"))
    replay = S.execute(manifest, manifest.parent/"replay.json", certificate_path=Path(report["certificate_path"]))
    assert replay["final_status"] == S.CERTIFIED


@pytest.mark.parametrize("kind", ["timeout", "zero", "unseparated"])
def test_no_certified_separator_result(capture_fixture, monkeypatch, kind):
    manifest, artifact, _ = capture_fixture
    monkeypatch.setattr(S, "EXPECTED_ARTIFACT_SHA", S.C.sha256(artifact))
    result = {"y": None, "reason": "NUMERICAL_PROPOSAL_TIMEOUT"} if kind == "timeout" else {
        "y": ([0.]*127 if kind == "zero" else [1.]+[0.]*126), "solver_status": 0}
    monkeypatch.setattr(S, "bounded_proposal", lambda *_: result)
    report = S.execute(manifest, manifest.parent/"none.json")
    assert report["final_status"] == S.NO_SEPARATOR
    assert report["certificate_path"] is None and report["exact_check"] is None


def test_artifact_pin_rejects_resealed_modified_capture(capture_fixture):
    manifest, _, _ = capture_fixture
    with pytest.raises(RuntimeError, match="authenticated ISIS artifact"):
        S.load_operands(manifest)


def test_output_must_not_overwrite_capture(capture_fixture, monkeypatch):
    manifest, artifact, _ = capture_fixture
    monkeypatch.setattr(S, "EXPECTED_ARTIFACT_SHA", S.C.sha256(artifact))
    before = S.C.sha256(manifest)
    with pytest.raises(RuntimeError, match="overwrite"):
        S.execute(manifest, manifest)
    assert S.C.sha256(manifest) == before


def test_independent_arithmetic_imports_standard_library_only():
    tree = ast.parse(Path(K.__file__).read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import): imports.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom): imports.append(node.module)
    assert set(imports) <= {"fractions", "hashlib", "json", "math", "struct"}
