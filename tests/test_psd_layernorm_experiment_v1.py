from __future__ import annotations

import copy
import importlib.util
import inspect
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]

if "gmpy2" not in sys.modules:
    sys.modules["gmpy2"] = types.SimpleNamespace()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PSD = load("psd_ln_experiment", "research_hab/coret_psd_layernorm_experiment_v1.py")
FINISH = load("psd_ln_finish", "scripts/run_sound_fp64_finish_3l_v1.py")
RUNNER = load(
    "psd_ln_experiment_runner",
    "scripts/run_sound_fp64_3l_psd_layernorm_experiment_v1.py")


class FakeState:
    def __init__(self, center=(1.0, -1.0), generator=(0.1, -0.1),
                 low=-1.0, high=1.0):
        self.zonotope_w = torch.tensor(
            [[center], [generator]], dtype=torch.float64)
        self.error_term_range_low = torch.tensor([low], dtype=torch.float64)
        self.error_term_range_high = torch.tensor([high], dtype=torch.float64)
        self.num_error_terms = 1
        self.num_words = 1
        self.word_embedding_size = 2
        self.device = torch.device("cpu")


def proof():
    return PSD.structural.SupportProof((1,), ("g0",), ("source",), 1)


@pytest.fixture(autouse=True)
def skip_structural_kernel_validation(monkeypatch):
    monkeypatch.setattr(PSD.structural, "validate_support",
                        lambda *_args, **_kwargs: {"valid": True})


def test_positive_psd_certificate_overrides_only_negative_generic_domain():
    state = FakeState()
    generic = torch.full((1, 2), -0.5, dtype=torch.float64)
    before = state.zonotope_w.clone()
    certificate = PSD.build_certificate(state, proof(), generic)
    assert certificate["independent_checker_accepts"] is True
    assert certificate["minimum_psd_lower"] > 0.8
    assert certificate["coefficient_clamping_used"] is False
    assert torch.equal(state.zonotope_w, before)


def test_witness_mutation_is_rejected():
    state = FakeState()
    generic = torch.full((1, 2), -0.5, dtype=torch.float64)
    certificate = PSD.build_certificate(state, proof(), generic)
    mutated = copy.deepcopy(certificate)
    mutated["token_certificates"][0]["witness_binary64_hex"][0] = 0.0.hex()
    with pytest.raises(RuntimeError, match="witness recheck"):
        PSD.verify_certificate(state, proof(), generic, mutated)


@pytest.mark.parametrize("field", ["state", "range"])
def test_state_or_range_mutation_is_rejected(field):
    state = FakeState()
    generic = torch.full((1, 2), -0.5, dtype=torch.float64)
    certificate = PSD.build_certificate(state, proof(), generic)
    if field == "state":
        state.zonotope_w[0, 0, 0] += 1.0
    else:
        state.error_term_range_high[0] = 0.5
    with pytest.raises(RuntimeError, match="authenticated state"):
        PSD.verify_certificate(state, proof(), generic, certificate)


def test_optimizer_diagnostic_is_not_trusted_but_checked_lower_is():
    state = FakeState()
    generic = torch.full((1, 2), -0.5, dtype=torch.float64)
    certificate = PSD.build_certificate(state, proof(), generic)
    certificate["token_certificates"][0]["optimizer_diagnostic"] = {
        "untrusted": 1e300}
    checked = PSD.verify_certificate(state, proof(), generic, certificate)
    assert checked["minimum_psd_lower"] == certificate[
        "token_certificates"][0]["outward_safe_lower"]


def test_nonpositive_psd_certificate_still_rejects_domain():
    state = FakeState(center=(0.0, 0.0), generator=(0.0, 0.0))
    generic = torch.full((1, 2), -0.5, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="witness recheck failed"):
        PSD.build_certificate(state, proof(), generic)


def test_experiment_is_opt_in_and_target_restricted():
    parameter = inspect.signature(FINISH.execute).parameters[
        "experimental_post_attention_layernorm"]
    assert parameter.default is None
    with pytest.raises(RuntimeError, match="only handle Block-2 post-attention"):
        PSD.execute_experimental_layernorm(
            residual=object(), proof=object(), normalizer=object(),
            delegate=object(), diagnostics={"label": "block2_output"})


def test_layernorm_index_guard_rejects_non_target_transition():
    state = FakeState()
    with pytest.raises(RuntimeError, match="non-target LayerNorm"):
        PSD._native_layernorm_with_semantic_lower(
            state, proof(), SimpleNamespace(),
            SimpleNamespace(_layer_norm_index=4), [0.5])


def _harness(monkeypatch, callback=None):
    state = FakeState()
    support = RUNNER.experiment.structural.SupportProof(
        (1,), ("g0",), ("source",), 1)
    identity = RUNNER.experiment.state_identity(state, support)
    harness = RUNNER.LayerNormExperimentHarness(
        RUNNER.PROPERTY_ID, RUNNER.TESTED_RADIUS, identity,
        {"sha256": "oracle", "record": {}})
    if callback is None:
        callback = lambda **_kwargs: {
            "output": object(), "proof": object(), "reserve": object(),
            "certificate": {"state_identity": identity}}
    monkeypatch.setattr(
        RUNNER.experiment, "execute_experimental_layernorm", callback)
    return harness, state, support


def test_multiple_callbacks_with_one_authenticated_target_pass(monkeypatch):
    harness, state, support = _harness(monkeypatch)
    sentinel = object()
    result = harness(
        residual=state, proof=support, normalizer=object(),
        delegate=SimpleNamespace(_layer_norm_index=4),
        diagnostics={"label": "block1_output"},
        original_layernorm=lambda **_kwargs: sentinel)
    assert result is sentinel
    harness(
        residual=state, proof=support, normalizer=object(),
        delegate=SimpleNamespace(_layer_norm_index=5),
        diagnostics={"label": RUNNER.experiment.TARGET_LABEL})
    harness.validate()
    assert harness.counters() == {
        "total_callback_invocations": 2,
        "non_target_invocations": 1,
        "target_stage_encounters": 1,
        "target_state_matches": 1,
        "psd_certificate_applications": 1,
        "psd_certificate_rejections": 0,
    }


def test_zero_target_matches_fails_closed(monkeypatch):
    harness, _state, _support = _harness(monkeypatch)
    with pytest.raises(RuntimeError, match="target-state match count"):
        harness.validate()


def test_two_target_matches_fail_closed(monkeypatch):
    harness, state, support = _harness(monkeypatch)
    arguments = dict(
        residual=state, proof=support, normalizer=object(),
        delegate=SimpleNamespace(_layer_norm_index=5),
        diagnostics={"label": RUNNER.experiment.TARGET_LABEL})
    harness(**arguments)
    with pytest.raises(RuntimeError, match="visited twice"):
        harness(**arguments)
    with pytest.raises(RuntimeError, match="target-state match count"):
        harness.validate()


def test_target_match_without_application_fails_closed(monkeypatch):
    def reject(**_kwargs):
        raise RuntimeError("synthetic checker rejection")

    harness, state, support = _harness(monkeypatch, reject)
    with pytest.raises(RuntimeError, match="synthetic checker rejection"):
        harness(
            residual=state, proof=support, normalizer=object(),
            delegate=SimpleNamespace(_layer_norm_index=5),
            diagnostics={"label": RUNNER.experiment.TARGET_LABEL})
    assert harness.target_state_matches == 1
    assert harness.psd_certificate_applications == 0
    with pytest.raises(RuntimeError, match="application count"):
        harness.validate()


def test_non_target_callback_preserves_original_result(monkeypatch):
    harness, state, support = _harness(monkeypatch)
    calls = []

    def original(**kwargs):
        calls.append(kwargs)
        return "unchanged"

    result = harness(
        residual=state, proof=support, normalizer=object(),
        delegate=SimpleNamespace(_layer_norm_index=2),
        diagnostics={"label": "block1_output"},
        original_layernorm=original)
    assert result == "unchanged"
    assert len(calls) == 1
    assert harness.psd_certificate_applications == 0


def test_exception_trace_and_message_persist(tmp_path, monkeypatch):
    harness, _state, _support = _harness(monkeypatch)
    original = RUNNER.finish3l.execute
    monkeypatch.setattr(
        RUNNER.finish3l, "execute",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("synthetic evaluator failure")))
    try:
        with RUNNER._installed_finish_hook(harness):
            with pytest.raises(ValueError, match="synthetic evaluator failure"):
                RUNNER.finish3l.execute()
    finally:
        RUNNER.finish3l.execute = original
    record = harness.record()
    assert record["evaluator_exception"]["exception_type"] == "ValueError"
    assert record["evaluator_exception"]["exception_message"] == (
        "synthetic evaluator failure")
    path = tmp_path / "trace.json"
    persisted = RUNNER._atomic_json(path, record)
    assert persisted["evaluator_exception"]["traceback_tail"]


def test_production_callback_default_remains_none():
    assert inspect.signature(FINISH.execute).parameters[
        "experimental_post_attention_layernorm"].default is None
