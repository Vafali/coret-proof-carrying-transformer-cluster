import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "psd_oracle", ROOT / "scripts/diagnose_psd_layernorm_variance_v1.py")
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


def checked(center, generators, low, high, witness=None):
    center = np.array(center, dtype=np.float64)
    generators = np.array(generators, dtype=np.float64).reshape(-1, len(center))
    low, high = np.array(low, dtype=np.float64), np.array(high, dtype=np.float64)
    if witness is None:
        witness, _ = ORACLE.optimize_witness(center, generators, low, high)
    return ORACLE.recheck_dual_witness(
        center, generators, low, high, np.array(witness, dtype=np.float64))


def grid(center, generators, low, high, points=1001):
    return ORACLE.brute_force_minimum(
        np.array(center, dtype=np.float64),
        np.array(generators, dtype=np.float64).reshape(-1, len(center)),
        np.array(low, dtype=np.float64), np.array(high, dtype=np.float64),
        points=points)


def test_zero_generator_exact_variance():
    result = checked([2.0, 0.0], [], [], [])
    assert result["psd_dual_candidate_lower_outward_safe"] <= 1.0
    assert result["psd_dual_candidate_lower_outward_safe"] > 0.999999999999


def test_one_generator_matches_grid_and_weak_duality():
    center, generators = [2.0, -1.0], [[0.25, -0.5]]
    result = checked(center, generators, [-1.0], [1.0])
    exact_grid = grid(center, generators, [-1.0], [1.0])
    assert result["psd_dual_candidate_lower_outward_safe"] <= exact_grid + 1e-14
    assert exact_grid - result["psd_dual_candidate_lower_outward_safe"] < 1e-10


def test_constant_vector_has_zero_minimum():
    result = checked([3.0, 3.0], [[1.0, 1.0]], [-1.0], [1.0])
    assert result["psd_dual_candidate_lower_outward_safe"] <= 0.0
    assert abs(result["psd_dual_candidate_lower_outward_safe"]) < 1e-15


def test_psd_certificate_positive_when_coarse_generic_lower_is_negative():
    # For z=b+a*e, the native affine square relaxation has center
    # b^2+.5a^2, retained row 2ba, and fresh radius .5a^2.  After summation,
    # this example therefore has generic lower 5.125-6-1.125=-2, although
    # its exact minimum variance is 0.25.
    centered, generator = np.array([2.0, -2.0]), np.array([1.5, -1.5])
    generic_center = np.mean(centered**2 + 0.5 * generator**2)
    generic_retained = abs(np.mean(2.0 * centered * generator))
    generic_fresh = np.mean(0.5 * generator**2)
    coarse_generic_lower = generic_center - generic_retained - generic_fresh
    result = checked(centered, [generator], [-1.0], [1.0])
    assert coarse_generic_lower <= 0.0
    assert result["psd_dual_candidate_lower_outward_safe"] > 0.249999999


def test_asymmetric_nonunit_range_uses_adverse_support():
    center, generators = [1.5, -0.5], [[0.4, -0.2]]
    result = checked(center, generators, [0.25], [2.0])
    exact_grid = grid(center, generators, [0.25], [2.0])
    assert result["source_set_range_model"].endswith("h_K(-A^T y)")
    assert result["psd_dual_candidate_lower_outward_safe"] <= exact_grid + 1e-14
    assert exact_grid - result["psd_dual_candidate_lower_outward_safe"] < 1e-9


def test_perturbed_untrusted_witness_remains_a_sound_lower_bound():
    center, generators = [2.0, -1.0], [[0.25, -0.5]]
    witness, _ = ORACLE.optimize_witness(
        np.array(center), np.array(generators), np.array([-1.0]),
        np.array([1.0]))
    witness = witness + np.array([0.3, -0.1])
    result = checked(center, generators, [-1.0], [1.0], witness)
    assert result["psd_dual_candidate_lower_outward_safe"] <= grid(
        center, generators, [-1.0], [1.0]) + 1e-14


def test_two_generator_result_is_below_brute_force_grid():
    center = [1.0, -0.25]
    generators = [[0.2, 0.5], [-0.3, 0.1]]
    low, high = [-1.0, -0.5], [1.0, 1.5]
    result = checked(center, generators, low, high)
    exact_grid = grid(center, generators, low, high, points=301)
    assert result["psd_dual_candidate_lower_outward_safe"] <= exact_grid + 1e-12


def test_inventory_is_read_only_and_reports_missing_tensors(tmp_path):
    pilot = tmp_path / "pilot"
    evaluation = (pilot / "properties" / ORACLE.PROPERTY_ID / "evaluations"
                  / "00_95_of_100.json")
    evaluation.parent.mkdir(parents=True)
    payload = {"multiplier": {"display": "0.95"}}
    payload["record_sha256"] = ORACLE.cluster_common.canonical(payload)
    evaluation.write_text(json.dumps(payload))
    result = ORACLE.inventory(pilot, ORACLE.PROPERTY_ID)
    assert result["coefficient_states_available"] is False
    assert result["scientific_queries"] == result["bound_calls"] == 0
    assert list(pilot.rglob("*.pt")) == []


def test_malformed_ranges_reject():
    with pytest.raises(RuntimeError, match="malformed"):
        checked([1.0, -1.0], [[0.1, 0.2]], [2.0], [1.0])


def test_torch_state_authenticates_schema_ranges_and_topology(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "state.pt"
    payload = {
        "schema": "TEST_STATE_V1",
        "pinned_revision": ORACLE.PINNED_REVISION,
        "weights": torch.zeros(3, 1, 128, dtype=torch.float64),
        "range_low": torch.tensor([-1.0, 0.25], dtype=torch.float64),
        "range_high": torch.tensor([1.0, 2.0], dtype=torch.float64),
        "proof": {"ids": ["a", "b"], "masks": [1, 1],
                  "reasons": ["native", "fp64_roundoff_coordinate_box"],
                  "num_tokens": 1},
    }
    torch.save(payload, path)
    state = ORACLE._load_torch_state(
        path, ORACLE.sha256(path), "TEST_STATE_V1", None)
    assert state["low"].tolist() == [-1.0, 0.25]
    with pytest.raises(RuntimeError, match="schema"):
        ORACLE._load_torch_state(
            path, ORACLE.sha256(path), "WRONG", None)
    with pytest.raises(RuntimeError, match="SHA"):
        ORACLE._load_torch_state(path, "0" * 64, "TEST_STATE_V1", None)
