from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research_hab"))

import coret_sound_fp64_block0_feasibility_v1 as experiment


PLAIN_FP64_SUPPORT = {
    "embedding_layernorm": 0.061768030795682716,
    "qk": 0.6511588598061171,
    "softmax": 0.020292222345278388,
    "attention_value": 0.035843053538558434,
    "post_attention_layernorm": 0.07336812505826751,
    "ffn_output": 0.024973556536643016,
    "output_layernorm": 0.08216187021647059,
}


@pytest.fixture(scope="module")
def result():
    return experiment.run_sound_fp64()


def test_complete_block0_and_positive_second_layernorm(result):
    assert result["dtype"] == "torch.float64"
    assert result["dispatch_counts"] == {
        "LayerNorm": 3, "generator_reduction": 1, "QK": 1,
        "softmax": 1, "A.V": 1, "ReLU": 1,
    }
    assert result["second_layernorm_variance_lower"] > 0.8
    assert result["second_layernorm_variance_upper_min"] >= result[
        "second_layernorm_variance_lower"]


def test_roundoff_is_in_state_and_growth_is_bounded(result):
    injections = result["numerical_injections"]
    assert len(injections) == 17
    assert all(item["added_generators"] > 0 for item in injections)
    assert all(math.isfinite(item["maximum_local_widening"])
               and item["maximum_local_widening"] >= 0
               for item in injections)
    assert result["final_proof_generator_count"] == 12826
    assert result["final_proof_generator_count"] < experiment.MAXIMUM_GENERATORS


def test_numerical_widening_remains_below_gate(result):
    states = {row["label"]: row for row in result["rows"]}
    ratios = {}
    for label, reference in PLAIN_FP64_SUPPORT.items():
        ratios[label] = max(0.0, states[label]["support_max"] - reference) / reference
    assert max(ratios.values()) < 0.1
    assert ratios["output_layernorm"] < 0.02


def test_mpfr_spots_and_one_ulp_mutations(result):
    assert len(result["mpfr_spots"]) == 16
    assert {item["label"] for item in result["mpfr_spots"]} == {
        "sqrt_relaxation", "reciprocal_relaxation", "affine_accumulation",
        "qk_coefficient_radius", "softmax_exp_relaxation",
        "attention_value", "layernorm_variance",
        "layernorm_final_product",
        "actual_q_affine_coefficient", "actual_qk_retained_coefficient",
        "actual_softmax_exp", "actual_av_retained_coefficient",
        "actual_layernorm_variance_nominal", "actual_sqrt_relaxation",
        "actual_reciprocal", "actual_layernorm_final_product",
    }
    assert all(item["one_ulp_inward_rejected"]
               for item in result["mpfr_spots"])
    actual = [item for item in result["mpfr_spots"]
              if item["label"].startswith("actual_")]
    assert all(item["state_reserve_contains_machine_error"]
               for item in actual)


def test_output_is_finite_and_usable_for_next_block(result):
    final = result["rows"][-1]
    assert final["label"] == "output_layernorm"
    assert math.isfinite(final["lower_min"])
    assert math.isfinite(final["upper_max"])
    assert final["lower_min"] <= final["upper_max"]
    assert final["support_max"] < 0.1
