#!/usr/bin/env python3
"""Torch/NumPy-free checker for the bounded production V/A.V transition."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import coret_production_prefix_checker_v1 as prefix
import coret_production_softmax_checker_v1 as softmax


PINNED_REVISION = "16ffe4075f1f8a7c87fa2a187d8c46cfd51e07bf"
AFFINE_GAMMA_129 = (129 * 2.0 ** -24) / (1 - 129 * 2.0 ** -24)
AV_LOCAL_ERROR_RESERVE = 2.0 ** -18
V_PARAMETER_SHA256 = {
    "weight": "b7c929880a3c56f662446cdffedd0cba7338deab3ba9e5dfe6a795744b2c15ed",
    "bias": "6ee34c7b7821ff98373c12219d45431dc8edae64efae61227146e118fd4905fe",
}
FIXTURE_STATE_SHA256 = {
    "a0_block0_v_projection": "c084027f3d252206d75d0fa344aebaf42913cd9a911cf6dc034d3f1a24b7a3cd",
    "a1_block0_v_heads": "b86fb0e32da46973008da1df06def2cc8bd07dccd2dab7e45b63c2c378ebf82d",
    "a2_block0_av_output_before_projection": "8bdebf8bc2e5c10818071f3f72dcfa508f32df063356975041d7d8ff69d057af",
}
FIXTURE_RANGE_SHA256 = {
    "low": "c3bcf2c0baa46c1bf4f1107e7f16bf3465cf819e8241d23d4bba745202b374fc",
    "high": "9345018cbce1e2a73cb192e2a168ee30a6ae2e53522d4a675aea4a17097e6f22",
}


def _flat(value):
    if isinstance(value, list):
        for item in value:
            yield from _flat(item)
    else:
        yield value


def _check_parent(root, graph):
    parent_path = root / graph["parent_softmax"]["relative_path"]
    raw = parent_path.read_bytes()
    if (hashlib.sha256(raw).hexdigest() != graph["parent_softmax"]["sha256"]
            or graph["run_manifest"]["parent_softmax_trace_sha256"]
            != graph["parent_softmax"]["sha256"]):
        raise AssertionError("parent softmax trace identity mismatch")
    result = softmax.check_production_softmax(parent_path.parent)
    if result.get("accepted") is not True:
        raise AssertionError("parent softmax checker rejected")
    parent_graph = json.loads(raw)
    probability_record = parent_graph["state_records"][-1]
    probability = softmax._state(parent_path.parent, probability_record)
    probability["weights_sha256"] = probability_record[
        "producer_tensor_content_ids"]["weights"]["sha256"]
    probability["range_low_sha256"] = probability_record[
        "native_range_metadata"]["low"]["sha256"]
    probability["range_high_sha256"] = probability_record[
        "native_range_metadata"]["high"]["sha256"]
    prefix_path = parent_path.parent / parent_graph["parent_prefix"]["relative_path"]
    prefix_graph = json.loads(prefix_path.read_text())
    reduced = prefix._state(prefix_path.parent, prefix_graph["state_records"][3])
    return probability, reduced


def _check_v_projection(root, transition, source, output):
    prefix._seal(transition, "V projection transition")
    expected_tau = {
        "projection": "V", "input_width": 128, "output_width": 128,
        "native_equation": "matmul(weight_transpose)_then_add_bias",
        "generator_transition": "ordered_identity",
        "support_transition": "token_mask_identity",
        "numerical_policy": "coefficient_row_local_abs_weight_plus_gamma129",
        "gamma_129_hex": float(AFFINE_GAMMA_129).hex(),
    }
    witness = transition.get("operator_witness", {})
    if (transition.get("operator_family") != "affine_projection"
            or transition.get("input_state_ids") != [source["id"]]
            or transition.get("predecessor_state_id") != source["id"]
            or transition.get("output_state_ids") != [output["id"]]
            or transition.get("tau_k") != expected_tau
            or witness.get("input_weights_sha256") != source["weights_sha256"]
            or witness.get("output_weights_sha256") != output["weights_sha256"]):
        raise AssertionError("V projection trace mismatch")
    if (witness["weight"].get("sha256") != V_PARAMETER_SHA256["weight"]
            or witness["bias"].get("sha256") != V_PARAMETER_SHA256["bias"]):
        raise AssertionError("V frozen parameter identity mismatch")
    weight = prefix._tensor2(root, witness["weight"])
    bias = prefix._tensor1(root, witness["bias"])
    if (source["shape"] != (901, 4, 128)
            or output["shape"] != (901, 4, 128)
            or len(weight) != 128
            or any(len(row) != 128 for row in weight)
            or len(bias) != 128):
        raise AssertionError("V projection dimension mismatch")
    if (output["ids"] != source["ids"]
            or output["ranges"] != source["ranges"]
            or output["masks"] != source["masks"]
            or output["reasons"] != source["reasons"]):
        raise AssertionError("V projection topology mismatch")

    input_eta = list(_flat(source["numerical"]))
    if not input_eta or any(value != input_eta[0] for value in input_eta):
        raise AssertionError("V affine fixture input sidecar is not uniform")
    eta = input_eta[0]
    weight_l1 = [math.fsum(abs(value) for value in row) for row in weight]
    maximum_required = 0.0
    for row in range(901):
        for token in range(4):
            maximum = max(abs(value) for value in source["weights"][row][token])
            for coordinate in range(128):
                required = (eta * weight_l1[coordinate]
                            + AFFINE_GAMMA_129 * maximum
                            * weight_l1[coordinate])
                if row == 0:
                    required += AFFINE_GAMMA_129 * abs(bias[coordinate])
                if output["numerical"][row][token][coordinate] < required:
                    raise AssertionError("V projection numerical sidecar too narrow")
                maximum_required = max(maximum_required, required)
    if len(set(_flat(output["numerical"]))) < 2:
        raise AssertionError("V projection sidecar was globally broadcast")
    return maximum_required


def _check_v_heads(transition, source, output):
    prefix._seal(transition, "V head mapping transition")
    expected_tau = {
        "projection": "V", "heads": 4, "head_width": 32,
        "reshape": [901, 4, 4, 32], "permutation": [2, 0, 1, 3],
        "generator_transition": "ordered_identity",
    }
    witness = transition.get("operator_witness", {})
    if (transition.get("operator_family") != "attention_head_mapping"
            or transition.get("input_state_ids") != [source["id"]]
            or transition.get("predecessor_state_id") != source["id"]
            or transition.get("output_state_ids") != [output["id"]]
            or transition.get("tau_k") != expected_tau
            or witness.get("input_weights_sha256") != source["weights_sha256"]
            or witness.get("output_weights_sha256") != output["weights_sha256"]
            or source["shape"] != (901, 4, 128)
            or output["shape"] != (4, 901, 4, 32)):
        raise AssertionError("V head mapping trace mismatch")
    if (output["ids"] != source["ids"]
            or output["ranges"] != source["ranges"]
            or output["masks"] != source["masks"]):
        raise AssertionError("V head topology mismatch")
    for head in range(4):
        for row in range(901):
            for token in range(4):
                start = head * 32
                if output["weights"][head][row][token] != source["weights"][
                        row][token][start:start + 32]:
                    raise AssertionError("V head permutation mismatch")
                if output["numerical"][head][row][token] != source[
                        "numerical"][row][token][start:start + 32]:
                    raise AssertionError("V head numerical mapping mismatch")


def _check_precise_dot(witness, root):
    from coret_native_numerical_checker_v1 import check_precise_dot_witness
    return check_precise_dot_witness(witness, root)


def _check_transposed_value_operand(root, precise, value):
    from coret_native_numerical_checker_v1 import _load_block
    transposed = _load_block(root, precise["right"])
    if transposed.shape != (4, 901, 32, 4):
        raise AssertionError("A.V transposed value operand shape mismatch")
    for head in range(4):
        for row in range(901):
            for column in range(32):
                for key in range(4):
                    if (transposed.at(head, row, column, key)
                            != value["weights"][head][row][key][column]):
                        raise AssertionError(
                            "A.V transposed value operand differs from V heads")


def _check_av_sidecar(probability, value, output, rigorous_backend):
    ga, gb = len(probability["ids"]), len(value["ids"])
    gmin, gmax = min(ga, gb), max(ga, gb)
    coordinate_results = rigorous_backend["coordinate_results"]
    maxima = {"center": 0.0, "retained": 0.0, "fresh": 0.0,
              "coordinate": 0.0, "local": 0.0}
    for head in range(4):
        for query in range(4):
            av = [probability["weights"][head][row][query]
                  for row in range(ga + 1)]
            ae = [probability["numerical"][head][row][query]
                  for row in range(ga + 1)]
            for column in range(32):
                bv = [[value["weights"][head][row][key][column]
                       for key in range(4)] for row in range(gb + 1)]
                be = [[value["numerical"][head][row][key][column]
                       for key in range(4)] for row in range(gb + 1)]

                def sensitivity(first, second):
                    return math.fsum(
                        abs(av[first][d]) * be[second][d]
                        + abs(bv[second][d]) * ae[first][d]
                        + ae[first][d] * be[second][d] for d in range(4))

                diagonal = [sensitivity(index, index)
                            for index in range(1, gmin + 1)]
                offset = head * 128 + query * 32 + column
                local = (float(coordinate_results[offset]["center_upper"])
                         + float(coordinate_results[offset]["retained_upper"]))
                if local > AV_LOCAL_ERROR_RESERVE:
                    raise AssertionError("A.V local numerical reserve is too narrow")
                center = sensitivity(0, 0) + 0.5 * math.fsum(diagonal) + local
                if output["numerical"][head][0][query][column] < center:
                    raise AssertionError("A.V center sidecar too narrow")

                retained = []
                for index in range(1, gmax + 1):
                    required = 0.0
                    if index <= gb:
                        required += sensitivity(0, index)
                    if index <= ga:
                        required += sensitivity(index, 0)
                    retained.append(required)
                    if output["numerical"][head][index][query][column] < required:
                        raise AssertionError("A.V retained sidecar too narrow")

                sum_a = [math.fsum(abs(av[i][d]) for i in range(1, ga + 1))
                         for d in range(4)]
                sum_b = [math.fsum(abs(bv[i][d]) for i in range(1, gb + 1))
                         for d in range(4)]
                sum_ae = [math.fsum(ae[i][d] for i in range(1, ga + 1))
                          for d in range(4)]
                sum_be = [math.fsum(be[i][d] for i in range(1, gb + 1))
                          for d in range(4)]
                all_pairs = math.fsum(
                    sum_a[d] * sum_be[d] + sum_b[d] * sum_ae[d]
                    + sum_ae[d] * sum_be[d] for d in range(4))
                fresh = all_pairs - 0.5 * math.fsum(diagonal)
                owner = 1 + gmax + offset
                for row in range(1 + gmax, 1 + gmax + 512):
                    stored = output["numerical"][head][row][query][column]
                    if row == owner:
                        if stored < fresh:
                            raise AssertionError("A.V fresh sidecar too narrow")
                    elif stored != 0.0:
                        raise AssertionError("A.V fresh sidecar owner mismatch")
                coordinate = center + math.fsum(retained) + fresh
                maxima["center"] = max(maxima["center"], center)
                maxima["retained"] = max(
                    maxima["retained"], max(retained))
                maxima["fresh"] = max(maxima["fresh"], fresh)
                maxima["coordinate"] = max(maxima["coordinate"], coordinate)
                maxima["local"] = max(maxima["local"], local)
    return maxima


def check_production_av(root):
    root = Path(root).resolve()
    graph = json.loads((root / "av_trace.json").read_text())
    if graph.get("schema") != prefix.SCHEMA:
        raise AssertionError("A.V trace schema mismatch")
    prefix._seal(graph, "A.V trace")
    prefix._seal(graph["run_manifest"], "A.V run manifest")
    manifest = graph["run_manifest"]
    if (manifest.get("pinned_deept_revision") != PINNED_REVISION
            or manifest.get("purpose") != "bounded_real_production_av_trace"
            or manifest.get("scientific_query") is not False
            or manifest.get("bound_entrypoint_called") is not False
            or manifest.get("prefix_stop")
            != "block0_native_precise_av_output_before_projection"):
        raise AssertionError("A.V run identity mismatch")
    probability, reduced = _check_parent(root, graph)

    records = graph["state_records"]
    if len(records) != 3:
        raise AssertionError("A.V state inventory mismatch")
    value_projection = prefix._state(root, records[0])
    value_heads = prefix._state(root, records[1])
    output = softmax._state(root, records[2])
    output["weights_sha256"] = records[2][
        "producer_tensor_content_ids"]["weights"]["sha256"]
    if graph["graph_nodes"] != [value_projection["id"], value_heads["id"],
                                output["id"]]:
        raise AssertionError("A.V state order mismatch")
    observed = {record["state_id"]: record[
        "producer_tensor_content_ids"]["weights"]["sha256"]
        for record in records}
    if observed != FIXTURE_STATE_SHA256:
        raise AssertionError("frozen production A.V state identity mismatch")
    native_ranges = records[-1]["native_range_metadata"]
    if (native_ranges.get("kind") != "explicit"
            or native_ranges["low"].get("sha256") != FIXTURE_RANGE_SHA256["low"]
            or native_ranges["high"].get("sha256") != FIXTURE_RANGE_SHA256["high"]):
        raise AssertionError("frozen production A.V ranges differ")
    if (graph["parent_reduced_state"] != {
            "state_id": reduced["id"],
            "weights_sha256": reduced["weights_sha256"]}):
        raise AssertionError("pre-attention hidden-state continuity mismatch")

    v_required = _check_v_projection(
        root, graph["transition_records"][0], reduced, value_projection)
    _check_v_heads(graph["transition_records"][1], value_projection,
                   value_heads)

    transition = graph["transition_records"][2]
    prefix._seal(transition, "A.V transition")
    tau = transition.get("tau_k", {})
    witness = transition.get("operator_witness", {})
    ga, gb, fresh = len(probability["ids"]), len(value_heads["ids"]), 512
    gmax = max(ga, gb)
    expected_fresh_ids = [f"av_0_fresh_{index:06d}" for index in range(fresh)]
    if (transition.get("operator_family") != "A.V"
            or transition.get("input_state_ids")
            != [probability["id"], value_heads["id"]]
            or transition.get("predecessor_state_ids")
            != [probability["id"], value_heads["id"]]
            or transition.get("output_state_ids") != [output["id"]]
            or [tau.get("heads"), tau.get("query_tokens"),
                tau.get("value_width"), tau.get("key_width")] != [4, 4, 32, 4]
            or tau.get("probability_generator_count") != ga
            or tau.get("value_generator_count") != gb
            or tau.get("retained_generator_count") != gmax
            or tau.get("fresh_generator_count") != fresh
            or tau.get("fresh_generator_ids") != expected_fresh_ids
            or tau.get("numerical_backend") != "checker_only_rigorous_fp64"
            or tau.get("upstream_sensitivity")
            != "dependency_aware_factorized_O_gD"
            or tau.get("sensitivity_terms") != [
                "abs_probability_times_eta_value",
                "abs_value_times_eta_probability",
                "eta_probability_times_eta_value"]
            or tau.get("local_error_reserve_hex")
            != float(AV_LOCAL_ERROR_RESERVE).hex()
            or tau.get("local_error_placement")
            != "once_in_center_coordinate"
            or tau.get("range_policy")
            != "inherit_probability_then_append_minus1_plus1"
            or witness.get("input_probability_weights_sha256")
            != probability["weights_sha256"]
            or witness.get("input_value_weights_sha256")
            != value_heads["weights_sha256"]
            or witness.get("softmax_range_low_sha256")
            != probability["range_low_sha256"]
            or witness.get("softmax_range_high_sha256")
            != probability["range_high_sha256"]
            or witness.get("output_weights_sha256") != output["weights_sha256"]
            or output["shape"] != (4, 1605, 4, 32)):
        raise AssertionError("native precise A.V trace mismatch")

    # The probability state carries the complete accepted softmax equality and
    # explicit range relation.  V shares the first 900 native IDs; its absent
    # softmax suffix is padded with exact zero by the precise-dot semantics.
    if (value_heads["ids"] != reduced["ids"]
            or probability["ids"][:gb] != value_heads["ids"]
            or output["ids"][:gmax] != probability["ids"]
            or output["ids"][gmax:] != expected_fresh_ids
            or output["ranges"] != probability["ranges"] + [(-1.0, 1.0)] * fresh):
        raise AssertionError("A.V relational generator/range transition mismatch")
    expected_masks = [15 if value_heads["masks"][index]
                      else probability["masks"][index]
                      for index in range(gb)]
    expected_masks += probability["masks"][gb:]
    expected_masks += [1 << query for _head in range(4)
                       for query in range(4) for _column in range(32)]
    if output["masks"] != expected_masks:
        raise AssertionError("A.V structural support transition mismatch")
    expected_reasons = (["native_same_id_union"] * gmax
                        + ["native_output_coordinate_fresh"] * fresh)
    if output["reasons"] != expected_reasons:
        raise AssertionError("A.V support provenance transition mismatch")

    diagnostics = witness.get("structural_diagnostics", {})
    nominal = 4 * 4 * 32 * ga * gb * 4
    executed = diagnostics.get("executed_quadratic_MACs")
    skipped = diagnostics.get("structurally_skipped_quadratic_MACs")
    if (diagnostics.get("schema")
            != "CORET_STRUCTURAL_SUPPORT_PRECISE_DOT_V1"
            or diagnostics.get("mode") != "A.V"
            or diagnostics.get("support_validated") is not True
            or diagnostics.get("nominal_quadratic_MACs") != nominal
            or not isinstance(executed, int) or not isinstance(skipped, int)
            or executed < 0 or skipped < 0 or executed + skipped != nominal
            or diagnostics.get("support_class_count_left")
            != len(set(probability["masks"]))
            or diagnostics.get("support_class_count_right")
            != len(set(value_heads["masks"]))
            or diagnostics.get("requested_generator_tile") != 112
            or diagnostics.get("effective_generator_tile") != 112
            or diagnostics.get("temporary_cap_bytes") != 128 * 1024 * 1024
            or diagnostics.get("peak_temporary_bytes") > 128 * 1024 * 1024):
        raise AssertionError("A.V structural execution inventory mismatch")

    precise = witness.get("independent_numerical_witness")
    if not isinstance(precise, dict):
        raise AssertionError("A.V independent numerical witness missing")
    if (precise.get("family") != "A.V"
            or precise.get("left", {}).get("sha256")
            != probability["weights_sha256"]
            or precise.get("output", {}).get("sha256")
            != output["weights_sha256"]
            or precise.get("right", {}).get("sha256")
            != witness.get("transposed_value_weights_sha256")):
        raise AssertionError("A.V numerical witness/state continuity mismatch")
    if os.environ.get("CORET_PRECISE_DOT_NUMERICAL_BACKEND") != "rigorous_fp64":
        raise AssertionError("A.V checker must use accepted rigorous FP64 backend")
    _check_transposed_value_operand(root, precise, value_heads)
    rigorous = _check_precise_dot(precise, root)
    if (rigorous.get("accepted") is not True
            or rigorous.get("family") != "A.V"
            or rigorous.get("numerical_backend")
            != "checker_only_rigorous_fp64"):
        raise AssertionError("rigorous A.V numerical checker rejected")
    sidecar = _check_av_sidecar(
        probability, value_heads, output, rigorous["backend"])

    transparency = manifest["producer_transparency"]
    output_sha = output["weights_sha256"]
    low_sha = native_ranges["low"]["sha256"]
    high_sha = native_ranges["high"]["sha256"]
    if transparency != {
            "instrumented_v_projection_sha256": value_projection["weights_sha256"],
            "uninstrumented_v_projection_sha256": value_projection["weights_sha256"],
            "instrumented_v_heads_sha256": value_heads["weights_sha256"],
            "uninstrumented_v_heads_sha256": value_heads["weights_sha256"],
            "instrumented_av_sha256": output_sha,
            "uninstrumented_av_sha256": output_sha,
            "instrumented_range_low_sha256": low_sha,
            "uninstrumented_range_low_sha256": low_sha,
            "instrumented_range_high_sha256": high_sha,
            "uninstrumented_range_high_sha256": high_sha}:
        raise AssertionError("A.V bitwise transparency mismatch")
    if graph.get("final_property_record") is not None:
        raise AssertionError("A.V prefix must not claim a property")

    low, high, numerical = softmax._concretize(
        output["weights"], output["numerical"], output["ranges"])
    native_radius = 0.0
    numerical_max = 0.0
    for head in range(4):
        for query in range(4):
            for column in range(32):
                center = output["weights"][head][0][query][column]
                native_radius = max(
                    native_radius, high[head][query][column] - center,
                    center - low[head][query][column])
                numerical_max = max(
                    numerical_max, numerical[head][query][column])
    sound_low = min(
        low[h][q][c] - numerical[h][q][c]
        for h in range(4) for q in range(4) for c in range(32))
    sound_high = max(
        high[h][q][c] + numerical[h][q][c]
        for h in range(4) for q in range(4) for c in range(32))
    if (not math.isfinite(sound_low) or not math.isfinite(sound_high)
            or numerical_max >= native_radius):
        raise AssertionError("A.V numerical state is not useful downstream")
    return {
        "accepted": True,
        "PRODUCTION_V_PROJECTION_PASS": True,
        "PRODUCTION_V_HEAD_MAPPING_PASS": True,
        "PRODUCTION_AV_NUMERICAL_CHECK_PASS": True,
        "PRODUCTION_AV_RELATIONAL_STATE_PASS": True,
        "PRODUCTION_AV_STATE_CONTINUITY_PASS": True,
        "PRODUCTION_AV_BITWISE_EQUIVALENCE_PASS": True,
        "v_shape": list(value_heads["shape"]),
        "av_shape": list(output["shape"]),
        "probability_generator_count": ga,
        "value_generator_count": gb,
        "output_generator_count": len(output["ids"]),
        "fresh_generator_count": fresh,
        "maximum_v_projection_required": v_required,
        "maximum_native_support": native_radius,
        "maximum_checker_numerical_contribution": numerical_max,
        "numerical_native_ratio": numerical_max / native_radius,
        "av_sound_range": [sound_low, sound_high],
        "maximum_local_precise_dot_error": sidecar["local"],
        "rigorous_av_checker_seconds": rigorous["backend"]["wall_seconds"],
    }
