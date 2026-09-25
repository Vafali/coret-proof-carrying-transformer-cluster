from __future__ import annotations

import json
from pathlib import Path

import pytest

import cluster_common as common
from make_property_shards import make_shards
from merge_worker_results import copy_no_conflict, validate_property


def artifact():
    return common.artifact_root()


def baseline_state():
    return common.load_baseline_state()


def test_frozen_source_hashes_and_fused_unreachable():
    source = json.loads((common.REPO / "frozen/source_tree_manifest.json").read_text())
    for row in source["files"]:
        path = common.REPO / row["path"]
        assert common.sha256(path) == row["sha256"]
    scientific = list((common.REPO / "research_hab").glob("*.py"))
    assert not any("fused_av" in path.read_text(errors="ignore") for path in scientific)
    assert common.sha256(common.RESEARCH / "coret_optimized_historical_127_v1.py") == \
        "8d3e2216b7174f853968c71eebc78318c86dff061875e194aee3d34c8a85d6cb"
    assert common.sha256(common.RESEARCH / "coret_structural_support_precise_dot_v1.py") == \
        "86ba56410c713283251264c618d7652f1f741531bda4bfdc0dcdf38df7a2a0c6"


def test_artifact_manifest_and_result_import():
    root = artifact()
    value = common.verify_artifact_manifest(root)
    assert value["artifact_count"] > 500
    production = common.load_production_manifest(root)
    assert production["property_count"] == 127
    state = baseline_state()
    assert len(common.completed_property_ids(root)) == state["completed_property_count"]
    expected_partial = ({state["partial_property"]}
                        if state["partial_property"] is not None else set())
    assert common.partial_property_ids(root) == expected_partial
    properties = common.baseline_root(root) / "properties"
    query_count = len(list(properties.glob(
        "*/queries/query_*_result_v1.json")))
    journal_count = sum(
        sum(1 for line in path.read_text().splitlines() if line)
        for path in properties.glob("*/query_events_v1.jsonl"))
    assert query_count == state["persisted_query_count"]
    assert journal_count == state["valid_journal_count"]
    next_root = properties / state["next_property"]
    assert not (next_root / "result_v1.json").exists()
    assert not list((next_root / "queries").glob("query_*_result_v1.json"))


@pytest.mark.parametrize("count", [2, 3, 5])
def test_shards_complete_disjoint_and_partial_once(count):
    root = artifact(); manifest = common.load_production_manifest(root)
    value = make_shards(manifest, common.completed_property_ids(root),
                        common.partial_property_ids(root), [1.0] * count)
    ids = [row["property_id"] for worker in value["workers"]
           for row in worker["properties"]]
    state = baseline_state()
    expected = 127 - state["completed_property_count"]
    assert len(ids) == expected == len(set(ids))
    partial = state["partial_property"]
    assert sum(pid == partial for pid in ids) == (1 if partial is not None else 0)
    again = make_shards(manifest, common.completed_property_ids(root),
                        common.partial_property_ids(root), [1.0] * count)
    assert value == again


def test_weighted_sharding_is_deterministic_and_changes_load():
    root = artifact(); manifest = common.load_production_manifest(root)
    args = (manifest, common.completed_property_ids(root),
            common.partial_property_ids(root), [1.0, 2.0, 4.0])
    first = make_shards(*args); second = make_shards(*args)
    assert first == second
    costs = [row["objective_cost"] for row in first["workers"]]
    assert costs[2] > costs[1] > costs[0]


def test_worker_result_isolation_and_import(tmp_path):
    source = tmp_path / "source"; source.mkdir()
    (source / "fixed.json").write_text("fixed")
    first = tmp_path / "worker1"; second = tmp_path / "worker2"
    import shutil
    shutil.copytree(source, first); shutil.copytree(source, second)
    (first / "new.json").write_text("one")
    assert not (second / "new.json").exists()
    assert (second / "fixed.json").read_text() == "fixed"


def test_merge_synthetic_outputs_and_conflict_rejection(tmp_path):
    destination = tmp_path / "merged.json"
    left = tmp_path / "left.json"; right = tmp_path / "right.json"
    left.write_text("same"); right.write_text("same")
    copy_no_conflict(left, destination)
    copy_no_conflict(right, destination)
    bad = tmp_path / "bad.json"; bad.write_text("different")
    with pytest.raises(RuntimeError, match="conflicting merge record"):
        copy_no_conflict(bad, destination)


def test_reused_completed_property_without_local_query_journal():
    root = common.baseline_root(artifact())
    result = validate_property(
        root, "deept_table7_stdln3_s002_line1468_tok04")
    assert result["reused_from_optimized_smoke"] is True
    assert result["fresh_verifier_evaluations_this_run"] == 0


def test_zero_agent_scientific_or_bound_calls_are_declared():
    artifact_manifest = common.verify_artifact_manifest(artifact())
    source_manifest = json.loads((common.REPO / "frozen/source_tree_manifest.json").read_text())
    assert artifact_manifest["scientific_queries"] == 0
    assert artifact_manifest["bound_entrypoint_calls"] == 0
    assert source_manifest["scientific_queries"] == 0
    assert source_manifest["bound_entrypoint_calls"] == 0


def test_calibration_selection_is_frozen_and_references_payload():
    selection = common.verified_json(
        common.REPO / "frozen/calibration_selection.json",
        "canonical_manifest_sha256")
    assert [row["name"] for row in selection["cases"]] == ["short", "long"]
    root = artifact() / "payload" / common.BASELINE_REL / "properties"
    for row in selection["cases"]:
        query = root / row["property_id"] / "queries"
        result = next(query.glob(f'query_{row["query_ordinal"]:02d}_*_result_v1.json'))
        certificate = next(query.glob(
            f'query_{row["query_ordinal"]:02d}_*_certificates_v1.json'))
        assert common.sha256(result) == row["reference_result_file_sha256"]
        assert common.sha256(certificate) == row["reference_certificate_file_sha256"]
