from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import a40_fresh_common as fresh
import cluster_common as common
from analyze_a40_interim import (analyze, classification, distribution,
    write_output)


PROPERTY_BY_WORKER = {
    0: "deept_table7_stdln3_s000_line504_tok05",
    1: "deept_table7_stdln3_s000_line504_tok10",
}


def artifact() -> Path:
    return common.artifact_root()


def synthetic_fresh_root(tmp_path: Path) -> Path:
    root = tmp_path / "fresh"
    source = common.baseline_root(artifact()) / "properties"
    for worker_id, property_id in PROPERTY_BY_WORKER.items():
        worker = root / f"worker_{worker_id}"
        destination = fresh.result_root(worker) / "properties" / property_id
        destination.parent.mkdir(parents=True)
        shutil.copytree(source / property_id, destination)
        marker = fresh.seed_record(artifact(), worker_id)
        worker.mkdir(parents=True, exist_ok=True)
        (worker / "fresh_worker_seed_v1.json").write_text(
            json.dumps(marker, sort_keys=True, indent=2) + "\n")
    return root


def test_statistics_and_classification_are_deterministic():
    stats = distribution([1.0, 2.0, 4.0])
    assert stats["median"] == 2.0
    assert stats["q1"] == 1.5
    assert stats["q3"] == 3.0
    clean = {"checker": 0}
    assert classification([1.01, 1.02], clean) == (
        "A40_INTERIM_STRONG", "YES")
    assert classification([0.99, 1.03], clean) == (
        "A40_INTERIM_MIXED", "NO")
    assert classification([0.98, 1.0], clean) == (
        "A40_INTERIM_WEAK", "NO")


def test_local_existing_schemas_validate_without_verifier(tmp_path):
    root = synthetic_fresh_root(tmp_path)
    value = analyze(root, artifact(), 2)
    assert value["snapshot"]["validated_completed"] == 2
    assert value["snapshot"]["worker_completed_counts"] == {"0": 1, "1": 1}
    assert value["status"] == "A40_INTERIM_STRONG"
    assert value["continue_next_chunk"] == "YES"
    assert value["threshold_counts"]["ratio_gt_1"] == 2
    assert value["integrity_and_failures"]["generic_fallbacks"] == 0
    assert value["analysis_scientific_queries_executed"] == 0
    assert value["analysis_bound_entrypoint_calls"] == 0


def test_expected_completed_is_fail_closed(tmp_path):
    root = synthetic_fresh_root(tmp_path)
    with pytest.raises(RuntimeError, match="completed property count differs"):
        analyze(root, artifact(), 3)


def test_duplicate_worker_property_is_rejected(tmp_path):
    root = synthetic_fresh_root(tmp_path)
    source = (fresh.result_root(root / "worker_0") / "properties"
              / PROPERTY_BY_WORKER[0])
    duplicate = (fresh.result_root(root / "worker_1") / "properties"
                 / PROPERTY_BY_WORKER[0])
    shutil.copytree(source, duplicate)
    with pytest.raises(RuntimeError, match="duplicate completed property"):
        analyze(root, artifact(), 3)


def test_output_is_exclusive_and_cannot_modify_inputs(tmp_path):
    root = synthetic_fresh_root(tmp_path)
    value = analyze(root, artifact(), 2)
    output = tmp_path / "analysis" / "a40.json"
    write_output(output, value, root, artifact())
    assert common.verified_json(output) == value
    with pytest.raises(FileExistsError):
        write_output(output, value, root, artifact())
    with pytest.raises(RuntimeError, match="outside input trees"):
        write_output(root / "forbidden.json", value, root, artifact())


def test_analyzer_contains_no_verifier_or_bound_entrypoint():
    text = (common.REPO / "scripts/analyze_a40_interim.py").read_text()
    assert "generate_query(" not in text
    assert "run_property(" not in text
    assert "get_bounds_difference_in_scores" not in text
    assert "subprocess" not in text
